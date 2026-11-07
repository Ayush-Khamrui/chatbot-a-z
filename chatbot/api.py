import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chatbot.config import Settings
from chatbot.model_gateway import UnknownModelError
from chatbot.orchestrator import (
    ChatOrchestrator,
    ChatTimeoutError,
    ChatUnavailableError,
    ConversationNotFoundError,
)

logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"message": "hi", "model": "flash-lite"}}
    )

    message: str = Field(min_length=1, max_length=16000)
    conversation_id: UUID | None = Field(
        default=None,
        description="Omit for a new chat; use an ID returned by a previous successful response.",
    )
    model: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    conversation_id: UUID
    reply: str
    model: str


class ModelInfo(BaseModel):
    alias: str
    provider: str
    model: str


def create_app(
    settings: Settings | None = None, orchestrator: ChatOrchestrator | None = None
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.orchestrator = orchestrator or ChatOrchestrator(settings)
        yield

    app = FastAPI(title="Chatbot A-Z API", version="0.1.0", lifespan=lifespan)

    def verify_api_key(x_api_key: str | None = Depends(api_key_header)) -> None:
        if x_api_key is None or not secrets.compare_digest(x_api_key, settings.chat_api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=list[ModelInfo], dependencies=[Depends(verify_api_key)])
    async def models() -> list[ModelInfo]:
        return [ModelInfo(**item) for item in app.state.orchestrator.gateway.catalog()]

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            conversation_id, reply, model_alias = await app.state.orchestrator.chat(
                request.message, request.conversation_id, request.model
            )
        except UnknownModelError as exc:
            raise HTTPException(status_code=400, detail="Unknown model alias") from exc
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Conversation not found. Omit conversation_id to start a new chat, "
                    "or use an ID from a previous successful response."
                ),
            ) from exc
        except ChatTimeoutError as exc:
            raise HTTPException(status_code=504, detail="Model response timed out") from exc
        except ChatUnavailableError as exc:
            raise HTTPException(status_code=502, detail="No response from model") from exc
        except Exception as exc:
            logger.exception("Chat request failed")
            raise HTTPException(status_code=502, detail="Chat service unavailable") from exc
        return ChatResponse(conversation_id=conversation_id, reply=reply, model=model_alias)

    return app
