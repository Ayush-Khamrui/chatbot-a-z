from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from chatbot.api import create_app
from chatbot.config import Settings
from chatbot.model_gateway import UnknownModelError
from chatbot.orchestrator import (
    ChatTimeoutError,
    ChatUnavailableError,
    ConversationNotFoundError,
)


class FakeOrchestrator:
    def __init__(self):
        self.gateway = self

    def catalog(self):
        return [{"alias": "default", "provider": "gemini", "model": "gemini-flash-latest"}]

    async def chat(self, message, conversation_id, model):
        if model == "unknown":
            raise UnknownModelError
        if message == "missing":
            raise ConversationNotFoundError
        if message == "empty response":
            raise ChatUnavailableError
        if message == "timeout":
            raise ChatTimeoutError
        return conversation_id or uuid4(), f"Reply: {message}", model or "default"


def app():
    settings = Settings(chat_api_key="a" * 32, google_api_key="test")
    return create_app(settings, FakeOrchestrator())


@pytest.mark.asyncio
async def test_chat_and_conversation_id():
    application = app()
    async with (
        application.router.lifespan_context(application),
        AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as api,
    ):
        response = await api.post(
            "/v1/chat", headers={"X-API-Key": "a" * 32}, json={"message": "Hello"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["reply"] == "Reply: Hello"
        assert body["model"] == "default"
        continued = await api.post(
            "/v1/chat",
            headers={"X-API-Key": "a" * 32},
            json={
                "message": "Again",
                "conversation_id": body["conversation_id"],
                "model": "flash-lite",
            },
        )
        assert continued.status_code == 200
        assert continued.json()["conversation_id"] == body["conversation_id"]
        assert continued.json()["model"] == "flash-lite"


@pytest.mark.asyncio
async def test_auth_validation_and_not_found():
    application = app()
    async with (
        application.router.lifespan_context(application),
        AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as api,
    ):
        assert (await api.post("/v1/chat", json={"message": "Hello"})).status_code == 401
        headers = {"X-API-Key": "a" * 32}
        assert (
            await api.post("/v1/chat", headers=headers, json={"message": "   "})
        ).status_code == 422
        assert (
            await api.post("/v1/chat", headers=headers, json={"message": "missing"})
        ).status_code == 404
        assert (
            await api.post("/v1/chat", headers=headers, json={"message": "empty response"})
        ).status_code == 502
        assert (
            await api.post("/v1/chat", headers=headers, json={"message": "timeout"})
        ).status_code == 504
        assert (
            await api.post("/v1/chat", headers=headers, json={"message": "hi", "model": "unknown"})
        ).status_code == 400
        assert (await api.get("/v1/models")).status_code == 401
        assert (await api.get("/v1/models", headers=headers)).json()[0]["alias"] == "default"
