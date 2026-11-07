import asyncio
import logging
from uuid import UUID, uuid4

from google.adk.sessions import DatabaseSessionService
from google.genai import types

from chatbot.config import Settings
from chatbot.model_gateway import ModelGateway

logger = logging.getLogger(__name__)
APP_NAME = "chatbot_a_z"
USER_ID = "api_client"


class ChatUnavailableError(Exception):
    """The agent did not produce a usable response."""


class ChatTimeoutError(ChatUnavailableError):
    """The agent exceeded the response deadline."""


class ConversationNotFoundError(Exception):
    """The requested conversation does not exist."""


class ChatOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.timeout_seconds = settings.chat_timeout_seconds
        self.sessions = DatabaseSessionService(db_url=settings.session_db_url)
        self.gateway = ModelGateway(settings, self.sessions, APP_NAME)

    async def chat(
        self, message: str, conversation_id: UUID | None, model: str | None = None
    ) -> tuple[UUID, str, str]:
        model_alias, runner = self.gateway.select(model)
        if conversation_id is None:
            conversation_id = uuid4()
            await self.sessions.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=str(conversation_id)
            )
        else:
            session = await self.sessions.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=str(conversation_id)
            )
            if session is None:
                raise ConversationNotFoundError

        content = types.Content(role="user", parts=[types.Part(text=message)])
        final_text: str | None = None
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async for event in runner.run_async(
                    user_id=USER_ID, session_id=str(conversation_id), new_message=content
                ):
                    if event.is_final_response() and event.content:
                        parts = [part.text for part in event.content.parts if part.text]
                        if parts:
                            final_text = "\n".join(parts)
        except TimeoutError as exc:
            logger.warning("ADK timed out for conversation %s", conversation_id)
            raise ChatTimeoutError from exc

        if not final_text:
            logger.error(
                "ADK completed without a text response for conversation %s", conversation_id
            )
            raise ChatUnavailableError
        return conversation_id, final_text, model_alias
