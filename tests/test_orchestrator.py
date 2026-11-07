from types import SimpleNamespace
from uuid import uuid4

import pytest
from google.adk.agents import Agent
from google.adk.models import BaseLlm, LlmResponse
from google.adk.runners import Runner
from google.genai import types

from chatbot.config import Settings
from chatbot.model_gateway import UnknownModelError
from chatbot.orchestrator import APP_NAME, ChatOrchestrator, ConversationNotFoundError


class FakeRunner:
    async def run_async(self, *, user_id, session_id, new_message):
        assert user_id == "api_client"
        assert session_id
        assert new_message.parts[0].text
        yield SimpleNamespace(
            is_final_response=lambda: True,
            content=SimpleNamespace(parts=[SimpleNamespace(text="Model response")]),
        )


class StubLlm(BaseLlm):
    async def generate_content_async(self, llm_request, stream=False):
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self.model)]),
            turnComplete=True,
        )


@pytest.mark.asyncio
async def test_session_persists_across_orchestrator_instances(tmp_path):
    settings = Settings(
        chat_api_key="a" * 32,
        google_api_key="test",
        session_db_url=f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}",
    )
    first = ChatOrchestrator(settings)
    assert first.gateway.runners["default"].agent.model.client_kwargs == {"api_key": "test"}
    first.gateway.runners["default"] = FakeRunner()
    conversation_id, reply, model = await first.chat("Hello", None)
    assert reply == "Model response"
    assert model == "default"

    second = ChatOrchestrator(settings)
    second.gateway.runners["flash-lite"] = FakeRunner()
    same_id, reply, model = await second.chat("Follow-up", conversation_id, "flash-lite")
    assert same_id == conversation_id
    assert reply == "Model response"
    assert model == "flash-lite"

    with pytest.raises(ConversationNotFoundError):
        await second.chat("Unknown", uuid4())
    with pytest.raises(UnknownModelError):
        await second.chat("No model", None, "unlisted")


@pytest.mark.asyncio
async def test_adk_runner_can_switch_models_in_one_conversation(tmp_path):
    settings = Settings(
        chat_api_key="a" * 32,
        google_api_key="test",
        session_db_url=f"sqlite+aiosqlite:///{tmp_path / 'adk.db'}",
    )
    orchestrator = ChatOrchestrator(settings)
    for alias in ("default", "flash-lite"):
        orchestrator.gateway.runners[alias] = Runner(
            agent=Agent(name="orchestrator", model=StubLlm(model=alias)),
            app_name=APP_NAME,
            session_service=orchestrator.sessions,
        )

    conversation_id, first_reply, _ = await orchestrator.chat("First", None)
    same_id, second_reply, selected = await orchestrator.chat(
        "Second", conversation_id, "flash-lite"
    )
    assert same_id == conversation_id
    assert first_reply == "default"
    assert second_reply == "flash-lite"
    assert selected == "flash-lite"
