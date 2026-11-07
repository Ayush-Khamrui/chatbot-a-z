"""Allowlisted model selection and ADK runner construction."""

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService

from chatbot.config import Settings

INSTRUCTION = (
    "You are the orchestrator for a helpful, accurate chat assistant. "
    "At this stage you have no external tools or document access. "
    "Answer directly when possible. If current or private information is required, "
    "state that you cannot verify it. Do not claim to have consulted external sources."
)


class UnknownModelError(Exception):
    """The requested model alias is not configured."""


class ModelGateway:
    def __init__(self, settings: Settings, sessions: BaseSessionService, app_name: str) -> None:
        if settings.default_model not in settings.model_routes:
            raise ValueError("DEFAULT_MODEL must exist in MODEL_ROUTES")
        if not settings.model_routes:
            raise ValueError("MODEL_ROUTES must not be empty")
        self.default_model = settings.default_model
        self.routes = settings.model_routes
        self.runners: dict[str, Runner] = {}
        for alias, target in self.routes.items():
            if (
                not alias
                or not alias.isascii()
                or not all(character.isalnum() or character in "-_" for character in alias)
            ):
                raise ValueError(f"Invalid model alias: {alias!r}")
            model = Gemini(
                model=target.model,
                client_kwargs={"api_key": settings.google_api_key},
            )
            agent = Agent(name="orchestrator", model=model, instruction=INSTRUCTION)
            self.runners[alias] = Runner(agent=agent, app_name=app_name, session_service=sessions)

    def select(self, alias: str | None) -> tuple[str, Runner]:
        selected = alias or self.default_model
        try:
            return selected, self.runners[selected]
        except KeyError as exc:
            raise UnknownModelError from exc

    def catalog(self) -> list[dict[str, str]]:
        return [
            {"alias": alias, "provider": target.provider, "model": target.model}
            for alias, target in sorted(self.routes.items())
        ]
