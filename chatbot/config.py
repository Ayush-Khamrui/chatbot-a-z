from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelTarget(BaseModel):
    provider: Literal["gemini"]
    model: str = Field(min_length=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chat_api_key: str = Field(min_length=24)
    google_api_key: str = Field(min_length=1)
    default_model: str = "default"
    model_routes: dict[str, ModelTarget] = Field(
        default_factory=lambda: {
            "default": ModelTarget(provider="gemini", model="gemini-flash-latest"),
            "flash-lite": ModelTarget(provider="gemini", model="gemini-3.5-flash-lite"),
            "stable-flash": ModelTarget(provider="gemini", model="gemini-3.8-flash"),
        }
    )
    session_db_url: str = "sqlite+aiosqlite:///./chat_sessions.db"
    chat_timeout_seconds: int = Field(default=60, ge=1, le=300)
