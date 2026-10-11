from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CUEAI_",
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://cueai:cueai@localhost:5432/cueai"
    cors_origins: list[str] = ["*"]
    app_env: Literal["dev", "prod", "test"] = "dev"
    db_echo: bool = False
    log_level: str = "INFO"

    # --- Chat agent -------------------------------------------------------------------
    # OpenAI serves the model. The key is read from CUEAI_OPENAI_API_KEY
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CUEAI_OPENAI_API_KEY"),
    )
    openai_model: str = "gpt-5.4-nano"
    # gpt-5 family only: none | minimal | low | medium | high. Leave empty for models
    # without reasoning (gpt-4.1, gpt-4o).
    openai_reasoning_effort: str | None = "low"
    # Where LangGraph keeps conversation checkpoints and long-term memory.
    # "postgres" reuses database_url; "memory" is in-process and forgets on restart.
    agent_persistence: Literal["postgres", "memory"] = "postgres"
    agent_db_pool_size: int = Field(default=5, ge=1)
    # Upper bound for one conversational turn (may involve up to three model calls).
    agent_turn_timeout_s: float = Field(default=120.0, gt=0)
    # Until authentication exists, requests without an X-User-Id header act as this user.
    default_user_id: str = "default"


@lru_cache
def get_settings() -> Settings:
    return Settings()
