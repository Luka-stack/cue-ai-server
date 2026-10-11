from functools import lru_cache
from typing import Literal

from pydantic import Field
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
    # Ollama serves the model locally; no API key is involved.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_num_ctx: int = Field(default=8192, ge=2048)
    ollama_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # Where LangGraph keeps conversation checkpoints and long-term memory.
    # "postgres" reuses database_url; "memory" is in-process and forgets on restart.
    agent_persistence: Literal["postgres", "memory"] = "postgres"
    agent_db_pool_size: int = Field(default=5, ge=1)
    # Upper bound for one conversational turn (may involve two or three model calls).
    agent_turn_timeout_s: float = Field(default=120.0, gt=0)
    # Until authentication exists, requests without an X-User-Id header act as this user.
    default_user_id: str = "default"


@lru_cache
def get_settings() -> Settings:
    return Settings()
