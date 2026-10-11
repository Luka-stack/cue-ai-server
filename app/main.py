import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.agent.runtime import AgentRuntime
from app.api.deps import AgentRuntimeDep, SessionDep
from app.api.v1.router import api_v1_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.db.session import engine

logger = logging.getLogger(__name__)


async def _check_openai(settings: Settings) -> tuple[bool, str]:
    """Does the API key work, and can it use the configured model?"""
    from openai import AsyncOpenAI, OpenAIError

    if settings.openai_api_key is None:
        return False, "CUEAI_OPENAI_API_KEY is not set"

    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    try:
        await client.models.retrieve(settings.openai_model)
    except OpenAIError as exc:
        return False, f"OpenAI model {settings.openai_model!r} unavailable: {exc}"
    finally:
        await client.close()

    return True, "ok"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # Fail fast if the database is unreachable. Schema is owned by Alembic.
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))

    # The agent's persistence lives in the same database; the model is only warned about,
    # so the items API keeps working while OpenAI is unreachable (chat then answers 503).
    app.state.agent = await AgentRuntime.create(settings)
    ok, detail = await _check_openai(settings)
    if ok:
        logger.info(
            "chat agent ready: model=%s persistence=%s",
            settings.openai_model,
            app.state.agent.persistence_backend,
        )
    else:
        logger.warning("chat agent degraded: %s", detail)

    try:
        yield
    finally:
        await app.state.agent.aclose()
        await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="CueAI Server",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.app_env != "prod" else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_v1_router, prefix="/api/v1")

    @app.get("/health", tags=["health"])
    async def health(session: SessionDep) -> dict[str, str]:
        await session.execute(text("SELECT 1"))
        return {"status": "ok"}

    @app.get("/health/agent", tags=["health"])
    async def health_agent(runtime: AgentRuntimeDep) -> dict[str, str]:
        """Readiness of the chat feature: the OpenAI key works and the model is available."""
        ok, detail = await _check_openai(settings)
        return {
            "status": "ok" if ok else "degraded",
            "model": runtime.model_name,
            "persistence": runtime.persistence_backend,
            "detail": detail,
        }

    return app


app = create_app()
