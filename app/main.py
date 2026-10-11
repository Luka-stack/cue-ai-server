import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.agent.runtime import AgentRuntime
from app.api.deps import AgentRuntimeDep, SessionDep
from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.db.session import engine

logger = logging.getLogger(__name__)


async def _check_ollama(base_url: str, model: str) -> tuple[bool, str]:
    """Is Ollama reachable and does it have the configured model pulled?"""
    from ollama import AsyncClient

    try:
        listing = await AsyncClient(host=base_url).list()
    except Exception as exc:  # connection refused, DNS, timeout, ...
        return False, f"Ollama unreachable at {base_url}: {exc}"
    available = [m.model for m in listing.models]
    if model not in available:
        return False, f"model {model!r} not pulled (run: ollama pull {model}); have {available}"
    return True, "ok"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # Fail fast if the database is unreachable. Schema is owned by Alembic.
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))

    # The agent's persistence lives in the same database; the model is only warned about,
    # so the items API keeps working while Ollama is down (chat then answers 503).
    app.state.agent = await AgentRuntime.create(settings)
    ok, detail = await _check_ollama(settings.ollama_base_url, settings.ollama_model)
    if ok:
        logger.info(
            "chat agent ready: model=%s persistence=%s",
            settings.ollama_model,
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
        """Readiness of the chat feature: Ollama reachable and the model pulled."""
        ok, detail = await _check_ollama(settings.ollama_base_url, runtime.model_name)
        return {
            "status": "ok" if ok else "degraded",
            "model": runtime.model_name,
            "persistence": runtime.persistence_backend,
            "detail": detail,
        }

    return app


app = create_app()
