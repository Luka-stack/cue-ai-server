"""Test fixtures.

Requires the compose Postgres to be running (``docker compose up -d db``).
A dedicated ``cueai_test`` database is dropped/recreated once per session and
its schema is applied via Alembic, so migrations are exercised by every run.

The chat agent runs against an in-memory checkpointer/store and a scripted chat model,
so the suite needs no OpenAI key and no LangGraph tables.
"""

import asyncio
import os
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from tests.scripted_model import ScriptedChatModel

# Must be set before app.core.config is imported anywhere.
os.environ["CUEAI_AGENT_PERSISTENCE"] = "memory"
os.environ["CUEAI_APP_ENV"] = "test"

ADMIN_URL = "postgresql+asyncpg://cueai:cueai@localhost:5432/postgres"
TEST_DB_NAME = "cueai_test"
TEST_URL = f"postgresql+asyncpg://cueai:cueai@localhost:5432/{TEST_DB_NAME}"


def _run_alembic_upgrade() -> None:
    from alembic.config import Config

    from alembic import command

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", TEST_URL)
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
async def test_engine() -> AsyncIterator[AsyncEngine]:
    admin_engine = create_async_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    async with admin_engine.connect() as conn:
        await conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB_NAME} WITH (FORCE)"))
        await conn.execute(text(f"CREATE DATABASE {TEST_DB_NAME}"))
    await admin_engine.dispose()

    # Alembic's env.py spins up its own event loop; keep it off ours.
    await asyncio.to_thread(_run_alembic_upgrade)

    engine = create_async_engine(TEST_URL)
    yield engine
    await engine.dispose()


@pytest.fixture(autouse=True)
async def _clean_tables(test_engine: AsyncEngine) -> AsyncIterator[None]:
    yield
    async with test_engine.begin() as conn:
        await conn.execute(text("TRUNCATE items, chat_threads"))


@pytest.fixture(scope="session")
def model() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture(scope="session")
async def client(test_engine: AsyncEngine, model: ScriptedChatModel) -> AsyncIterator[AsyncClient]:
    from app.agent.runtime import AgentRuntime
    from app.core.config import get_settings
    from app.db.session import get_db_session
    from app.main import create_app

    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def _test_db_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # Lifespan is intentionally not run: it pings the dev-settings engine and OpenAI,
    # while every request-scoped DB access goes through the overridden dependency and
    # the agent runtime is built here with the scripted model.
    app = create_app()
    app.dependency_overrides[get_db_session] = _test_db_session
    # The agent reads and writes items through its own sessions, on the test database.
    app.state.agent = await AgentRuntime.create(
        get_settings(), model=model, session_factory=session_factory
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
    await app.state.agent.aclose()
