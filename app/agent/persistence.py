"""Checkpointer (short-term memory) and store (long-term memory) backends.

``postgres`` is the production backend: both live in the application database, in tables
owned by LangGraph's own migrations (``checkpoint_migrations`` / ``store_migrations``),
kept separate from Alembic on purpose -- LangGraph evolves that schema itself.

``memory`` keeps everything in-process. It is for tests and throwaway local runs: every
restart forgets all threads and all memories.
"""

import logging
from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from sqlalchemy.engine import make_url

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class AgentPersistence:
    checkpointer: BaseCheckpointSaver
    store: BaseStore
    backend: str
    _pool: object | None = None

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()  # type: ignore[attr-defined]


def _psycopg_conninfo(sqlalchemy_url: str) -> str:
    """Turn the app's SQLAlchemy URL (asyncpg driver) into a plain libpq URL for psycopg."""
    return (
        make_url(sqlalchemy_url).set(drivername="postgresql").render_as_string(hide_password=False)
    )


async def create_persistence(settings: Settings) -> AgentPersistence:
    if settings.agent_persistence == "memory":
        logger.warning("agent persistence is in-memory: threads and memories reset on restart")
        return AgentPersistence(InMemorySaver(), InMemoryStore(), backend="memory")

    # Imported lazily so the in-memory backend never needs psycopg at all.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=_psycopg_conninfo(settings.database_url),
        min_size=1,
        max_size=settings.agent_db_pool_size,
        open=False,
        # Required by LangGraph's Postgres backends.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    # Fail fast at startup rather than retrying for psycopg_pool's default 30 s.
    await pool.open(wait=True, timeout=10)

    checkpointer = AsyncPostgresSaver(pool)
    store = AsyncPostgresStore(pool)
    # Idempotent: applies LangGraph's own migrations for its tables.
    await checkpointer.setup()
    await store.setup()

    return AgentPersistence(checkpointer, store, backend="postgres", _pool=pool)
