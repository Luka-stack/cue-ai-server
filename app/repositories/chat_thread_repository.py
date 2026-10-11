import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_thread import ChatThread


class ChatThreadRepository:
    """Persistence for thread metadata. Messages live in the LangGraph checkpointer."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, thread: ChatThread) -> ChatThread:
        self._session.add(thread)
        await self._session.flush()
        await self._session.refresh(thread)
        return thread

    async def get(self, thread_id: uuid.UUID) -> ChatThread | None:
        return await self._session.get(ChatThread, thread_id)

    async def list_for_user(self, user_id: str, *, limit: int, offset: int) -> Sequence[ChatThread]:
        stmt = (
            select(ChatThread)
            .where(ChatThread.user_id == user_id)
            .order_by(ChatThread.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return (await self._session.scalars(stmt)).all()

    async def touch(self, thread: ChatThread, *, title: str | None = None) -> ChatThread:
        """Bump ``updated_at`` (and set the title if given) after a turn."""
        if title is not None:
            thread.title = title
        thread.updated_at = func.now()
        await self._session.flush()
        await self._session.refresh(thread)
        return thread

    async def delete(self, thread: ChatThread) -> None:
        await self._session.delete(thread)
        await self._session.flush()
