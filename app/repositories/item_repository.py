import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.item import Item

TodayBounds = tuple[datetime, datetime]


class ItemRepository:
    """All persistence for items, always scoped to one owner. No HTTP concerns, no
    business rules."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(
        self,
        *,
        user_id: str,
        filter_: str,
        today_bounds: TodayBounds,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Item]:
        stmt = select(Item).where(Item.user_id == user_id)

        if filter_ == "notes":
            stmt = stmt.where(Item.type == "note")
        elif filter_ == "completed":
            stmt = stmt.where(Item.type == "todo", Item.completed.is_(True))
        elif filter_ == "todos":
            stmt = stmt.where(Item.type == "todo")
        elif filter_ == "today":
            start, end = today_bounds
            stmt = stmt.where(Item.type == "todo", Item.date >= start, Item.date < end)

        # Client ordering contract: notes first, then todos by date descending,
        # undated todos last; newest-created wins ties.
        stmt = stmt.order_by(
            case((Item.type == "todo", 1), else_=0),
            Item.date.desc().nulls_last(),
            Item.created_at.desc(),
        )

        if limit is not None:
            stmt = stmt.limit(limit)
        if offset:
            stmt = stmt.offset(offset)

        result = await self._session.scalars(stmt)
        return result.all()

    async def list_by_type(
        self, *, user_id: str, type_: str, limit: int
    ) -> Sequence[Item]:
        """Oldest first, so the agent sees memories in the order they were created."""
        stmt = (
            select(Item)
            .where(Item.user_id == user_id, Item.type == type_)
            .order_by(Item.created_at, Item.id)
            .limit(limit)
        )
        return (await self._session.scalars(stmt)).all()

    # Annotated as Sequence: inside this class body, ``list`` is the method above.
    async def get_many(
        self, item_ids: Sequence[uuid.UUID], *, user_id: str
    ) -> Sequence[Item]:
        if not item_ids:
            return []
        stmt = select(Item).where(Item.user_id == user_id, Item.id.in_(item_ids))
        return (await self._session.scalars(stmt)).all()

    async def get(self, item_id: uuid.UUID, *, user_id: str) -> Item | None:
        item = await self._session.get(Item, item_id)
        # Someone else's item is indistinguishable from a missing one.
        if item is None or item.user_id != user_id:
            return None
        return item

    async def add(self, item: Item) -> Item:
        self._session.add(item)
        await self._session.flush()
        await self._session.refresh(item)
        return item

    async def delete(self, item: Item) -> None:
        await self._session.delete(item)
        await self._session.flush()

    async def counts(
        self, *, user_id: str, today_bounds: TodayBounds
    ) -> dict[str, int]:
        start, end = today_bounds
        is_todo = Item.type == "todo"
        stmt = select(
            func.count().label("all"),
            func.count().filter(Item.type == "note").label("notes"),
            func.count().filter(is_todo, Item.completed.is_(True)).label("completed"),
            func.count()
            .filter(is_todo, Item.date >= start, Item.date < end)
            .label("today"),
        ).where(Item.user_id == user_id)
        row = (await self._session.execute(stmt)).one()
        return {
            "all": row.all,
            "notes": row.notes,
            "completed": row.completed,
            "today": row.today,
        }
