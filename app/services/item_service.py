import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from app.core.exceptions import InvalidItemOperationError, ItemNotFoundError
from app.core.timezone import resolve_zone
from app.models.item import DEFAULT_TODO_STATUS, Item, is_completed
from app.repositories.item_repository import ItemRepository, TodayBounds
from app.schemas.item import ItemUpdate, NoteCreate, ToDoCreate

_TODO_ONLY_FIELDS = ("date", "completed", "status", "solutions", "time_to_complete")


class ItemService:
    """Items of one user. The chat agent reads and writes the same rows, so edits made
    here are what the agent sees on its next turn."""

    def __init__(self, repo: ItemRepository, user_id: str) -> None:
        self._repo = repo
        self._user_id = user_id

    async def list_items(
        self,
        *,
        filter_: str = "all",
        tz: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Item]:
        return await self._repo.list(
            user_id=self._user_id,
            filter_=filter_,
            today_bounds=self._today_bounds(tz),
            limit=limit,
            offset=offset,
        )

    async def get_item(self, item_id: str) -> Item:
        return await self._get_or_raise(item_id)

    async def create_item(self, payload: NoteCreate | ToDoCreate) -> Item:
        # Payload validation (required title, note needs title or content)
        # already happened in the schema layer.
        item = Item(user_id=self._user_id, **payload.model_dump())
        return await self._repo.add(item)

    async def update_item(self, item_id: str, payload: ItemUpdate) -> Item:
        item = await self._get_or_raise(item_id)
        changes = payload.model_dump(exclude_unset=True)

        if item.type == "note":
            if any(field in changes for field in _TODO_ONLY_FIELDS):
                raise InvalidItemOperationError(
                    "a note cannot have a date, completed state, status, solutions or "
                    "time to complete"
                )
            new_title = changes.get("title", item.title)
            new_content = changes.get("content", item.content)
            if not (new_title or "").strip() and not (new_content or "").strip():
                raise InvalidItemOperationError("a note requires a non-empty title or content")
        else:  # todo
            if "title" in changes and not (changes["title"] or "").strip():
                raise InvalidItemOperationError("a todo requires a non-empty title")
            self._sync_status(item, changes)

        for field, value in changes.items():
            setattr(item, field, value)
        return await self._repo.add(item)

    @staticmethod
    def _sync_status(item: Item, changes: dict) -> None:
        """Keep ``completed == (status == "done")`` whichever of the two the client sent.

        Unchecking a done todo returns it to "not started"; unchecking one that is in
        progress or archived leaves its status alone.
        """
        for field in ("status", "completed"):
            if field in changes and changes[field] is None:
                raise InvalidItemOperationError(f"a todo's {field} cannot be null")

        if "status" in changes:
            if "completed" in changes and changes["completed"] != is_completed(changes["status"]):
                raise InvalidItemOperationError(
                    "completed must be true exactly when status is 'done'"
                )
            changes["completed"] = is_completed(changes["status"])
        elif "completed" in changes:
            if changes["completed"]:
                changes["status"] = "done"
            elif item.status == "done":
                changes["status"] = DEFAULT_TODO_STATUS

    async def delete_item(self, item_id: str) -> None:
        item = await self._get_or_raise(item_id)
        await self._repo.delete(item)

    async def get_counts(self, *, tz: str | None = None) -> dict[str, int]:
        return await self._repo.counts(user_id=self._user_id, today_bounds=self._today_bounds(tz))

    async def _get_or_raise(self, item_id: str) -> Item:
        try:
            key = uuid.UUID(item_id)
        except ValueError:
            raise ItemNotFoundError(item_id) from None
        item = await self._repo.get(key, user_id=self._user_id)
        if item is None:
            raise ItemNotFoundError(item_id)
        return item

    @staticmethod
    def _today_bounds(tz: str | None) -> TodayBounds:
        """[local midnight, next local midnight) in UTC for the given IANA zone."""
        now_local = datetime.now(resolve_zone(tz))
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = start_local + timedelta(days=1)
        return start_local.astimezone(UTC), end_local.astimezone(UTC)
