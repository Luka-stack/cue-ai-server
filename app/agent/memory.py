"""The agent's long-term memory for notes and todos, backed by the ``items`` table.

The items table is the single source of truth: the app edits it through the items API,
and the agent reads and writes the very same rows. Anything the user creates, edits,
completes or deletes in the app is what the agent sees on its next turn, in any thread.

The agent works with documents in its extractor schemas (``Note`` / ``ToDo`` in
``app.agent.schemas``), whose field names the prompts were validated against. This module
translates between those documents and item rows:

    Note.title             <-> item.title
    Note.content           <-> item.content
    ToDo.title             <-> item.title
    ToDo.content           <-> item.content
    ToDo.deadline          <-> item.deadline      (naive local time <-> UTC, via the user's tz)
    ToDo.time_to_complete  <-> item.time_to_complete
    ToDo.status            <-> item.completed     (item.completed follows: true if "completed")
"""

import logging
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, tzinfo
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.schemas import MemoryKindName, Note, ToDo
from app.agent.state import MEMORY_SEARCH_LIMIT, ProposalRecord
from app.models.item import Item
from app.repositories.item_repository import ItemRepository

logger = logging.getLogger(__name__)

Document = dict[str, Any]

# ----------------------------------------------------------------- item -> document


def _local_naive(value: datetime | None, zone: tzinfo) -> datetime | None:
    """UTC instant -> wall-clock time in the user's zone, as the model reads and writes it."""
    if value is None:
        return None
    return value.astimezone(zone).replace(tzinfo=None)


def note_document(item: Item) -> Document:
    return Note.model_validate(
        {"title": item.title or "", "content": item.content}
    ).model_dump(mode="json")


def todo_document(item: Item, zone: tzinfo) -> Document:
    doc: Document = {
        "title": item.title or "",
        "content": item.content,
        "time_to_complete": item.time_to_complete,
        "deadline": _local_naive(item.deadline, zone),
        "status": "completed" if item.completed else "todo",
    }
    # Round-tripping through the schema serializes exactly as the extractor's output
    # does, so an unchanged field compares equal in the review diff.
    return ToDo.model_validate(doc).model_dump(mode="json")


def to_document(kind: MemoryKindName, item: Item, zone: tzinfo) -> Document:
    return note_document(item) if kind == "note" else todo_document(item, zone)


def to_prompt_document(kind: MemoryKindName, item: Item, zone: tzinfo) -> Document:
    """What the chat model is shown."""
    return to_document(kind, item, zone)


# ----------------------------------------------------------------- document -> item


def _to_utc(value: str | None, zone: tzinfo) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        # The model writes wall-clock time ("end of November"), in the user's zone.
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(UTC)


def _blank_to_none(value: str | None) -> str | None:
    return value if value and value.strip() else None


def _positive_or_none(value: int | None) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _note_fields(key: str, value: Any, zone: tzinfo) -> dict[str, Any]:
    if key in ("title", "content"):
        return {key: _blank_to_none(value)}
    return {}


def _todo_fields(key: str, value: Any, zone: tzinfo) -> dict[str, Any]:
    if key == "title":
        return {"title": value}
    if key == "content":
        return {"content": _blank_to_none(value)}
    if key == "deadline":
        return {"deadline": _to_utc(value, zone)}
    if key == "time_to_complete":
        return {"time_to_complete": _positive_or_none(value)}
    if key == "status":
        return {"completed": value == "completed"}
    return {}


_FIELD_MAPPERS: dict[MemoryKindName, Callable[[str, Any, tzinfo], dict[str, Any]]] = {
    "note": _note_fields,
    "todo": _todo_fields,
}


def to_item_fields(
    kind: MemoryKindName,
    document: Document,
    zone: tzinfo,
    *,
    keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Item column values for ``document``, restricted to ``keys`` when given."""
    mapper = _FIELD_MAPPERS[kind]
    fields: dict[str, Any] = {}
    for key in document if keys is None else keys:
        fields.update(mapper(key, document.get(key), zone))
    return fields


def changed_keys(before: Document, after: Document) -> list[str]:
    return [key for key, value in after.items() if before.get(key) != value]


# ------------------------------------------------------------------------ the store


class ItemMemory:
    """Reads and writes one user's notes and todos as item rows.

    Each call runs in its own session and transaction: the graph runs outside any
    request's unit of work, and a turn can span several model calls.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def _items(self, kind: MemoryKindName, user_id: str) -> list[Item]:
        async with self._session_factory() as session:
            repo = ItemRepository(session)
            return list(
                await repo.list_by_type(
                    user_id=user_id, type_=kind, limit=MEMORY_SEARCH_LIMIT
                )
            )

    async def load(
        self, kind: MemoryKindName, user_id: str, zone: tzinfo
    ) -> dict[str, Document]:
        """``{item id: document}`` in the extractor's schema, for Trustcall to patch."""
        return {
            str(item.id): to_document(kind, item, zone)
            for item in await self._items(kind, user_id)
        }

    async def load_for_prompt(
        self, kind: MemoryKindName, user_id: str, zone: tzinfo
    ) -> list[Document]:
        return [
            to_prompt_document(kind, item, zone)
            for item in await self._items(kind, user_id)
        ]

    async def apply(
        self,
        kind: MemoryKindName,
        user_id: str,
        records: list[ProposalRecord],
        zone: tzinfo,
    ) -> None:
        """Write an accepted proposal, atomically.

        Updates write only the fields the proposal changes (``value`` against the
        ``before`` the user reviewed). If the user edited the same item in the app while
        the review was pending, their other edits are kept instead of being overwritten
        by the draft's stale copy of them.
        """
        async with self._session_factory() as session, session.begin():
            repo = ItemRepository(session)
            ids = {rec["doc_id"]: _as_uuid(rec["doc_id"]) for rec in records}
            existing = {
                str(item.id): item
                for item in await repo.get_many(list(ids.values()), user_id=user_id)
            }

            for rec in records:
                item = existing.get(str(ids[rec["doc_id"]]))
                if item is None:
                    if not rec["is_new"]:
                        # Deleted in the app while the review was pending. The user just
                        # accepted this exact document, so it is recreated.
                        logger.info(
                            "recreating %s %s deleted during review",
                            kind,
                            rec["doc_id"],
                        )

                    fields = to_item_fields(kind, rec["value"], zone)

                    if kind == "todo":
                        if rec["is_new"]:
                            fields["completed"] = False
                        else:
                            fields.setdefault("completed", False)

                    session.add(
                        Item(
                            id=ids[rec["doc_id"]], user_id=user_id, type=kind, **fields
                        )
                    )
                    continue

                keys = changed_keys(rec["before"], rec["value"])
                for field, value in to_item_fields(
                    kind, rec["value"], zone, keys=keys
                ).items():
                    setattr(item, field, value)


def _as_uuid(doc_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(doc_id)
    except ValueError:
        # Only reachable if the model invented a non-UUID id for a new document.
        logger.warning("replacing non-UUID document id %r", doc_id)
        return uuid.uuid4()
