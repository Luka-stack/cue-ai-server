import uuid
from datetime import datetime
from typing import Literal, get_args

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# A todo's progress, as the chat agent tracks it. ``completed`` is kept equal to
# ``status == "done"`` (a CHECK constraint enforces it), so clients that only know the
# boolean keep working.
TodoStatus = Literal["not started", "in progress", "done", "archived"]
TODO_STATUSES: tuple[str, ...] = get_args(TodoStatus)
DEFAULT_TODO_STATUS: TodoStatus = "not started"


def is_completed(status: str) -> bool:
    return status == "done"


_STATUS_SQL = ", ".join(f"'{s}'" for s in TODO_STATUSES)


class Item(Base):
    """Single table for both item types; the ``type`` column discriminates.

    This table is the single source of truth for notes and todos: the app edits it
    through the items API, and the chat agent reads and writes it as its long-term
    memory. ``date``, ``completed``, ``status``, ``solutions`` and ``time_to_complete``
    are only meaningful for todos — a CHECK constraint keeps notes from ever carrying them.
    """

    __tablename__ = "items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    type: Mapped[str] = mapped_column(String(8), nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # One of TODO_STATUSES for todos, NULL for notes.
    status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Concrete options for getting a todo done (the agent suggests these).
    solutions: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    # Estimated effort in minutes.
    time_to_complete: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint("type IN ('note', 'todo')", name="type_valid"),
        CheckConstraint(
            "type <> 'note' OR (date IS NULL AND completed IS NULL AND status IS NULL "
            "AND solutions IS NULL AND time_to_complete IS NULL)",
            name="note_has_no_todo_fields",
        ),
        CheckConstraint(f"status IS NULL OR status IN ({_STATUS_SQL})", name="status_valid"),
        CheckConstraint(
            "type <> 'todo' OR (status IS NOT NULL AND completed IS NOT NULL "
            "AND completed = (status = 'done'))",
            name="todo_status_matches_completed",
        ),
        CheckConstraint(
            "time_to_complete IS NULL OR time_to_complete > 0", name="time_to_complete_positive"
        ),
        Index("ix_items_user_id_type_date", "user_id", "type", "date"),
    )
