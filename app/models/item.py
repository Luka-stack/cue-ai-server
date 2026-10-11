import uuid
from datetime import datetime

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
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Item(Base):
    """Single table for both item types; the ``type`` column discriminates.

    This table is the single source of truth for notes and todos: the app edits it
    through the items API, and the chat agent reads and writes it as its long-term
    memory. ``deadline``, ``completed``, and ``time_to_complete``
    are only meaningful for todos — a CHECK constraint keeps notes from ever carrying them.
    """

    __tablename__ = "items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    type: Mapped[str] = mapped_column(String(8), nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    deadline: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
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
            "type <> 'note' OR (deadline IS NULL AND completed IS NULL AND time_to_complete IS NULL)",
            name="note_has_no_todo_fields",
        ),
        CheckConstraint(
            "type <> 'todo' OR completed IS NOT NULL", name="todo_has_completed"
        ),
        CheckConstraint(
            "time_to_complete IS NULL OR time_to_complete > 0",
            name="time_to_complete_positive",
        ),
        Index("ix_items_user_id_type_deadline", "user_id", "type", "deadline"),
    )
