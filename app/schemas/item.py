"""Pydantic schemas mirroring the client contract in cue-ai/models/index.ts.

Notes never expose ``deadline``/``completed`` (the fields are absent from the
model, not null), and todo dates serialize as UTC ISO-8601 with a ``Z``
suffix — exactly what the React Native client's mocks contain.
"""

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)


def _require_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError(
            "datetime must include a timezone offset (e.g. 2026-07-27T10:00:00Z)"
        )
    return dt


def _serialize_utc_z(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


UTCDateTime = Annotated[
    datetime,
    AfterValidator(_require_aware),
    PlainSerializer(_serialize_utc_z, return_type=str, when_used="json"),
]

# The client contract says id is a string; the DB stores a UUID.
IdStr = Annotated[str, BeforeValidator(str)]

# Todo-only extras the chat agent fills in; the app may edit them too.
Minutes = Annotated[int, Field(gt=0, le=60 * 24 * 365)]


class _SchemaBase(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


# --------------------------------------------------------------------------- read


class NoteRead(_SchemaBase):
    id: IdStr
    type: Literal["note"]
    title: str | None = None
    content: str | None = None
    updated_at: UTCDateTime


class ToDoRead(_SchemaBase):
    id: IdStr
    type: Literal["todo"]
    title: str
    deadline: UTCDateTime | None = None
    content: str | None = None
    completed: bool = False
    time_to_complete: int | None = None
    updated_at: UTCDateTime


ItemRead = Annotated[NoteRead | ToDoRead, Field(discriminator="type")]


class CountsRead(BaseModel):
    today: int
    notes: int
    all: int
    completed: int


# ------------------------------------------------------------------------- create


class NoteCreate(_SchemaBase):
    type: Literal["note"]
    title: str | None = None
    content: str | None = None

    @model_validator(mode="after")
    def _require_title_or_content(self) -> "NoteCreate":
        if not (self.title or "").strip() and not (self.content or "").strip():
            raise ValueError("a note requires a non-empty title or content")
        return self


class ToDoCreate(_SchemaBase):
    type: Literal["todo"]
    title: str = Field(min_length=1)
    deadline: UTCDateTime | None = None
    content: str | None = None
    completed: bool = False
    time_to_complete: Minutes | None = None

    @model_validator(mode="after")
    def _require_nonblank_title(self) -> "ToDoCreate":
        if not self.title.strip():
            raise ValueError("a todo requires a non-empty title")
        return self


ItemCreate = Annotated[NoteCreate | ToDoCreate, Field(discriminator="type")]


# ------------------------------------------------------------------------- update


class ItemUpdate(_SchemaBase):
    """Partial update. Type-specific rules are enforced in the service layer,
    since the target item's type is only known after loading it."""

    title: str | None = None
    content: str | None = None
    deadline: UTCDateTime | None = None
    completed: bool | None = None
    time_to_complete: Minutes | None = None
