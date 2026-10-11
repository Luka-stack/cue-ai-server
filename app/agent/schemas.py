"""Structured outputs the agent extracts (Trustcall tool schemas) and its routing tool."""

from datetime import datetime
from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


class Note(BaseModel):
    """A note the user asked to remember."""

    title: str = Field(description="A brief one-line summary of the note.")
    content: str | None = Field(
        description="The body of the note. Leave null when the title alone already says "
        "everything.",
        default=None,
    )


class ToDo(BaseModel):
    """A task the user needs to complete."""

    task: str = Field(description="The task to be completed.")
    time_to_complete: int | None = Field(
        description="Estimated time to complete the task (minutes).",
        default=None,
    )
    deadline: datetime | None = Field(
        description="When the task needs to be completed by, if the user stated or clearly "
        "implied one. Leave this null otherwise -- never invent a date.",
        default=None,
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service "
        "providers, or concrete options relevant to completing the task)",
        default_factory=list,
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started",
    )

    @field_validator("deadline")
    @classmethod
    def _wall_clock(cls, value: datetime | None) -> datetime | None:
        """Deadlines are the user's local wall-clock time. The model tends to add an offset
        (usually "Z", as the date-time format asks for); it is dropped, not converted."""
        return value.replace(tzinfo=None) if value else value


MemoryKindName = Literal["note", "todo"]


class UpdateMemory(TypedDict):
    """Decision on what memory type to update"""

    update_type: Literal["note", "todo", "instructions"]
