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

    title: str = Field(description="A short one-line summary of the task.")
    content: str | None = Field(
        default=None,
        description="Details of the task. Only when the task is complicated enough to need "
        "a plan, add short list of specific, actionable solutions (ideas, service "
        "providers, concrete options). Leave null for simple tasks.",
    )
    deadline: datetime | None = Field(
        description="When the task needs to be completed by, if the user stated or clearly "
        "implied one. Leave this null otherwise -- never invent a date.",
        default=None,
    )
    status: Literal["todo", "completed"] = Field(
        default="todo",
        description="'completed' only when the user says the task is done.",
    )
    time_to_complete: int | None = Field(
        description="Estimated time to complete the task (minutes).",
        default=None,
    )

    @field_validator("deadline")
    @classmethod
    def _wall_clock(cls, value: datetime | None) -> datetime | None:
        """Deadlines are the user's local wall-clock time. The model tends to add an offset
        (usually "Z", as the date-time format asks for); it is dropped, not converted.
        """
        return value.replace(tzinfo=None) if value else value


MemoryKindName = Literal["note", "todo"]


class UpdateMemory(TypedDict):
    """Decision on what memory type to update"""

    update_type: Literal["note", "todo", "instructions"]
