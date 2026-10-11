"""Structured outputs the agent extracts (Trustcall tool schemas) and its routing tool."""

from datetime import datetime
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


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
        min_length=1,
        default_factory=list,
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started",
    )


MemoryKindName = Literal["note", "todo"]


class UpdateMemory(TypedDict):
    """Decision on what memory type to update"""

    update_type: Literal["note", "todo", "instructions"]


TodoPeriod = Literal[
    "all", "today", "tomorrow", "this_week", "next_week", "next_7_days", "dates"
]


class FindTodos(TypedDict):
    """Look up the user's open ToDos. Call this whenever the user ASKS about their ToDos:
    to list them, what they need to do, or what is due -- today, tomorrow, this week, on a
    date. It only reads; never call UpdateMemory for a question."""

    period: Annotated[
        TodoPeriod,
        ...,
        "Which ToDos to list. `all`: every open ToDo -- use it whenever the user names no "
        "day or range ('list all my todos', 'what do I need to do?'). `today`, "
        "`tomorrow`: that day. `this_week`, `next_week`: Monday to Sunday. `next_7_days`: "
        "today and the six days after. `dates`: any other day or range, given in start "
        "and end.",
    ]
    start: Annotated[
        str | None,
        None,
        "Only for period `dates`: first day, YYYY-MM-DD in the user's local time.",
    ]
    end: Annotated[
        str | None,
        None,
        "Only for period `dates`: last day (inclusive), YYYY-MM-DD in the user's local "
        "time. The same as start for a single day.",
    ]
