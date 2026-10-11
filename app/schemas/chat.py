"""API contract for the chat agent and its long-term memory."""

from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

IdStr = Annotated[str, BeforeValidator(str)]


class _SchemaBase(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


# ------------------------------------------------------------------------- threads


class ThreadCreate(_SchemaBase):
    title: str | None = Field(default=None, max_length=200)


class ThreadRead(_SchemaBase):
    id: IdStr
    title: str | None = None
    created_at: datetime
    updated_at: datetime


# ------------------------------------------------------------------------ messages


class MessageCreate(_SchemaBase):
    content: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def _require_nonblank(self) -> "MessageCreate":
        if not self.content.strip():
            raise ValueError("message content must not be blank")
        return self


class ToolCallRead(BaseModel):
    name: str
    args: dict[str, Any]


class ChatMessageRead(BaseModel):
    """One message of the conversation, in the shape the client renders."""

    id: str
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCallRead] | None = None


def _text_of(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # Multimodal / block content: keep only the text blocks.
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def to_chat_message(message: AnyMessage) -> ChatMessageRead | None:
    """Map a LangChain message to the API shape; ``None`` for kinds the client never shows."""
    if isinstance(message, HumanMessage):
        return ChatMessageRead(id=message.id or "", role="user", content=_text_of(message))
    if isinstance(message, AIMessage):
        tool_calls = [ToolCallRead(name=tc["name"], args=tc["args"]) for tc in message.tool_calls]
        return ChatMessageRead(
            id=message.id or "",
            role="assistant",
            content=_text_of(message),
            tool_calls=tool_calls or None,
        )
    if isinstance(message, ToolMessage):
        return ChatMessageRead(id=message.id or "", role="tool", content=_text_of(message))
    return None


def is_user_visible(message: ChatMessageRead) -> bool:
    """Hide the agent's internal plumbing: tool results and tool-call-only assistant turns."""
    if message.role == "tool":
        return False
    return bool(message.content.strip())


# -------------------------------------------------------------------------- review


class ProposalRecordRead(BaseModel):
    doc_id: str
    is_new: bool
    before: dict[str, Any]
    value: dict[str, Any]


class PendingReviewRead(BaseModel):
    """A memory write the agent staged and is waiting for the user to approve."""

    kind: Literal["note", "todo"]
    action_request: str
    summary: str
    help: str
    records: list[ProposalRecordRead]


class ReviewDecision(_SchemaBase):
    action: Literal["accept", "reject", "revise"]
    feedback: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _revise_needs_feedback(self) -> "ReviewDecision":
        if self.action == "revise" and not (self.feedback or "").strip():
            raise ValueError("a 'revise' decision needs feedback describing the change")
        return self


TurnStatus = Literal["completed", "awaiting_review"]


class TurnRead(BaseModel):
    """Result of sending a message or answering a review."""

    thread: ThreadRead
    status: TurnStatus
    # Only the messages this turn added, so the client can append rather than reload.
    messages: list[ChatMessageRead]
    pending_review: PendingReviewRead | None = None


class ThreadDetailRead(BaseModel):
    thread: ThreadRead
    status: TurnStatus
    messages: list[ChatMessageRead]
    pending_review: PendingReviewRead | None = None


# -------------------------------------------------------------------------- memory

# Notes and todos are items: read them through /api/v1/items.
MemoryCollection = Literal["instructions"]


class MemoryRead(BaseModel):
    id: str
    collection: MemoryCollection
    value: dict[str, Any]
    created_at: datetime
    updated_at: datetime
