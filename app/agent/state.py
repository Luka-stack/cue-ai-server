"""Graph state, the per-kind memory registry, and the pure helpers the nodes share."""

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel

from app.agent.prompts import NOTE_RULES, TODO_RULES
from app.agent.schemas import MemoryKindName, Note, ToDo

# Upper bound on how many notes (and, separately, todos) are loaded into a prompt. Memory
# is small per user, so it is loaded whole; the cap only keeps a runaway list from
# overflowing the model's context window.
MEMORY_SEARCH_LIMIT = 200

ReviewAction = Literal["accept", "reject", "revise"]


class ProposalRecord(TypedDict):
    """One document the extractor wants to write: a new one or a patch to an existing one."""

    doc_id: str
    is_new: bool
    before: dict[str, Any]
    value: dict[str, Any]


class Proposal(TypedDict):
    """A staged, not-yet-saved write waiting for the user's decision."""

    kind: MemoryKindName
    tool_call_id: str
    records: list[ProposalRecord]
    summary: str


class MemoryState(MessagesState):
    """MessagesState plus the write that is currently waiting for the user's approval."""

    # Staged by propose_memory, consumed by save_memory, cleared once the turn is resolved.
    proposal: Proposal | None
    # Everything the user asked to change while reviewing this turn's draft.
    review_feedback: list[str]


@dataclass(frozen=True)
class MemoryKind:
    """Everything that differs between the reviewable memory types.

    Both are stored as rows of the ``items`` table, with ``type`` equal to the kind name.
    """

    label: str
    tool_name: str
    schema: type[BaseModel]
    rules: str


MEMORY_KINDS: dict[MemoryKindName, MemoryKind] = {
    "note": MemoryKind("note", "Note", Note, NOTE_RULES),
    "todo": MemoryKind("ToDo", "ToDo", ToDo, TODO_RULES),
}

# Instructions are agent-internal (never shown in the app), so they stay in the LangGraph
# store under ("instructions", user_id).
INSTRUCTIONS_COLLECTION = "instructions"
INSTRUCTIONS_KEY = "user_instructions"


def conversation_only(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Strip the agent's own UpdateMemory plumbing out of the history."""
    kept: list[AnyMessage] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, AIMessage) and m.tool_calls:
            if not m.content:
                continue
            m = AIMessage(content=m.content)  # keep the prose, drop the tool call
        kept.append(m)
    return kept


def _format_fields(value: dict[str, Any]) -> str:
    """Render one extracted document, skipping the fields the model left empty."""
    filled = [(k, v) for k, v in value.items() if v not in (None, "", [], {})]
    return "\n".join(f"        {k}: {v}" for k, v in filled) or "        (empty)"


def _format_changes(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Render only the fields a patch actually changes, so the review stays skimmable."""
    changed = [(k, before.get(k), v) for k, v in after.items() if before.get(k) != v]
    return (
        "\n".join(f"        {k}: {old!r}  ->  {new!r}" for k, old, new in changed)
        or "        (no field actually changes)"
    )


def format_proposal(kind: MemoryKindName, records: list[ProposalRecord]) -> str:
    """Render the pending write as something a human can approve at a glance."""
    label = MEMORY_KINDS[kind].label
    blocks = []
    for i, rec in enumerate(records, 1):
        if rec["is_new"]:
            blocks.append(f"    [{i}] NEW {label}\n" + _format_fields(rec["value"]))
        else:
            blocks.append(
                f"    [{i}] UPDATE existing {label}  ({rec['doc_id']})\n"
                + _format_changes(rec["before"], rec["value"])
            )
    return "\n\n".join(blocks)


_ACCEPT = {
    "y", "yes", "yep", "yeah", "ok", "okay", "accept", "approve", "approved",
    "save", "go", "sure", "looks good", "lgtm",
}  # fmt: skip
_REJECT = {
    "n", "no", "nope", "reject", "cancel", "discard", "skip", "dont", "don't",
    "never mind", "nevermind",
}  # fmt: skip


def parse_decision(decision: Any) -> tuple[ReviewAction, str]:
    """Turn whatever the graph was resumed with into an (action, feedback) pair.

    Accepts a structured ``{"action": ..., "feedback": ...}`` dict (what the API sends) or
    a plain string -- "yes" / "no", and anything else is read as "change it like this".
    """
    if isinstance(decision, dict):
        action = (decision.get("action") or "revise").strip().lower()
        feedback = (decision.get("feedback") or "").strip()
        if action not in ("accept", "reject", "revise"):
            action = "revise"
        if action == "revise" and not feedback:
            raise ValueError("A 'revise' decision needs feedback describing the change.")
        return action, feedback  # type: ignore[return-value]

    text = str(decision).strip()
    key = text.lower().rstrip(".!")
    if key in _ACCEPT:
        return "accept", ""
    if key in _REJECT:
        return "reject", ""
    return "revise", text
