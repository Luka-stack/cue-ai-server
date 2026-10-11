"""Graph nodes. Ported from the notebook; made async, with injected dependencies.

The chat model is injected rather than imported so the graph can be built against a
fake model in tests and against Ollama in production. Notes and todos are read from and
written to the ``items`` table through :class:`ItemMemory` (the same rows the app edits);
only the ToDo-handling instructions stay in the LangGraph store.
"""

import logging
import uuid
from datetime import date, datetime, timedelta, tzinfo
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    merge_message_runs,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt
from trustcall import create_extractor

from app.agent.memory import ItemMemory
from app.agent.prompts import (
    CREATE_INSTRUCTIONS,
    MODEL_SYSTEM_MESSAGE,
    REVISION_RULES,
    trustcall_instruction,
)
from app.agent.schemas import FindTodos, UpdateMemory
from app.agent.state import (
    INSTRUCTIONS_COLLECTION,
    INSTRUCTIONS_KEY,
    MEMORY_KINDS,
    MemoryState,
    Proposal,
    ProposalRecord,
    conversation_only,
    format_proposal,
    parse_decision,
)
from app.core.timezone import resolve_zone

logger = logging.getLogger(__name__)

FIND_TODOS = FindTodos.__name__


def _user_id(config: RunnableConfig) -> str:
    try:
        return config["configurable"]["user_id"]
    except KeyError:  # pragma: no cover - programming error, not a runtime condition
        raise RuntimeError("graph config must carry configurable.user_id") from None


def _extractor_view(document: dict) -> dict:
    """The document as handed to Trustcall for patching, minus empty lists.

    ``ToDo.solutions`` requires at least one entry when present, and falls back to its
    default only when absent. A todo created in the app has none, so an explicit ``[]``
    would make every patch fail validation and send Trustcall into retries.
    """
    return {key: value for key, value in document.items() if value != []}


def _zone(config: RunnableConfig) -> tzinfo:
    """The user's time zone; deadlines are read and written in it. Validated at the API."""
    return resolve_zone(config["configurable"].get("tz"))


class MemoryAgentNodes:
    """The five nodes of the memory agent, bound to one chat model and one memory."""

    def __init__(self, model: BaseChatModel, memory: ItemMemory) -> None:
        self._model = model
        self._memory = memory

    # ------------------------------------------------------------------ conversation

    async def task_maistro(
        self, state: MemoryState, config: RunnableConfig, store: BaseStore
    ) -> dict:
        """Load memories and use them to personalize the chatbot's response."""
        user_id = _user_id(config)
        zone = _zone(config)

        # Read fresh on every turn, so edits made in the app are picked up immediately.
        notes_docs = await self._memory.load_for_prompt("note", user_id, zone)
        notes = "\n".join(f"{doc}" for doc in notes_docs)

        todo_docs = await self._memory.load_for_prompt("todo", user_id, zone)
        todo = "\n".join(f"{doc}" for doc in todo_docs)

        instructions_items = await store.asearch((INSTRUCTIONS_COLLECTION, user_id), limit=1)
        instructions = instructions_items[0].value if instructions_items else ""

        # The user's wall-clock time, so relative dates ("tomorrow") resolve in their zone.
        now = datetime.now(zone).strftime("%A, %Y-%m-%d %H:%M")
        system_msg = MODEL_SYSTEM_MESSAGE.format(
            now=now, notes=notes, todo=todo, instructions=instructions
        )

        messages = [SystemMessage(content=system_msg)] + state["messages"]
        last = state["messages"][-1]
        if isinstance(last, ToolMessage) and last.name == FIND_TODOS:
            # The lookup is done: answer from it. Without tools the model can't loop
            # back into another lookup or turn the question into a save.
            response = await self._model.ainvoke(messages)
        else:
            response = await self._model.bind_tools([UpdateMemory, FindTodos]).ainvoke(
                messages
            )

        # Ollama has no `parallel_tool_calls=False`, so enforce a single tool call by hand:
        # route_message and the update nodes only ever handle tool_calls[0].
        if len(response.tool_calls) > 1:
            response.tool_calls = response.tool_calls[:1]

        return {"messages": [response]}

    # --------------------------------------------------------------- human in the loop

    async def propose_memory(
        self, state: MemoryState, config: RunnableConfig
    ) -> Command[Literal["review_memory", "task_maistro"]]:
        """Extract the note / ToDo the user asked for and stage it -- nothing is written yet."""
        user_id = _user_id(config)
        zone = _zone(config)

        # task_maistro already decided which kind of memory this turn belongs to
        tool_call = state["messages"][-1].tool_calls[0]
        kind = tool_call["args"]["update_type"]
        spec = MEMORY_KINDS[kind]

        # {item id: document in the extractor's schema}
        stored = await self._memory.load(kind, user_id, zone)

        # Anything the user asked for while reviewing an earlier draft of this same turn.
        # This is deliberately kept out of `messages`: a HumanMessage sitting between the
        # AI tool call and its ToolMessage would break the pairing the chat model expects.
        feedback = state.get("review_feedback") or []
        draft = state.get("proposal") if feedback else None

        if draft:
            # Revising: show the extractor ONLY the draft it just produced. Handing it the
            # whole store makes qwen3 merge the unsaved draft into whichever document
            # happens to be there already.
            existing_memories = [
                (rec["doc_id"], spec.tool_name, _extractor_view(rec["value"]))
                for rec in draft["records"]
            ]
            in_scope: set[str] | None = {rec["doc_id"] for rec in draft["records"]}
            rules = spec.rules + REVISION_RULES
        else:
            existing_memories = (
                [(key, spec.tool_name, _extractor_view(value)) for key, value in stored.items()]
                if stored
                else None
            )
            in_scope = None
            rules = spec.rules

        feedback_msgs = [
            HumanMessage(content=f"The user reviewed your draft and asked for this change: {f}")
            for f in feedback
        ]

        updated_messages = list(
            merge_message_runs(
                messages=[SystemMessage(content=trustcall_instruction(rules))]
                + conversation_only(state["messages"][:-1])
                + feedback_msgs
            )
        )

        extractor = create_extractor(
            self._model,
            tools=[spec.schema],
            tool_choice=spec.tool_name,
            enable_inserts=True,
        )
        result = await extractor.ainvoke(
            {"messages": updated_messages, "existing": existing_memories}
        )

        # The extractor hands back the finished documents, so the review can show the real
        # end state rather than the raw PatchDoc calls.
        records: list[ProposalRecord] = []
        for r, rmeta in zip(result["responses"], result["response_metadata"], strict=True):
            doc_id = rmeta.get("json_doc_id") or str(uuid.uuid4())
            if in_scope is not None and doc_id in stored and doc_id not in in_scope:
                # A revision may only change the draft under review; a revision that
                # silently rewrites a memory the user was not reviewing is the worst
                # outcome the review step can produce.
                logger.warning(
                    "dropped a patch to %s: a revision may only change the draft under review",
                    doc_id,
                )
                continue
            records.append(
                {
                    "doc_id": doc_id,
                    "is_new": doc_id not in stored,
                    "before": stored.get(doc_id, {}),
                    "value": r.model_dump(mode="json"),
                }
            )

        if not records and draft:
            # The correction produced nothing usable -- prose instead of a tool call, or
            # only patches that were out of scope. Put the previous draft back in front of
            # the user instead of silently dropping the memory they asked for.
            logger.warning("the last correction could not be applied; re-presenting the draft")
            summary = (
                "    (your last correction could not be applied -- this is the previous "
                "draft, reword it or accept it as is)\n\n" + draft["summary"]
            )
            return Command(goto="review_memory", update={"proposal": {**draft, "summary": summary}})

        if not records:
            # Ollama cannot force a tool call, so the extractor sometimes answers in prose.
            return Command(
                goto="task_maistro",
                update={
                    "messages": [
                        ToolMessage(
                            content=f"The extractor returned no {spec.label} -- there is "
                            "nothing to review or save.",
                            tool_call_id=tool_call["id"],
                        )
                    ],
                    "proposal": None,
                    "review_feedback": [],
                },
            )

        touched = {rec["doc_id"] for rec in records if not rec["is_new"]}
        if len(touched) > 1:
            # qwen3 occasionally overgeneralizes a detail mentioned for one item to every
            # existing document in the turn. The review is the real backstop.
            logger.warning(
                "turn would touch %d existing %s documents at once (%s)",
                len(touched),
                spec.label,
                touched,
            )

        proposal: Proposal = {
            "kind": kind,
            "tool_call_id": tool_call["id"],
            "records": records,
            "summary": format_proposal(kind, records),
        }
        return Command(goto="review_memory", update={"proposal": proposal})

    async def review_memory(
        self, state: MemoryState
    ) -> Command[Literal["save_memory", "propose_memory", "task_maistro"]]:
        """Pause the graph and hand the staged write to the user for approval."""
        proposal = state["proposal"]
        assert proposal is not None, "review_memory reached without a proposal"
        spec = MEMORY_KINDS[proposal["kind"]]

        # Everything above this line is a deterministic read of state, so resuming re-runs
        # the node without re-running the extractor: what the user approved is exactly
        # what is saved.
        decision = interrupt(
            {
                "action_request": f"save this {spec.label}",
                "kind": proposal["kind"],
                "summary": proposal["summary"],
                "help": "Resume with 'yes' to save, 'no' to discard, or describe the "
                "change you want.",
            }
        )

        action, feedback = parse_decision(decision)

        if action == "accept":
            return Command(goto="save_memory")

        if action == "reject":
            return Command(
                goto="task_maistro",
                update={
                    "messages": [
                        ToolMessage(
                            content=f"The user reviewed the proposed {spec.label} and "
                            "declined it. Nothing was saved.",
                            tool_call_id=proposal["tool_call_id"],
                        )
                    ],
                    "proposal": None,
                    "review_feedback": [],
                },
            )

        # Re-extract with the correction in hand; the new draft is reviewed as well.
        # `proposal` is deliberately left in state: propose_memory revises that draft
        # rather than extracting again from scratch against the whole store.
        return Command(
            goto="propose_memory",
            update={"review_feedback": (state.get("review_feedback") or []) + [feedback]},
        )

    async def save_memory(self, state: MemoryState, config: RunnableConfig) -> dict:
        """Write the approved proposal to the items table, where the app sees it."""
        proposal = state["proposal"]
        assert proposal is not None, "save_memory reached without a proposal"

        await self._memory.apply(
            proposal["kind"], _user_id(config), proposal["records"], _zone(config)
        )

        return {
            "messages": [
                ToolMessage(
                    content="The user approved these changes and they are now saved:\n"
                    f"{proposal['summary']}",
                    tool_call_id=proposal["tool_call_id"],
                )
            ],
            "proposal": None,
            "review_feedback": [],
        }

    # ------------------------------------------------------------------ questions

    async def find_todos(self, state: MemoryState, config: RunnableConfig) -> dict:
        """Answer "what is due ..." with an exact, date-filtered read of the todos.

        The model only names the period; the dates are worked out here, because a small
        model's date arithmetic is not reliable.
        """
        tool_call = state["messages"][-1].tool_calls[0]
        zone = _zone(config)
        start, end, label = _period_bounds(tool_call["args"], datetime.now(zone).date())

        todos = await self._memory.find_todos(_user_id(config), zone, start, end)

        lines = [
            f"- {todo['task']} (due: {_spoken_deadline(todo.get('deadline'))}"
            + (f", status: {todo['status']}" if todo["status"] != "not started" else "")
            + ")"
            for todo in todos
        ]
        content = f"{label}:\n" + "\n".join(lines) if lines else f"{label}: none."

        return {
            "messages": [
                ToolMessage(content=content, name=FIND_TODOS, tool_call_id=tool_call["id"])
            ]
        }

    # ------------------------------------------------------------------ instructions

    async def update_instructions(
        self, state: MemoryState, config: RunnableConfig, store: BaseStore
    ) -> dict:
        """Reflect on the chat history and update the ToDo-handling instructions."""
        user_id = _user_id(config)
        namespace = (INSTRUCTIONS_COLLECTION, user_id)

        existing = await store.aget(namespace, INSTRUCTIONS_KEY)
        system_msg = CREATE_INSTRUCTIONS.format(
            current_instructions=existing.value if existing else None
        )
        new_memory = await self._model.ainvoke(
            [SystemMessage(content=system_msg)]
            + conversation_only(state["messages"][:-1])
            + [HumanMessage(content="Please update the instructions based on the conversation")]
        )

        await store.aput(namespace, INSTRUCTIONS_KEY, {"memory": new_memory.content})

        tool_call = state["messages"][-1].tool_calls[0]
        return {
            "messages": [ToolMessage(content="updated instructions", tool_call_id=tool_call["id"])]
        }


def _spoken_deadline(value: str | None) -> str:
    """``2026-10-10T23:00:00`` -> ``Saturday, October 10, 2026 at 11:00 PM``.

    Written out so the model copies it instead of converting a 24-hour clock itself,
    which it gets wrong now and then.
    """
    if not value:
        return "no deadline"
    moment = datetime.fromisoformat(value)
    hour = moment.hour % 12 or 12
    meridiem = "AM" if moment.hour < 12 else "PM"
    return (
        f"{moment:%A, %B} {moment.day}, {moment.year} at {hour}:{moment:%M} {meridiem}"
    )


def _parse_day(value: object) -> date | None:
    """A ``YYYY-MM-DD`` tool argument, or None when missing or malformed."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _spoken_day(day: date) -> str:
    return f"{day:%A, %B} {day.day}"


def _period_bounds(args: dict, today: date) -> tuple[date | None, date | None, str]:
    """FindTodos arguments -> (first day, last day, heading for the result).

    Unknown or incomplete input falls back to every open todo: listing too much is a
    better failure than silently hiding todos.
    """
    period = args.get("period")
    monday = today - timedelta(days=today.weekday())

    if period == "today":
        return today, today, f"Open ToDos due today ({_spoken_day(today)})"
    if period == "tomorrow":
        day = today + timedelta(days=1)
        return day, day, f"Open ToDos due tomorrow ({_spoken_day(day)})"
    if period == "this_week":
        sunday = monday + timedelta(days=6)
        return monday, sunday, f"Open ToDos due this week (through {_spoken_day(sunday)})"
    if period == "next_week":
        start, end = monday + timedelta(days=7), monday + timedelta(days=13)
        return (
            start,
            end,
            f"Open ToDos due next week ({_spoken_day(start)} to {_spoken_day(end)})",
        )
    if period == "next_7_days":
        end = today + timedelta(days=6)
        return today, end, f"Open ToDos due in the next 7 days (through {_spoken_day(end)})"
    if period == "dates":
        start, end = _parse_day(args.get("start")), _parse_day(args.get("end"))
        start, end = start or end, end or start
        if start and end:
            start, end = min(start, end), max(start, end)
            span = (
                _spoken_day(start)
                if start == end
                else f"{_spoken_day(start)} to {_spoken_day(end)}"
            )
            return start, end, f"Open ToDos due {span}"

    return None, None, "All open ToDos"


def route_message(  # type: ignore[valid-type]
    state: MemoryState,
) -> Literal[END, "propose_memory", "update_instructions", "find_todos"]:
    """Decide whether the last assistant turn asked to update memory, and which kind."""
    message = state["messages"][-1]
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return END

    if tool_calls[0]["name"] == FIND_TODOS:
        # A read: no review needed, the answer comes back through task_maistro.
        return "find_todos"

    # bind_tools does not validate args, and qwen3 occasionally adds keys of its own,
    # so read update_type defensively rather than trusting the schema.
    update_type = tool_calls[0]["args"].get("update_type")
    if update_type in MEMORY_KINDS:
        # Notes and ToDos are user-visible memory, so they go through review first.
        return "propose_memory"
    if update_type == "instructions":
        # Instructions are internal plumbing the user is never told about, so they are
        # written straight through.
        return "update_instructions"

    logger.warning("unexpected update_type %r; ending turn", update_type)
    return END
