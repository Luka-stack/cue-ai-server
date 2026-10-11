"""Graph nodes. Ported from the notebook; made async, with injected dependencies.

The chat model is injected rather than imported so the graph can be built against a
fake model in tests and against OpenAI in production. Notes and todos are read from and
written to the ``items`` table through :class:`ItemMemory` (the same rows the app edits);
only the ToDo-handling instructions stay in the LangGraph store.
"""

import uuid
from datetime import datetime, tzinfo
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
from app.agent.schemas import UpdateMemory
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


def _user_id(config: RunnableConfig) -> str:
    try:
        return config["configurable"]["user_id"]
    except KeyError:  # pragma: no cover - programming error, not a runtime condition
        raise RuntimeError("graph config must carry configurable.user_id") from None


def _zone(config: RunnableConfig) -> tzinfo:
    """The user's time zone; deadlines are read and written in it. Validated at the API."""
    return resolve_zone(config["configurable"].get("tz"))


class MemoryAgentNodes:
    """The nodes of the memory agent, bound to one chat model and one memory."""

    def __init__(self, model: BaseChatModel, memory: ItemMemory) -> None:
        self._model = model
        self._memory = memory
        # strict: the arguments always match UpdateMemory, so route_message can trust them.
        # One tool call per turn: route_message and the update nodes handle tool_calls[0].
        self._chat_model = model.bind_tools(
            [UpdateMemory], strict=True, parallel_tool_calls=False
        )

    # ------------------------------------------------------------------ conversation

    async def cue_brain(
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

        instructions_items = await store.asearch(
            (INSTRUCTIONS_COLLECTION, user_id), limit=1
        )
        instructions = instructions_items[0].value if instructions_items else ""

        # The user's wall-clock time, so relative dates ("tomorrow") resolve in their zone.
        now = datetime.now(zone).strftime("%A, %Y-%m-%d %H:%M")
        system_msg = MODEL_SYSTEM_MESSAGE.format(
            now=now, notes=notes, todo=todo, instructions=instructions
        )

        response = await self._chat_model.ainvoke(
            [SystemMessage(content=system_msg)] + state["messages"]
        )
        return {"messages": [response]}

    # --------------------------------------------------------------- human in the loop

    async def propose_memory(
        self, state: MemoryState, config: RunnableConfig
    ) -> Command[Literal["review_memory", "cue_brain"]]:
        """Extract the note / ToDo the user asked for and stage it -- nothing is written yet."""
        user_id = _user_id(config)
        zone = _zone(config)

        # cue_brain already decided which kind of memory this turn belongs to
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
            # Revising: the extractor patches the draft it just produced, not the store.
            existing = [
                (rec["doc_id"], spec.tool_name, rec["value"])
                for rec in draft["records"]
            ]
            rules = spec.rules + REVISION_RULES
        else:
            existing = [(key, spec.tool_name, value) for key, value in stored.items()]
            rules = spec.rules

        messages = list(
            merge_message_runs(
                [
                    SystemMessage(
                        content=trustcall_instruction(rules, now=datetime.now(zone))
                    )
                ]
                + conversation_only(state["messages"][:-1])
                + [
                    HumanMessage(
                        f"The user reviewed your draft and asked for this change: {f}"
                    )
                    for f in feedback
                ]
            )
        )

        extractor = create_extractor(
            self._model,
            tools=[spec.schema],
            tool_choice=spec.tool_name,
            enable_inserts=True,
        )
        result = await extractor.ainvoke(
            {"messages": messages, "existing": existing or None}
        )

        # The extractor hands back the finished documents, so the review can show the real
        # end state rather than the raw PatchDoc calls.
        records: list[ProposalRecord] = []
        for r, rmeta in zip(
            result["responses"], result["response_metadata"], strict=True
        ):
            doc_id = rmeta.get("json_doc_id") or str(uuid.uuid4())
            records.append(
                {
                    "doc_id": doc_id,
                    "is_new": doc_id not in stored,
                    "before": stored.get(doc_id, {}),
                    "value": r.model_dump(mode="json"),
                }
            )

        if not records:
            return Command(
                goto="cue_brain",
                update={
                    "messages": [
                        ToolMessage(
                            content=f"No {spec.label} was extracted -- nothing was saved.",
                            tool_call_id=tool_call["id"],
                        )
                    ],
                    "proposal": None,
                    "review_feedback": [],
                },
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
    ) -> Command[Literal["save_memory", "propose_memory", "cue_brain"]]:
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
                goto="cue_brain",
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
            update={
                "review_feedback": (state.get("review_feedback") or []) + [feedback]
            },
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
            + [
                HumanMessage(
                    content="Please update the instructions based on the conversation"
                )
            ]
        )

        await store.aput(namespace, INSTRUCTIONS_KEY, {"memory": new_memory.text})

        tool_call = state["messages"][-1].tool_calls[0]
        return {
            "messages": [
                ToolMessage(
                    content="updated instructions", tool_call_id=tool_call["id"]
                )
            ]
        }


def route_message(  # type: ignore[valid-type]
    state: MemoryState,
) -> Literal[END, "propose_memory", "update_instructions"]:
    """Decide whether the last assistant turn asked to update memory, and which kind."""
    tool_calls = getattr(state["messages"][-1], "tool_calls", None)
    if not tool_calls:
        return END
    if tool_calls[0]["args"]["update_type"] == "instructions":
        # Instructions are internal plumbing the user is never told about, so they are
        # written straight through.
        return "update_instructions"
    # Notes and ToDos are user-visible memory, so they go through review first.
    return "propose_memory"
