"""Application-facing wrapper around the compiled graph.

One :class:`AgentRuntime` lives for the whole process (``app.state.agent``). It owns the
persistence backends and the compiled graph, serializes concurrent access to a thread,
applies a timeout to every LLM turn, and translates LangGraph's state into plain data
the service layer can hand to the API.
"""

import asyncio
import logging
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import SearchItem
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.graph import build_graph
from app.agent.llm import create_chat_model
from app.agent.memory import ItemMemory
from app.agent.persistence import AgentPersistence, create_persistence
from app.agent.state import (
    INSTRUCTIONS_COLLECTION,
    MEMORY_SEARCH_LIMIT,
    Proposal,
    ReviewAction,
)
from app.core.config import Settings
from app.core.exceptions import (
    AgentTimeoutError,
    AgentUnavailableError,
    NoPendingReviewError,
    ReviewPendingError,
    ThreadBusyError,
)

logger = logging.getLogger(__name__)

TurnStatus = Literal["completed", "awaiting_review"]

# Notes and todos are items (read them through the items API); only the agent-internal
# instructions live in the LangGraph store.
MEMORY_COLLECTIONS: dict[str, str] = {"instructions": INSTRUCTIONS_COLLECTION}


@dataclass(frozen=True)
class PendingReview:
    """The staged write the graph is paused on, ready for the user's decision."""

    kind: str
    action_request: str
    summary: str
    help: str
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class ThreadSnapshot:
    """The state of one conversation thread as the API sees it."""

    messages: list[AnyMessage]
    status: TurnStatus
    pending_review: PendingReview | None


@dataclass(frozen=True)
class TurnResult(ThreadSnapshot):
    """A snapshot plus the messages this particular turn added."""

    new_messages: list[AnyMessage] = field(default_factory=list)


class AgentRuntime:
    def __init__(
        self,
        graph: CompiledStateGraph,
        persistence: AgentPersistence,
        *,
        turn_timeout_s: float,
        model_name: str,
    ) -> None:
        self._graph = graph
        self._persistence = persistence
        self._turn_timeout_s = turn_timeout_s
        self.model_name = model_name
        # One lock per active thread; entries vanish once nothing references them.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    @classmethod
    async def create(
        cls,
        settings: Settings,
        *,
        model: BaseChatModel | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> "AgentRuntime":
        """Build the runtime. ``model`` and ``session_factory`` default to OpenAI and the
        application database; tests pass their own."""
        if session_factory is None:
            from app.db.session import async_session_factory as session_factory

        persistence = await create_persistence(settings)
        model = model or create_chat_model(settings)
        graph = build_graph(
            model,
            ItemMemory(session_factory),
            checkpointer=persistence.checkpointer,
            store=persistence.store,
        )
        return cls(
            graph,
            persistence,
            turn_timeout_s=settings.agent_turn_timeout_s,
            model_name=settings.openai_model,
        )

    async def aclose(self) -> None:
        await self._persistence.aclose()

    @property
    def persistence_backend(self) -> str:
        return self._persistence.backend

    # ------------------------------------------------------------------- threads

    async def send_message(
        self, thread_id: str, user_id: str, content: str, *, tz: str | None = None
    ) -> TurnResult:
        """Run one conversational turn. Stops at the review if the agent wants to write memory.

        ``tz`` is the user's IANA time zone: deadlines are shown to and read from the
        model in local time, and stored in UTC.
        """
        async with self._acquire(thread_id):
            before = await self.get_snapshot(thread_id, user_id)
            if before.status == "awaiting_review":
                raise ReviewPendingError(thread_id)
            return await self._run(
                thread_id, user_id, {"messages": [HumanMessage(content=content)]}, before, tz
            )

    async def review(
        self,
        thread_id: str,
        user_id: str,
        action: ReviewAction,
        feedback: str | None,
        *,
        tz: str | None = None,
    ) -> TurnResult:
        """Answer the pending review: accept, reject, or revise with feedback."""
        async with self._acquire(thread_id):
            before = await self.get_snapshot(thread_id, user_id)
            if before.status != "awaiting_review":
                raise NoPendingReviewError(thread_id)
            decision = {"action": action, "feedback": feedback or ""}
            return await self._run(thread_id, user_id, Command(resume=decision), before, tz)

    async def get_snapshot(self, thread_id: str, user_id: str) -> ThreadSnapshot:
        state = await self._graph.aget_state(self._config(thread_id, user_id))
        messages = list(state.values.get("messages") or []) if state.values else []
        pending = self._pending_review(state)
        return ThreadSnapshot(
            messages=messages,
            status="awaiting_review" if pending else "completed",
            pending_review=pending,
        )

    async def delete_thread(self, thread_id: str) -> None:
        async with self._acquire(thread_id):
            await self._persistence.checkpointer.adelete_thread(thread_id)

    # -------------------------------------------------------------------- memory

    async def list_memories(self, user_id: str, collection: str) -> Sequence[SearchItem]:
        namespace = (MEMORY_COLLECTIONS[collection], user_id)
        return await self._persistence.store.asearch(namespace, limit=MEMORY_SEARCH_LIMIT)

    # ----------------------------------------------------------------- internals

    def _acquire(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[thread_id] = lock
        if lock.locked():
            raise ThreadBusyError(thread_id)
        return lock

    @staticmethod
    def _config(thread_id: str, user_id: str, tz: str | None = None) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id, "user_id": user_id, "tz": tz}}

    async def _run(
        self,
        thread_id: str,
        user_id: str,
        graph_input: Any,
        before: ThreadSnapshot,
        tz: str | None,
    ) -> TurnResult:
        config = self._config(thread_id, user_id, tz)
        try:
            await asyncio.wait_for(
                self._graph.ainvoke(graph_input, config), timeout=self._turn_timeout_s
            )
        except TimeoutError:
            logger.error("agent turn on thread %s exceeded %.0fs", thread_id, self._turn_timeout_s)
            raise AgentTimeoutError() from None
        except openai.APIError as exc:
            # Connection failures, auth, rate limits and server errors alike: the chat is
            # unavailable, the rest of the API is not.
            logger.error("chat model error: %s", exc)
            raise AgentUnavailableError() from exc

        after = await self.get_snapshot(thread_id, user_id)
        seen = {m.id for m in before.messages}
        new_messages = [m for m in after.messages if m.id not in seen]
        return TurnResult(
            messages=after.messages,
            status=after.status,
            pending_review=after.pending_review,
            new_messages=new_messages,
        )

    @staticmethod
    def _pending_review(state: Any) -> PendingReview | None:
        interrupts = list(getattr(state, "interrupts", None) or [])
        if not interrupts:
            interrupts = [i for task in state.tasks for i in task.interrupts]
        if not interrupts:
            return None
        payload = interrupts[0].value
        proposal: Proposal | None = state.values.get("proposal")
        return PendingReview(
            kind=payload["kind"],
            action_request=payload["action_request"],
            summary=payload["summary"],
            help=payload["help"],
            records=list(proposal["records"]) if proposal else [],
        )
