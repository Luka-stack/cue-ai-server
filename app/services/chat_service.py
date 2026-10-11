import uuid
from collections.abc import Sequence

from app.agent.runtime import AgentRuntime, PendingReview, ThreadSnapshot, TurnResult
from app.core.exceptions import ThreadNotFoundError
from app.core.timezone import resolve_zone
from app.models.chat_thread import ChatThread
from app.repositories.chat_thread_repository import ChatThreadRepository
from app.schemas.chat import (
    ChatMessageRead,
    PendingReviewRead,
    ProposalRecordRead,
    ReviewDecision,
    ThreadDetailRead,
    ThreadRead,
    TurnRead,
    is_user_visible,
    to_chat_message,
)

_TITLE_MAX_LEN = 80


class ChatService:
    """Conversations with the memory agent, scoped to one user."""

    def __init__(self, repo: ChatThreadRepository, runtime: AgentRuntime, user_id: str) -> None:
        self._repo = repo
        self._runtime = runtime
        self._user_id = user_id

    # ------------------------------------------------------------------- threads

    async def create_thread(self, title: str | None = None) -> ChatThread:
        return await self._repo.add(ChatThread(user_id=self._user_id, title=title))

    async def list_threads(self, *, limit: int, offset: int) -> Sequence[ChatThread]:
        return await self._repo.list_for_user(self._user_id, limit=limit, offset=offset)

    async def get_thread(self, thread_id: str, *, include_internal: bool) -> ThreadDetailRead:
        thread = await self._get_or_raise(thread_id)
        snapshot = await self._runtime.get_snapshot(str(thread.id), self._user_id)
        return ThreadDetailRead(
            thread=ThreadRead.model_validate(thread),
            status=snapshot.status,
            messages=self._messages(snapshot.messages, include_internal),
            pending_review=self._pending(snapshot),
        )

    async def delete_thread(self, thread_id: str) -> None:
        thread = await self._get_or_raise(thread_id)
        await self._runtime.delete_thread(str(thread.id))
        await self._repo.delete(thread)

    # --------------------------------------------------------------------- turns

    async def send_message(self, thread_id: str, content: str, *, tz: str | None) -> TurnRead:
        resolve_zone(tz)  # reject an unknown zone before running the model
        thread = await self._get_or_raise(thread_id)
        result = await self._runtime.send_message(str(thread.id), self._user_id, content, tz=tz)
        title = None if thread.title else _derive_title(content)
        thread = await self._repo.touch(thread, title=title)
        return self._turn(thread, result)

    async def review(self, thread_id: str, decision: ReviewDecision, *, tz: str | None) -> TurnRead:
        resolve_zone(tz)
        thread = await self._get_or_raise(thread_id)
        result = await self._runtime.review(
            str(thread.id), self._user_id, decision.action, decision.feedback, tz=tz
        )
        thread = await self._repo.touch(thread)
        return self._turn(thread, result)

    # ----------------------------------------------------------------- internals

    async def _get_or_raise(self, thread_id: str) -> ChatThread:
        try:
            key = uuid.UUID(thread_id)
        except ValueError:
            raise ThreadNotFoundError(thread_id) from None
        thread = await self._repo.get(key)
        # A thread belonging to someone else is indistinguishable from a missing one.
        if thread is None or thread.user_id != self._user_id:
            raise ThreadNotFoundError(thread_id)
        return thread

    def _turn(self, thread: ChatThread, result: TurnResult) -> TurnRead:
        return TurnRead(
            thread=ThreadRead.model_validate(thread),
            status=result.status,
            messages=self._messages(result.new_messages, include_internal=False),
            pending_review=self._pending(result),
        )

    @staticmethod
    def _messages(messages, include_internal: bool) -> list[ChatMessageRead]:
        mapped = [m for m in (to_chat_message(m) for m in messages) if m is not None]
        if include_internal:
            return mapped
        return [m for m in mapped if is_user_visible(m)]

    @staticmethod
    def _pending(snapshot: ThreadSnapshot) -> PendingReviewRead | None:
        pending: PendingReview | None = snapshot.pending_review
        if pending is None:
            return None
        return PendingReviewRead(
            kind=pending.kind,
            action_request=pending.action_request,
            summary=pending.summary,
            help=pending.help,
            records=[ProposalRecordRead(**rec) for rec in pending.records],
        )


def _derive_title(content: str) -> str:
    first_line = content.strip().splitlines()[0].strip()
    if len(first_line) <= _TITLE_MAX_LEN:
        return first_line
    return first_line[: _TITLE_MAX_LEN - 1].rstrip() + "…"
