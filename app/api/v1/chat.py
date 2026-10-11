from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import ChatServiceDep
from app.schemas.chat import (
    MessageCreate,
    ReviewDecision,
    ThreadCreate,
    ThreadDetailRead,
    ThreadRead,
    TurnRead,
)

router = APIRouter(prefix="/chat", tags=["chat"])

TzQuery = Annotated[
    str | None,
    Query(
        description="The user's IANA time zone (e.g. Europe/Warsaw). Deadlines the user "
        "mentions are read in it and stored in UTC; UTC when omitted."
    ),
]


@router.post("/threads", response_model=ThreadRead, status_code=status.HTTP_201_CREATED)
async def create_thread(service: ChatServiceDep, payload: ThreadCreate | None = None):
    """Start a new conversation. Long-term memory is shared across all of a user's threads."""
    return await service.create_thread(title=payload.title if payload else None)


@router.get("/threads", response_model=list[ThreadRead])
async def list_threads(
    service: ChatServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return await service.list_threads(limit=limit, offset=offset)


@router.get("/threads/{thread_id}", response_model=ThreadDetailRead)
async def get_thread(
    thread_id: str,
    service: ChatServiceDep,
    include_internal: Annotated[
        bool, Query(description="Also return tool messages and tool-call-only assistant turns.")
    ] = False,
):
    """Full history plus the review the agent is waiting on, if any."""
    return await service.get_thread(thread_id, include_internal=include_internal)


@router.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(thread_id: str, service: ChatServiceDep):
    """Delete the thread and its checkpoints. Long-term memory is untouched."""
    await service.delete_thread(thread_id)


@router.post(
    "/threads/{thread_id}/messages",
    response_model=TurnRead,
    responses={
        409: {"description": "Thread is busy, or a memory review is pending on it"},
        503: {"description": "Chat model unavailable"},
        504: {"description": "Chat model timed out"},
    },
)
async def send_message(
    thread_id: str, payload: MessageCreate, service: ChatServiceDep, tz: TzQuery = None
):
    """Send one user message and run the agent.

    If the agent wants to save a note or a ToDo, the response comes back with
    ``status = "awaiting_review"`` and a ``pending_review``; nothing is stored until the
    client answers via the review endpoint.
    """
    return await service.send_message(thread_id, payload.content, tz=tz)


@router.post(
    "/threads/{thread_id}/review",
    response_model=TurnRead,
    responses={
        409: {"description": "Thread is busy, or has no pending review"},
        503: {"description": "Chat model unavailable"},
        504: {"description": "Chat model timed out"},
    },
)
async def review(
    thread_id: str, payload: ReviewDecision, service: ChatServiceDep, tz: TzQuery = None
):
    """Answer a pending review: accept saves it, reject discards it, revise re-drafts it
    with the given feedback (and comes back for another review)."""
    return await service.review(thread_id, payload, tz=tz)
