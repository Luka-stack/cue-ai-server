from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.runtime import AgentRuntime
from app.core.config import get_settings
from app.db.session import get_db_session
from app.repositories.chat_thread_repository import ChatThreadRepository
from app.repositories.item_repository import ItemRepository
from app.services.chat_service import ChatService
from app.services.item_service import ItemService
from app.services.memory_service import MemoryService

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_current_user_id(
    x_user_id: Annotated[
        str | None,
        Header(
            max_length=128,
            description="Identifies the user until real authentication exists. "
            "Defaults to the server's configured default user.",
        ),
    ] = None,
) -> str:
    """Placeholder for authentication: swap this for a token-validating dependency later
    and every items/chat/memory route picks it up unchanged."""
    return (x_user_id or "").strip() or get_settings().default_user_id


CurrentUserId = Annotated[str, Depends(get_current_user_id)]


def get_item_service(session: SessionDep, user_id: CurrentUserId) -> ItemService:
    return ItemService(ItemRepository(session), user_id)


ItemServiceDep = Annotated[ItemService, Depends(get_item_service)]


# ------------------------------------------------------------------------------ chat


def get_agent_runtime(request: Request) -> AgentRuntime:
    return request.app.state.agent


AgentRuntimeDep = Annotated[AgentRuntime, Depends(get_agent_runtime)]


def get_chat_service(
    session: SessionDep, runtime: AgentRuntimeDep, user_id: CurrentUserId
) -> ChatService:
    return ChatService(ChatThreadRepository(session), runtime, user_id)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


def get_memory_service(runtime: AgentRuntimeDep, user_id: CurrentUserId) -> MemoryService:
    return MemoryService(runtime, user_id)


MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]
