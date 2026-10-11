from app.agent.runtime import AgentRuntime
from app.schemas.chat import MemoryCollection, MemoryRead


class MemoryService:
    """Read access to the agent's long-term memory for one user.

    Writes deliberately have no endpoint: memory changes only through a reviewed and
    accepted agent proposal, so what the store holds is always something the user saw
    and approved.
    """

    def __init__(self, runtime: AgentRuntime, user_id: str) -> None:
        self._runtime = runtime
        self._user_id = user_id

    async def list(self, collection: MemoryCollection) -> list[MemoryRead]:
        items = await self._runtime.list_memories(self._user_id, collection)
        return [
            MemoryRead(
                id=item.key,
                collection=collection,
                value=item.value,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in items
        ]
