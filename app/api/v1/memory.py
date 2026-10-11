from fastapi import APIRouter

from app.api.deps import MemoryServiceDep
from app.schemas.chat import MemoryCollection, MemoryRead

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/{collection}", response_model=list[MemoryRead])
async def list_memory(collection: MemoryCollection, service: MemoryServiceDep):
    """Agent-internal memory (the ToDo-handling instructions). Read-only.

    Notes and todos are not here: they are items, shared with the app, at /api/v1/items.
    """
    return await service.list(collection)
