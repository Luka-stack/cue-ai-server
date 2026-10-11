from typing import Annotated, Literal

from fastapi import APIRouter, Query, status

from app.api.deps import ItemServiceDep
from app.schemas.item import CountsRead, ItemCreate, ItemRead, ItemUpdate

router = APIRouter(prefix="/items", tags=["items"])

ItemFilter = Literal["all", "today", "notes", "completed", "todos"]
TzQuery = Annotated[
    str | None,
    Query(
        description="IANA timezone (e.g. Europe/Warsaw) used to resolve 'today'; UTC default."
    ),
]


@router.get("", response_model=list[ItemRead], response_model_exclude_none=True)
async def list_items(
    service: ItemServiceDep,
    filter: ItemFilter = "all",  # noqa: A002 — query param name is part of the API contract
    tz: TzQuery = None,
    limit: Annotated[int | None, Query(ge=1, le=500)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return await service.list_items(filter_=filter, tz=tz, limit=limit, offset=offset)


# NOTE: /items/counts must be declared before /items/{item_id},
# otherwise "counts" would be captured as an item id.
@router.get("/counts", response_model=CountsRead)
async def get_counts(service: ItemServiceDep, tz: TzQuery = None):
    return await service.get_counts(tz=tz)


@router.get("/{item_id}", response_model=ItemRead, response_model_exclude_none=True)
async def get_item(item_id: str, service: ItemServiceDep):
    return await service.get_item(item_id)


@router.post(
    "",
    response_model=ItemRead,
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_item(payload: ItemCreate, service: ItemServiceDep):
    return await service.create_item(payload)


@router.patch("/{item_id}", response_model=ItemRead, response_model_exclude_none=True)
async def update_item(item_id: str, payload: ItemUpdate, service: ItemServiceDep):
    return await service.update_item(item_id, payload)


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: str, service: ItemServiceDep):
    await service.delete_item(item_id)
