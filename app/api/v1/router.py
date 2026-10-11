from fastapi import APIRouter

from app.api.v1 import chat, items, memory

api_v1_router = APIRouter()
api_v1_router.include_router(items.router)
api_v1_router.include_router(chat.router)
api_v1_router.include_router(memory.router)
