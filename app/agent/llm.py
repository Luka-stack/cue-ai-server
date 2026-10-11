"""Chat model factory."""

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from app.core.config import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Local model served by Ollama -- no API key needed.

    ``num_ctx`` is set explicitly: Ollama's default is well below qwen3's ceiling, and an
    over-long prompt is silently truncated, which quietly breaks tool calling. The default
    of 8192 keeps qwen3:8b's weights and KV cache GPU-resident on an 8 GB card.
    """
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,  # deterministic tool calling
        reasoning=False,  # maps to Ollama `think`: no <think> blocks, much faster
        num_ctx=settings.ollama_num_ctx,
        # The model is checked in the app lifespan instead, so a missing Ollama degrades the
        # chat feature to 503 rather than preventing the whole API from starting.
        validate_model_on_init=False,
    )
