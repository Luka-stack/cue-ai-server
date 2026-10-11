"""Chat model factory."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    """The OpenAI chat model the agent runs on."""
    if settings.openai_api_key is None:
        raise RuntimeError("CUEAI_OPENAI_API_KEY must be set")

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        reasoning_effort=settings.openai_reasoning_effort or None,
        # The Responses API supports tools together with reasoning on every gpt-5 model.
        use_responses_api=True,
    )
