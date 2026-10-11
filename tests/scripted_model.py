"""A chat model that replays scripted responses, so the agent graph runs without OpenAI."""

import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": args, "id": str(uuid.uuid4()), "type": "tool_call"}


def ai(content: str = "", *calls: dict[str, Any]) -> AIMessage:
    return AIMessage(content=content, tool_calls=list(calls))


class ScriptedChatModel(BaseChatModel):
    """Pops one pre-written ``AIMessage`` per model call, in order.

    ``bind_tools`` returns the model itself: the script already contains whatever tool
    calls the test wants the "model" to make, so tool binding is a no-op.
    """

    queue: list[AIMessage] = []
    calls: int = 0
    # The prompt of every call, so tests can assert what the model was shown.
    received: list[list[BaseMessage]] = []

    def script(self, *responses: AIMessage) -> None:
        self.queue[:] = list(responses)
        self.received[:] = []
        self.calls = 0

    def system_prompt(self, call: int = 0) -> str:
        """Text of the system message the model got on the given call."""
        return str(self.received[call][0].content)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.queue:
            raise AssertionError("scripted model was called more times than scripted")
        self.calls += 1
        self.received.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.queue.pop(0))])

    @property
    def _llm_type(self) -> str:
        return "scripted"
