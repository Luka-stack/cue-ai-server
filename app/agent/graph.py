"""Graph assembly."""

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from app.agent.memory import ItemMemory
from app.agent.nodes import MemoryAgentNodes, route_message
from app.agent.state import MemoryState


def build_graph(
    model: BaseChatModel,
    memory: ItemMemory,
    *,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
) -> CompiledStateGraph:
    """Compile the memory agent.

    ``checkpointer`` holds short-term (within-thread) memory and the paused review.
    ``memory`` holds long-term notes and todos (the items table the app edits);
    ``store`` holds the long-term ToDo-handling instructions.
    """
    nodes = MemoryAgentNodes(model, memory)

    builder = StateGraph(MemoryState)
    builder.add_node("cue_brain", nodes.cue_brain)
    builder.add_node("propose_memory", nodes.propose_memory)
    builder.add_node("review_memory", nodes.review_memory)
    builder.add_node("save_memory", nodes.save_memory)
    builder.add_node("update_instructions", nodes.update_instructions)

    builder.add_edge(START, "cue_brain")
    builder.add_conditional_edges("cue_brain", route_message)
    # propose_memory and review_memory route themselves with Command, so the only static
    # edges left here are the ways back into the conversation.
    builder.add_edge("save_memory", "cue_brain")
    builder.add_edge("update_instructions", "cue_brain")

    return builder.compile(checkpointer=checkpointer, store=store)
