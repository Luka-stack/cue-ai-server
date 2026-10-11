"""CueAI memory agent.

A LangGraph agent (ported from ``module-5/cueai-agent.ipynb``) that keeps the user's
notes and ToDo list in long-term memory. Every write to memory is staged, handed to
the user for review through a LangGraph ``interrupt``, and only persisted once the
user accepts it.

Public entry point: :class:`app.agent.runtime.AgentRuntime`.
"""
