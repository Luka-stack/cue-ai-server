"""Event loop factory for uvicorn (``--loop app.core.event_loop:loop_factory``).

psycopg's async mode (used by the LangGraph Postgres checkpointer/store) cannot run on
Windows' default ProactorEventLoop. On Windows this returns a selector loop; elsewhere it
prefers uvloop when installed (``uvicorn[standard]`` ships it on Linux), matching what
``--loop auto`` would pick.
"""

import asyncio
import sys


def loop_factory() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    try:
        import uvloop
    except ImportError:  # pragma: no cover - uvloop is a Linux/macOS extra
        return asyncio.new_event_loop()
    return uvloop.new_event_loop()
