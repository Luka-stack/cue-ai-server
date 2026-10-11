#!/bin/sh
set -e

# Apply migrations unless explicitly skipped (e.g. multi-replica deployments
# where a dedicated release step owns the schema).
if [ "${SKIP_MIGRATIONS:-0}" != "1" ]; then
    alembic upgrade head
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop app.core.event_loop:loop_factory
