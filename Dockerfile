FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Install dependencies first so they cache independently of source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY scripts ./scripts

FROM python:3.12-slim-bookworm AS runtime

RUN groupadd --system appuser && useradd --system --gid appuser appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app /app
RUN chmod +x /app/scripts/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
