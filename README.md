# CueAI Server

Backend API for the CueAI capture-inbox app (React Native client in `../cue-ai`). Users capture **notes** and **to-dos**; the app shows them on a Today tab with counts, filters, item details, and completion toggles, and talks to a **chat agent** that turns what the user says into notes and to-dos, with every write reviewed by the user before it is stored.

Notes and to-dos live in **one place, the `items` table**. The app edits it through the items API and the agent reads and writes the same rows as its long-term memory. A to-do created, edited, completed or deleted in the app is what the agent sees on its very next turn, in any thread, and a to-do saved in chat shows up in the app's list.

Built with FastAPI, async SQLAlchemy 2.0, PostgreSQL, Alembic, LangGraph + LangChain (OpenAI), and [uv](https://docs.astral.sh/uv/).

## Quick start (Docker only)

```bash
echo CUEAI_OPENAI_API_KEY=sk-... > .env   # the agent's model runs on OpenAI
docker compose up --build
```

API at http://localhost:8000 — interactive docs at http://localhost:8000/docs (disabled when `CUEAI_APP_ENV=prod`). Migrations run automatically on container start.

## Local development

Requires Python 3.12+, uv, Docker (for Postgres), and an OpenAI API key.

```bash
# 1. Start Postgres
docker compose up -d db

# 2. Install dependencies
uv sync

# 3. Apply migrations
uv run alembic upgrade head

# 4. Configure the OpenAI key (once)
cp .env.example .env   # then set CUEAI_OPENAI_API_KEY

# 5. Run the dev server (hot reload)
uv run uvicorn app.main:app --reload --loop app.core.event_loop:loop_factory
```

The `--loop` flag matters on Windows: the agent's Postgres pool (psycopg) cannot run on the default proactor event loop, so `app.core.event_loop.loop_factory` picks a selector loop there (and uvloop elsewhere). The Docker entrypoint passes it too.

Configuration is read from environment variables / `.env` (copy `.env.example`). All variables use the `CUEAI_` prefix — see `app/core/config.py`.

### Everyday commands

```bash
uv run pytest                                        # tests (needs the compose db running; no OpenAI key needed)
uv run ruff check . && uv run ruff format .          # lint + format
uv run alembic revision --autogenerate -m "message"  # new migration
uv run alembic upgrade head                          # apply migrations
```

## API

All routes are mounted under `/api/v1`.

### Items

| Method | Path                                                                         | Description                                 |
| ------ | ---------------------------------------------------------------------------- | ------------------------------------------- |
| GET    | `/api/v1/items?filter=all\|today\|notes\|completed&tz=<IANA>&limit=&offset=` | List items                                  |
| GET    | `/api/v1/items/counts?tz=<IANA>`                                             | `{today, notes, all, completed}`            |
| GET    | `/api/v1/items/{id}`                                                         | Get one item                                |
| POST   | `/api/v1/items`                                                              | Create a note or todo (201)                 |
| PATCH  | `/api/v1/items/{id}`                                                         | Partial update (e.g. `{"completed": true}`) |
| DELETE | `/api/v1/items/{id}`                                                         | Delete (204)                                |

### Chat agent

| Method | Path                                               | Description                                                                                            |
| ------ | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| POST   | `/api/v1/chat/threads`                             | Start a conversation (201). Optional body `{"title": "..."}`                                           |
| GET    | `/api/v1/chat/threads?limit=&offset=`              | The user's threads, most recently active first                                                         |
| GET    | `/api/v1/chat/threads/{id}?include_internal=false` | Full history + pending review                                                                          |
| DELETE | `/api/v1/chat/threads/{id}`                        | Delete the thread and its checkpoints (204). Memory is kept                                            |
| POST   | `/api/v1/chat/threads/{id}/messages`               | Send `{"content": "..."}` and run the agent                                                            |
| POST   | `/api/v1/chat/threads/{id}/review`                 | Answer a pending review: `{"action": "accept" \| "reject" \| "revise", "feedback"?: "..."}`            |
| GET    | `/api/v1/memory/instructions`                      | The agent's internal to-do instructions (read-only). Notes and to-dos are read through `/api/v1/items` |

### Health

| Method | Path            | Description                                                     |
| ------ | --------------- | --------------------------------------------------------------- |
| GET    | `/health`       | Liveness + DB ping                                              |
| GET    | `/health/agent` | Chat readiness: the OpenAI key works and the model is available |

### Identifying the user

There is no authentication yet. Items, chat and memory routes read an `X-User-Id` header and fall back to `CUEAI_DEFAULT_USER_ID`. Items, threads and memory are strictly scoped to that id (another user's item or thread is a 404). Items that existed before owners were introduced belong to the user `default`. Swapping `get_current_user_id` in `app/api/deps.py` for a token-validating dependency is the only change needed once auth exists.

### How a chat turn works

Sending a message returns a `TurnRead`:

```jsonc
{
  "thread": { "id": "<uuid>", "title": "Remember the spare key…", "created_at": "...", "updated_at": "..." },
  "status": "completed" | "awaiting_review",
  "messages": [ /* only the messages this turn added */
    { "id": "...", "role": "user", "content": "Remember the spare key is under the flowerpot" }
  ],
  "pending_review": {              // present when status == "awaiting_review"
    "kind": "note",                // or "todo"
    "action_request": "save this note",
    "summary": "    [1] NEW note\n        title: Spare Key Location\n        content: ...",
    "help": "Resume with 'yes' to save, 'no' to discard, or describe the change you want.",
    "records": [
      { "doc_id": "<uuid>", "is_new": true, "before": {}, "value": { "title": "Spare Key Location", "content": "..." } }
    ]
  }
}
```

- **`completed`**: the agent replied (the last message has `role: "assistant"`) and nothing was written.
- **`awaiting_review`**: the agent wants to save a note or a to-do. Nothing is stored yet. The client renders `pending_review.records` as a card (that is the "Here's what I caught — want me to add it?" moment in the app) and answers via the review endpoint:
  - `accept` writes the records to the items table, then the agent confirms in an assistant message. Refetch items and counts afterwards.
  - `reject` discards the draft; the agent acknowledges that nothing was saved.
  - `revise` + `feedback` re-runs the extractor with the correction applied to the draft only, and returns for another review.
- While a review is pending, sending another message is a **409**; answer the review first. Concurrent requests on one thread are also a 409 (the thread is busy).
- Pass the phone's time zone as `?tz=<IANA>` on both the message and the review call, like the items API. Deadlines the user mentions ("end of November") are read in that zone and stored in UTC, and the agent is shown deadlines in local time. UTC when omitted; an unknown zone is a 422.
- **503** means OpenAI is unreachable or rejected the request (key, quota, rate limit); **504** means the turn exceeded `CUEAI_AGENT_TURN_TIMEOUT_S`.

`records[].value` follows the extractor schemas in `app/agent/schemas.py`: a note is `{title, content?}`; a to-do is `{task, time_to_complete?, deadline?, solutions[], status}`. `records[].doc_id` is the item id: after an accept, the item exists under exactly that id. For an update to an existing item, `is_new` is false and `before` holds the previous value so the client can show a diff.

How a to-do document maps onto an item:

| Agent field        | Item field         | Notes                                             |
| ------------------ | ------------------ | ------------------------------------------------- |
| `task`             | `title`            |                                                   |
| `deadline`         | `date`             | local wall-clock time in the request's `tz` ↔ UTC |
| `solutions`        | `solutions`        |                                                   |
| `time_to_complete` | `time_to_complete` | minutes                                           |
| `status`           | `status`           | same four values; `completed` follows it          |
| —                  | `content`          | shown to the agent, never changed by it           |

An accepted update writes only the fields the proposal changes. If the user edits the same item in the app while the review is pending, those edits survive the accept. An item deleted in the app while its review is pending is recreated on accept, because the user just approved that exact content.

Messages (`GET /threads/{id}`) hide the agent's internal plumbing by default: tool messages and assistant turns that only carry a tool call. Pass `include_internal=true` to see them (assistant messages then carry `tool_calls: [{name, args}]`).

### Item shapes (match the client contract in `cue-ai/models/index.ts`)

```jsonc
// note — never carries "date" or "completed"
{ "id": "<uuid>", "type": "note", "title": "Gift ideas", "content": "..." }

// todo — "title" is required; "date" is UTC ISO-8601 with a Z suffix
{ "id": "<uuid>", "type": "todo", "title": "Buy milk", "date": "2026-07-27T10:00:00Z", "completed": false }

// todo with the optional fields the agent fills in (omitted when empty)
{ "id": "<uuid>", "type": "todo", "title": "Fix the door", "completed": false,
  "status": "in progress", "solutions": ["WD-40", "Call a locksmith"], "time_to_complete": 30 }
```

Validation rules:

- A note needs a non-empty `title` **or** `content`.
- A todo needs a non-empty `title`; `date` must carry a timezone offset (naive datetimes are rejected with 422).
- Unknown fields are rejected (`extra="forbid"`).
- PATCH bodies are partial; `{"date": null}` clears a todo's date. Setting `date`, `completed`, `solutions` or `time_to_complete` on a note is a 422.
- `solutions` is a list of up to 20 non-empty strings; `time_to_complete` is a positive number of minutes.
- A todo's `status` is one of `not started`, `in progress`, `done`, `archived` (the same values the agent uses), and `completed` is always `status == "done"`. A client may send either one and the other follows. Checking a todo sets `done`. Unchecking a `done` todo returns it to `not started`, while unchecking an `in progress` or `archived` one keeps its status. Sending both with values that disagree is a 422, as is a null status or completed. The `completed` filter and count mean `done`. Archived todos still appear in the other filters; the client can hide them by status.

### Semantics worth knowing

- **Ordering**: notes first (newest first), then todos by `date` descending, undated todos last.
- **`today`**: todos whose `date` falls in `[local midnight, next local midnight)` of the `tz` query param (IANA name, e.g. `Europe/Warsaw`); defaults to UTC. The RN client can send `Intl.DateTimeFormat().resolvedOptions().timeZone`.
- **Dates round-trip semantically, not byte-identically**: input offsets are normalized to UTC `Z` form (e.g. `12:00+02:00` → `10:00:00Z`).
- **Pagination**: optional `limit` (max 500) / `offset`; the response stays a bare array to match the client contract.

## Architecture

```
app/
├── main.py          # app factory, lifespan (DB ping, agent runtime), CORS, exception handlers
├── core/            # settings (pydantic-settings), domain exceptions, logging
├── db/              # engine/session lifecycle (unit-of-work per request)
├── models/          # SQLAlchemy ORM: items (owned per user), chat_threads (metadata only)
├── schemas/         # Pydantic contracts: items, chat (threads, turns, review, memory)
├── repositories/    # all SQL, no business rules
├── services/        # business rules: items, chat (thread ownership, turns), memory (read)
├── api/             # DI wiring (deps.py: user id, agent runtime) + versioned routers
└── agent/           # the LangGraph memory agent
    ├── schemas.py   # Note / ToDo extraction schemas, UpdateMemory routing tool
    ├── prompts.py   # system prompt, Trustcall extraction rules
    ├── state.py     # MemoryState, memory-kind registry, proposal formatting, decision parsing
    ├── memory.py    # ItemMemory: notes/todos as item rows, document <-> item mapping
    ├── nodes.py     # cue_brain → propose_memory → review_memory (interrupt) → save_memory
    ├── graph.py     # StateGraph assembly
    ├── llm.py       # ChatOpenAI factory (Responses API)
    ├── persistence.py  # checkpointer + store: Postgres (prod) or in-memory
    └── runtime.py   # process-wide AgentRuntime: per-thread locks, timeouts, snapshots
```

Layering is `api → services → repositories → db`, with `services → agent.runtime` for chat. Items code and chat code do not touch each other.

### The agent

1. `cue_brain` loads the user's notes and to-dos from the items table, and the instructions from the store, into the system prompt and answers, optionally calling the `UpdateMemory` tool (strict schema, one call per turn) to say which kind of memory this turn is about. Questions about saved notes and to-dos ("what's due this week?") are answered straight from that prompt, without a tool call.
2. Notes and to-dos go through **`propose_memory`** (a [Trustcall](https://github.com/hinthornw/trustcall) extractor stages new documents and field-level patches), then **`review_memory`** pauses the graph with a LangGraph `interrupt`, and only **`save_memory`** writes to the items table, after the user accepts. Revisions re-run the extractor against the draft alone, so a correction can never leak into a memory the user was not reviewing.
3. Instructions (`update_type == "instructions"`) are written straight through; the user is never told about them.

Adaptations for the server: the nodes are `async`, the chat model and the item memory are injected (so tests run the real graph with a scripted model), and notes and to-dos live in the items table instead of the LangGraph store (`app/agent/memory.py` translates between the extractor's documents and item rows, so the schemas stay those of the notebook).

**Persistence.** Short-term memory (the thread's messages and any paused review) lives in a LangGraph Postgres checkpointer; notes and to-dos in the items table; the to-do instructions, namespaced per user, in a LangGraph Postgres store. The checkpointer and store use the app database through their own `psycopg` pool and create their tables (`checkpoints*`, `store*`, `*_migrations`) on startup via LangGraph's own migrations — deliberately outside Alembic, which owns only the app's tables and is configured to ignore LangGraph's (`alembic/env.py`), so autogenerate never proposes dropping them. `CUEAI_AGENT_PERSISTENCE=memory` swaps both for in-process backends (tests, throwaway runs).

**Runtime guarantees.** One `AgentRuntime` per process holds a lock per active thread (a second request on the same thread is a 409, never an interleaved run), bounds every turn with `CUEAI_AGENT_TURN_TIMEOUT_S`, and maps OpenAI API errors to 503. If OpenAI is unreachable at startup the API still boots (a missing key fails startup); `/health/agent` reports `degraded` and chat requests fail with 503 until it is back.

## Migrations policy

The container entrypoint runs `alembic upgrade head` before starting uvicorn — right for a single-instance deployment. For multi-replica deployments, set `SKIP_MIGRATIONS=1` on the app containers and run migrations as a dedicated release step. LangGraph's checkpointer/store setup is idempotent and safe to run from every replica.

## Tests

`uv run pytest` drops/recreates a dedicated `cueai_test` database on the compose Postgres and applies the schema via Alembic, so every run exercises the migrations. Items tests cover the POST validation matrix, JSON shape guarantees, ordering, timezone-aware `today` boundaries, PATCH rules, and counts arithmetic. Chat tests run the real graph with an in-memory checkpointer/store and a scripted chat model (`tests/scripted_model.py`), covering thread scoping, plain turns, the accept / reject / revise review paths, instruction writes and the 409/422 guards. `tests/test_agent_items.py` covers the shared memory: items created or edited in the app appear in the next prompt, accepted to-dos appear in the items API with the mapped fields and time zone conversion, partial writes keep concurrent app edits, and items are scoped per user. No OpenAI key required.
