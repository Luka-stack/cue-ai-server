"""Chat agent API: threads, turns, the human-in-the-loop review, and memory reads.

Accepted notes and todos land in the items table, so they are read back through the
items API. Each test uses its own user id because the in-memory instruction store is
shared for the
whole session.
"""

import uuid

import pytest

from tests.scripted_model import ai, tool_call

THREADS = "/api/v1/chat/threads"
MEMORY = "/api/v1/memory"
ITEMS = "/api/v1/items"


async def _items(client, user, filter_: str = "all") -> list[dict]:
    resp = await client.get(ITEMS, headers=user, params={"filter": filter_})
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def user() -> dict[str, str]:
    return {"X-User-Id": f"user-{uuid.uuid4()}"}


async def _new_thread(client, user) -> str:
    resp = await client.post(THREADS, headers=user)
    assert resp.status_code == 201
    return resp.json()["id"]


def _note_script(title: str, content: str | None, reply: str):
    return (
        ai("", tool_call("UpdateMemory", {"update_type": "note"})),
        ai("", tool_call("Note", {"title": title, "content": content})),
        ai(reply),
    )


# ------------------------------------------------------------------------- threads


async def test_create_and_list_threads(client, user):
    resp = await client.post(THREADS, headers=user, json={"title": "Groceries"})
    assert resp.status_code == 201
    body = resp.json()
    uuid.UUID(body["id"])
    assert body["title"] == "Groceries"

    resp = await client.get(THREADS, headers=user)
    assert resp.status_code == 200
    assert [t["id"] for t in resp.json()] == [body["id"]]


async def test_threads_are_scoped_to_user(client, user):
    thread_id = await _new_thread(client, user)
    other = {"X-User-Id": "someone-else"}

    assert (await client.get(THREADS, headers=other)).json() == []
    assert (await client.get(f"{THREADS}/{thread_id}", headers=other)).status_code == 404
    assert (await client.delete(f"{THREADS}/{thread_id}", headers=other)).status_code == 404


async def test_unknown_thread_is_404(client, user):
    assert (await client.get(f"{THREADS}/not-a-uuid", headers=user)).status_code == 404
    assert (await client.get(f"{THREADS}/{uuid.uuid4()}", headers=user)).status_code == 404


async def test_new_thread_is_empty_and_completed(client, user):
    thread_id = await _new_thread(client, user)
    resp = await client.get(f"{THREADS}/{thread_id}", headers=user)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["messages"] == []
    assert body["pending_review"] is None


# --------------------------------------------------------------------------- turns


async def test_plain_chat_turn(client, user, model):
    thread_id = await _new_thread(client, user)
    model.script(ai("Hello! How can I help?"))

    resp = await client.post(
        f"{THREADS}/{thread_id}/messages", headers=user, json={"content": "Hi there"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["pending_review"] is None
    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("user", "Hi there"),
        ("assistant", "Hello! How can I help?"),
    ]
    # First message names the thread.
    assert body["thread"]["title"] == "Hi there"


async def test_blank_message_rejected(client, user):
    thread_id = await _new_thread(client, user)
    resp = await client.post(
        f"{THREADS}/{thread_id}/messages", headers=user, json={"content": "  "}
    )
    assert resp.status_code == 422


async def test_note_is_staged_then_saved_on_accept(client, user, model):
    thread_id = await _new_thread(client, user)
    model.script(*_note_script("Spare key", "Under the blue flowerpot", "Saved the note."))

    resp = await client.post(
        f"{THREADS}/{thread_id}/messages",
        headers=user,
        json={"content": "Remember the spare key is under the blue flowerpot"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "awaiting_review"
    review = body["pending_review"]
    assert review["kind"] == "note"
    assert review["action_request"] == "save this note"
    assert len(review["records"]) == 1
    record = review["records"][0]
    assert record["is_new"] is True
    assert record["value"] == {"title": "Spare key", "content": "Under the blue flowerpot"}
    # The tool-call-only assistant turn is internal; only the user's message is visible.
    assert [m["role"] for m in body["messages"]] == ["user"]

    # Nothing is stored until the user accepts.
    assert await _items(client, user) == []

    resp = await client.post(
        f"{THREADS}/{thread_id}/review", headers=user, json={"action": "accept"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["pending_review"] is None
    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("assistant", "Saved the note.")
    ]

    # The accepted note is an item the app lists, under the id the review showed.
    items = await _items(client, user)
    assert items[0].pop("updated_at")
    assert items == [
        {
            "id": record["doc_id"],
            "type": "note",
            "title": "Spare key",
            "content": "Under the blue flowerpot",
        }
    ]


async def test_reject_discards_the_draft(client, user, model):
    thread_id = await _new_thread(client, user)
    model.script(
        ai("", tool_call("UpdateMemory", {"update_type": "todo"})),
        ai("", tool_call("ToDo", {"task": "Buy cat food", "solutions": ["pet store"]})),
        ai("Alright, I did not add it."),
    )

    resp = await client.post(
        f"{THREADS}/{thread_id}/messages", headers=user, json={"content": "I need cat food"}
    )
    assert resp.json()["status"] == "awaiting_review"
    assert resp.json()["pending_review"]["kind"] == "todo"

    resp = await client.post(
        f"{THREADS}/{thread_id}/review", headers=user, json={"action": "reject"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
    assert resp.json()["messages"][-1]["content"] == "Alright, I did not add it."
    assert await _items(client, user) == []


async def test_revise_redrafts_and_reviews_again(client, user, model):
    thread_id = await _new_thread(client, user)
    model.script(
        ai("", tool_call("UpdateMemory", {"update_type": "note"})),
        ai("", tool_call("Note", {"title": "Spare key", "content": "under the pot"})),
    )
    resp = await client.post(
        f"{THREADS}/{thread_id}/messages", headers=user, json={"content": "spare key note"}
    )
    first = resp.json()["pending_review"]["records"][0]

    # The revision patches the draft in place (Trustcall PatchDoc on the draft's doc id).
    model.script(
        ai(
            "",
            tool_call(
                "PatchDoc",
                {
                    "json_doc_id": first["doc_id"],
                    "planned_edits": "mention the blue pot",
                    "patches": [
                        {"op": "replace", "path": "/content", "value": "under the blue pot"}
                    ],
                },
            ),
        ),
    )
    resp = await client.post(
        f"{THREADS}/{thread_id}/review",
        headers=user,
        json={"action": "revise", "feedback": "say it's the blue pot"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "awaiting_review"
    record = body["pending_review"]["records"][0]
    assert record["doc_id"] == first["doc_id"]
    assert record["value"]["content"] == "under the blue pot"

    model.script(ai("Saved."))
    resp = await client.post(
        f"{THREADS}/{thread_id}/review", headers=user, json={"action": "accept"}
    )
    assert resp.json()["status"] == "completed"
    assert [n["content"] for n in await _items(client, user)] == ["under the blue pot"]


async def test_revise_requires_feedback(client, user):
    thread_id = await _new_thread(client, user)
    resp = await client.post(
        f"{THREADS}/{thread_id}/review", headers=user, json={"action": "revise"}
    )
    assert resp.status_code == 422


async def test_review_without_pending_is_409(client, user):
    thread_id = await _new_thread(client, user)
    resp = await client.post(
        f"{THREADS}/{thread_id}/review", headers=user, json={"action": "accept"}
    )
    assert resp.status_code == 409


async def test_message_while_review_pending_is_409(client, user, model):
    thread_id = await _new_thread(client, user)
    model.script(*_note_script("A", None, "ok")[:2])
    await client.post(f"{THREADS}/{thread_id}/messages", headers=user, json={"content": "note A"})

    resp = await client.post(
        f"{THREADS}/{thread_id}/messages", headers=user, json={"content": "another thing"}
    )
    assert resp.status_code == 409
    assert "pending" in resp.json()["detail"]


async def test_thread_detail_exposes_history_and_pending_review(client, user, model):
    thread_id = await _new_thread(client, user)
    model.script(ai("Hi!"))
    await client.post(f"{THREADS}/{thread_id}/messages", headers=user, json={"content": "hello"})
    model.script(*_note_script("Vet", "Dr. K", "saved")[:2])
    await client.post(f"{THREADS}/{thread_id}/messages", headers=user, json={"content": "vet note"})

    resp = await client.get(f"{THREADS}/{thread_id}", headers=user)
    body = resp.json()
    assert body["status"] == "awaiting_review"
    assert body["pending_review"]["records"][0]["value"]["title"] == "Vet"
    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("user", "hello"),
        ("assistant", "Hi!"),
        ("user", "vet note"),
    ]

    resp = await client.get(
        f"{THREADS}/{thread_id}", headers=user, params={"include_internal": True}
    )
    roles = [m["role"] for m in resp.json()["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert resp.json()["messages"][-1]["tool_calls"] == [
        {"name": "UpdateMemory", "args": {"update_type": "note"}}
    ]


async def test_memory_is_shared_across_threads_of_a_user(client, user, model):
    first = await _new_thread(client, user)
    model.script(*_note_script("Wifi", "hunter2", "saved"))
    await client.post(f"{THREADS}/{first}/messages", headers=user, json={"content": "wifi note"})
    await client.post(f"{THREADS}/{first}/review", headers=user, json={"action": "accept"})

    second = await _new_thread(client, user)
    assert (await client.get(f"{THREADS}/{second}", headers=user)).json()["messages"] == []
    assert [n["title"] for n in await _items(client, user)] == ["Wifi"]

    # ...and invisible to anyone else.
    assert await _items(client, {"X-User-Id": "someone-else"}) == []


async def test_instructions_are_written_without_review(client, user, model):
    thread_id = await _new_thread(client, user)
    model.script(
        ai("", tool_call("UpdateMemory", {"update_type": "instructions"})),
        ai("Always include local vendors."),
        ai("Noted."),
    )
    resp = await client.post(
        f"{THREADS}/{thread_id}/messages",
        headers=user,
        json={"content": "When adding todos, include local vendors."},
    )
    assert resp.json()["status"] == "completed"
    instructions = (await client.get(f"{MEMORY}/instructions", headers=user)).json()
    assert len(instructions) == 1
    assert instructions[0]["value"] == {"memory": "Always include local vendors."}


async def test_unknown_memory_collection_is_422(client, user):
    assert (await client.get(f"{MEMORY}/secrets", headers=user)).status_code == 422
    # Notes and todos are served by the items API, not the memory endpoint.
    assert (await client.get(f"{MEMORY}/notes", headers=user)).status_code == 422


async def test_delete_thread_removes_history_but_keeps_memory(client, user, model):
    thread_id = await _new_thread(client, user)
    model.script(*_note_script("Keep", None, "saved"))
    await client.post(
        f"{THREADS}/{thread_id}/messages", headers=user, json={"content": "keep this"}
    )
    await client.post(f"{THREADS}/{thread_id}/review", headers=user, json={"action": "accept"})

    resp = await client.delete(f"{THREADS}/{thread_id}", headers=user)
    assert resp.status_code == 204
    assert (await client.get(f"{THREADS}/{thread_id}", headers=user)).status_code == 404
    assert (await client.get(THREADS, headers=user)).json() == []
    assert len(await _items(client, user)) == 1
