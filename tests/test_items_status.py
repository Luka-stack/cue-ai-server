"""Todo status and its sync with ``completed``.

``status`` mirrors what the chat agent tracks; ``completed`` is always
``status == "done"``, whichever of the two the client sends.
"""

import pytest

ITEMS = "/api/v1/items"


async def _create(client, payload: dict) -> dict:
    resp = await client.post(ITEMS, json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _patch(client, item_id: str, payload: dict):
    return await client.patch(f"{ITEMS}/{item_id}", json=payload)


async def test_new_todo_defaults_to_not_started(client):
    todo = await _create(client, {"type": "todo", "title": "Buy milk"})
    assert todo["status"] == "not started"
    assert todo["completed"] is False


@pytest.mark.parametrize(
    ("payload", "status", "completed"),
    [
        ({"status": "in progress"}, "in progress", False),
        ({"status": "archived"}, "archived", False),
        ({"status": "done"}, "done", True),
        ({"completed": True}, "done", True),
        ({"completed": False}, "not started", False),
        ({"status": "done", "completed": True}, "done", True),
    ],
)
async def test_create_derives_the_missing_field(client, payload, status, completed):
    todo = await _create(client, {"type": "todo", "title": "x", **payload})
    assert (todo["status"], todo["completed"]) == (status, completed)


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "done", "completed": False},
        {"status": "in progress", "completed": True},
        {"status": "someday"},
    ],
)
async def test_create_rejects_contradictory_or_unknown_status(client, payload):
    resp = await client.post(ITEMS, json={"type": "todo", "title": "x", **payload})
    assert resp.status_code == 422


async def test_checking_a_todo_marks_it_done(client):
    todo = await _create(client, {"type": "todo", "title": "x", "status": "in progress"})
    resp = await _patch(client, todo["id"], {"completed": True})
    assert (resp.json()["status"], resp.json()["completed"]) == ("done", True)


async def test_unchecking_a_done_todo_returns_it_to_not_started(client):
    todo = await _create(client, {"type": "todo", "title": "x", "status": "done"})
    resp = await _patch(client, todo["id"], {"completed": False})
    assert (resp.json()["status"], resp.json()["completed"]) == ("not started", False)


@pytest.mark.parametrize("status", ["in progress", "archived"])
async def test_unchecking_keeps_an_open_status(client, status):
    todo = await _create(client, {"type": "todo", "title": "x", "status": status})
    resp = await _patch(client, todo["id"], {"completed": False})
    assert resp.json()["status"] == status


async def test_setting_status_updates_completed(client):
    todo = await _create(client, {"type": "todo", "title": "x"})

    resp = await _patch(client, todo["id"], {"status": "done"})
    assert resp.json()["completed"] is True

    resp = await _patch(client, todo["id"], {"status": "archived"})
    assert (resp.json()["status"], resp.json()["completed"]) == ("archived", False)


@pytest.mark.parametrize(
    "payload",
    [
        {"status": None},
        {"completed": None},
        {"status": "done", "completed": False},
        {"status": "someday"},
    ],
)
async def test_patch_rejects_invalid_status(client, payload):
    todo = await _create(client, {"type": "todo", "title": "x"})
    resp = await _patch(client, todo["id"], payload)
    assert resp.status_code == 422


async def test_notes_have_no_status(client):
    note = await _create(client, {"type": "note", "title": "x"})
    assert "status" not in note

    assert (await _patch(client, note["id"], {"status": "done"})).status_code == 422
    resp = await client.post(ITEMS, json={"type": "note", "title": "x", "status": "done"})
    assert resp.status_code == 422


async def test_completed_filter_and_counts_follow_status(client):
    await _create(client, {"type": "todo", "title": "open", "status": "in progress"})
    await _create(client, {"type": "todo", "title": "finished", "status": "done"})

    completed = (await client.get(ITEMS, params={"filter": "completed"})).json()
    assert [t["title"] for t in completed] == ["finished"]
    assert (await client.get(f"{ITEMS}/counts")).json()["completed"] == 1
