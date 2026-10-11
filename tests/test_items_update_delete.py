import uuid

ITEMS = "/api/v1/items"


async def _create(client, payload: dict) -> dict:
    resp = await client.post(ITEMS, json=payload)
    assert resp.status_code == 201
    return resp.json()


async def test_toggle_todo_completed(client):
    todo = await _create(client, {"type": "todo", "title": "Buy milk"})
    assert todo["completed"] is False

    resp = await client.patch(f"{ITEMS}/{todo['id']}", json={"completed": True})
    assert resp.status_code == 200
    assert resp.json()["completed"] is True

    resp = await client.patch(f"{ITEMS}/{todo['id']}", json={"completed": False})
    assert resp.json()["completed"] is False


async def test_patch_todo_clear_date_with_null(client):
    todo = await _create(
        client,
        {"type": "todo", "title": "Buy milk", "deadline": "2026-07-27T10:00:00Z"},
    )
    resp = await client.patch(f"{ITEMS}/{todo['id']}", json={"deadline": None})
    assert resp.status_code == 200
    assert "deadline" not in resp.json()


async def test_patch_todo_empty_title_rejected(client):
    todo = await _create(client, {"type": "todo", "title": "Buy milk"})
    for bad_title in (None, "", "   "):
        resp = await client.patch(f"{ITEMS}/{todo['id']}", json={"title": bad_title})
        assert resp.status_code == 422


async def test_patch_note_completed_or_date_rejected(client):
    note = await _create(client, {"type": "note", "title": "Idea"})
    resp = await client.patch(f"{ITEMS}/{note['id']}", json={"completed": True})
    assert resp.status_code == 422

    resp = await client.patch(
        f"{ITEMS}/{note['id']}", json={"deadline": "2026-07-27T10:00:00Z"}
    )
    assert resp.status_code == 422


async def test_patch_note_cannot_empty_both_title_and_content(client):
    note = await _create(client, {"type": "note", "title": "Idea"})
    resp = await client.patch(f"{ITEMS}/{note['id']}", json={"title": None})
    assert resp.status_code == 422

    # But replacing title with content is fine.
    resp = await client.patch(
        f"{ITEMS}/{note['id']}", json={"title": None, "content": "body"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "title" not in body
    assert body["content"] == "body"


async def test_patch_updates_content(client):
    note = await _create(client, {"type": "note", "title": "Idea"})
    resp = await client.patch(f"{ITEMS}/{note['id']}", json={"content": "more detail"})
    assert resp.status_code == 200
    assert resp.json()["content"] == "more detail"


async def test_patch_missing_item_404(client):
    resp = await client.patch(f"{ITEMS}/{uuid.uuid4()}", json={"title": "x"})
    assert resp.status_code == 404


async def test_delete_then_get_404(client):
    note = await _create(client, {"type": "note", "title": "Idea"})
    resp = await client.delete(f"{ITEMS}/{note['id']}")
    assert resp.status_code == 204
    assert resp.content == b""

    resp = await client.get(f"{ITEMS}/{note['id']}")
    assert resp.status_code == 404


async def test_delete_missing_item_404(client):
    resp = await client.delete(f"{ITEMS}/{uuid.uuid4()}")
    assert resp.status_code == 404
