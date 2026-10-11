import uuid

ITEMS = "/api/v1/items"


async def test_create_note_returns_201_and_contract_shape(client):
    resp = await client.post(
        ITEMS, json={"type": "note", "title": "Gift ideas for mom"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["type"] == "note"
    assert body["title"] == "Gift ideas for mom"
    uuid.UUID(body["id"])  # id is a string UUID
    # A note must not carry todo-only keys — absent, not null.
    assert "deadline" not in body
    assert "completed" not in body
    assert "content" not in body  # unset optional fields are omitted


async def test_create_note_with_only_content(client):
    resp = await client.post(ITEMS, json={"type": "note", "content": "just a thought"})
    assert resp.status_code == 201
    assert resp.json()["content"] == "just a thought"
    assert "title" not in resp.json()


async def test_create_note_without_title_and_content_rejected(client):
    resp = await client.post(ITEMS, json={"type": "note"})
    assert resp.status_code == 422

    resp = await client.post(ITEMS, json={"type": "note", "title": "  ", "content": ""})
    assert resp.status_code == 422


async def test_create_todo_returns_contract_shape(client):
    resp = await client.post(
        ITEMS,
        json={"type": "todo", "title": "Buy milk", "deadline": "2026-07-27T10:00:00Z"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["type"] == "todo"
    assert body["title"] == "Buy milk"
    assert body["completed"] is False
    assert body["deadline"] == "2026-07-27T10:00:00Z"  # UTC with Z suffix


async def test_create_todo_normalizes_offset_to_utc_z(client):
    resp = await client.post(
        ITEMS,
        json={"type": "todo", "title": "Call", "deadline": "2026-07-27T12:00:00+02:00"},
    )
    assert resp.status_code == 201
    assert resp.json()["deadline"] == "2026-07-27T10:00:00Z"


async def test_create_todo_without_title_rejected(client):
    resp = await client.post(ITEMS, json={"type": "todo"})
    assert resp.status_code == 422

    resp = await client.post(ITEMS, json={"type": "todo", "title": "   "})
    assert resp.status_code == 422


async def test_create_todo_naive_date_rejected(client):
    resp = await client.post(
        ITEMS, json={"type": "todo", "title": "x", "deadline": "2026-07-27T10:00:00"}
    )
    assert resp.status_code == 422


async def test_create_unknown_type_rejected(client):
    resp = await client.post(ITEMS, json={"type": "reminder", "title": "x"})
    assert resp.status_code == 422


async def test_create_with_unknown_field_rejected(client):
    resp = await client.post(ITEMS, json={"type": "note", "title": "x", "date": "oops"})
    assert resp.status_code == 422


async def test_note_cannot_carry_todo_fields(client):
    resp = await client.post(
        ITEMS, json={"type": "note", "title": "x", "completed": True}
    )
    assert resp.status_code == 422
