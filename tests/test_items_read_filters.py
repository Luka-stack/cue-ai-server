import uuid

ITEMS = "/api/v1/items"


async def _seed(client) -> dict[str, str]:
    """Create a small fixture set; returns title -> id."""
    payloads = [
        {"type": "note", "title": "Note A"},
        {"type": "note", "title": "Note B"},
        {"type": "todo", "title": "Old todo", "date": "2023-12-01T10:00:00Z"},
        {"type": "todo", "title": "New todo", "date": "2026-07-27T10:00:00Z"},
        {"type": "todo", "title": "Undated todo"},
        {"type": "todo", "title": "Done todo", "date": "2026-01-01T08:00:00Z", "completed": True},
    ]
    ids = {}
    for payload in payloads:
        resp = await client.post(ITEMS, json=payload)
        assert resp.status_code == 201
        ids[payload["title"]] = resp.json()["id"]
    return ids


async def test_list_ordering_notes_first_then_todos_date_desc_nulls_last(client):
    await _seed(client)
    resp = await client.get(ITEMS)
    assert resp.status_code == 200
    titles = [item["title"] for item in resp.json()]

    # Notes first (newest created first), then todos by date desc, undated last.
    assert titles[:2] == ["Note B", "Note A"]
    assert titles[2:] == ["New todo", "Done todo", "Old todo", "Undated todo"]


async def test_list_filter_notes(client):
    await _seed(client)
    resp = await client.get(ITEMS, params={"filter": "notes"})
    body = resp.json()
    assert {item["type"] for item in body} == {"note"}
    assert len(body) == 2


async def test_list_filter_completed(client):
    await _seed(client)
    resp = await client.get(ITEMS, params={"filter": "completed"})
    body = resp.json()
    assert [item["title"] for item in body] == ["Done todo"]


async def test_list_filter_invalid_rejected(client):
    resp = await client.get(ITEMS, params={"filter": "bogus"})
    assert resp.status_code == 422


async def test_list_limit_offset(client):
    await _seed(client)
    resp = await client.get(ITEMS, params={"limit": 2, "offset": 1})
    body = resp.json()
    assert [item["title"] for item in body] == ["Note A", "New todo"]


async def test_get_item_by_id(client):
    ids = await _seed(client)
    resp = await client.get(f"{ITEMS}/{ids['New todo']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "New todo"
    assert body["date"] == "2026-07-27T10:00:00Z"


async def test_get_missing_item_404(client):
    resp = await client.get(f"{ITEMS}/{uuid.uuid4()}")
    assert resp.status_code == 404
    assert "detail" in resp.json()


async def test_get_malformed_id_404(client):
    resp = await client.get(f"{ITEMS}/not-a-uuid")
    assert resp.status_code == 404
