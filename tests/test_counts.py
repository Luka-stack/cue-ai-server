from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

ITEMS = "/api/v1/items"
COUNTS = f"{ITEMS}/counts"
WARSAW = ZoneInfo("Europe/Warsaw")


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


async def _create(client, payload: dict) -> dict:
    resp = await client.post(ITEMS, json=payload)
    assert resp.status_code == 201
    return resp.json()


async def test_counts_arithmetic(client):
    now = datetime.now(UTC)
    await _create(client, {"type": "note", "title": "Note A"})
    await _create(client, {"type": "note", "content": "Note B"})
    await _create(client, {"type": "todo", "title": "Due now", "date": _iso(now)})
    await _create(client, {"type": "todo", "title": "Past", "date": _iso(now - timedelta(days=3))})
    await _create(
        client,
        {
            "type": "todo",
            "title": "Past done",
            "date": _iso(now - timedelta(days=3)),
            "completed": True,
        },
    )

    resp = await client.get(COUNTS)
    assert resp.status_code == 200
    assert resp.json() == {"today": 1, "notes": 2, "all": 5, "completed": 1}


async def test_counts_empty_db(client):
    resp = await client.get(COUNTS)
    assert resp.json() == {"today": 0, "notes": 0, "all": 0, "completed": 0}


async def test_counts_today_respects_timezone(client):
    """An item very early in Warsaw's day belongs to Warsaw's 'today', while its
    membership in UTC's 'today' depends on the window — assert both precisely."""
    now_warsaw = datetime.now(WARSAW)
    warsaw_day_start = now_warsaw.replace(hour=0, minute=0, second=0, microsecond=0)
    early_local = warsaw_day_start + timedelta(minutes=15)  # 00:15 Warsaw time today

    await _create(client, {"type": "todo", "title": "Early bird", "date": _iso(early_local)})

    resp = await client.get(COUNTS, params={"tz": "Europe/Warsaw"})
    assert resp.json()["today"] == 1

    now_utc = datetime.now(UTC)
    utc_day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    in_utc_today = utc_day_start <= early_local.astimezone(UTC) < utc_day_start + timedelta(days=1)
    resp = await client.get(COUNTS)
    assert resp.json()["today"] == (1 if in_utc_today else 0)


async def test_list_today_filter_matches_counts(client):
    now = datetime.now(UTC)
    await _create(client, {"type": "todo", "title": "Due now", "date": _iso(now)})
    await _create(
        client, {"type": "todo", "title": "Future", "date": _iso(now + timedelta(days=5))}
    )

    resp = await client.get(ITEMS, params={"filter": "today"})
    titles = [item["title"] for item in resp.json()]
    assert titles == ["Due now"]

    counts = (await client.get(COUNTS)).json()
    assert counts["today"] == 1


async def test_counts_invalid_timezone_rejected(client):
    resp = await client.get(COUNTS, params={"tz": "Mars/Olympus_Mons"})
    assert resp.status_code == 422

    resp = await client.get(ITEMS, params={"filter": "today", "tz": "Mars/Olympus_Mons"})
    assert resp.status_code == 422


async def test_counts_route_not_shadowed_by_item_id(client):
    resp = await client.get(COUNTS)
    assert resp.status_code == 200
    assert set(resp.json()) == {"today", "notes", "all", "completed"}


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
