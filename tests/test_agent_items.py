"""The items table is the agent's memory: what the app edits, the chat sees, and what the
chat saves, the app lists.

These run the real graph with the scripted model, against the test database.
"""

import uuid

import pytest

from tests.scripted_model import ai, tool_call

ITEMS = "/api/v1/items"
THREADS = "/api/v1/chat/threads"


@pytest.fixture
def user() -> dict[str, str]:
    return {"X-User-Id": f"user-{uuid.uuid4()}"}


async def _new_thread(client, user) -> str:
    resp = await client.post(THREADS, headers=user)
    assert resp.status_code == 201
    return resp.json()["id"]


async def _create_item(client, user, payload: dict) -> dict:
    resp = await client.post(ITEMS, headers=user, json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _send(client, user, thread_id, content, **params):
    resp = await client.post(
        f"{THREADS}/{thread_id}/messages", headers=user, json={"content": content}, params=params
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _review(client, user, thread_id, action="accept", **params):
    resp = await client.post(
        f"{THREADS}/{thread_id}/review", headers=user, json={"action": action}, params=params
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _patch(doc_id: str, planned: str, *patches: dict) -> dict:
    return tool_call(
        "PatchDoc", {"json_doc_id": doc_id, "planned_edits": planned, "patches": list(patches)}
    )


# -------------------------------------------------------------------- app -> chat


async def test_todo_created_in_app_is_known_in_a_new_chat(client, user, model):
    """The user's scenario: add a todo in the app, open a fresh chat, ask what to do."""
    await _create_item(
        client,
        user,
        {
            "type": "todo",
            "title": "Renew the car insurance",
            "date": "2026-10-15T10:00:00Z",
            "content": "Compare at least two offers",
        },
    )
    await _create_item(client, user, {"type": "note", "title": "Gate code", "content": "4411"})

    thread_id = await _new_thread(client, user)
    model.script(ai("You need to renew the car insurance by October 15."))
    body = await _send(client, user, thread_id, "What do I have to do?", tz="Europe/Warsaw")
    assert body["messages"][-1]["content"].startswith("You need to renew")

    prompt = model.system_prompt()
    assert "'task': 'Renew the car insurance'" in prompt
    # Shown in the user's local time: 10:00Z is 12:00 in Warsaw in October.
    assert "'deadline': '2026-10-15T12:00:00'" in prompt
    assert "'status': 'not started'" in prompt
    assert "'content': 'Compare at least two offers'" in prompt
    assert "'title': 'Gate code'" in prompt


async def test_edits_in_the_app_are_picked_up_on_the_next_turn(client, user, model):
    todo = await _create_item(client, user, {"type": "todo", "title": "Call the plumber"})
    note = await _create_item(client, user, {"type": "note", "title": "Old wifi password"})
    thread_id = await _new_thread(client, user)

    model.script(ai("ok"))
    await _send(client, user, thread_id, "hi")
    assert "'status': 'not started'" in model.system_prompt()

    # Same thread, no restart: complete the todo, rename the note, delete nothing.
    await client.patch(f"{ITEMS}/{todo['id']}", headers=user, json={"completed": True})
    await client.patch(f"{ITEMS}/{note['id']}", headers=user, json={"title": "New wifi password"})

    model.script(ai("ok"))
    await _send(client, user, thread_id, "anything left?")
    prompt = model.system_prompt()
    assert "'status': 'done'" in prompt
    assert "New wifi password" in prompt
    assert "Old wifi password" not in prompt

    await client.delete(f"{ITEMS}/{note['id']}", headers=user)
    model.script(ai("ok"))
    await _send(client, user, thread_id, "and now?")
    assert "wifi password" not in model.system_prompt()


async def test_chat_never_sees_another_users_items(client, user, model):
    await _create_item(client, {"X-User-Id": "someone-else"}, {"type": "todo", "title": "Secret"})
    thread_id = await _new_thread(client, user)
    model.script(ai("Nothing to do."))
    await _send(client, user, thread_id, "What do I have to do?")
    assert "Secret" not in model.system_prompt()


# -------------------------------------------------------------------- chat -> app


async def test_todo_saved_in_chat_appears_in_the_app(client, user, model):
    thread_id = await _new_thread(client, user)
    model.script(
        ai("", tool_call("UpdateMemory", {"update_type": "todo"})),
        ai(
            "",
            tool_call(
                "ToDo",
                {
                    "task": "Book a vet appointment",
                    "time_to_complete": 15,
                    "deadline": "2026-11-30T00:00:00",
                    "solutions": ["Call Dr. Kowalski"],
                },
            ),
        ),
        ai("Added to your list."),
    )
    body = await _send(
        client, user, thread_id, "Book the vet by end of November", tz="Europe/Warsaw"
    )
    assert body["status"] == "awaiting_review"
    doc_id = body["pending_review"]["records"][0]["doc_id"]

    await _review(client, user, thread_id, tz="Europe/Warsaw")

    resp = await client.get(f"{ITEMS}/{doc_id}", headers=user)
    assert resp.status_code == 200
    assert resp.json() == {
        "id": doc_id,
        "type": "todo",
        "title": "Book a vet appointment",
        # Midnight in Warsaw (UTC+1 in November) is 23:00 the day before in UTC.
        "date": "2026-11-29T23:00:00Z",
        "completed": False,
        "status": "not started",
        "solutions": ["Call Dr. Kowalski"],
        "time_to_complete": 15,
    }
    assert (await client.get(f"{ITEMS}/counts", headers=user)).json()["all"] == 1


async def test_chat_patch_changes_only_what_the_user_asked(client, user, model):
    """Patching one field keeps everything else the app holds, including content the
    extractor schema has no field for."""
    todo = await _create_item(
        client,
        user,
        {
            "type": "todo",
            "title": "Fix the front door",
            "content": "The lock sticks in cold weather",
            "solutions": ["WD-40"],
            "time_to_complete": 30,
        },
    )
    thread_id = await _new_thread(client, user)
    model.script(
        ai("", tool_call("UpdateMemory", {"update_type": "todo"})),
        ai(
            "",
            _patch(
                todo["id"],
                "add a deadline",
                {"op": "replace", "path": "/deadline", "value": "2026-10-01T09:00:00"},
            ),
        ),
        ai("Deadline set."),
    )
    body = await _send(client, user, thread_id, "The door has to be fixed by Oct 1, 9am")
    record = body["pending_review"]["records"][0]
    assert record["is_new"] is False
    assert record["doc_id"] == todo["id"]
    assert "deadline: None  ->  '2026-10-01T09:00:00'" in body["pending_review"]["summary"]
    # Nothing else is reported as changing.
    assert body["pending_review"]["summary"].count("->") == 1

    await _review(client, user, thread_id)

    after = (await client.get(f"{ITEMS}/{todo['id']}", headers=user)).json()
    assert after == {**todo, "date": "2026-10-01T09:00:00Z"}


async def test_app_edit_during_pending_review_is_not_overwritten(client, user, model):
    todo = await _create_item(client, user, {"type": "todo", "title": "Buy paint"})
    thread_id = await _new_thread(client, user)
    model.script(
        ai("", tool_call("UpdateMemory", {"update_type": "todo"})),
        ai(
            "",
            _patch(
                todo["id"],
                "estimate the time",
                {"op": "replace", "path": "/time_to_complete", "value": 45},
            ),
        ),
        ai("Updated."),
    )
    await _send(client, user, thread_id, "Buying paint takes about 45 minutes")

    # While the review is pending, the user renames and completes the todo in the app.
    await client.patch(
        f"{ITEMS}/{todo['id']}",
        headers=user,
        json={"title": "Buy white paint", "completed": True},
    )

    await _review(client, user, thread_id)
    after = (await client.get(f"{ITEMS}/{todo['id']}", headers=user)).json()
    assert after["title"] == "Buy white paint"
    assert after["completed"] is True
    assert after["time_to_complete"] == 45


async def test_item_deleted_during_pending_review_is_recreated_on_accept(client, user, model):
    note = await _create_item(client, user, {"type": "note", "title": "Vet", "content": "Dr. K"})
    thread_id = await _new_thread(client, user)
    model.script(
        ai("", tool_call("UpdateMemory", {"update_type": "note"})),
        ai(
            "",
            _patch(
                note["id"],
                "add opening days",
                {"op": "replace", "path": "/content", "value": "Dr. K, closed Mondays"},
            ),
        ),
        ai("Updated."),
    )
    await _send(client, user, thread_id, "Add that the vet is closed on Mondays")
    await client.delete(f"{ITEMS}/{note['id']}", headers=user)

    await _review(client, user, thread_id)
    resp = await client.get(f"{ITEMS}/{note['id']}", headers=user)
    assert resp.status_code == 200
    assert resp.json()["content"] == "Dr. K, closed Mondays"


async def test_completing_via_chat_marks_the_item_completed(client, user, model):
    todo = await _create_item(client, user, {"type": "todo", "title": "Pay rent"})
    thread_id = await _new_thread(client, user)
    model.script(
        ai("", tool_call("UpdateMemory", {"update_type": "todo"})),
        ai(
            "",
            _patch(todo["id"], "mark done", {"op": "replace", "path": "/status", "value": "done"}),
        ),
        ai("Marked as done."),
    )
    await _send(client, user, thread_id, "I paid the rent")
    await _review(client, user, thread_id)

    after = (await client.get(f"{ITEMS}/{todo['id']}", headers=user)).json()
    assert (after["status"], after["completed"]) == ("done", True)
    assert [
        t["title"]
        for t in (await client.get(ITEMS, headers=user, params={"filter": "completed"})).json()
    ] == ["Pay rent"]


@pytest.mark.parametrize("status", ["in progress", "archived"])
async def test_status_set_in_chat_is_kept_and_seen_next_turn(client, user, model, status):
    """The agent's own statuses survive the round trip through the items table."""
    todo = await _create_item(client, user, {"type": "todo", "title": "Paint the fence"})
    thread_id = await _new_thread(client, user)
    model.script(
        ai("", tool_call("UpdateMemory", {"update_type": "todo"})),
        ai(
            "",
            _patch(todo["id"], "set status", {"op": "replace", "path": "/status", "value": status}),
        ),
        ai("Updated."),
    )
    body = await _send(client, user, thread_id, "Update the fence todo")
    assert f"status: 'not started'  ->  '{status}'" in body["pending_review"]["summary"]
    await _review(client, user, thread_id)

    after = (await client.get(f"{ITEMS}/{todo['id']}", headers=user)).json()
    assert (after["status"], after["completed"]) == (status, False)

    model.script(ai("ok"))
    await _send(client, user, thread_id, "What is the fence status?")
    assert f"'status': '{status}'" in model.system_prompt()


async def test_status_set_in_the_app_is_seen_by_the_chat(client, user, model):
    todo = await _create_item(client, user, {"type": "todo", "title": "Write the report"})
    await client.patch(f"{ITEMS}/{todo['id']}", headers=user, json={"status": "in progress"})

    thread_id = await _new_thread(client, user)
    model.script(ai("You are working on the report."))
    await _send(client, user, thread_id, "What am I working on?")
    assert "'status': 'in progress'" in model.system_prompt()


async def test_chat_with_unknown_timezone_is_422(client, user):
    thread_id = await _new_thread(client, user)
    resp = await client.post(
        f"{THREADS}/{thread_id}/messages",
        headers=user,
        json={"content": "hi"},
        params={"tz": "Mars/Olympus_Mons"},
    )
    assert resp.status_code == 422


# ------------------------------------------------------------------ items contract


async def test_items_are_scoped_to_the_user(client, user):
    item = await _create_item(client, user, {"type": "note", "title": "Mine"})
    other = {"X-User-Id": "someone-else"}

    assert (await client.get(ITEMS, headers=other)).json() == []
    assert (await client.get(f"{ITEMS}/{item['id']}", headers=other)).status_code == 404
    assert (
        await client.patch(f"{ITEMS}/{item['id']}", headers=other, json={"title": "x"})
    ).status_code == 404
    assert (await client.delete(f"{ITEMS}/{item['id']}", headers=other)).status_code == 404
    assert (await client.get(f"{ITEMS}/counts", headers=other)).json()["all"] == 0
    assert (await client.get(f"{ITEMS}/{item['id']}", headers=user)).status_code == 200


async def test_todo_extra_fields_round_trip_and_are_rejected_on_notes(client, user):
    todo = await _create_item(
        client,
        user,
        {"type": "todo", "title": "Paint", "solutions": ["Hardware store"], "time_to_complete": 60},
    )
    assert todo["solutions"] == ["Hardware store"]
    assert todo["time_to_complete"] == 60

    resp = await client.patch(f"{ITEMS}/{todo['id']}", headers=user, json={"solutions": None})
    assert "solutions" not in resp.json()

    resp = await client.post(
        ITEMS, headers=user, json={"type": "note", "title": "x", "solutions": ["y"]}
    )
    assert resp.status_code == 422

    note = await _create_item(client, user, {"type": "note", "title": "x"})
    for field, value in (("solutions", ["y"]), ("time_to_complete", 5)):
        resp = await client.patch(f"{ITEMS}/{note['id']}", headers=user, json={field: value})
        assert resp.status_code == 422

    resp = await client.post(
        ITEMS, headers=user, json={"type": "todo", "title": "x", "time_to_complete": 0}
    )
    assert resp.status_code == 422


# ------------------------------------------------------------------ questions


async def test_find_todos_returns_only_open_todos_due_in_the_range(client, user, model):
    """"What's due tomorrow?" is a read: the lookup filters by local day, never proposes."""
    # 22:30Z on the 15th is already the 16th in Warsaw, so it belongs to the 16th.
    await _create_item(
        client, user, {"type": "todo", "title": "Call the plumber", "date": "2026-10-15T22:30:00Z"}
    )
    await _create_item(
        client, user, {"type": "todo", "title": "Pay rent", "date": "2026-10-15T08:00:00Z"}
    )
    done = await _create_item(
        client, user, {"type": "todo", "title": "Water plants", "date": "2026-10-16T09:00:00Z"}
    )
    await client.patch(f"{ITEMS}/{done['id']}", headers=user, json={"completed": True})
    await _create_item(client, user, {"type": "todo", "title": "Someday task"})

    thread_id = await _new_thread(client, user)
    model.script(
        ai(
            "",
            tool_call(
                "FindTodos", {"period": "dates", "start": "2026-10-16", "end": "2026-10-16"}
            ),
        ),
        ai("Tomorrow you need to call the plumber."),
    )
    body = await _send(client, user, thread_id, "What's due tomorrow?", tz="Europe/Warsaw")

    assert body["status"] == "completed"
    assert body["pending_review"] is None
    assert body["messages"][-1]["content"] == "Tomorrow you need to call the plumber."

    # The second model call saw the lookup's result as the last message.
    lookup = str(model.received[1][-1].content)
    assert "Call the plumber" in lookup
    assert "Friday, October 16, 2026 at 12:30 AM" in lookup
    for left_out in ("Pay rent", "Water plants", "Someday task"):
        assert left_out not in lookup


async def test_find_todos_without_a_range_lists_every_open_todo(client, user, model):
    await _create_item(
        client, user, {"type": "todo", "title": "Later", "date": "2026-11-01T09:00:00Z"}
    )
    await _create_item(
        client, user, {"type": "todo", "title": "Sooner", "date": "2026-10-20T09:00:00Z"}
    )
    await _create_item(client, user, {"type": "todo", "title": "Undated"})

    thread_id = await _new_thread(client, user)
    model.script(
        ai("", tool_call("FindTodos", {"period": "all"})),
        ai("Here is everything."),
    )
    await _send(client, user, thread_id, "What do I need to do?", tz="UTC")

    lookup = str(model.received[1][-1].content)
    assert lookup.index("Sooner") < lookup.index("Later") < lookup.index("Undated")


def test_find_todos_periods_resolve_to_local_days():
    from datetime import date

    from app.agent.nodes import _period_bounds

    saturday = date(2026, 10, 10)

    def bounds(**args):
        return _period_bounds(args, saturday)[:2]

    assert bounds(period="all") == (None, None)
    assert bounds(period="today") == (saturday, saturday)
    assert bounds(period="tomorrow") == (date(2026, 10, 11),) * 2
    assert bounds(period="this_week") == (date(2026, 10, 5), date(2026, 10, 11))
    assert bounds(period="next_week") == (date(2026, 10, 12), date(2026, 10, 18))
    assert bounds(period="next_7_days") == (saturday, date(2026, 10, 16))
    assert bounds(period="dates", start="2026-10-20") == (date(2026, 10, 20),) * 2
    assert bounds(period="dates", start="2026-10-22", end="2026-10-20") == (
        date(2026, 10, 20),
        date(2026, 10, 22),
    )
    # Anything unusable lists everything rather than hiding todos.
    assert bounds(period="dates") == (None, None)
    assert bounds(period="someday") == (None, None)
    assert bounds() == (None, None)
