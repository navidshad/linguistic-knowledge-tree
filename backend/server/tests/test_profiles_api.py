"""Profile CRUD + chat persistence over the API.

Each test points ``KLG_PROFILE_DIR`` at a tmp dir (so nothing lands in the real
``data/profiles`` and tests are isolated) and forces the deterministic chat
stack. The existing ``test_api.py`` / ``test_chat.py`` set no profile dir and use
built-ins, so they're unaffected.
"""
import pytest
from fastapi.testclient import TestClient

from klg_server.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("KLG_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("KLG_GEMINI_MOCK", "1")
    monkeypatch.setenv("KLG_TRACE", "0")
    import klg_server.routers.chat as chat_mod
    chat_mod._GRADE_CACHE.clear()


def test_create_lists_with_editable_flag():
    r = client.post("/api/profiles", json={"name": "Navid"})
    assert r.status_code == 201
    pid = r.json()["id"]
    assert r.json()["editable"] is True

    profiles = {p["id"]: p["editable"] for p in client.get("/api/profiles").json()}
    assert profiles == {pid: True}  # /profiles is user profiles only

    learners = {p["id"]: p["editable"] for p in client.get("/api/learners").json()}
    assert learners[pid] is True and learners["demo"] is False
    assert {"demo", "beginner", "intermediate"} <= set(learners)  # built-ins intact


def test_seeded_profile_status_and_append():
    pid = client.post("/api/profiles", json={"name": "S", "seed_level": "A1"}).json()["id"]
    st = client.get(f"/api/learner/{pid}/status")
    assert st.status_code == 200
    known = st.json()["counts"].get("known", 0)
    assert known > 0  # A1 seeded

    # A node above A1 starts unknown; marking it known persists and lifts the count.
    target = "comparatives"  # A2
    assert st.json()["statuses"][target] != "known"
    r = client.post(f"/api/profiles/{pid}/events", json={"node_ids": [target], "correct": True})
    assert r.status_code == 200
    assert r.json()["statuses"][target] == "known"
    assert r.json()["counts"]["known"] == known + 1


def test_chat_persists_and_dedups_on_resend():
    pid = client.post("/api/profiles", json={"name": "Talker"}).json()["id"]
    body = {"messages": [{"role": "user", "text": "she is reading a book"}],
            "activated": [], "profile_id": pid}
    client.post("/api/chat", json=body)
    client.post("/api/chat", json=body)  # client resends the same one-turn history

    convo = client.get(f"/api/profiles/{pid}/conversation").json()["messages"]
    # Exactly one user + the latest tutor reply — the resend didn't duplicate.
    assert [m["role"] for m in convo] == ["user", "tutor"]


def test_chat_resumes_saved_conversation():
    pid = client.post("/api/profiles", json={"name": "Resume"}).json()["id"]
    client.post("/api/chat", json={"messages": [{"role": "user", "text": "I have finished"}],
                                   "activated": [], "profile_id": pid})
    convo = client.get(f"/api/profiles/{pid}/conversation").json()["messages"]
    assert convo and convo[0]["text"] == "I have finished"


def test_built_in_profiles_are_read_only():
    assert client.post("/api/profiles/demo/events", json={"node_ids": ["x"]}).status_code == 409
    assert client.delete("/api/profiles/demo").status_code == 409
    # chat against a built-in stays stateless — nothing is persisted for it.
    client.post("/api/chat", json={"messages": [{"role": "user", "text": "hi"}],
                                   "activated": [], "profile_id": "demo"})
    assert client.get("/api/profiles/demo/conversation").status_code == 404


def test_delete_profile():
    pid = client.post("/api/profiles", json={"name": "Doomed"}).json()["id"]
    assert client.delete(f"/api/profiles/{pid}").status_code == 204
    assert client.get(f"/api/learner/{pid}/status").status_code == 404
    assert pid not in {p["id"] for p in client.get("/api/profiles").json()}


def test_create_requires_a_name():
    assert client.post("/api/profiles", json={"name": "   "}).status_code == 422


def test_unknown_profile_event_404():
    assert client.post("/api/profiles/ghost/events", json={"node_ids": ["x"]}).status_code == 404
