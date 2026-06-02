from fastapi.testclient import TestClient

from klg_server.main import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_map():
    r = client.get("/api/map")
    assert r.status_code == 200
    d = r.json()
    assert len(d["nodes"]) == 113
    assert len(d["edges"]) == 111
    assert {c["id"] for c in d["categories"]}  # non-empty


def test_learner_status_demo():
    r = client.get("/api/learner/demo/status")
    assert r.status_code == 200
    d = r.json()
    assert d["counts"]["known"] == 48
    assert d["counts"]["interior_gap"] == 3
    assert d["statuses"]["second_conditional"] == "interior_gap"
    assert d["statuses"]["present_simple"] == "known"


def test_unknown_learner_404():
    assert client.get("/api/learner/nobody/status").status_code == 404


def test_post_status():
    r = client.post("/api/status", json={"activated": ["present_simple", "present_continuous"]})
    assert r.status_code == 200
    d = r.json()
    assert d["counts"]["known"] == 2
    assert d["statuses"]["present_simple"] == "known"
    # past_simple's only prerequisite (present_simple) is known -> frontier
    assert d["statuses"]["past_simple"] == "frontier"


def test_post_status_empty():
    r = client.post("/api/status", json={"activated": []})
    assert r.status_code == 200
    assert r.json()["counts"].get("known", 0) == 0
