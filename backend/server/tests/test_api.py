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


def test_learners_list():
    r = client.get("/api/learners")
    assert r.status_code == 200
    ids = {p["id"] for p in r.json()}
    assert {"demo", "beginner", "intermediate"} <= ids


def test_synthetic_learner_status_differs_from_demo():
    demo_known = client.get("/api/learner/demo/status").json()["counts"]["known"]
    beginner = client.get("/api/learner/beginner/status")
    assert beginner.status_code == 200
    # a beginner knows a real, smaller slice of the map than the demo learner
    assert 0 < beginner.json()["counts"]["known"] < demo_known


def test_learner_status_exposes_mastery():
    d = client.get("/api/learner/demo/status").json()
    m = d["mastery"]
    assert len(m) == 113                          # every node scored
    assert m["present_simple"] >= 0.5             # a known node is confident
    assert 0.0 < m["second_conditional"] < 0.5    # interior gap: lifted but not "known"


def test_post_status_has_no_mastery():
    # The what-if endpoint is handed an activated set, not evidence — no mastery.
    r = client.post("/api/status", json={"activated": ["present_simple"]})
    assert r.json()["mastery"] is None


def test_timeline_endpoint_shape():
    r = client.get("/api/learner/intermediate/timeline?frames=10")
    assert r.status_code == 200
    d = r.json()
    assert d["learner_id"] == "intermediate"
    assert len(d["frames"]) == 10
    times = [f["t"] for f in d["frames"]]
    assert times == sorted(times)
    assert (d["t_start"], d["t_end"]) == (times[0], times[-1])
    f0 = d["frames"][0]
    assert set(f0) == {"t", "counts", "statuses", "mastery"}
    assert len(f0["mastery"]) == 113


def test_timeline_frames_are_clamped():
    assert len(client.get("/api/learner/demo/timeline?frames=999").json()["frames"]) == 60


def test_timeline_unknown_learner_404():
    assert client.get("/api/learner/nobody/timeline").status_code == 404
