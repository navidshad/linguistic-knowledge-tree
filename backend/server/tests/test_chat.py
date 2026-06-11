"""POST /api/chat — the Gemini chat demo activates the map (Phase 6, Step 5).

Deterministic: a low-threshold ``HashingEmbedder`` mapper + the Gemini mock, so no
model download and no API key. Verifies turns -> nodes -> live status + per-node
evidence, and that the legacy endpoints are unchanged.
"""
import pytest
from fastapi.testclient import TestClient

import klg_server.routers.chat as chat_mod
from klg_ai.loader import load_map
from klg_ai.semantic import HashingEmbedder, SemanticMapper, build_node_vectors
from klg_server.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _deterministic(monkeypatch):
    monkeypatch.setenv("KLG_GEMINI_MOCK", "1")
    monkeypatch.setenv("KLG_TRACE", "0")  # don't write trace files in most tests
    chat_mod._GRADE_CACHE.clear()  # grades are cached per (text, nodes) across requests
    emb = HashingEmbedder(dim=64)
    mapper = SemanticMapper(
        emb, build_node_vectors(emb, load_map()),
        threshold=0.0, top_k=2, knowledge_injection=True, syntax_map=load_map(),
    )
    monkeypatch.setattr(chat_mod, "_mapper", lambda: mapper)


def test_chat_empty_conversation():
    r = client.post("/api/chat", json={"messages": [], "activated": []})
    assert r.status_code == 200
    d = r.json()
    assert d["reply"]                       # deterministic mock greeting
    assert d["mapped_nodes"] == []
    assert d["evidence"] == []
    assert len(d["statuses"]) == 113        # full map overlay
    assert d["counts"].get("known", 0) == 0


def test_chat_turn_maps_and_activates():
    turn = "she is reading a book and i have finished my homework"
    r = client.post("/api/chat", json={"messages": [{"role": "user", "text": turn}]})
    assert r.status_code == 200
    d = r.json()
    assert d["reply"]
    assert d["mapped_nodes"]                              # low threshold -> non-empty
    assert set(d["mapped_nodes"]) <= set(d["statuses"])
    assert set(d["confidences"]) == set(d["mapped_nodes"])
    assert d["evidence"]
    assert all(e["turn_indices"] == [0] for e in d["evidence"])  # the only user turn
    assert d["mastery"]                                   # mastery overlay populated


def test_chat_mock_reply_is_deterministic_and_echoes():
    turn = "I like coffee"
    a = client.post("/api/chat", json={"messages": [{"role": "user", "text": turn}]}).json()
    b = client.post("/api/chat", json={"messages": [{"role": "user", "text": turn}]}).json()
    assert a["reply"] == b["reply"]
    assert "I like coffee" in a["reply"]


def test_chat_does_not_break_legacy_endpoints():
    assert client.get("/api/learner/demo/status").json()["counts"]["known"] == 48
    assert len(client.get("/api/map").json()["nodes"]) == 113


def _first_mapped_node(text: str) -> str:
    """The first node the (mock-graded, all-correct) mapper assigns to a turn."""
    d = client.post("/api/chat", json={"messages": [{"role": "user", "text": text}]}).json()
    assert d["mapped_nodes"]
    return d["mapped_nodes"][0]


def test_incorrect_usage_is_negative_evidence(monkeypatch):
    turn = "she is reading a book and i have finished my homework"
    target = _first_mapped_node(turn)
    baseline = client.post(
        "/api/chat", json={"messages": [{"role": "user", "text": turn}]}).json()
    assert baseline["grades"][target] is True
    assert baseline["mastery"][target] >= 0.5  # all-correct grading -> known

    # Grade the target node as used-but-wrong; everything else stays correct.
    def grade_target_wrong(text, candidates, **kw):
        return {nid: {"used": True, "correct": nid != target} for nid, _ in candidates}

    chat_mod._GRADE_CACHE.clear()
    monkeypatch.setattr(chat_mod, "gemini_grade", grade_target_wrong)
    d = client.post("/api/chat", json={"messages": [{"role": "user", "text": turn}]}).json()
    assert d["grades"][target] is False
    assert d["mastery"][target] < baseline["mastery"][target]  # wrong use lowers mastery
    assert d["statuses"][target] != "known"                    # and never fabricates known
    ev = next(e for e in d["evidence"] if e["node_id"] == target)
    assert ev["incorrect_turn_indices"] == [0]


def test_unused_candidates_are_pruned(monkeypatch):
    turn = "she is reading a book and i have finished my homework"
    target = _first_mapped_node(turn)

    def grade_target_unused(text, candidates, **kw):
        return {nid: {"used": nid != target, "correct": nid != target} for nid, _ in candidates}

    chat_mod._GRADE_CACHE.clear()
    monkeypatch.setattr(chat_mod, "gemini_grade", grade_target_unused)
    d = client.post("/api/chat", json={"messages": [{"role": "user", "text": turn}]}).json()
    assert target not in d["mapped_nodes"]                       # pruned from the lit-up set
    assert all(e["node_id"] != target for e in d["evidence"])    # and from the evidence
    assert d["mastery"][target] == 0.0                           # no event emitted for it


def test_chat_writes_per_session_trace(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("KLG_TRACE", "1")
    monkeypatch.setenv("KLG_TRACE_DIR", str(tmp_path))

    body = {"messages": [{"role": "user", "text": "she is reading a book"}],
            "activated": [], "session_id": "sess-A"}
    assert client.post("/api/chat", json=body).status_code == 200

    f = tmp_path / "sess-A.json"
    assert f.exists()
    doc = json.loads(f.read_text())          # a single valid JSON object
    assert doc["session"] == "sess-A"
    rec = doc["turns"][-1]
    assert rec["input"] == "she is reading a book"
    assert rec["turn"] == 0
    assert "matched" in rec["mapper"]
    assert isinstance(rec["touched"], list)
    assert isinstance(rec["gnn_inferred"], list)
    assert "status_counts" in rec and "tutor_reply" in rec
    # each request appends one turn; a different session writes a separate file
    client.post("/api/chat", json={**body, "messages": body["messages"] + [
        {"role": "tutor", "text": "ok"}, {"role": "user", "text": "I have finished"}]})
    assert len(json.loads(f.read_text())["turns"]) == 2
    client.post("/api/chat", json={**body, "session_id": "sess-B"})
    assert (tmp_path / "sess-B.json").exists()


def test_chat_returns_edge_adjustments_field(monkeypatch):
    # KGT is on by default in chat: the field is a (possibly empty) list.
    turn = "she is reading a book and i have finished my homework"
    d = client.post("/api/chat", json={"messages": [{"role": "user", "text": turn}]}).json()
    assert isinstance(d["edge_adjustments"], list)

    # The kill-switch turns the personal graph off entirely.
    monkeypatch.setenv("KLG_CHAT_KGT", "0")
    d = client.post("/api/chat", json={"messages": [{"role": "user", "text": turn}]}).json()
    assert d["edge_adjustments"] is None
