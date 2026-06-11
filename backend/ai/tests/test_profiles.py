"""Persistent learner profiles (file-backed store + activation wiring).

Every test points ``KLG_PROFILE_DIR`` at a tmp dir, so the real ``data/profiles``
is never touched and tests don't see each other's profiles. The central
guarantee: built-in synthetic learners resolve exactly as before, so the rest of
the suite is unaffected.
"""
import json

import pytest

from klg_ai import (
    Event,
    append_events,
    create_profile,
    delete_profile,
    list_profiles,
    load_conversation,
    load_events,
    mastery_from_events,
    save_conversation,
    threshold_activated,
)
from klg_ai.core.activation import _cefr_nodes, _events_for, demo_events, list_learners
from klg_ai.core.graph import default_graph


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("KLG_PROFILE_DIR", str(tmp_path))


def test_create_and_load_round_trip():
    doc = create_profile("Navid")
    pid = doc["id"]
    append_events(pid, [Event(pid, ("present_simple",), True)])
    events = load_events(pid)
    assert len(events) == 1
    assert events[0].node_ids == ("present_simple",)
    assert events[0].learner_id == pid  # reattached from the file id


def test_seed_from_cefr_marks_those_nodes_known():
    a1 = _cefr_nodes("A1")
    doc = create_profile("Seeded", seed_level="A1", seed_nodes=a1)
    events = load_events(doc["id"])
    assert len(events) == len(a1) * 3  # reviews_per_node=3
    known = threshold_activated(mastery_from_events(default_graph(), events))
    assert set(a1) <= known


def test_append_ts_is_monotonic_and_wall_clock_independent():
    pid = create_profile("Mono")["id"]
    append_events(pid, [Event(pid, ("a",), True)])
    append_events(pid, [Event(pid, ("b",), True), Event(pid, ("c",), True)])
    ts = [e.ts for e in load_events(pid)]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)  # strictly increasing, unique
    assert ts == [0.0, 1.0, 2.0]  # deterministic counter, not wall-clock


def test_built_ins_untouched_with_store_active():
    # The store is set, but built-ins still resolve to their synthetic streams.
    assert _events_for("demo") == demo_events()
    assert _events_for("nope") is None
    create_profile("Extra")
    learners = {p.id: p.editable for p in list_learners()}
    assert learners["demo"] is False and learners["struggling"] is False
    assert {"demo", "beginner", "intermediate"} <= set(learners)
    assert any(editable for editable in learners.values())  # the created profile


def test_conversation_round_trip():
    pid = create_profile("Chatter")["id"]
    convo = [{"role": "user", "text": "hi"}, {"role": "tutor", "text": "hello"}]
    save_conversation(pid, convo)
    assert load_conversation(pid) == convo


def test_delete_and_reserved_ids():
    pid = create_profile("Temp")["id"]
    assert delete_profile(pid) is True
    assert load_events(pid) is None
    assert delete_profile("demo") is False     # built-in id is reserved
    assert delete_profile("missing") is False


def test_new_profile_cannot_shadow_a_reserved_id():
    # Slug+suffix never equals a bare reserved id, and the suffix keeps it unique.
    doc = create_profile("demo")
    assert doc["id"] != "demo"
    assert load_events("demo") is None  # store has no "demo" file; built-in handles it


def test_corrupt_file_is_skipped(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    good = create_profile("Good")["id"]
    ids = {m.id for m in list_profiles()}
    assert good in ids and "broken" not in ids


def test_load_events_is_none_for_unknown():
    assert load_events("does-not-exist") is None
    assert load_conversation("does-not-exist") is None
