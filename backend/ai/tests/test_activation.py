"""End-to-end activation engine: events -> evidence -> propagation -> threshold.

Skips cleanly if torch_geometric is unavailable (the default config propagates).
"""
from collections import Counter

import pytest

pytest.importorskip("torch_geometric")

import klg_ai as k
from klg_ai.core.activation import EngineConfig, activated_from_events
from klg_ai.data.adapters.synthetic import generate_events


def test_demo_activation_matches_known_set():
    assert k.get_activation("demo") == set(k.DEMO_KNOWN)


def test_unknown_learner_is_none():
    assert k.get_activation("nobody") is None
    assert k.compute_mastery("nobody") is None


def test_demo_status_counts_preserved():
    st = k.compute_status(k.default_graph(), k.get_activation("demo"))
    counts = Counter(s.value for s in st.values())
    assert counts["known"] == 48
    assert counts["interior_gap"] == 3


def test_demo_gaps_lifted_but_not_fabricated():
    m = k.compute_mastery("demo")
    for gap in ("second_conditional", "indirect_questions", "introductory_it_there"):
        assert 0.0 < m[gap] < k.DEFAULT_CONFIG.known_threshold


def test_propagation_off_equals_direct_evidence():
    # The demo's direct evidence alone == its known set (propagation adds no
    # *status* changes for it, only confidence underneath).
    act = k.get_activation("demo", config=EngineConfig(propagation=False))
    assert act == set(k.DEMO_KNOWN)


def test_forgetting_fades_mastery_over_time():
    fresh = k.get_activation("demo", now=0.0)
    faded = k.get_activation("demo", now=400.0)  # ~13 half-lives later
    assert len(faded) < len(fresh)


def test_inferred_ceiling_must_be_below_threshold():
    with pytest.raises(ValueError):
        EngineConfig(known_threshold=0.5, inferred_ceiling=0.5)


def test_synthetic_learner_activates_its_known_core():
    known = ["present_simple", "present_continuous", "past_simple"]
    events = generate_events(known, seed=1, reviews_per_node=5, accuracy=1.0)
    act = activated_from_events(k.default_graph(), events)
    assert set(known) <= act


def test_timeline_unknown_learner_is_none():
    assert k.mastery_timeline("nobody") is None


def test_timeline_frames_are_time_ordered_and_cover_every_node():
    frames = k.mastery_timeline("intermediate", frames=12)
    assert len(frames) == 12
    times = [f.t for f in frames]
    assert times == sorted(times)
    n_nodes = k.default_graph().number_of_nodes()
    assert all(len(f.mastery) == n_nodes for f in frames)  # every node scored each frame


def test_timeline_shows_growth_then_decay():
    # Total mastery rises as a synthetic learner's reviews accrue, peaks once the
    # evidence window closes, then falls as forgetting takes over.
    frames = k.mastery_timeline("intermediate", frames=24)
    totals = [sum(f.mastery.values()) for f in frames]
    peak = max(range(len(totals)), key=lambda i: totals[i])
    assert totals[0] < totals[peak]    # grew from the first frame
    assert totals[-1] < totals[peak]   # decayed by the last frame


def test_timeline_demo_starts_full_then_forgets():
    # All demo evidence lands at t=0, so frame 0 is the full known set (48) and
    # later frames can only forget it (pure inference never refills it).
    frames = k.mastery_timeline("demo", frames=10)
    assert len(frames[0].activated) == 48
    assert len(frames[-1].activated) < 48
