"""KGT per-learner edge tuning (thesis RQ5, §3.8).

The closed-form rule itself is torch-free; tests that run propagation over the
tuned graph skip cleanly without torch_geometric. The non-negotiable invariant
throughout: KGT redistributes *inference*, so the demo contract (48 known /
3 interior gaps) and both propagation guarantees survive with the toggle on.
"""
import pytest

import networkx as nx

from klg_ai.activation import (
    DEMO_KNOWN,
    EngineConfig,
    compute_edge_adjustments,
    demo_events,
    generate_events,
)
from klg_ai.events import Event
from klg_ai.kgt import tune_edges

KGT = EngineConfig(kgt=True)


def _chain() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edge("a", "b", label="A")  # a prereq of b
    g.nodes["a"]["label"] = "A"
    g.nodes["b"]["label"] = "B"
    return g


def _events(node: str, *, correct: bool, n: int = 4) -> list[Event]:
    return [Event(learner_id="t", node_ids=(node,), correct=correct, ts=0.0) for _ in range(n)]


def test_no_events_no_adjustments():
    res = tune_edges(_chain(), [], KGT)
    assert res.factors == {}
    assert res.adjustments == []


def test_evidence_on_one_endpoint_only_is_not_contradiction():
    # b demonstrably known, a never seen: absence of evidence must not move the edge.
    res = tune_edges(_chain(), _events("b", correct=True), KGT)
    assert res.factors == {}


def test_contradiction_weakens_the_backward_message():
    # Knows the dependent b, consistently fails the prereq a.
    events = _events("b", correct=True) + _events("a", correct=False)
    res = tune_edges(_chain(), events, KGT)
    m_back, m_fwd = res.factors[("a", "b")]
    assert m_back < 1.0
    assert m_fwd == pytest.approx(1.0)  # the forward readiness message is untouched here


def test_strong_contradiction_removes_the_edge():
    events = _events("b", correct=True, n=5) + _events("a", correct=False, n=6)
    res = tune_edges(_chain(), events, KGT)
    (adj,) = res.adjustments
    assert adj.kind == "removed"
    assert adj.factor_back <= 0.1


def test_agreement_strengthens_both_messages():
    events = _events("a", correct=True) + _events("b", correct=True)
    res = tune_edges(_chain(), events, KGT)
    m_back, m_fwd = res.factors[("a", "b")]
    assert m_back > 1.0 and m_fwd > 1.0


def test_factors_respect_the_configured_bounds():
    events = _events("b", correct=True, n=20) + _events("a", correct=False, n=20)
    for cfg in (KGT, EngineConfig(kgt=True, kgt_factor_min=0.25, kgt_factor_max=1.5)):
        res = tune_edges(_chain(), events, cfg)
        for m_back, m_fwd in res.factors.values():
            assert cfg.kgt_factor_min <= m_back <= cfg.kgt_factor_max
            assert cfg.kgt_factor_min <= m_fwd <= cfg.kgt_factor_max


def test_adjustments_carry_kind_and_reason():
    events = _events("b", correct=True) + _events("a", correct=False)
    for adj in tune_edges(_chain(), events, KGT).adjustments:
        assert adj.kind in ("strengthened", "weakened", "removed")
        assert adj.reason  # human-readable, non-empty
        assert "B" in adj.reason and "A" in adj.reason  # cites the node labels


def test_tune_edges_is_deterministic():
    events = _events("b", correct=True) + _events("a", correct=False)
    a = tune_edges(_chain(), events, KGT)
    b = tune_edges(_chain(), events, KGT)
    assert a.factors == b.factors and a.adjustments == b.adjustments


def test_demo_learner_has_no_reportable_adjustments():
    # One-shot single reviews never clear the "consistent feedback" gate.
    assert compute_edge_adjustments("demo") == []


def test_struggling_learner_has_weakened_edges():
    adjustments = compute_edge_adjustments("struggling")
    weakened = {(a.source, a.target) for a in adjustments if a.factor_back < 1.0}
    assert ("relative_pronouns", "relative_clauses_defining") in weakened
    assert ("second_conditional", "third_conditional") in weakened


def test_generate_events_failed_nodes_mostly_incorrect():
    events = generate_events(["x"], failed=["y"], seed=3, accuracy=1.0, reviews_per_node=4)
    by_node = {"x": [], "y": []}
    for e in events:
        by_node[e.node_ids[0]].append(e.correct)
    assert all(by_node["x"])
    assert not any(by_node["y"])


def test_generate_events_without_failed_is_byte_compatible():
    # The new keyword must not disturb existing streams (seeded rng draw order).
    assert generate_events(["x", "y"], seed=7) == generate_events(["x", "y"], seed=7, failed=[])


# --- with propagation over the tuned graph ---------------------------------

torch_tests = pytest.importorskip("torch_geometric", reason="propagation tests need PyG")

from klg_ai.activation import compute_mastery, get_activation  # noqa: E402
from klg_ai.graph import default_graph  # noqa: E402
from klg_ai.propagation import propagate  # noqa: E402


def test_demo_contract_survives_kgt():
    assert get_activation("demo", config=KGT) == set(DEMO_KNOWN)


def test_kgt_cuts_inference_into_contradicted_nodes():
    # The struggling learner fails relative_pronouns outright; without KGT the
    # graph lifts it to the ceiling on neighbours alone, with KGT the lift is cut.
    off = compute_mastery("struggling", config=EngineConfig())
    on = compute_mastery("struggling", config=KGT)
    assert on["relative_pronouns"] < off["relative_pronouns"]


def test_kgt_invariants_hold_on_the_tuned_graph():
    g = default_graph()
    events = demo_events() + generate_events(
        ["relative_clauses_defining"], failed=["relative_pronouns"],
        learner_id="demo", seed=5, reviews_per_node=4,
    )
    factors = tune_edges(g, events, KGT).factors
    from klg_ai.evidence import direct_scores
    direct = direct_scores(events)
    mastery = propagate(g, direct, KGT, edge_factors=factors)
    for n, m in mastery.items():
        assert 0.0 <= m <= 1.0
        assert m >= direct.get(n, 0.0) - 1e-6          # direct evidence never lowered
        if direct.get(n, 0.0) == 0.0:                  # inference-only stays sub-threshold
            assert m <= KGT.inferred_ceiling + 1e-6


def test_max_reinforcement_cannot_fabricate_known():
    # Force every edge to the max factor: zero-evidence nodes must stay capped.
    g = default_graph()
    factors = {(u, v): (KGT.kgt_factor_max, KGT.kgt_factor_max) for u, v in g.edges}
    direct = {n: 0.9 for n in list(g.nodes)[:20]}
    mastery = propagate(g, direct, KGT, edge_factors=factors)
    for n, m in mastery.items():
        if n not in direct:
            assert m <= KGT.inferred_ceiling + 1e-6


def test_edge_factors_none_matches_default_propagation():
    g = default_graph()
    direct = {n: 0.7 for n in list(g.nodes)[:10]}
    cfg = EngineConfig()
    assert propagate(g, direct, cfg) == propagate(g, direct, cfg, edge_factors=None)
