"""GNN propagation invariants (thesis RQ3).

Uses a tiny prerequisite chain a -> b -> c for controllable, exact assertions.
Skips cleanly if torch_geometric is unavailable.
"""
import pytest

pytest.importorskip("torch_geometric")

import networkx as nx

from klg_ai.core.activation import DEFAULT_CONFIG
from klg_ai.core.propagation import propagate


def _chain() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from([("a", "b"), ("b", "c")])  # a prereq of b prereq of c
    return g


def test_known_node_is_never_lowered():
    m = propagate(_chain(), {"a": 0.9, "b": 0.0, "c": 0.0}, DEFAULT_CONFIG)
    assert m["a"] >= 0.9 - 1e-6


def test_zero_evidence_node_capped_below_threshold():
    # b has no evidence but sits between known a and c.
    m = propagate(_chain(), {"a": 0.9, "b": 0.0, "c": 0.9}, DEFAULT_CONFIG)
    assert m["b"] <= DEFAULT_CONFIG.inferred_ceiling + 1e-6
    assert m["b"] < DEFAULT_CONFIG.known_threshold


def test_neighbours_lift_a_zero_evidence_gap():
    m = propagate(_chain(), {"a": 0.9, "b": 0.0, "c": 0.9}, DEFAULT_CONFIG)
    assert m["b"] > 0.0  # graph inference lifts it above its (zero) direct score


def test_propagation_corroborates_weak_evidence():
    alone = propagate(_chain(), {"a": 0.0, "b": 0.4, "c": 0.0}, DEFAULT_CONFIG)["b"]
    supported = propagate(_chain(), {"a": 0.9, "b": 0.4, "c": 0.9}, DEFAULT_CONFIG)["b"]
    assert supported > alone


def test_backward_inference_is_stronger_than_forward():
    # Knowing the *dependent* (c) says more about b than knowing the *prereq* (a).
    from_dependent = propagate(_chain(), {"a": 0.0, "b": 0.0, "c": 0.9}, DEFAULT_CONFIG)["b"]
    from_prereq = propagate(_chain(), {"a": 0.9, "b": 0.0, "c": 0.0}, DEFAULT_CONFIG)["b"]
    assert from_dependent > from_prereq
