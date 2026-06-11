"""Regression test: the demo learner must reproduce the verified status counts."""
from collections import Counter

from klg_ai.core.activation import DEMO_KNOWN
from klg_ai.core.graph import build_graph
from klg_ai.core.loader import DEFAULT_MAP_PATH, load_map
from klg_ai.core.status import Status, compute_status


def _status():
    m = load_map(DEFAULT_MAP_PATH)
    g = build_graph(m)
    return compute_status(g, set(DEMO_KNOWN))


def test_status_counts():
    counts = Counter(s.value for s in _status().values())
    assert counts["known"] == 48
    assert counts["interior_gap"] == 3
    assert counts["frontier"] == 31
    assert counts["further"] == 31
    assert sum(counts.values()) == 113


def test_interior_gaps_are_the_expected_three():
    st = _status()
    gaps = {n for n, s in st.items() if s is Status.INTERIOR_GAP}
    assert gaps == {"second_conditional", "indirect_questions", "introductory_it_there"}


def test_known_nodes_match_activation():
    st = _status()
    known = {n for n, s in st.items() if s is Status.KNOWN}
    assert known == set(DEMO_KNOWN)
