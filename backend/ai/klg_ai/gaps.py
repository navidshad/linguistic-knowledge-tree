"""Gap detection: the 'nodes in between' two activated concepts.

An interior gap is an unactivated node that sits on a prerequisite path between
two activated nodes — i.e. it has at least one activated ancestor AND one
activated descendant. These are concepts the learner skipped between things they
already know (e.g. 'second conditional' when they know first and third).
"""
from __future__ import annotations

import networkx as nx


def interior_gaps(g: nx.DiGraph, activated: set[str]) -> set[str]:
    gaps: set[str] = set()
    for n in g.nodes:
        if n in activated:
            continue
        if (nx.ancestors(g, n) & activated) and (nx.descendants(g, n) & activated):
            gaps.add(n)
    return gaps


def connecting_subgraph(g: nx.DiGraph, activated: set[str]) -> set[str]:
    """The learner's 'relevant span': activated nodes plus the interior gaps
    that connect them."""
    return set(activated) | interior_gaps(g, activated)
