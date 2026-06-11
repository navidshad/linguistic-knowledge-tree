"""Gap detection: the 'nodes in between' two activated concepts.

An interior gap is an unactivated node that sits on a prerequisite path between
two activated nodes — i.e. it has at least one activated ancestor AND one
activated descendant. These are concepts the learner skipped between things they
already know (e.g. 'second conditional' when they know first and third).

``gap_scores`` (thesis §3.7) ranks the *actionable* gaps — interior gaps and
frontier nodes — for the separate "what's-next" recommender this system hands
off to (see ``docs/knowledge-state-api.md``). Recommendation policy itself
stays out of scope.
"""
from __future__ import annotations

from collections.abc import Mapping

import networkx as nx

# CEFR levels in curriculum order; earlier levels weigh more (a foundational
# gap blocks more of the map than an advanced one).
_CEFR_ORDER = ("A1", "A2", "B1", "B2", "C1", "C2")


def _level_weight(cefr: str) -> float:
    try:
        i = _CEFR_ORDER.index(cefr)
    except ValueError:
        i = len(_CEFR_ORDER) - 1  # unknown level: weigh least
    return (len(_CEFR_ORDER) - i) / len(_CEFR_ORDER)


def gap_scores(
    g: nx.DiGraph,
    statuses: Mapping[str, str],
    mastery: Mapping[str, float],
) -> dict[str, float]:
    """Pedagogical priority per actionable node: ``level_weight × (1 − mastery)``.

    Scores only interior-gap and frontier nodes — the recommender's candidate
    set (known needs no teaching; ``further`` fails prerequisite feasibility,
    §3.7's prune step). A1 weighs 1.0 down to C2 at 1/6, so foundational gaps
    outrank advanced ones at equal mastery. ``statuses`` values may be
    ``Status`` enums or their string values (``Status`` is a str-Enum; comparing
    on strings keeps this module import-free of ``status.py``, which imports us).
    """
    actionable = ("interior_gap", "frontier")
    out: dict[str, float] = {}
    for n, s in statuses.items():
        if s not in actionable:
            continue
        w = _level_weight(g.nodes[n].get("cefr", ""))
        out[n] = w * (1.0 - mastery.get(n, 0.0))
    return out


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
