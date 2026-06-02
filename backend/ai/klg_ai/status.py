"""Compute each node's status for a learner, from the graph + activated set.

Status is a pure function of (graph, activated set):

    known        node is activated (mastered)
    interior_gap unactivated, between two activated nodes (skipped)
    frontier     unactivated, all direct prerequisites activated (learn next)
    further      everything else not yet reachable
"""
from __future__ import annotations

from enum import Enum

import networkx as nx

from .gaps import interior_gaps


class Status(str, Enum):
    KNOWN = "known"
    INTERIOR_GAP = "interior_gap"
    FRONTIER = "frontier"
    FURTHER = "further"


def compute_status(g: nx.DiGraph, activated: set[str]) -> dict[str, Status]:
    gaps = interior_gaps(g, activated)
    status: dict[str, Status] = {}
    for n in g.nodes:
        if n in activated:
            status[n] = Status.KNOWN
        elif n in gaps:
            status[n] = Status.INTERIOR_GAP
        elif all(p in activated for p in g.predecessors(n)):
            status[n] = Status.FRONTIER
        else:
            status[n] = Status.FURTHER
    return status
