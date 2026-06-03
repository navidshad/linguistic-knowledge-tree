"""Build the prerequisite graph (networkx) from the static map."""
from __future__ import annotations

from functools import lru_cache

import networkx as nx

from .loader import SyntaxMap, load_map


def build_graph(m: SyntaxMap) -> nx.DiGraph:
    """Directed graph; an edge u -> v means u is a prerequisite of v."""
    g = nx.DiGraph()
    for n in m.nodes:
        g.add_node(n.id, label=n.label, category=n.category, cefr=n.cefr, description=n.description)
    for e in m.edges:
        g.add_edge(e.source, e.target, type=e.type)
    return g


@lru_cache(maxsize=1)
def default_graph() -> nx.DiGraph:
    """The prerequisite graph for the default static map, built once and cached.

    Lets the engine compute activation without the caller threading the graph
    through (the server keeps its own cached copy in ``klg_server.deps``).
    """
    return build_graph(load_map())
