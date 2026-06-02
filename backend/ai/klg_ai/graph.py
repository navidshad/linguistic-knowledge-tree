"""Build the prerequisite graph (networkx) from the static map."""
from __future__ import annotations

import networkx as nx

from .loader import SyntaxMap


def build_graph(m: SyntaxMap) -> nx.DiGraph:
    """Directed graph; an edge u -> v means u is a prerequisite of v."""
    g = nx.DiGraph()
    for n in m.nodes:
        g.add_node(n.id, label=n.label, category=n.category, cefr=n.cefr, description=n.description)
    for e in m.edges:
        g.add_edge(e.source, e.target, type=e.type)
    return g
