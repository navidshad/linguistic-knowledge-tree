"""klg_ai — the Linguistic Knowledge Graph AI engine.

Pure-Python, no web dependencies: load the static map, build the prerequisite
graph, compute per-learner activation, status, and gaps.
"""
from .activation import DEMO_KNOWN, Event, get_activation
from .gaps import connecting_subgraph, interior_gaps
from .graph import build_graph
from .loader import DEFAULT_MAP_PATH, Edge, Node, SyntaxMap, load_map
from .status import Status, compute_status

__all__ = [
    "DEFAULT_MAP_PATH", "load_map", "SyntaxMap", "Node", "Edge",
    "build_graph",
    "compute_status", "Status",
    "interior_gaps", "connecting_subgraph",
    "get_activation", "DEMO_KNOWN", "Event",
]
