"""Shared dependencies: load the map + build the graph once (cached)."""
from __future__ import annotations

from functools import lru_cache

import networkx as nx

from klg_ai.core.graph import build_graph
from klg_ai.core.loader import DEFAULT_MAP_PATH, SyntaxMap, load_map


@lru_cache(maxsize=1)
def get_map() -> SyntaxMap:
    return load_map(DEFAULT_MAP_PATH)


@lru_cache(maxsize=1)
def get_graph() -> nx.DiGraph:
    return build_graph(get_map())
