"""Load and represent the static English syntax map."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

# repo_root/map/english-syntax-map.v0.json  (loader.py is backend/ai/klg_ai/core/loader.py)
DEFAULT_MAP_PATH = Path(__file__).resolve().parents[4] / "map" / "english-syntax-map.v0.json"


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    category: str
    cefr: str
    description: str = ""


@dataclass(frozen=True)
class Edge:
    source: str  # the prerequisite
    target: str  # the dependent concept
    type: str = "prerequisite"


@dataclass
class SyntaxMap:
    meta: dict
    categories: list[dict]
    nodes: list[Node]
    edges: list[Edge]

    @property
    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}

    def node(self, node_id: str) -> Node:
        return next(n for n in self.nodes if n.id == node_id)


def load_map(path: str | Path = DEFAULT_MAP_PATH) -> SyntaxMap:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [
        Node(
            id=n["id"],
            label=n["label"],
            category=n["category"],
            cefr=n["cefr"],
            description=n.get("description", ""),
        )
        for n in data["nodes"]
    ]
    edges = [
        Edge(source=e["from"], target=e["to"], type=e.get("type", "prerequisite"))
        for e in data["edges"]
    ]
    return SyntaxMap(meta=data.get("meta", {}), categories=data["categories"], nodes=nodes, edges=edges)
