#!/usr/bin/env python3
"""Validate the static English syntax map.

Checks JSON validity, referential integrity (edge endpoints, node categories,
CEFR levels), absence of self-loops, and that the prerequisite graph is a DAG
(no cycles). Prints a summary and exits non-zero on any error.

Usage:
    python3 packages/map/validate.py [path-to-map.json]
"""
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

CEFR_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}


def validate(path: Path) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON: {e}")
        return 1

    categories = {c["id"]: c for c in data.get("categories", [])}
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    edges = data.get("edges", [])

    if len(categories) != len(data.get("categories", [])):
        errors.append("Duplicate category ids found.")
    if len(nodes) != len(data.get("nodes", [])):
        errors.append("Duplicate node ids found.")

    # Node-level checks
    for n in data.get("nodes", []):
        nid = n.get("id", "<missing id>")
        if n.get("category") not in categories:
            errors.append(f"Node '{nid}' has unknown category '{n.get('category')}'.")
        if n.get("cefr") not in CEFR_LEVELS:
            errors.append(f"Node '{nid}' has invalid CEFR level '{n.get('cefr')}'.")

    # Edge-level checks + adjacency for cycle detection
    succ: dict[str, list[str]] = defaultdict(list)
    indeg: dict[str, int] = {nid: 0 for nid in nodes}
    for e in edges:
        a, b = e.get("from"), e.get("to")
        if a not in nodes:
            errors.append(f"Edge from unknown node '{a}'.")
        if b not in nodes:
            errors.append(f"Edge to unknown node '{b}'.")
        if a == b:
            errors.append(f"Self-loop on '{a}'.")
        if a in nodes and b in nodes and a != b:
            succ[a].append(b)
            indeg[b] += 1

    # Cycle detection (Kahn's algorithm)
    if not any("unknown node" in e for e in errors):
        q = deque([nid for nid, d in indeg.items() if d == 0])
        seen = 0
        indeg_copy = dict(indeg)
        while q:
            u = q.popleft()
            seen += 1
            for v in succ[u]:
                indeg_copy[v] -= 1
                if indeg_copy[v] == 0:
                    q.append(v)
        if seen != len(nodes):
            errors.append(
                f"Prerequisite graph has a cycle ({len(nodes) - seen} node(s) "
                "involved). It must be a DAG."
            )

    # Soft checks
    orphans = [
        nid for nid in nodes
        if indeg[nid] == 0 and not succ[nid]
    ]
    if orphans:
        warnings.append(f"{len(orphans)} isolated node(s) (no edges): {', '.join(orphans)}")

    # Report
    print(f"Map: {path}")
    print(f"  categories: {len(categories)}")
    print(f"  nodes:      {len(nodes)}")
    print(f"  edges:      {len(edges)}")
    by_level = defaultdict(int)
    for n in nodes.values():
        by_level[n.get("cefr")] += 1
    print("  by CEFR:    " + ", ".join(f"{lvl}={by_level.get(lvl, 0)}" for lvl in sorted(CEFR_LEVELS)))

    for w in warnings:
        print(f"  WARN: {w}")
    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("\nOK: map is valid (referential integrity + DAG).")
    return 0


if __name__ == "__main__":
    default = Path(__file__).parent / "english-syntax-map.v0.json"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    sys.exit(validate(target))
