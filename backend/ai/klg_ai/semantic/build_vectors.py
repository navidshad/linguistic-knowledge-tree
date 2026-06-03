"""CLI: build and save the committed node-vector artifact.

    python -m klg_ai.semantic.build_vectors --out docs/phase6/node_vectors.npz

Embeds every map node (vanilla + K-BERT graph-injected) with the real
sentence-transformers backend and writes the ``.npz`` the server/eval load. Prints
the map hash so a map edit that invalidates the artifact is visible. Pass
``--embedder hash`` to produce a deterministic dependency-free artifact (lower
quality; for environments without sentence-transformers).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ..loader import DEFAULT_MAP_PATH, load_map
from .embedder import HashingEmbedder, SentenceTransformerEmbedder, default_embedder
from .node_vectors import DEFAULT_VECTORS_PATH, build_node_vectors, save_node_vectors


def _map_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the Phase 6 node-vector artifact")
    p.add_argument("--out", type=Path, default=DEFAULT_VECTORS_PATH)
    p.add_argument("--ki-weight", type=float, default=0.25)
    p.add_argument("--embedder", choices=["auto", "minilm", "hash"], default="auto")
    args = p.parse_args(argv)

    emb = {
        "hash": HashingEmbedder,
        "minilm": SentenceTransformerEmbedder,
        "auto": default_embedder,
    }[args.embedder]()

    m = load_map()
    print(
        f"map: {len(m.nodes)} nodes, hash {_map_hash(DEFAULT_MAP_PATH)}; "
        f"embedder {getattr(emb, 'name', '?')}"
    )
    nv = build_node_vectors(emb, m, ki_weight=args.ki_weight)
    out = save_node_vectors(nv, args.out)
    print(f"wrote {out}  (matrix {nv.matrix.shape}, model {nv.model_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
