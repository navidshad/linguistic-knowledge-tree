"""CLI: intrinsic RQ2 — embedding mapper vs rule-based silver labels on SLAM.

    python -m klg_ai.eval.mapping_run --data data/slam --course en_es --split dev \
        --max-learners 500 --out docs/phase6/mapping_results.json

Streams the sampled learners' exercises, runs the rule / BERT / K-BERT mappers,
and writes the precision/recall/F1 + confusion report the Phase 6 doc/dashboard
use. Set ``KLG_EMBEDDER=hash`` to run without sentence-transformers (lower quality).
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from klg_ai.data.adapters.slam import iter_exercises
from klg_ai.semantic.mapper import default_mapper
from klg_ai.data.dataset import _STAMP, _eval_users
from klg_ai.eval.mapping_eval import RuleMapper, compare_mappers

_REPO = Path(__file__).resolve().parents[4]


def _default_data_dir() -> Path:
    for c in (_REPO / "data" / "slam", _REPO / "data" / "slam" / "raw"):
        if (c / f"en_es.{_STAMP}.train").exists():
            return c
    return _REPO / "data" / "slam"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 6 RQ2 — semantic mapper vs rule silver labels")
    p.add_argument("--data", type=Path, default=_default_data_dir())
    p.add_argument("--course", default="en_es")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    p.add_argument("--max-learners", type=int, default=500, help="seeded learner sample; 0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--out", type=Path, default=_REPO / "data" / "eval" / "mapping_results.json")
    args = p.parse_args(argv)

    split_path = args.data / f"{args.course}.{_STAMP}.{args.split}"
    if not split_path.exists():
        raise FileNotFoundError(f"missing SLAM file: {split_path}")

    users: set[str] | None = None
    if args.max_learners and args.max_learners > 0:
        pool = _eval_users(split_path)
        users = set(random.Random(args.seed).sample(pool, min(args.max_learners, len(pool))))

    print(f"Loading {args.course}/{args.split} exercises from {split_path} "
          f"(max_learners={args.max_learners}) ...")
    exercises = [ex for ex in iter_exercises(split_path) if users is None or ex.user in users]
    print(f"  {len(exercises)} exercises. Building mappers "
          f"(threshold={args.threshold}, top_k={args.top_k}) ...")

    mappers = {
        "rule": RuleMapper(),
        "bert": default_mapper(knowledge_injection=False, threshold=args.threshold, top_k=args.top_k),
        "kbert": default_mapper(knowledge_injection=True, threshold=args.threshold, top_k=args.top_k),
    }
    result = compare_mappers(exercises, mappers)
    result["meta"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "course": args.course,
        "split": args.split,
        "max_learners": args.max_learners,
        "seed": args.seed,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "embedder": mappers["bert"].embedder.name,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n  {'mapper':<10}{'micro-P':>9}{'micro-R':>9}{'micro-F1':>10}{'coverage':>10}{'exact':>8}")
    print("  " + "-" * 56)
    for name, rep in result["reports"].items():
        mi = rep["micro"]
        print(f"  {name:<10}{mi['precision']:>9.3f}{mi['recall']:>9.3f}{mi['f1']:>10.3f}"
              f"{rep['coverage']:>10.3f}{rep['exact_set_rate']:>8.3f}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
