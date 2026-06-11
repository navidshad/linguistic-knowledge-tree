"""CLI: run the Phase 5 SLAM validation and write the results JSON.

    python -m klg_ai.eval.run --data data/slam --course en_es --split dev \
        --max-learners 400 --out data/eval/results.json

Loads the SLAM track, runs the RQ ablations (engine variants, DKT, baselines),
prints a leaderboard, and writes the results JSON the dashboard/API serve.
``--max-learners`` bounds the cost (seeded learner sample); pass 0 for all
learners (a full en_es run is large). The cap is recorded in the output.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from klg_ai.eval.ablations import run_ablations
from klg_ai.data.sources import DATA_SOURCES, load_source

_REPO = Path(__file__).resolve().parents[4]


def _default_data_dir() -> Path:
    for c in (_REPO / "data" / "slam", _REPO / "data" / "slam" / "raw"):
        if (c / "en_es.slam.20190204.train").exists():
            return c
    return _REPO / "data" / "slam"


def _resolve_out(out: Path | None, *, train_prop: bool, kgt: bool, mapper: str) -> Path:
    """Default the results path per variant so a variant run never overwrites the
    Phase-5 ``results.json``. ``--train-prop`` wins (it may be combined with
    ``--kgt`` to produce one superset file)."""
    if out:
        return out
    base = _REPO / "data" / "eval"
    if train_prop:
        return base / "results_trainprop.json"
    if kgt:
        return base / "results_kgt.json"
    return base / ("results.json" if mapper == "rule" else f"results_{mapper}.json")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SLAM validation + RQ ablations (RQ1/3/4/5)")
    p.add_argument("--data-source", default="slam", choices=sorted(DATA_SOURCES),
                   help="open dataset to load (registry in klg_ai.data.sources; slam only today)")
    p.add_argument("--data", type=Path, default=_default_data_dir(),
                   help="directory with <course>.slam.20190204.{train,dev,test,*.key}")
    p.add_argument("--course", default="en_es")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--max-learners", type=int, default=400,
                   help="cap on learners (seeded sample); 0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dkt-epochs", type=int, default=10)
    p.add_argument("--mapper", default="rule", choices=["rule", "bert", "kbert"],
                   help="evidence->node mapping: rule (Phase 5) | bert | kbert (Phase 6 RQ2)")
    p.add_argument("--kgt", action="store_true",
                   help="add the RQ5 personalization arms (engine_kgt + per-learner retrain) "
                        "and measure per-model compute cost (Phase 7)")
    p.add_argument("--retrain-epochs", type=int, default=30,
                   help="Adam epochs for the per-learner retrain arm (with --kgt)")
    p.add_argument("--train-prop", action="store_true",
                   help="add the RQ3 trained-propagation arms (globally fit edge weights, "
                        "per-edge + 2-scalar) and measure compute cost")
    p.add_argument("--prop-epochs", type=int, default=40,
                   help="Adam epochs for the global propagation fit (with --train-prop)")
    p.add_argument("--out", type=Path, default=None,
                   help="results JSON (default: data/eval/results.json; results_<mapper>.json "
                        "for bert/kbert; results_kgt.json with --kgt; results_trainprop.json with "
                        "--train-prop — a variant run never clobbers the Phase-5 rule results)")
    args = p.parse_args(argv)

    out = _resolve_out(args.out, train_prop=args.train_prop, kgt=args.kgt, mapper=args.mapper)

    max_learners = None if args.max_learners == 0 else args.max_learners
    print(f"Loading {args.data_source}:{args.course}/{args.split} from {args.data} "
          f"(max_learners={max_learners}, seed={args.seed}, mapper={args.mapper}) ...")
    learners = load_source(args.data_source, args.data, course=args.course, split=args.split,
                           max_learners=max_learners, seed=args.seed, mapper=args.mapper)
    n_eval = sum(len(ld.evalset) for ld in learners)
    print(f"  {len(learners)} learners, {n_eval} eval instances. Running models ...")

    results = run_ablations(learners, course=args.course, split=args.split,
                            dkt_epochs=args.dkt_epochs, seed=args.seed, mapper=args.mapper,
                            kgt=args.kgt, retrain_epochs=args.retrain_epochs,
                            train_prop=args.train_prop, prop_epochs=args.prop_epochs)
    results["meta"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": args.data_source,
        "max_learners": max_learners,
        "seed": args.seed,
        "dkt_epochs": args.dkt_epochs,
        "mapper": args.mapper,
    }
    if args.kgt or args.train_prop:
        import torch
        results["meta"]["torch_threads"] = torch.get_num_threads()
    if args.kgt:
        results["meta"]["kgt"] = True
        results["meta"]["retrain_epochs"] = args.retrain_epochs
    if args.train_prop:
        results["meta"]["train_prop"] = True
        results["meta"]["prop_epochs"] = args.prop_epochs

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    d = results["dataset"]
    print(f"\n{d['source']} — {d['course']}/{d['split']}: "
          f"{d['n_learners']} learners, {d['n_eval_instances']} instances "
          f"({d['n_cold_instances']} cold), mistake rate {d['mistake_base_rate']:.3f}, "
          f"coverage {d['node_coverage']:.3f}\n")
    cost_col = "  ms/learner" if (args.kgt or args.train_prop) else ""
    print(f"  {'model':<32}{'AUROC':>8}{'F1':>8}{'acc':>8}{'logloss':>9}{'AUROC_cold':>12}{cost_col}")
    print("  " + "-" * (76 + len(cost_col)))
    for m in results["models"]:
        mt = m["metrics"]
        cold = m.get("metrics_cold")
        cold_auroc = f"{cold['auroc']:>12.3f}" if cold else f"{'-':>12}"
        cost = m.get("cost")
        cost_s = f"{cost['seconds_per_learner'] * 1000:>12.1f}" if cost else ""
        print(f"  {m['label']:<32}{mt['auroc']:>8.3f}{mt['F1']:>8.3f}"
              f"{mt['accuracy']:>8.3f}{mt['avglogloss']:>9.3f}{cold_auroc}{cost_s}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
