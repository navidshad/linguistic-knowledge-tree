"""The RQ ablation runs: engine variants vs. DKT vs. simple baselines.

Assembles the comparison table the thesis reports. Every model is scored on the
*same* eval instances with the same metrics, so the differences are attributable:

* **RQ1 (graph vs. sequence)** — engine (full) vs. DKT, both over the same
  node-tagged interactions; per-skill/per-user/global means anchor the scale.
* **RQ3 (propagation)** — engine with GNN propagation on vs. off.
* **RQ4 (forgetting)** — engine with recency decay on vs. off.

Returns a JSON-serializable dict (dataset summary + per-model metrics + ROC
points + RQ groupings) consumed by the CLI, the API, and the viewer dashboard.
"""
from __future__ import annotations

from ..activation import EngineConfig
from .baselines import DKTPredictor
from .dataset import LearnerData, all_eval_instances, cold_eval_mask
from .metrics import evaluate, roc_curve
from .predict import (
    EnginePredictor,
    GlobalMeanPredictor,
    PerSkillMeanPredictor,
    PerUserMeanPredictor,
)


def _models(dkt_epochs: int, seed: int):
    """(predictor, rq_group, display_label) for every model in the comparison."""
    return [
        (EnginePredictor(EngineConfig(), "engine_full"),
         "engine", "Engine (graph + forgetting)"),
        (EnginePredictor(EngineConfig(propagation=False), "engine_no_prop"),
         "RQ3", "Engine − propagation"),
        (EnginePredictor(EngineConfig(forgetting=False), "engine_no_forget"),
         "RQ4", "Engine − forgetting"),
        (EnginePredictor(EngineConfig(propagation=False, forgetting=False), "engine_neither"),
         "ablation", "Engine − both"),
        (DKTPredictor(epochs=dkt_epochs, seed=seed),
         "RQ1", "DKT (sequence)"),
        (PerSkillMeanPredictor(), "baseline", "Per-skill mean (difficulty)"),
        (PerUserMeanPredictor(), "baseline", "Per-user mean (ability)"),
        (GlobalMeanPredictor(), "baseline", "Global mean"),
    ]


def _coverage(learners: list[LearnerData]) -> float:
    mapped = total = 0
    for it in all_eval_instances(learners):
        total += 1
        mapped += 1 if it.node_ids else 0
    return mapped / total if total else 0.0


def run_ablations(
    learners: list[LearnerData],
    *,
    course: str = "en_es",
    split: str = "dev",
    dkt_epochs: int = 10,
    seed: int = 0,
    mapper: str = "rule",
) -> dict:
    """Run every model and return the full results dict."""
    instances = all_eval_instances(learners)
    actual = [float(it.label) for it in instances]  # 1 = mistake
    cold = cold_eval_mask(learners)
    cold_idx = [i for i, c in enumerate(cold) if c]
    cold_actual = [actual[i] for i in cold_idx]

    models: list[dict] = []
    for predictor, group, label in _models(dkt_epochs, seed):
        predicted = predictor.predict(learners)
        if len(predicted) != len(actual):
            raise RuntimeError(
                f"{predictor.name}: {len(predicted)} preds vs {len(actual)} instances")
        # Cold-node slice: eval items on nodes the learner never practiced (RQ3).
        cold_metrics = evaluate(cold_actual, [predicted[i] for i in cold_idx]) if cold_idx else None
        models.append({
            "name": predictor.name,
            "group": group,
            "label": label,
            "metrics": evaluate(actual, predicted),
            "metrics_cold": cold_metrics,
            "roc": roc_curve(actual, predicted),
        })

    models.sort(key=lambda m: m["metrics"]["auroc"], reverse=True)
    return {
        "dataset": {
            "source": "Duolingo SLAM 2018",
            "course": course,
            "split": split,
            "mapper": mapper,
            "n_learners": len(learners),
            "n_eval_instances": len(instances),
            "n_cold_instances": len(cold_idx),
            "mistake_base_rate": (sum(actual) / len(actual)) if actual else 0.0,
            "node_coverage": _coverage(learners),
        },
        "rqs": {
            "RQ1_graph_vs_sequence": ["engine_full", "dkt", "per_skill_mean"],
            "RQ3_propagation": ["engine_full", "engine_no_prop"],
            "RQ3_propagation_cold": ["engine_full", "engine_no_prop"],
            "RQ4_forgetting": ["engine_full", "engine_no_forget"],
        },
        "models": models,
    }
