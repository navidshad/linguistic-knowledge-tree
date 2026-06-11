"""The RQ ablation runs: engine variants vs. DKT vs. simple baselines.

Assembles the comparison table the thesis reports. Every model is scored on the
*same* eval instances with the same metrics, so the differences are attributable:

* **RQ1 (graph vs. sequence)** — engine (full) vs. DKT, both over the same
  node-tagged interactions; per-skill/per-user/global means anchor the scale.
* **RQ3 (propagation)** — engine with GNN propagation on vs. off.
* **RQ4 (forgetting)** — engine with recency decay on vs. off.
* **RQ5 (personalization, ``kgt=True`` runs only)** — closed-form KGT edge
  tuning vs. per-learner gradient retraining over the same hypothesis space;
  each model also gets a measured compute ``cost``, since RQ5 is a cost
  question as much as a fit question. The default ``kgt=False`` output is
  key-for-key identical to Phase 5.

Returns a JSON-serializable dict (dataset summary + per-model metrics + ROC
points + RQ groupings) consumed by the CLI, the API, and the viewer dashboard.
"""
from __future__ import annotations

import time

from klg_ai.core.activation import EngineConfig
from klg_ai.eval.baselines import DKTPredictor
from klg_ai.data.dataset import LearnerData, all_eval_instances, cold_eval_mask
from klg_ai.eval.metrics import evaluate, roc_curve
from klg_ai.eval.predict import (
    EnginePredictor,
    GlobalMeanPredictor,
    PerSkillMeanPredictor,
    PerUserMeanPredictor,
)


def _models(dkt_epochs: int, seed: int, *, kgt: bool = False, retrain_epochs: int = 30,
            train_prop: bool = False, prop_epochs: int = 40):
    """(predictor, rq_group, display_label) for every model in the comparison."""
    models = [
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
    if kgt:
        from klg_ai.eval.retrain_predictor import PerLearnerRetrainPredictor
        models += [
            (EnginePredictor(EngineConfig(kgt=True), "engine_kgt"),
             "RQ5", "Engine + KGT (personal edges)"),
            (PerLearnerRetrainPredictor(epochs=retrain_epochs, seed=seed),
             "RQ5", "Engine + per-learner retrain"),
        ]
    if train_prop:
        from klg_ai.eval.train_prop_predictor import TrainedEnginePredictor
        models += [
            (TrainedEnginePredictor(EngineConfig(), granularity="edge", epochs=prop_epochs, seed=seed),
             "RQ3", "Engine + trained propagation (per-edge)"),
            (TrainedEnginePredictor(EngineConfig(), granularity="scalar", epochs=prop_epochs, seed=seed),
             "RQ3", "Engine + trained propagation (2-scalar)"),
        ]
    return models


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
    kgt: bool = False,
    retrain_epochs: int = 30,
    train_prop: bool = False,
    prop_epochs: int = 40,
) -> dict:
    """Run every model and return the full results dict.

    ``kgt=True`` (the Phase-7 RQ5 run) adds the two personalization arms;
    ``train_prop=True`` (the RQ3 trained-propagation run) adds the globally
    trained-weight arms. Either adds a measured per-model compute ``cost``; the
    default output (both off) is key-for-key identical to Phase 5.
    """
    measure_cost = kgt or train_prop
    instances = all_eval_instances(learners)
    actual = [float(it.label) for it in instances]  # 1 = mistake
    cold = cold_eval_mask(learners)
    cold_idx = [i for i, c in enumerate(cold) if c]
    cold_actual = [actual[i] for i in cold_idx]

    if measure_cost:
        # Warm up the lazy torch/PyG import + layer build before timing, so the
        # first engine arm's cost isn't inflated by one-time setup.
        from klg_ai.core.graph import default_graph
        from klg_ai.core.propagation import propagate
        propagate(default_graph(), {}, EngineConfig())

    models: list[dict] = []
    for predictor, group, label in _models(dkt_epochs, seed, kgt=kgt, retrain_epochs=retrain_epochs,
                                           train_prop=train_prop, prop_epochs=prop_epochs):
        t0 = time.perf_counter()
        predicted = predictor.predict(learners)
        dt = time.perf_counter() - t0
        if len(predicted) != len(actual):
            raise RuntimeError(
                f"{predictor.name}: {len(predicted)} preds vs {len(actual)} instances")
        # Cold-node slice: eval items on nodes the learner never practiced (RQ3).
        cold_metrics = evaluate(cold_actual, [predicted[i] for i in cold_idx]) if cold_idx else None
        entry = {
            "name": predictor.name,
            "group": group,
            "label": label,
            "metrics": evaluate(actual, predicted),
            "metrics_cold": cold_metrics,
            "roc": roc_curve(actual, predicted),
        }
        if measure_cost:
            # Cost is a research variable here (RQ5 KGT-vs-retrain; trained-vs-fixed
            # one-time fit): fit+predict wall-clock, same machine/process for every
            # arm so the ratio is meaningful.
            entry["cost"] = {
                "fit_predict_seconds": round(dt, 3),
                "seconds_per_learner": round(dt / max(1, len(learners)), 5),
            }
            curve = getattr(predictor, "curve_", None)
            if curve:
                entry["retrain_curve"] = curve
        models.append(entry)

    models.sort(key=lambda m: m["metrics"]["auroc"], reverse=True)
    rqs = {
        "RQ1_graph_vs_sequence": ["engine_full", "dkt", "per_skill_mean"],
        "RQ3_propagation": ["engine_full", "engine_no_prop"],
        "RQ3_propagation_cold": ["engine_full", "engine_no_prop"],
        "RQ4_forgetting": ["engine_full", "engine_no_forget"],
    }
    if kgt:
        rqs["RQ5_personalization"] = ["engine_full", "engine_kgt", "engine_retrain"]
    if train_prop:
        rqs["RQ3_trained_propagation"] = ["engine_full", "engine_no_prop", "engine_trained"]
        rqs["RQ3_trained_propagation_cold"] = ["engine_full", "engine_no_prop", "engine_trained"]
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
        "rqs": rqs,
        "models": models,
    }
