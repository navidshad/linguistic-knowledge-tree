"""Phase 5 eval harness: metrics, dataset split, predictors, ablations.

Metrics are pinned to Duolingo SLAM eval.py's oracle; everything else runs on the
in-repo mini fixture (no gated dataset). Engine/DKT tests skip without torch.
"""
from pathlib import Path

import pytest

from klg_ai.eval import metrics as M
from klg_ai.eval.dataset import all_eval_instances, cold_eval_mask, load_track

FIX = Path(__file__).parent / "fixtures" / "slam"


def _learners():
    return load_track(FIX, course="mini", split="dev", max_learners=None)


# --- metrics (no torch) ----------------------------------------------------

def test_metrics_match_slam_oracle():
    m = M.evaluate([1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
                   [0.8, 0.2, 0.6, 0.3, 0.1, 0.2, 0.3, 0.9, 0.2, 0.7])
    assert round(m["accuracy"], 3) == 0.700
    assert round(m["avglogloss"], 3) == 0.613
    assert round(m["auroc"], 3) == 0.740
    assert round(m["F1"], 3) == 0.667


def test_auroc_single_class_is_half():
    assert M.compute_auroc([1, 1, 1], [0.2, 0.7, 0.5]) == 0.5


def test_roc_curve_endpoints_and_monotone():
    pts = M.roc_curve([1, 0, 1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2, 0.6, 0.4])
    assert pts[0] == {"fpr": 0.0, "tpr": 0.0}
    assert pts[-1] == {"fpr": 1.0, "tpr": 1.0}
    fprs = [p["fpr"] for p in pts]
    tprs = [p["tpr"] for p in pts]
    assert fprs == sorted(fprs) and tprs == sorted(tprs)


# --- dataset (no torch) ----------------------------------------------------

def test_load_track_basic():
    learners = _learners()
    assert {ld.user for ld in learners} == {"learnerA", "learnerB"}
    assert all(ld.train and ld.evalset for ld in learners)


def test_split_is_causal_eval_after_train():
    for ld in _learners():
        assert ld.eval_ref_day > max(it.day for it in ld.train)


def test_train_events_only_mapped_with_correctness():
    a = next(ld for ld in _learners() if ld.user == "learnerA")
    events = a.train_events()
    assert events and all(e.node_ids for e in events)
    pc = [e for e in events if "present_continuous" in e.node_ids]
    assert any(e.correct is False for e in pc)  # "reading" was labeled a mistake


def test_min_train_filter_drops_learners():
    assert load_track(FIX, course="mini", split="dev", min_train=1000) == []


def test_cold_mask_aligned_and_flags_unpracticed_nodes():
    learners = _learners()
    mask = cold_eval_mask(learners)
    assert len(mask) == len(all_eval_instances(learners))
    assert sum(mask) == 2  # learnerB's "Where"/"are" hit nodes never seen in train


# --- predictors + ablations (need torch / torch_geometric) -----------------

def test_predictors_emit_aligned_probabilities():
    pytest.importorskip("torch_geometric")
    from klg_ai.activation import EngineConfig
    from klg_ai.eval.baselines import DKTPredictor
    from klg_ai.eval.predict import (
        EnginePredictor, GlobalMeanPredictor, PerSkillMeanPredictor,
        PerUserMeanPredictor,
    )

    learners = _learners()
    n = len(all_eval_instances(learners))
    predictors = [
        EnginePredictor(EngineConfig()),
        PerSkillMeanPredictor(),
        PerUserMeanPredictor(),
        GlobalMeanPredictor(),
        DKTPredictor(epochs=2),
    ]
    for p in predictors:
        out = p.predict(learners)
        assert len(out) == n, p.name
        assert all(0.0 <= v <= 1.0 for v in out), p.name


def test_engine_propagation_changes_cold_predictions():
    pytest.importorskip("torch_geometric")
    from klg_ai.activation import EngineConfig
    from klg_ai.eval.predict import EnginePredictor

    learners = _learners()
    cold = cold_eval_mask(learners)
    with_prop = EnginePredictor(EngineConfig(propagation=True)).predict(learners)
    no_prop = EnginePredictor(EngineConfig(propagation=False)).predict(learners)
    # On at least one cold node, the graph must move the prediction.
    assert any(abs(with_prop[i] - no_prop[i]) > 1e-6 for i, c in enumerate(cold) if c)


def test_run_ablations_structure():
    pytest.importorskip("torch_geometric")
    from klg_ai.eval.ablations import run_ablations

    res = run_ablations(_learners(), dkt_epochs=2)
    assert {"dataset", "rqs", "models"} <= res.keys()
    names = {m["name"] for m in res["models"]}
    assert {"engine_full", "engine_no_prop", "engine_no_forget",
            "dkt", "per_skill_mean", "global_mean"} <= names
    for m in res["models"]:
        assert 0.0 <= m["metrics"]["auroc"] <= 1.0
        assert m["metrics"]["n"] == res["dataset"]["n_eval_instances"]
    for group in res["rqs"].values():
        assert all(any(mm["name"] == nm for mm in res["models"]) for nm in group)
