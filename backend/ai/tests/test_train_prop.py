"""RQ3 trained propagation: the global edge-weight trainer + harness wiring.

Runs on the in-repo mini SLAM fixture (no gated dataset). The trainer needs
torch; the predictor/ablations need torch_geometric (the propagation forward).
"""
from pathlib import Path

import pytest

from klg_ai.data.dataset import all_eval_instances, load_track

FIX = Path(__file__).parent / "fixtures" / "slam"


def _learners():
    return load_track(FIX, course="mini", split="dev", max_learners=None)


# --- no torch: path discipline + data-source registry ----------------------

def test_resolve_out_never_clobbers_results_json():
    from klg_ai.eval.run import _resolve_out
    assert _resolve_out(None, train_prop=True, kgt=False, mapper="rule").name == "results_trainprop.json"
    assert _resolve_out(None, train_prop=True, kgt=True, mapper="rule").name == "results_trainprop.json"
    assert _resolve_out(None, train_prop=False, kgt=True, mapper="rule").name == "results_kgt.json"
    assert _resolve_out(None, train_prop=False, kgt=False, mapper="rule").name == "results.json"


def test_data_source_registry():
    from klg_ai.data.sources import DATA_SOURCES, load_source
    assert "slam" in DATA_SOURCES
    learners = load_source("slam", FIX, course="mini", split="dev")
    assert {ld.user for ld in learners} == {"learnerA", "learnerB"}
    with pytest.raises(ValueError):
        load_source("nope", FIX)


# --- global trainer (needs torch) ------------------------------------------

def test_fit_global_descends_shapes_and_deterministic():
    pytest.importorskip("torch")
    from klg_ai.core.activation import EngineConfig
    from klg_ai.core.graph import default_graph
    from klg_ai.tuning.train_prop import fit_global_edge_factors

    g = default_graph()
    learners = _learners()
    t = fit_global_edge_factors(g, learners, EngineConfig(), epochs=20, seed=0)
    assert len(t.losses) == 20
    assert t.losses[-1] <= t.losses[0] + 1e-9            # full-batch fit is non-increasing
    for mb, mf in t.factors.values():
        assert 0.0 < mb < 2.0 and 0.0 < mf < 2.0         # m = 2*sigmoid(theta) bounds
    t2 = fit_global_edge_factors(g, learners, EngineConfig(), epochs=20, seed=0)
    assert t.losses == t2.losses                          # zero-init + fixed epochs -> deterministic
    wj = t.as_weights_json(g)
    assert wj["granularity"] == "edge" and wj["edges"] and "alpha_back" in next(iter(wj["edges"].values()))


def test_fit_global_scalar_shares_one_multiplier():
    pytest.importorskip("torch")
    from klg_ai.core.activation import EngineConfig
    from klg_ai.core.graph import default_graph
    from klg_ai.tuning.train_prop import fit_global_edge_factors

    t = fit_global_edge_factors(default_graph(), _learners(), EngineConfig(),
                                granularity="scalar", epochs=10, seed=0)
    assert len(set(t.factors.values())) <= 1              # every edge shares the one (back, fwd) pair


def test_fit_global_epochs_zero_is_empty():
    pytest.importorskip("torch")
    from klg_ai.core.activation import EngineConfig
    from klg_ai.core.graph import default_graph
    from klg_ai.tuning.train_prop import fit_global_edge_factors

    t = fit_global_edge_factors(default_graph(), _learners(), EngineConfig(), epochs=0)
    assert t.factors == {} and t.losses == []


def test_fit_global_rejects_bad_granularity():
    pytest.importorskip("torch")
    from klg_ai.core.graph import default_graph
    from klg_ai.tuning.train_prop import fit_global_edge_factors

    with pytest.raises(ValueError):
        fit_global_edge_factors(default_graph(), _learners(), granularity="nope")


# --- trained predictor (needs torch_geometric) -----------------------------

def test_trained_predictor_aligned_bounded_and_seeded():
    pytest.importorskip("torch_geometric")
    from klg_ai.eval.train_prop_predictor import TrainedEnginePredictor

    learners = _learners()
    n = len(all_eval_instances(learners))
    a = TrainedEnginePredictor(epochs=10, seed=0).predict(learners)
    b = TrainedEnginePredictor(epochs=10, seed=0).predict(learners)
    assert len(a) == n and all(0.0 <= x <= 1.0 for x in a)
    assert a == b


def test_trained_predictor_epochs0_equals_fixed_engine():
    # With no training the trained arm must reduce *exactly* to the fixed engine,
    # so any AUROC difference downstream is attributable to the learned weights.
    pytest.importorskip("torch_geometric")
    from klg_ai.core.activation import EngineConfig
    from klg_ai.eval.predict import EnginePredictor
    from klg_ai.eval.train_prop_predictor import TrainedEnginePredictor

    learners = _learners()
    trained0 = TrainedEnginePredictor(EngineConfig(), epochs=0).predict(learners)
    fixed = EnginePredictor(EngineConfig()).predict(learners)
    assert [round(x, 9) for x in trained0] == [round(x, 9) for x in fixed]


# --- ablations wiring ------------------------------------------------------

def test_run_ablations_train_prop_adds_rq3_arms_and_cost():
    pytest.importorskip("torch_geometric")
    from klg_ai.eval.ablations import run_ablations

    res = run_ablations(_learners(), dkt_epochs=2, train_prop=True, prop_epochs=8)
    names = {m["name"] for m in res["models"]}
    assert {"engine_trained", "engine_trained_scalar"} <= names
    assert res["rqs"]["RQ3_trained_propagation"] == ["engine_full", "engine_no_prop", "engine_trained"]
    et = next(m for m in res["models"] if m["name"] == "engine_trained")
    assert et["cost"]["seconds_per_learner"] >= 0.0
    assert [p["epoch"] for p in et["retrain_curve"]] == list(range(1, 9))


def test_run_ablations_default_has_no_trained_arms():
    pytest.importorskip("torch_geometric")
    from klg_ai.eval.ablations import run_ablations

    res = run_ablations(_learners(), dkt_epochs=2)
    names = {m["name"] for m in res["models"]}
    assert "engine_trained" not in names
    assert "RQ3_trained_propagation" not in res["rqs"]
