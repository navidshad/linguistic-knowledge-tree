"""Intrinsic RQ2 eval: mapper-vs-silver-labels metrics (Phase 6 / RQ2, Step 3).

Deterministic via ``HashingEmbedder`` on the in-repo SLAM fixture.
"""
from pathlib import Path

from klg_ai.data.adapters.slam import iter_exercises
from klg_ai.eval.mapping_eval import RuleMapper, compare_mappers
from klg_ai.core.loader import load_map
from klg_ai.semantic import HashingEmbedder, SemanticMapper, build_node_vectors

FIX = Path(__file__).parent / "fixtures" / "slam"
TRAIN = FIX / "mini.slam.20190204.train"


def _mappers():
    emb = HashingEmbedder(dim=64)
    nv = build_node_vectors(emb, load_map())
    return {
        "rule": RuleMapper(),
        "bert": SemanticMapper(emb, nv, threshold=0.0, top_k=2),
        "kbert": SemanticMapper(emb, nv, threshold=0.0, top_k=2, knowledge_injection=True),
    }


def test_rule_recovers_itself_perfectly():
    res = compare_mappers(list(iter_exercises(TRAIN)), _mappers())
    rule = res["reports"]["rule"]
    assert rule["micro"]["f1"] == 1.0
    assert rule["exact_set_rate"] == 1.0
    assert rule["coverage"] == rule["gold_coverage"]
    assert rule["counts"]["fp"] == 0 and rule["counts"]["fn"] == 0


def test_semantic_metrics_in_range():
    exs = list(iter_exercises(TRAIN))
    res = compare_mappers(exs, _mappers())
    assert res["n_exercises"] == len(exs)
    for name in ("bert", "kbert"):
        rep = res["reports"][name]
        for k in ("precision", "recall", "f1"):
            assert 0.0 <= rep["micro"][k] <= 1.0
            assert 0.0 <= rep["macro"][k] <= 1.0
        assert 0.0 <= rep["coverage"] <= 1.0
        assert 0.0 <= rep["exact_set_rate"] <= 1.0


def test_report_structure_and_confusions():
    rep = compare_mappers(list(iter_exercises(TRAIN)), _mappers())["reports"]["bert"]
    assert set(rep) >= {
        "micro", "macro", "coverage", "gold_coverage", "exact_set_rate",
        "per_node", "top_confusions", "counts",
    }
    for c in rep["top_confusions"]:
        assert set(c) == {"gold", "pred", "count"}
