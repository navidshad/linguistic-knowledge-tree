"""SemanticMapper: BERT vs K-BERT text->node mapping (Phase 6 / RQ2, Step 2).

Deterministic via ``HashingEmbedder``; assertions are structural (shape, ordering,
threshold/top-k, the injection toggle), not about embedding quality.
"""
from pathlib import Path

import numpy as np

from klg_ai.adapters.slam import iter_exercises
from klg_ai.loader import load_map
from klg_ai.semantic import HashingEmbedder, SemanticMapper, build_node_vectors, node_text
from klg_ai.semantic.node_vectors import _category_labels

FIX = Path(__file__).parent / "fixtures" / "slam"
TRAIN = FIX / "mini.slam.20190204.train"

_EMB = HashingEmbedder(dim=64)
_MAP = load_map()
_NV = build_node_vectors(_EMB, _MAP)


def _mapper(**kw) -> SemanticMapper:
    return SemanticMapper(_EMB, _NV, **kw)


def test_empty_text_maps_to_nothing():
    assert _mapper().map_text("   ").node_ids == ()


def test_topk_and_threshold_respected():
    m = _mapper(threshold=0.0, top_k=2)
    match = m.map_text("she is reading a book")
    assert len(match.node_ids) <= 2
    assert set(match.node_ids) <= _MAP.node_ids
    assert set(match.node_ids) == set(match.scores)
    # high threshold filters everything out
    assert _mapper(threshold=2.0).map_text("she is reading a book").node_ids == ()


def test_scores_are_descending_cosine():
    match = _mapper(threshold=0.0, top_k=5).map_text("the books are bigger")
    vals = [match.scores[n] for n in match.node_ids]
    assert vals == sorted(vals, reverse=True)


def test_determinism():
    a = _mapper(threshold=0.0).map_text("present perfect")
    b = _mapper(threshold=0.0).map_text("present perfect")
    assert a.node_ids == b.node_ids and a.scores == b.scores


def test_knowledge_injection_changes_scores():
    # Pick a node whose K-BERT (graph-injected) vector actually differs.
    inj = next(
        i for i in range(len(_NV.node_ids)) if not np.allclose(_NV.matrix[i], _NV.matrix_kbert[i])
    )
    node = _MAP.node(_NV.node_ids[inj])
    text = node_text(node, _category_labels(_MAP))
    bert = _mapper(threshold=0.0, top_k=3).map_text(text)
    kbert = _mapper(threshold=0.0, top_k=3, knowledge_injection=True).map_text(text)
    assert bert.scores != kbert.scores


def test_map_exercise_shape_compatible_with_rule_mapper():
    ex = next(iter_exercises(TRAIN))
    out = _mapper(threshold=0.0, top_k=2).map_exercise(ex)
    assert isinstance(out, dict)
    valid_ids = {t.instance_id for t in ex.tokens}
    for iid, match in out.items():
        assert iid in valid_ids
        assert set(match.node_ids) <= _MAP.node_ids
        assert match.node_ids  # omitted entirely if it matched nothing


def test_map_token_in_context_marks_span():
    # Structural: a token embedded in its sentence yields a (possibly empty) match.
    match = _mapper(threshold=0.0, top_k=1).map_token_in_context("reading", "she is reading")
    assert isinstance(match.node_ids, tuple)
