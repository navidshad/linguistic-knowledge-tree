"""Intrinsic RQ2: how well does the embedding mapper recover the rule-based labels?

The Phase-5 rule mapper (``adapters.slam_mapping.map_exercise``) is the **silver
standard** — a hand-built, auditable tag->node function. Here we treat its output
as gold and ask how close an *unsupervised* semantic mapper (BERT vs K-BERT) gets
when it only sees the token's text in context (no morphosyntactic tags), and
whether 1-hop prerequisite-graph injection closes the gap.

Granularity is **token-in-context per ``instance_id``** — the gold is token-keyed,
so this is the native comparison. Each (token, node) is a binary decision; we
report micro and macro (per-node) precision / recall / F1, node coverage, the
exact-set match rate, and the top node confusions. The ``rule`` variant scored
against itself is the perfect reference row.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from klg_ai.data.adapters.slam_mapping import map_exercise


class RuleMapper:
    """Adapts the rule-based ``map_exercise`` to the mapper interface (the gold row)."""

    name = "rule"

    def map_exercise(self, ex):  # noqa: D401 - thin adapter
        return map_exercise(ex)


def _sets(mapping) -> dict[str, set[str]]:
    """Normalize ``{iid: tuple}`` (rule) or ``{iid: SemanticMatch}`` -> ``{iid: set}``."""
    out: dict[str, set[str]] = {}
    for iid, val in mapping.items():
        out[iid] = set(getattr(val, "node_ids", val))
    return out


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f}


def _score_one(exercises, gold_by_iid: dict[str, set[str]], mapper, *, max_confusions: int) -> dict:
    tp = fp = fn = 0
    per_node: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # [tp, fp, fn]
    confusions: Counter = Counter()
    n_tokens = gold_cov = pred_cov = gold_nonempty = exact = 0

    for ex in exercises:
        pred = _sets(mapper.map_exercise(ex))
        for tok in ex.tokens:
            g = gold_by_iid.get(tok.instance_id, set())
            p = pred.get(tok.instance_id, set())
            n_tokens += 1
            gold_cov += bool(g)
            pred_cov += bool(p)
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
            for n in g & p:
                per_node[n][0] += 1
            for n in p - g:
                per_node[n][1] += 1
            for n in g - p:
                per_node[n][2] += 1
            for gnode in g - p:
                for pnode in p - g:
                    confusions[(gnode, pnode)] += 1
            if g:
                gold_nonempty += 1
                exact += g == p

    macro = [_prf(t, f, n) for t, f, n in per_node.values() if (t + n) > 0]  # support = tp+fn
    nmac = len(macro) or 1
    return {
        "name": getattr(mapper, "name", "?"),
        "micro": _prf(tp, fp, fn),
        "macro": {k: sum(m[k] for m in macro) / nmac for k in ("precision", "recall", "f1")},
        "coverage": pred_cov / n_tokens if n_tokens else 0.0,
        "gold_coverage": gold_cov / n_tokens if n_tokens else 0.0,
        "exact_set_rate": exact / gold_nonempty if gold_nonempty else 0.0,
        "n_tokens": n_tokens,
        "counts": {"tp": tp, "fp": fp, "fn": fn},
        "per_node": {
            n: {**_prf(c[0], c[1], c[2]), "support": c[0] + c[2], "tp": c[0], "fp": c[1], "fn": c[2]}
            for n, c in sorted(per_node.items())
        },
        "top_confusions": [
            {"gold": g, "pred": p, "count": c} for (g, p), c in confusions.most_common(max_confusions)
        ],
    }


def compare_mappers(exercises, mappers: dict[str, object], *, max_confusions: int = 15) -> dict:
    """Score each named mapper against the rule-based silver labels.

    ``mappers`` maps a display name -> an object with ``map_exercise(ex)``. Include
    a ``RuleMapper`` to get the perfect reference row. Returns a JSON-serializable
    dict: ``{n_exercises, reports: {name: {...}}}``.
    """
    exercises = list(exercises)
    gold_by_iid: dict[str, set[str]] = {}
    for ex in exercises:
        for iid, nodes in map_exercise(ex).items():
            gold_by_iid[iid] = set(nodes)
    return {
        "n_exercises": len(exercises),
        "reports": {
            name: _score_one(exercises, gold_by_iid, mapper, max_confusions=max_confusions)
            for name, mapper in mappers.items()
        },
    }
