"""IELTS mock-test case study — real-learner external validation.

Each raw transcript (``data/ielts/raw/test_NN.md``) is a real, publicly-sourced
IELTS speaking test: an overall band ``Score``, the source video ``url``, the
``Candidate``/``Examiner`` dialogue, and the examiner's spoken ``Feedback``. The
band (scored by a human examiner) is the external ground-truth label; the
candidate's own utterances are the model input.

This module is the bridge: it parses each transcript, runs the candidate's
speech through the engine's free-text mapper (the same semantic mapper the chat
demo uses), folds the matched concepts as ``dialog`` evidence, and reports the
system's CEFR read so it can be compared against the IELTS band. Pure ``klg_ai``
(no web / no Gemini): grading is a server concern, so this is the system's
*as-is, ungraded* read — honest about the Phase-6 RQ2 recall limitation.

Run: ``python -m klg_ai.eval.ielts`` (writes per-test JSON under
``data/ielts/processed/`` + a pinned aggregate to ``docs/ielts-case-study/``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from klg_ai.core.activation import EngineConfig, mastery_from_events, threshold_activated
from klg_ai.core.events import Event
from klg_ai.core.loader import SyntaxMap, load_map
from klg_ai.core.status import compute_status

_REPO = Path(__file__).resolve().parents[4]
RAW_DIR = _REPO / "data" / "ielts" / "raw"
PROCESSED_DIR = _REPO / "data" / "ielts" / "processed"
PINNED = _REPO / "docs" / "ielts-case-study" / "results.json"

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_INDEX = {c: i for i, c in enumerate(CEFR_ORDER)}

# IELTS overall band → CEFR (Cambridge/IELTS official alignment, midpoint read).
# Used only to label the expected level next to the system read — not a model input.
def band_to_cefr(band: float) -> str:
    if band < 4.0:
        return "A2"
    if band < 5.0:
        return "A2/B1"
    if band < 6.5:
        return "B1"
    if band < 7.0:
        return "B2"
    if band < 8.0:
        return "B2/C1"
    return "C1/C2"


@dataclass
class Test:
    id: str
    band: float
    url: str
    candidate: list[str]
    feedback: str = ""


@dataclass
class TestResult:
    id: str
    band: float
    band_cefr: str
    url: str
    n_utterances: int
    proposals: dict[str, int] = field(default_factory=dict)   # node_id -> times matched
    cefr_profile: dict[str, int] = field(default_factory=dict)  # cefr -> proposal count
    system_cefr_index: float = 0.0   # proposal-weighted mean CEFR (0=A1 … 5=C2)
    ceiling_cefr: str | None = None  # highest CEFR with >= 2 proposals (robust ceiling)
    known: list[str] = field(default_factory=list)            # engine known-set (ungraded)
    known_cefr: dict[str, int] = field(default_factory=dict)


_BAND_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_test(path: Path) -> Test:
    """Parse one transcript. Tolerant of header casing (``score:``/``Score:``),
    key (``link:``/``url:``), and band forms (``9``, ``7.0``, ``Band 9``)."""
    text = path.read_text(encoding="utf-8")
    band = 0.0
    url = ""
    for line in text.splitlines():
        low = line.lower()
        if low.startswith("score") or low.startswith("detail"):
            m = _BAND_RE.search(line)
            if m:
                band = float(m.group(1))
        elif low.startswith("url") or low.startswith("link"):
            url = line.split(":", 1)[1].strip()
    # Candidate turns only (the model input); examiner prompts/feedback excluded.
    candidate = [c.strip() for c in re.findall(r"^Candidate:\s*(.+)$", text, flags=re.M)]
    # Strip parenthetical transcriber notes, keep substantive turns (>= 4 words).
    candidate = [re.sub(r"\((?:Note|Transcript)[^)]*\)", "", c).strip() for c in candidate]
    candidate = [c for c in candidate if len(c.split()) >= 4]
    fb = re.search(r"^(?:Examiner Feedback|Feedback):\s*(.+)$", text, flags=re.M | re.S)
    feedback = fb.group(1).strip() if fb else ""
    return Test(id=path.stem, band=band, url=url, candidate=candidate, feedback=feedback)


def analyze(test: Test, mapper, syntax_map: SyntaxMap, *, config: EngineConfig) -> TestResult:
    cefr_of = {n.id: n.cefr for n in syntax_map.nodes}
    proposals: dict[str, int] = {}
    for utt in test.candidate:
        match = mapper.map_text(utt, active_nodes=set())
        for nid in match.node_ids:
            proposals[nid] = proposals.get(nid, 0) + 1

    cefr_profile: dict[str, int] = {}
    for nid, c in proposals.items():
        cefr_profile[cefr_of[nid]] = cefr_profile.get(cefr_of[nid], 0) + c

    total = sum(cefr_profile.values())
    sys_idx = (
        sum(CEFR_INDEX[c] * n for c, n in cefr_profile.items()) / total if total else 0.0
    )
    ceiling = None
    for c in reversed(CEFR_ORDER):
        if cefr_profile.get(c, 0) >= 2:
            ceiling = c
            break

    # Ungraded engine read: every matched concept folded as correct dialog evidence.
    events = [
        Event(learner_id=test.id, node_ids=tuple(match_ids), correct=True, ts=float(i),
              source="dialog")
        for i, utt in enumerate(test.candidate)
        if (match_ids := mapper.map_text(utt, active_nodes=set()).node_ids)
    ]
    from klg_ai.core.graph import build_graph
    g = build_graph(syntax_map)
    mastery = mastery_from_events(g, events, config)
    activated = threshold_activated(mastery, config)
    statuses = {n: s.value for n, s in compute_status(g, activated).items()}
    known = sorted((n for n, s in statuses.items() if s == "known"), key=lambda n: CEFR_INDEX[cefr_of[n]])
    known_cefr: dict[str, int] = {}
    for n in known:
        known_cefr[cefr_of[n]] = known_cefr.get(cefr_of[n], 0) + 1

    return TestResult(
        id=test.id, band=test.band, band_cefr=band_to_cefr(test.band), url=test.url,
        n_utterances=len(test.candidate), proposals=dict(sorted(proposals.items())),
        cefr_profile=cefr_profile, system_cefr_index=round(sys_idx, 3),
        ceiling_cefr=ceiling, known=known, known_cefr=known_cefr,
    )


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def run(threshold: float = 0.22) -> dict:
    from klg_ai.semantic.mapper import default_mapper

    syntax_map = load_map()
    mapper = default_mapper(knowledge_injection=True, threshold=threshold, top_k=3,
                            syntax_map=syntax_map)
    config = EngineConfig()

    tests = [parse_test(p) for p in sorted(RAW_DIR.glob("test_*.md"))]
    results = [analyze(t, mapper, syntax_map, config=config) for t in tests]
    results.sort(key=lambda r: r.band)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        (PROCESSED_DIR / f"{r.id}.json").write_text(json.dumps(vars(r), indent=2))

    bands = [r.band for r in results]
    sys_idx = [r.system_cefr_index for r in results]
    corr = _pearson(bands, sys_idx)
    aggregate = {
        "n_tests": len(results),
        "threshold": threshold,
        "mapper": "kbert (semantic, ungraded)",
        "band_range": [min(bands), max(bands)],
        "pearson_band_vs_system_cefr": round(corr, 3),
        "tests": [vars(r) for r in results],
    }
    PINNED.parent.mkdir(parents=True, exist_ok=True)
    PINNED.write_text(json.dumps(aggregate, indent=2))
    return aggregate


def _print_table(agg: dict) -> None:
    print(f"\n{'test':9s} {'band':>5s} {'IELTS→CEFR':>11s}  {'sysCEFR':>7s} {'ceil':>4s}  "
          f"{'known by CEFR':<22s}")
    print("-" * 78)
    for r in agg["tests"]:
        kc = " ".join(f"{c}:{n}" for c, n in sorted(r["known_cefr"].items(),
                       key=lambda kv: CEFR_INDEX[kv[0]])) or "—"
        print(f"{r['id']:9s} {r['band']:>5.1f} {r['band_cefr']:>11s}  "
              f"{r['system_cefr_index']:>7.2f} {str(r['ceiling_cefr'] or '—'):>4s}  {kc:<22s}")
    print("-" * 78)
    print(f"Pearson(band, system CEFR index) = {agg['pearson_band_vs_system_cefr']:+.3f}  "
          f"(n={agg['n_tests']}, bands {agg['band_range'][0]}–{agg['band_range'][1]})")


if __name__ == "__main__":
    _print_table(run())
