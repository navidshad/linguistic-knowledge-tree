"""Graded IELTS benchmark — the no-tagging baseline.

Runs the same 11 transcripts through the **live chat pipeline** (semantic mapper +
Gemini *grading*) rather than the ungraded mapper of ``ielts.py``. Gemini here only
*prunes* the mapper's proposals (precision); it does **not** propose concepts — so
this is the honest "system as-is, without Gemini tagging" baseline that a future
Gemini-tagging arm is compared against.

Requires the backend running with a Gemini key (``uvicorn klg_server.main:app``).
Writes the ``graded`` arm into ``docs/ielts-case-study/results.json`` alongside the
ungraded arm. Run: ``python -m klg_ai.eval.ielts_graded``.

Three discrimination metrics per test (we report all, then pick the fairest):
  - ``cefr_com``  : mastery-weighted centre-of-mass CEFR (0=A1 … 5=C2)
  - ``ceiling``   : highest CEFR with mastery >= 0.3 (the reached ceiling)
  - ``adv_mass``  : fraction of mastery mass at B2+ (advanced grammar share)
"""
from __future__ import annotations

import json
import urllib.request

from klg_ai.core.loader import load_map
from klg_ai.eval.ielts import (
    CEFR_INDEX, CEFR_ORDER, PINNED, RAW_DIR, _pearson, band_to_cefr, parse_test,
)

BASE_URL = "http://localhost:8000"


def _post(messages: list[dict], tagger: str | None = None) -> dict:
    payload = {"messages": messages, "activated": [],
               "session_id": "ielts-graded", "profile_id": None}
    if tagger:
        payload["tagger"] = tagger
    req = urllib.request.Request(BASE_URL + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def run_graded(arm: str = "graded", tagger: str | None = None,
               mapper_label: str = "kbert + gemini grading (no tagging)") -> dict:
    smap = load_map()
    cefr_of = {n.id: n.cefr for n in smap.nodes}
    tests = [parse_test(p) for p in sorted(RAW_DIR.glob("test_*.md"))]

    rows = []
    for t in tests:
        resp = _post([{"role": "user", "text": u} for u in t.candidate], tagger=tagger)
        mastery = {n: m for n, m in (resp.get("mastery") or {}).items() if m > 0}
        statuses = resp.get("statuses") or {}
        total = sum(mastery.values())
        com = sum(CEFR_INDEX[cefr_of[n]] * m for n, m in mastery.items()) / total if total else 0.0
        adv = sum(m for n, m in mastery.items() if CEFR_INDEX[cefr_of[n]] >= 3) / total if total else 0.0
        ceiling = None
        for c in reversed(CEFR_ORDER):
            if any(cefr_of[n] == c and m >= 0.3 for n, m in mastery.items()):
                ceiling = c
                break
        known = sorted((n for n, s in statuses.items() if s == "known"),
                       key=lambda n: CEFR_INDEX[cefr_of[n]])
        known_cefr: dict[str, int] = {}
        for n in known:
            known_cefr[cefr_of[n]] = known_cefr.get(cefr_of[n], 0) + 1
        rows.append({
            "id": t.id, "band": t.band, "band_cefr": band_to_cefr(t.band), "url": t.url,
            "n_utterances": len(t.candidate),
            "cefr_com": round(com, 3), "ceiling_cefr": ceiling, "adv_mass": round(adv, 3),
            "n_known": len(known), "known_cefr": known_cefr,
            "counts": resp.get("counts", {}),
        })
        print(f"  {t.id}  band {t.band:<4}  com {com:5.2f}  ceil {str(ceiling or '-'):>3}  "
              f"adv {adv:4.2f}  known {len(known)}")

    rows.sort(key=lambda r: r["band"])
    bands = [r["band"] for r in rows]
    metrics = {
        "pearson_band_vs_cefr_com": round(_pearson(bands, [r["cefr_com"] for r in rows]), 3),
        "pearson_band_vs_adv_mass": round(_pearson(bands, [r["adv_mass"] for r in rows]), 3),
        "pearson_band_vs_ceiling": round(_pearson(
            bands, [CEFR_INDEX.get(r["ceiling_cefr"], 0) for r in rows]), 3),
    }
    result = {"mapper": mapper_label, "metrics": metrics, "tests": rows}

    agg = json.loads(PINNED.read_text()) if PINNED.exists() else {}
    agg[arm] = result
    PINNED.write_text(json.dumps(agg, indent=2))
    return result


def _print_summary(g: dict) -> None:
    print(f"\n{'test':9s} {'band':>5s} {'IELTS→CEFR':>11s}  {'cefr_com':>8s} {'ceiling':>7s} "
          f"{'adv_mass':>8s}  {'known by CEFR'}")
    print("-" * 84)
    for r in g["tests"]:
        kc = " ".join(f"{c}:{n}" for c, n in sorted(r["known_cefr"].items(),
                      key=lambda kv: CEFR_INDEX[kv[0]])) or "—"
        print(f"{r['id']:9s} {r['band']:>5.1f} {r['band_cefr']:>11s}  {r['cefr_com']:>8.2f} "
              f"{str(r['ceiling_cefr'] or '—'):>7s} {r['adv_mass']:>8.2f}  {kc}")
    print("-" * 84)
    for k, v in g["metrics"].items():
        print(f"  {k} = {v:+.3f}")


if __name__ == "__main__":
    _print_summary(run_graded())
