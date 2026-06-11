# Phase 7 — KGT Personalization (RQ5)

**RQ5 (§1.5):** *"What is the most computationally efficient technical strategy
for personalization — retraining the entire model per user, or using
lightweight methods like Knowledge Graph Tuning (KGT)?"*

**TL;DR — the answer is KGT, and the margin is the compute.** Both strategies
personalize the same thing (a `(back, fwd)` multiplier per prerequisite edge of
the learner's personal graph) and land on **identical predictive fit**
(AUROC 0.6406 on SLAM en_es, same as the global engine — personalization, like
propagation in RQ3, is representational rather than a ranking lift on
next-token prediction). The difference is everything else:

| | Engine (global) | **Engine + KGT** | Engine + per-learner retrain |
|---|---|---|---|
| AUROC | 0.6406 | **0.6406** | 0.6406 |
| AUROC (cold) | 0.5001 | 0.5023 | 0.4999 |
| ms / learner (fit+predict) | 3.1 | **4.0** | 363.4 |
| fitting | — | **one closed-form pass** | 30 Adam epochs, 222 params/learner |
| interpretable? | — | **every adjustment carries its evidence** | "gradient-fitted" |

**KGT personalizes at +0.9 ms/learner over the global engine — ×90 cheaper
than per-learner gradient retraining — while producing human-readable,
per-edge explanations (§3.8's interpretability promise).** Run:
`python -m klg_ai.eval.run --data data/slam --max-learners 500 --seed 0 --kgt`
(committed artifact: `docs/phase7/results.json`; M1 Pro, torch 8 threads,
identical Phase-5 settings — the 8 Phase-5 model entries inside the Phase-7
results are numerically identical to `docs/phase5/results.json`).

## The KGT rule (`klg_ai/kgt.py`)

Closed-form, no gradients, pure function of `(graph, events, config, now)` —
the personal graph is recomputed from the event stream, never stored.

Per node, the same causal/source-weighted/recency-decayed fold as
`evidence.direct_scores`, split into two gated signals
(`gate g = 1 − exp(−mass / kgt_mass0)`, `kgt_mass0 = 2.0` — stricter than the
mastery score's 0.7, so only *consistent* feedback moves an edge):

```
k = acc · g          # demonstrably knows         k + w ≤ 1;
w = (1 − acc) · g    # demonstrably fails         both 0 without evidence
```

Per edge `u → v` (u prerequisite of v), multipliers on the two propagation
messages, clamped to `[0, 2]`:

```
agree       = k_u · k_v     →  reinforce both messages         (γ⁺ = 0.5)
contra_back = k_v · w_u     →  weaken the backward inference   (γ⁻ = 1.25)
contra_fwd  = k_u · w_v     →  weaken the forward readiness

m_back = clamp(1 + γ⁺·agree − γ⁻·contra_back, 0, 2)     (m_fwd analogous)
```

Factor ≈ 0 is edge **removal**, < 1 **weakened**, > 1 **strengthened** —
§3.8's add/remove/reweight. (Adding edges *outside* the curriculum DAG is out
of scope: the static map is the hypothesis space, as in Sun et al. 2024, which
re-weights existing KG structure.) A contradiction weakens exactly the message
it falsifies: knowing the dependent while failing the prerequisite says the
*backward* "v ⇒ u" inference is wrong for this learner; the converse says the
*forward* readiness signal misleads.

**Worked example (the built-in "struggling" learner):** 4/4 correct on
*Defining Relative Clauses*, 0/4 on its prerequisite *Relative Pronouns* →
`m_back = 0.16` (weakened to near-removal). Without KGT the graph lifts
Relative Pronouns to the inference ceiling (mastery 0.45) on neighbours alone
— despite the learner failing it every time; with KGT the unearned inference is
cut to 0.15. Every reported adjustment carries its evidence:

> *"Knows 'Defining Relative Clauses' (4/4 correct …) but struggles with its
> prerequisite 'Relative Pronouns' (0/4 correct …) — backward inference ×0.16"*

**Invariants (by construction + `tests/test_kgt.py`):** factors multiply the
raw alphas in `propagation._build_edges` *before* per-target normalization, so
propagation stays a convex combination; self-loops are never scaled (no
division by zero even with every incident edge removed); the
`inferred_ceiling` cap is untouched, so KGT redistributes inference but never
fabricates "known" — the demo contract (48 known / 3 interior gaps) holds with
the toggle on. Absence of evidence is never contradiction: an edge moves only
when *both* endpoints have evidence.

## The retrain comparator (`klg_ai/eval/retrain.py`)

The "retrain the model per user" arm, deliberately given the **same hypothesis
space** so the comparison isolates fitting strategy: per learner, one
`(m_back, m_fwd)` per edge, reparameterized `m(θ) = 2·sigmoid(θ)` (θ = 0 ⇒
m = 1 ≡ the global graph), fitted by full-batch Adam (30 epochs, lr 0.05) on
BCE between the engine's mastery readout and the learner's own train
correctness, with L2 anchoring θ at the global weights. The propagation
forward pass is re-implemented in differentiable torch (the production PyG
layer runs under `no_grad`); direct scores are held fixed. Deterministic: zero
init, fixed epochs, seeded.

Cost protocol: every model's `predict()` is timed with `perf_counter` in one
sequential process on the same machine; the lazy torch/PyG import is warmed up
before timing; calibration (`fit_platt`) is identical across engine arms, so
its cost cancels. `meta` records `torch_threads`.

**A finding worth keeping:** the gradient arm *cannot even reach* the
ceiling-saturated contradiction KGT fixes. Where the raw inferred lift exceeds
`inferred_ceiling`, the clamp's gradient is zero — so for nodes like the
struggling learner's *Relative Pronouns* (parked exactly at the ceiling), no
learning signal flows to the offending edges, while the closed-form rule cuts
them directly. Smooth loss descent elsewhere, blindness exactly where the
§3.8 story lives.

## Robustness of the AUROC null

KGT genuinely fires on SLAM — it is not a no-op: at the default gate it
adjusts edges for **500/500 learners** (6,006 reportable adjustments, ≈12 per
learner) and moves 75,486 of 75,512 predictions. The null is flat across the
knob sweep:

| config | AUROC | AUROC (cold) | preds moved |
|---|---|---|---|
| `mass0=2.0, γ⁻=1.25` (default) | 0.6406 | 0.5023 | 75,486 |
| `mass0=1.0, γ⁻=1.25` | 0.6406 | 0.5022 | 75,432 |
| `mass0=0.5, γ⁻=2.0` | 0.6406 | 0.5025 | 75,496 |
| `mass0=1.0, γ⁻=2.0, γ⁺=1.0` | 0.6406 | 0.5052 | 75,486 |

Mirrors RQ3: on a 30-day window of practice data, re-weighting *inference*
edges re-ranks little — the value is the corrected personal representation
(what the viewer shows) and the cost/interpretability profile, not next-token
AUROC.

## What shipped (7-B and the §3.7 handoff)

- **Engine:** `klg_ai.kgt` (`tune_edges`, `EdgeAdjustment` with per-edge
  `reason`), `EngineConfig.kgt` toggle (+ 6 knobs, default **off** — all
  Phase-5/6 outputs unchanged), `propagate(..., edge_factors=)`,
  `compute_edge_adjustments`, `learner_events`; a built-in **"struggling"**
  learner (knows two B1/B2 dependents, consistently fails their prerequisites)
  so the contradiction case is demo-able; `generate_events(failed=...)`.
- **Eval:** `--kgt` adds `engine_kgt` + `engine_retrain` (group RQ5), per-model
  `cost`, the retrain loss curve, and the `RQ5_personalization` group; writes
  `results_kgt.json` (never clobbers Phase-5 `results.json` — guarded by
  tests).
- **API:** `GET /api/learner/{id}/status?kgt=1` → personalized mastery +
  `edge_adjustments`; `gap_scores` (§3.7: `level_weight(cefr) × (1 − mastery)`
  over interior-gap + frontier nodes) now always on the status payload;
  `POST /api/learner/{id}/retrain?epochs=N` runs the gradient comparator live
  (sub-second) and returns per-epoch loss + factor snapshots;
  `POST /api/chat` runs KGT on the conversation (wrong usage weakens edges
  live; `KLG_CHAT_KGT=0` kill-switch, `KLG_CHAT_KGT_MASS0` default 1.0);
  `GET /api/metrics` resolves the Phase-7 results first (a strict superset of
  the Phase-5 table). Contract doc for the future recommender:
  `docs/knowledge-state-api.md`.
- **Viewer:** Map tab **"Personal graph (KGT)"** toggle — reinforced edges
  green/thick, weakened dashed red, removed dotted; NodeDetails shows each
  incident adjustment with its ×factor and evidence reason; **"Retrain on this
  learner"** replays the gradient fit epoch by epoch (scrubber + play/pause +
  live loss) ending in the wall-time verdict (e.g. *"87 ms vs KGT 0.3 ms —
  ×289"* for the struggling learner); Validation tab **RQ5 card** (parity
  bars, log-scale cost bars, convergence curve); Chat tab notes when the
  conversation adapts personal edges.

## Limitations / future work

- **Reweight-only:** KGT tunes existing curriculum edges; discovering *new*
  per-learner edges is out of scope (and would need a different safety story).
- **Factors are pre-normalization:** the viewer shows the raw multipliers; the
  effective convex weights differ after per-target renormalization.
- **30-day window:** SLAM's short history limits how much consistent
  contradiction can accumulate; richer longitudinal data (Subturtle's
  `leitner_review_log`) is where per-learner structure should matter more.
- **The retrain arm is the fair in-family comparator**, not the strongest
  possible per-user model; its ceiling-clamp blindness (above) suggests a
  soft-clamp variant if it is ever pursued seriously.
