# Trained Propagation Weights (RQ3) — Validation

**Question (RQ3, revisited):** the GNN propagation layer ships with *hand-set,
untrained* edge weights (`prop_alpha_back = 0.5`, `prop_alpha_fwd = 0.15`).
Phase 5 found propagation **null on predictive AUROC** — but with a hedge: nobody
ever *fit* those weights to data. This work removes the hedge by training the
propagation weights on the open Duolingo SLAM data, then re-comparing trained vs.
fixed vs. propagation-off.

> **Scope.** This is the **knowledge-illustration** model (model + visualize a
> learner's mastery over the prerequisite graph), not recommendation. Training is
> a **backend, terminal-run pipeline**; the result surfaces in the app only via the
> read-only Validation dashboard. The live engine/API/viewer overlay is untouched
> (still fixed-weight).

## Method

**Global, not per-learner.** One weight set is fit on *all* learners' pooled train
interactions and used for everyone — distinct from the per-learner gradient
*retrain* of RQ5 (`tuning/retrain.py`), which fits a separate θ per learner. The
two share only the differentiable propagation forward.

- **`tuning/train_prop.py` · `fit_global_edge_factors`** — reuses the differentiable
  propagation forward (self-loops, per-target normalization, `prop_rounds` rounds,
  lift clamped at `inferred_ceiling`), but **batches all learners**: the graph is
  shared and only each learner's direct-score vector differs, so one sparse
  diffusion propagates every learner's column at once. Supervision is pooled BCE
  over every `(learner, train-item)` pair; θ is reparameterized
  `m = factor_max·sigmoid(θ)` (θ=0 ⇒ ×1) with L2 anchoring at the global weights.
  Deterministic (zero init, full-batch Adam, fixed epochs).
- **Two granularities:**
  - **per-edge** (`engine_trained`) — a `(back, fwd)` multiplier per prerequisite
    edge (~222 params); the headline trained model.
  - **2-scalar** (`engine_trained_scalar`) — just the two global `α` weights,
    broadcast to all edges; a robustness check that the result is not an artifact
    of one hand-set constant.
- **Protocol** — identical to Phase 5: temporal train/eval split, point-in-time
  causal mastery, global Platt calibration, AUROC headline + the cold-node slice.
  The trained weights are frozen before they touch the eval split. `epochs=0`
  reduces the trained predictor *exactly* to the fixed engine (test-guarded), so
  any AUROC difference is attributable to the learned weights.

## Results

Duolingo SLAM 2018 `en_es`, dev split, 500-learner seeded sample (seed 0); 75,512
held-out tokens (1,597 on cold nodes), 14% mistake rate. Artifact:
`docs/trained-prop/results.json`.

| Model | AUROC | AUROC (cold) |
|---|---|---|
| Engine − propagation (`engine_no_prop`) | 0.641 | 0.500 |
| Engine, fixed weights (`engine_full`) | 0.641 | 0.500 |
| **Engine + trained propagation, per-edge** (`engine_trained`) | **0.641** | **0.505** |
| Engine + trained propagation, 2-scalar (`engine_trained_scalar`) | 0.641 | 0.500 |
| DKT (sequence) | 0.656 | 0.620 |

**Verdict — the RQ3 null is robust to training.** Fitting the propagation weights
to the data moves predictive AUROC by **±0.000**: per-edge (~222 params) and the
2-scalar fit both land at **0.641**, identical to the hand-set 0.5/0.15 weights and
to propagation-*off*. On the cold-node slice the per-edge fit nudges AUROC from
0.500 to **0.505** — a hair above chance, not a meaningful lift.

This **upgrades the Phase-5 hedge** ("the weights were never trained") into a firm
finding: *even fit to the data, the prerequisite graph carries no predictive signal
for next-step correctness on SLAM.* Two structural reasons reinforce it: (1) cold /
inference-only tokens are ~2% of the eval set, so propagation can barely move the
headline regardless of weights; (2) the `inferred_ceiling` clamp caps any inferred
lift below the known threshold by design. The graph's value is **representational**
— it supplies inferred mastery for the interior-gap overlay (the knowledge
*illustration*) — not predictive. This is consistent with the Phase-5 RQ3/RQ4 and
Phase-6 RQ2 themes: the structure illustrates knowledge, the direct evidence
predicts it.

## Reproduce

```bash
# backend-only, manual terminal run (licensed SLAM data in the git-ignored data/)
python -m klg_ai.eval.run --data-source slam --kgt --train-prop \
    --max-learners 500 --split dev --out data/eval/results_trainprop.json
# results_trainprop.json is a strict superset of the Phase-7 results (keeps RQ5,
# adds the RQ3-trained arms); the Phase-5 results.json is never touched.
```

The Validation tab then shows an **RQ3 · Trained propagation** card (fixed vs.
trained per-edge vs. 2-scalar vs. no-prop, overall + cold AUROC, plus the global
fit's loss curve), alongside the unchanged RQ1/RQ4/RQ5 cards.

## Notes

- **No new runtime deps** — the trainer uses `torch` (already a dep) for gradients
  only; the runtime propagation forward stays parameter-free.
- **Dataset-agnostic** — the trainer/eval consume `list[LearnerData]`; a new open
  source (EdNet, …) is just a new loader in `klg_ai/data/sources.py`. SLAM only today.
- **Artifact** — the learned weights serialize via `GlobalPropTrace.as_weights_json`
  (per-edge multipliers + effective `α`s) for the thesis figure.
