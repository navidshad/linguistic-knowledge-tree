# CLAUDE.md — Linguistic Knowledge Tree

Master's thesis (VGTU, MSc AI): models, computes, and **visualizes** a learner's
English-grammar mastery as a static CEFR syntax map with a per-learner overlay.
Thesis paper in `paper-work/`; phased plan in `docs/roadmap.md`.

## What this system is (and isn't)

- A **static, global English syntax map** (A1–C2), sourced from CEFR / English
  Grammar Profile. **Same map for every learner**; learner data only *activates*
  nodes. Edges = prerequisite / learning-order; the graph is a **DAG**.
- Per learner, each node gets a **computed status** = pure function of
  `(graph + activated set)`:
  - `known` — activated
  - `interior_gap` — unactivated node *between* two activated nodes (a known
    ancestor AND a known descendant) → the "nodes in between" feature
  - `frontier` — unactivated, all direct prerequisites known
  - `further` — everything else
- **Out of scope:** deciding *what to learn next* (recommendation / policy) —
  delegated to a separate system later. This system represents + visualizes +
  computes status only.
- Two layers: syntax/skill nodes = the map (what's shown); lexical/phrase data =
  the *evidence* that activates them.
- Scope is **A1–C2** (full range). Note: thesis §1.8 still says "Intermediate
  English" — update when convenient.

## Structure

- `map/` — static map JSON (`english-syntax-map.v0.json`, 113 nodes, valid DAG) +
  `validate.py` (run after any edit). v0 is a curated seed, expand toward full EGP.
- `backend/ai/` (`klg_ai`) — pure-Python engine, **no web deps**: loader, graph,
  activation, status, gaps.
- `backend/server/` (`klg_server`) — FastAPI: `/api/map`,
  `/api/learner/{id}/status`, `POST /api/status` (arbitrary activated set, for
  interactive what-if).
- `frontend/` — Vue 3 + Vite + TS + Pinia + Cytoscape.js. Modular:
  components / composables / stores / services / types / constants.
- Backend is **Python** (decided). At ship time it runs as a Python service that
  Subturtle's Node server calls; the Vue frontend ports to Subturtle's Nuxt.

## Run

```bash
# backend
python3 -m venv backend/.venv && source backend/.venv/bin/activate
pip install -e backend/ai
pip install -e "backend/server[dev]"
pytest backend/ai backend/server -q
uvicorn klg_server.main:app --reload --port 8000     # docs at /docs

# frontend (separate terminal)
cd frontend && npm install && npm run dev             # :5173, expects API on :8000

# validate the map after editing it
python3 map/validate.py
```

(`.claude/launch.json` has a local preview config; `.claude/` is gitignored.)

## Validation plan (the thesis "proof")

Validate activation/propagation on open English-L2 knowledge-tracing data —
**Duolingo SLAM** (primary) + **EdNet** (scale check) — with **pykt** baselines
(DKT, etc.). RQ→phase mapping in `docs/roadmap.md`. C2 is data-thin, so it'll be
structurally demonstrated, empirically lighter (Subturtle's idiom/discourse/
formality signals partly offset this at the top).

## Target platform — Subturtle (SEPARATE repo)

`/Users/navid-shad/Projects/CodeBridger/learn-by-subtitle/subturtle-dashboard-app`

- Frontend: **Nuxt 3** (Vue 3 + TS + Pinia + Tailwind + Vitest). Server:
  **Node / `@modular-rest`** (MongoDB).
- Learner data available: conversational **dialog turns**; **phrases** with
  `linguistic_data` + typed `chunks` (collocation / phrasal_verb / idiom /
  discourse_marker + formality); **Leitner SRS** (`boxLevel` per phrase).
  NOT available: structured grammar-exercise attempt logs.
- ⚠️ **Leitner stores only the current snapshot — no review history.** KT needs
  the time-series of attempts. Spec to add an append-only `leitner_review_log`:
  `docs/subturtle-review-event-log-spec.md`. Emit point:
  `LeitnerService.submitReview()` in
  `server/src/modules/leitner_box/service.ts`. **Start logging early — history
  cannot be backfilled.**
- Evidence strength for activation: Leitner recall (graded) > dialog (productive
  use) > phrase save (exposure).

## Git workflow

`main` (stable, tracks `origin/main`) → `dev` (integration; has Phase 1) →
`feat/*` (features, branch off `dev`). Phase 1 = commit `c2bb07b`. `dev` and
feature branches are local-only (not pushed yet).

## Status

- **Phase 1 — DONE** (on `dev`): static map + engine + API + Vue viewer.
  Viewer features: matrix / concentric (A1 center → C2 outer) / force layouts;
  CEFR level filter; learner-overlay toggle; interactive node activation
  (right-click / details button → `POST /api/status`).
- **Phases 2–4A — DONE** (merged into `dev`).
  `get_activation()` is now a real event-driven engine, not a fixture. Pipeline,
  all pure functions of the inputs:
  `events → evidence (source-weighted review>dialog>exposure, recency-decayed)
  → GNN propagation (PyG MessagePassing, lazy-imported) → threshold → known set`.
  New `klg_ai` modules: `events`, `forgetting`, `evidence`, `propagation`,
  `adapters/synthetic`. `EngineConfig` carries the `forgetting`/`propagation`
  on-off switches for the RQ3/RQ4 ablations. Collapses the roadmap's Phase 2
  (activation) + Phase 4-A (propagation + forgetting) model layers into one
  branch. Propagation weights are **fixed/untrained** in v1 — the architecture is
  there so Phase 5 can train it on Duolingo/EdNet (RQ3).
  - **Contract preserved:** `get_activation` still returns `set[str]`; the demo
    learner still == `DEMO_KNOWN` (48 known / 3 interior_gap), so API + viewer are
    unchanged. Inferred-only lift is capped below the known threshold, so pure
    inference never fabricates "known" (interior gaps don't fill themselves in).
  - Continuous mastery is exposed via `compute_mastery()`; surfaced to the API +
    viewer in Phase 4-B (below).
  - Deps added: `torch`, `torch-geometric`. Data source is still fixture-level
    (demo + seeded synthetic generator); the real Duolingo SLAM adapter is Phase 5.
- **Phase 4-B — DONE** (merged into `dev`): continuous
  mastery surfaced and made *visible*.
  - API: `mastery` (per-node [0,1]) added to `GET /api/learner/{id}/status`; new
    `GET /api/learner/{id}/timeline?frames=N` samples mastery across the learner's
    history. `evidence.direct_scores` is now **point-in-time / causal** (ignores
    evidence logged after `now`), so an earlier `now` reconstructs past knowledge —
    what lets the scrubber show mastery *grow*. New engine helpers:
    `threshold_activated`, `mastery_timeline`, `event_span`.
  - Viewer: **confidence overlay** — node opacity ∝ mastery (status = color,
    confidence = opacity), so GNN-lifted interior gaps *glow* below the known
    threshold; **timeline scrubber** (slider + play/pause) replays growth →
    forgetting; mastery shown as a bar in the node panel.
  - Contract preserved: status counts unchanged (demo still 48/3). `mastery` is
    nullable — the `POST /api/status` what-if has no evidence, so it's `null` and
    the viewer approximates opacity from the discrete status there.
- **Phase 5 — DONE** (merged into `dev`): validation on real
  open data. Full write-up: `docs/phase5-validation.md`.
  - **Data:** Duolingo SLAM 2018 **en_es** (English learners, L1 Spanish; 2.6M
    token instances, 2,593 learners, 14% mistake rate). License-gated on Harvard
    Dataverse — **not committed**; `data/` is git-ignored. **en_es = English
    tokens** (the direction is ambiguous in the literature — confirmed from the
    bytes). Label **1 = mistake**, so `Event.correct = (label == 0)`.
  - New `klg_ai.adapters.slam` + `slam_mapping` (parse mirrors official
    `baseline.py`; rule-based morphosyntactic tag → concept node, dependency-aware
    for verb constructions, 72.5% token coverage). New `klg_ai.eval` package:
    `dataset` (temporal split + cold-node mask), `predict` (engine-as-predictor +
    Platt calibration), `baselines` (in-repo **DKT** LSTM + difficulty/ability/
    chance), `metrics` (ported from SLAM `eval.py`, pinned to its oracle),
    `ablations`, `run` (CLI: `python -m klg_ai.eval.run`).
  - **Results (RQs):** engine **0.641** ≈ DKT **0.656** > per-skill 0.605 >
    chance 0.500 (RQ1: graph competitive with sequence). Propagation null on
    predictive AUROC (RQ3) but supplies inferred mastery to 76% of cold nodes
    (representational, not predictive). Forgetting negligible on the 30-day
    window (RQ4). No new runtime deps (metrics pure-Python; DKT uses torch).
  - **Viewer:** new **Validation** tab (RQ cards + ROC curves + ablation table)
    served by `GET /api/metrics` (reads `data/eval/results.json` else committed
    `docs/phase5/results.json`). Map tab + status contract unchanged (demo 48/3).
  - Open: pykt/EdNet scale-check and *training* the propagation weights (RQ3) are
    deferred — the architecture supports it; Phase 5 delivers the validated
    comparison with fixed weights.
- **Phase 6 — DONE** (on `feat/phase6-semantic-embedding`, off `dev`): semantic
  embedding (BERT vs K-BERT) for evidence→node mapping (RQ2) + a Gemini chat demo.
  Full write-up: `docs/phase6-validation.md`.
  - **Semantic mapper** — new `klg_ai.semantic` package: `embedder` (lazy
    sentence-transformers MiniLM, **CPU-pinned** for bit-stable vectors; a
    dependency-free `HashingEmbedder` fake forced by `KLG_EMBEDDER=hash` so tests/CI
    need no model), `node_vectors` (per-node embeddings; **K-BERT** = blend each
    node's text with its 1-hop prerequisite-graph neighbours — the RQ2 toggle is
    just which matrix the mapper dots against), `mapper` (`SemanticMapper.map_text`
    / `map_token_in_context` / `map_exercise`, cosine + top-k + threshold), `build_vectors`
    CLI. Committed artifact: `docs/phase6/node_vectors.npz` (MiniLM, dim 384).
    `sentence-transformers` is an optional `[semantic]` extra (lazy-imported).
  - **RQ2 eval** — `klg_ai.eval.mapping_eval` + `mapping_run` CLI: **intrinsic**,
    token-in-context vs the rule-based `map_exercise` as **silver labels** on SLAM
    en_es (micro/macro P/R/F1, coverage, exact-set, confusions). **Extrinsic** reuses
    the Phase-5 harness via an **additive** `load_track/run_ablations/run(--mapper
    {rule,bert,kbert})` flag defaulting to `rule` (KLG_SEMANTIC_THRESHOLD env) — the
    Phase-5 numbers + `docs/phase5/results.json` are never touched.
  - **Result (RQ2):** semantic embeddings recover the rule mapping only **weakly**
    (peak micro-F1 ≈ 0.14 at threshold 0.25 vs the rule oracle's 1.0) and
    **K-BERT ≈ BERT** — 1-hop prerequisite-graph injection gives no consistent lift.
    Frozen sentence embeddings capture grammatical *construction* poorly; the
    morphosyntactic rule mapper stays the stronger evidence source. (Mirrors the
    Phase-5 RQ3 theme: the graph is representational, not a predictive lift.)
  - **Chat demo (6-B)** — `POST /api/chat` (`routers/chat.py` + `gemini.py`):
    stateless like `POST /api/status`; a **Gemini** tutor (minimal REST, **deterministic
    mock** when `KLG_GEMINI_MOCK=1`/no key) replies, and each learner turn is mapped→
    nodes (validated mapper, `dialog` source) → live map overlay. Viewer: a **Chat**
    tab (`ChatPanel.vue` + `stores/chat.ts`) where the map lights up as you talk, and
    a NodeDetails **Evidence** section showing the turns behind a node. Map/Validation
    tabs + demo contract (48/3) unchanged. Chat threshold `KLG_CHAT_THRESHOLD` (0.22)
    is below the eval default since a whole turn embeds to a "mixed" point.
  - Open: train the K-BERT/propagation weights (vs fixed); richer node text + fuller
    EGP coverage to lift mapping F1; optional Gemini-assisted turn tagging.
