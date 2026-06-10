# Implementation Roadmap — Learner Linguistic Knowledge Graph

**Thesis:** Modeling and Visualization of a Learner's Linguistic Knowledge Graph (VGTU, MSc AI)
**Goal of build:** validate the thesis design empirically, then clone the repo to ship for Subturtle.
**Two tracks, advancing together:** **(A) Model** (graph + activation + gaps) and **(B) Visualization** (web app). Every phase delivers something visible.

---

## Locked design decisions

| Decision | Choice |
|---|---|
| Graph structure | **Static, global** map of English syntax. Same map for every learner. |
| Map source | CEFR / **English Grammar Profile** (curriculum-defined). |
| Scope (ceiling) | **A1–C2**, full range. Curation depth front-loaded on the A2–C1 core; C1/C2 as a refinement layer deepened later. |
| Node granularity | Two levels: **category** (noun, verb, tense, clause…) → **concept** (present perfect, relative clauses, comparatives…). |
| Edge meaning | **Prerequisite / learning-order** (what enables what). |
| Per-learner data | **Activation overlay** on the static map; updated **iteratively** as more data arrives. |
| Node status | Computed, not authored: **known ● / interior-gap ○ / frontier ◐ / further ○**. |
| "In-between" gaps | Connecting subgraph between activated nodes → surface unactivated nodes on the path. |
| Recommendation ("what's next") | **Out of scope** — delegated to a separate system. We expose the knowledge state via API. |
| Validation data | Open English-L2 KT data: **Duolingo SLAM** (primary), **EdNet** (scale check). |
| Productization | Subturtle adapter + dashboard integration, **after** validation. |

---

## Architecture

```
[ Evidence sources ]                          [ Static Map  (EGP/CEFR JSON) ]
  Duolingo / EdNet  (validation)                         │
  Subturtle         (later)        ──►[ Adapters ]──►[ Activation + Propagation + Forgetting Engine ]
                                                         │
                                          [ Status Computer:  known / interior-gap / frontier / further ]
                                                         │
                                                  [ API  (FastAPI) ]──►[ "What's-next" recommender — SEPARATE system, later ]
                                                         │
                                          [ Web App — Cytoscape.js graph visualization ]
```

**Stack**
- **Backend / model:** Python — `networkx` (graph algorithms: shortest path, connecting subgraph, frontier), `PyTorch` + `PyTorch Geometric` (GNN propagation), `transformers` (BERT/K-BERT, later), `pykt` (KT baselines for validation), `FastAPI` (serve per-learner map + status).
- **Frontend:** **Vue 3 + Vite + TypeScript + Pinia** + **Cytoscape.js** (matrix layout: category lanes × CEFR columns). `Sigma.js`/WebGL as a fallback if the full ~1,200-node map gets heavy. (Ports to Subturtle's Nuxt later.)
- **Contracts (language-agnostic):** the static-map JSON, the `Event` schema, and the knowledge-state API. These are what make "clone & ship to Subturtle" a re-implementation of the engine behind the same contracts, not a rewrite.

> **Decision point — backend language.** Python is recommended for the thesis (ML-native, best for validation). Subturtle is TS/Node. Plan: build the prototype in Python behind a clean HTTP API; the frontend is stack-agnostic; port the engine to TS only if Subturtle later needs it in-process. The map JSON + API contract carry over unchanged either way.

**Repo layout (monorepo)**
```
linguistic-knowledge-tree/
  paper-work/        # thesis
  docs/              # specs, this roadmap, design notes
  map/               # static syntax map: JSON + validator + schema
  backend/
    ai/              # klg_ai — engine: loader, graph, activation, status, gaps
                     #          (adapters + eval/pykt baselines land here too)
    server/          # klg_server — FastAPI: serve map + per-learner status
  frontend/          # Vue 3 + Vite + TS + Pinia + Cytoscape.js
```

---

## Stages & Phases

**Status legend:** ✅ Done · 🟡 Partial · ⬜ Todo.
(Phases 0–6 are merged into `dev`; Phase 7 is on `feat/phase7-kgt-personalization`, off `dev`.)

### Stage I — Map & Explore

**Phase 0 — Scaffolding (end-to-end skeleton)** · ✅ Done
- A: Monorepo, core data contracts (static-map JSON schema, `ActivationState`, `NodeStatus`, `Event`), CI.
- B: Render a hardcoded 5-node map in the web app, colored by status; backend serves it.
- **Deliverable:** the full stack is wired — backend JSON → frontend renders a colored DAG.

**Phase 1 — The static map + map browser** · ✅ Done
- A: Build **v0 EGP/CEFR map** (categories A1–C2, representative concept nodes, prerequisite edges) as versioned JSON; backend serves the full map.
- B: Interactive DAG of the whole map (Cytoscape + dagre), banded by CEFR level (A1→C2), grouped by category, zoom/pan, node tooltips. No learner overlay yet.
- **Deliverable:** browse the entire English syntax map visually. *(= paper §1.4.4 Phase 1: visualize curriculum for teachers.)*

### Stage II — Learner Overlay (the core product vision)

**Phase 2 — Activation from data + overlay** · ✅ Done
- A: Evidence→node mapping (rule-based v1: graded attempts on items tagged to a node raise its activation). Status v1: **known / frontier / further** (frontier = unactivated neighbors of known).
- B: Overlay a learner's activation on the map; learner selector; "more data → map updates."
- **Deliverable:** pick a learner, watch their boundary + frontier light up on the real map.

**Phase 3 — Gap detection / in-between nodes** · ✅ Done ← headline differentiator
- A: Connecting subgraph / shortest paths between activated nodes → surface **interior-gap** nodes; common-ancestor handling for cross-branch pairs. Status v2: all four statuses.
- B: Render interior gaps distinctly (e.g., dashed red), draw the connecting paths; toggle **"my relevant subgraph"** vs full map.
- **Deliverable:** "learner activates two distant nodes → system reveals the nodes in between." *(= thesis Objective 3, §3.7.)*

### Stage III — Dynamic & Validated (the thesis proof)

**Phase 4 — Propagation + forgetting** · ✅ Done
- A: GNN-style **propagation** (activating a node lifts inferred mastery of related nodes); **forgetting/decay** over time (Leitner-style) so the map is temporally dynamic.
- B: Confidence encoded as **node opacity ∝ mastery** (status = color, confidence = opacity), so GNN-lifted interior gaps glow below the known threshold; **timeline scrubber** (slider + play/pause) replays growth + forgetting. Mastery surfaced via `mastery` on `GET /api/learner/{id}/status` + `GET /api/learner/{id}/timeline`; evidence scoring is now point-in-time/causal.
- **Deliverable:** the map infers beyond direct evidence and evolves over time. *(= §3.5, §3.6, §2.6.)*

**Phase 5 — Validation on open data** · ✅ Done (on `feat/phase5-validation`) ← turns the thesis from theoretical to validated
- A: **Duolingo SLAM en_es** adapter (real morphosyntactic tag → concept-node mapping, 72.5% token coverage) → engine run → **next-step prediction, AUROC vs. in-repo DKT** + difficulty/ability/chance baselines. Ablations per RQ: propagation on/off (RQ3, incl. a cold-node slice), forgetting on/off (RQ4), graph vs. sequence (RQ1). (DKT chosen as a self-contained sequence baseline instead of pykt; EdNet scale-check deferred.)
- B: **Validation dashboard** tab in the viewer (RQ cards, ROC curves, ablation table) served from `GET /api/metrics`.
- **Deliverable:** the numbers that answer the RQs — engine 0.641 ≈ DKT 0.656 > difficulty 0.605 > chance 0.500; propagation/forgetting null on predictive AUROC but propagation gives representational coverage on cold nodes. Full write-up: `docs/phase5-validation.md`.

### Stage IV — Semantic & Personal

**Phase 6 — Semantic embedding (BERT/K-BERT) for evidence→node mapping** · ✅ Done (on `feat/phase6-semantic-embedding`) ← lets *unlabeled text* activate the map
- A: `klg_ai.semantic` — a local **BERT** (sentence-transformers MiniLM, CPU-pinned, lazy) text→node mapper, with a **K-BERT** variant that fuses each node's 1-hop prerequisite-graph neighbours into its embedding (the RQ2 toggle). Committed node-vector artifact (`docs/phase6/node_vectors.npz`); a deterministic `HashingEmbedder` keeps tests offline. **Intrinsic RQ2** (`klg_ai.eval.mapping_eval`): scored token-in-context against the rule-based `map_exercise` as silver labels on SLAM en_es. **Extrinsic** RQ2 reuses the Phase-5 harness via an additive `mapper={rule,bert,kbert}` flag (Phase-5 numbers untouched).
- B: **Chat tab** — a Gemini tutor (`POST /api/chat`, REST + deterministic mock); each learner turn is mapped→nodes by the validated mapper, folded as `dialog` evidence, and the map lights up live. Click a node → the **Evidence** panel shows the chat turns that activated it.
- **Deliverable:** learner dialog activates the map automatically. **RQ2 result:** semantic embeddings recover the rule mapping only weakly (peak micro-F1 ≈ 0.14 vs the oracle's 1.0) and K-BERT ≈ BERT (1-hop graph injection gives no consistent lift) — frozen sentence embeddings capture grammatical *construction* weakly; the morphosyntactic rule mapper stays the stronger evidence source. Full write-up: `docs/phase6-validation.md`. *(= §3.4; prerequisite for Subturtle transcripts.)*

**Phase 7 — Personalization (KGT) + recommender handoff** · ✅ Done (on `feat/phase7-kgt-personalization`)
- A: **Knowledge Graph Tuning** (`klg_ai.kgt`) — closed-form per-learner edge multipliers from the learner's own feedback (consistent contradiction weakens/removes the inference message it falsifies; agreement reinforces), applied pre-normalization so both propagation guarantees survive. RQ5 ablation on the Phase-5 harness (`--kgt`): **engine+KGT = per-learner gradient retrain = global engine at AUROC 0.6406, but at 4.0 vs 363.4 ms/learner — ×90 cheaper** (null robust across a knob sweep; KGT fires on 500/500 learners). Knowledge-state API documented for the recommender: `docs/knowledge-state-api.md`; `gap_scores` (§3.7) computed on the status payload.
- B: Viewer **"Personal graph (KGT)"** toggle (green/dashed-red/dotted edges + per-edge evidence reasons in NodeDetails), live **retrain-process animation** (epoch scrubber + loss + wall-time verdict vs KGT one-shot), Validation-tab RQ5 card (parity + log-scale cost bars + convergence curve), chat KGT (wrong usage weakens edges live). New "struggling" demo learner carries the contradiction story.
- **Deliverable:** adapted personal graphs + a documented API boundary for the delegated recommender. Full write-up: `docs/phase7-validation.md`; committed artifact `docs/phase7/results.json`.

### Stage V — Ship

**Phase 8 — Subturtle integration** · ⬜ Todo
- A: Subturtle adapter — `leitner_review_log` (once added), phrase/chunk/`linguistic_data` evidence, dialog turns → `Event` schema → activation.
- B: Embed/port the visualization into the Subturtle dashboard (or consume via API).
- **Deliverable:** real Subturtle learners on the real map. *(Post-thesis productization.)*

---

## RQ → Phase coverage

| RQ | Phase |
|---|---|
| RQ1 — graph/hybrid vs sequential | 5 |
| RQ2 — BERT vs K-BERT | 6 |
| RQ3 — GNN propagation for gaps | 3, 4, 5 |
| RQ4 — forgetting / dynamics | 4, 5 |
| RQ5 — KGT personalization vs retrain | 7 ✓ |

## Dependencies & risks

- **Static map quality (P1)** gates everything. EGP isn't a clean download — v0 is a curated seed, expanded toward full EGP coverage incrementally.
- **C2 validation is data-starved** — open KT data thins above ~C1, so C2 is structurally demonstrated, empirically lighter. Subturtle's idiom/discourse/formality signals partly offset this for the upper levels.
- **Subturtle review history (P8)** depends on the `leitner_review_log` (see `docs/subturtle-review-event-log-spec.md`) — start logging early; history can't be backfilled.
- **Scope note:** paper §1.8 says "Intermediate English" — update to A1–C2.

## Immediate next step

All five RQs now have empirical answers (RQ1/RQ3/RQ4 → Phase 5, RQ2 → Phase 6, RQ5 → Phase 7). Next: **paper catch-up** (translate the phase write-ups into the results chapters; fix §1.8) and **Phase 8** — Subturtle integration (start the `leitner_review_log` early — history can't be backfilled; spec in `docs/subturtle-review-event-log-spec.md`). Housekeeping: merge `feat/phase7-kgt-personalization` → `dev`. Open items carried forward: train the K-BERT injection / propagation weights instead of fixing them (RQ2/RQ3), richer node text + fuller EGP coverage to lift semantic-mapping F1, optional Gemini-assisted turn tagging, and the EdNet scale-check.
