# Phase 5 — Validation on open data (Duolingo SLAM)

Turns the thesis from *theoretical* to *validated*: runs the activation engine as
a knowledge-tracing predictor on real second-language data and scores it against
a sequence-model baseline (DKT) and simple priors, with the RQ ablations.

> **TL;DR.** On Duolingo SLAM **en_es** (English learners, L1 Spanish), the
> structural graph engine predicts next-step correctness at **AUROC ≈ 0.64**,
> competitive with an in-repo **DKT** (0.66) and clearly above item-difficulty
> (0.61) and chance (0.50) — using only a 113-node syntax map and *untrained*
> propagation weights. Propagation’s contribution is **representational** (it
> fills in mastery for evidence-poor nodes — the interior-gap overlay) rather
> than a lift in predictive AUROC; forgetting is negligible on the 30-day window.

## The data

The 2018 Duolingo Shared Task on Second Language Acquisition Modeling (SLAM):
per-token graded exercises from learners' first ~30 days. We use the **en_es**
track — learners *learning English* (so the tokens are English and carry the
morphosyntactic annotation we map onto the syntax map). `es_en` / `fr_en` are
Spanish / French tokens and are ignored.

| | learners | exercises | token instances | mistake rate |
|---|---:|---:|---:|---:|
| en_es train | 2,593 | 824,012 | 2,622,957 | — |
| en_es dev | — | — | 387,374 | 0.143 |

Each token line gives a Universal POS tag, a fine-grained Penn tag (`fPOS`), UD
morphological features, and a dependency relation + head. **Label 1 = the learner
made a mistake**, 0 = correct (so a learner `Event` is correct iff the label is
0). `days` (float, larger = newer) maps onto the engine's day clock.

**Getting it.** The tarballs are gated behind a Harvard Dataverse guestbook (DOI
`10.7910/DVN/8SWHNO`). They are **not** committed (large + license-gated; `data/`
is git-ignored). To fetch with a Dataverse API token:

```bash
# POST a guestbook response -> signed URL -> download (per file id)
curl -s -X POST -H "X-Dataverse-key: $DV" -H 'Content-Type: application/json' \
  -d '{"guestbookResponse":{"name":"…","email":"…","institution":"…","answers":[]}}' \
  "https://dataverse.harvard.edu/api/access/datafile/3357629?signed=true"   # en_es
# then GET the returned signedUrl, extract into data/slam/
```

File ids: en_es `3357629`, es_en `3357630`, fr_en `3357627`, starter_code `3357628`.

## Method

```
SLAM tokens ─▶ adapter (parse + tag→node) ─▶ per-learner Events
            ─▶ temporal split (train → dev)
            ─▶ predictor (engine | DKT | priors) ─▶ P(mistake) per held-out token
            ─▶ SLAM metrics (AUROC / F1 / acc / logloss) + cold-node slice
```

**Adapter** (`klg_ai/adapters/slam.py`, `slam_mapping.py`). Parsing mirrors the
official `baseline.py` exactly. The tag→node mapper has two layers:

- *Lexical / token-local* — articles, pronouns, plural nouns, comparatives,
  modals, prepositions, linkers, negation, wh-words … read off one token's
  POS + features + surface form.
- *Verb constructions* — read the dependency tree: *be* + V-ing → (present/past)
  continuous, *have* + V-en → (present/past) perfect, *be* + V-en → passive,
  modal + V → the modal's concept, copular *be* → present/past simple. The node
  is attributed to the main verb **and** its auxiliaries.

A token can map to several nodes; function words / proper nouns that carry no
grammar signal map to nothing. Every emitted id is filtered against the live map.
Coverage on en_es dev: **72.5%** of token instances map to ≥1 of 35/113 nodes
(the A1–B1 core, as expected for beginners).

**Eval** (`klg_ai/eval/`). The SLAM split is temporal per learner, so predicting
dev is genuine next-step prediction. The **engine predictor** folds each
learner's train events into point-in-time mastery (forgetting + GNN propagation,
both toggleable), reads each dev token's mapped-node mastery *causally* at the
learner's first dev day, and turns mastery into a probability via a global Platt
(1-D logistic) calibration fit on train. AUROC is rank-based, so the calibration
leaves it unchanged — it only makes log-loss/accuracy meaningful. Unmapped tokens
back off to the global base rate. Metrics are ported from SLAM `eval.py` (pinned
to its self-test: acc .700, logloss .613, auroc .740, F1 .667).

**Baselines.** In-repo **DKT** (LSTM over the same node-tagged interaction
sequence — order only, no graph; RQ1), **per-skill mean** (item difficulty),
**per-user mean** (learner ability), **global mean** (chance).

**Cold-node slice.** Eval tokens on nodes the learner *never practiced* — the
sharp test for propagation, since only the graph can say anything there.

## Results

en_es / dev, 500-learner seeded sample (`docs/phase5/results.json`, seed 0):

| Model | AUROC | AUROC (cold) | F1 | Acc | Log-loss |
|---|---:|---:|---:|---:|---:|
| DKT (sequence) | **0.656** | 0.620 | 0.037 | 0.860 | 0.387 |
| Per-user mean (ability) | 0.649 | 0.657 | 0.030 | 0.861 | 0.388 |
| **Engine (graph + forgetting)** | **0.641** | 0.500 | 0.000 | 0.860 | 0.400 |
| Engine − propagation | 0.641 | 0.500 | 0.000 | 0.860 | 0.401 |
| Engine − forgetting | 0.640 | 0.500 | 0.000 | 0.860 | 0.400 |
| Per-skill mean (difficulty) | 0.605 | 0.676 | 0.000 | 0.860 | 0.399 |
| Global mean | 0.500 | 0.500 | 0.000 | 0.860 | 0.407 |

(F1/accuracy are uninformative at a 14% mistake base rate — almost nothing is
predicted ≥0.5 P(mistake); AUROC and log-loss are the meaningful metrics. The
SLAM leaderboard's top systems reach ~0.82 AUROC using rich *lexical* features
— token identity, user, etc. — which this deliberately structural model omits.)

### Reading the research questions

- **RQ1 — graph vs. sequence.** The structural engine (0.641) is competitive with
  the neural sequence model DKT (0.656) and well above difficulty (0.605) and
  chance (0.500). A coarse, *untrained* prerequisite graph carries most of the
  signal a same-skill LSTM extracts from order — encouraging for a representation
  whose real job is visualization, not leaderboard AUROC.
- **RQ3 — propagation.** No change in predictive AUROC (0.641 vs 0.641; cold
  0.500 vs 0.500). This is expected and explainable: dev tokens almost always
  land on nodes the learner already practiced, where direct evidence dominates
  and propagation (which only lifts evidence-poor nodes) cannot move the
  prediction. Its value is **representational**: on cold nodes it supplies
  inferred mastery for **76%** of cases (e.g. an unseen *present perfect* lifted
  to ~0.29 from practiced prerequisites), where “off” is a flat zero — exactly
  the interior-gap “glow” the viewer shows. Propagation helps you *see* a plausible
  knowledge state, not predict first-attempt correctness (which is driven by item
  difficulty / learner ability — note per-skill/per-user lead on the cold slice).
- **RQ4 — forgetting.** Negligible (0.641 vs 0.640): SLAM's ~30-day window is too
  short for exponential decay (30-day half-life) to separate fresh from stale
  evidence. The mechanism is validated structurally (the Phase 4-B scrubber) and
  is expected to matter on longer Subturtle histories.

## Reproduce

```bash
source backend/.venv/bin/activate
pytest backend/ai backend/server -q                       # 75 tests, incl. SLAM oracle
python -m klg_ai.eval.run --data data/slam --course en_es --split dev \
    --max-learners 500 --dkt-epochs 10 --out data/eval/results.json
cp data/eval/results.json docs/phase5/results.json        # refresh the committed artifact
```

`--max-learners 0` runs all 2,593 learners (large). The viewer's **Validation**
tab reads `GET /api/metrics` (served from `data/eval/results.json` if present,
else the committed `docs/phase5/results.json`).

## Files

| Path | Role |
|---|---|
| `backend/ai/klg_ai/adapters/slam.py` | SLAM parser + `Event` builder |
| `backend/ai/klg_ai/adapters/slam_mapping.py` | morphosyntactic tag → concept node |
| `backend/ai/klg_ai/eval/{dataset,predict,baselines,metrics,ablations,run}.py` | eval harness |
| `backend/ai/tests/{test_slam_adapter,test_eval}.py` + `tests/fixtures/slam/` | tests + fixture |
| `backend/server/klg_server/routers/metrics.py` | `GET /api/metrics` |
| `frontend/src/components/MetricsDashboard.vue` | the Validation tab |
| `docs/phase5/results.json` | committed results artifact |

## Limitations / notes

- The map is a 113-node v0 seed; richer EGP coverage would raise mapping coverage
  above 72.5% and give the graph more to work with.
- C2 is data-thin in SLAM (beginner window) — structurally demonstrated,
  empirically light (per the thesis plan; Subturtle's idiom/discourse signals
  offset this later).
- Propagation weights are fixed/untrained here; the architecture supports
  learning them (a future extension) — the point of Phase 5 is the *validated
  comparison*, not a tuned model.
