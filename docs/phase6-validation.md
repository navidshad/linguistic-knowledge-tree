# Phase 6 — Semantic embedding (BERT vs K-BERT) + Gemini chat demo

Generalizes the Phase-5 **rule-based** morphosyntactic mapper into a **semantic
text→node mapper** so *unlabeled* free text can activate the syntax map — the
prerequisite for ingesting Subturtle's dialog turns. Two tracks: (A) the
quantitative **RQ2** study (BERT vs K-BERT, offline on SLAM), and (B) a live
**Gemini chat** demo where a learner's turns light up the map.

> **TL;DR.** Frozen sentence embeddings recover the hand-built rule mapping only
> **weakly** — micro-F1 ≈ **0.13** (macro ≈ 0.14) against the rule oracle's 1.0 — and
> the K-BERT knowledge-injection variant is **statistically indistinguishable** from
> plain BERT (1-hop prerequisite-graph injection gives no consistent lift). The
> grammatical *construction* signal the morphosyntactic rule mapper exploits is
> largely orthogonal to what a general-purpose bi-encoder captures, so the rule
> mapper remains the stronger evidence source. This mirrors Phase 5's RQ3 finding:
> the graph is **representational**, not a predictive lift. The chat demo, however,
> shows the pipeline works end-to-end — free dialog turns *do* activate plausible
> map regions in real time.

## The two tracks

```
Track A (RQ2, offline, reproducible)          Track B (live demo)
  SLAM token  ─embed in context─▶ cosine        learner turn ─embed─▶ cosine
     │  vs rule map_exercise (silver labels)        │  → dialog Event → engine
     ▼                                              ▼
  micro/macro P/R/F1, coverage, confusions       map lights up + node evidence
  BERT vs K-BERT vs rule (oracle)                Gemini tutor reply (mock-able)
```

## Data & method (Track A)

Same **Duolingo SLAM 2018 en_es** as Phase 5 (English learners, L1 Spanish;
license-gated, `data/` git-ignored). The **rule-based** `adapters.slam_mapping.map_exercise`
output is the **silver standard**: for each token we ask how close an *unsupervised*
embedding mapper gets using only the token's **text in context** (no morphosyntactic
tags). Granularity is **token-in-context per `instance_id`** (native to the token-keyed
gold): the token is embedded as its reconstructed sentence with the token span marked
(`"She [ is ] reading"`) — a bare token ("is") is semantically empty, the marked
sentence is not. Each `(token, node)` is a binary decision scored as micro / macro
(per-node) precision / recall / F1, plus node coverage, exact-set-match, and the top
node confusions.

- **Embedder:** `sentence-transformers/all-MiniLM-L6-v2` (dim 384), **CPU-pinned**
  for bit-stable vectors, lazy-imported (optional `[semantic]` extra). A dependency-free
  `HashingEmbedder` (forced by `KLG_EMBEDDER=hash`) keeps the unit tests offline.
- **Node vectors** (`docs/phase6/node_vectors.npz`, committed): each node embedded from
  `"<label>. <description> (category: <cat>)"`.
- **Selection:** cosine ≥ threshold, top-k (the operating point below is the dev-tuned
  threshold 0.25, top-k 3; reported, not optimized on test).

### The K-BERT mechanism (the RQ2 ablation)

The single variable the ablation toggles is **which node matrix the mapper dots against**:

- **BERT** — each node's vanilla embedding (`matrix`).
- **K-BERT** — each node's vector blended with an embedding of its **1-hop
  prerequisite-graph neighbours** (parent prerequisites + dependent children labels),
  `0.25·embed(neighbours) + 0.75·embed(own)`, re-normalized (`matrix_kbert`). This is the
  bi-encoder analogue of K-BERT's "inject KG triples into the text representation" — here
  the triples are prerequisite edges. The evidence text and matching code are identical
  across arms, so any difference is attributable to the injection alone.

**Faithfulness caveats** (vs Liu et al. 2020): injection is **post-hoc text fusion** on a
**frozen** encoder, not attention-level soft-position/visible-matrix integration; only
1-hop, prerequisite-typed edges; only the *node* side is injected (the evidence side is
plain BERT). It is "K-BERT-inspired," not K-BERT proper.

## Results (RQ2)

en_es / dev, 400-learner seeded sample (seed 0), threshold 0.25, top-k 3
(`docs/phase6/mapping_results.json`). Micro = over all (token, node) pairs; the **rule**
row is the oracle (it *is* the gold).

| Mapper | micro-P | micro-R | micro-F1 | macro-F1 | coverage | exact-set |
|---|---:|---:|---:|---:|---:|---:|
| **rule** (oracle) | 1.000 | 1.000 | **1.000** | 1.000 | 0.724 | 1.000 |
| BERT | 0.100 | 0.201 | **0.134** | 0.144 | 0.668 | 0.052 |
| K-BERT | 0.093 | 0.203 | 0.127 | 0.141 | 0.677 | 0.035 |

(61,398 tokens; macro = unweighted mean over the 35 nodes with gold support. "exact-set" =
fraction of gold-nonempty tokens whose predicted node set equals the gold set.)

### Reading RQ2

- **BERT vs K-BERT** — within noise of each other: BERT micro-F1 **0.134** vs K-BERT
  **0.127** at threshold 0.25 (BERT marginally ahead; K-BERT trades a little precision for a
  little recall/coverage), macro-F1 0.144 vs 0.141. A threshold sweep flips the sign — on a
  50-learner sample K-BERT edges ahead only in the high-precision regime (0.085 vs 0.075 at
  0.35). **Net: 1-hop prerequisite-graph injection gives no consistent improvement.** The
  prerequisite structure is a *learning-order* signal, largely orthogonal to the surface
  semantics a frozen bi-encoder encodes, so fusing neighbour text neither reliably sharpens
  nor blurs the match.
- **Semantic vs rule** — both embedding arms sit far below the rule oracle (F1 1.0). A
  morphosyntactic tag → node rule reads *grammatical construction* (tense, aspect, voice,
  POS) directly; MiniLM sentence similarity is dominated by lexical/topical content, so it
  recovers **lexically distinctive** concepts reasonably (per-node F1 ≈ 0.4–0.5 for
  `articles_the`/`articles_a_an`, `adverbs_frequency`, `possessive_pronouns`, `can_ability`)
  but systematically confuses construction-level ones — most `present_simple` tokens are
  mis-mapped to pronoun/structural nodes (`personal_pronouns`, `passive_present_past`,
  `reported_questions`; see `top_confusions` in the JSON). The honest conclusion: **for
  evidence→node mapping the auditable rule mapper is the stronger source; semantic
  embeddings are a *complement* for genuinely unlabeled free text, not a replacement.**
- **Extrinsic** — swapping the embedding-derived tags into the Phase-5 engine harness
  (additive `--mapper kbert`, threshold 0.25) is a **positive** confirmatory result: at this
  threshold the K-BERT mapper's eval coverage (0.73) is comparable to rule's (0.72), and the
  engine predicts next-step correctness at **AUROC 0.670** on a 40-learner dev sample —
  competitive with the Phase-5 rule-based engine (0.641, larger sample). So although the
  per-token mapping is noisy (F1 ≈ 0.13), that noise **largely averages out at the
  prediction level**: the tags still carry enough per-node/per-learner difficulty signal for
  the engine. Propagation/forgetting stay null and per-user ability leads (0.723), exactly as
  in Phase 5. (40 learners — illustrative, not the headline; the committed Phase-5 rule
  numbers in `docs/phase5/results.json` are never touched.)

This is a useful *negative* result for the thesis: it quantifies exactly how much a
general-purpose embedding recovers of a curriculum-grounded grammatical mapping, and shows
the graph's value here (as in Phase 5) is representational rather than a predictive lift.

## The chat demo (Track B / 6-B)

`POST /api/chat` is stateless (mirrors `POST /api/status`): the client sends the
conversation + current known-set; the server maps each **learner** turn to nodes via the
validated mapper, emits them as `dialog`-source `Event`s, folds them into point-in-time
mastery, and returns the tutor reply + recomputed status/mastery + the per-node text
evidence. The tutor reply comes from **Gemini** (minimal REST, `gemini-flash-latest`, thinking
disabled so the short reply isn't starved of tokens; override via `GEMINI_MODEL`) or a
**deterministic mock** (`KLG_GEMINI_MOCK=1` / no key) — the reply never blocks the
thesis-relevant part (mapping the turn), so the map lights up even if Gemini is down. Only
the learner's production is evidence (review > **dialog** > exposure).

The viewer adds a **Chat** tab: type in English and the map activates live; click a node
and NodeDetails shows the **chat turns that activated it** (the 6-B "text evidence"). A
whole turn embeds to a "mixed" point, so the chat uses a lower threshold
(`KLG_CHAT_THRESHOLD`, default 0.22) than the RQ2 eval. Example: *"The blue car is bigger
than mine"* → **comparatives** (known); *"If it rains I will stay home"* →
zero/first conditional.

## Reproduce

```bash
source backend/.venv/bin/activate
pip install -e "backend/ai[semantic]"                      # MiniLM (one-time ~80MB model download)
pytest backend/ai backend/server -q                        # 98 tests (KLG_EMBEDDER=hash → offline)

python -m klg_ai.semantic.build_vectors                    # rebuild docs/phase6/node_vectors.npz
python -m klg_ai.eval.mapping_run --course en_es --split dev \
    --max-learners 400 --threshold 0.25 --out docs/phase6/mapping_results.json   # intrinsic RQ2
KLG_SEMANTIC_THRESHOLD=0.25 python -m klg_ai.eval.run --course en_es --split dev \
    --max-learners 200 --mapper kbert --out docs/phase6/results_kbert.json        # extrinsic (optional)

# live chat (no key needed): KLG_GEMINI_MOCK=1 uvicorn klg_server.main:app --port 8000
#   then the frontend Chat tab; set GEMINI_API_KEY for a real tutor.
```

## Files

| Path | Role |
|---|---|
| `backend/ai/klg_ai/semantic/{embedder,node_vectors,mapper,build_vectors}.py` | BERT/K-BERT mapper + committed-artifact builder |
| `backend/ai/klg_ai/eval/{mapping_eval,mapping_run}.py` | intrinsic RQ2 (vs rule silver labels) |
| `backend/ai/klg_ai/eval/{dataset,ablations,run}.py` | extrinsic RQ2 via additive `mapper={rule,bert,kbert}` |
| `backend/server/klg_server/{gemini.py, routers/chat.py}` | Gemini client (+ mock) and `POST /api/chat` |
| `frontend/src/{components/ChatPanel.vue, stores/chat.ts}` + `App.vue`, `NodeDetails.vue` | Chat tab + live overlay + evidence panel |
| `docs/phase6/node_vectors.npz` | committed MiniLM node vectors (BERT + K-BERT) |
| `docs/phase6/mapping_results.json` | committed RQ2 results |
| `backend/{ai,server}/tests/test_semantic_*.py`, `test_mapping_eval.py`, `test_chat.py` | tests (deterministic, offline) |

## Limitations / notes

- The map is a 113-node v0 seed with terse descriptions; richer node text + fuller EGP
  coverage would raise semantic-mapping F1 and give the K-injection more to work with.
- K-BERT here is the frozen-bi-encoder analogue (post-hoc 1-hop fusion), not the original
  attention-level architecture — and the injection weights are **fixed/untrained**.
  Training the injection (and the GNN propagation, RQ3) is the natural next step.
- The chat mapping is approximate by construction (it inherits the weak BERT mapping); it
  demonstrates the *pipeline* (free text → live activation + evidence), not high-precision
  tagging. Gemini-assisted turn tagging is an easy future toggle.
- SLAM is a beginner window, so the A1–B1 core dominates both the gold and the predictions;
  upper-CEFR mapping is demonstrated structurally, thin empirically (as in Phase 5).
