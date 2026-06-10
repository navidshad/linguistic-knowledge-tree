# Knowledge-State API — the recommender handoff contract

**Audience:** the separate "what's-next" recommender system (thesis §3.7) and
the Subturtle integration (Phase 8). This system **represents, computes, and
visualizes** a learner's knowledge state; deciding *what to learn next* is
explicitly out of scope and delegated across this API boundary.

Everything below is stateless and a **pure function of
`(static map, learner events, config, now)`** — there is no stored per-learner
model. Two identical requests return identical responses.

## The map (shared, static)

`GET /api/map` → the global English-syntax DAG (same for every learner):
`nodes[{id, label, category, cefr, description}]`,
`edges[{source, target, type:"prerequisite"}]` — `source` is a prerequisite of
`target`. The map is versioned JSON (`map/english-syntax-map.v0.json`); node ids
are the stable keys every other payload uses.

## Per-learner knowledge state

`GET /api/learner/{id}/status[?kgt=1]` → the full contract in one response:

| Field | Meaning |
|---|---|
| `statuses` | node_id → `known` \| `interior_gap` \| `frontier` \| `further` (see below) |
| `counts` | status → node count |
| `mastery` | node_id → [0, 1], the continuous score behind the discrete status |
| `gap_scores` | node_id → (0, 1], pedagogical priority — **the recommender's ranking input** |
| `edge_adjustments` | only with `?kgt=1`: the learner's personal-graph deltas (below) |

**Statuses** are computed from `(graph, activated set)` where
`activated = {n : mastery[n] ≥ known_threshold (0.5)}`:

- `known` — mastery cleared the threshold.
- `interior_gap` — unactivated but *between* known nodes (a known ancestor AND
  a known descendant): something the learner skipped.
- `frontier` — unactivated, every direct prerequisite known: learnable now.
- `further` — everything else (fails prerequisite feasibility).

**Gap scores** (§3.7) rank the *actionable* candidate set — interior gaps and
frontier nodes only:

```
gap_score(n) = level_weight(cefr(n)) × (1 − mastery(n)),   level_weight: A1=1.0 … C2=1/6
```

Known nodes need no teaching and `further` nodes are pruned (prerequisites not
met), so they never appear. Higher = more urgent; foundational gaps outrank
advanced ones at equal mastery. A minimal recommender can simply sort by
`gap_score` and apply its own policy on top.

**Timeline:** `GET /api/learner/{id}/timeline?frames=N` returns the same state
sampled across the learner's history (mastery is evaluated causally at each
frame's reference time) — useful for progress/decay analytics.

**What-if:** `POST /api/status` with `{activated: [node_id, ...]}` computes
statuses for an arbitrary known-set. No evidence exists there, so `mastery`,
`gap_scores`, and `edge_adjustments` are `null`.

## Personal graph (KGT, thesis §3.8 / RQ5)

With `?kgt=1`, mastery is computed over the learner's **personalized** copy of
the graph: each prerequisite edge carries a `(factor_back, factor_fwd)`
multiplier derived in closed form from the learner's own feedback (consistent
contradiction weakens the inference an edge carries; factor ≈ 0 removes it,
> 1 reinforces it). `edge_adjustments` reports every meaningful deviation:

```json
{"source": "relative_pronouns", "target": "relative_clauses_defining",
 "factor_back": 0.16, "factor_fwd": 1.0, "kind": "weakened",
 "reason": "Knows 'Defining Relative Clauses' (4/4 correct …) but struggles with its prerequisite …"}
```

A recommender MAY use these as interpretable signals (e.g. a `removed` backward
edge says "do not infer the prerequisite from the dependent for this learner —
it likely needs explicit teaching"). `POST /api/learner/{id}/retrain` exists
for demonstration/inspection of the gradient alternative; it is not part of the
recommendation contract.

## Evidence ingestion (the other direction)

Learner data enters the engine as `Event`s (`klg_ai.events.Event`):

```
{learner_id, node_ids: (node_id, ...), correct: bool, ts: float days, source: "review"|"dialog"|"exposure"}
```

Source weights: graded SRS `review` (1.0) > productive `dialog` (0.6) > passive
`exposure` (0.3); evidence decays with a 30-day half-life. This is the schema
the Subturtle adapter (Phase 8, `leitner_review_log`) must emit.

## What the recommender must NOT assume

- **Sub-threshold mastery on evidence-free nodes is inference, not knowledge.**
  Graph propagation lifts such nodes at most to `inferred_ceiling` (0.45 <
  known threshold 0.5) — treat 0 < mastery < 0.45 on an unpracticed node as
  "the graph suspects partial readiness", never as observed skill.
- **Statuses are not stable over time.** Forgetting decays evidence; the same
  request a month later can demote `known` nodes. Re-query, don't cache.
- **`gap_scores` is a ranking, not a curriculum.** It deliberately encodes no
  policy (pacing, review-vs-new tradeoffs, motivation) — that is the
  recommender's job.
- **Edges are learning-order prerequisites, not strict logical implications** —
  and with `?kgt=1` they are per-learner: don't assume the same effective graph
  across learners.
