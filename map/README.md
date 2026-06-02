# Static English Syntax Map

The fixed, global concept graph the whole system is built on. **Same map for every learner** — a learner's data only *activates* nodes on it; it never changes the structure.

- `english-syntax-map.v0.json` — the map (v0 seed, ~113 nodes, A1–C2).
- `validate.py` — integrity checker (run it after any edit).

## Schema

```jsonc
{
  "meta": { "version", "title", "source", "scope", "edgeSemantics" },
  "categories": [ { "id", "label" } ],
  "nodes": [
    {
      "id":          "present_perfect_simple",   // unique, snake_case
      "label":       "Present Perfect Simple",   // display name
      "category":    "tense_aspect",             // must match a categories[].id
      "cefr":        "A2",                        // A1|A2|B1|B2|C1|C2
      "description": "Past with present relevance."
    }
  ],
  "edges": [
    { "from": "past_simple", "to": "present_perfect_simple", "type": "prerequisite" }
  ]
}
```

## Edge meaning

An edge `from -> to` means **`from` is a prerequisite of `to`** (learning-order). The prerequisite graph must be a **DAG** (no cycles) — `validate.py` enforces this. Edges may cross categories (e.g. `present_perfect_simple -> passive_present_past`); cross-category edges are what make "show the nodes in between two distant activated nodes" meaningful.

## How node status is derived (computed downstream, not stored here)

Given a learner's **activated** node set `K`:

| Status | Rule |
|---|---|
| **known** | node ∈ K |
| **interior-gap** | not in K, has a known ancestor **and** a known descendant (skipped between two known nodes) |
| **frontier** | not in K, all direct prerequisites known, no known descendant (immediately learnable) |
| **further** | everything else not in K |

The map itself carries no learner data — status is a function of `(map + activated set)`.

## Validate

```bash
python3 packages/map/validate.py
```

## Roadmap note

v0 is a **curated seed**, not full English Grammar Profile coverage (~1,200 points). Expand incrementally — ideally by ingesting an EGP export. Keep edits valid (`validate.py`) and bump `meta.version`.
