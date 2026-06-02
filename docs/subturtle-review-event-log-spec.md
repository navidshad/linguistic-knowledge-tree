# Task Spec — Subturtle Review-Event Log (KT-readiness)

**Owner:** Navid · **Target repo:** `CodeBridger/learn-by-subtitle/subturtle-dashboard-app` · **Module:** `server/src/modules/leitner_box`
**Priority:** P0 (time-sensitive — every day without it is trajectory data that cannot be recovered)
**Effort:** ~half a day (one new collection + one emit point + tests)

---

## 1. Why

The thesis (Linguistic Knowledge Graph) needs to **track a learner's knowledge over time**. The standard way to validate a knowledge tracer is *next-step prediction*: given a learner's ordered history of attempts `(concept, correct, timestamp)`, predict the next outcome.

Subturtle's Leitner system currently stores only the **current snapshot** of each item:

```ts
// leitner_box/db.ts — LeitnerItem
{ phraseId, boxLevel, nextReviewDate, lastAttemptDate, consecutiveIncorrect }
```

Every review **mutates this in place** (`submitReview`, service.ts:168), so the *sequence of attempts is overwritten and lost*. We keep only the latest box level and the last attempt date — never the trajectory.

**Knowledge tracing needs the trajectory, not the snapshot.** This spec adds an append-only event log so that, by the time we clone the thesis repo to ship for Subturtle, we already have months of real review history to validate and personalize on. Snapshot data can be recomputed any time; **history, once not recorded, is gone forever.**

## 2. Goal / Scope

**In scope (P0):** Persist one immutable event per review attempt, capturing the box transition and correctness, in a dedicated collection.

**Out of scope (this task):** The ML model itself; dialog-turn and phrase-save events (see §7, later phase).

## 3. Data model — new collection `leitner_review_log`

Add to `server/src/config.ts`:

```ts
export const LEITNER_REVIEW_LOG_COLLECTION = "leitner_review_log";
```

Add to `server/src/modules/leitner_box/db.ts`:

```ts
export interface LeitnerReviewLog {
  userId: string;
  phraseId: string;        // cross-DB ref to `phrase`, same as LeitnerItem
  reviewedAt: Date;        // attempt time (== the `now` used in submitReview)
  isCorrect: boolean;      // the graded outcome
  boxBefore: number;       // box level BEFORE this review; 0 if first-ever review of the item
  boxAfter: number;        // box level AFTER this review
  consecutiveIncorrectAfter: number;
  intervalDays: number;    // scheduled interval applied (boxIntervals[boxAfter-1]); fuels forgetting analysis
  source: "scheduled" | "custom" | "manual" | "snapshot"; // how the review was triggered
  sessionId?: string;      // optional: groups attempts within one review session
}

const leitnerReviewLogSchema = new Schema<LeitnerReviewLog>(
  {
    userId: { type: String, required: true },
    phraseId: { type: String, ref: PHRASE_COLLECTION, required: true },
    reviewedAt: { type: Date, required: true },
    isCorrect: { type: Boolean, required: true },
    boxBefore: { type: Number, required: true },
    boxAfter: { type: Number, required: true },
    consecutiveIncorrectAfter: { type: Number, default: 0 },
    intervalDays: { type: Number },
    source: {
      type: String,
      enum: ["scheduled", "custom", "manual", "snapshot"],
      default: "scheduled",
    },
    sessionId: { type: String },
  },
  { timestamps: true }
);

// KT trajectory query: per-user history in time order
leitnerReviewLogSchema.index({ userId: 1, reviewedAt: 1 });
// Per-item history (forgetting curve per phrase)
leitnerReviewLogSchema.index({ userId: 1, phraseId: 1, reviewedAt: 1 });

export const leitnerReviewLogCollection = defineCollection({
  database: DATABASE_LEITNER,
  collection: LEITNER_REVIEW_LOG_COLLECTION,
  schema: leitnerReviewLogSchema,
  permissions: [
    // Read-only for the owner; writes happen server-side via the service only.
    new Permission({ accessType: "owner", read: true, write: false }),
    new Permission({ accessType: "admin", read: true, write: true }),
  ],
});

// remember to export it from module.exports alongside leitnerSystemCollection
module.exports = [leitnerSystemCollection, leitnerReviewLogCollection];
```

> `PHRASE_COLLECTION` is already imported context in the leitner module via config; add it to the import in `db.ts` if not present.

**Why a separate collection (not embedded in `leitner_system.items`):** the system doc already holds an unbounded `items[]` array; an append-only log grows without limit and must not bloat the hot document that's read on every `getDueItems`/`getStats` call.

## 4. Where to emit — `LeitnerService.submitReview()` (service.ts:168)

This is the **only** place reviews are graded. `addPhraseToBox` is a manual placement (not a graded attempt) and `getDueItems`/`getCustomReviewItems` are reads — do **not** log from those.

`submitReview` has two branches (new item vs existing item). Both compute `nextBox`, the new consecutive-incorrect count, and `now`. **Recommended:** lift those into shared variables and emit one log write after the box update. Sketch:

```ts
// inside submitReview, after the updateOne that persists the box transition:
const boxBefore = item ? item.boxLevel : 0;
const consecutiveIncorrectAfter = isCorrect
  ? 0
  : (item ? (item.consecutiveIncorrect || 0) + 1 : 1);

const logCol = await getCollection(DATABASE_LEITNER, LEITNER_REVIEW_LOG_COLLECTION);
await logCol.create({
  userId,
  phraseId,
  reviewedAt: now,
  isCorrect,
  boxBefore,
  boxAfter: nextBox,
  consecutiveIncorrectAfter,
  intervalDays: (system.settings.boxIntervals || [])[nextBox - 1],
  source: "scheduled", // thread through from the controller if you distinguish custom/manual sessions
});
```

> Note the new-item branch currently inlines `consecutiveIncorrect: isCorrect ? 0 : 1` and sets `nextBox = isCorrect ? 2 : 1`. Make sure `boxBefore = 0` and `consecutiveIncorrectAfter` are captured correctly there too. The cleanest fix is to unify both branches to set `nextBox` / `nextConsecutiveIncorrect` variables, then do a single update + single log emit.

The log write should **not** block or fail the review. Wrap it so a logging error is caught and reported, never bubbling into the user's request:

```ts
.catch((e) => console.error("[LeitnerService] review-log write failed:", e));
```

## 5. Optional one-time seed (gives existing users a starting point)

You cannot reconstruct lost history, but you can seed **one** `source: "snapshot"` row per current `LeitnerItem` from `(boxLevel, lastAttemptDate, consecutiveIncorrect)`. This bootstraps the graph for existing users. Mark them `source: "snapshot"` so the ML side can **exclude them from clean next-step evaluation** (they're a single derived point, not a real attempt). One-off script, run once.

## 6. Acceptance criteria

- [ ] New collection `leitner_review_log` registered and indexed.
- [ ] Every `submitReview` call appends exactly **one** log row; no other code path writes review logs.
- [ ] `boxBefore`/`boxAfter`/`isCorrect`/`consecutiveIncorrectAfter` exactly match the transition applied to `leitner_system.items`.
- [ ] First-ever review of a phrase logs `boxBefore = 0`.
- [ ] Log rows are **never** updated or deleted by any code path (append-only).
- [ ] A failing log write does not fail the review request.
- [ ] Tests in `leitner_box/__tests__` extended: assert row count increments and field values per branch (new item correct, new item wrong, existing item promote, existing item demote-once, existing item reset-to-1).

## 7. Later (separate tasks, not P0)

Same idea, two more event sources — they make the *productive-use* and *exposure* signals available to the model:

- **`phrase_save` (exposure):** emit on phrase create (hook into `phrase_bundle/triggers.ts`). Weak signal — learner encountered, didn't test.
- **`dialog_turn` (productive use):** from live-session dialog turns; requires the text→concept extraction layer to tag which chunks were used. Medium signal.

All three event types collapse to the same shape the thesis pipeline consumes: `(learner_id, concept_ids, correct, ts, source)` — so logging them now means the future Subturtle data adapter is a thin mapping, not a rewrite.
