<script setup lang="ts">
import { computed, onMounted } from "vue";
import { storeToRefs } from "pinia";
import { useMetricsStore } from "../stores/metrics";
import type { ModelResult, RocPoint } from "../types";

const metricsStore = useMetricsStore();
const { data, loading, error } = storeToRefs(metricsStore);

onMounted(() => metricsStore.load());

// Per-model colors: greens = engine variants, blue = sequence (DKT),
// amber/orange = simple baselines, grey = chance.
const COLOR: Record<string, string> = {
  engine_full: "#2e7d32",
  engine_no_prop: "#66bb6a",
  engine_no_forget: "#43a047",
  engine_neither: "#a5d6a7",
  dkt: "#0288d1",
  per_skill_mean: "#f9a825",
  per_user_mean: "#fb8c00",
  global_mean: "#90a4ae",
  engine_kgt: "#7b1fa2",
  engine_retrain: "#ad1457",
};

const models = computed(() => data.value?.models ?? []);
function model(name: string): ModelResult | undefined {
  return models.value.find((m) => m.name === name);
}
function fmt(x: number | undefined | null, d = 3): string {
  return x === undefined || x === null ? "—" : x.toFixed(d);
}
// AUROC lives in a narrow band (~0.5–0.7); stretch that range so bars are legible.
function barWidth(auroc: number | undefined): string {
  if (auroc === undefined) return "0%";
  const lo = 0.5, hi = 0.72;
  return `${Math.max(0, Math.min(1, (auroc - lo) / (hi - lo))) * 100}%`;
}

// ROC overlay — a handful of representative models.
const ROC_MODELS = ["engine_full", "dkt", "per_skill_mean", "global_mean"];
const PLOT = 240;
const PAD = 30;
const SPAN = PLOT - PAD * 2;
function rocPoints(roc: RocPoint[]): string {
  // (0,0) bottom-left → (1,1) top-right within the padded plot box.
  return roc
    .map((p) => `${PAD + p.fpr * SPAN},${PLOT - PAD - p.tpr * SPAN}`)
    .join(" ");
}
const rocModels = computed(() =>
  ROC_MODELS.map(model).filter((m): m is ModelResult => !!m),
);

function color(name: string): string {
  return COLOR[name] ?? "#607d8b";
}

// --- RQ5 (Phase 7): personalization — fit AND compute cost -----------------
const RQ5_MODELS = ["engine_full", "engine_kgt", "engine_retrain"];
const hasRq5 = computed(() => !!model("engine_kgt"));
function msPerLearner(name: string): number | null {
  const s = model(name)?.cost?.seconds_per_learner;
  return s === undefined || s === null ? null : s * 1000;
}
// The headline RQ5 number: how many times the per-learner retrain costs vs KGT.
const rq5CostRatio = computed(() => {
  const kgt = msPerLearner("engine_kgt");
  const retrain = msPerLearner("engine_retrain");
  return kgt && retrain && kgt > 0 ? Math.round(retrain / kgt) : null;
});
// Cost bars on a log scale (costs span orders of magnitude).
const costBarWidth = computed(() => {
  const vals = RQ5_MODELS.map(msPerLearner).filter((v): v is number => !!v && v > 0);
  const max = Math.max(...vals, 1);
  return (name: string) => {
    const v = msPerLearner(name);
    if (!v || v <= 0) return "0%";
    const lo = Math.log10(0.01); // 0.01 ms floor
    return `${Math.max(0.04, (Math.log10(v) - lo) / (Math.log10(max) - lo)) * 100}%`;
  };
});
// Retrain convergence plot (mean train loss per epoch).
const retrainCurve = computed(() => model("engine_retrain")?.retrain_curve ?? null);
const CURVE_W = 240;
const CURVE_H = 90;
const CURVE_PAD = 8;
const curvePoints = computed(() => {
  const c = retrainCurve.value;
  if (!c || c.length < 2) return "";
  const losses = c.map((p) => p.loss);
  const lo = Math.min(...losses);
  const hi = Math.max(...losses);
  const span = hi - lo || 1;
  return c
    .map((p, i) => {
      const x = CURVE_PAD + (i / (c.length - 1)) * (CURVE_W - 2 * CURVE_PAD);
      // SVG y grows downward: highest loss at the top, the fit descends.
      const y = CURVE_PAD + ((hi - p.loss) / span) * (CURVE_H - 2 * CURVE_PAD);
      return `${x},${y}`;
    })
    .join(" ");
});
</script>

<template>
  <section class="dash">
    <div v-if="loading" class="msg">Loading validation results…</div>
    <div v-else-if="error" class="msg err">
      No validation results available ({{ error }}).<br />
      Generate them with
      <code>python -m klg_ai.eval.run --data data/slam --max-learners 500</code>.
    </div>

    <template v-else-if="data">
      <div class="head">
        <h2>{{ hasRq5 ? "Phases 5 & 7 — Validation on open data" : "Phase 5 — Validation on open data" }}</h2>
        <p class="summary">
          {{ data.dataset.source }} · <b>{{ data.dataset.course }}</b>/{{ data.dataset.split }} ·
          {{ data.dataset.n_learners.toLocaleString() }} learners ·
          {{ data.dataset.n_eval_instances.toLocaleString() }} held-out tokens
          ({{ data.dataset.n_cold_instances.toLocaleString() }} on cold nodes) ·
          mistake rate {{ fmt(data.dataset.mistake_base_rate) }} ·
          node coverage {{ fmt(data.dataset.node_coverage) }}
        </p>
        <p class="note">
          Next-step correctness prediction; AUROC is the headline metric (the
          14% mistake base rate makes accuracy/F1 uninformative). Higher AUROC is
          better; 0.5 = chance.
        </p>
      </div>

      <div class="cards">
        <!-- RQ1 -->
        <article class="card">
          <h3>RQ1 · Graph vs. sequence</h3>
          <div
            v-for="name in ['engine_full', 'dkt', 'per_skill_mean', 'global_mean']"
            :key="name"
            class="bar-row"
          >
            <span class="bar-label">{{ model(name)?.label }}</span>
            <span class="bar-track">
              <span
                class="bar-fill"
                :style="{ width: barWidth(model(name)?.metrics.auroc), background: color(name) }"
              />
            </span>
            <span class="bar-val">{{ fmt(model(name)?.metrics.auroc) }}</span>
          </div>
          <p class="cap">
            The structural graph engine is competitive with the neural sequence
            model (DKT) and clearly beats item-difficulty and chance.
          </p>
        </article>

        <!-- RQ3 -->
        <article class="card">
          <h3>RQ3 · GNN propagation</h3>
          <table class="mini">
            <thead>
              <tr><th></th><th>AUROC</th><th>cold</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>Propagation on</td>
                <td>{{ fmt(model("engine_full")?.metrics.auroc) }}</td>
                <td>{{ fmt(model("engine_full")?.metrics_cold?.auroc) }}</td>
              </tr>
              <tr>
                <td>Propagation off</td>
                <td>{{ fmt(model("engine_no_prop")?.metrics.auroc) }}</td>
                <td>{{ fmt(model("engine_no_prop")?.metrics_cold?.auroc) }}</td>
              </tr>
            </tbody>
          </table>
          <p class="cap">
            No change in predictive AUROC — but on <em>cold</em> nodes (never
            practiced) propagation supplies inferred mastery where “off” is flat
            zero. Its value is representational (the interior-gap overlay), not
            cold-item prediction.
          </p>
        </article>

        <!-- RQ4 -->
        <article class="card">
          <h3>RQ4 · Forgetting / decay</h3>
          <table class="mini">
            <thead>
              <tr><th></th><th>AUROC</th><th>logloss</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>Forgetting on</td>
                <td>{{ fmt(model("engine_full")?.metrics.auroc) }}</td>
                <td>{{ fmt(model("engine_full")?.metrics.avglogloss) }}</td>
              </tr>
              <tr>
                <td>Forgetting off</td>
                <td>{{ fmt(model("engine_no_forget")?.metrics.auroc) }}</td>
                <td>{{ fmt(model("engine_no_forget")?.metrics.avglogloss) }}</td>
              </tr>
            </tbody>
          </table>
          <p class="cap">
            Negligible over SLAM's ~30-day window — too short for decay to
            separate fresh from stale evidence.
          </p>
        </article>

        <!-- RQ5 (Phase 7) -->
        <article v-if="hasRq5" class="card">
          <h3>RQ5 · Personalization: KGT vs retrain</h3>
          <div v-for="name in RQ5_MODELS" :key="name" class="bar-row">
            <span class="bar-label">{{ model(name)?.label }}</span>
            <span class="bar-track">
              <span
                class="bar-fill"
                :style="{ width: barWidth(model(name)?.metrics.auroc), background: color(name) }"
              />
            </span>
            <span class="bar-val">{{ fmt(model(name)?.metrics.auroc) }}</span>
          </div>

          <div class="cost-head">Compute cost (ms / learner, log scale)</div>
          <div v-for="name in RQ5_MODELS" :key="name + '-cost'" class="bar-row">
            <span class="bar-label">{{ model(name)?.label }}</span>
            <span class="bar-track">
              <span
                class="bar-fill"
                :style="{ width: costBarWidth(name), background: color(name) }"
              />
            </span>
            <span class="bar-val">{{ msPerLearner(name) === null ? "—" : msPerLearner(name)!.toFixed(1) }}</span>
          </div>

          <template v-if="curvePoints">
            <div class="cost-head">Retrain convergence (mean train loss / epoch)</div>
            <svg :viewBox="`0 0 ${CURVE_W} ${CURVE_H}`" class="curve-svg">
              <polyline :points="curvePoints" fill="none" :stroke="color('engine_retrain')" stroke-width="2" />
            </svg>
            <p class="curve-note">KGT needs no epochs — one closed-form pass.</p>
          </template>

          <p class="cap">
            Same personalization space (per-edge multipliers), two fitting
            strategies: the closed-form KGT rule matches per-learner gradient
            retraining on predictive fit<template v-if="rq5CostRatio">
            at <b>1/{{ rq5CostRatio }}</b> of the compute</template> — and stays
            interpretable (every adjustment carries its evidence).
          </p>
        </article>

        <!-- ROC -->
        <article class="card roc">
          <h3>ROC curves</h3>
          <svg :viewBox="`0 0 ${PLOT} ${PLOT}`" class="roc-svg">
            <!-- axes -->
            <line :x1="PAD" :y1="PLOT - PAD" :x2="PLOT - PAD" :y2="PLOT - PAD" class="axis" />
            <line :x1="PAD" :y1="PLOT - PAD" :x2="PAD" :y2="PAD" class="axis" />
            <!-- chance diagonal -->
            <line :x1="PAD" :y1="PLOT - PAD" :x2="PLOT - PAD" :y2="PAD" class="diag" />
            <polyline
              v-for="m in rocModels"
              :key="m.name"
              :points="rocPoints(m.roc)"
              fill="none"
              :stroke="color(m.name)"
              stroke-width="2"
            />
            <text :x="PLOT / 2 - 12" :y="PLOT - 8" class="axlab">FPR →</text>
            <text :x="12" :y="PLOT / 2" class="axlab" :transform="`rotate(-90 12 ${PLOT / 2})`">TPR →</text>
          </svg>
          <ul class="legend">
            <li v-for="m in rocModels" :key="m.name">
              <span class="sw" :style="{ background: color(m.name) }" />
              {{ m.label }} ({{ fmt(m.metrics.auroc) }})
            </li>
          </ul>
        </article>
      </div>

      <!-- full table -->
      <table class="full">
        <thead>
          <tr>
            <th class="l">Model</th>
            <th>AUROC</th>
            <th>AUROC (cold)</th>
            <th>F1</th>
            <th>Accuracy</th>
            <th>Log-loss</th>
            <th v-if="hasRq5">ms/learner</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="m in models" :key="m.name" :class="{ hl: m.name === 'engine_full' }">
            <td class="l">
              <span class="dot" :style="{ background: color(m.name) }" />{{ m.label }}
            </td>
            <td><b>{{ fmt(m.metrics.auroc) }}</b></td>
            <td>{{ fmt(m.metrics_cold?.auroc) }}</td>
            <td>{{ fmt(m.metrics.F1) }}</td>
            <td>{{ fmt(m.metrics.accuracy) }}</td>
            <td>{{ fmt(m.metrics.avglogloss) }}</td>
            <td v-if="hasRq5">
              {{ m.cost ? (m.cost.seconds_per_learner * 1000).toFixed(1) : "—" }}
            </td>
          </tr>
        </tbody>
      </table>

      <p v-if="data.meta" class="foot">
        Generated {{ data.meta.generated_at }} · seed {{ data.meta.seed }} ·
        DKT epochs {{ data.meta.dkt_epochs }} · learner cap {{ data.meta.max_learners ?? "all" }}.
      </p>
    </template>
  </section>
</template>

<style scoped>
.dash { height: 100%; overflow-y: auto; padding: 18px 24px; }
.msg { padding: 24px; font-size: 14px; }
.msg.err { color: var(--gap); line-height: 1.7; }
.head h2 { font-size: 16px; margin: 0 0 6px; }
.summary { font-size: 13px; color: var(--ink); margin: 0 0 4px; }
.note { font-size: 12px; color: var(--muted); margin: 0 0 16px; max-width: 70ch; line-height: 1.5; }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 20px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
.card h3 { font-size: 13px; margin: 0 0 10px; font-weight: 600; }
.cap { font-size: 11px; color: var(--muted); margin: 10px 0 0; line-height: 1.5; }

.bar-row { display: grid; grid-template-columns: 1fr 90px 38px; align-items: center; gap: 8px; margin-bottom: 7px; }
.bar-label { font-size: 11px; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { background: #eceff1; border-radius: 4px; height: 10px; overflow: hidden; }
.bar-fill { display: block; height: 100%; border-radius: 4px; }
.bar-val { font-size: 11px; font-variant-numeric: tabular-nums; text-align: right; color: var(--ink); }

.mini { width: 100%; border-collapse: collapse; font-size: 12px; }
.mini th { font-weight: 500; color: var(--muted); text-align: right; padding: 2px 6px; }
.mini th:first-child { text-align: left; }
.mini td { padding: 3px 6px; text-align: right; font-variant-numeric: tabular-nums; }
.mini td:first-child { text-align: left; color: var(--ink); }

.cost-head { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin: 12px 0 6px; }
.curve-svg { width: 100%; max-width: 240px; display: block; background: #fafafa; border-radius: 4px; }
.curve-note { font-size: 10px; color: var(--muted); margin: 4px 0 0; }

.roc { grid-column: span 1; }
.roc-svg { width: 100%; max-width: 280px; display: block; }
.axis { stroke: #b0bec5; stroke-width: 1; }
.diag { stroke: #cfd8dc; stroke-width: 1; stroke-dasharray: 3 3; }
.axlab { font-size: 8px; fill: var(--muted); }
.legend { list-style: none; margin: 8px 0 0; padding: 0; font-size: 11px; color: var(--ink); }
.legend li { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
.sw { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }

.full { width: 100%; border-collapse: collapse; font-size: 12px; }
.full th { text-align: right; padding: 6px 10px; color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--line); }
.full th.l { text-align: left; }
.full td { text-align: right; padding: 6px 10px; font-variant-numeric: tabular-nums; border-bottom: 1px solid #eceff1; }
.full td.l { text-align: left; color: var(--ink); }
.full tr.hl { background: #f1f8e9; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; vertical-align: middle; }
.foot { font-size: 11px; color: var(--muted); margin-top: 14px; }
</style>
