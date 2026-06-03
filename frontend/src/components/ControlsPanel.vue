<script setup lang="ts">
import { computed } from "vue";
import { useViewStore } from "../stores/view";
import { useLearnerStore } from "../stores/learner";
import { CEFR_COLOR, CEFR_ORDER, STATUS_COLOR } from "../constants";
import type { LayoutName } from "../types";

const view = useViewStore();
const learner = useLearnerStore();

const LAYOUTS: { value: LayoutName; label: string }[] = [
  { value: "matrix", label: "Matrix (category × CEFR)" },
  { value: "concentric", label: "Concentric (A1 → C2)" },
  { value: "force", label: "Force-directed" },
];

function onLayout(e: Event) {
  view.setLayout((e.target as HTMLSelectElement).value as LayoutName);
}
function onOverlay(e: Event) {
  view.overlayOn = (e.target as HTMLInputElement).checked;
}
function onSubgraph(e: Event) {
  view.subgraphOnly = (e.target as HTMLInputElement).checked;
}
function onLearner(e: Event) {
  learner.load((e.target as HTMLSelectElement).value);
}
const selectedDescription = computed(
  () => learner.learners.find((l) => l.id === learner.learnerId)?.description ?? "",
);
</script>

<template>
  <aside class="panel">
    <h2>Learner</h2>
    <select class="select" autocomplete="off" :value="learner.learnerId" @change="onLearner">
      <option v-for="l in learner.learners" :key="l.id" :value="l.id">{{ l.label }}</option>
    </select>
    <p class="hint">{{ selectedDescription }}</p>

    <h2>Layout</h2>
    <select class="select" autocomplete="off" :value="view.layout" @change="onLayout">
      <option v-for="o in LAYOUTS" :key="o.value" :value="o.value">{{ o.label }}</option>
    </select>

    <h2>Language levels</h2>
    <div class="cefr">
      <button
        v-for="l in CEFR_ORDER"
        :key="l"
        class="chip"
        :class="{ off: !view.enabledLevels.has(l) }"
        :style="{ borderColor: CEFR_COLOR[l] }"
        @click="view.toggleLevel(l)"
      >
        {{ l }}
      </button>
    </div>
    <p class="hint">Click a level to show/hide its nodes.</p>

    <h2>Learner overlay</h2>
    <label class="toggle">
      <input type="checkbox" autocomplete="off" :checked="view.overlayOn" @change="onOverlay" />
      Show learner status
    </label>
    <label class="toggle" :class="{ disabled: !view.overlayOn }">
      <input type="checkbox" autocomplete="off" :checked="view.subgraphOnly" :disabled="!view.overlayOn" @change="onSubgraph" />
      Only relevant subgraph
    </label>
    <p class="hint">Right-click a node to mark it known / not known.</p>
    <button class="reset" @click="learner.load(learner.learnerId)">Reset what-if edits</button>

    <h2>Status legend</h2>
    <div class="row"><span class="sw" :style="{ background: STATUS_COLOR.known }" /> Known</div>
    <div class="row"><span class="sw" :style="{ background: STATUS_COLOR.interior_gap }" /> Interior gap</div>
    <div class="row"><span class="sw" :style="{ background: STATUS_COLOR.frontier }" /> Frontier</div>
    <div class="row"><span class="sw" :style="{ background: STATUS_COLOR.further }" /> Further</div>
  </aside>
</template>

<style scoped>
.panel { background: var(--panel); border-right: 1px solid var(--line); padding: 14px; overflow-y: auto; font-size: 13px; }
h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 16px 0 8px; }
h2:first-child { margin-top: 0; }
.select { width: 100%; padding: 6px; border: 1px solid var(--line); border-radius: 6px; font-size: 13px; background: #fff; }
.cefr { display: flex; gap: 5px; flex-wrap: wrap; }
.chip { font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 5px; border: 3px solid; background: #fff; color: var(--ink); cursor: pointer; }
.chip.off { opacity: 0.35; background: #f5f5f5; text-decoration: line-through; }
.toggle { display: flex; align-items: center; gap: 8px; margin: 6px 0; cursor: pointer; }
.toggle.disabled { opacity: 0.45; cursor: default; }
.hint { font-size: 12px; color: var(--muted); line-height: 1.5; margin: 6px 0; }
.reset { margin-top: 8px; padding: 6px 10px; font-size: 12px; border: 1px solid var(--line); border-radius: 6px; background: #fff; cursor: pointer; }
.reset:hover { background: #f5f5f5; }
.row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
.sw { width: 14px; height: 14px; border-radius: 4px; flex: none; }
</style>
