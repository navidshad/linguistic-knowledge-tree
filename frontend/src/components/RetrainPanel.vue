<script setup lang="ts">
// RQ5 demo (Phase 7): run the per-learner gradient retrain live and replay it
// epoch by epoch — edges re-weight on the map as the loss descends — then show
// the wall-time verdict against KGT's one-shot closed-form tuning.
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useRetrainStore } from "../stores/retrain";
import { useLearnerStore } from "../stores/learner";

const retrain = useRetrainStore();
const learner = useLearnerStore();
const { result, index, playing, engaged, loading, error, epochs, costRatio } = storeToRefs(retrain);

const currentLoss = computed(() => epochs.value[index.value]?.loss);

function onScrub(e: Event) {
  retrain.pause();
  retrain.setIndex(Number((e.target as HTMLInputElement).value));
}
</script>

<template>
  <div class="retrain">
    <button class="run" :disabled="loading" @click="retrain.run(learner.learnerId)">
      {{ loading ? "Fitting…" : result ? "Retrain again" : "Retrain on this learner" }}
    </button>
    <p v-if="error" class="hint err">{{ error }}</p>

    <template v-if="result && engaged">
      <div class="row">
        <button class="play" @click="playing ? retrain.pause() : retrain.play()">
          {{ playing ? "❚❚" : "▶" }}
        </button>
        <input
          class="scrub"
          type="range"
          min="0"
          :max="epochs.length - 1"
          :value="index"
          @input="onScrub"
        />
        <span class="epoch">ep {{ index + 1 }}/{{ epochs.length }}</span>
      </div>
      <p class="loss">train loss <b>{{ currentLoss?.toFixed(4) }}</b></p>
      <p class="verdict">
        Gradient retrain: <b>{{ result.wall_ms.toFixed(0) }} ms</b> ·
        KGT one-shot: <b>{{ result.kgt_wall_ms.toFixed(1) }} ms</b>
        <template v-if="costRatio"> — <b>×{{ costRatio }}</b> the compute for the same personalization space.</template>
      </p>
      <p class="hint">Scrub to watch the fit converge: edges re-weight each epoch.</p>
      <button class="back" @click="retrain.disengage()">Show KGT result instead</button>
    </template>
    <button v-else-if="result" class="back" @click="retrain.engaged = true">
      Show the retrain replay again
    </button>
  </div>
</template>

<style scoped>
.retrain { margin-top: 4px; }
.run { width: 100%; padding: 6px 10px; font-size: 12px; border: 1px solid var(--line); border-radius: 6px; background: #fff; cursor: pointer; }
.run:hover:enabled { background: #f5f5f5; }
.run:disabled { opacity: 0.6; cursor: default; }
.row { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
.play { width: 30px; height: 28px; flex: none; border: 1px solid var(--line); border-radius: 6px; background: #fff; cursor: pointer; font-size: 11px; }
.play:hover { background: #f5f5f5; }
.scrub { flex: 1; min-width: 0; }
.epoch { font-size: 11px; font-weight: 600; white-space: nowrap; }
.loss { font-size: 12px; margin: 6px 0 0; font-variant-numeric: tabular-nums; }
.verdict { font-size: 12px; line-height: 1.5; margin: 6px 0 0; }
.hint { font-size: 12px; color: var(--muted); line-height: 1.5; margin: 6px 0 0; }
.hint.err { color: var(--gap, #c62828); }
.back { margin-top: 8px; padding: 4px 8px; font-size: 11px; border: 1px solid var(--line); border-radius: 6px; background: #fff; cursor: pointer; color: var(--muted); }
.back:hover { background: #f5f5f5; }
</style>
