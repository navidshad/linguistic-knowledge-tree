// Live retrain animation (Phase 7, RQ5): runs the per-learner gradient
// comparator server-side and replays the fit epoch by epoch — edges visibly
// re-weight as the loss descends — next to KGT's one-shot wall time. Mirrors
// the timeline store's scrubber mechanics (precomputed frames, instant scrub).
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { api } from "../services/api";
import type { RetrainEpoch, RetrainResult } from "../types";

const STEP_MS = 200; // playback cadence between epochs

export const useRetrainStore = defineStore("retrain", () => {
  const result = ref<RetrainResult | null>(null);
  const index = ref(0); // current epoch frame
  const playing = ref(false);
  const engaged = ref(false); // is the animation driving the edge overlay?
  const loading = ref(false);
  const error = ref<string | null>(null);
  let timer: ReturnType<typeof setInterval> | null = null;

  const epochs = computed<RetrainEpoch[]>(() => result.value?.epochs ?? []);
  // The epoch the edge overlay should render — null unless engaged, so the map
  // falls back to the learner's KGT adjustments.
  const currentEpoch = computed<RetrainEpoch | null>(() =>
    engaged.value ? epochs.value[index.value] ?? null : null,
  );
  // Cost ratio for the side-by-side caption ("retrain is N× slower than KGT").
  const costRatio = computed(() => {
    const r = result.value;
    return r && r.kgt_wall_ms > 0 ? Math.round(r.wall_ms / r.kgt_wall_ms) : null;
  });

  function pause() {
    playing.value = false;
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  function setIndex(i: number) {
    index.value = Math.max(0, Math.min(i, epochs.value.length - 1));
  }

  function play() {
    if (playing.value || epochs.value.length === 0) return;
    if (index.value >= epochs.value.length - 1) index.value = 0; // replay
    playing.value = true;
    timer = setInterval(() => {
      if (index.value >= epochs.value.length - 1) pause(); // hold the final fit
      else index.value += 1;
    }, STEP_MS);
  }

  // Run the gradient fit for a learner and start replaying it.
  async function run(learnerId: string, epochCount = 30) {
    pause();
    loading.value = true;
    error.value = null;
    try {
      result.value = await api.postRetrain(learnerId, epochCount);
      index.value = 0;
      engaged.value = true;
      play();
    } catch (e) {
      error.value = (e as Error).message;
    } finally {
      loading.value = false;
    }
  }

  // Hand the edge overlay back to KGT (keep the result for re-play).
  function disengage() {
    pause();
    engaged.value = false;
  }

  // Drop everything (learner switch / KGT toggle off).
  function reset() {
    pause();
    result.value = null;
    index.value = 0;
    engaged.value = false;
    error.value = null;
  }

  return {
    result, index, playing, engaged, loading, error,
    epochs, currentEpoch, costRatio,
    run, play, pause, setIndex, disengage, reset,
  };
});
