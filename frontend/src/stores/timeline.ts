// Timeline scrubber state (Phase 4-B): plays back mastery sampled across a
// learner's history so propagation + forgetting become visible over time. The
// frames are precomputed server-side (one request), so scrubbing is instant.
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { api } from "../services/api";
import type { Timeline, TimelineFrame } from "../types";

const STEP_MS = 350; // playback cadence between frames

export const useTimelineStore = defineStore("timeline", () => {
  const enabled = ref(false); // is the scrubber engaged (driving the view)?
  const timeline = ref<Timeline | null>(null);
  const index = ref(0); // current frame
  const playing = ref(false);
  const loading = ref(false);
  const error = ref<string | null>(null);
  let timer: ReturnType<typeof setInterval> | null = null;

  const frames = computed<TimelineFrame[]>(() => timeline.value?.frames ?? []);
  // The frame the view should render — null unless engaged with frames loaded,
  // so the rest of the app falls back to the learner's live status.
  const currentFrame = computed<TimelineFrame | null>(() =>
    enabled.value ? frames.value[index.value] ?? null : null,
  );
  // Day relative to the learner's first evidence (t is on an arbitrary origin).
  const day = computed(() => {
    const tl = timeline.value;
    const f = frames.value[index.value];
    return tl && f ? Math.round(f.t - tl.t_start) : 0;
  });

  async function load(learnerId: string) {
    loading.value = true;
    error.value = null;
    try {
      timeline.value = await api.getTimeline(learnerId);
      index.value = 0;
    } catch (e) {
      error.value = (e as Error).message;
    } finally {
      loading.value = false;
    }
  }

  function pause() {
    playing.value = false;
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  function setIndex(i: number) {
    index.value = Math.max(0, Math.min(i, frames.value.length - 1));
  }

  function play() {
    if (playing.value || frames.value.length === 0) return;
    if (index.value >= frames.value.length - 1) index.value = 0; // replay from start
    playing.value = true;
    timer = setInterval(() => {
      if (index.value >= frames.value.length - 1) pause(); // stop at the end
      else index.value += 1;
    }, STEP_MS);
  }

  // Engage the scrubber for a learner, loading frames on first use.
  async function enable(learnerId: string) {
    enabled.value = true;
    if (!timeline.value) await load(learnerId);
  }

  function disable() {
    pause();
    enabled.value = false;
  }

  // Drop cached frames (e.g. when the learner changes).
  function reset() {
    pause();
    timeline.value = null;
    index.value = 0;
    enabled.value = false;
  }

  return {
    enabled, timeline, index, playing, loading, error,
    frames, currentFrame, day,
    load, enable, disable, reset, play, pause, setIndex,
  };
});
