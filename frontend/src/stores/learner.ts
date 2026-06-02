import { defineStore } from "pinia";
import { ref } from "vue";
import { api } from "../services/api";
import type { LearnerProfile, LearnerStatus, Status } from "../types";

export const useLearnerStore = defineStore("learner", () => {
  const learnerId = ref("demo");
  const learners = ref<LearnerProfile[]>([]); // selectable learners (from the API)
  const data = ref<LearnerStatus | null>(null);
  const activated = ref<Set<string>>(new Set()); // editable known-set for what-if
  const loading = ref(false);
  const error = ref<string | null>(null);

  async function loadLearners() {
    try {
      learners.value = await api.getLearners();
    } catch (e) {
      error.value = (e as Error).message;
    }
  }

  function syncActivatedFromData() {
    if (!data.value) return;
    activated.value = new Set(
      Object.entries(data.value.statuses).filter(([, s]) => s === "known").map(([id]) => id),
    );
  }

  async function load(id: string = learnerId.value) {
    learnerId.value = id;
    loading.value = true;
    error.value = null;
    try {
      data.value = await api.getLearnerStatus(id);
      syncActivatedFromData();
    } catch (e) {
      error.value = (e as Error).message;
    } finally {
      loading.value = false;
    }
  }

  // Recompute statuses for the current editable activated set (engine drives it).
  async function refresh() {
    error.value = null;
    try {
      data.value = await api.postStatus([...activated.value]);
    } catch (e) {
      error.value = (e as Error).message;
    }
  }

  async function toggle(nodeId: string) {
    const next = new Set(activated.value);
    next.has(nodeId) ? next.delete(nodeId) : next.add(nodeId);
    activated.value = next;
    await refresh();
  }

  function statusOf(nodeId: string): Status | undefined {
    return data.value?.statuses[nodeId];
  }

  return { learnerId, learners, data, activated, loading, error, loadLearners, load, refresh, toggle, statusOf };
});
