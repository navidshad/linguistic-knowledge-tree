import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { api } from "../services/api";
import type { Cefr, EdgeAdjustment, LearnerProfile, LearnerStatus, Status } from "../types";

export const useLearnerStore = defineStore("learner", () => {
  const learnerId = ref("demo");
  const learners = ref<LearnerProfile[]>([]); // selectable learners (from the API)
  const data = ref<LearnerStatus | null>(null);
  const activated = ref<Set<string>>(new Set()); // editable known-set for what-if
  const loading = ref(false);
  const error = ref<string | null>(null);

  // The learner's personal-graph deltas (present after a kgt=true load).
  const edgeAdjustments = computed<EdgeAdjustment[] | null>(
    () => data.value?.edge_adjustments ?? null,
  );

  // Is the active learner a file-backed user profile (vs. a read-only built-in)?
  const isEditable = computed(
    () => learners.value.find((l) => l.id === learnerId.value)?.editable ?? false,
  );

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

  async function load(id: string = learnerId.value, kgt = false) {
    learnerId.value = id;
    loading.value = true;
    error.value = null;
    try {
      data.value = await api.getLearnerStatus(id, kgt);
      syncActivatedFromData();
    } catch (e) {
      error.value = (e as Error).message;
    } finally {
      loading.value = false;
    }
  }

  // Persist a known/not-known mark to an editable profile as review evidence,
  // then take the recomputed status (mastery + KGT) the server returns.
  async function markNode(nodeId: string, known: boolean, kgt = false) {
    error.value = null;
    try {
      data.value = await api.postProfileEvent(
        learnerId.value, { node_ids: [nodeId], correct: known, source: "review" }, kgt,
      );
      syncActivatedFromData();
    } catch (e) {
      error.value = (e as Error).message;
    }
  }

  async function createProfile(name: string, seedLevel: Cefr | null = null) {
    const p = await api.createProfile({ name, seed_level: seedLevel });
    await loadLearners();
    await load(p.id);
    return p;
  }

  async function deleteProfile(id: string) {
    await api.deleteProfile(id);
    await loadLearners();
    if (learnerId.value === id) await load("demo");
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

  return {
    learnerId, learners, data, activated, loading, error, edgeAdjustments, isEditable,
    loadLearners, load, refresh, toggle, statusOf, markNode, createProfile, deleteProfile,
  };
});
