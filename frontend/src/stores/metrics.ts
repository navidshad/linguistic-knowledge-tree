import { defineStore } from "pinia";
import { ref } from "vue";
import { api } from "../services/api";
import type { ValidationResults } from "../types";

// Phase 5 validation results (Duolingo SLAM). Loaded once, lazily, when the
// validation tab is first opened.
export const useMetricsStore = defineStore("metrics", () => {
  const data = ref<ValidationResults | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  async function load() {
    if (data.value || loading.value) return;
    loading.value = true;
    error.value = null;
    try {
      data.value = await api.getMetrics();
    } catch (e) {
      error.value = (e as Error).message;
    } finally {
      loading.value = false;
    }
  }

  return { data, loading, error, load };
});
