import { defineStore } from "pinia";
import { ref } from "vue";
import { api } from "../services/api";
import type { SyntaxMap } from "../types";

export const useMapStore = defineStore("map", () => {
  const map = ref<SyntaxMap | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  async function load() {
    loading.value = true;
    error.value = null;
    try {
      map.value = await api.getMap();
    } catch (e) {
      error.value = (e as Error).message;
    } finally {
      loading.value = false;
    }
  }

  return { map, loading, error, load };
});
