import { defineStore } from "pinia";
import { ref } from "vue";
import { CEFR_ORDER } from "../constants";
import type { Cefr, LayoutName } from "../types";

export const useViewStore = defineStore("view", () => {
  const layout = ref<LayoutName>("matrix");
  const enabledLevels = ref<Set<Cefr>>(new Set(CEFR_ORDER)); // all levels visible
  const overlayOn = ref(true); // show learner status overlay vs bare map
  const subgraphOnly = ref(false); // hide "further" nodes

  function setLayout(l: LayoutName) {
    layout.value = l;
  }

  function toggleLevel(l: Cefr) {
    const next = new Set(enabledLevels.value);
    next.has(l) ? next.delete(l) : next.add(l);
    enabledLevels.value = next;
  }

  return { layout, enabledLevels, overlayOn, subgraphOnly, setLayout, toggleLevel };
});
