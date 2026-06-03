import { defineStore } from "pinia";
import { ref } from "vue";
import { CEFR_ORDER } from "../constants";
import type { Cefr, LayoutName, Tab } from "../types";

export const useViewStore = defineStore("view", () => {
  const tab = ref<Tab>("map"); // map browser vs. Phase-5 validation dashboard
  const layout = ref<LayoutName>("matrix");
  const enabledLevels = ref<Set<Cefr>>(new Set(CEFR_ORDER)); // all levels visible
  const overlayOn = ref(true); // show learner status overlay vs bare map
  const subgraphOnly = ref(false); // hide "further" nodes
  const confidenceOn = ref(false); // node opacity tracks mastery (Phase 4-B)

  function setTab(t: Tab) {
    tab.value = t;
  }

  function setLayout(l: LayoutName) {
    layout.value = l;
  }

  function toggleLevel(l: Cefr) {
    const next = new Set(enabledLevels.value);
    next.has(l) ? next.delete(l) : next.add(l);
    enabledLevels.value = next;
  }

  return { tab, layout, enabledLevels, overlayOn, subgraphOnly, confidenceOn, setTab, setLayout, toggleLevel };
});
