import { defineStore } from "pinia";
import { ref } from "vue";
import { CEFR_ORDER } from "../constants";
import type { Cefr, LayoutName, Tab } from "../types";

export const useViewStore = defineStore("view", () => {
  // Top-level screen: the profile gallery (landing) vs. an open profile's
  // workspace. You pick/create a profile first, then drill into its detail.
  const screen = ref<"profiles" | "workspace">("profiles");
  const tab = ref<Tab>("map"); // within a profile: map browser, chat, or validation
  const layout = ref<LayoutName>("matrix");
  const enabledLevels = ref<Set<Cefr>>(new Set(CEFR_ORDER)); // all levels visible
  const overlayOn = ref(true); // show learner status overlay vs bare map
  const subgraphOnly = ref(false); // hide "further" nodes
  const confidenceOn = ref(false); // node opacity tracks mastery (Phase 4-B)
  const kgtOn = ref(false); // personalized graph: KGT edge adjustments (Phase 7, RQ5)

  function enterWorkspace() {
    screen.value = "workspace";
  }

  function goToProfiles() {
    screen.value = "profiles";
  }

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

  return {
    screen, tab, layout, enabledLevels, overlayOn, subgraphOnly, confidenceOn, kgtOn,
    enterWorkspace, goToProfiles, setTab, setLayout, toggleLevel,
  };
});
