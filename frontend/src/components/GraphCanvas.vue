<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { Cefr, LayoutName, Status, SyntaxMap } from "../types";
import { createGraph, type GraphHandle } from "../composables/useCytoscape";

const props = defineProps<{
  map: SyntaxMap;
  statuses: Record<string, Status>;
  mastery: Record<string, number>;
  confidenceOn: boolean;
  layout: LayoutName;
  enabledLevels: Set<Cefr>;
  overlayOn: boolean;
  subgraphOnly: boolean;
}>();

const emit = defineEmits<{
  (e: "select", nodeId: string | null): void;
  (e: "toggle", nodeId: string): void;
}>();

const container = ref<HTMLElement | null>(null);
let handle: GraphHandle | null = null;

onMounted(async () => {
  if (!container.value) return;
  handle = createGraph(container.value, props.map, props.statuses, {
    onSelect: (id) => emit("select", id),
    onToggle: (id) => emit("toggle", id),
  });
  // Let the CSS-grid container settle before laying out / fitting, so the
  // initial render can't race the mount.
  await nextTick();
  if (!handle) return;
  handle.setOverlay(props.overlayOn);
  handle.setMastery(props.mastery);
  handle.setConfidence(props.confidenceOn);
  handle.setEnabledLevels(props.enabledLevels);
  handle.setSubgraphOnly(props.subgraphOnly);
  handle.setLayout(props.layout);
});

onBeforeUnmount(() => handle?.destroy());

watch(() => props.statuses, (s) => handle?.setStatuses(s), { deep: true });
watch(() => props.mastery, (m) => handle?.setMastery(m), { deep: true });
watch(() => props.confidenceOn, (o) => handle?.setConfidence(o));
watch(() => props.layout, (l) => handle?.setLayout(l));
watch(() => props.enabledLevels, (l) => handle?.setEnabledLevels(l));
watch(() => props.overlayOn, (o) => handle?.setOverlay(o));
watch(() => props.subgraphOnly, (o) => handle?.setSubgraphOnly(o));
</script>

<template>
  <div ref="container" class="graph"></div>
</template>

<style scoped>
.graph {
  width: 100%;
  height: 100%;
  background: var(--bg);
}
</style>
