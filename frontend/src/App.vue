<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { storeToRefs } from "pinia";
import { useMapStore } from "./stores/map";
import { useLearnerStore } from "./stores/learner";
import { useViewStore } from "./stores/view";
import ControlsPanel from "./components/ControlsPanel.vue";
import GraphCanvas from "./components/GraphCanvas.vue";
import StatusBar from "./components/StatusBar.vue";
import NodeDetails from "./components/NodeDetails.vue";
import type { MapNode } from "./types";

const mapStore = useMapStore();
const learnerStore = useLearnerStore();
const viewStore = useViewStore();
const { map } = storeToRefs(mapStore);
const { data: learnerData } = storeToRefs(learnerStore);
const { layout, enabledLevels, overlayOn, subgraphOnly } = storeToRefs(viewStore);

const selectedId = ref<string | null>(null);

const ready = computed(() => !!map.value && !!learnerData.value);
const statuses = computed(() => learnerData.value?.statuses ?? {});
const selectedNode = computed<MapNode | null>(() =>
  map.value && selectedId.value ? map.value.nodes.find((n) => n.id === selectedId.value) ?? null : null,
);
const selectedStatus = computed(() => (selectedId.value ? learnerStore.statusOf(selectedId.value) : undefined));
const mode = computed(() => learnerData.value?.learner_id ?? learnerStore.learnerId);

function onToggle(id: string) {
  learnerStore.toggle(id);
}

onMounted(() => {
  mapStore.load();
  learnerStore.load("demo");
});
</script>

<template>
  <div class="app">
    <header>
      <h1>Learner Linguistic Knowledge Graph</h1>
      <span class="sub">Static syntax map (v0) · learner: {{ mode }}</span>
      <StatusBar class="bar" :data="learnerData" />
    </header>

    <ControlsPanel />

    <main>
      <div v-if="mapStore.loading || learnerStore.loading" class="msg">Loading…</div>
      <div v-else-if="mapStore.error || learnerStore.error" class="msg err">
        API error: {{ mapStore.error || learnerStore.error }}<br />
        Is the backend running on <code>:8000</code>?
      </div>
      <GraphCanvas
        v-else-if="ready && map"
        :map="map"
        :statuses="statuses"
        :layout="layout"
        :enabled-levels="enabledLevels"
        :overlay-on="overlayOn"
        :subgraph-only="subgraphOnly"
        @select="selectedId = $event"
        @toggle="onToggle"
      />
    </main>

    <NodeDetails v-if="map" :node="selectedNode" :status="selectedStatus" :map="map" @toggle="onToggle" />
  </div>
</template>

<style scoped>
.app {
  display: grid;
  grid-template-columns: 250px 1fr 300px;
  grid-template-rows: auto 1fr;
  height: 100vh;
}
header {
  grid-column: 1 / -1;
  display: flex;
  align-items: baseline;
  gap: 16px;
  padding: 10px 16px;
  background: var(--panel);
  border-bottom: 1px solid var(--line);
}
header h1 { font-size: 15px; margin: 0; font-weight: 600; }
header .sub { font-size: 12px; color: var(--muted); }
header .bar { margin-left: auto; }
main { position: relative; overflow: hidden; }
.msg { padding: 20px; font-size: 14px; }
.msg.err { color: var(--gap); line-height: 1.6; }
code { background: #eceff1; padding: 1px 5px; border-radius: 3px; }
</style>
