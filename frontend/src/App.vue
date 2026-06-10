<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { storeToRefs } from "pinia";
import { useMapStore } from "./stores/map";
import { useLearnerStore } from "./stores/learner";
import { useViewStore } from "./stores/view";
import { useTimelineStore } from "./stores/timeline";
import { useChatStore } from "./stores/chat";
import ControlsPanel from "./components/ControlsPanel.vue";
import ChatPanel from "./components/ChatPanel.vue";
import GraphCanvas from "./components/GraphCanvas.vue";
import StatusBar from "./components/StatusBar.vue";
import NodeDetails from "./components/NodeDetails.vue";
import MetricsDashboard from "./components/MetricsDashboard.vue";
import { STATUS_MASTERY } from "./constants";
import type { MapNode, Status } from "./types";

const mapStore = useMapStore();
const learnerStore = useLearnerStore();
const viewStore = useViewStore();
const timelineStore = useTimelineStore();
const chatStore = useChatStore();
const { map } = storeToRefs(mapStore);
const { data: learnerData } = storeToRefs(learnerStore);
const { tab, layout, enabledLevels, overlayOn, subgraphOnly, confidenceOn } = storeToRefs(viewStore);
const { currentFrame } = storeToRefs(timelineStore);
const {
  statuses: chatStatuses,
  mastery: chatMastery,
  counts: chatCounts,
  evidenceByNode,
  messages: chatMessages,
} = storeToRefs(chatStore);

const selectedId = ref<string | null>(null);

const ready = computed(() => !!map.value && !!learnerData.value);

// What the graph renders: in chat mode the live conversation overlay; otherwise
// the active timeline frame if the scrubber is engaged, else the learner's status.
// Centralising this keeps the graph, status bar and node panel on the same instant.
const displayStatuses = computed<Record<string, Status>>(() => {
  if (tab.value === "chat") return chatStatuses.value;
  return currentFrame.value?.statuses ?? learnerData.value?.statuses ?? {};
});
const displayMastery = computed<Record<string, number>>(() => {
  if (tab.value === "chat") return chatMastery.value;
  if (currentFrame.value) return currentFrame.value.mastery;
  const m = learnerData.value?.mastery;
  if (m) return m;
  // What-if mode: no evidence to score, so approximate confidence from status.
  return Object.fromEntries(
    Object.entries(displayStatuses.value).map(([id, s]) => [id, STATUS_MASTERY[s]]),
  );
});
const displayCounts = computed(() => {
  if (tab.value === "chat") return chatCounts.value;
  return currentFrame.value?.counts ?? learnerData.value?.counts ?? {};
});
const displayTotal = computed(() => Object.keys(displayStatuses.value).length);

// Chat mode glows by mastery (confidence overlay) so the map fills in as the
// learner talks; the Map tab keeps its user-controlled confidence toggle.
const confidenceActive = computed(() => (tab.value === "chat" ? true : confidenceOn.value));

const selectedNode = computed<MapNode | null>(() =>
  map.value && selectedId.value ? map.value.nodes.find((n) => n.id === selectedId.value) ?? null : null,
);
const selectedStatus = computed(() => (selectedId.value ? displayStatuses.value[selectedId.value] : undefined));
const selectedMastery = computed(() => (selectedId.value ? displayMastery.value[selectedId.value] : undefined));
const selectedEvidence = computed(() =>
  tab.value === "chat" && selectedId.value ? evidenceByNode.value[selectedId.value] : undefined,
);
const mode = computed(() => learnerData.value?.learner_id ?? learnerStore.learnerId);

function onToggle(id: string) {
  if (tab.value !== "map") return; // chat/validation overlays are not hand-editable
  learnerStore.toggle(id);
}

onMounted(() => {
  mapStore.load();
  learnerStore.loadLearners();
  learnerStore.load("demo");
});
</script>

<template>
  <div class="app">
    <header>
      <h1>Learner Linguistic Knowledge Graph</h1>
      <nav class="tabs">
        <button :class="{ active: tab === 'map' }" @click="viewStore.setTab('map')">Map</button>
        <button :class="{ active: tab === 'chat' }" @click="viewStore.setTab('chat')">Chat</button>
        <button :class="{ active: tab === 'metrics' }" @click="viewStore.setTab('metrics')">Validation</button>
      </nav>
      <template v-if="tab !== 'metrics'">
        <span class="sub">
          {{ tab === "chat" ? "Live chat · your turns activate the map" : "Static syntax map (v0) · learner: " + mode }}
        </span>
        <StatusBar
          class="bar"
          :counts="displayCounts"
          :total="displayTotal"
          :day="currentFrame && tab === 'map' ? timelineStore.day : null"
        />
      </template>
    </header>

    <template v-if="tab === 'metrics'">
      <MetricsDashboard class="metrics-pane" />
    </template>
    <template v-else>
      <ControlsPanel v-if="tab === 'map'" />
      <ChatPanel v-else />

      <main>
        <div v-if="mapStore.loading || learnerStore.loading" class="msg">Loading…</div>
        <div v-else-if="mapStore.error || learnerStore.error" class="msg err">
          API error: {{ mapStore.error || learnerStore.error }}<br />
          Is the backend running on <code>:8000</code>?
        </div>
        <GraphCanvas
          v-else-if="ready && map"
          :map="map"
          :statuses="displayStatuses"
          :mastery="displayMastery"
          :confidence-on="confidenceActive"
          :layout="layout"
          :enabled-levels="enabledLevels"
          :overlay-on="overlayOn"
          :subgraph-only="subgraphOnly"
          @select="selectedId = $event"
          @toggle="onToggle"
        />
      </main>

      <NodeDetails
        v-if="map"
        :node="selectedNode"
        :status="selectedStatus"
        :mastery="selectedMastery"
        :map="map"
        :evidence="selectedEvidence"
        :chat-turns="tab === 'chat' ? chatMessages : undefined"
        @toggle="onToggle"
      />
    </template>
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
.tabs { display: flex; gap: 4px; }
.tabs button {
  font-size: 12px; padding: 3px 12px; border: 1px solid var(--line);
  background: transparent; color: var(--muted); border-radius: 6px; cursor: pointer;
}
.tabs button.active { background: var(--ink); color: #fff; border-color: var(--ink); }
.metrics-pane { grid-column: 1 / -1; overflow: hidden; min-height: 0; }
main { position: relative; overflow: hidden; }
.msg { padding: 20px; font-size: 14px; }
.msg.err { color: var(--gap); line-height: 1.6; }
code { background: #eceff1; padding: 1px 5px; border-radius: 3px; }
</style>
