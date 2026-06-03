<script setup lang="ts">
import { computed } from "vue";
import type { ChatTurn, MapNode, NodeEvidence, Status, SyntaxMap } from "../types";
import { CEFR_COLOR, STATUS_COLOR, STATUS_LABEL } from "../constants";

const props = defineProps<{
  node: MapNode | null;
  status: Status | undefined;
  mastery: number | undefined;
  map: SyntaxMap;
  // Phase 6: the chat turns that activated this node (the 6-B evidence view).
  evidence?: NodeEvidence;
  chatTurns?: ChatTurn[];
}>();

const evidenceTurns = computed(() =>
  props.evidence && props.chatTurns
    ? props.evidence.turn_indices.map((i) => props.chatTurns![i]?.text).filter(Boolean)
    : [],
);

const emit = defineEmits<{ (e: "toggle", nodeId: string): void }>();

const masteryPct = computed(() =>
  props.mastery === undefined ? null : Math.round(props.mastery * 100),
);

function labelOf(id: string): string {
  return props.map.nodes.find((n) => n.id === id)?.label ?? id;
}

const categoryLabel = computed(() =>
  props.node ? props.map.categories.find((c) => c.id === props.node!.category)?.label ?? "" : "",
);
const prerequisites = computed(() =>
  props.node ? props.map.edges.filter((e) => e.target === props.node!.id).map((e) => labelOf(e.source)) : [],
);
const unlocks = computed(() =>
  props.node ? props.map.edges.filter((e) => e.source === props.node!.id).map((e) => labelOf(e.target)) : [],
);
const isKnown = computed(() => props.status === "known");
</script>

<template>
  <aside class="details">
    <h2>Node details</h2>
    <p v-if="!node" class="empty">Click a node to inspect it. Right-click toggles known.</p>
    <template v-else>
      <p class="title">{{ node.label }}</p>
      <div class="badges">
        <span
          v-if="status"
          class="badge"
          :style="{ background: STATUS_COLOR[status], color: status === 'frontier' ? '#1a1a1a' : '#fff' }"
        >{{ STATUS_LABEL[status] }}</span>
        <span class="badge" :style="{ background: CEFR_COLOR[node.cefr] }">{{ node.cefr }}</span>
      </div>

      <div v-if="masteryPct !== null" class="mastery">
        <div class="mhead"><span class="k">Mastery (confidence)</span><b>{{ masteryPct }}%</b></div>
        <div class="bar">
          <div
            class="fill"
            :style="{ width: masteryPct + '%', background: status ? STATUS_COLOR[status] : '#90a4ae' }"
          />
        </div>
      </div>

      <div v-if="evidenceTurns.length" class="evidence">
        <div class="mhead">
          <span class="k">Evidence — chat turns</span>
          <b v-if="evidence">{{ Math.round(evidence.confidence * 100) }}%</b>
        </div>
        <ul><li v-for="(t, i) in evidenceTurns" :key="i">“{{ t }}”</li></ul>
      </div>

      <button class="toggle-btn" :class="{ on: isKnown }" @click="emit('toggle', node.id)">
        {{ isKnown ? "Mark as not known" : "Mark as known" }}
      </button>

      <p class="desc">{{ node.description }}</p>
      <div class="kv"><span class="k">Category</span><div>{{ categoryLabel }}</div></div>
      <div class="kv">
        <span class="k">Prerequisites</span>
        <ul v-if="prerequisites.length"><li v-for="p in prerequisites" :key="p">{{ p }}</li></ul>
        <div v-else>—</div>
      </div>
      <div class="kv">
        <span class="k">Unlocks</span>
        <ul v-if="unlocks.length"><li v-for="u in unlocks" :key="u">{{ u }}</li></ul>
        <div v-else>—</div>
      </div>
    </template>
  </aside>
</template>

<style scoped>
.details { background: var(--panel); border-left: 1px solid var(--line); padding: 14px; overflow-y: auto; font-size: 13px; }
h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 8px; }
.empty { color: var(--muted); font-style: italic; line-height: 1.5; }
.title { font-size: 15px; font-weight: 600; margin: 0 0 6px; }
.badges { margin-bottom: 10px; }
.badge { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 10px; color: #fff; margin-right: 6px; }
.mastery { margin-bottom: 12px; }
.mhead { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
.mhead .k { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
.bar { height: 8px; background: #eceff1; border-radius: 5px; overflow: hidden; }
.fill { height: 100%; border-radius: 5px; transition: width 0.2s ease; }
.evidence { margin-bottom: 12px; }
.evidence ul { margin: 4px 0 0; padding-left: 16px; }
.evidence li { margin: 3px 0; color: #455a64; line-height: 1.4; font-style: italic; }
.toggle-btn { width: 100%; padding: 7px; font-size: 13px; border: 1px solid var(--line); border-radius: 6px; background: #fff; cursor: pointer; margin-bottom: 12px; }
.toggle-btn:hover { background: #f5f5f5; }
.toggle-btn.on { border-color: var(--known); color: var(--known); }
.desc { font-size: 13px; color: #455a64; line-height: 1.45; }
.kv { margin: 10px 0; }
.kv .k { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
.kv ul { margin: 4px 0 0; padding-left: 16px; }
.kv li { margin: 2px 0; }
</style>
