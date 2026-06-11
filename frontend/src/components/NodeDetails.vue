<script setup lang="ts">
import { computed } from "vue";
import type { ChatTurn, EdgeAdjustment, MapNode, NodeEvidence, Status, SyntaxMap } from "../types";
import { CEFR_COLOR, KGT_EDGE_COLOR, STATUS_COLOR, STATUS_LABEL } from "../constants";

const props = defineProps<{
  node: MapNode | null;
  status: Status | undefined;
  mastery: number | undefined;
  map: SyntaxMap;
  // Phase 6: the chat turns that activated this node (the 6-B evidence view).
  evidence?: NodeEvidence;
  chatTurns?: ChatTurn[];
  // Phase 7: KGT adjustments on this node's incident edges (the §3.8
  // interpretability view — why the personal graph differs here).
  edgeAdjustments?: EdgeAdjustment[];
}>();

const evidenceTurns = computed(() => {
  if (!props.evidence || !props.chatTurns) return [];
  const wrong = new Set(props.evidence.incorrect_turn_indices ?? []);
  return props.evidence.turn_indices
    .map((i) => ({ text: props.chatTurns![i]?.text ?? "", wrong: wrong.has(i) }))
    .filter((t) => t.text);
});

const emit = defineEmits<{ (e: "toggle", nodeId: string): void; (e: "close"): void }>();

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

// The factor that drove each adjustment (the one furthest from ×1).
function dominantFactor(a: EdgeAdjustment): number {
  return Math.abs(a.factor_back - 1) >= Math.abs(a.factor_fwd - 1) ? a.factor_back : a.factor_fwd;
}
function kgtColor(a: EdgeAdjustment): string {
  return KGT_EDGE_COLOR[a.kind];
}
</script>

<template>
  <aside class="details">
    <div class="dhead">
      <h2>Node details</h2>
      <button class="close" title="Close" @click="emit('close')">×</button>
    </div>
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
        <ul>
          <li v-for="(t, i) in evidenceTurns" :key="i" :class="{ wrong: t.wrong }">
            “{{ t.text }}”<span v-if="t.wrong"> — used incorrectly ✗</span>
          </li>
        </ul>
      </div>

      <div v-if="edgeAdjustments?.length" class="kgt">
        <div class="mhead"><span class="k">Personal edge adjustments</span></div>
        <ul>
          <li v-for="a in edgeAdjustments" :key="a.source + '→' + a.target">
            <span class="kgt-badge" :style="{ background: kgtColor(a) }">
              {{ a.kind }} ×{{ dominantFactor(a).toFixed(2) }}
            </span>
            <span class="kgt-edge">{{ labelOf(a.source) }} → {{ labelOf(a.target) }}</span>
            <p class="kgt-reason">{{ a.reason }}</p>
          </li>
        </ul>
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
.dhead { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.dhead h2 { margin: 0; }
.close { font-size: 18px; line-height: 1; border: none; background: none; color: var(--muted); cursor: pointer; padding: 0 4px; }
.close:hover { color: var(--ink); }
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
.evidence li.wrong { color: var(--gap, #c62828); }
.kgt { margin-bottom: 12px; }
.kgt ul { list-style: none; margin: 4px 0 0; padding: 0; }
.kgt li { margin: 8px 0; }
.kgt-badge { display: inline-block; font-size: 10px; padding: 1px 7px; border-radius: 9px; color: #fff; margin-right: 6px; }
.kgt-edge { font-size: 12px; color: var(--ink); }
.kgt-reason { font-size: 11px; color: var(--muted); line-height: 1.45; margin: 3px 0 0; }
.toggle-btn { width: 100%; padding: 7px; font-size: 13px; border: 1px solid var(--line); border-radius: 6px; background: #fff; cursor: pointer; margin-bottom: 12px; }
.toggle-btn:hover { background: #f5f5f5; }
.toggle-btn.on { border-color: var(--known); color: var(--known); }
.desc { font-size: 13px; color: #455a64; line-height: 1.45; }
.kv { margin: 10px 0; }
.kv .k { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
.kv ul { margin: 4px 0 0; padding-left: 16px; }
.kv li { margin: 2px 0; }
</style>
