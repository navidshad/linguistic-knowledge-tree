<script setup lang="ts">
import type { Status } from "../types";

// Counts/total are passed in (rather than a LearnerStatus) so the bar reflects
// whatever the view shows — the learner's live status or a timeline frame.
defineProps<{
  counts: Partial<Record<Status, number>>;
  total: number;
  day: number | null; // timeline day when the scrubber is engaged, else null
}>();
</script>

<template>
  <div v-if="total" class="stats">
    <span v-if="day !== null" class="day">Day {{ day }}</span>
    <span>Known <b>{{ counts.known ?? 0 }}</b></span>
    <span>Gaps <b class="gap">{{ counts.interior_gap ?? 0 }}</b></span>
    <span>Frontier <b class="fr">{{ counts.frontier ?? 0 }}</b></span>
    <span>Further <b>{{ counts.further ?? 0 }}</b></span>
    <span>Total <b>{{ total }}</b></span>
  </div>
</template>

<style scoped>
.stats { display: flex; gap: 14px; font-size: 12px; color: var(--muted); align-items: baseline; }
.stats b { color: var(--ink); }
.gap { color: var(--gap); }
.fr { color: #f57f17; }
.day { font-weight: 600; color: var(--ink); background: #eceff1; padding: 1px 8px; border-radius: 10px; }
</style>
