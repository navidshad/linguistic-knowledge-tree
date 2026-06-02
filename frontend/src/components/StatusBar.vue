<script setup lang="ts">
import type { LearnerStatus } from "../types";

defineProps<{ data: LearnerStatus | null }>();
</script>

<template>
  <div v-if="data" class="stats">
    <span>Known <b>{{ data.counts.known ?? 0 }}</b></span>
    <span>Gaps <b class="gap">{{ data.counts.interior_gap ?? 0 }}</b></span>
    <span>Frontier <b class="fr">{{ data.counts.frontier ?? 0 }}</b></span>
    <span>Further <b>{{ data.counts.further ?? 0 }}</b></span>
    <span>Total <b>{{ Object.values(data.statuses).length }}</b></span>
  </div>
</template>

<style scoped>
.stats { display: flex; gap: 14px; font-size: 12px; color: var(--muted); }
.stats b { color: var(--ink); }
.gap { color: var(--gap); }
.fr { color: #f57f17; }
</style>
