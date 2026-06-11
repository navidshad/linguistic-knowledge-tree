<script setup lang="ts">
import { computed, ref } from "vue";
import { useLearnerStore } from "../stores/learner";
import { useViewStore } from "../stores/view";
import { CEFR_ORDER } from "../constants";
import type { Cefr } from "../types";

const learner = useLearnerStore();
const view = useViewStore();

const builtinLearners = computed(() => learner.learners.filter((l) => !l.editable));
const userProfiles = computed(() => learner.learners.filter((l) => l.editable));

const newName = ref("");
const newSeed = ref<Cefr | "">("");
const creating = ref(false);
const busy = ref(false);

// Open a profile → load its state and drill into the workspace.
async function open(id: string) {
  if (busy.value) return;
  busy.value = true;
  try {
    await learner.load(id);
    view.enterWorkspace();
  } finally {
    busy.value = false;
  }
}

async function onCreate() {
  const name = newName.value.trim();
  if (!name || creating.value) return;
  creating.value = true;
  try {
    const p = await learner.createProfile(name, newSeed.value || null);
    newName.value = "";
    newSeed.value = "";
    view.enterWorkspace(); // jump straight into the new profile
    return p;
  } finally {
    creating.value = false;
  }
}

async function onDelete(id: string, label: string, e: Event) {
  e.stopPropagation(); // don't also open the card
  if (confirm(`Delete profile "${label}"? This can't be undone.`)) {
    await learner.deleteProfile(id);
  }
}
</script>

<template>
  <div class="profiles">
    <div class="inner">
      <header class="hero">
        <h1>Choose a learner</h1>
        <p>Pick a profile to open its knowledge map, or create a new one. Your
          profiles persist — chat, marked nodes, and the seed are saved to disk.</p>
      </header>

      <section class="create">
        <h2>New profile</h2>
        <form class="form" @submit.prevent="onCreate">
          <input v-model="newName" class="name-input" placeholder="Profile name…" autocomplete="off" />
          <select v-model="newSeed" class="seed" autocomplete="off">
            <option value="">No starting level</option>
            <option v-for="l in CEFR_ORDER" :key="l" :value="l">Knows up to {{ l }}</option>
          </select>
          <button type="submit" class="create-btn" :disabled="!newName.trim() || creating">
            {{ creating ? "Creating…" : "Create & open" }}
          </button>
        </form>
      </section>

      <section v-if="userProfiles.length">
        <h2>Your profiles</h2>
        <div class="grid">
          <button v-for="p in userProfiles" :key="p.id" class="card editable" @click="open(p.id)">
            <span class="del" title="Delete profile" @click="onDelete(p.id, p.label, $event)">×</span>
            <span class="tag yours">Yours</span>
            <span class="name">{{ p.label }}</span>
            <span class="desc">{{ p.description }}</span>
          </button>
        </div>
      </section>

      <section>
        <h2>Pre-defined <span class="ro">read-only</span></h2>
        <div class="grid">
          <button v-for="p in builtinLearners" :key="p.id" class="card" @click="open(p.id)">
            <span class="tag">Demo</span>
            <span class="name">{{ p.label }}</span>
            <span class="desc">{{ p.description }}</span>
          </button>
        </div>
      </section>

      <p v-if="learner.error" class="err">{{ learner.error }}</p>
    </div>
  </div>
</template>

<style scoped>
.profiles { height: 100%; overflow-y: auto; background: var(--bg); }
.inner { max-width: 860px; margin: 0 auto; padding: 40px 24px 60px; }
.hero h1 { font-size: 24px; margin: 0 0 8px; font-weight: 650; }
.hero p { font-size: 14px; color: var(--muted); line-height: 1.6; margin: 0 0 28px; max-width: 620px; }
section { margin-bottom: 32px; }
h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 12px; }
h2 .ro { font-weight: 400; text-transform: none; letter-spacing: 0; opacity: 0.7; }

.create { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 18px; }
.form { display: flex; gap: 10px; flex-wrap: wrap; }
.name-input { flex: 2; min-width: 180px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 7px; font-size: 14px; }
.seed { flex: 1; min-width: 150px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 7px; font-size: 14px; background: #fff; }
.create-btn { padding: 9px 18px; font-size: 14px; border: 1px solid var(--ink); background: var(--ink); color: #fff; border-radius: 7px; cursor: pointer; white-space: nowrap; }
.create-btn:disabled { opacity: 0.45; cursor: default; }

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
.card {
  position: relative; text-align: left; display: flex; flex-direction: column; gap: 6px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px; cursor: pointer; transition: border-color 0.15s, box-shadow 0.15s, transform 0.05s;
}
.card:hover { border-color: var(--ink); box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06); }
.card:active { transform: translateY(1px); }
.card.editable { border-left: 3px solid var(--known); }
.card .name { font-size: 15px; font-weight: 600; }
.card .desc { font-size: 12px; color: var(--muted); line-height: 1.5; }
.tag { align-self: flex-start; font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); background: #eceff1; padding: 2px 7px; border-radius: 10px; }
.tag.yours { color: #fff; background: var(--known); }
.del { position: absolute; top: 8px; right: 10px; font-size: 18px; line-height: 1; color: var(--muted); cursor: pointer; padding: 2px 4px; border-radius: 4px; }
.del:hover { color: var(--gap); background: #fdecea; }
.err { color: var(--gap); font-size: 13px; }
</style>
