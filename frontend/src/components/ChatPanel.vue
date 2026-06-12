<script setup lang="ts">
import { nextTick, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { useChatStore } from "../stores/chat";

const chat = useChatStore();
const { messages, sending, error, lastMapped, lastGrades, edgeAdjustments, tagger } = storeToRefs(chat);
const draft = ref("");
const listEl = ref<HTMLElement | null>(null);

async function send() {
  const text = draft.value;
  draft.value = "";
  await chat.send(text);
}

watch(
  () => messages.value.length,
  async () => {
    await nextTick();
    listEl.value?.scrollTo({ top: listEl.value.scrollHeight });
  },
);
</script>

<template>
  <aside class="chat">
    <div class="head">
      <h2>Chat tutor</h2>
      <button class="reset" title="Clear the conversation" @click="chat.reset()">Reset</button>
    </div>
    <label class="tagger" title="Gemini reads each turn and names the grammar used (higher recall) instead of the embedding mapper proposing and Gemini only vetoing">
      <input
        type="checkbox"
        :checked="tagger === 'gemini'"
        :disabled="sending"
        @change="tagger = ($event.target as HTMLInputElement).checked ? 'gemini' : 'semantic'"
      />
      Gemini tagging <span class="hint-badge">higher recall</span>
    </label>
    <p v-if="!messages.length" class="hint">
      Chat in English with the AI tutor. Each turn you write is mapped to grammar
      concepts and lights up the map&nbsp;→
    </p>
    <div ref="listEl" class="msgs">
      <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
        <span class="who">{{ m.role === "user" ? "You" : "Tutor" }}</span>
        <p>{{ m.text }}</p>
      </div>
      <div v-if="sending" class="msg tutor pending"><span class="who">Tutor</span><p>…</p></div>
    </div>
    <p v-if="lastMapped.length" class="mapped">
      detected:
      <span v-for="n in lastMapped" :key="n" class="tag" :class="{ wrong: lastGrades[n] === false }">
        {{ n }} {{ lastGrades[n] === false ? "✗" : "✓" }}
      </span>
    </p>
    <p v-if="edgeAdjustments?.length" class="kgt-note">
      {{ edgeAdjustments.length }} edge{{ edgeAdjustments.length === 1 ? "" : "s" }} of your
      personal graph adapted — click a highlighted edge's node for the reason.
    </p>
    <p v-if="error" class="err">{{ error }}</p>
    <form class="composer" @submit.prevent="send">
      <input v-model="draft" placeholder="Say something in English…" :disabled="sending" />
      <button type="submit" :disabled="sending || !draft.trim()">Send</button>
    </form>
  </aside>
</template>

<style scoped>
.chat { background: var(--panel); border-right: 1px solid var(--line); padding: 12px; display: flex; flex-direction: column; min-height: 0; font-size: 13px; }
.head { display: flex; justify-content: space-between; align-items: center; }
h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 8px; }
.reset { font-size: 10px; border: 1px solid var(--line); background: #fff; border-radius: 5px; padding: 2px 7px; cursor: pointer; color: var(--muted); }
.tagger { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--muted); margin: 0 0 8px; cursor: pointer; }
.tagger input { margin: 0; cursor: pointer; }
.hint-badge { font-size: 9px; background: #ede7f6; color: #6a1b9a; padding: 1px 5px; border-radius: 4px; letter-spacing: 0.02em; }
.hint { color: var(--muted); font-style: italic; line-height: 1.5; margin: 0 0 8px; }
.msgs { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; padding-right: 2px; }
.msg .who { font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
.msg p { margin: 1px 0 0; padding: 6px 9px; border-radius: 8px; line-height: 1.4; }
.msg.user p { background: var(--ink); color: #fff; }
.msg.tutor p { background: #eceff1; color: #1a1a1a; }
.msg.pending p { opacity: 0.5; }
.mapped { font-size: 11px; color: var(--known, #2e7d32); margin: 6px 0 0; }
.mapped .tag { margin-right: 8px; white-space: nowrap; }
.mapped .tag.wrong { color: var(--gap, #c62828); }
.kgt-note { font-size: 11px; color: #7b1fa2; margin: 6px 0 0; line-height: 1.45; }
.err { color: var(--gap, #c62828); font-size: 11px; margin: 6px 0 0; }
.composer { display: flex; gap: 6px; margin-top: 8px; }
.composer input { flex: 1; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; font-size: 13px; }
.composer button { padding: 6px 12px; border: 1px solid var(--ink); background: var(--ink); color: #fff; border-radius: 6px; cursor: pointer; font-size: 13px; }
.composer button:disabled { opacity: 0.5; cursor: default; }
</style>
