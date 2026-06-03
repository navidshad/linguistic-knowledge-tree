import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { api } from "../services/api";
import type { ChatTurn, NodeEvidence, Status } from "../types";

// The Gemini chat demo: each learner turn is mapped to concept nodes server-side
// and folded as `dialog` evidence, so the map overlay (statuses/mastery) reflects
// the conversation. State is the server's recomputed knowledge state each turn.
export const useChatStore = defineStore("chat", () => {
  const sessionId = ref(crypto.randomUUID()); // keys the server-side per-session trace
  const messages = ref<ChatTurn[]>([]);
  const statuses = ref<Record<string, Status>>({});
  const mastery = ref<Record<string, number>>({});
  const counts = ref<Partial<Record<Status, number>>>({});
  const evidenceByNode = ref<Record<string, NodeEvidence>>({});
  const lastMapped = ref<string[]>([]); // nodes the most recent turn lit up
  const sending = ref(false);
  const error = ref<string | null>(null);

  // The current known-set, carried back to the server for the K-BERT readiness bias.
  const activated = computed(() =>
    Object.entries(statuses.value)
      .filter(([, s]) => s === "known")
      .map(([id]) => id),
  );

  async function send(text: string) {
    const t = text.trim();
    if (!t || sending.value) return;
    messages.value = [...messages.value, { role: "user", text: t }];
    sending.value = true;
    error.value = null;
    try {
      const res = await api.postChat(messages.value, activated.value, sessionId.value);
      messages.value = [...messages.value, { role: "tutor", text: res.reply }];
      statuses.value = res.statuses;
      mastery.value = res.mastery;
      counts.value = res.counts;
      lastMapped.value = res.mapped_nodes;
      // The server returns the full accumulated evidence, so just index it.
      evidenceByNode.value = Object.fromEntries(res.evidence.map((e) => [e.node_id, e]));
    } catch (e) {
      error.value = (e as Error).message;
      messages.value = messages.value.slice(0, -1); // roll back the optimistic user turn
    } finally {
      sending.value = false;
    }
  }

  function reset() {
    messages.value = [];
    statuses.value = {};
    mastery.value = {};
    counts.value = {};
    evidenceByNode.value = {};
    lastMapped.value = [];
    error.value = null;
    sessionId.value = crypto.randomUUID(); // fresh session → fresh trace file
  }

  return {
    sessionId, messages, statuses, mastery, counts, evidenceByNode, lastMapped, sending, error,
    activated, send, reset,
  };
});
