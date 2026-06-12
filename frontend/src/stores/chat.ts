import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { api } from "../services/api";
import type { ChatTurn, EdgeAdjustment, NodeEvidence, Status } from "../types";

// The Gemini chat demo: each learner turn is mapped to concept nodes server-side
// and folded as `dialog` evidence, so the map overlay (statuses/mastery) reflects
// the conversation. State is the server's recomputed knowledge state each turn.
export const useChatStore = defineStore("chat", () => {
  const sessionId = ref<string>(crypto.randomUUID()); // keys the server-side per-session trace
  const profileId = ref<string | null>(null); // when set, the conversation persists
  // Evidence→node mapper: "semantic" (default, the validated K-BERT mapper + Gemini
  // grading) or "gemini" (Gemini tags concepts directly — higher recall, lights the
  // map up far more reliably). A live demo preference, not the validated default.
  const tagger = ref<"semantic" | "gemini">("semantic");
  const messages = ref<ChatTurn[]>([]);
  const statuses = ref<Record<string, Status>>({});
  const mastery = ref<Record<string, number>>({});
  const counts = ref<Partial<Record<Status, number>>>({});
  const evidenceByNode = ref<Record<string, NodeEvidence>>({});
  const lastMapped = ref<string[]>([]); // nodes the most recent turn lit up
  const lastGrades = ref<Record<string, boolean>>({}); // node -> used correctly?
  const edgeAdjustments = ref<EdgeAdjustment[] | null>(null); // KGT live on the conversation
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
      const res = await api.postChat(
        messages.value, activated.value, sessionId.value, profileId.value ?? undefined,
        tagger.value === "gemini" ? "gemini" : undefined,
      );
      messages.value = [...messages.value, { role: "tutor", text: res.reply }];
      statuses.value = res.statuses;
      mastery.value = res.mastery;
      counts.value = res.counts;
      lastMapped.value = res.mapped_nodes;
      lastGrades.value = res.grades;
      edgeAdjustments.value = res.edge_adjustments ?? null;
      // The server returns the full accumulated evidence, so just index it.
      evidenceByNode.value = Object.fromEntries(res.evidence.map((e) => [e.node_id, e]));
    } catch (e) {
      error.value = (e as Error).message;
      messages.value = messages.value.slice(0, -1); // roll back the optimistic user turn
    } finally {
      sending.value = false;
    }
  }

  function _clear() {
    messages.value = [];
    statuses.value = {};
    mastery.value = {};
    counts.value = {};
    evidenceByNode.value = {};
    lastMapped.value = [];
    lastGrades.value = {};
    edgeAdjustments.value = null;
    error.value = null;
  }

  function reset() {
    _clear();
    profileId.value = null; // back to an ephemeral (non-persisted) conversation
    sessionId.value = crypto.randomUUID(); // fresh session → fresh trace file
  }

  // Resume a persistent profile: hydrate the saved transcript + an overlay
  // snapshot (KGT on, matching how chat turns recompute). No tutor turn is
  // generated, so reopening never appends a spurious reply.
  async function loadForProfile(id: string) {
    _clear();
    profileId.value = id;
    sessionId.value = `profile-${id}`;
    try {
      const convo = await api.getConversation(id);
      messages.value = convo.messages;
      const st = await api.getLearnerStatus(id, true);
      statuses.value = st.statuses;
      mastery.value = st.mastery ?? {};
      counts.value = st.counts;
      edgeAdjustments.value = st.edge_adjustments ?? null;
    } catch (e) {
      error.value = (e as Error).message;
    }
  }

  return {
    sessionId, profileId, tagger, messages, statuses, mastery, counts, evidenceByNode,
    lastMapped, lastGrades, edgeAdjustments, sending, error, activated,
    send, reset, loadForProfile,
  };
});
