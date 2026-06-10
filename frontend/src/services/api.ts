// Typed client for the FastAPI backend. Transport is isolated here so it can be
// swapped (e.g. for Subturtle's client) without touching stores/components.
import type {
  ChatResponse,
  ChatTurn,
  Conversation,
  LearnerProfile,
  LearnerStatus,
  ProfileCreate,
  ProfileEvent,
  RetrainResult,
  SyntaxMap,
  Timeline,
  ValidationResults,
} from "../types";

const BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
  return (await res.json()) as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
  return (await res.json()) as T;
}

async function del(path: string): Promise<void> {
  const res = await fetch(`${BASE}${path}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
}

export const api = {
  getMap: () => get<SyntaxMap>("/api/map"),
  // Built-ins + file-backed user profiles, merged (each carries `editable`).
  getLearners: () => get<LearnerProfile[]>("/api/learners"),
  // Persistent profile CRUD (Phase 8).
  createProfile: (body: ProfileCreate) => post<LearnerProfile>("/api/profiles", body),
  deleteProfile: (id: string) => del(`/api/profiles/${id}`),
  // Append evidence (e.g. marking a node known) → returns the recomputed status.
  postProfileEvent: (id: string, body: ProfileEvent, kgt = false) =>
    post<LearnerStatus>(`/api/profiles/${id}/events${kgt ? "?kgt=1" : ""}`, body),
  getConversation: (id: string) => get<Conversation>(`/api/profiles/${id}/conversation`),
  // kgt=true computes mastery over the learner's personalized graph (Phase 7,
  // RQ5) and includes the edge adjustments behind it.
  getLearnerStatus: (id: string, kgt = false) =>
    get<LearnerStatus>(`/api/learner/${id}/status${kgt ? "?kgt=1" : ""}`),
  // Run the RQ5 retrain comparator live: per-epoch loss + edge factors.
  postRetrain: (id: string, epochs = 30) =>
    post<RetrainResult>(`/api/learner/${id}/retrain?epochs=${epochs}`, {}),
  // Mastery sampled across the learner's history, for the timeline scrubber.
  getTimeline: (id: string, frames = 24) =>
    get<Timeline>(`/api/learner/${id}/timeline?frames=${frames}`),
  // Interactive what-if: compute statuses for an arbitrary activated set.
  postStatus: (activated: string[]) => post<LearnerStatus>("/api/status", { activated }),
  // Phase 5 validation results (SLAM ablation table + ROC curves).
  getMetrics: () => get<ValidationResults>("/api/metrics"),
  // Phase 6 chat: a tutor turn + the knowledge state the learner's turns imply.
  // sessionId keys the server-side per-session pipeline trace; profileId (Phase 8)
  // persists the conversation + dialog evidence to that profile.
  postChat: (messages: ChatTurn[], activated: string[], sessionId?: string, profileId?: string) =>
    post<ChatResponse>("/api/chat", {
      messages, activated, session_id: sessionId, profile_id: profileId,
    }),
};
