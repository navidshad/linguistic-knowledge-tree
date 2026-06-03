// Typed client for the FastAPI backend. Transport is isolated here so it can be
// swapped (e.g. for Subturtle's client) without touching stores/components.
import type {
  LearnerProfile,
  LearnerStatus,
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

export const api = {
  getMap: () => get<SyntaxMap>("/api/map"),
  getLearners: () => get<LearnerProfile[]>("/api/learners"),
  getLearnerStatus: (id: string) => get<LearnerStatus>(`/api/learner/${id}/status`),
  // Mastery sampled across the learner's history, for the timeline scrubber.
  getTimeline: (id: string, frames = 24) =>
    get<Timeline>(`/api/learner/${id}/timeline?frames=${frames}`),
  // Interactive what-if: compute statuses for an arbitrary activated set.
  postStatus: (activated: string[]) => post<LearnerStatus>("/api/status", { activated }),
  // Phase 5 validation results (SLAM ablation table + ROC curves).
  getMetrics: () => get<ValidationResults>("/api/metrics"),
};
