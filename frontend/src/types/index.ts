// Shared types — mirror the backend Pydantic response schemas.

export type Cefr = "A1" | "A2" | "B1" | "B2" | "C1" | "C2";
export type Status = "known" | "interior_gap" | "frontier" | "further";
export type LayoutName = "matrix" | "concentric" | "force";
export type Tab = "map" | "metrics" | "chat";

export interface Category {
  id: string;
  label: string;
}

export interface MapNode {
  id: string;
  label: string;
  category: string;
  cefr: Cefr;
  description: string;
}

export interface MapEdge {
  source: string;
  target: string;
  type: string;
}

export interface SyntaxMap {
  meta: Record<string, unknown>;
  categories: Category[];
  nodes: MapNode[];
  edges: MapEdge[];
}

// Phase 7 (RQ5): one KGT re-weighting in the learner's personal graph.
export interface EdgeAdjustment {
  source: string; // the prerequisite
  target: string; // the dependent
  factor_back: number; // multiplier on the backward inference message
  factor_fwd: number; // multiplier on the forward readiness message
  kind: "strengthened" | "weakened" | "removed";
  reason: string; // human-readable, cites the learner's evidence
}

export interface LearnerStatus {
  learner_id: string;
  counts: Partial<Record<Status, number>>;
  statuses: Record<string, Status>;
  // Continuous per-node mastery [0, 1] behind the status (confidence overlay).
  // null for the what-if endpoint (an activated set with no evidence to score).
  mastery: Record<string, number> | null;
  // §3.7 handoff: priority for interior-gap/frontier nodes (null in what-if mode).
  gap_scores?: Record<string, number> | null;
  // Personal-graph deltas — only present when requested with kgt=1.
  edge_adjustments?: EdgeAdjustment[] | null;
}

// A learner's knowledge state at one point in time — the timeline scrubber frame.
export interface TimelineFrame {
  t: number;
  counts: Partial<Record<Status, number>>;
  statuses: Record<string, Status>;
  mastery: Record<string, number>;
}

export interface Timeline {
  learner_id: string;
  t_start: number;
  t_end: number;
  frames: TimelineFrame[];
}

export interface LearnerProfile {
  id: string;
  label: string;
  description: string;
  editable?: boolean; // built-ins read-only; file-backed user profiles editable
}

// Create a persistent profile (optionally seeded to a starting CEFR band).
export interface ProfileCreate {
  name: string;
  seed_level?: Cefr | null;
}

// One piece of evidence appended to a profile (e.g. marking a node known).
export interface ProfileEvent {
  node_ids: string[];
  correct: boolean;
  source: "review" | "dialog" | "exposure";
}

export interface Conversation {
  profile_id: string;
  messages: ChatTurn[];
}

// --- Phase 5: validation metrics (Duolingo SLAM) ---

export interface MetricSet {
  n: number;
  base_rate: number;
  auroc: number;
  F1: number;
  accuracy: number;
  avglogloss: number;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
}

export interface ModelResult {
  name: string;
  group: string;
  label: string;
  metrics: MetricSet;
  metrics_cold: MetricSet | null;
  roc: RocPoint[];
  // Phase 7 (--kgt runs): measured fit+predict compute cost per model.
  cost?: { fit_predict_seconds: number; seconds_per_learner: number } | null;
  // Per-epoch mean train loss (the retrain arm only) — the convergence plot.
  retrain_curve?: { epoch: number; loss: number }[] | null;
}

export interface ValidationDataset {
  source: string;
  course: string;
  split: string;
  n_learners: number;
  n_eval_instances: number;
  n_cold_instances: number;
  mistake_base_rate: number;
  node_coverage: number;
}

export interface ValidationResults {
  dataset: ValidationDataset;
  rqs: Record<string, string[]>;
  models: ModelResult[];
  meta?: Record<string, unknown>;
}

// --- Phase 6: Gemini chat demo (dialog turns activate the map) ---

export interface ChatTurn {
  role: "user" | "tutor";
  text: string;
}

export interface NodeEvidence {
  node_id: string;
  confidence: number;
  turn_indices: number[];
  incorrect_turn_indices: number[]; // subset graded as wrong usage
}

export interface ChatResponse {
  reply: string;
  mapped_nodes: string[];
  confidences: Record<string, number>;
  grades: Record<string, boolean>; // node -> used correctly? (latest turn)
  counts: Partial<Record<Status, number>>;
  statuses: Record<string, Status>;
  mastery: Record<string, number>;
  evidence: NodeEvidence[];
  // Phase 7: KGT live on the conversation (wrong usage weakens inference edges).
  edge_adjustments?: EdgeAdjustment[] | null;
}

// --- Phase 7: live per-learner retrain (the RQ5 demo) ---

export interface RetrainEpoch {
  epoch: number;
  loss: number;
  edge_adjustments: EdgeAdjustment[];
}

export interface RetrainResult {
  learner_id: string;
  n_items: number;
  wall_ms: number; // the full gradient fit
  kgt_wall_ms: number; // closed-form KGT on the same events
  epochs: RetrainEpoch[];
}
