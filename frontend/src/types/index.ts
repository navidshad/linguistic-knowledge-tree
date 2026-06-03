// Shared types — mirror the backend Pydantic response schemas.

export type Cefr = "A1" | "A2" | "B1" | "B2" | "C1" | "C2";
export type Status = "known" | "interior_gap" | "frontier" | "further";
export type LayoutName = "matrix" | "concentric" | "force";

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

export interface LearnerStatus {
  learner_id: string;
  counts: Partial<Record<Status, number>>;
  statuses: Record<string, Status>;
  // Continuous per-node mastery [0, 1] behind the status (confidence overlay).
  // null for the what-if endpoint (an activated set with no evidence to score).
  mastery: Record<string, number> | null;
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
}
