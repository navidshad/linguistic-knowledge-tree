// Shared visual constants — decoupled from cytoscape so stores/components can
// import them without pulling in the graph library.
import type { Cefr, Status } from "./types";

export const CEFR_ORDER: Cefr[] = ["A1", "A2", "B1", "B2", "C1", "C2"];

export const CEFR_INDEX: Record<Cefr, number> = { A1: 0, A2: 1, B1: 2, B2: 3, C1: 4, C2: 5 };

export const CEFR_COLOR: Record<Cefr, string> = {
  A1: "#4fc3f7", A2: "#29b6f6", B1: "#0288d1", B2: "#01579b", C1: "#5e35b1", C2: "#311b92",
};

export const STATUS_COLOR: Record<Status, string> = {
  known: "#2e7d32", interior_gap: "#c62828", frontier: "#f9a825", further: "#cfd8dc",
};

export const STATUS_LABEL: Record<Status, string> = {
  known: "Known", interior_gap: "Interior gap", frontier: "Frontier", further: "Further",
};

// Confidence overlay maps mastery [0, 1] -> node opacity [MIN, 1], so even a
// zero-evidence node stays faintly visible rather than vanishing.
export const CONFIDENCE_MIN_OPACITY = 0.15;

// KGT personal-graph edge colors (Phase 7): reinforced vs weakened/cut edges.
export const KGT_EDGE_COLOR = {
  strengthened: "#2e7d32",
  weakened: "#c62828",
  removed: "#c62828",
} as const;

// What-if mode has no evidence to score, so approximate confidence from the
// discrete status, letting the opacity overlay degrade gracefully there.
export const STATUS_MASTERY: Record<Status, number> = {
  known: 1, interior_gap: 0.4, frontier: 0.2, further: 0.05,
};
