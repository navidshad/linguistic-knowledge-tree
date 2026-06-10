// Graph rendering, isolated from Vue. Builds the layouts (matrix / concentric /
// force), applies status + CEFR styling, and exposes a handle the GraphCanvas
// component drives. Visual filters are independent CSS classes (any one with
// display:none hides the node), so they compose freely.
import cytoscape from "cytoscape";
import type { Core, ElementDefinition } from "cytoscape";
import { CEFR_COLOR, CEFR_INDEX, CEFR_ORDER, CONFIDENCE_MIN_OPACITY, KGT_EDGE_COLOR, STATUS_COLOR } from "../constants";
import type { Cefr, EdgeAdjustment, LayoutName, Status, SyntaxMap } from "../types";

const COL_W = 250;
const ROW_H = 58;
const LANE_GAP = 50;
const HEADER_Y = -104;
const LABEL_X = -205;

type XY = { x: number; y: number };

function computeMatrixPositions(map: SyntaxMap) {
  const levelIdx = CEFR_INDEX;
  const catOrder = map.categories.map((c) => c.id);
  const byCat: Record<string, SyntaxMap["nodes"]> = {};
  catOrder.forEach((c) => (byCat[c] = []));
  map.nodes.forEach((n) => byCat[n.category]?.push(n));

  const pos: Record<string, XY> = {};
  const laneCenter: Record<string, number> = {};
  let laneTop = 0;

  for (const cat of catOrder) {
    const cells = Object.fromEntries(CEFR_ORDER.map((l) => [l, [] as SyntaxMap["nodes"]])) as Record<Cefr, SyntaxMap["nodes"]>;
    byCat[cat].forEach((n) => cells[n.cefr].push(n));
    const laneH = Math.max(1, ...CEFR_ORDER.map((l) => cells[l].length)) * ROW_H;
    CEFR_ORDER.forEach((l) => {
      const stackH = cells[l].length * ROW_H;
      cells[l].forEach((n, i) => {
        pos[n.id] = { x: levelIdx[l] * COL_W, y: laneTop + (laneH - stackH) / 2 + i * ROW_H };
      });
    });
    laneCenter[cat] = laneTop + laneH / 2 - ROW_H / 2;
    laneTop += laneH + LANE_GAP;
  }
  return { pos, laneCenter };
}

const STYLE = [
  { selector: "node", style: {
      label: "data(label)", "font-size": 9, "text-wrap": "wrap", "text-max-width": "96px",
      "text-valign": "center", "text-halign": "center", shape: "round-rectangle",
      width: 112, height: 44, "border-width": 4, color: "#fff",
  } },
  { selector: 'node[status="known"]', style: { "background-color": STATUS_COLOR.known } },
  { selector: 'node[status="interior_gap"]', style: { "background-color": STATUS_COLOR.interior_gap } },
  { selector: 'node[status="frontier"]', style: { "background-color": STATUS_COLOR.frontier, color: "#1a1a1a" } },
  { selector: 'node[status="further"]', style: { "background-color": STATUS_COLOR.further, color: "#546e7a" } },
  ...CEFR_ORDER.map((l) => ({ selector: `node[cefr="${l}"]`, style: { "border-color": CEFR_COLOR[l] } })),
  // bare-map mode: neutral fill regardless of status (placed AFTER status rules to win)
  { selector: "node.neutral", style: { "background-color": "#eceff1", color: "#607d8b" } },
  // confidence overlay: node opacity tracks mastery (status color unchanged, so
  // color = status and opacity = how strongly the engine believes it).
  { selector: "node.confidence", style: { opacity: `mapData(mastery, 0, 1, ${CONFIDENCE_MIN_OPACITY}, 1)` } },
  { selector: ".pseudo", style: { "background-opacity": 0, "border-width": 0, events: "no", width: 1, height: 1 } },
  { selector: ".header", style: { "font-size": 22, "font-weight": "bold", color: "#455a64" } },
  { selector: ".lane", style: {
      "font-size": 11, color: "#607d8b", "text-halign": "left", "text-valign": "center",
      "text-wrap": "wrap", "text-max-width": "150px", "text-margin-x": -10,
  } },
  { selector: "edge", style: {
      width: 1.4, "line-color": "#d0d7db", "target-arrow-color": "#aebcc2",
      "target-arrow-shape": "triangle", "arrow-scale": 0.8, "curve-style": "bezier", opacity: 0.7,
  } },
  // KGT personal-graph deltas (Phase 7): edge style encodes how this learner's
  // own feedback re-weighted the edge (placed after the base rule to win).
  { selector: "edge.kgt-strong", style: {
      width: 3, "line-color": KGT_EDGE_COLOR.strengthened, "target-arrow-color": KGT_EDGE_COLOR.strengthened, opacity: 1,
  } },
  { selector: "edge.kgt-weak", style: {
      width: 2.5, "line-color": KGT_EDGE_COLOR.weakened, "target-arrow-color": KGT_EDGE_COLOR.weakened,
      "line-style": "dashed", opacity: 0.9,
  } },
  { selector: "edge.kgt-removed", style: {
      width: 2, "line-color": KGT_EDGE_COLOR.removed, "target-arrow-color": KGT_EDGE_COLOR.removed,
      "line-style": "dotted", opacity: 0.45,
  } },
  { selector: "node:selected", style: { "border-color": "#000", "border-width": 5 } },
  // any of these hides the element
  { selector: ".lvl-off", style: { display: "none" } },
  { selector: ".sub-hidden", style: { display: "none" } },
  { selector: ".pseudo-off", style: { display: "none" } },
] as unknown as cytoscape.StylesheetCSS[];

export interface GraphCallbacks {
  onSelect: (nodeId: string | null) => void;
  onToggle: (nodeId: string) => void;
}

export interface GraphHandle {
  cy: Core;
  setStatuses(statuses: Record<string, Status>): void;
  setMastery(mastery: Record<string, number>): void;
  setConfidence(on: boolean): void;
  setOverlay(on: boolean): void;
  setEnabledLevels(levels: Set<Cefr>): void;
  setSubgraphOnly(on: boolean): void;
  setEdgeAdjustments(adjustments: EdgeAdjustment[] | null): void;
  setLayout(name: LayoutName): void;
  fit(): void;
  destroy(): void;
}

const KGT_CLASS: Record<EdgeAdjustment["kind"], string> = {
  strengthened: "kgt-strong",
  weakened: "kgt-weak",
  removed: "kgt-removed",
};

export function createGraph(
  container: HTMLElement,
  map: SyntaxMap,
  statuses: Record<string, Status>,
  cb: GraphCallbacks,
): GraphHandle {
  const { pos, laneCenter } = computeMatrixPositions(map);
  const catLabel = Object.fromEntries(map.categories.map((c) => [c.id, c.label]));

  const matrixPositions: Record<string, XY> = { ...pos };
  CEFR_ORDER.forEach((l) => (matrixPositions["__hdr_" + l] = { x: CEFR_INDEX[l] * COL_W, y: HEADER_Y }));
  map.categories.forEach((c) => (matrixPositions["__lane_" + c.id] = { x: LABEL_X, y: laneCenter[c.id] }));

  const elements: ElementDefinition[] = [
    ...map.nodes.map((n) => ({ data: { id: n.id, label: n.label, cefr: n.cefr, status: statuses[n.id] ?? "further", mastery: 0 } })),
    ...CEFR_ORDER.map((l) => ({ data: { id: "__hdr_" + l, label: l }, classes: "pseudo header", selectable: false, grabbable: false })),
    ...map.categories.map((c) => ({ data: { id: "__lane_" + c.id, label: catLabel[c.id] }, classes: "pseudo lane", selectable: false, grabbable: false })),
    ...map.edges.map((e, i) => ({ data: { id: "e" + i, source: e.source, target: e.target } })),
  ];

  const cy = cytoscape({
    container,
    elements,
    layout: { name: "preset", positions: (n: cytoscape.NodeSingular) => matrixPositions[n.id()], fit: false } as unknown as cytoscape.PresetLayoutOptions,
    style: STYLE,
    wheelSensitivity: 0.25,
  });

  const realNodes = () => cy.nodes().filter((n) => !n.hasClass("pseudo"));
  const fit = () => cy.fit(cy.elements(":visible"), 50);
  fit();

  cy.on("tap", "node", (e) => {
    const id = e.target.id();
    cb.onSelect(id.startsWith("__") ? null : id);
  });
  cy.on("tap", (e) => {
    if (e.target === cy) cb.onSelect(null);
  });
  cy.on("cxttap", "node", (e) => {
    const id = e.target.id();
    if (!id.startsWith("__")) cb.onToggle(id);
  });

  function setLayout(name: LayoutName) {
    if (name === "matrix") {
      cy.nodes(".pseudo").removeClass("pseudo-off");
      cy.layout({ name: "preset", positions: (n: cytoscape.NodeSingular) => matrixPositions[n.id()], fit: false } as unknown as cytoscape.PresetLayoutOptions).run();
    } else {
      cy.nodes(".pseudo").addClass("pseudo-off");
      const eles = realNodes();
      if (name === "concentric") {
        eles.layout({
          name: "concentric",
          concentric: (n: cytoscape.NodeSingular) => 5 - (CEFR_INDEX[n.data("cefr") as Cefr] ?? 0),
          levelWidth: () => 1,
          minNodeSpacing: 18,
          avoidOverlap: true,
          animate: false,
          fit: false,
        } as cytoscape.ConcentricLayoutOptions).run();
      } else {
        eles.layout({ name: "cose", animate: false, fit: false, padding: 40, idealEdgeLength: () => 70, nodeRepulsion: () => 6000 } as cytoscape.CoseLayoutOptions).run();
      }
    }
    fit();
  }

  return {
    cy,
    setStatuses(s) {
      cy.batch(() => {
        Object.entries(s).forEach(([id, st]) => {
          const el = cy.getElementById(id);
          if (el.nonempty()) el.data("status", st);
        });
      });
    },
    setMastery(m) {
      cy.batch(() => {
        realNodes().forEach((n) => {
          n.data("mastery", m[n.id()] ?? 0);
        });
      });
    },
    setConfidence(on) {
      if (on) realNodes().addClass("confidence");
      else realNodes().removeClass("confidence");
    },
    setOverlay(on) {
      if (on) realNodes().removeClass("neutral");
      else realNodes().addClass("neutral");
    },
    setEnabledLevels(levels) {
      cy.batch(() => {
        realNodes().forEach((n) => {
          if (levels.has(n.data("cefr") as Cefr)) n.removeClass("lvl-off");
          else n.addClass("lvl-off");
        });
      });
      fit();
    },
    setSubgraphOnly(on) {
      if (on) realNodes().filter('[status="further"]').addClass("sub-hidden");
      else cy.nodes(".sub-hidden").removeClass("sub-hidden");
      fit();
    },
    setEdgeAdjustments(adjustments) {
      cy.batch(() => {
        cy.edges().removeClass("kgt-strong kgt-weak kgt-removed");
        (adjustments ?? []).forEach((a) => {
          cy.edges(`[source="${a.source}"][target="${a.target}"]`).addClass(KGT_CLASS[a.kind]);
        });
      });
    },
    setLayout,
    fit,
    destroy() {
      cy.destroy();
    },
  };
}
