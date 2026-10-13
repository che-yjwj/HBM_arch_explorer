import type { BaselineSnapshot, CollectiveOp, OperatorKind } from "../domain/types";
import { defaultAttentionWorkload, defaultCollectiveWorkload, defaultGemmWorkload, defaultHardware, defaultMoeWorkload } from "../presets/defaults";

// UI state limited to: input drafts, selected step, one inspector target, baseline, target time.
// (07-implementation-plan.md "UI 상태와 재계산")

export type InspectorTarget =
  | { type: "resource"; id: string }
  | { type: "transfer"; id: string }
  | { type: "metric"; id: string }
  | null;

export interface GemmDraft {
  m: string;
  n: string;
  k: string;
  tileM: string;
  tileN: string;
  tileK: string;
}

export interface AttentionDraft {
  batch: string;
  heads: string;
  seq: string;
  headDim: string;
}

export interface MoeDraft {
  tokens: string;
  hidden: string;
  ffn: string;
  experts: string;
  topK: string;
}

export interface CollectiveDraft {
  op: CollectiveOp;
  sBytes: string;
}

export interface HardwareDraft {
  rankCount: string;
  computeFlops: string;
  computeEfficiency: string;
  sramCapacityBytes: string;
  sramBwBytes: string;
  sramEfficiency: string;
  hbmBwBytes: string;
  hbmEfficiency: string;
  linkBwBytes: string;
  linkEfficiency: string;
  linkLatencySeconds: string;
}

export interface UiState {
  operator: OperatorKind;
  gemmDraft: GemmDraft;
  attentionDraft: AttentionDraft;
  moeDraft: MoeDraft;
  collectiveDraft: CollectiveDraft;
  hardwareDraft: HardwareDraft;
  selectedStepId: string | null;
  inspectorTarget: InspectorTarget;
  baseline: BaselineSnapshot | null;
  targetSecondsDraft: string;
  inputPanelOpen: boolean;
}

export type UiAction =
  | { type: "SET_OPERATOR"; operator: OperatorKind }
  | { type: "SET_GEMM_FIELD"; field: keyof GemmDraft; value: string }
  | { type: "SET_ATTENTION_FIELD"; field: keyof AttentionDraft; value: string }
  | { type: "SET_MOE_FIELD"; field: keyof MoeDraft; value: string }
  | { type: "SET_COLLECTIVE_FIELD"; field: keyof CollectiveDraft; value: string }
  | { type: "SET_HARDWARE_FIELD"; field: keyof HardwareDraft; value: string }
  | { type: "SELECT_STEP"; stepId: string | null }
  | { type: "SET_INSPECTOR"; target: InspectorTarget }
  | { type: "RESTORE_WORKLOAD_DEFAULTS" }
  | { type: "RESTORE_HARDWARE_DEFAULTS" }
  | { type: "SET_BASELINE"; snapshot: BaselineSnapshot }
  | { type: "CLEAR_BASELINE" }
  | { type: "SET_TARGET"; value: string }
  | { type: "TOGGLE_INPUT_PANEL" }
  | { type: "VALIDATE_SELECTION"; validStepIds: string[]; firstStepId: string | null };

export function gemmDraftFromDefaults(d: { m: number; n: number; k: number; tileM: number; tileN: number; tileK: number }): GemmDraft {
  return { m: String(d.m), n: String(d.n), k: String(d.k), tileM: String(d.tileM), tileN: String(d.tileN), tileK: String(d.tileK) };
}

export function attentionDraftFromDefaults(d: { batch: number; heads: number; seq: number; headDim: number }): AttentionDraft {
  return { batch: String(d.batch), heads: String(d.heads), seq: String(d.seq), headDim: String(d.headDim) };
}

export function moeDraftFromDefaults(d: { tokens: number; hidden: number; ffn: number; experts: number; topK: number }): MoeDraft {
  return { tokens: String(d.tokens), hidden: String(d.hidden), ffn: String(d.ffn), experts: String(d.experts), topK: String(d.topK) };
}

export function collectiveDraftFromDefaults(d: { op: CollectiveOp; sBytes: number }): CollectiveDraft {
  return { op: d.op, sBytes: `${d.sBytes} B` };
}

export function hardwareDraftFromDefaults(hw: {
  rankCount: number;
  computeFlops: number;
  computeEfficiency: number;
  sramCapacityBytes: number;
  sramBwBytes: number;
  sramEfficiency: number;
  hbmBwBytes: number;
  hbmEfficiency: number;
  linkBwBytes: number;
  linkEfficiency: number;
  linkLatencySeconds: number;
}): HardwareDraft {
  return {
    rankCount: String(hw.rankCount),
    computeFlops: String(hw.computeFlops),
    computeEfficiency: String(hw.computeEfficiency),
    sramCapacityBytes: String(hw.sramCapacityBytes),
    sramBwBytes: String(hw.sramBwBytes),
    sramEfficiency: String(hw.sramEfficiency),
    hbmBwBytes: String(hw.hbmBwBytes),
    hbmEfficiency: String(hw.hbmEfficiency),
    linkBwBytes: String(hw.linkBwBytes),
    linkEfficiency: String(hw.linkEfficiency),
    linkLatencySeconds: String(hw.linkLatencySeconds),
  };
}

export function uiReducer(state: UiState, action: UiAction): UiState {
  switch (action.type) {
    case "SET_OPERATOR":
      return {
        ...state,
        operator: action.operator,
        gemmDraft: action.operator === "gemm" ? gemmDraftFromDefaults(defaultGemmWorkload()) : state.gemmDraft,
        attentionDraft: action.operator === "attention" ? attentionDraftFromDefaults(defaultAttentionWorkload()) : state.attentionDraft,
        moeDraft: action.operator === "moe" ? moeDraftFromDefaults(defaultMoeWorkload()) : state.moeDraft,
        collectiveDraft: action.operator === "collective" ? collectiveDraftFromDefaults(defaultCollectiveWorkload()) : state.collectiveDraft,
        selectedStepId: null,
        inspectorTarget: null,
      };
    case "SET_GEMM_FIELD":
      return { ...state, gemmDraft: { ...state.gemmDraft, [action.field]: action.value } };
    case "SET_ATTENTION_FIELD":
      return { ...state, attentionDraft: { ...state.attentionDraft, [action.field]: action.value } };
    case "SET_MOE_FIELD":
      return { ...state, moeDraft: { ...state.moeDraft, [action.field]: action.value } };
    case "SET_COLLECTIVE_FIELD":
      return { ...state, collectiveDraft: { ...state.collectiveDraft, [action.field]: action.value } };
    case "SET_HARDWARE_FIELD":
      return { ...state, hardwareDraft: { ...state.hardwareDraft, [action.field]: action.value } };
    case "SELECT_STEP":
      return { ...state, selectedStepId: action.stepId, inspectorTarget: null };
    case "SET_INSPECTOR":
      return { ...state, inspectorTarget: action.target };
    case "RESTORE_WORKLOAD_DEFAULTS":
      if (state.operator === "gemm") return { ...state, gemmDraft: gemmDraftFromDefaults(defaultGemmWorkload()) };
      if (state.operator === "attention") return { ...state, attentionDraft: attentionDraftFromDefaults(defaultAttentionWorkload()) };
      if (state.operator === "moe") return { ...state, moeDraft: moeDraftFromDefaults(defaultMoeWorkload()) };
      return { ...state, collectiveDraft: collectiveDraftFromDefaults(defaultCollectiveWorkload()) };
    case "RESTORE_HARDWARE_DEFAULTS":
      return { ...state, hardwareDraft: hardwareDraftFromDefaults(defaultHardware()) };
    case "SET_BASELINE":
      return { ...state, baseline: action.snapshot };
    case "CLEAR_BASELINE":
      return { ...state, baseline: null };
    case "SET_TARGET":
      return { ...state, targetSecondsDraft: action.value };
    case "TOGGLE_INPUT_PANEL":
      return { ...state, inputPanelOpen: !state.inputPanelOpen };
    case "VALIDATE_SELECTION": {
      if (state.selectedStepId && action.validStepIds.includes(state.selectedStepId)) return state;
      return { ...state, selectedStepId: action.firstStepId, inspectorTarget: null };
    }
    default:
      return state;
  }
}
