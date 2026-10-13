import { describe, expect, it } from "vitest";
import { defaultAttentionWorkload, defaultCollectiveWorkload, defaultGemmWorkload, defaultHardware, defaultMoeWorkload } from "../presets/defaults";
import { analyze } from "../engine/analyze";
import { attentionDraftFromDefaults, collectiveDraftFromDefaults, gemmDraftFromDefaults, hardwareDraftFromDefaults, moeDraftFromDefaults, uiReducer } from "./state";
import type { UiState } from "./state";

describe("UI operator draft state", () => {
  it("resets the switched-to draft while retaining hardware and baseline", () => {
    const workload = defaultGemmWorkload();
    const hardware = defaultHardware();
    const result = analyze(workload, hardware);
    if (!result.valid) throw new Error("expected valid baseline");
    const initial: UiState = {
      operator: "gemm",
      gemmDraft: { ...gemmDraftFromDefaults(workload), m: "19" },
      attentionDraft: { ...attentionDraftFromDefaults(defaultAttentionWorkload()), seq: "77" },
      moeDraft: moeDraftFromDefaults(defaultMoeWorkload()),
      collectiveDraft: collectiveDraftFromDefaults(defaultCollectiveWorkload()),
      hardwareDraft: { ...hardwareDraftFromDefaults(hardware), rankCount: "3" },
      selectedStepId: "step-load",
      inspectorTarget: { type: "resource", id: "res-hbm" },
      baseline: { workload, hardware, modelVersion: result.modelVersion, result },
      targetSecondsDraft: "9",
      inputPanelOpen: true,
    };
    const next = uiReducer(initial, { type: "SET_OPERATOR", operator: "attention" });
    expect(next.attentionDraft.seq).toBe(String(defaultAttentionWorkload().seq));
    expect(next.hardwareDraft.rankCount).toBe("3");
    expect(next.baseline).toBe(initial.baseline);
    expect(next.selectedStepId).toBeNull();
  });
});
