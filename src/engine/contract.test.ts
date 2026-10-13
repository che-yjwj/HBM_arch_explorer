import { describe, expect, it } from "vitest";
import type { CoreAnalysisResult, ValidCoreAnalysisResult } from "../domain/types";
import { defaultAttentionWorkload, defaultCollectiveWorkload, defaultGemmWorkload, defaultHardware, defaultMoeWorkload } from "../presets/defaults";
import { analyze } from "./analyze";
import { validateAnalysisContract } from "./contract";

describe("analysis result contracts", () => {
  it("all operator defaults have valid references and acyclic dependencies", () => {
    for (const workload of [defaultGemmWorkload(), defaultAttentionWorkload(), defaultMoeWorkload(), defaultCollectiveWorkload()]) {
      const result = analyze(workload, defaultHardware());
      expect(result.valid, `${workload.kind}: ${JSON.stringify(result.diagnostics)}`).toBe(true);
      expect(validateAnalysisContract(result)).toEqual([]);
    }
  });

  it("detects formula and step cycles", () => {
    const valid = analyze(defaultGemmWorkload(), defaultHardware()) as ValidCoreAnalysisResult;
    const broken = structuredClone(valid) as CoreAnalysisResult & { valid: true };
    const mA = broken.metrics.find((m) => m.id === "metric-flops")!;
    const mB = broken.metrics.find((m) => m.id === "metric-total-time")!;
    const fA = broken.formulas.find((f) => f.id === mA.formulaId)!;
    const fB = broken.formulas.find((f) => f.id === mB.formulaId)!;
    fA.inputMetricIds = [mB.id];
    fB.inputMetricIds = [mA.id];
    broken.steps[0].precedingStepIds = [broken.steps[2].id];
    const issues = validateAnalysisContract(broken);
    expect(issues.some((issue) => issue.code === "formula-cycle")).toBe(true);
    expect(issues.some((issue) => issue.code === "step-cycle")).toBe(true);
  });

  it("rejects dangling assumption provenance", () => {
    const valid = analyze(defaultGemmWorkload(), defaultHardware()) as ValidCoreAnalysisResult;
    const broken = structuredClone(valid);
    broken.formulas.find((formula) => formula.assumptionIds.length > 0)!.assumptionIds = ["missing-assumption"];
    expect(validateAnalysisContract(broken).some((issue) => issue.code === "missing-assumption")).toBe(true);
  });
});
