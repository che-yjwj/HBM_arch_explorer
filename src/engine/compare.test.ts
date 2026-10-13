import { describe, expect, it } from "vitest";
import { defaultCollectiveWorkload, defaultGemmWorkload, defaultHardware } from "../presets/defaults";
import { analyze } from "./analyze";
import { compare } from "./compare";

describe("baseline comparison boundaries", () => {
  it("keeps valid result comparison while explicitly stopping an invalid target judgment", () => {
    const workload = defaultGemmWorkload();
    const hardware = defaultHardware();
    const result = analyze(workload, hardware);
    if (!result.valid) throw new Error("expected valid baseline");
    const compared = compare({ workload, hardware, modelVersion: result.modelVersion, result }, workload, hardware, result, null, true);
    expect(compared.compatible).toBe(true);
    expect(compared.absoluteDeltaSeconds).toBe(0);
    expect(compared.targetInvalid).toBe(true);
    expect(compared.targetMet).toBeNull();
  });

  it("marks a Collective algorithm change incompatible", () => {
    const hardware = defaultHardware();
    const workload = defaultCollectiveWorkload();
    const baselineResult = analyze(workload, hardware);
    const currentWorkload = { ...workload, op: "all-gather" as const };
    const currentResult = analyze(currentWorkload, hardware);
    if (!baselineResult.valid || !currentResult.valid) throw new Error("expected valid Collective results");
    const compared = compare({ workload, hardware, modelVersion: baselineResult.modelVersion, result: baselineResult }, currentWorkload, hardware, currentResult, null);
    expect(compared.compatible).toBe(false);
    expect(compared.incompatibleReason).toContain("Collective");
  });
});
