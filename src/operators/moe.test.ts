import { describe, expect, it } from "vitest";
import { analyzeMoe } from "./moe";
import { defaultHardware } from "../presets/defaults";
import type { MoeWorkload } from "../domain/types";

function hw(rankCount: number, overrides: Partial<ReturnType<typeof defaultHardware>> = {}) {
  return { ...defaultHardware(), rankCount, ...overrides };
}

function wl(overrides: Partial<MoeWorkload> = {}): MoeWorkload {
  return { kind: "moe", precision: "bf16", tokens: 8, hidden: 2, ffn: 4, experts: 2, topK: 1, ...overrides };
}

describe("MoE analyze — 06 hand-checked expectations", () => {
  it("T=8,D=2,F=4,E=P=2,k=1,q=2 -> Router 64 FLOPs (total), Expert 256 FLOPs (total), Dispatch/Combine 8 bytes/rank", () => {
    const r = analyzeMoe(wl(), hw(2));
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    expect(r.metrics.find((m) => m.id === "metric-router-flops-total")!.value).toBe(64);
    expect(r.metrics.find((m) => m.id === "metric-expert-flops-total")!.value).toBe(256);
    expect(r.metrics.find((m) => m.id === "metric-dispatch-bytes")!.value).toBe(8);
    expect(r.metrics.find((m) => m.id === "metric-combine-bytes")!.value).toBe(8);
  });

  it("E mod P != 0 is rejected", () => {
    const r = analyzeMoe(wl({ experts: 3 }), hw(2));
    expect(r.valid).toBe(false);
  });

  it("Top-k > Experts is rejected", () => {
    const r = analyzeMoe(wl({ topK: 5, experts: 2 }), hw(2));
    expect(r.valid).toBe(false);
  });
});
