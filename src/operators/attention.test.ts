import { describe, expect, it } from "vitest";
import { analyzeAttention } from "./attention";
import { defaultHardware } from "../presets/defaults";
import type { AttentionWorkload } from "../domain/types";

function hw(overrides: Partial<ReturnType<typeof defaultHardware>> = {}) {
  return { ...defaultHardware(), ...overrides };
}

function wl(overrides: Partial<AttentionWorkload> = {}): AttentionWorkload {
  return { kind: "attention", precision: "bf16", batch: 1, heads: 1, seq: 2, headDim: 2, ...overrides };
}

describe("Attention analyze — 06 hand-checked expectations", () => {
  it("B=H=1,S=D=2 -> Matmul (QK+PV) 32 FLOPs, Softmax 20 FLOPs, HBM 96 bytes", () => {
    const r = analyzeAttention(wl(), hw());
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    const qk = r.metrics.find((m) => m.id === "metric-qk-flops")!.value;
    const pv = r.metrics.find((m) => m.id === "metric-pv-flops")!.value;
    expect(qk + pv).toBe(32);
    expect(r.metrics.find((m) => m.id === "metric-softmax-flops")!.value).toBe(20);
    expect(r.metrics.find((m) => m.id === "metric-hbm-total")!.value).toBe(96);
  });

  it("throughput approximation with C_eff=100 FLOP/s -> QK 0.16s, Softmax 0.20s, PV 0.16s", () => {
    const r = analyzeAttention(wl(), hw({ computeFlops: 100, computeEfficiency: 1 }));
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    expect(r.metrics.find((m) => m.id === "metric-qk-time")!.value).toBeCloseTo(0.16, 10);
    expect(r.metrics.find((m) => m.id === "metric-softmax-time")!.value).toBeCloseTo(0.2, 10);
    expect(r.metrics.find((m) => m.id === "metric-pv-time")!.value).toBeCloseTo(0.16, 10);
  });

  it("Q/K/V are distinct transfers with their own tensor and byte metric, not merged into one", () => {
    const r = analyzeAttention(wl(), hw());
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    const q = r.transfers.find((t) => t.id === "transfer-q")!;
    const k = r.transfers.find((t) => t.id === "transfer-k")!;
    const v = r.transfers.find((t) => t.id === "transfer-v")!;
    expect(q.tensorId).toBe("tensor-q");
    expect(k.tensorId).toBe("tensor-k");
    expect(v.tensorId).toBe("tensor-v");
    expect(new Set([q.bytesMetricId, k.bytesMetricId, v.bytesMetricId]).size).toBe(3);
  });

  it("Score and Probability Store/Load transfers use byte metrics in opposite directions", () => {
    const r = analyzeAttention(wl(), hw());
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    for (const [stem, tensorId] of [["score", "tensor-score"], ["prob", "tensor-prob"]] as const) {
      const store = r.transfers.find((transfer) => transfer.id === `transfer-${stem}-store`)!;
      const load = r.transfers.find((transfer) => transfer.id === `transfer-${stem}-load`)!;
      const storeMetric = r.metrics.find((metric) => metric.id === store.bytesMetricId)!;
      const loadMetric = r.metrics.find((metric) => metric.id === load.bytesMetricId)!;
      expect(store.tensorId).toBe(tensorId);
      expect(load.tensorId).toBe(tensorId);
      expect(storeMetric.unit).toBe("byte");
      expect(loadMetric.unit).toBe("byte");
      expect(store.from).toBe("res-compute");
      expect(store.to).toBe("res-hbm");
      expect(load.from).toBe("res-hbm");
      expect(load.to).toBe("res-compute");
    }
  });

  it("SRAM sensitivity is unsupported for Attention", async () => {
    const { SUPPORTED_SENSITIVITY_RESOURCES } = await import("./attention");
    expect(SUPPORTED_SENSITIVITY_RESOURCES).not.toContain("sram");
  });
});
