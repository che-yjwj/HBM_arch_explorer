import { describe, expect, it } from "vitest";
import { analyzeCollective } from "./collective";
import { defaultHardware } from "../presets/defaults";
import type { CollectiveWorkload } from "../domain/types";

const MiB = 1024 * 1024;

function hw(rankCount: number, overrides: Partial<ReturnType<typeof defaultHardware>> = {}) {
  return { ...defaultHardware(), rankCount, linkEfficiency: 1, linkBwBytes: 1, linkLatencySeconds: 0, ...overrides };
}

function wl(op: CollectiveWorkload["op"], sBytes: number): CollectiveWorkload {
  return { kind: "collective", precision: "bf16", op, sBytes };
}

describe("Collective analyze — 06 hand-checked expectations", () => {
  it("All-Reduce P=4, S=256MiB -> send 384MiB/rank, 6 rounds", () => {
    const r = analyzeCollective(wl("all-reduce", 256 * MiB), hw(4));
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    expect(r.metrics.find((m) => m.id === "metric-send-bytes")!.value).toBe(384 * MiB);
    expect(r.metrics.find((m) => m.id === "metric-rounds")!.value).toBe(6);
  });

  it("All-Gather P=4, initial shard S=64MiB -> send 192MiB/rank, final tensor 256MiB, 3 rounds", () => {
    const r = analyzeCollective(wl("all-gather", 64 * MiB), hw(4));
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    expect(r.metrics.find((m) => m.id === "metric-send-bytes")!.value).toBe(192 * MiB);
    expect(r.tensors.find((t) => t.id === "tensor-out")!.bytes).toBe(256 * MiB);
    expect(r.metrics.find((m) => m.id === "metric-rounds")!.value).toBe(3);
  });

  it("Reduce-Scatter P=4, S=256MiB -> send 192MiB/rank, output 64MiB, 3 rounds", () => {
    const r = analyzeCollective(wl("reduce-scatter", 256 * MiB), hw(4));
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    expect(r.metrics.find((m) => m.id === "metric-send-bytes")!.value).toBe(192 * MiB);
    expect(r.tensors.find((t) => t.id === "tensor-out")!.bytes).toBe(64 * MiB);
    expect(r.metrics.find((m) => m.id === "metric-rounds")!.value).toBe(3);
  });

  it("All-to-All P=4, S=256MiB (self included) -> send 192MiB/rank, 3 rounds", () => {
    const r = analyzeCollective(wl("all-to-all", 256 * MiB), hw(4));
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    expect(r.metrics.find((m) => m.id === "metric-send-bytes")!.value).toBe(192 * MiB);
    expect(r.metrics.find((m) => m.id === "metric-rounds")!.value).toBe(3);
  });

  it("single rank P=1 -> explicit zero send/recv/rounds/time metrics, not just absent", () => {
    const r = analyzeCollective(wl("all-reduce", 256 * MiB), hw(1));
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    expect(r.totalSeconds).toBe(0);
    expect(r.metrics.find((m) => m.id === "metric-send-bytes")?.value).toBe(0);
    expect(r.metrics.find((m) => m.id === "metric-recv-bytes")?.value).toBe(0);
    expect(r.metrics.find((m) => m.id === "metric-rounds")?.value).toBe(0);
    expect(r.metrics.find((m) => m.id === "metric-total-time")?.value).toBe(0);
    expect(r.steps[0].limitingResource).toBeNull();
  });

  it("non-divisible: All-Reduce P=3, S=256 MiB, q=2 -> error with nearest valid size 268435458", () => {
    const r = analyzeCollective(wl("all-reduce", 256 * MiB), hw(3));
    expect(r.valid).toBe(false);
    if (r.valid) throw new Error("expected invalid");
    expect(r.diagnostics[0].fix).toContain("268435458");
  });

  it("divisible: All-Reduce P=3, S=12 bytes -> shard 4B, send 16B/rank, 4 rounds", () => {
    const r = analyzeCollective(wl("all-reduce", 12), hw(3));
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    expect(r.metrics.find((m) => m.id === "metric-send-bytes")!.value).toBe(16);
    expect(r.metrics.find((m) => m.id === "metric-rounds")!.value).toBe(4);
  });

  it("All-Gather input distinction: P=3, S=8 bytes -> valid initial shard, final 24B, send 16B/rank, 2 rounds", () => {
    const r = analyzeCollective(wl("all-gather", 8), hw(3));
    expect(r.valid).toBe(true);
    if (!r.valid) throw new Error("expected valid");
    expect(r.tensors.find((t) => t.id === "tensor-out")!.bytes).toBe(24);
    expect(r.metrics.find((m) => m.id === "metric-send-bytes")!.value).toBe(16);
    expect(r.metrics.find((m) => m.id === "metric-rounds")!.value).toBe(2);
  });
});
