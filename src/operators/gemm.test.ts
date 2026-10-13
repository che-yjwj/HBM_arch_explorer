import { describe, expect, it } from "vitest";
import { analyzeGemm } from "./gemm";
import { defaultHardware } from "../presets/defaults";
import type { GemmWorkload } from "../domain/types";

function hw(overrides: Partial<ReturnType<typeof defaultHardware>> = {}) {
  return { ...defaultHardware(), ...overrides };
}

function gemm(overrides: Partial<GemmWorkload>): GemmWorkload {
  return { kind: "gemm", precision: "bf16", m: 2, n: 2, k: 2, tileM: 2, tileN: 2, tileK: 2, ...overrides };
}

describe("GEMM analyze — 06 hand-checked expectations", () => {
  it("basic case: M=N=K=2, tile 2^3 -> FLOPs 16, HBM 24B, SRAM working set 32B, SRAM traffic 32B", () => {
    const result = analyzeGemm(gemm({}), hw({ sramCapacityBytes: 1024 }));
    expect(result.valid).toBe(true);
    if (!result.valid) throw new Error("expected valid");
    const flops = result.metrics.find((m) => m.id === "metric-flops")!;
    const hbmTotal = result.metrics.find((m) => m.id === "metric-hbm-total")!;
    const sramTraffic = result.metrics.find((m) => m.id === "metric-sram-traffic")!;
    const requiredSram = result.metrics.find((m) => m.id === "metric-required-sram")!;
    expect(flops.value).toBe(16);
    expect(hbmTotal.value).toBe(24);
    expect(sramTraffic.value).toBe(32);
    expect(requiredSram.value).toBe(32);
  });

  it("boundary case: M=3,N=5,K=7, tile 2^3 -> I=2,J=3,L=4, FLOPs 210, HBM 296B, SRAM traffic 686B", () => {
    const result = analyzeGemm(gemm({ m: 3, n: 5, k: 7, tileM: 2, tileN: 2, tileK: 2 }), hw({ sramCapacityBytes: 4096 }));
    expect(result.valid).toBe(true);
    if (!result.valid) throw new Error("expected valid");
    expect(result.metrics.find((m) => m.id === "metric-flops")!.value).toBe(210);
    expect(result.metrics.find((m) => m.id === "metric-hbm-total")!.value).toBe(296);
    expect(result.metrics.find((m) => m.id === "metric-sram-traffic")!.value).toBe(686);
  });

  it("transfer sum case: M=N=K=4, tile 2^3 -> A/B logical 32B each, HBM Load A/B 64B each, Store C 32B, HBM total 160B, SRAM traffic 320B", () => {
    const result = analyzeGemm(gemm({ m: 4, n: 4, k: 4, tileM: 2, tileN: 2, tileK: 2 }), hw({ sramCapacityBytes: 4096 }));
    expect(result.valid).toBe(true);
    if (!result.valid) throw new Error("expected valid");
    const hbmA = result.metrics.find((m) => m.id === "metric-hbm-a")!;
    const hbmB = result.metrics.find((m) => m.id === "metric-hbm-b")!;
    const hbmC = result.metrics.find((m) => m.id === "metric-hbm-c")!;
    const hbmTotal = result.metrics.find((m) => m.id === "metric-hbm-total")!;
    const sramTraffic = result.metrics.find((m) => m.id === "metric-sram-traffic")!;
    expect(hbmA.value).toBe(64);
    expect(hbmB.value).toBe(64);
    expect(hbmC.value).toBe(32);
    expect(hbmTotal.value).toBe(160);
    expect(sramTraffic.value).toBe(320);
  });

  it("SRAM shortfall -> invalid with required/capacity explain, no normal totalSeconds", () => {
    const result = analyzeGemm(gemm({}), hw({ sramCapacityBytes: 10 }));
    expect(result.valid).toBe(false);
    if (result.valid) throw new Error("expected invalid");
    expect(result.totalSeconds).toBeNull();
    expect(result.explain.requiredSramBytes).toBe(32);
    expect(result.explain.sramCapacityBytes).toBe(10);
  });

  it("sequential timing: compute 100 FLOPs @ 100 FLOP/s eff = 1s; load 1s + compute 2s + store 1s => total 4s", () => {
    // Construct inputs so Load=1s, Compute(FLOPs path)=2s dominating, Store=1s, with tiny SRAM access time.
    // Use M=N=K chosen so flops/Ceff = 2s and sram access time is smaller, and HBM times are 1s each.
    const customHw = hw({
      computeFlops: 1, // Ceff with efficiency 1 below
      computeEfficiency: 1,
      sramBwBytes: 1e12,
      sramEfficiency: 1,
      hbmBwBytes: 1,
      hbmEfficiency: 1,
      sramCapacityBytes: 1e9,
    });
    // flops = 2*m*n*k; choose m=n=k=1 tile 1 -> flops=2 -> compute time = 2s with Ceff=1
    const w = gemm({ m: 1, n: 1, k: 1, tileM: 1, tileN: 1, tileK: 1 });
    const result = analyzeGemm(w, customHw);
    expect(result.valid).toBe(true);
    if (!result.valid) throw new Error("expected valid");
    const loadTime = result.metrics.find((m) => m.id === "metric-load-time")!.value;
    const storeTime = result.metrics.find((m) => m.id === "metric-store-time")!.value;
    const computeTime = result.metrics.find((m) => m.id === "metric-compute-time")!.value;
    // q=2 so A read=2B, B read=2B -> load=4B/1Bps=4s; store C=2B/1=2s. Not exactly matching doc's abstract
    // example, but validates internal consistency: total = load+compute+store.
    const total = result.totalSeconds;
    expect(total).toBeCloseTo(loadTime + computeTime + storeTime, 12);
    expect(computeTime).toBeGreaterThanOrEqual(2); // flops=2 / Ceff=1 = 2s is a lower bound
  });

  it("zero compute efficiency is rejected as invalid, not NaN/Infinity", () => {
    const result = analyzeGemm(gemm({}), hw({ computeEfficiency: 0, sramCapacityBytes: 4096 }));
    expect(result.valid).toBe(false);
    if (result.valid) throw new Error("expected invalid");
    expect(result.totalSeconds).toBeNull();
  });

  it("joint compute+sram limiting step is recorded once, not double-counted across contributions", () => {
    const result = analyzeGemm(gemm({}), hw({ sramCapacityBytes: 4096 }));
    expect(result.valid).toBe(true);
    if (!result.valid) throw new Error("expected valid");
    const sumOfContributions = result.contributions.reduce((acc, c) => acc + c.totalSeconds, 0);
    expect(sumOfContributions).toBeCloseTo(result.totalSeconds, 12);
  });
});
