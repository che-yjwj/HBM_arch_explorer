import type { AttentionWorkload, CollectiveWorkload, GemmWorkload, Hardware, MoeWorkload } from "../domain/types";

// Single definition point for default virtual hardware and workload values (07-implementation-plan.md).
export function defaultHardware(): Hardware {
  return {
    id: "hw-default",
    name: "가상 Hardware",
    rankCount: 4,
    computeFlops: 200e12, // 200 TFLOP/s
    computeEfficiency: 0.7,
    sramCapacityBytes: 32 * 1024 * 1024, // 32 MiB
    sramBwBytes: 10000e9, // 10,000 GB/s
    sramEfficiency: 0.7,
    hbmBwBytes: 1000e9, // 1,000 GB/s
    hbmEfficiency: 0.7,
    linkBwBytes: 100e9, // 100 GB/s per rank
    linkEfficiency: 0.7,
    linkLatencySeconds: 2e-6, // 2 microseconds
  };
}

export function defaultGemmWorkload(): GemmWorkload {
  return { kind: "gemm", precision: "bf16", m: 4096, n: 4096, k: 4096, tileM: 128, tileN: 128, tileK: 128 };
}

export function defaultAttentionWorkload(): AttentionWorkload {
  return { kind: "attention", precision: "bf16", batch: 1, heads: 32, seq: 1024, headDim: 128 };
}

export function defaultMoeWorkload(): MoeWorkload {
  return { kind: "moe", precision: "bf16", tokens: 1024, hidden: 4096, ffn: 14336, experts: 8, topK: 2 };
}

export function defaultCollectiveWorkload(): CollectiveWorkload {
  return { kind: "collective", precision: "bf16", op: "all-reduce", sBytes: 256 * 1024 * 1024 };
}
