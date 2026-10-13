import type { CoreAnalysisResult, Hardware, Workload } from "../domain/types";
import { analyzeGemm } from "../operators/gemm";
import { analyzeAttention } from "../operators/attention";
import { analyzeMoe } from "../operators/moe";
import { analyzeCollective } from "../operators/collective";
import { MAX_RANK_COUNT, SAFE_INT_MAX } from "./numeric";
import { validateAnalysisContract } from "./contract";

const MODEL_VERSION: Record<Workload["kind"], string> = {
  gemm: "gemm-v1",
  attention: "attention-v1",
  moe: "moe-v1",
  collective: "collective-v1",
};

function invalidHardwareMessage(hw: Hardware): string | null {
  if (!Number.isSafeInteger(hw.rankCount) || hw.rankCount < 1 || hw.rankCount > MAX_RANK_COUNT) {
    return `Rank 수는 1~${MAX_RANK_COUNT} 범위의 안전 정수여야 합니다`;
  }
  const positive: Array<[string, number]> = [
    ["Compute 성능", hw.computeFlops], ["SRAM 용량", hw.sramCapacityBytes],
    ["SRAM BW", hw.sramBwBytes], ["HBM BW", hw.hbmBwBytes], ["Link BW", hw.linkBwBytes],
  ];
  for (const [label, value] of positive) {
    if (!Number.isFinite(value) || value <= 0) return `${label}은 양의 유한수여야 합니다`;
  }
  if (!Number.isSafeInteger(hw.sramCapacityBytes) || hw.sramCapacityBytes > SAFE_INT_MAX) {
    return "SRAM 용량은 안전 정수 byte여야 합니다";
  }
  const efficiencies: Array<[string, number]> = [
    ["Compute 효율", hw.computeEfficiency], ["SRAM 효율", hw.sramEfficiency],
    ["HBM 효율", hw.hbmEfficiency], ["Link 효율", hw.linkEfficiency],
  ];
  for (const [label, value] of efficiencies) {
    if (!Number.isFinite(value) || value <= 0 || value > 1) return `${label}은 0 초과 1 이하여야 합니다`;
  }
  if (!Number.isFinite(hw.linkLatencySeconds) || hw.linkLatencySeconds < 0) {
    return "Link latency는 0 이상의 유한수여야 합니다";
  }
  return null;
}

function invalidCore(workload: Workload, message: string): CoreAnalysisResult {
  return {
    valid: false,
    modelVersion: MODEL_VERSION[workload.kind],
    resources: [], tensors: [], steps: [], transfers: [], metrics: [], formulas: [], assumptions: [],
    diagnostics: [{ level: "error", message, relatedInputIds: ["hardware"] }],
    totalSeconds: null,
    explain: {},
  };
}

function validateCalculatedResult(workload: Workload, hardware: Hardware, result: CoreAnalysisResult): CoreAnalysisResult {
  if (!result.valid) return result;
  for (const tensor of result.tensors) {
    if (!Number.isFinite(tensor.bytes) || tensor.bytes < 0 || tensor.bytes > SAFE_INT_MAX || tensor.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 0)) {
      return invalidCore(workload, `${tensor.label}: Tensor 크기/shape가 안전한 비음수 값이 아닙니다`);
    }
  }
  for (const metric of result.metrics) {
    if (!Number.isFinite(metric.value) || metric.value < 0) {
      return invalidCore(workload, `${metric.label}: 계산값이 유한한 0 이상 값이 아닙니다`);
    }
    if ((metric.unit === "byte" || metric.unit === "flop" || metric.unit === "count") && metric.value > SAFE_INT_MAX) {
      return invalidCore(workload, `${metric.label}: 계산 결과가 안전 정수 범위를 초과했습니다`);
    }
    if (metric.unit === "count" && !Number.isSafeInteger(metric.value)) {
      return invalidCore(workload, `${metric.label}: 계산 결과가 안전 정수가 아닙니다`);
    }
  }
  const timeById = new Map(result.metrics.filter((m) => m.unit === "s").map((m) => [m.id, m.value]));
  for (const step of result.steps) {
    const seconds = timeById.get(step.timeMetricId);
    if (seconds === undefined) return invalidCore(workload, `단계 ${step.label}의 시간 Metric을 찾을 수 없습니다`);
    if (seconds === 0) {
      const allowedNoWork = (workload.kind === "collective" && hardware.rankCount === 1)
        || (workload.kind === "moe" && hardware.rankCount === 1 && (step.id === "step-dispatch" || step.id === "step-combine"));
      if (!allowedNoWork) return invalidCore(workload, `${step.label}: 양의 작업 시간 계산이 0으로 underflow했습니다`);
    }
  }
  if (!Number.isFinite(result.totalSeconds) || result.totalSeconds < 0) {
    return invalidCore(workload, "총시간 계산이 유한한 0 이상 값이 아닙니다");
  }
  const contractIssues = validateAnalysisContract(result);
  if (contractIssues.length > 0) return invalidCore(workload, `결과 계약 오류: ${contractIssues[0].message}`);
  return result;
}

/** Operator dispatch. Does not call evaluate() or sensitivity() — no recursion (07-implementation-plan.md). */
export function analyze(workload: Workload, hardware: Hardware): CoreAnalysisResult {
  const hardwareError = invalidHardwareMessage(hardware);
  if (hardwareError) return invalidCore(workload, hardwareError);
  let result: CoreAnalysisResult;
  switch (workload.kind) {
    case "gemm":
      result = analyzeGemm(workload, hardware);
      break;
    case "attention":
      result = analyzeAttention(workload, hardware);
      break;
    case "moe":
      result = analyzeMoe(workload, hardware);
      break;
    case "collective":
      result = analyzeCollective(workload, hardware);
      break;
  }
  return validateCalculatedResult(workload, hardware, result);
}
