import type { BaselineSnapshot, Hardware, LimitingResource, ValidCoreAnalysisResult, Workload } from "../domain/types";
import { addFinite } from "./numeric";

export interface CompareResult {
  compatible: boolean;
  incompatibleReason: string | null;
  totalSecondsBaseline: number;
  totalSecondsCurrent: number | null; // null when current is invalid
  absoluteDeltaSeconds: number | null;
  relativeDeltaRatio: number | null; // null if baseline is 0 or incompatible
  workloadChanged: boolean;
  executionChanged: boolean;
  changedHardwareSpecs: string[];
  hbmTrafficBaseline: number | null;
  hbmTrafficCurrent: number | null;
  linkTrafficBaseline: number | null;
  linkTrafficCurrent: number | null;
  sramRequiredBaseline: number | null;
  sramRequiredCurrent: number | null;
  limitingResourcesBaseline: LimitingResource[];
  limitingResourcesCurrent: LimitingResource[];
  targetSeconds: number | null;
  targetInvalid: boolean;
  targetMet: boolean | null; // null if no target or current invalid
}

function sameWorkloadShape(a: Workload, b: Workload): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "gemm" && b.kind === "gemm") {
    return a.m === b.m && a.n === b.n && a.k === b.k;
  }
  if (a.kind === "attention" && b.kind === "attention") {
    return a.batch === b.batch && a.heads === b.heads && a.seq === b.seq && a.headDim === b.headDim;
  }
  if (a.kind === "moe" && b.kind === "moe") {
    return a.tokens === b.tokens && a.hidden === b.hidden && a.ffn === b.ffn && a.experts === b.experts && a.topK === b.topK;
  }
  if (a.kind === "collective" && b.kind === "collective") {
    return a.op === b.op && a.sBytes === b.sBytes;
  }
  return true;
}

export function compare(
  baseline: BaselineSnapshot,
  currentWorkload: Workload,
  currentHardware: Hardware,
  currentResult: ValidCoreAnalysisResult | null,
  targetSeconds: number | null,
  targetInvalid = false,
): CompareResult {
  const sameOperator = baseline.workload.kind === currentWorkload.kind;
  const sameModelVersion = currentResult ? baseline.modelVersion === currentResult.modelVersion : true;
  const incompatibleAlgorithm = sameOperator && baseline.workload.kind === "collective" && currentWorkload.kind === "collective" && baseline.workload.op !== currentWorkload.op;
  const compatible = sameOperator && sameModelVersion && !incompatibleAlgorithm;

  // Rank count changes total work for MoE/Collective (rank-dependent models), so count as a workload change there.
  const rankSensitive = currentWorkload.kind === "moe" || currentWorkload.kind === "collective";
  const rankChanged = rankSensitive && baseline.hardware.rankCount !== currentHardware.rankCount;

  const workloadChanged = sameOperator && (!sameWorkloadShape(baseline.workload, currentWorkload) || rankChanged);
  const executionChanged = sameOperator && baseline.workload.kind === "gemm" && currentWorkload.kind === "gemm"
    && (baseline.workload.tileM !== currentWorkload.tileM || baseline.workload.tileN !== currentWorkload.tileN || baseline.workload.tileK !== currentWorkload.tileK);

  const hardwareFields: Array<[keyof Hardware, string]> = [
    ["computeFlops", "Compute 성능"], ["computeEfficiency", "Compute 효율"],
    ["sramCapacityBytes", "SRAM 용량"], ["sramBwBytes", "SRAM BW"], ["sramEfficiency", "SRAM 효율"],
    ["hbmBwBytes", "HBM BW"], ["hbmEfficiency", "HBM 효율"],
    ["linkBwBytes", "Link BW"], ["linkEfficiency", "Link 효율"], ["linkLatencySeconds", "Link latency"],
    ["rankCount", "Rank 수"],
  ];
  const changedHardwareSpecs = hardwareFields.filter(([field]) => baseline.hardware[field] !== currentHardware[field]).map(([, label]) => label);

  const traffic = (result: ValidCoreAnalysisResult, trafficClass: "hbm" | "link"): number | null => result.transfers
    .filter((transfer) => transfer.trafficClass === trafficClass)
    .reduce<number | null>((sum, transfer) => sum === null ? null : addFinite(sum, result.metrics.find((metric) => metric.id === transfer.bytesMetricId)?.value ?? 0), 0);
  const metricValue = (result: ValidCoreAnalysisResult, id: string) => result.metrics.find((metric) => metric.id === id)?.value ?? null;

  const totalSecondsBaseline = baseline.result.totalSeconds;
  const totalSecondsCurrent = currentResult ? currentResult.totalSeconds : null;

  let absoluteDeltaSeconds: number | null = null;
  let relativeDeltaRatio: number | null = null;
  if (compatible && totalSecondsCurrent !== null) {
    absoluteDeltaSeconds = totalSecondsCurrent - totalSecondsBaseline;
    relativeDeltaRatio = totalSecondsBaseline > 0 ? absoluteDeltaSeconds / totalSecondsBaseline : null;
  }

  let targetMet: boolean | null = null;
  if (!targetInvalid && targetSeconds !== null && Number.isFinite(targetSeconds) && targetSeconds > 0 && totalSecondsCurrent !== null) {
    targetMet = totalSecondsCurrent <= targetSeconds;
  }

  return {
    compatible,
    incompatibleReason: compatible ? null : !sameOperator ? "Operator가 다릅니다" : incompatibleAlgorithm ? "Collective 알고리즘이 다릅니다" : "모델 버전이 다릅니다",
    totalSecondsBaseline,
    totalSecondsCurrent,
    absoluteDeltaSeconds,
    relativeDeltaRatio,
    workloadChanged,
    executionChanged,
    changedHardwareSpecs,
    hbmTrafficBaseline: traffic(baseline.result, "hbm"),
    hbmTrafficCurrent: currentResult ? traffic(currentResult, "hbm") : null,
    linkTrafficBaseline: traffic(baseline.result, "link"),
    linkTrafficCurrent: currentResult ? traffic(currentResult, "link") : null,
    sramRequiredBaseline: metricValue(baseline.result, "metric-required-sram"),
    sramRequiredCurrent: currentResult ? metricValue(currentResult, "metric-required-sram") : null,
    limitingResourcesBaseline: [...new Set(baseline.result.steps.map((step) => step.limitingResource).filter((resource): resource is LimitingResource => resource !== null))],
    limitingResourcesCurrent: currentResult ? [...new Set(currentResult.steps.map((step) => step.limitingResource).filter((resource): resource is LimitingResource => resource !== null))] : [],
    targetSeconds,
    targetInvalid,
    targetMet,
  };
}
