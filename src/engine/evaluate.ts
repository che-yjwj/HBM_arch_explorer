import type { AnalysisResult, Hardware, ResourceKind, SensitivityResult, Workload } from "../domain/types";
import { analyze } from "./analyze";
import { SUPPORTED_SENSITIVITY_RESOURCES as GEMM_SUPPORTED } from "../operators/gemm";
import { SUPPORTED_SENSITIVITY_RESOURCES as ATTENTION_SUPPORTED } from "../operators/attention";
import { SUPPORTED_SENSITIVITY_RESOURCES as MOE_SUPPORTED } from "../operators/moe";
import { SUPPORTED_SENSITIVITY_RESOURCES as COLLECTIVE_SUPPORTED } from "../operators/collective";

const ALL_RESOURCE_KINDS: ResourceKind[] = ["compute", "sram", "hbm", "link"];

function doubledHardware(hw: Hardware, kind: ResourceKind): Hardware {
  switch (kind) {
    case "compute":
      return { ...hw, computeFlops: hw.computeFlops * 2 };
    case "sram":
      return { ...hw, sramBwBytes: hw.sramBwBytes * 2 };
    case "hbm":
      return { ...hw, hbmBwBytes: hw.hbmBwBytes * 2 };
    case "link":
      return { ...hw, linkBwBytes: hw.linkBwBytes * 2 };
  }
}

function supportedKinds(workload: Workload): ResourceKind[] {
  switch (workload.kind) {
    case "gemm":
      return [...GEMM_SUPPORTED];
    case "attention":
      return [...ATTENTION_SUPPORTED];
    case "moe":
      return [...MOE_SUPPORTED];
    case "collective":
      return [...COLLECTIVE_SUPPORTED];
  }
}

export function sensitivity(workload: Workload, hardware: Hardware): SensitivityResult[] {
  const results: SensitivityResult[] = [];
  const supported = new Set(supportedKinds(workload));
  for (const kind of ALL_RESOURCE_KINDS) {
    if (!supported.has(kind)) {
      results.push({
        resourceKind: kind,
        status: "unsupported",
        baselineSeconds: null,
        changedSeconds: null,
        savedSeconds: null,
        speedup: null,
        reductionRatio: null,
        reason: "모델 미지원",
      });
      continue;
    }
    const baseCore = analyze(workload, hardware);
    if (!baseCore.valid) {
      results.push({
        resourceKind: kind,
        status: "invalid",
        baselineSeconds: null,
        changedSeconds: null,
        savedSeconds: null,
        speedup: null,
        reductionRatio: null,
        reason: "기준 결과가 유효하지 않습니다",
      });
      continue;
    }
    const changedHw = doubledHardware(hardware, kind);
    const changedCore = analyze(workload, changedHw);
    if (!changedCore.valid) {
      results.push({
        resourceKind: kind,
        status: "invalid",
        baselineSeconds: baseCore.totalSeconds,
        changedSeconds: null,
        savedSeconds: null,
        speedup: null,
        reductionRatio: null,
        reason: "2배 변경 후 재계산이 유효하지 않습니다",
      });
      continue;
    }
    const tBase = baseCore.totalSeconds;
    const tChanged = changedCore.totalSeconds;
    if (tBase === 0 && tChanged === 0) {
      results.push({
        resourceKind: kind,
        status: "no-work",
        baselineSeconds: 0,
        changedSeconds: 0,
        savedSeconds: 0,
        speedup: null,
        reductionRatio: null,
        reason: "해당 없음: 통신 없음",
      });
      continue;
    }
    if (tBase <= 0 || tChanged < 0 || !Number.isFinite(tBase) || !Number.isFinite(tChanged)) {
      results.push({
        resourceKind: kind,
        status: "invalid",
        baselineSeconds: tBase,
        changedSeconds: tChanged,
        savedSeconds: null,
        speedup: null,
        reductionRatio: null,
        reason: "계산 오류: 비정상적인 기준/변경 시간",
      });
      continue;
    }
    const saved = tBase - tChanged;
    results.push({
      resourceKind: kind,
      status: "ok",
      baselineSeconds: tBase,
      changedSeconds: tChanged,
      savedSeconds: saved,
      speedup: tBase / tChanged,
      reductionRatio: 1 - tChanged / tBase,
      reason: null,
    });
  }
  return results;
}

/** analyze once; if valid, attach sensitivity. Does not recurse into evaluate from sensitivity. */
export function evaluate(workload: Workload, hardware: Hardware): AnalysisResult {
  const core = analyze(workload, hardware);
  if (!core.valid) {
    return { core, sensitivity: [] };
  }
  return { core, sensitivity: sensitivity(workload, hardware) };
}
