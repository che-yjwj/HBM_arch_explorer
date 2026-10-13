// GEMM/GEMV operator: validation, formulas, steps, transfers. (02-analysis-models.md GEMM section)
import type {
  Assumption,
  CoreAnalysisResult,
  Diagnostic,
  GemmWorkload,
  Hardware,
  InvalidCoreAnalysisResult,
  Resource,
  Step,
  Tensor,
  TimeContribution,
  Transfer,
  ValidCoreAnalysisResult,
} from "../domain/types";
import { MetricBuilder, resetIdCounter } from "../engine/metrics";
import { addFinite, divideChecked, safeAdd, safeMul } from "../engine/numeric";

const Q_BYTES = 2; // BF16
const A_BYTES = 4; // FP32 accumulator
export const MODEL_VERSION = "gemm-v1";

export interface GemmValidation {
  diagnostics: Diagnostic[];
}

export function validateGemm(w: GemmWorkload): GemmValidation {
  const diagnostics: Diagnostic[] = [];
  const check = (v: number, label: string, id: string) => {
    if (!Number.isFinite(v) || v <= 0 || !Number.isInteger(v) || !Number.isSafeInteger(v)) {
      diagnostics.push({
        level: "error",
        message: `${label}은 양의 안전 정수여야 합니다 (입력: ${v})`,
        relatedInputIds: [id],
      });
    }
  };
  check(w.m, "M", "m");
  check(w.n, "N", "n");
  check(w.k, "K", "k");
  check(w.tileM, "타일 M", "tileM");
  check(w.tileN, "타일 N", "tileN");
  check(w.tileK, "타일 K", "tileK");
  return { diagnostics };
}

const ASSUMPTIONS: Assumption[] = [
  {
    id: "asm-gemm-output-stationary",
    description: "Output-stationary 순서: 각 출력 타일에 대해 K 타일을 순회하고 마지막에 C를 저장",
    appliesTo: "gemm",
    limitations: "레지스터 내부 접근과 bank conflict는 제외",
  },
  {
    id: "asm-gemm-no-double-buffer",
    description: "Double buffering 없음. SRAM 부족 시 자동 타일 변경이나 spill을 하지 않음",
    appliesTo: "gemm",
    limitations: "필요 SRAM = q(tM'tK' + tK'tN') + a*tM'tN'",
  },
];

export function analyzeGemm(w: GemmWorkload, hw: Hardware): CoreAnalysisResult {
  resetIdCounter();
  const { diagnostics: inputDiagnostics } = validateGemm(w);
  if (inputDiagnostics.length > 0) {
    return invalidResult(inputDiagnostics, {});
  }

  const { m, n, k, tileM, tileN, tileK } = w;
  const q = Q_BYTES;
  const a = A_BYTES;

  const I = Math.ceil(m / tileM);
  const J = Math.ceil(n / tileN);
  const L = Math.ceil(k / tileK);

  const flops = safeMul(2, m, n, k);
  const hbmA = safeMul(q, m, k, J);
  const hbmB = safeMul(q, k, n, I);
  const hbmC = safeMul(q, m, n);
  if (flops === null || hbmA === null || hbmB === null || hbmC === null) {
    return invalidResult([{ level: "error", message: "FLOPs/HBM byte 계산이 안전 정수 상한을 초과했습니다", relatedInputIds: ["m", "n", "k"] }], {});
  }
  const hbmTotal = safeAdd(hbmA, hbmB, hbmC);
  if (hbmTotal === null) {
    return invalidResult([{ level: "error", message: "HBM traffic 합계가 안전 정수 상한을 초과했습니다", relatedInputIds: ["m", "n", "k"] }], {});
  }

  const sramLoadA = safeMul(q, m, k, J);
  const sramLoadB = safeMul(q, k, n, I);
  const accumLoops = safeMul(2, L);
  const sramAccum = accumLoops === null ? null : safeMul(a, m, n, accumLoops - 1);
  if (sramLoadA === null || sramLoadB === null || sramAccum === null) {
    return invalidResult([{ level: "error", message: "SRAM traffic 계산이 안전 정수 상한을 초과했습니다", relatedInputIds: ["m", "n", "k", "tileK"] }], {});
  }
  const sramLoadTraffic = safeAdd(sramLoadA, sramLoadB);
  const sramTraffic = sramLoadTraffic === null ? null : safeAdd(sramLoadTraffic, sramAccum);
  if (sramLoadTraffic === null || sramTraffic === null) {
    return invalidResult([{ level: "error", message: "SRAM traffic 합계가 안전 정수 범위를 초과했습니다", relatedInputIds: ["m", "n", "k"] }], {});
  }

  const tMp = Math.min(m, tileM);
  const tNp = Math.min(n, tileN);
  const tKp = Math.min(k, tileK);
  const requiredSramA = safeMul(q, tMp, tKp);
  const requiredSramB = safeMul(q, tKp, tNp);
  const requiredSramC = safeMul(a, tMp, tNp);
  if (requiredSramA === null || requiredSramB === null || requiredSramC === null) {
    return invalidResult([{ level: "error", message: "필요 SRAM 계산이 안전 정수 상한을 초과했습니다", relatedInputIds: ["tileM", "tileN", "tileK"] }], {});
  }
  const requiredSram = safeAdd(requiredSramA, requiredSramB, requiredSramC);
  if (requiredSram === null) return invalidResult([{ level: "error", message: "필요 SRAM 합계가 안전 정수 범위를 초과했습니다", relatedInputIds: ["tileM", "tileN", "tileK"] }], {});

  if (requiredSram > hw.sramCapacityBytes) {
    return invalidResult(
      [
        {
          level: "error",
          message: `SRAM 부족: 필요 ${requiredSram} bytes > 설정 ${hw.sramCapacityBytes} bytes`,
          relatedInputIds: ["tileM", "tileN", "tileK", "hardware.sramCapacityBytes"],
          fix: `SRAM 용량을 ${requiredSram} bytes 이상으로 늘리거나 타일을 줄이세요`,
        },
      ],
      { requiredSramBytes: requiredSram, sramCapacityBytes: hw.sramCapacityBytes },
    );
  }

  const cEff = hw.computeFlops * hw.computeEfficiency;
  const sramEffBw = hw.sramBwBytes * hw.sramEfficiency;
  const hbmEffBw = hw.hbmBwBytes * hw.hbmEfficiency;

  if (cEff <= 0 || sramEffBw <= 0 || hbmEffBw <= 0 || !Number.isFinite(cEff) || !Number.isFinite(sramEffBw) || !Number.isFinite(hbmEffBw)) {
    return invalidResult(
      [{ level: "error", message: "유효 성능 또는 BW가 0 이하이거나 유한하지 않습니다", relatedInputIds: ["hardware"] }],
      {},
    );
  }

  const computeTimeOnly = divideChecked(flops, cEff);
  const sramAccessTime = divideChecked(sramTraffic, sramEffBw);
  if (computeTimeOnly === null || sramAccessTime === null) {
    return invalidResult([{ level: "error", message: "Compute 시간 계산이 overflow/underflow로 유한한 양수를 만들지 못했습니다", relatedInputIds: [] }], {});
  }
  const computeStepTime = Math.max(computeTimeOnly, sramAccessTime);
  const hbmLoadBytes = safeAdd(hbmA, hbmB);
  const loadTime = hbmLoadBytes === null ? null : divideChecked(hbmLoadBytes, hbmEffBw);
  const storeTime = divideChecked(hbmC, hbmEffBw);
  if (loadTime === null || storeTime === null) {
    return invalidResult([{ level: "error", message: "Load/Store 시간 계산이 overflow/underflow로 유한한 양수를 만들지 못했습니다", relatedInputIds: [] }], {});
  }
  const totalSeconds = addFinite(loadTime, computeStepTime, storeTime);
  if (totalSeconds === null) {
    return invalidResult([{ level: "error", message: "총시간 계산이 유한하지 않습니다 (overflow)", relatedInputIds: [] }], {});
  }

  // Resources: single rank, one HBM/SRAM/Compute each.
  const resHbm: Resource = { id: "res-hbm", kind: "hbm", rankIndex: 0 };
  const resSram: Resource = { id: "res-sram", kind: "sram", rankIndex: 0 };
  const resCompute: Resource = { id: "res-compute", kind: "compute", rankIndex: 0 };
  const resources = [resHbm, resSram, resCompute];

  const tensorABytes = safeMul(q, m, k)!;
  const tensorBBytes = safeMul(q, k, n)!;
  const tensorCBytes = safeMul(q, m, n)!;
  const tensorA: Tensor = { id: "tensor-a", label: "A", shape: [m, k], precision: "bf16", bytes: tensorABytes, scope: "global" };
  const tensorB: Tensor = { id: "tensor-b", label: "B", shape: [k, n], precision: "bf16", bytes: tensorBBytes, scope: "global" };
  const tensorC: Tensor = { id: "tensor-c", label: "C", shape: [m, n], precision: "bf16", bytes: tensorCBytes, scope: "global" };
  const tensors = [tensorA, tensorB, tensorC];

  const b = new MetricBuilder();
  const iM = b.input("input-m", "M", m, "count");
  const iN = b.input("input-n", "N", n, "count");
  const iK = b.input("input-k", "K", k, "count");
  const iQ = b.input("input-q", "q (BF16 byte)", q, "byte");
  const iA = b.input("input-a", "a (FP32 byte)", a, "byte");
  const iI = b.input("input-i", "I (M 타일 수)", I, "count");
  const iJ = b.input("input-j", "J (N 타일 수)", J, "count");
  const iL = b.input("input-l", "L (K 타일 수)", L, "count");
  const iTMp = b.input("input-tmp", "tM' (min(M,타일M))", tMp, "count");
  const iTNp = b.input("input-tnp", "tN' (min(N,타일N))", tNp, "count");
  const iTKp = b.input("input-tkp", "tK' (min(K,타일K))", tKp, "count");
  const iCEff = b.input("input-c-eff", "유효 Compute 성능", cEff, "flop/s");
  const iSramEffBw = b.input("input-sram-eff-bw", "유효 SRAM BW", sramEffBw, "byte/s");
  const iHbmEffBw = b.input("input-hbm-eff-bw", "유효 HBM BW", hbmEffBw, "byte/s");

  const mFlops = b.derived({
    id: "metric-flops",
    label: "FLOPs",
    value: flops,
    unit: "flop",
    scope: "global",
    formula: { description: "FLOPs = 2MNK", expression: "2*M*N*K", inputMetricIds: [iM.id, iN.id, iK.id], substituted: { M: m, N: n, K: k } },
  });

  const mHbmA = b.derived({
    id: "metric-hbm-a",
    label: "HBM A 읽기",
    value: hbmA,
    unit: "byte",
    scope: "global",
    relatedResourceIds: [resHbm.id],
    formula: { description: "HBM A 읽기 = q*M*K*J", expression: "q*M*K*J", inputMetricIds: [iQ.id, iM.id, iK.id, iJ.id], substituted: { q, M: m, K: k, J } },
  });
  const mHbmB = b.derived({
    id: "metric-hbm-b",
    label: "HBM B 읽기",
    value: hbmB,
    unit: "byte",
    scope: "global",
    relatedResourceIds: [resHbm.id],
    formula: { description: "HBM B 읽기 = q*K*N*I", expression: "q*K*N*I", inputMetricIds: [iQ.id, iK.id, iN.id, iI.id], substituted: { q, K: k, N: n, I } },
  });
  const mHbmC = b.derived({
    id: "metric-hbm-c",
    label: "HBM C 쓰기",
    value: hbmC,
    unit: "byte",
    scope: "global",
    relatedResourceIds: [resHbm.id],
    formula: { description: "HBM C 쓰기 = q*M*N", expression: "q*M*N", inputMetricIds: [iQ.id, iM.id, iN.id], substituted: { q, M: m, N: n } },
  });
  b.derived({
    id: "metric-hbm-total",
    label: "HBM traffic 합계",
    value: hbmTotal,
    unit: "byte",
    scope: "global",
    relatedResourceIds: [resHbm.id],
    formula: {
      description: "Q = q(MKJ + KNI + MN)",
      expression: "hbmA + hbmB + hbmC",
      inputMetricIds: [mHbmA.id, mHbmB.id, mHbmC.id],
      substituted: { hbmA, hbmB, hbmC },
    },
  });

  const mSramTraffic = b.derived({
    id: "metric-sram-traffic",
    label: "SRAM traffic",
    value: sramTraffic,
    unit: "byte",
    scope: "global",
    relatedResourceIds: [resSram.id],
    formula: {
      description: "SRAM traffic = q(MKJ+KNI) + aMN(2L-1)",
      expression: "q*(M*K*J + K*N*I) + a*M*N*(2*L-1)",
      inputMetricIds: [iQ.id, iA.id, iM.id, iN.id, iK.id, iI.id, iJ.id, iL.id],
      substituted: { q, a, M: m, N: n, K: k, I, J, L },
      assumptionIds: ["asm-gemm-output-stationary"],
    },
  });
  const mSramLoadA = b.derived({
    id: "metric-sram-load-a", label: "SRAM A 읽기", value: sramLoadA, unit: "byte", scope: "global", relatedResourceIds: [resSram.id],
    formula: { description: "출력 타일 반복에 따른 A SRAM 읽기", expression: "q*M*K*J", inputMetricIds: [iQ.id, iM.id, iK.id, iJ.id], substituted: { q, M: m, K: k, J } },
  });
  const mSramLoadB = b.derived({
    id: "metric-sram-load-b", label: "SRAM B 읽기", value: sramLoadB, unit: "byte", scope: "global", relatedResourceIds: [resSram.id],
    formula: { description: "출력 타일 반복에 따른 B SRAM 읽기", expression: "q*K*N*I", inputMetricIds: [iQ.id, iK.id, iN.id, iI.id], substituted: { q, K: k, N: n, I } },
  });
  const mSramAccum = b.derived({
    id: "metric-sram-accum", label: "SRAM 누산값 이동", value: sramAccum, unit: "byte", scope: "global", relatedResourceIds: [resSram.id],
    formula: { description: "K 타일 사이 누산값 읽기·쓰기", expression: "a*M*N*(2*L-1)", inputMetricIds: [iA.id, iM.id, iN.id, iL.id], substituted: { a, M: m, N: n, L } },
  });

  b.derived({
    id: "metric-required-sram",
    label: "필요 SRAM (타일 working set)",
    value: requiredSram,
    unit: "byte",
    scope: "global",
    relatedResourceIds: [resSram.id],
    formula: {
      description: "필요 SRAM = q(tM'tK' + tK'tN') + a*tM'tN'",
      expression: "q*(tMp*tKp + tKp*tNp) + a*tMp*tNp",
      inputMetricIds: [iQ.id, iA.id, iTMp.id, iTKp.id, iTNp.id],
      substituted: { q, a, tMp, tKp, tNp },
      assumptionIds: ["asm-gemm-no-double-buffer"],
    },
  });

  const mLoadTime = b.derived({
    id: "metric-load-time",
    label: "Load 시간",
    value: loadTime,
    unit: "s",
    scope: "global",
    formula: {
      description: "Load 시간 = (HBM A + HBM B) / 유효 HBM BW",
      expression: "(hbmA + hbmB) / hbmEffBw",
      inputMetricIds: [mHbmA.id, mHbmB.id, iHbmEffBw.id],
      substituted: { hbmA, hbmB, hbmEffBw },
    },
  });

  const mComputeOnly = b.derived({
    id: "metric-compute-only-time",
    label: "Compute 연산 시간",
    value: computeTimeOnly,
    unit: "s",
    scope: "global",
    formula: { description: "FLOPs / 유효 Compute 성능", expression: "flops / cEff", inputMetricIds: [mFlops.id, iCEff.id], substituted: { flops, cEff } },
  });
  const mSramAccess = b.derived({
    id: "metric-sram-access-time",
    label: "SRAM 접근 시간",
    value: sramAccessTime,
    unit: "s",
    scope: "global",
    formula: {
      description: "SRAM bytes / 유효 SRAM BW",
      expression: "sramTraffic / sramEffBw",
      inputMetricIds: [mSramTraffic.id, iSramEffBw.id],
      substituted: { sramTraffic, sramEffBw },
    },
  });
  const mComputeTime = b.derived({
    id: "metric-compute-time",
    label: "Compute 단계 시간",
    value: computeStepTime,
    unit: "s",
    scope: "global",
    formula: {
      description: "max(Compute 연산 시간, SRAM 접근 시간)",
      expression: "max(computeTimeOnly, sramAccessTime)",
      inputMetricIds: [mComputeOnly.id, mSramAccess.id],
      substituted: { computeTimeOnly, sramAccessTime },
    },
  });

  const mStoreTime = b.derived({
    id: "metric-store-time",
    label: "Store 시간",
    value: storeTime,
    unit: "s",
    scope: "global",
    formula: { description: "HBM C / 유효 HBM BW", expression: "hbmC / hbmEffBw", inputMetricIds: [mHbmC.id, iHbmEffBw.id], substituted: { hbmC, hbmEffBw } },
  });

  b.derived({
    id: "metric-total-time",
    label: "총시간",
    value: totalSeconds,
    unit: "s",
    scope: "global",
    formula: {
      description: "총시간 = Load + Compute + Store (순차)",
      expression: "loadTime + computeStepTime + storeTime",
      inputMetricIds: [mLoadTime.id, mComputeTime.id, mStoreTime.id],
      substituted: { loadTime, computeStepTime, storeTime },
    },
  });

  const stepLoad: Step = {
    id: "step-load",
    label: "Load A/B",
    description: "HBM에서 A/B를 읽어 SRAM에 적재합니다",
    precedingStepIds: [],
    resourceIds: [resHbm.id, resSram.id],
    tensorIds: [tensorA.id, tensorB.id],
    timeMetricId: mLoadTime.id,
    limitingResource: loadTime > 0 ? "hbm" : null,
    isAggregate: true,
  };
  // Relative tolerance per 06: |x-y| <= 1e-9 * max(x,y) (not max(x,y,1) — no artificial floor).
  const EPS_REL = 1e-9;
  const tie = Math.abs(computeTimeOnly - sramAccessTime) <= EPS_REL * Math.max(computeTimeOnly, sramAccessTime);
  const stepCompute: Step = {
    id: "step-compute",
    label: "Compute",
    description: "SRAM의 타일로 연산하고 누산값을 SRAM에 유지합니다",
    precedingStepIds: [stepLoad.id],
    resourceIds: [resCompute.id, resSram.id],
    tensorIds: [tensorA.id, tensorB.id, tensorC.id],
    timeMetricId: mComputeTime.id,
    limitingResource: computeStepTime === 0 ? null : tie ? "compute+sram" : computeTimeOnly >= sramAccessTime ? "compute" : "sram",
    isAggregate: true,
  };
  const stepStore: Step = {
    id: "step-store",
    label: "Store C",
    description: "연산 결과 C를 SRAM에서 HBM으로 씁니다",
    precedingStepIds: [stepCompute.id],
    resourceIds: [resSram.id, resHbm.id],
    tensorIds: [tensorC.id],
    timeMetricId: mStoreTime.id,
    limitingResource: storeTime > 0 ? "hbm" : null,
    isAggregate: true,
  };
  const steps = [stepLoad, stepCompute, stepStore];

  const transfers: Transfer[] = [
    { id: "transfer-a", from: resHbm.id, to: resSram.id, tensorId: tensorA.id, bytesMetricId: mHbmA.id, stepId: stepLoad.id, trafficClass: "hbm" },
    { id: "transfer-b", from: resHbm.id, to: resSram.id, tensorId: tensorB.id, bytesMetricId: mHbmB.id, stepId: stepLoad.id, trafficClass: "hbm" },
    { id: "transfer-a-sram-read", from: resSram.id, to: resCompute.id, tensorId: tensorA.id, bytesMetricId: mSramLoadA.id, stepId: stepCompute.id, trafficClass: "sram" },
    { id: "transfer-b-sram-read", from: resSram.id, to: resCompute.id, tensorId: tensorB.id, bytesMetricId: mSramLoadB.id, stepId: stepCompute.id, trafficClass: "sram" },
    { id: "transfer-c-sram-write", from: resCompute.id, to: resSram.id, tensorId: tensorC.id, bytesMetricId: mSramAccum.id, stepId: stepCompute.id, trafficClass: "sram" },
    { id: "transfer-c", from: resSram.id, to: resHbm.id, tensorId: tensorC.id, bytesMetricId: mHbmC.id, stepId: stepStore.id, trafficClass: "hbm" },
  ];

  const contributions: TimeContribution[] = [];
  const byLimit = new Map<string, { stepIds: string[]; seconds: number }>();
  for (const s of steps) {
    if (!s.limitingResource) continue;
    const metric = b.get(s.timeMetricId)!;
    if (metric.value <= 0) continue;
    const entry = byLimit.get(s.limitingResource) ?? { stepIds: [], seconds: 0 };
    entry.stepIds.push(s.id);
    entry.seconds += metric.value;
    byLimit.set(s.limitingResource, entry);
  }
  for (const [limit, entry] of byLimit) {
    contributions.push({
      limitingResources: limit as any,
      stepIds: entry.stepIds,
      totalSeconds: entry.seconds,
      ratioOfTotal: totalSeconds > 0 ? entry.seconds / totalSeconds : null,
    });
  }

  const result: ValidCoreAnalysisResult = {
    valid: true,
    modelVersion: MODEL_VERSION,
    resources,
    tensors,
    steps,
    transfers,
    metrics: b.metrics,
    formulas: b.formulas,
    assumptions: ASSUMPTIONS,
    diagnostics: [],
    totalSeconds,
    contributions,
  };
  return result;
}

function invalidResult(diagnostics: Diagnostic[], explain: Record<string, number>): InvalidCoreAnalysisResult {
  return {
    valid: false,
    modelVersion: MODEL_VERSION,
    resources: [],
    tensors: [],
    steps: [],
    transfers: [],
    metrics: [],
    formulas: [],
    assumptions: ASSUMPTIONS,
    diagnostics,
    totalSeconds: null,
    explain,
  };
}

export const SUPPORTED_SENSITIVITY_RESOURCES = ["compute", "sram", "hbm"] as const;
