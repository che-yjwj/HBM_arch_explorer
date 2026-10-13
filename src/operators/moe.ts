// MoE operator: Router -> Dispatch -> Expert GEMM -> Combine, equal routing. (02-analysis-models.md MoE section)
import type {
  Assumption,
  CoreAnalysisResult,
  Diagnostic,
  Hardware,
  InvalidCoreAnalysisResult,
  MoeWorkload,
  Resource,
  Step,
  Tensor,
  TimeContribution,
  Transfer,
} from "../domain/types";
import { MetricBuilder, resetIdCounter } from "../engine/metrics";
import { addFinite, divideChecked, multiplyFiniteChecked, safeMul } from "../engine/numeric";

const Q_BYTES = 2; // BF16
export const MODEL_VERSION = "moe-v1";

const ASSUMPTIONS: Assumption[] = [
  {
    id: "asm-moe-equal-routing",
    description: "P개 Rank에 E개 Expert를 균등 배치. 균등 라우팅의 평균 토큰 수 사용 (분수 평균 허용)",
    appliesTo: "moe",
    limitations: "Router 선택·정렬, activation 함수, combine 가중합, 통신 metadata는 제외",
  },
];

function validateMoe(w: MoeWorkload, rankCount: number): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  const check = (v: number, label: string, id: string) => {
    if (!Number.isFinite(v) || v <= 0 || !Number.isInteger(v) || !Number.isSafeInteger(v)) {
      diagnostics.push({ level: "error", message: `${label}은 양의 안전 정수여야 합니다 (입력: ${v})`, relatedInputIds: [id] });
    }
  };
  check(w.tokens, "Tokens", "tokens");
  check(w.hidden, "Hidden", "hidden");
  check(w.ffn, "FFN", "ffn");
  check(w.experts, "Experts", "experts");
  check(w.topK, "Top-k", "topK");
  if (diagnostics.length > 0) return diagnostics;
  if (w.topK > w.experts) {
    diagnostics.push({ level: "error", message: `Top-k(${w.topK})는 Experts(${w.experts}) 이하여야 합니다`, relatedInputIds: ["topK"] });
  }
  if (w.experts % rankCount !== 0) {
    diagnostics.push({
      level: "error",
      message: `Experts(${w.experts})는 Rank 수(${rankCount})로 나누어져야 합니다 (E mod P != 0)`,
      relatedInputIds: ["experts", "hardware.rankCount"],
      fix: `Rank 수를 ${w.experts}의 약수로 맞추거나 Experts를 ${rankCount}의 배수로 조정하세요`,
    });
  }
  return diagnostics;
}

export function analyzeMoe(w: MoeWorkload, hw: Hardware): CoreAnalysisResult {
  resetIdCounter();
  const p = hw.rankCount;
  const diagnostics = validateMoe(w, p);
  if (diagnostics.length > 0) return invalidResult(diagnostics);

  const { tokens: T, hidden: D, ffn: F, experts: E, topK: k } = w;
  const q = Q_BYTES;

  const routerFlopsTotal = safeMul(2, T, D, E);
  const expertFlopsTotal = safeMul(4, T, k, D, F);
  const expertWeightsReadPerRank = safeMul(2, E / p, D, F, q);
  const routerWeightsReadPerRank = safeMul(D, E, q);
  if (routerFlopsTotal === null || expertFlopsTotal === null || expertWeightsReadPerRank === null || routerWeightsReadPerRank === null) {
    return invalidResult([{ level: "error", message: "FLOPs/byte 계산이 안전 정수 상한을 초과했습니다", relatedInputIds: ["tokens", "hidden", "ffn", "experts", "topK"] }]);
  }
  const routerFlopsPerRank = routerFlopsTotal / p;
  const expertFlopsPerRank = expertFlopsTotal / p;
  // Dispatch/Combine use average per-rank traffic over equal routing — fractional values are expected (02 문서), so not integer-checked.
  const routedTokenBytes = safeMul(T, k, D, q);
  const allTokenBytes = safeMul(T, D, q);
  if (routedTokenBytes === null || allTokenBytes === null) {
    return invalidResult([{ level: "error", message: "토큰 이동량이 안전 정수 범위를 초과했습니다", relatedInputIds: ["tokens", "hidden", "topK"] }]);
  }
  const dispatchSendPerRank = (routedTokenBytes / p) * ((p - 1) / p);
  const combineSendPerRank = dispatchSendPerRank;
  const localTokenReadPerRank = allTokenBytes / p;
  const outputWritePerRank = allTokenBytes / p;

  const cEff = hw.computeFlops * hw.computeEfficiency;
  const hbmEffBw = hw.hbmBwBytes * hw.hbmEfficiency;
  const linkEffBw = hw.linkBwBytes * hw.linkEfficiency;
  if (cEff <= 0 || hbmEffBw <= 0 || linkEffBw <= 0 || ![cEff, hbmEffBw, linkEffBw].every(Number.isFinite)) {
    return invalidResult([{ level: "error", message: "유효 성능 또는 BW가 0 이하이거나 유한하지 않습니다", relatedInputIds: ["hardware"] }]);
  }

  const localTokenLoadTime = divideChecked(localTokenReadPerRank, hbmEffBw);
  const routerWeightLoadTime = divideChecked(routerWeightsReadPerRank, hbmEffBw);
  const routerTime = divideChecked(routerFlopsPerRank, cEff);
  const dispatchComm = p > 1 ? divideChecked(dispatchSendPerRank, linkEffBw) : 0;
  const expertWeightLoadTime = divideChecked(expertWeightsReadPerRank, hbmEffBw);
  const expertComputeTime = divideChecked(expertFlopsPerRank, cEff);
  const combineComm = p > 1 ? divideChecked(combineSendPerRank, linkEffBw) : 0;
  const outputStoreTime = divideChecked(outputWritePerRank, hbmEffBw);

  const parts = [localTokenLoadTime, routerWeightLoadTime, routerTime, dispatchComm, expertWeightLoadTime, expertComputeTime, combineComm, outputStoreTime];
  if (parts.some((t) => t === null)) {
    return invalidResult([{ level: "error", message: "시간 계산이 overflow/underflow로 유한한 양수를 만들지 못했습니다", relatedInputIds: [] }]);
  }
  const [ltl, rwl, rt, dc, ewl, ect, cc, ost] = parts as number[];
  const inputLoadTime = addFinite(ltl, rwl);
  const allToAllLatency = p > 1 ? multiplyFiniteChecked(p - 1, hw.linkLatencySeconds) : 0;
  const dispatchTime = p > 1 && allToAllLatency !== null ? addFinite(allToAllLatency, dc) : p === 1 ? 0 : null;
  const combineTime = p > 1 && allToAllLatency !== null ? addFinite(allToAllLatency, cc) : p === 1 ? 0 : null;
  const totalSeconds = inputLoadTime === null || dispatchTime === null || combineTime === null
    ? null
    : addFinite(inputLoadTime, rt, dispatchTime, ewl, ect, combineTime, ost);
  if (totalSeconds === null || inputLoadTime === null || dispatchTime === null || combineTime === null) {
    return invalidResult([{ level: "error", message: "총시간 계산이 유한하지 않습니다", relatedInputIds: [] }]);
  }

  const resHbm: Resource = { id: "res-hbm", kind: "hbm", rankIndex: 0 };
  const resCompute: Resource = { id: "res-compute", kind: "compute", rankIndex: 0 };
  const resLink: Resource = { id: "res-link", kind: "link", rankIndex: -1 };
  const resources = [resHbm, resCompute, resLink];

  const tensorTokens: Tensor = { id: "tensor-tokens", label: "입력 토큰", shape: [T, D], precision: "bf16", bytes: allTokenBytes, scope: "global" };
  const tensorRouterW: Tensor = { id: "tensor-router-weights", label: "Router Weight", shape: [D, E], precision: "bf16", bytes: routerWeightsReadPerRank, scope: "per-rank" };
  const tensorExpertW: Tensor = { id: "tensor-expert-weights", label: "Expert Weight", shape: [E / p, D, F], precision: "bf16", bytes: expertWeightsReadPerRank, scope: "per-rank" };
  const tensorDispatch: Tensor = { id: "tensor-dispatch", label: "Dispatch 토큰", shape: [T, D], precision: "bf16", bytes: dispatchSendPerRank, scope: "per-rank" };
  const tensorCombine: Tensor = { id: "tensor-combine", label: "Combine 출력", shape: [T, D], precision: "bf16", bytes: combineSendPerRank, scope: "per-rank" };
  const tensorOut: Tensor = { id: "tensor-out", label: "출력", shape: [T, D], precision: "bf16", bytes: allTokenBytes, scope: "global" };
  const tensors = [tensorTokens, tensorRouterW, tensorExpertW, tensorDispatch, tensorCombine, tensorOut];

  const b = new MetricBuilder();
  const iT = b.input("input-t", "Tokens", T, "count");
  const iD = b.input("input-d", "Hidden", D, "count");
  const iF = b.input("input-f", "FFN", F, "count");
  const iE = b.input("input-e", "Experts", E, "count");
  const iK = b.input("input-k", "Top-k", k, "count");
  const iP = b.input("input-p", "P (Rank 수)", p, "count");
  const iQ = b.input("input-q", "q (BF16 byte)", q, "byte");
  const iCEff = b.input("input-c-eff", "유효 Compute 성능", cEff, "flop/s");
  const iHbmEffBw = b.input("input-hbm-eff-bw", "유효 HBM BW", hbmEffBw, "byte/s");
  const iLinkEffBw = b.input("input-link-eff-bw", "유효 Link BW", linkEffBw, "byte/s");
  const iLatency = b.input("input-link-latency", "Link latency", hw.linkLatencySeconds, "s");

  const mRouterTotal = b.derived({ id: "metric-router-flops-total", label: "Router FLOPs (전체)", value: routerFlopsTotal, unit: "flop", scope: "global", formula: { description: "Router FLOPs = 2TDE", expression: "2*T*D*E", inputMetricIds: [iT.id, iD.id, iE.id], substituted: { T, D, E } } });
  const mExpertTotal = b.derived({ id: "metric-expert-flops-total", label: "Expert FLOPs (전체)", value: expertFlopsTotal, unit: "flop", scope: "global", formula: { description: "Expert FLOPs = 4TkDF", expression: "4*T*k*D*F", inputMetricIds: [iT.id, iK.id, iD.id, iF.id], substituted: { T, k, D, F } } });
  const mLocalTokenRead = b.derived({ id: "metric-local-token-bytes", label: "로컬 토큰 읽기", value: localTokenReadPerRank, unit: "byte", scope: "per-rank", formula: { description: "TDq/P", expression: "T*D*q/P", inputMetricIds: [iT.id, iD.id, iQ.id, iP.id], substituted: { T, D, q, P: p } } });
  const mRouterWRead = b.derived({ id: "metric-router-weight-bytes", label: "Router Weight 읽기", value: routerWeightsReadPerRank, unit: "byte", scope: "per-rank", formula: { description: "DEq", expression: "D*E*q", inputMetricIds: [iD.id, iE.id, iQ.id], substituted: { D, E, q } } });
  const mDispatch = b.derived({ id: "metric-dispatch-bytes", label: "Dispatch Rank당 송신량", value: dispatchSendPerRank, unit: "byte", scope: "per-rank", relatedResourceIds: [resLink.id], formula: { description: "(TkDq/P)*(P-1)/P", expression: "(T*k*D*q/P)*(P-1)/P", inputMetricIds: [iT.id, iK.id, iD.id, iQ.id, iP.id], substituted: { T, k, D, q, P: p } } });
  const mCombine = b.derived({ id: "metric-combine-bytes", label: "Combine Rank당 송신량", value: combineSendPerRank, unit: "byte", scope: "per-rank", relatedResourceIds: [resLink.id], formula: { description: "Dispatch와 동일", expression: "dispatchBytes", inputMetricIds: [mDispatch.id], substituted: { dispatchSendPerRank } } });
  const mExpertWRead = b.derived({ id: "metric-expert-weight-bytes", label: "Expert Weight 읽기", value: expertWeightsReadPerRank, unit: "byte", scope: "per-rank", formula: { description: "2(E/P)DFq", expression: "2*(E/P)*D*F*q", inputMetricIds: [iE.id, iP.id, iD.id, iF.id, iQ.id], substituted: { E, P: p, D, F, q } } });
  const mOutputWrite = b.derived({ id: "metric-output-bytes", label: "출력 쓰기", value: outputWritePerRank, unit: "byte", scope: "per-rank", formula: { description: "TDq/P", expression: "T*D*q/P", inputMetricIds: [iT.id, iD.id, iQ.id, iP.id], substituted: { T, D, q, P: p } } });

  const mInputLoad = b.derived({ id: "metric-input-load-time", label: "Input/Router Weight Load 시간", value: inputLoadTime, unit: "s", scope: "per-rank", formula: { description: "로컬 토큰 + Router weight 읽기 / 유효 HBM BW", expression: "(localTokenRead+routerWeightsRead)/hbmEffBw", inputMetricIds: [mLocalTokenRead.id, mRouterWRead.id, iHbmEffBw.id], substituted: { localTokenReadPerRank, routerWeightsReadPerRank, hbmEffBw } } });
  const mRouterTime = b.derived({ id: "metric-router-time", label: "Router 시간", value: rt, unit: "s", scope: "per-rank", formula: { description: "Rank당 Router FLOPs / 유효 Compute 성능", expression: "routerFlopsPerRank/cEff", inputMetricIds: [mRouterTotal.id, iP.id, iCEff.id], substituted: { routerFlopsPerRank, cEff } } });
  const mDispatchTime = b.derived({ id: "metric-dispatch-time", label: "Dispatch 시간", value: dispatchTime, unit: "s", scope: "per-rank", formula: { description: "Pairwise All-to-All 모델: (P-1)*latency + 송신/유효 Link BW", expression: "(P-1)*latency + dispatchBytes/linkEffBw", inputMetricIds: [iP.id, iLatency.id, mDispatch.id, iLinkEffBw.id], substituted: { P: p, latency: hw.linkLatencySeconds, dispatchSendPerRank, linkEffBw } } });
  const mExpertWeightLoad = b.derived({ id: "metric-expert-weight-load-time", label: "Expert Weight Load 시간", value: ewl, unit: "s", scope: "per-rank", formula: { description: "2(E/P)DFq / 유효 HBM BW", expression: "expertWeightsRead/hbmEffBw", inputMetricIds: [mExpertWRead.id, iHbmEffBw.id], substituted: { expertWeightsReadPerRank, hbmEffBw } } });
  const mExpertCompute = b.derived({ id: "metric-expert-compute-time", label: "Expert Compute 시간", value: ect, unit: "s", scope: "per-rank", formula: { description: "Rank당 Expert FLOPs / 유효 Compute 성능", expression: "expertFlopsPerRank/cEff", inputMetricIds: [mExpertTotal.id, iP.id, iCEff.id], substituted: { expertFlopsPerRank, cEff } } });
  const mCombineTime = b.derived({ id: "metric-combine-time", label: "Combine 시간", value: combineTime, unit: "s", scope: "per-rank", formula: { description: "Pairwise All-to-All 모델", expression: "(P-1)*latency + combineBytes/linkEffBw", inputMetricIds: [iP.id, iLatency.id, mCombine.id, iLinkEffBw.id], substituted: { P: p, latency: hw.linkLatencySeconds, combineSendPerRank, linkEffBw } } });
  const mOutputStore = b.derived({ id: "metric-output-store-time", label: "Output Store 시간", value: ost, unit: "s", scope: "per-rank", formula: { description: "출력 쓰기 / 유효 HBM BW", expression: "outputWrite/hbmEffBw", inputMetricIds: [mOutputWrite.id, iHbmEffBw.id], substituted: { outputWritePerRank, hbmEffBw } } });

  b.derived({
    id: "metric-total-time",
    label: "총시간 (균등 Rank 임계 경로)",
    value: totalSeconds,
    unit: "s",
    scope: "per-rank",
    formula: {
      description: "총시간 = 단계 시간 합 (순차, 균등 Rank 임계 경로)",
      expression: "inputLoad+router+dispatch+expertWeightLoad+expertCompute+combine+outputStore",
      inputMetricIds: [mInputLoad.id, mRouterTime.id, mDispatchTime.id, mExpertWeightLoad.id, mExpertCompute.id, mCombineTime.id, mOutputStore.id],
      substituted: { inputLoadTime, routerTime: rt, dispatchTime, expertWeightLoadTime: ewl, expertComputeTime: ect, combineTime, outputStoreTime: ost },
      assumptionIds: ["asm-moe-equal-routing"],
    },
  });

  const steps: Step[] = [
    { id: "step-input-load", label: "Input/Router Weight Load", description: "HBM에서 로컬 토큰과 Router weight를 읽습니다", precedingStepIds: [], resourceIds: [resHbm.id], tensorIds: [tensorTokens.id, tensorRouterW.id], timeMetricId: mInputLoad.id, limitingResource: inputLoadTime > 0 ? "hbm" : null, isAggregate: true },
    { id: "step-router", label: "Router", description: "Compute에서 Router를 계산합니다", precedingStepIds: ["step-input-load"], resourceIds: [resCompute.id], tensorIds: [tensorTokens.id], timeMetricId: mRouterTime.id, limitingResource: rt > 0 ? "compute" : null, isAggregate: true },
    { id: "step-dispatch", label: "Dispatch", description: "Link를 통해 토큰을 Expert가 있는 Rank로 보냅니다", precedingStepIds: ["step-router"], resourceIds: [resLink.id], tensorIds: [tensorDispatch.id], timeMetricId: mDispatchTime.id, limitingResource: dispatchTime > 0 ? "link" : null, isAggregate: true },
    { id: "step-expert-weight-load", label: "Expert Weight Load", description: "HBM에서 Expert weight를 읽습니다", precedingStepIds: ["step-dispatch"], resourceIds: [resHbm.id], tensorIds: [tensorExpertW.id], timeMetricId: mExpertWeightLoad.id, limitingResource: ewl > 0 ? "hbm" : null, isAggregate: true },
    { id: "step-expert-compute", label: "Expert Compute", description: "Compute에서 Expert GEMM을 계산합니다", precedingStepIds: ["step-expert-weight-load"], resourceIds: [resCompute.id], tensorIds: [tensorExpertW.id], timeMetricId: mExpertCompute.id, limitingResource: ect > 0 ? "compute" : null, isAggregate: true },
    { id: "step-combine", label: "Combine", description: "Link를 통해 Expert 출력을 원래 Rank로 모읍니다", precedingStepIds: ["step-expert-compute"], resourceIds: [resLink.id], tensorIds: [tensorCombine.id], timeMetricId: mCombineTime.id, limitingResource: combineTime > 0 ? "link" : null, isAggregate: true },
    { id: "step-output-store", label: "Output Store", description: "HBM에 최종 출력을 씁니다", precedingStepIds: ["step-combine"], resourceIds: [resHbm.id], tensorIds: [tensorOut.id], timeMetricId: mOutputStore.id, limitingResource: ost > 0 ? "hbm" : null, isAggregate: true },
  ];

  const transfers: Transfer[] = [
    { id: "transfer-local-token", from: resHbm.id, to: resCompute.id, tensorId: tensorTokens.id, bytesMetricId: mLocalTokenRead.id, stepId: "step-input-load", trafficClass: "hbm" },
    { id: "transfer-router-weight", from: resHbm.id, to: resCompute.id, tensorId: tensorRouterW.id, bytesMetricId: mRouterWRead.id, stepId: "step-input-load", trafficClass: "hbm" },
    { id: "transfer-dispatch", from: resLink.id, to: resLink.id, tensorId: tensorDispatch.id, bytesMetricId: mDispatch.id, stepId: "step-dispatch", trafficClass: "link" },
    { id: "transfer-expert-weight", from: resHbm.id, to: resCompute.id, tensorId: tensorExpertW.id, bytesMetricId: mExpertWRead.id, stepId: "step-expert-weight-load", trafficClass: "hbm" },
    { id: "transfer-combine", from: resLink.id, to: resLink.id, tensorId: tensorCombine.id, bytesMetricId: mCombine.id, stepId: "step-combine", trafficClass: "link" },
    { id: "transfer-output", from: resCompute.id, to: resHbm.id, tensorId: tensorOut.id, bytesMetricId: mOutputWrite.id, stepId: "step-output-store", trafficClass: "hbm" },
  ];

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
  const contributions: TimeContribution[] = [...byLimit.entries()].map(([limit, entry]) => ({
    limitingResources: limit as any,
    stepIds: entry.stepIds,
    totalSeconds: entry.seconds,
    ratioOfTotal: totalSeconds > 0 ? entry.seconds / totalSeconds : null,
  }));

  return {
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
}

function invalidResult(diagnostics: Diagnostic[]): InvalidCoreAnalysisResult {
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
    explain: {},
  };
}

/** SRAM sensitivity is unsupported — no buffer/SRAM-BW model defined for MoE (02-analysis-models.md). */
export const SUPPORTED_SENSITIVITY_RESOURCES = ["compute", "hbm", "link"] as const;
