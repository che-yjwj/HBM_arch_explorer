// Attention operator: dense non-causal prefill, Q/K/V already materialized. (02-analysis-models.md Attention section)
import type {
  Assumption,
  AttentionWorkload,
  CoreAnalysisResult,
  Diagnostic,
  Hardware,
  InvalidCoreAnalysisResult,
  Resource,
  Step,
  Tensor,
  TimeContribution,
  Transfer,
} from "../domain/types";
import { MetricBuilder, resetIdCounter } from "../engine/metrics";
import { addFinite, divideChecked, safeAdd, safeMul } from "../engine/numeric";

const Q_BYTES = 2; // BF16
const A_BYTES = 4; // FP32 intermediates
export const MODEL_VERSION = "attention-v1";

const ASSUMPTIONS: Assumption[] = [
  {
    id: "asm-attention-simple-memory",
    description: "중간 행렬을 HBM에 저장하는 단순 모델. Q/K/V 재읽기·타일 재사용·SRAM BW/용량 제한은 포함하지 않음",
    appliesTo: "attention",
    limitations: "GEMM의 메모리 모델과 정확도가 같다고 표시하지 않음",
  },
  {
    id: "asm-attention-throughput-approx",
    description: "PV의 FLOPs와 Softmax의 FLOPs 상당량에도 QKᵀ와 동일한 유효 Compute 성능을 적용하는 시간 근사",
    appliesTo: "attention",
    limitations: "서로 다른 정밀도·연산의 실제 처리량이 같다는 주장이 아님",
  },
];

function validateAttention(w: AttentionWorkload): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  const check = (v: number, label: string, id: string) => {
    if (!Number.isFinite(v) || v <= 0 || !Number.isInteger(v) || !Number.isSafeInteger(v)) {
      diagnostics.push({ level: "error", message: `${label}은 양의 안전 정수여야 합니다 (입력: ${v})`, relatedInputIds: [id] });
    }
  };
  check(w.batch, "Batch", "batch");
  check(w.heads, "Heads", "heads");
  check(w.seq, "Sequence", "seq");
  check(w.headDim, "Head dimension", "headDim");
  return diagnostics;
}

export function analyzeAttention(w: AttentionWorkload, hw: Hardware): CoreAnalysisResult {
  resetIdCounter();
  const diagnostics = validateAttention(w);
  if (diagnostics.length > 0) return invalidResult(diagnostics);

  const { batch: B, heads: H, seq: S, headDim: D } = w;
  const q = Q_BYTES;
  const a = A_BYTES;

  const matmulFlopsEach = safeMul(2, B, H, S, S, D); // QK^T FLOPs, and PV FLOPs (same shape)
  const softmaxFlops = safeMul(5, B, H, S, S);
  const perTensorBytes = safeMul(B, H, S, D, q);
  const scoreProbBytes = safeMul(B, H, S, S, a);
  if (matmulFlopsEach === null || softmaxFlops === null || perTensorBytes === null || scoreProbBytes === null) {
    return invalidResult([{ level: "error", message: "FLOPs/byte 계산이 안전 정수 상한을 초과했습니다", relatedInputIds: ["batch", "heads", "seq", "headDim"] }]);
  }
  const qkFlops = matmulFlopsEach;
  const pvFlops = matmulFlopsEach;
  const hbmTensorTraffic = safeMul(4, perTensorBytes);
  const hbmIntermediateTraffic = safeMul(4, scoreProbBytes);
  const hbmTotal = hbmTensorTraffic === null || hbmIntermediateTraffic === null ? null : safeAdd(hbmTensorTraffic, hbmIntermediateTraffic);
  if (hbmTotal === null) {
    return invalidResult([{ level: "error", message: "HBM traffic 합계가 안전 정수 범위를 초과했습니다", relatedInputIds: ["batch", "heads", "seq", "headDim"] }]);
  }

  const cEff = hw.computeFlops * hw.computeEfficiency;
  const hbmEffBw = hw.hbmBwBytes * hw.hbmEfficiency;
  if (cEff <= 0 || hbmEffBw <= 0 || !Number.isFinite(cEff) || !Number.isFinite(hbmEffBw)) {
    return invalidResult([{ level: "error", message: "유효 성능 또는 BW가 0 이하이거나 유한하지 않습니다", relatedInputIds: ["hardware"] }]);
  }

  const loadQ = divideChecked(perTensorBytes, hbmEffBw);
  const loadK = divideChecked(perTensorBytes, hbmEffBw);
  const loadV = divideChecked(perTensorBytes, hbmEffBw);
  const qkTime = divideChecked(qkFlops, cEff);
  const scoreTrafficBytes = safeMul(2, scoreProbBytes);
  const scoreStoreLoadTime = scoreTrafficBytes === null ? null : divideChecked(scoreTrafficBytes, hbmEffBw);
  const softmaxTime = divideChecked(softmaxFlops, cEff);
  const probStoreLoadTime = scoreTrafficBytes === null ? null : divideChecked(scoreTrafficBytes, hbmEffBw);
  const pvTime = divideChecked(pvFlops, cEff);
  const outStore = divideChecked(perTensorBytes, hbmEffBw);

  const parts = [loadQ, loadK, loadV, qkTime, scoreStoreLoadTime, softmaxTime, probStoreLoadTime, pvTime, outStore];
  if (parts.some((t) => t === null)) {
    return invalidResult([{ level: "error", message: "시간 계산이 overflow/underflow로 유한한 양수를 만들지 못했습니다", relatedInputIds: [] }]);
  }
  const [lQ, lK, lV, qkT, scoreIo, smT, probIo, pvT, outS] = parts as number[];
  const loadQKV = addFinite(lQ, lK, lV);
  const totalSeconds = loadQKV === null ? null : addFinite(loadQKV, qkT, scoreIo, smT, probIo, pvT, outS);
  if (totalSeconds === null || loadQKV === null) {
    return invalidResult([{ level: "error", message: "총시간 계산이 유한하지 않습니다", relatedInputIds: [] }]);
  }

  const resHbm: Resource = { id: "res-hbm", kind: "hbm", rankIndex: 0 };
  const resCompute: Resource = { id: "res-compute", kind: "compute", rankIndex: 0 };
  const resources = [resHbm, resCompute];

  const tensorQ: Tensor = { id: "tensor-q", label: "Q", shape: [B, H, S, D], precision: "bf16", bytes: perTensorBytes, scope: "global" };
  const tensorK: Tensor = { id: "tensor-k", label: "K", shape: [B, H, S, D], precision: "bf16", bytes: perTensorBytes, scope: "global" };
  const tensorV: Tensor = { id: "tensor-v", label: "V", shape: [B, H, S, D], precision: "bf16", bytes: perTensorBytes, scope: "global" };
  const tensorScore: Tensor = { id: "tensor-score", label: "Score", shape: [B, H, S, S], precision: "fp32", bytes: scoreProbBytes, scope: "global" };
  const tensorProb: Tensor = { id: "tensor-prob", label: "Probability", shape: [B, H, S, S], precision: "fp32", bytes: scoreProbBytes, scope: "global" };
  const tensorOut: Tensor = { id: "tensor-out", label: "Output", shape: [B, H, S, D], precision: "bf16", bytes: perTensorBytes, scope: "global" };
  const tensors = [tensorQ, tensorK, tensorV, tensorScore, tensorProb, tensorOut];

  const b = new MetricBuilder();
  const iB = b.input("input-b", "Batch", B, "count");
  const iH = b.input("input-h", "Heads", H, "count");
  const iS = b.input("input-s", "Sequence", S, "count");
  const iD = b.input("input-d", "Head dimension", D, "count");
  const iQb = b.input("input-q", "q (BF16 byte)", q, "byte");
  const iAb = b.input("input-a", "a (FP32 byte)", a, "byte");
  const iCEff = b.input("input-c-eff", "유효 Compute 성능", cEff, "flop/s");
  const iHbmEffBw = b.input("input-hbm-eff-bw", "유효 HBM BW", hbmEffBw, "byte/s");

  const mPerTensorBytes = b.derived({
    id: "metric-per-tensor-bytes",
    label: "Q/K/V/Output 텐서 크기",
    value: perTensorBytes,
    unit: "byte",
    scope: "global",
    formula: { description: "BHSDq", expression: "B*H*S*D*q", inputMetricIds: [iB.id, iH.id, iS.id, iD.id, iQb.id], substituted: { B, H, S, D, q } },
  });
  const mQBytes = b.derived({ id: "metric-q-bytes", label: "Q 읽기 bytes", value: perTensorBytes, unit: "byte", scope: "global", formula: { description: "Q 논리 크기", expression: "perTensorBytes", inputMetricIds: [mPerTensorBytes.id], substituted: { perTensorBytes } } });
  const mKBytes = b.derived({ id: "metric-k-bytes", label: "K 읽기 bytes", value: perTensorBytes, unit: "byte", scope: "global", formula: { description: "K 논리 크기", expression: "perTensorBytes", inputMetricIds: [mPerTensorBytes.id], substituted: { perTensorBytes } } });
  const mVBytes = b.derived({ id: "metric-v-bytes", label: "V 읽기 bytes", value: perTensorBytes, unit: "byte", scope: "global", formula: { description: "V 논리 크기", expression: "perTensorBytes", inputMetricIds: [mPerTensorBytes.id], substituted: { perTensorBytes } } });
  const mQkFlops = b.derived({
    id: "metric-qk-flops",
    label: "QKᵀ FLOPs",
    value: qkFlops,
    unit: "flop",
    scope: "global",
    formula: { description: "QKᵀ FLOPs = 2BHS²D", expression: "2*B*H*S*S*D", inputMetricIds: [iB.id, iH.id, iS.id, iD.id], substituted: { B, H, S, D } },
  });
  const mPvFlops = b.derived({
    id: "metric-pv-flops",
    label: "PV FLOPs",
    value: pvFlops,
    unit: "flop",
    scope: "global",
    formula: { description: "PV FLOPs = 2BHS²D", expression: "2*B*H*S*S*D", inputMetricIds: [iB.id, iH.id, iS.id, iD.id], substituted: { B, H, S, D } },
  });
  const mSoftmaxFlops = b.derived({
    id: "metric-softmax-flops",
    label: "Softmax FLOPs 상당",
    value: softmaxFlops,
    unit: "flop",
    scope: "global",
    formula: { description: "Softmax 비용 근사 = 5BHS²", expression: "5*B*H*S*S", inputMetricIds: [iB.id, iH.id, iS.id], substituted: { B, H, S } },
  });
  const mScoreProbBytes = b.derived({
    id: "metric-score-prob-bytes",
    label: "Score/Probability 텐서 크기",
    value: scoreProbBytes,
    unit: "byte",
    scope: "global",
    formula: { description: "BHS²a", expression: "B*H*S*S*a", inputMetricIds: [iB.id, iH.id, iS.id, iAb.id], substituted: { B, H, S, a } },
  });
  b.derived({
    id: "metric-hbm-total",
    label: "HBM traffic 합계",
    value: hbmTotal,
    unit: "byte",
    scope: "global",
    relatedResourceIds: [resHbm.id],
    formula: {
      description: "4BHSDq + 4BHS²a",
      expression: "4*perTensorBytes + 4*scoreProbBytes",
      inputMetricIds: [mPerTensorBytes.id, mScoreProbBytes.id],
      substituted: { perTensorBytes, scoreProbBytes },
    },
  });

  const mLoadQ = b.derived({ id: "metric-load-q-time", label: "Q Load 시간", value: lQ, unit: "s", scope: "global", formula: { description: "Q 읽기 / 유효 HBM BW", expression: "perTensorBytes/hbmEffBw", inputMetricIds: [mPerTensorBytes.id, iHbmEffBw.id], substituted: { perTensorBytes, hbmEffBw } } });
  const mLoadK = b.derived({ id: "metric-load-k-time", label: "K Load 시간", value: lK, unit: "s", scope: "global", formula: { description: "K 읽기 / 유효 HBM BW", expression: "perTensorBytes/hbmEffBw", inputMetricIds: [mPerTensorBytes.id, iHbmEffBw.id], substituted: { perTensorBytes, hbmEffBw } } });
  const mLoadV = b.derived({ id: "metric-load-v-time", label: "V Load 시간", value: lV, unit: "s", scope: "global", formula: { description: "V 읽기 / 유효 HBM BW", expression: "perTensorBytes/hbmEffBw", inputMetricIds: [mPerTensorBytes.id, iHbmEffBw.id], substituted: { perTensorBytes, hbmEffBw } } });
  const mLoadQkv = b.derived({
    id: "metric-load-qkv-time",
    label: "QKV Load 시간",
    value: loadQKV,
    unit: "s",
    scope: "global",
    formula: { description: "Q+K+V 읽기 시간 합", expression: "loadQ+loadK+loadV", inputMetricIds: [mLoadQ.id, mLoadK.id, mLoadV.id], substituted: { loadQ: lQ, loadK: lK, loadV: lV } },
  });
  const mQkTime = b.derived({ id: "metric-qk-time", label: "QKᵀ 시간", value: qkT, unit: "s", scope: "global", formula: { description: "FLOPs / 유효 Compute 성능", expression: "qkFlops/cEff", inputMetricIds: [mQkFlops.id, iCEff.id], substituted: { qkFlops, cEff }, assumptionIds: ["asm-attention-throughput-approx"] } });
  const mScoreIo = b.derived({ id: "metric-score-io-time", label: "Score Store/Load 시간", value: scoreIo, unit: "s", scope: "global", formula: { description: "Score FP32 저장+읽기 / 유효 HBM BW", expression: "2*scoreProbBytes/hbmEffBw", inputMetricIds: [mScoreProbBytes.id, iHbmEffBw.id], substituted: { scoreProbBytes, hbmEffBw } } });
  const mSoftmaxTime = b.derived({ id: "metric-softmax-time", label: "Softmax 시간", value: smT, unit: "s", scope: "global", formula: { description: "Softmax FLOPs 상당 / 유효 Compute 성능", expression: "softmaxFlops/cEff", inputMetricIds: [mSoftmaxFlops.id, iCEff.id], substituted: { softmaxFlops, cEff }, assumptionIds: ["asm-attention-throughput-approx"] } });
  const mProbIo = b.derived({ id: "metric-prob-io-time", label: "Probability Store/Load 시간", value: probIo, unit: "s", scope: "global", formula: { description: "Probability FP32 저장+읽기 / 유효 HBM BW", expression: "2*scoreProbBytes/hbmEffBw", inputMetricIds: [mScoreProbBytes.id, iHbmEffBw.id], substituted: { scoreProbBytes, hbmEffBw } } });
  const mPvTime = b.derived({ id: "metric-pv-time", label: "PV 시간", value: pvT, unit: "s", scope: "global", formula: { description: "FLOPs / 유효 Compute 성능", expression: "pvFlops/cEff", inputMetricIds: [mPvFlops.id, iCEff.id], substituted: { pvFlops, cEff }, assumptionIds: ["asm-attention-throughput-approx"] } });
  const mOutStore = b.derived({ id: "metric-out-store-time", label: "Output Store 시간", value: outS, unit: "s", scope: "global", formula: { description: "출력 쓰기 / 유효 HBM BW", expression: "perTensorBytes/hbmEffBw", inputMetricIds: [mPerTensorBytes.id, iHbmEffBw.id], substituted: { perTensorBytes, hbmEffBw } } });
  b.derived({
    id: "metric-total-time",
    label: "총시간",
    value: totalSeconds,
    unit: "s",
    scope: "global",
    formula: {
      description: "총시간 = 각 단계 시간 합 (순차)",
      expression: "loadQKV + qkTime + scoreIo + softmaxTime + probIo + pvTime + outStore",
      inputMetricIds: [mLoadQkv.id, mQkTime.id, mScoreIo.id, mSoftmaxTime.id, mProbIo.id, mPvTime.id, mOutStore.id],
      substituted: { loadQKV, qkTime: qkT, scoreStoreLoadTime: scoreIo, softmaxTime: smT, probStoreLoadTime: probIo, pvTime: pvT, outStore: outS },
    },
  });

  const steps: Step[] = [
    { id: "step-qkv-load", label: "QKV Load", description: "HBM에서 Q/K/V를 읽습니다", precedingStepIds: [], resourceIds: [resHbm.id], tensorIds: [tensorQ.id, tensorK.id, tensorV.id], timeMetricId: mLoadQkv.id, limitingResource: loadQKV > 0 ? "hbm" : null, isAggregate: true },
    { id: "step-qk", label: "QKᵀ", description: "Compute에서 Q·Kᵀ를 계산합니다", precedingStepIds: ["step-qkv-load"], resourceIds: [resCompute.id], tensorIds: [tensorQ.id, tensorK.id, tensorScore.id], timeMetricId: mQkTime.id, limitingResource: qkT > 0 ? "compute" : null, isAggregate: true },
    { id: "step-score-io", label: "Score Store/Load", description: "Score를 HBM에 쓰고 다시 읽습니다 (FP32)", precedingStepIds: ["step-qk"], resourceIds: [resHbm.id], tensorIds: [tensorScore.id], timeMetricId: mScoreIo.id, limitingResource: scoreIo > 0 ? "hbm" : null, isAggregate: true },
    { id: "step-softmax", label: "Softmax", description: "Compute에서 Softmax를 근사 계산합니다", precedingStepIds: ["step-score-io"], resourceIds: [resCompute.id], tensorIds: [tensorScore.id, tensorProb.id], timeMetricId: mSoftmaxTime.id, limitingResource: smT > 0 ? "compute" : null, isAggregate: true },
    { id: "step-prob-io", label: "Probability Store/Load", description: "Probability를 HBM에 쓰고 다시 읽습니다 (FP32)", precedingStepIds: ["step-softmax"], resourceIds: [resHbm.id], tensorIds: [tensorProb.id], timeMetricId: mProbIo.id, limitingResource: probIo > 0 ? "hbm" : null, isAggregate: true },
    { id: "step-pv", label: "PV", description: "Compute에서 Probability·V를 계산합니다", precedingStepIds: ["step-prob-io"], resourceIds: [resCompute.id], tensorIds: [tensorProb.id, tensorV.id, tensorOut.id], timeMetricId: mPvTime.id, limitingResource: pvT > 0 ? "compute" : null, isAggregate: true },
    { id: "step-out-store", label: "Output Store", description: "결과를 HBM에 씁니다", precedingStepIds: ["step-pv"], resourceIds: [resHbm.id], tensorIds: [tensorOut.id], timeMetricId: mOutStore.id, limitingResource: outS > 0 ? "hbm" : null, isAggregate: true },
  ];

  // Distinct per-tensor transfers — Q/K/V are not merged into a single "Q" transfer.
  const transfers: Transfer[] = [
    { id: "transfer-q", from: resHbm.id, to: resCompute.id, tensorId: tensorQ.id, bytesMetricId: mQBytes.id, stepId: "step-qkv-load", trafficClass: "hbm" },
    { id: "transfer-k", from: resHbm.id, to: resCompute.id, tensorId: tensorK.id, bytesMetricId: mKBytes.id, stepId: "step-qkv-load", trafficClass: "hbm" },
    { id: "transfer-v", from: resHbm.id, to: resCompute.id, tensorId: tensorV.id, bytesMetricId: mVBytes.id, stepId: "step-qkv-load", trafficClass: "hbm" },
    { id: "transfer-score-store", from: resCompute.id, to: resHbm.id, tensorId: tensorScore.id, bytesMetricId: mScoreProbBytes.id, stepId: "step-score-io", trafficClass: "hbm" },
    { id: "transfer-score-load", from: resHbm.id, to: resCompute.id, tensorId: tensorScore.id, bytesMetricId: mScoreProbBytes.id, stepId: "step-score-io", trafficClass: "hbm" },
    { id: "transfer-prob-store", from: resCompute.id, to: resHbm.id, tensorId: tensorProb.id, bytesMetricId: mScoreProbBytes.id, stepId: "step-prob-io", trafficClass: "hbm" },
    { id: "transfer-prob-load", from: resHbm.id, to: resCompute.id, tensorId: tensorProb.id, bytesMetricId: mScoreProbBytes.id, stepId: "step-prob-io", trafficClass: "hbm" },
    { id: "transfer-out", from: resCompute.id, to: resHbm.id, tensorId: tensorOut.id, bytesMetricId: mPerTensorBytes.id, stepId: "step-out-store", trafficClass: "hbm" },
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

/** SRAM sensitivity is unsupported — no SRAM BW model defined for Attention (02-analysis-models.md). */
export const SUPPORTED_SENSITIVITY_RESOURCES = ["compute", "hbm"] as const;
