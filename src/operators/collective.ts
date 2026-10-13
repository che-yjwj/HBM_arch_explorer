// Collective operator: Ring All-Reduce/All-Gather/Reduce-Scatter, Pairwise All-to-All. (02-analysis-models.md Collective section)
import type {
  Assumption,
  CollectiveWorkload,
  CoreAnalysisResult,
  Diagnostic,
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
import { addFinite, divideChecked, multiplyFiniteChecked, safeMul } from "../engine/numeric";

const Q_BYTES = 2; // BF16
export const MODEL_VERSION = "collective-v1";

const ASSUMPTIONS: Assumption[] = [
  {
    id: "asm-collective-logical",
    description: "통신은 논리적 Rank 간 경로이며 물리적 NVLink·NIC·스위치 사양을 의미하지 않음",
    appliesTo: "collective",
    limitations: "Reduction 연산 비용과 로컬 메모리 복사 비용은 제외",
  },
];

interface CollectiveShape {
  sendBytesPerRank: number;
  rounds: number;
  outputBytes: number; // the resulting tensor size most relevant to the op
  outputLabel: string;
}

function shapeFor(op: CollectiveWorkload["op"], p: number, s: number): CollectiveShape | null {
  switch (op) {
    case "all-reduce": {
      const send = safeMul(2, p - 1, s);
      if (send === null) return null;
      const rounds = safeMul(2, p - 1);
      if (rounds === null) return null;
      return { sendBytesPerRank: send / p, rounds, outputBytes: s, outputLabel: "All-Reduce 결과(각 Rank 전체 텐서)" };
    }
    case "all-gather": {
      const send = safeMul(p - 1, s);
      if (send === null) return null;
      const out = safeMul(p, s);
      if (out === null) return null;
      return { sendBytesPerRank: send, rounds: p - 1, outputBytes: out, outputLabel: "All-Gather 최종 텐서" };
    }
    case "reduce-scatter": {
      const send = safeMul(p - 1, s);
      if (send === null) return null;
      return { sendBytesPerRank: send / p, rounds: p - 1, outputBytes: s / p, outputLabel: "Reduce-Scatter 출력 shard" };
    }
    case "all-to-all": {
      const send = safeMul(p - 1, s);
      if (send === null) return null;
      return { sendBytesPerRank: send / p, rounds: p - 1, outputBytes: s, outputLabel: "All-to-All 전체 송신 buffer" };
    }
  }
}

/** Validates S's integer-byte and BF16-element alignment, plus the P*q divisibility required for all ops except all-gather. */
export function validateCollective(w: CollectiveWorkload, rankCount: number): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  const q = Q_BYTES;
  if (!Number.isFinite(w.sBytes) || w.sBytes <= 0 || !Number.isInteger(w.sBytes) || !Number.isSafeInteger(w.sBytes)) {
    diagnostics.push({ level: "error", message: "S는 양의 안전 정수 byte여야 합니다", relatedInputIds: ["sBytes"] });
    return diagnostics;
  }
  if (w.sBytes % q !== 0) {
    diagnostics.push({ level: "error", message: `S는 BF16 원소 정렬(${q}의 배수)이어야 합니다`, relatedInputIds: ["sBytes"] });
    return diagnostics;
  }
  if (w.op !== "all-gather") {
    const divisor = rankCount * q;
    if (w.sBytes % divisor !== 0) {
      const nearestCandidate = Math.ceil(w.sBytes / divisor) * divisor;
      const nearest = Number.isSafeInteger(nearestCandidate) ? nearestCandidate : null;
      diagnostics.push({
        level: "error",
        message: `S는 P*q(${divisor})의 배수여야 합니다. S mod ${divisor} = ${w.sBytes % divisor}`,
        relatedInputIds: ["sBytes"],
        fix: nearest === null ? "안전 정수 범위 안의 P*q 배수로 S를 줄이세요" : `가장 가까운 유효 크기: ${nearest} bytes`,
      });
    }
  }
  return diagnostics;
}

export function analyzeCollective(w: CollectiveWorkload, hw: Hardware): CoreAnalysisResult {
  resetIdCounter();
  const p = hw.rankCount;
  const diagnostics = validateCollective(w, p);
  if (diagnostics.length > 0) return invalidResult(diagnostics);

  const linkEffBw = hw.linkBwBytes * hw.linkEfficiency;
  if (linkEffBw <= 0 || !Number.isFinite(linkEffBw)) {
    return invalidResult([{ level: "error", message: "Link 유효 BW가 0 이하이거나 유한하지 않습니다", relatedInputIds: ["hardware"] }]);
  }

  const b = new MetricBuilder();
  const iP = b.input("input-p", "P (Rank 수)", p, "count");
  const iS = b.input("input-s", "S", w.sBytes, "byte");
  const iLinkEffBw = b.input("input-link-eff-bw", "유효 Link BW", linkEffBw, "byte/s");
  const iLatency = b.input("input-link-latency", "Link latency", hw.linkLatencySeconds, "s");

  const resLink: Resource = { id: "res-link", kind: "link", rankIndex: -1 };

  if (p === 1) {
    const mSend = b.derived({ id: "metric-send-bytes", label: "Rank당 송신량", value: 0, unit: "byte", scope: "per-rank", relatedResourceIds: [resLink.id], formula: { description: "P=1: 송신 없음", expression: "0", inputMetricIds: [iP.id], substituted: { P: p } } });
    const mRecv = b.derived({ id: "metric-recv-bytes", label: "Rank당 수신량", value: 0, unit: "byte", scope: "per-rank", relatedResourceIds: [resLink.id], formula: { description: "P=1: 수신 없음", expression: "0", inputMetricIds: [iP.id], substituted: { P: p } } });
    const mRounds = b.derived({ id: "metric-rounds", label: "라운드 수", value: 0, unit: "count", scope: "global", formula: { description: "P=1: 라운드 없음", expression: "0", inputMetricIds: [iP.id], substituted: { P: p } } });
    const mTime = b.derived({ id: "metric-total-time", label: "총시간", value: 0, unit: "s", scope: "per-rank", formula: { description: "P=1: 통신 없음, 총시간 0", expression: "0", inputMetricIds: [iP.id], substituted: { P: p } } });
    void mSend;
    void mRecv;
    void mRounds;
    const step: Step = {
      id: "step-comm",
      label: "통신 없음 (P=1)",
      description: "단일 Rank: 송신·수신·시간·라운드 모두 0",
      precedingStepIds: [],
      resourceIds: [resLink.id],
      tensorIds: [],
      timeMetricId: mTime.id,
      limitingResource: null,
      isAggregate: true,
    };
    return {
      valid: true,
      modelVersion: MODEL_VERSION,
      resources: [resLink],
      tensors: [],
      steps: [step],
      transfers: [],
      metrics: b.metrics,
      formulas: b.formulas,
      assumptions: ASSUMPTIONS,
      diagnostics: [],
      totalSeconds: 0,
      contributions: [],
    };
  }

  const shape = shapeFor(w.op, p, w.sBytes);
  if (shape === null) {
    return invalidResult([{ level: "error", message: "통신량 계산이 안전 정수 상한을 초과했습니다", relatedInputIds: ["sBytes"] }]);
  }
  const commTime = divideChecked(shape.sendBytesPerRank, linkEffBw);
  if (commTime === null) {
    return invalidResult([{ level: "error", message: "통신 시간 계산이 overflow/underflow로 유한한 양수를 만들지 못했습니다", relatedInputIds: [] }]);
  }
  const latencyTime = multiplyFiniteChecked(shape.rounds, hw.linkLatencySeconds);
  const totalSeconds = latencyTime === null ? null : addFinite(latencyTime, commTime);
  if (totalSeconds === null) {
    return invalidResult([{ level: "error", message: "통신 시간이 유한하지 않습니다", relatedInputIds: [] }]);
  }

  const resources = [resLink];
  const tensorIn: Tensor = { id: "tensor-s", label: "S", shape: [w.sBytes], precision: "bf16", bytes: w.sBytes, scope: "per-rank" };
  const tensorOut: Tensor = {
    id: "tensor-out",
    label: shape.outputLabel,
    shape: [shape.outputBytes],
    precision: "bf16",
    bytes: shape.outputBytes,
    scope: w.op === "all-gather" ? "global" : "per-rank",
  };

  const mSend = b.derived({
    id: "metric-send-bytes",
    label: "Rank당 송신량",
    value: shape.sendBytesPerRank,
    unit: "byte",
    scope: "per-rank",
    relatedResourceIds: [resLink.id],
    formula: { description: `${w.op} Rank당 송신량`, expression: "f(P,S)", inputMetricIds: [iP.id, iS.id], substituted: { P: p, S: w.sBytes } },
  });
  const mRounds = b.derived({
    id: "metric-rounds",
    label: "라운드 수",
    value: shape.rounds,
    unit: "count",
    scope: "global",
    formula: { description: `${w.op} 라운드 수`, expression: "f(P)", inputMetricIds: [iP.id], substituted: { P: p } },
  });
  const mCommTime = b.derived({
    id: "metric-comm-time",
    label: "통신 시간(지연 제외)",
    value: commTime,
    unit: "s",
    scope: "per-rank",
    formula: {
      description: "통신 시간(지연 제외) = Rank당 송신 bytes / 유효 Link BW",
      expression: "sendBytes/linkEffBw",
      inputMetricIds: [mSend.id, iLinkEffBw.id],
      substituted: { sendBytes: shape.sendBytesPerRank, linkEffBw },
    },
  });
  const mTotal = b.derived({
    id: "metric-total-time",
    label: "총시간",
    value: totalSeconds,
    unit: "s",
    scope: "per-rank",
    formula: {
      description: "총시간 = 라운드 수 * latency + 통신 시간(지연 제외)",
      expression: "rounds*latency + commTime",
      inputMetricIds: [mRounds.id, iLatency.id, mCommTime.id],
      substituted: { rounds: shape.rounds, latency: hw.linkLatencySeconds, commTime },
    },
  });

  const step: Step = {
    id: "step-comm",
    label: `${w.op} 통신`,
    description: "논리적 Rank 사이에서 데이터를 교환합니다 (물리 라우팅 미모델링)",
    precedingStepIds: [],
    resourceIds: [resLink.id],
    tensorIds: [tensorIn.id, tensorOut.id],
    timeMetricId: mTotal.id,
    limitingResource: totalSeconds > 0 ? "link" : null,
    isAggregate: true,
  };

  const transfers: Transfer[] = [
    { id: "transfer-send", from: resLink.id, to: resLink.id, tensorId: tensorIn.id, bytesMetricId: mSend.id, stepId: step.id, trafficClass: "link" },
  ];

  const contributions: TimeContribution[] =
    totalSeconds > 0 ? [{ limitingResources: "link", stepIds: [step.id], totalSeconds, ratioOfTotal: 1 }] : [];

  const result: ValidCoreAnalysisResult = {
    valid: true,
    modelVersion: MODEL_VERSION,
    resources,
    tensors: [tensorIn, tensorOut],
    steps: [step],
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

/** HBM and Compute sensitivity are unsupported for Collective (02-analysis-models.md). */
export const SUPPORTED_SENSITIVITY_RESOURCES = ["link"] as const;
