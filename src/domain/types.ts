// Domain types. No React / React Flow dependency here (see 07-implementation-plan.md).

export type OperatorKind = "gemm" | "attention" | "moe" | "collective";

export type Precision = "bf16"; // P0: single BF16/FP32 combo (02-analysis-models.md)

export interface GemmWorkload {
  kind: "gemm";
  precision: Precision;
  m: number;
  n: number;
  k: number;
  tileM: number;
  tileN: number;
  tileK: number;
}

export type CollectiveOp = "all-reduce" | "all-gather" | "reduce-scatter" | "all-to-all";

export interface CollectiveWorkload {
  kind: "collective";
  precision: Precision;
  op: CollectiveOp;
  /** S in bytes, meaning depends on op (02-analysis-models.md) */
  sBytes: number;
}

export interface AttentionWorkload {
  kind: "attention";
  precision: Precision;
  batch: number;
  heads: number;
  seq: number;
  headDim: number;
}

export interface MoeWorkload {
  kind: "moe";
  precision: Precision;
  tokens: number;
  hidden: number;
  ffn: number;
  experts: number;
  topK: number;
}

export type Workload = GemmWorkload | CollectiveWorkload | AttentionWorkload | MoeWorkload;

export interface Hardware {
  id: string;
  name: string;
  rankCount: number;
  computeFlops: number; // per-rank nominal FLOP/s
  computeEfficiency: number; // (0,1]
  sramCapacityBytes: number; // per-rank
  sramBwBytes: number; // per-rank nominal byte/s
  sramEfficiency: number;
  hbmBwBytes: number; // per-rank nominal byte/s
  hbmEfficiency: number;
  linkBwBytes: number; // per-rank nominal byte/s
  linkEfficiency: number;
  linkLatencySeconds: number;
}

export type ResourceKind = "hbm" | "sram" | "compute" | "link";

export interface Resource {
  id: string;
  kind: ResourceKind;
  rankIndex: number; // representative rank index, or -1 for "all ranks" aggregate
}

export interface Tensor {
  id: string;
  label: string;
  shape: number[];
  precision: Precision | "fp32";
  bytes: number;
  scope: "global" | "per-rank";
}

export type TrafficClass = "hbm" | "sram" | "link";

export interface Transfer {
  id: string;
  from: string; // Resource id
  to: string; // Resource id
  tensorId: string;
  bytesMetricId: string;
  stepId: string;
  trafficClass: TrafficClass;
}

export type MetricUnit = "s" | "byte" | "flop" | "flop/s" | "byte/s" | "ratio" | "count" | "none";

export interface Metric {
  id: string;
  label: string;
  value: number;
  unit: MetricUnit;
  scope: "global" | "per-rank";
  formulaId: string | null;
  relatedStepIds: string[];
  relatedResourceIds: string[];
}

export interface Formula {
  id: string;
  description: string;
  expression: string; // display-only, never eval'd
  inputMetricIds: string[];
  substituted: Record<string, number>;
  assumptionIds: string[];
}

export interface Assumption {
  id: string;
  description: string;
  appliesTo: string;
  limitations: string;
}

export type DiagnosticLevel = "error" | "warning";

export interface Diagnostic {
  level: DiagnosticLevel;
  message: string;
  relatedInputIds: string[];
  fix?: string;
}

export type LimitingResource = "compute" | "sram" | "hbm" | "link" | "compute+sram";

export interface Step {
  id: string;
  label: string;
  description: string; // "읽고 -> 계산 -> 쓰기" one-liner
  precedingStepIds: string[];
  resourceIds: string[];
  tensorIds: string[];
  timeMetricId: string;
  limitingResource: LimitingResource | null; // null only if step time is 0
  isAggregate: boolean;
}

export type SensitivityStatus = "ok" | "no-work" | "unsupported" | "invalid";

export interface SensitivityResult {
  resourceKind: ResourceKind;
  status: SensitivityStatus;
  baselineSeconds: number | null;
  changedSeconds: number | null;
  savedSeconds: number | null;
  speedup: number | null;
  reductionRatio: number | null;
  reason: string | null;
}

export interface TimeContribution {
  limitingResources: LimitingResource;
  stepIds: string[];
  totalSeconds: number;
  ratioOfTotal: number | null;
}

interface CoreAnalysisResultBase {
  modelVersion: string;
  resources: Resource[];
  tensors: Tensor[];
  steps: Step[];
  transfers: Transfer[];
  metrics: Metric[];
  formulas: Formula[];
  assumptions: Assumption[];
  diagnostics: Diagnostic[];
}

export interface ValidCoreAnalysisResult extends CoreAnalysisResultBase {
  valid: true;
  totalSeconds: number;
  contributions: TimeContribution[];
}

export interface InvalidCoreAnalysisResult extends CoreAnalysisResultBase {
  valid: false;
  totalSeconds: null;
  /** safely-derived explanatory values only, e.g. required SRAM bytes */
  explain: Record<string, number>;
}

export type CoreAnalysisResult = ValidCoreAnalysisResult | InvalidCoreAnalysisResult;

export interface AnalysisResult {
  core: CoreAnalysisResult;
  sensitivity: SensitivityResult[]; // one per supported-or-attempted resource kind
}

export interface BaselineSnapshot {
  workload: Workload;
  hardware: Hardware;
  modelVersion: string;
  result: ValidCoreAnalysisResult;
}
