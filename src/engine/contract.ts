import type { CoreAnalysisResult, Formula, Metric, Step } from "../domain/types";

export interface ContractIssue { code: string; message: string; }

/** Validate reference integrity and acyclicity before an analysis result reaches the UI. */
export function validateAnalysisContract(result: CoreAnalysisResult): ContractIssue[] {
  if (!result.valid) return [];
  const issues: ContractIssue[] = [];
  const metrics = new Map<string, Metric>();
  const formulas = new Map<string, Formula>();
  const steps = new Map<string, Step>();
  const resources = new Set<string>();
  const tensors = new Set<string>();
  const assumptions = new Set(result.assumptions.map((assumption) => assumption.id));
  for (const resource of result.resources) {
    if (resources.has(resource.id)) issues.push({ code: "duplicate-resource", message: `중복 Resource ID: ${resource.id}` });
    resources.add(resource.id);
  }
  for (const tensor of result.tensors) {
    if (tensors.has(tensor.id)) issues.push({ code: "duplicate-tensor", message: `중복 Tensor ID: ${tensor.id}` });
    tensors.add(tensor.id);
  }
  if (assumptions.size !== result.assumptions.length) issues.push({ code: "duplicate-assumption", message: "중복 Assumption ID가 있습니다" });
  for (const metric of result.metrics) {
    if (metrics.has(metric.id)) issues.push({ code: "duplicate-metric", message: `중복 Metric ID: ${metric.id}` });
    metrics.set(metric.id, metric);
  }
  for (const formula of result.formulas) {
    if (formulas.has(formula.id)) issues.push({ code: "duplicate-formula", message: `중복 Formula ID: ${formula.id}` });
    formulas.set(formula.id, formula);
  }
  for (const step of result.steps) {
    if (steps.has(step.id)) issues.push({ code: "duplicate-step", message: `중복 Step ID: ${step.id}` });
    steps.set(step.id, step);
  }

  for (const metric of result.metrics) {
    if (metric.formulaId !== null && !formulas.has(metric.formulaId)) {
      issues.push({ code: "missing-formula", message: `${metric.id}가 참조하는 Formula ${metric.formulaId}가 없습니다` });
    }
    for (const id of metric.relatedStepIds) if (!steps.has(id)) issues.push({ code: "missing-metric-step", message: `${metric.id}의 관련 Step ${id}가 없습니다` });
    for (const id of metric.relatedResourceIds) if (!resources.has(id)) issues.push({ code: "missing-metric-resource", message: `${metric.id}의 관련 Resource ${id}가 없습니다` });
  }
  for (const formula of result.formulas) {
    for (const inputId of formula.inputMetricIds) {
      if (!metrics.has(inputId)) issues.push({ code: "missing-input-metric", message: `${formula.id}가 참조하는 Metric ${inputId}가 없습니다` });
    }
    for (const assumptionId of formula.assumptionIds) {
      if (!assumptions.has(assumptionId)) issues.push({ code: "missing-assumption", message: `${formula.id}가 참조하는 Assumption ${assumptionId}가 없습니다` });
    }
  }
  for (const step of result.steps) {
    if (!metrics.has(step.timeMetricId)) issues.push({ code: "missing-step-time", message: `${step.id}의 시간 Metric ${step.timeMetricId}가 없습니다` });
    else if (metrics.get(step.timeMetricId)?.unit !== "s") issues.push({ code: "invalid-step-time-unit", message: `${step.id}의 시간 Metric 단위가 초(s)가 아닙니다` });
    for (const id of step.precedingStepIds) if (!steps.has(id)) issues.push({ code: "missing-preceding-step", message: `${step.id}의 선행 Step ${id}가 없습니다` });
    for (const id of step.resourceIds) if (!resources.has(id)) issues.push({ code: "missing-step-resource", message: `${step.id}의 Resource ${id}가 없습니다` });
    for (const id of step.tensorIds) if (!tensors.has(id)) issues.push({ code: "missing-step-tensor", message: `${step.id}의 Tensor ${id}가 없습니다` });
  }
  for (const transfer of result.transfers) {
    if (!resources.has(transfer.from)) issues.push({ code: "missing-transfer-source", message: `${transfer.id}의 출발 Resource가 없습니다` });
    if (!resources.has(transfer.to)) issues.push({ code: "missing-transfer-target", message: `${transfer.id}의 도착 Resource가 없습니다` });
    if (!tensors.has(transfer.tensorId)) issues.push({ code: "missing-transfer-tensor", message: `${transfer.id}의 Tensor가 없습니다` });
    if (!steps.has(transfer.stepId)) issues.push({ code: "missing-transfer-step", message: `${transfer.id}의 Step이 없습니다` });
    const bytes = metrics.get(transfer.bytesMetricId);
    if (!bytes) issues.push({ code: "missing-transfer-bytes", message: `${transfer.id}의 bytes Metric이 없습니다` });
    else if (bytes.unit !== "byte") issues.push({ code: "invalid-transfer-bytes-unit", message: `${transfer.id}의 Metric 단위가 byte가 아닙니다` });
  }

  const visitAcyclic = (ids: string[], getNext: (id: string) => string[], code: string, label: string) => {
    const visiting = new Set<string>();
    const visited = new Set<string>();
    const visit = (id: string) => {
      if (visited.has(id)) return;
      if (visiting.has(id)) {
        issues.push({ code, message: `${label} 참조에 순환이 있습니다: ${id}` });
        return;
      }
      visiting.add(id);
      for (const next of getNext(id)) if (ids.includes(next)) visit(next);
      visiting.delete(id);
      visited.add(id);
    };
    for (const id of ids) visit(id);
  };
  const metricIds = [...metrics.keys()];
  visitAcyclic(metricIds, (id) => {
    const metric = metrics.get(id);
    const formula = metric?.formulaId ? formulas.get(metric.formulaId) : undefined;
    return formula?.inputMetricIds ?? [];
  }, "formula-cycle", "Formula");
  const stepIds = [...steps.keys()];
  visitAcyclic(stepIds, (id) => steps.get(id)?.precedingStepIds ?? [], "step-cycle", "Step");
  return issues;
}
