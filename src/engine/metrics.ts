import type { Formula, Metric, MetricUnit } from "../domain/types";

let counter = 0;
/** Deterministic-enough IDs for a single analyze() call; reset per call via resetIdCounter(). */
export function resetIdCounter(): void {
  counter = 0;
}
export function nextId(prefix: string): string {
  counter += 1;
  return `${prefix}-${counter}`;
}

export interface MetricSpec {
  id: string;
  label: string;
  value: number;
  unit: MetricUnit;
  scope: "global" | "per-rank";
  relatedStepIds?: string[];
  relatedResourceIds?: string[];
  formula?: {
    description: string;
    expression: string;
    inputMetricIds: string[];
    substituted: Record<string, number>;
    assumptionIds?: string[];
  };
}

/** Builds a Metric and, if a formula spec is given, its paired Formula sharing the same substituted values. */
export function makeMetric(spec: MetricSpec): { metric: Metric; formula: Formula | null } {
  if (!spec.formula) {
    return {
      metric: {
        id: spec.id,
        label: spec.label,
        value: spec.value,
        unit: spec.unit,
        scope: spec.scope,
        formulaId: null,
        relatedStepIds: spec.relatedStepIds ?? [],
        relatedResourceIds: spec.relatedResourceIds ?? [],
      },
      formula: null,
    };
  }
  const formulaId = nextId("f");
  const formula: Formula = {
    id: formulaId,
    description: spec.formula.description,
    expression: spec.formula.expression,
    inputMetricIds: spec.formula.inputMetricIds,
    substituted: spec.formula.substituted,
    assumptionIds: spec.formula.assumptionIds ?? [],
  };
  const metric: Metric = {
    id: spec.id,
    label: spec.label,
    value: spec.value,
    unit: spec.unit,
    scope: spec.scope,
    formulaId,
    relatedStepIds: spec.relatedStepIds ?? [],
    relatedResourceIds: spec.relatedResourceIds ?? [],
  };
  return { metric, formula };
}

/**
 * Accumulates Metrics/Formulas for one analyze() call and tracks every raw scalar input as its own
 * leaf Metric (formulaId null), so every Formula.inputMetricIds can point at a real, navigable Metric
 * instead of only showing an inline substituted number (03-data-and-engine.md Trace 연결).
 */
export class MetricBuilder {
  readonly metrics: Metric[] = [];
  readonly formulas: Formula[] = [];
  private readonly byId = new Map<string, Metric>();

  /** Registers a raw input value (workload/hardware field) as a leaf Metric. Idempotent per id. */
  input(id: string, label: string, value: number, unit: MetricUnit, scope: "global" | "per-rank" = "global"): Metric {
    const existing = this.byId.get(id);
    if (existing) return existing;
    const metric: Metric = { id, label, value, unit, scope, formulaId: null, relatedStepIds: [], relatedResourceIds: [] };
    this.metrics.push(metric);
    this.byId.set(id, metric);
    return metric;
  }

  /** Registers a derived Metric with its Formula; inputMetricIds should reference real Metric ids (incl. input()s). */
  derived(spec: MetricSpec): Metric {
    const { metric, formula } = makeMetric(spec);
    this.metrics.push(metric);
    if (formula) this.formulas.push(formula);
    this.byId.set(metric.id, metric);
    return metric;
  }

  get(id: string): Metric | undefined {
    return this.byId.get(id);
  }
}
