import { useState } from "react";
import type { CoreAnalysisResult, ResourceKind, ValidCoreAnalysisResult } from "../../domain/types";
import type { HardwareDraft, InspectorTarget } from "../state";
import { formatBytes, formatBwGBs, formatFlopsPerSec, formatSeconds, RESOURCE_LABEL } from "../format";

interface Props {
  target: InspectorTarget;
  result: CoreAnalysisResult;
  hardwareDraft: HardwareDraft;
  onHardwareFieldChange: (field: keyof HardwareDraft, value: string) => void;
  onSelectMetric: (metricId: string) => void;
  onSelectStep: (stepId: string) => void;
  onSelectResource: (resourceId: string) => void;
  onEditInput: (metricId: string) => void;
  onClose: () => void;
}

const RESOURCE_FIELD_MAP: Record<ResourceKind, Array<{ field: keyof HardwareDraft; label: string }>> = {
  hbm: [
    { field: "hbmBwBytes", label: "HBM BW (byte/s)" },
    { field: "hbmEfficiency", label: "HBM 효율 (0~1)" },
  ],
  sram: [
    { field: "sramCapacityBytes", label: "SRAM 용량 (byte)" },
    { field: "sramBwBytes", label: "SRAM BW (byte/s)" },
    { field: "sramEfficiency", label: "SRAM 효율 (0~1)" },
  ],
  compute: [
    { field: "computeFlops", label: "Compute 성능 (FLOP/s)" },
    { field: "computeEfficiency", label: "Compute 효율 (0~1)" },
  ],
  link: [
    { field: "linkBwBytes", label: "Link BW (byte/s)" },
    { field: "linkEfficiency", label: "Link 효율 (0~1)" },
    { field: "linkLatencySeconds", label: "Link latency (s)" },
  ],
};

export function Inspector({ target, result, hardwareDraft, onHardwareFieldChange, onSelectMetric, onSelectStep, onSelectResource, onEditInput, onClose }: Props) {
  const [metricHistory, setMetricHistory] = useState<string[]>([]);
  if (!target) {
    return <div style={{ color: "#777", fontSize: 13 }}>자원·경로·수치를 선택하면 설정과 근거가 여기 표시됩니다.</div>;
  }

  if (target.type === "resource") {
    const kind = resourceIdToKind(target.id);
    const fields = kind ? RESOURCE_FIELD_MAP[kind] : [];
    return (
      <div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <strong>{kind ? RESOURCE_LABEL[kind] : target.id} 설정</strong>
          <button type="button" onClick={onClose}>
            닫기
          </button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 8 }}>
          {fields.map(({ field, label }) => (
            <label key={field} style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
              {label}
              <input
                id={`hardware-${field}`}
                value={hardwareDraft[field]}
                onChange={(e) => onHardwareFieldChange(field, e.target.value)}
                style={{ padding: 4 }}
              />
              {readableHardwareValue(field, hardwareDraft[field]) && (
                <span style={{ color: "#666", fontSize: 11 }}>
                  보기 좋은 단위: {readableHardwareValue(field, hardwareDraft[field])}
                </span>
              )}
            </label>
          ))}
        </div>
      </div>
    );
  }

  if (target.type === "metric") {
    if (!result.valid) return <div>유효하지 않은 결과에는 수치 근거가 없습니다.</div>;
    return <MetricTrace
      result={result}
      metricId={target.id}
      onSelectMetric={(metricId) => { setMetricHistory((history) => [...history, target.id]); onSelectMetric(metricId); }}
      onSelectStep={onSelectStep}
      onSelectResource={onSelectResource}
      onEditInput={onEditInput}
      onBack={metricHistory.length > 0 ? () => { const previous = metricHistory[metricHistory.length - 1]; setMetricHistory((history) => history.slice(0, -1)); onSelectMetric(previous); } : undefined}
      onClose={onClose}
    />;
  }

  if (target.type === "transfer") {
    if (!result.valid) return <div>유효하지 않은 결과에는 전송 정보가 없습니다.</div>;
    const transfer = result.transfers.find((t) => t.id === target.id);
    if (!transfer) return <div>전송 경로를 찾을 수 없습니다.</div>;
    const metric = result.metrics.find((m) => m.id === transfer.bytesMetricId);
    const tensor = result.tensors.find((t) => t.id === transfer.tensorId);
    return (
      <div>
        <strong>경로: {tensor?.label}</strong>
        <button type="button" onClick={onClose} style={{ marginLeft: 8 }}>
          닫기
        </button>
        <p style={{ fontSize: 13 }}>
          {transfer.trafficClass === "link" && transfer.from === transfer.to
            ? transfer.id === "transfer-combine" ? "peer Ranks → Rank 0 (논리 Link)" : "Rank 0 ↔ peer Ranks (논리 Link, 대칭 송수신)"
            : `${resourceLabel(transfer.from)} → ${resourceLabel(transfer.to)}`}, trafficClass={transfer.trafficClass}
        </p>
        <p style={{ fontSize: 13 }}>
          누적 이동량: {metric ? formatBytes(metric.value) : "–"} (Tensor 논리 크기 {tensor ? formatBytes(tensor.bytes) : "–"})
        </p>
      </div>
    );
  }

  return null;
}

function readableHardwareValue(field: keyof HardwareDraft, raw: string): string | null {
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) return null;
  if (field === "sramCapacityBytes") return formatBytes(value);
  if (field === "computeFlops") return formatFlopsPerSec(value);
  if (field === "sramBwBytes" || field === "hbmBwBytes" || field === "linkBwBytes") return formatBwGBs(value);
  return null;
}

function resourceIdToKind(id: string): ResourceKind | null {
  if (id === "res-hbm") return "hbm";
  if (id === "res-sram") return "sram";
  if (id === "res-compute") return "compute";
  if (id === "res-link") return "link";
  return null;
}

function resourceLabel(id: string): string {
  const kind = resourceIdToKind(id);
  return kind ? RESOURCE_LABEL[kind] : id;
}

function MetricTrace({ result, metricId, onSelectMetric, onSelectStep, onSelectResource, onEditInput, onBack, onClose }: { result: ValidCoreAnalysisResult; metricId: string; onSelectMetric: (id: string) => void; onSelectStep: (id: string) => void; onSelectResource: (id: string) => void; onEditInput: (id: string) => void; onBack?: () => void; onClose: () => void }) {
  const metric = result.metrics.find((m) => m.id === metricId);
  if (!metric) return <div>지표를 찾을 수 없습니다.</div>;
  const formula = metric.formulaId ? result.formulas.find((f) => f.id === metric.formulaId) : null;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <strong>
          {metric.label}: {metric.unit === "s" ? formatSeconds(metric.value) : metric.value}
          {metric.unit === "byte" ? " B" : ""}
        </strong>
        {onBack && <button type="button" onClick={onBack}>이전 근거</button>}
        <button type="button" onClick={onClose}>
          닫기
        </button>
      </div>
      {formula && (
        <div style={{ marginTop: 8, fontSize: 13 }}>
          <div>
            산식: <code>{formula.expression}</code>
          </div>
          <div>{formula.description}</div>
          {formula.inputMetricIds.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <strong>입력 지표</strong>
              <ul>
                {formula.inputMetricIds.map((inputId) => {
                  const inputMetric = result.metrics.find((item) => item.id === inputId);
                  if (!inputMetric) return <li key={inputId}>{inputId} (누락)</li>;
                  return (
                    <li key={inputId}>
                      <button type="button" onClick={() => onSelectMetric(inputId)}>{inputMetric.label}: {inputMetric.value} {inputMetric.unit}</button>
                      {inputMetric.formulaId === null && inputFieldId(inputId, result.modelVersion) && (
                        <button type="button" onClick={() => onEditInput(inputFieldId(inputId, result.modelVersion)!)} style={{ marginLeft: 4 }}>입력 수정</button>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
          <table style={{ marginTop: 4 }}>
            <tbody>
              {Object.entries(formula.substituted).map(([k, v]) => (
                <tr key={k}>
                  <td style={{ paddingRight: 8, color: "#555" }}>{k}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {formula.assumptionIds.length > 0 && (
            <ul style={{ fontSize: 12, color: "#555" }}>
              {formula.assumptionIds.map((aid) => {
                const a = result.assumptions.find((x) => x.id === aid);
                return a ? <li key={aid}>{a.description}</li> : null;
              })}
            </ul>
          )}
        </div>
      )}
      {(() => {
        const relatedSteps = result.steps.filter((step) => step.timeMetricId === metric.id || metric.relatedStepIds.includes(step.id));
        return relatedSteps.length > 0 || metric.relatedResourceIds.length > 0 ? (
          <div style={{ marginTop: 8 }}>
            {relatedSteps.map((step) => <button key={step.id} type="button" onClick={() => onSelectStep(step.id)}>단계로 이동: {step.label}</button>)}
            {metric.relatedResourceIds.map((resourceId) => <button key={resourceId} type="button" onClick={() => onSelectResource(resourceId)} style={{ marginLeft: 4 }}>자원 설정: {resourceLabel(resourceId)}</button>)}
          </div>
        ) : null;
      })()}
      {metric.formulaId === null && (
        <div style={{ fontSize: 12, color: "#777", marginTop: 4 }}>원시 입력값이며 추가 산식이 없습니다.</div>
      )}
    </div>
  );
}

function inputFieldId(metricId: string, modelVersion: string): string | null {
  const direct: Record<string, string> = {
    "input-m": "gemm-m", "input-n": "gemm-n", "input-b": "attention-batch", "input-h": "attention-heads",
    "input-t": "moe-tokens", "input-e": "moe-experts", "input-f": "moe-ffn",
    "input-p": "hardware-rankCount", "input-i": "gemm-tileM", "input-j": "gemm-tileN", "input-l": "gemm-tileK",
    "input-tmp": "gemm-tileM", "input-tnp": "gemm-tileN", "input-tkp": "gemm-tileK",
    "input-c-eff": "hardware-computeFlops", "input-sram-eff-bw": "hardware-sramBwBytes", "input-hbm-eff-bw": "hardware-hbmBwBytes", "input-link-eff-bw": "hardware-linkBwBytes", "input-link-latency": "hardware-linkLatencySeconds",
  };
  if (modelVersion === "gemm-v1" && metricId === "input-k") return "gemm-k";
  if (modelVersion === "attention-v1" && metricId === "input-s") return "attention-seq";
  if (modelVersion === "attention-v1" && metricId === "input-d") return "attention-headDim";
  if (modelVersion === "moe-v1" && metricId === "input-d") return "moe-hidden";
  if (modelVersion === "moe-v1" && metricId === "input-k") return "moe-topK";
  if (modelVersion === "collective-v1" && metricId === "input-s") return "collective-sBytes";
  if (modelVersion === "gemm-v1" && metricId.startsWith("input-t")) return direct[metricId] ?? "gemm-tileM";
  if (modelVersion === "moe-v1" && metricId === "input-p") return "hardware-rankCount";
  if (direct[metricId] && !(metricId === "input-p" && modelVersion !== "moe-v1" && modelVersion !== "collective-v1")) return direct[metricId];
  return null;
}
