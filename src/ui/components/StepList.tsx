import type { Metric, Step } from "../../domain/types";
import { formatSeconds } from "../format";

interface Props {
  steps: Step[];
  metrics: Metric[];
  selectedStepId: string | null;
  onSelect: (id: string) => void;
}

export function StepList({ steps, metrics, selectedStepId, onSelect }: Props) {
  const idx = steps.findIndex((s) => s.id === selectedStepId);

  const goPrev = () => idx > 0 && onSelect(steps[idx - 1].id);
  const goNext = () => idx >= 0 && idx < steps.length - 1 && onSelect(steps[idx + 1].id);

  return (
    <div aria-label="분석 단계 목록">
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button type="button" onClick={goPrev} disabled={idx <= 0}>
          ← 이전
        </button>
        <button type="button" onClick={goNext} disabled={idx < 0 || idx >= steps.length - 1}>
          다음 →
        </button>
      </div>
      <ol style={{ listStyle: "none", padding: 0, display: "flex", flexDirection: "column", gap: 6 }}>
        {steps.map((step) => {
          const metric = metrics.find((m) => m.id === step.timeMetricId);
          const selected = step.id === selectedStepId;
          return (
            <li key={step.id}>
              <button
                type="button"
                onClick={() => onSelect(step.id)}
                aria-current={selected}
                style={{
                  width: "100%",
                  textAlign: "left",
                  padding: "8px 12px",
                  border: selected ? "2px solid #0969da" : "1px solid #ccc",
                  borderRadius: 6,
                  background: selected ? "#eef6ff" : "#fff",
                  cursor: "pointer",
                }}
              >
                <strong>{step.label}</strong> — {metric ? formatSeconds(metric.value) : "–"}
                {step.limitingResource && (
                  <span style={{ marginLeft: 8, fontSize: 12, color: "#555" }}>이 단계 제한: {step.limitingResource}</span>
                )}
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
