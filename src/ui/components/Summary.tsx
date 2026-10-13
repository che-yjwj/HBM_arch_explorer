import type { AnalysisResult } from "../../domain/types";
import { formatSeconds, formatRatio } from "../format";

interface Props {
  result: AnalysisResult;
  onOpenDetail: () => void;
  onOpenMetric: (id: string) => void;
}

export function Summary({ result, onOpenDetail, onOpenMetric }: Props) {
  const { core, sensitivity } = result;

  if (!core.valid) {
    return (
      <div role="alert" style={{ padding: 12, background: "#fff8e6", border: "1px solid #e0a500", borderRadius: 6 }}>
        <strong>입력 확인 필요</strong>
        <ul>
          {core.diagnostics.map((d, i) => (
            <li key={i}>
              {d.message}
              {d.fix ? ` — ${d.fix}` : ""}
            </li>
          ))}
        </ul>
      </div>
    );
  }

  const best = sensitivity
    .filter((s) => s.status === "ok" && s.savedSeconds !== null && s.savedSeconds > 0)
    .sort((a, b) => (b.savedSeconds ?? 0) - (a.savedSeconds ?? 0))[0];

  return (
    <div style={{ padding: 12, background: "#f6f8fa", borderRadius: 6 }}>
      <div>
        <strong>추정 총시간: {formatSeconds(core.totalSeconds)}</strong>
        <span style={{ marginLeft: 8, fontSize: 12, color: "#666" }}>(순차 실행 가정의 추정)</span>
      </div>
      <div style={{ marginTop: 4, fontSize: 13 }}>
        {best ? (
          <>
            개선 후보: {best.resourceKind} 2배 → {formatSeconds(best.savedSeconds ?? 0)} 절감 ({best.reductionRatio !== null ? formatRatio(best.reductionRatio) : "–"})
          </>
        ) : (
          "개선 후보 해당 없음"
        )}
      </div>
      <button type="button" onClick={onOpenDetail} style={{ marginTop: 8 }}>
        자세히 보기
      </button>
      <button type="button" onClick={() => onOpenMetric("metric-total-time")} style={{ marginTop: 8, marginLeft: 8 }}>
        총시간 산식과 입력 근거
      </button>
    </div>
  );
}
