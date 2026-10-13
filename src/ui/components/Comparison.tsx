import type { BaselineSnapshot } from "../../domain/types";
import type { CompareResult } from "../../engine/compare";
import { formatBytes, formatSeconds, formatRatio } from "../format";

interface Props {
  baseline: BaselineSnapshot | null;
  compareResult: CompareResult | null;
  targetDraft: string;
  onTargetChange: (v: string) => void;
  onSetBaseline: () => void;
  onClearBaseline: () => void;
  canSetBaseline: boolean;
}

export function Comparison({ baseline, compareResult, targetDraft, onTargetChange, onSetBaseline, onClearBaseline, canSetBaseline }: Props) {
  return (
    <div>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button type="button" onClick={onSetBaseline} disabled={!canSetBaseline}>
          기준으로 고정
        </button>
        {baseline && (
          <button type="button" onClick={onClearBaseline}>
            기준 해제
          </button>
        )}
      </div>

      {baseline && compareResult && (
        <div style={{ marginTop: 12 }}>
          {!compareResult.compatible && (
            <div role="alert" style={{ color: "#cf222e" }}>
              비교 비호환: {compareResult.incompatibleReason} — 기준을 재설정하세요
            </div>
          )}
          {compareResult.compatible && (
            <table style={{ width: "100%", fontSize: 13 }}>
              <tbody>
                <tr>
                  <td>기준안 총시간</td>
                  <td>{formatSeconds(compareResult.totalSecondsBaseline)}</td>
                </tr>
                <tr>
                  <td>현재안 총시간</td>
                  <td>{compareResult.totalSecondsCurrent !== null ? formatSeconds(compareResult.totalSecondsCurrent) : "현재 입력 오류"}</td>
                </tr>
                <tr>
                  <td>절대 차이</td>
                  <td>{compareResult.absoluteDeltaSeconds !== null ? formatSeconds(Math.abs(compareResult.absoluteDeltaSeconds)) + (compareResult.absoluteDeltaSeconds <= 0 ? " 감소" : " 증가") : "–"}</td>
                </tr>
                <tr>
                  <td>상대 변화율</td>
                  <td>{compareResult.relativeDeltaRatio !== null ? formatRatio(compareResult.relativeDeltaRatio) : "해당 없음"}</td>
                </tr>
                <tr><td>HBM traffic</td><td>{formatTraffic(compareResult.hbmTrafficBaseline, compareResult.hbmTrafficCurrent)}</td></tr>
                <tr><td>Link traffic</td><td>{formatTraffic(compareResult.linkTrafficBaseline, compareResult.linkTrafficCurrent)}</td></tr>
                {(compareResult.sramRequiredBaseline !== null || compareResult.sramRequiredCurrent !== null) && (
                  <tr><td>필요 SRAM</td><td>{formatTraffic(compareResult.sramRequiredBaseline, compareResult.sramRequiredCurrent)}</td></tr>
                )}
                <tr><td>제한 자원</td><td>{compareResult.limitingResourcesBaseline.join(", ") || "없음"} → {compareResult.limitingResourcesCurrent.join(", ") || "없음"}</td></tr>
                {compareResult.changedHardwareSpecs.length > 0 && (
                  <tr><td>변경 Hardware</td><td>{compareResult.changedHardwareSpecs.join(", ")}</td></tr>
                )}
                {compareResult.workloadChanged && (
                  <tr>
                    <td colSpan={2} style={{ color: "#9a6700" }}>
                      워크로드도 변경됨: 시간 차이를 하드웨어 속도 향상으로 해석하지 않습니다
                    </td>
                  </tr>
                )}
                {compareResult.executionChanged && (
                  <tr><td colSpan={2} style={{ color: "#9a6700" }}>타일 실행 설정이 변경됨: 트래픽·SRAM 요구량 변화를 함께 확인하세요</td></tr>
                )}
              </tbody>
            </table>
          )}

          <label style={{ display: "block", marginTop: 12, fontSize: 13 }}>
            목표 시간 (선택, 초)
            <input value={targetDraft} onChange={(e) => onTargetChange(e.target.value)} style={{ padding: 4, marginLeft: 8 }} />
          </label>
          {compareResult.targetInvalid && <div role="alert" style={{ color: "#cf222e", marginTop: 4 }}>목표 시간은 0보다 큰 초 단위 숫자여야 합니다. 목표 판정만 보류합니다.</div>}
          {!compareResult.targetInvalid && compareResult.targetMet !== null && (
            <div style={{ marginTop: 4 }}>{compareResult.targetMet ? "모델상 목표 충족" : "모델상 목표 미충족"}</div>
          )}
        </div>
      )}
      {!baseline && <div style={{ fontSize: 13, color: "#666", marginTop: 8 }}>기준안이 없습니다. 현재 유효한 결과를 기준으로 고정할 수 있습니다.</div>}
    </div>
  );
}

function formatTraffic(baseline: number | null, current: number | null): string {
  return `${baseline === null ? "–" : formatBytes(baseline)} → ${current === null ? "–" : formatBytes(current)}`;
}
