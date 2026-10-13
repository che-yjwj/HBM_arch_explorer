import type { SensitivityResult } from "../../domain/types";
import { formatSeconds, formatRatio, RESOURCE_LABEL } from "../format";

export function SensitivityTable({ rows }: { rows: SensitivityResult[] }) {
  return (
    <table style={{ width: "100%", fontSize: 13 }}>
      <thead>
        <tr>
          <th style={{ textAlign: "left" }}>자원</th>
          <th>상태</th>
          <th>기준</th>
          <th>2배 변경</th>
          <th>절감</th>
          <th>Speedup</th>
          <th style={{ textAlign: "left" }}>이유</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.resourceKind}>
            <td>{RESOURCE_LABEL[r.resourceKind]}</td>
            <td>{statusLabel(r.status)}</td>
            <td>{r.baselineSeconds !== null ? formatSeconds(r.baselineSeconds) : "–"}</td>
            <td>{r.changedSeconds !== null ? formatSeconds(r.changedSeconds) : "–"}</td>
            <td>{r.reductionRatio !== null ? formatRatio(r.reductionRatio) : r.status === "no-work" ? "해당 없음" : "–"}</td>
            <td>{r.speedup !== null ? r.speedup.toFixed(3) : "–"}</td>
            <td>{r.reason ?? "모델 가정에서 계산된 결과"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function statusLabel(status: SensitivityResult["status"]): string {
  switch (status) {
    case "ok":
      return "정상";
    case "no-work":
      return "해당 없음";
    case "unsupported":
      return "모델 미지원";
    case "invalid":
      return "오류";
  }
}
