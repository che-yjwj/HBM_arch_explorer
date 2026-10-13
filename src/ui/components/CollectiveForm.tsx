import type { CollectiveOp } from "../../domain/types";
import type { CollectiveDraft } from "../state";
import type { FieldError } from "../parseInputs";

interface Props {
  draft: CollectiveDraft;
  errors: FieldError[];
  onChangeOp: (op: CollectiveOp) => void;
  onChangeBytes: (value: string) => void;
  onRestoreDefaults: () => void;
}

const OP_LABELS: Record<CollectiveOp, string> = {
  "all-reduce": "All-Reduce",
  "all-gather": "All-Gather",
  "reduce-scatter": "Reduce-Scatter",
  "all-to-all": "All-to-All",
};

const S_MEANING: Record<CollectiveOp, string> = {
  "all-reduce": "S = 각 Rank 입력 텐서 전체 bytes",
  "all-gather": "S = 각 Rank 초기 shard bytes (전체 입력이 아님)",
  "reduce-scatter": "S = 각 Rank 전체 입력 bytes",
  "all-to-all": "S = 각 Rank의 전체 send buffer bytes (self 포함)",
};

export function CollectiveForm({ draft, errors, onChangeOp, onChangeBytes, onRestoreDefaults }: Props) {
  const errorFor = (field: string) => errors.find((e) => e.field === field)?.message;
  return (
    <div>
      <label style={{ display: "flex", flexDirection: "column", fontSize: 12, marginBottom: 8 }}>
        연산
        <select value={draft.op} onChange={(e) => onChangeOp(e.target.value as CollectiveOp)}>
          {Object.entries(OP_LABELS).map(([op, label]) => (
            <option key={op} value={op}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <div style={{ fontSize: 12, color: "#444", marginBottom: 8 }}>{S_MEANING[draft.op]}</div>
      <label style={{ display: "flex", flexDirection: "column", fontSize: 12 }}>
        S (예: 256MiB 또는 268435456 B)
        <input
          id="collective-sBytes"
          value={draft.sBytes}
          onChange={(e) => onChangeBytes(e.target.value)}
          aria-invalid={!!errorFor("sBytes")}
          style={{ padding: 4, border: errorFor("sBytes") ? "1px solid #cf222e" : "1px solid #ccc" }}
        />
        {errorFor("sBytes") && <span style={{ color: "#cf222e", fontSize: 11 }}>{errorFor("sBytes")}</span>}
      </label>
      <button type="button" onClick={onRestoreDefaults} style={{ marginTop: 8 }}>
        현재 Operator 기본값 복원
      </button>
    </div>
  );
}
