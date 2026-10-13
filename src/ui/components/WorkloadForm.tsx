import type { GemmDraft } from "../state";
import type { FieldError } from "../parseInputs";
import { GEMM_FORM_GRID } from "./formStyles";

interface Props {
  draft: GemmDraft;
  errors: FieldError[];
  onChange: (field: keyof GemmDraft, value: string) => void;
  onRestoreDefaults: () => void;
}

const FIELDS: Array<{ field: keyof GemmDraft; label: string }> = [
  { field: "m", label: "M — 출력 행 수" },
  { field: "n", label: "N — 출력 열 수" },
  { field: "k", label: "K — 축소 차원" },
  { field: "tileM", label: "타일 M" },
  { field: "tileN", label: "타일 N" },
  { field: "tileK", label: "타일 K" },
];

export function WorkloadForm({ draft, errors, onChange, onRestoreDefaults }: Props) {
  const errorFor = (field: string) => errors.find((e) => e.field === field)?.message;
  return (
    <div>
      <div style={{ marginBottom: 8, fontSize: 13, color: "#444" }}>
        A[{draft.m},{draft.k}] × B[{draft.k},{draft.n}] → C[{draft.m},{draft.n}]
      </div>
      <div style={GEMM_FORM_GRID}>
        {FIELDS.map(({ field, label }) => (
          <label key={field} style={{ display: "flex", flexDirection: "column", fontSize: 12 }}>
            {label}
            <input
              id={`gemm-${field}`}
              value={draft[field]}
              onChange={(e) => onChange(field, e.target.value)}
              aria-invalid={!!errorFor(field)}
              style={{ padding: 4, border: errorFor(field) ? "1px solid #cf222e" : "1px solid #ccc" }}
            />
            {errorFor(field) && <span style={{ color: "#cf222e", fontSize: 11 }}>{errorFor(field)}</span>}
          </label>
        ))}
      </div>
      <button type="button" onClick={onRestoreDefaults} style={{ marginTop: 8 }}>
        현재 Operator 기본값 복원
      </button>
    </div>
  );
}
