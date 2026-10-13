import type { MoeDraft } from "../state";
import type { FieldError } from "../parseInputs";
import { MOE_FORM_GRID } from "./formStyles";

interface Props {
  draft: MoeDraft;
  errors: FieldError[];
  onChange: (field: keyof MoeDraft, value: string) => void;
  onRestoreDefaults: () => void;
}

const FIELDS: Array<{ field: keyof MoeDraft; label: string }> = [
  { field: "tokens", label: "Tokens (T)" },
  { field: "hidden", label: "Hidden (D)" },
  { field: "ffn", label: "FFN (F)" },
  { field: "experts", label: "Experts (E)" },
  { field: "topK", label: "Top-k" },
];

export function MoeForm({ draft, errors, onChange, onRestoreDefaults }: Props) {
  const errorFor = (field: string) => errors.find((e) => e.field === field)?.message;
  return (
    <div>
      <div style={{ marginBottom: 8, fontSize: 13, color: "#444" }}>
        Router → Dispatch → Expert GEMM (D→F→D) → Combine, E개 Expert를 Rank에 균등 배치 (E mod P = 0 필요)
      </div>
      <div style={MOE_FORM_GRID}>
        {FIELDS.map(({ field, label }) => (
          <label key={field} style={{ display: "flex", flexDirection: "column", fontSize: 12 }}>
            {label}
            <input
              id={`moe-${field}`}
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
