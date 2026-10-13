import type { AttentionDraft } from "../state";
import type { FieldError } from "../parseInputs";
import { ATTENTION_FORM_GRID } from "./formStyles";

interface Props {
  draft: AttentionDraft;
  errors: FieldError[];
  onChange: (field: keyof AttentionDraft, value: string) => void;
  onRestoreDefaults: () => void;
}

const FIELDS: Array<{ field: keyof AttentionDraft; label: string }> = [
  { field: "batch", label: "Batch" },
  { field: "heads", label: "Heads" },
  { field: "seq", label: "Sequence" },
  { field: "headDim", label: "Head dimension" },
];

export function AttentionForm({ draft, errors, onChange, onRestoreDefaults }: Props) {
  const errorFor = (field: string) => errors.find((e) => e.field === field)?.message;
  return (
    <div>
      <div style={{ marginBottom: 8, fontSize: 13, color: "#444" }}>
        Q/K/V shape = [{draft.batch},{draft.heads},{draft.seq},{draft.headDim}] (dense non-causal prefill)
      </div>
      <div style={ATTENTION_FORM_GRID}>
        {FIELDS.map(({ field, label }) => (
          <label key={field} style={{ display: "flex", flexDirection: "column", fontSize: 12 }}>
            {label}
            <input
              id={`attention-${field}`}
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
