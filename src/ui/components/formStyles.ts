export const GEMM_FORM_GRID = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 300px), 1fr))",
  gap: 8,
} as const;

export const ATTENTION_FORM_GRID = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 225px), 1fr))",
  gap: 8,
} as const;

export const MOE_FORM_GRID = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 180px), 1fr))",
  gap: 8,
} as const;
