import { describe, expect, it } from "vitest";
import { GEMM_FORM_GRID, ATTENTION_FORM_GRID, MOE_FORM_GRID } from "./formStyles";

describe("Form grid style constants — responsive breakpoints preserve desktop column counts", () => {
  it("GEMM grid uses responsive auto-fit with 300px min (3 cols at 968px, 1 col at 390px)", () => {
    expect(GEMM_FORM_GRID.display).toBe("grid");
    expect(GEMM_FORM_GRID.gridTemplateColumns).toContain("auto-fit");
    expect(GEMM_FORM_GRID.gridTemplateColumns).toContain("300px");
    expect(GEMM_FORM_GRID.gridTemplateColumns).toContain("min(100%");
    expect(GEMM_FORM_GRID.gap).toBe(8);
  });

  it("Attention grid uses responsive auto-fit with 225px min (4 cols at 968px, 1 col at 390px)", () => {
    expect(ATTENTION_FORM_GRID.display).toBe("grid");
    expect(ATTENTION_FORM_GRID.gridTemplateColumns).toContain("auto-fit");
    expect(ATTENTION_FORM_GRID.gridTemplateColumns).toContain("225px");
    expect(ATTENTION_FORM_GRID.gridTemplateColumns).toContain("min(100%");
    expect(ATTENTION_FORM_GRID.gap).toBe(8);
  });

  it("MoE grid uses responsive auto-fit with 180px min (5 cols at 968px, 1 col at 390px)", () => {
    expect(MOE_FORM_GRID.display).toBe("grid");
    expect(MOE_FORM_GRID.gridTemplateColumns).toContain("auto-fit");
    expect(MOE_FORM_GRID.gridTemplateColumns).toContain("180px");
    expect(MOE_FORM_GRID.gridTemplateColumns).toContain("min(100%");
    expect(MOE_FORM_GRID.gap).toBe(8);
  });
});
