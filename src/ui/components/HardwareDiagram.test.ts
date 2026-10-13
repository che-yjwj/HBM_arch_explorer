import { describe, expect, it } from "vitest";
import { layoutFor } from "./HardwareDiagram";
import type { ResourceKind } from "../../domain/types";

const BOX_WIDTH = 140;

describe("HardwareDiagram layoutFor — no overflow regardless of resource-kind count", () => {
  const cases: ResourceKind[][] = [
    ["link"], // Collective: single-kind case that previously overflowed (X_POS.link=610 vs width=160)
    ["hbm", "compute"], // Attention
    ["hbm", "compute", "link"], // MoE
    ["hbm", "sram", "compute"], // GEMM
  ];

  for (const kinds of cases) {
    it(`fits every box within the container width for [${kinds.join(",")}]`, () => {
      const { present, xFor, width } = layoutFor(kinds);
      expect(present.length).toBe(kinds.length);
      for (const kind of present) {
        const x = xFor(kind);
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x + BOX_WIDTH).toBeLessThanOrEqual(width);
      }
    });
  }

  it("single-kind layout is compact, not stretched to the 4-slot width", () => {
    const single = layoutFor(["link"]);
    const quad = layoutFor(["hbm", "sram", "compute", "link"]);
    expect(single.width).toBeLessThan(quad.width);
  });
});
