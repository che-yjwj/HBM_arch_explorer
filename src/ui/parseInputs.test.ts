import { describe, expect, it } from "vitest";
import { defaultCollectiveWorkload } from "../presets/defaults";
import { collectiveDraftFromDefaults } from "./state";
import { parseCollectiveDraft } from "./parseInputs";

describe("Collective byte input", () => {
  it("parses unit suffixed bytes exactly and keeps defaults parseable", () => {
    expect(parseCollectiveDraft({ op: "all-reduce", sBytes: "1.5MiB" }).workload?.sBytes).toBe(1_572_864);
    const draft = collectiveDraftFromDefaults(defaultCollectiveWorkload());
    expect(parseCollectiveDraft(draft).errors).toEqual([]);
  });

  it("rejects fractional byte values", () => {
    expect(parseCollectiveDraft({ op: "all-reduce", sBytes: "1.5B" }).workload).toBeNull();
  });
});
