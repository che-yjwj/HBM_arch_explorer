import { describe, expect, it } from "vitest";
import { multiplyFiniteChecked, parseByteSize, parseEfficiency, parsePositiveInt, parseRankCount, safeAdd, safeMul, SAFE_INT_MAX } from "./numeric";

describe("numeric safety edges", () => {
  it("rejects 0 efficiency (must be > 0)", () => {
    const r = parseEfficiency("0", "효율");
    expect(r.ok).toBe(false);
  });

  it("rejects negative efficiency", () => {
    const r = parseEfficiency("-0.5", "효율");
    expect(r.ok).toBe(false);
  });

  it("accepts efficiency exactly 1 (upper bound inclusive)", () => {
    const r = parseEfficiency("1", "효율");
    expect(r.ok).toBe(true);
  });

  it("rejects efficiency above 1", () => {
    const r = parseEfficiency("1.0001", "효율");
    expect(r.ok).toBe(false);
  });

  it("rejects rank count 0", () => {
    const r = parseRankCount("0");
    expect(r.ok).toBe(false);
  });

  it("accepts rank count at the upper bound 4096", () => {
    const r = parseRankCount("4096");
    expect(r.ok).toBe(true);
  });

  it("rejects rank count above the upper bound 4097", () => {
    const r = parseRankCount("4097");
    expect(r.ok).toBe(false);
  });

  it("accepts the safe integer ceiling 2^53-1", () => {
    const r = parsePositiveInt(String(SAFE_INT_MAX), "값");
    expect(r.ok).toBe(true);
  });

  it("rejects one above the safe integer ceiling", () => {
    const r = parsePositiveInt("9007199254740992", "값");
    expect(r.ok).toBe(false);
  });

  it("rejects fractional input for integer fields", () => {
    const r = parsePositiveInt("4.5", "값");
    expect(r.ok).toBe(false);
  });

  it("rejects empty input", () => {
    const r = parsePositiveInt("", "값");
    expect(r.ok).toBe(false);
  });

  it("safeMul returns null on overflow past the safe integer ceiling", () => {
    expect(safeMul(SAFE_INT_MAX, 2)).toBeNull();
  });

  it("safeAdd returns null on overflow past the safe integer ceiling", () => {
    expect(safeAdd(SAFE_INT_MAX, 1)).toBeNull();
  });

  it("safeMul computes normally within bounds", () => {
    expect(safeMul(3, 4, 5)).toBe(60);
  });

  it("checked finite multiplication rejects underflow and overflow", () => {
    expect(multiplyFiniteChecked(1e-300, 1e-300)).toBeNull();
    expect(multiplyFiniteChecked(Number.MAX_VALUE, 2)).toBeNull();
    expect(multiplyFiniteChecked(6, 0.5)).toBe(3);
  });

  it("parseByteSize rejects fractional byte results instead of rounding", () => {
    const r = parseByteSize("1.5B", "용량");
    expect(r.ok).toBe(false);
  });

  it("parseByteSize accepts exact fractional-unit results (1.5 KiB = 1536 B)", () => {
    const r = parseByteSize("1.5KiB", "용량");
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value).toBe(1536);
  });

  it("parseByteSize rejects zero bytes", () => {
    const r = parseByteSize("0B", "용량");
    expect(r.ok).toBe(false);
  });
});
