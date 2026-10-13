// Safe numeric parsing, unit conversion, and overflow-checked integer arithmetic.
// (03-data-and-engine.md: 2^53-1 safe integer ceiling; byte strings parsed via decimal digits, not float rounding)

export const SAFE_INT_MAX = Number.MAX_SAFE_INTEGER; // 2^53 - 1
export const MAX_RANK_COUNT = 4096;

export type Parsed<T> = { ok: true; value: T } | { ok: false; error: string };

/** Parses a decimal string as a positive safe integer. Rejects floats, signs other than +, exponents, empty input. */
export function parsePositiveInt(raw: string, label: string): Parsed<number> {
  const trimmed = raw.trim();
  if (trimmed === "") return { ok: false, error: `${label}: 입력이 비었습니다` };
  if (!/^[0-9]+$/.test(trimmed)) {
    return { ok: false, error: `${label}: 양의 정수만 허용합니다 (입력: "${raw}")` };
  }
  // Compare as string first to avoid precision loss before Number() rounds.
  const normalized = trimmed.replace(/^0+(?=\d)/, "");
  if (normalized.length > String(SAFE_INT_MAX).length) {
    return { ok: false, error: `${label}: 안전 정수 상한(${SAFE_INT_MAX})을 초과했습니다` };
  }
  const value = Number(normalized);
  if (!Number.isSafeInteger(value) || value <= 0) {
    return { ok: false, error: `${label}: 안전 정수 상한(${SAFE_INT_MAX})을 초과했거나 0 이하입니다` };
  }
  return { ok: true, value };
}

/** Parses a positive finite number (for BW, performance, efficiency inputs that may be decimal). */
export function parsePositiveFinite(raw: string, label: string): Parsed<number> {
  const trimmed = raw.trim();
  if (trimmed === "") return { ok: false, error: `${label}: 입력이 비었습니다` };
  if (!/^[0-9]*\.?[0-9]+$/.test(trimmed)) {
    return { ok: false, error: `${label}: 양의 유한수만 허용합니다 (입력: "${raw}")` };
  }
  const value = Number(trimmed);
  if (!Number.isFinite(value) || value <= 0) {
    return { ok: false, error: `${label}: 양의 유한수만 허용합니다` };
  }
  return { ok: true, value };
}

export function parseNonNegativeFinite(raw: string, label: string): Parsed<number> {
  const trimmed = raw.trim();
  if (trimmed === "") return { ok: false, error: `${label}: 입력이 비었습니다` };
  if (!/^[0-9]*\.?[0-9]+$/.test(trimmed)) {
    return { ok: false, error: `${label}: 0 이상의 유한수만 허용합니다 (입력: "${raw}")` };
  }
  const value = Number(trimmed);
  if (!Number.isFinite(value) || value < 0) {
    return { ok: false, error: `${label}: 0 이상의 유한수만 허용합니다` };
  }
  return { ok: true, value };
}

/** Efficiency must be in (0, 1]. */
export function parseEfficiency(raw: string, label: string): Parsed<number> {
  const parsed = parsePositiveFinite(raw, label);
  if (!parsed.ok) return parsed;
  if (parsed.value > 1) return { ok: false, error: `${label}: 효율은 0 초과 1 이하여야 합니다` };
  return parsed;
}

export function parseRankCount(raw: string): Parsed<number> {
  const parsed = parsePositiveInt(raw, "Rank 수");
  if (!parsed.ok) return parsed;
  if (parsed.value > MAX_RANK_COUNT) {
    return { ok: false, error: `Rank 수: 지원 범위(1~${MAX_RANK_COUNT})를 초과했습니다` };
  }
  return parsed;
}

/** Checked multiplication over safe integers; returns null on overflow. */
export function safeMul(...values: number[]): number | null {
  let acc = 1;
  for (const v of values) {
    if (!Number.isFinite(v)) return null;
    const next = acc * v;
    if (!Number.isSafeInteger(next)) return null;
    acc = next;
  }
  return acc;
}

export function safeAdd(...values: number[]): number | null {
  let acc = 0;
  for (const v of values) {
    if (!Number.isFinite(v)) return null;
    const next = acc + v;
    if (!Number.isSafeInteger(next)) return null;
    acc = next;
  }
  return acc;
}

/** Adds finite real values (for seconds and rates) and rejects overflow. */
export function addFinite(...values: number[]): number | null {
  let acc = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) return null;
    acc += value;
    if (!Number.isFinite(acc)) return null;
  }
  return acc;
}

/** Multiplies finite real values and rejects overflow or a positive-product underflow to zero. */
export function multiplyFiniteChecked(left: number, right: number): number | null {
  if (!Number.isFinite(left) || !Number.isFinite(right)) return null;
  const result = left * right;
  if (!Number.isFinite(result) || (left > 0 && right > 0 && result <= 0)) return null;
  return result;
}

export function isFiniteNumber(v: number): boolean {
  return typeof v === "number" && Number.isFinite(v);
}

/**
 * Divides and rejects both overflow (non-finite) and underflow (a positive numerator producing
 * a non-positive result due to float rounding, e.g. a tiny byte count over a huge BW). Division by
 * zero is the caller's responsibility to prevent (BW/efficiency are validated > 0 beforehand).
 */
export function divideChecked(numerator: number, denominator: number): number | null {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) return null;
  const result = numerator / denominator;
  if (!Number.isFinite(result)) return null;
  if (numerator > 0 && result <= 0) return null; // underflowed to zero or negative
  return result;
}

/**
 * Parses a decimal byte-size string with a unit suffix into an exact integer byte count.
 * Supports: B, KB(10^3), MB(10^6), GB(10^9), KiB(2^10), MiB(2^20), GiB(2^30).
 * Rejects fractional byte results (e.g. "1.5 B") instead of rounding.
 */
export function parseByteSize(raw: string, label: string): Parsed<number> {
  const trimmed = raw.trim();
  const match = /^([0-9]+)(?:\.([0-9]+))?\s*(B|KB|MB|GB|KiB|MiB|GiB)$/.exec(trimmed);
  if (!match) {
    return {
      ok: false,
      error: `${label}: "숫자 단위" 형식이어야 합니다 (예: 256MiB). 입력: "${raw}"`,
    };
  }
  const [, intPart, fracPart, unit] = match;
  const scale: Record<string, bigint> = {
    B: 1n,
    KB: 1000n,
    MB: 1000000n,
    GB: 1000000000n,
    KiB: 1024n,
    MiB: 1048576n,
    GiB: 1073741824n,
  };
  const unitScale = scale[unit];
  const frac = fracPart ?? "";
  // value = (intPart.frac) * unitScale, computed in integer domain with frac as a fractional digit string.
  const scaledInt = BigInt(intPart) * unitScale;
  if (frac.length === 0) {
    return finishByteParse(scaledInt, label, raw);
  }
  // fractional part contributes frac * unitScale / 10^frac.length; must be an integer.
  const fracNumerator = BigInt(frac) * unitScale;
  const fracDenominator = 10n ** BigInt(frac.length);
  if (fracNumerator % fracDenominator !== 0n) {
    return { ok: false, error: `${label}: 소수 byte는 허용하지 않습니다 (입력: "${raw}")` };
  }
  const total = scaledInt + fracNumerator / fracDenominator;
  return finishByteParse(total, label, raw);
}

function finishByteParse(total: bigint, label: string, raw: string): Parsed<number> {
  if (total <= 0n) return { ok: false, error: `${label}: 0보다 큰 byte 값이 필요합니다` };
  if (total > BigInt(SAFE_INT_MAX)) {
    return { ok: false, error: `${label}: 안전 정수 상한을 초과했습니다 (입력: "${raw}")` };
  }
  return { ok: true, value: Number(total) };
}

export function formatBytesExact(bytes: number): string {
  return `${bytes} B`;
}
