import type { AttentionWorkload, CollectiveWorkload, GemmWorkload, Hardware, MoeWorkload } from "../domain/types";
import { parseByteSize, parseEfficiency, parseNonNegativeFinite, parsePositiveFinite, parsePositiveInt, parseRankCount } from "../engine/numeric";
import type { AttentionDraft, CollectiveDraft, GemmDraft, HardwareDraft, MoeDraft } from "./state";

export interface FieldError {
  field: string;
  message: string;
}

export function parseGemmDraft(draft: GemmDraft): { workload: GemmWorkload | null; errors: FieldError[] } {
  const errors: FieldError[] = [];
  const m = parsePositiveInt(draft.m, "M");
  const n = parsePositiveInt(draft.n, "N");
  const k = parsePositiveInt(draft.k, "K");
  const tileM = parsePositiveInt(draft.tileM, "타일 M");
  const tileN = parsePositiveInt(draft.tileN, "타일 N");
  const tileK = parsePositiveInt(draft.tileK, "타일 K");
  for (const [field, p] of [
    ["m", m],
    ["n", n],
    ["k", k],
    ["tileM", tileM],
    ["tileN", tileN],
    ["tileK", tileK],
  ] as const) {
    if (!p.ok) errors.push({ field, message: p.error });
  }
  if (errors.length > 0) return { workload: null, errors };
  return {
    workload: {
      kind: "gemm",
      precision: "bf16",
      m: (m as any).value,
      n: (n as any).value,
      k: (k as any).value,
      tileM: (tileM as any).value,
      tileN: (tileN as any).value,
      tileK: (tileK as any).value,
    },
    errors: [],
  };
}

export function parseAttentionDraft(draft: AttentionDraft): { workload: AttentionWorkload | null; errors: FieldError[] } {
  const errors: FieldError[] = [];
  const batch = parsePositiveInt(draft.batch, "Batch");
  const heads = parsePositiveInt(draft.heads, "Heads");
  const seq = parsePositiveInt(draft.seq, "Sequence");
  const headDim = parsePositiveInt(draft.headDim, "Head dimension");
  for (const [field, p] of [
    ["batch", batch],
    ["heads", heads],
    ["seq", seq],
    ["headDim", headDim],
  ] as const) {
    if (!p.ok) errors.push({ field, message: p.error });
  }
  if (errors.length > 0) return { workload: null, errors };
  return {
    workload: {
      kind: "attention",
      precision: "bf16",
      batch: (batch as any).value,
      heads: (heads as any).value,
      seq: (seq as any).value,
      headDim: (headDim as any).value,
    },
    errors: [],
  };
}

export function parseMoeDraft(draft: MoeDraft): { workload: MoeWorkload | null; errors: FieldError[] } {
  const errors: FieldError[] = [];
  const tokens = parsePositiveInt(draft.tokens, "Tokens");
  const hidden = parsePositiveInt(draft.hidden, "Hidden");
  const ffn = parsePositiveInt(draft.ffn, "FFN");
  const experts = parsePositiveInt(draft.experts, "Experts");
  const topK = parsePositiveInt(draft.topK, "Top-k");
  for (const [field, p] of [
    ["tokens", tokens],
    ["hidden", hidden],
    ["ffn", ffn],
    ["experts", experts],
    ["topK", topK],
  ] as const) {
    if (!p.ok) errors.push({ field, message: p.error });
  }
  if (errors.length > 0) return { workload: null, errors };
  return {
    workload: {
      kind: "moe",
      precision: "bf16",
      tokens: (tokens as any).value,
      hidden: (hidden as any).value,
      ffn: (ffn as any).value,
      experts: (experts as any).value,
      topK: (topK as any).value,
    },
    errors: [],
  };
}

export function parseCollectiveDraft(draft: CollectiveDraft): { workload: CollectiveWorkload | null; errors: FieldError[] } {
  const errors: FieldError[] = [];
  const sBytes = parseByteSize(draft.sBytes, "S");
  if (!sBytes.ok) errors.push({ field: "sBytes", message: sBytes.error });
  if (errors.length > 0) return { workload: null, errors };
  return {
    workload: { kind: "collective", precision: "bf16", op: draft.op, sBytes: (sBytes as any).value },
    errors: [],
  };
}

export function parseHardwareDraft(draft: HardwareDraft): { hardware: Hardware | null; errors: FieldError[] } {
  const errors: FieldError[] = [];
  const rankCount = parseRankCount(draft.rankCount);
  const computeFlops = parsePositiveFinite(draft.computeFlops, "Compute 성능");
  const computeEfficiency = parseEfficiency(draft.computeEfficiency, "Compute 효율");
  const sramCapacityBytes = parsePositiveInt(draft.sramCapacityBytes, "SRAM 용량");
  const sramBwBytes = parsePositiveFinite(draft.sramBwBytes, "SRAM BW");
  const sramEfficiency = parseEfficiency(draft.sramEfficiency, "SRAM 효율");
  const hbmBwBytes = parsePositiveFinite(draft.hbmBwBytes, "HBM BW");
  const hbmEfficiency = parseEfficiency(draft.hbmEfficiency, "HBM 효율");
  const linkBwBytes = parsePositiveFinite(draft.linkBwBytes, "Link BW");
  const linkEfficiency = parseEfficiency(draft.linkEfficiency, "Link 효율");
  const linkLatencySeconds = parseNonNegativeFinite(draft.linkLatencySeconds, "Link latency");

  const entries = [
    ["rankCount", rankCount],
    ["computeFlops", computeFlops],
    ["computeEfficiency", computeEfficiency],
    ["sramCapacityBytes", sramCapacityBytes],
    ["sramBwBytes", sramBwBytes],
    ["sramEfficiency", sramEfficiency],
    ["hbmBwBytes", hbmBwBytes],
    ["hbmEfficiency", hbmEfficiency],
    ["linkBwBytes", linkBwBytes],
    ["linkEfficiency", linkEfficiency],
    ["linkLatencySeconds", linkLatencySeconds],
  ] as const;
  for (const [field, p] of entries) {
    if (!p.ok) errors.push({ field, message: p.error });
  }
  if (errors.length > 0) return { hardware: null, errors };

  return {
    hardware: {
      id: "hw-current",
      name: "가상 Hardware",
      rankCount: (rankCount as any).value,
      computeFlops: (computeFlops as any).value,
      computeEfficiency: (computeEfficiency as any).value,
      sramCapacityBytes: (sramCapacityBytes as any).value,
      sramBwBytes: (sramBwBytes as any).value,
      sramEfficiency: (sramEfficiency as any).value,
      hbmBwBytes: (hbmBwBytes as any).value,
      hbmEfficiency: (hbmEfficiency as any).value,
      linkBwBytes: (linkBwBytes as any).value,
      linkEfficiency: (linkEfficiency as any).value,
      linkLatencySeconds: (linkLatencySeconds as any).value,
    },
    errors: [],
  };
}
