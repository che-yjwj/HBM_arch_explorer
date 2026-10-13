export function formatSeconds(s: number): string {
  if (s === 0) return "0 µs";
  if (s < 1e-3) return `${(s * 1e6).toFixed(2)} µs`;
  if (s < 1) return `${(s * 1e3).toFixed(3)} ms`;
  return `${s.toFixed(4)} s`;
}

export function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(2)} KiB`;
  if (b < 1024 * 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(2)} MiB`;
  return `${(b / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
}

export function formatFlops(f: number): string {
  if (f < 1e9) return `${f} FLOPs`;
  if (f < 1e12) return `${(f / 1e9).toFixed(2)} GFLOPs`;
  return `${(f / 1e12).toFixed(2)} TFLOPs`;
}

export function formatFlopsPerSec(f: number): string {
  if (f < 1e3) return `${Number(f.toPrecision(4))} FLOP/s`;
  if (f < 1e6) return `${(f / 1e3).toFixed(2)} kFLOP/s`;
  if (f < 1e9) return `${(f / 1e6).toFixed(2)} MFLOP/s`;
  if (f < 1e12) return `${(f / 1e9).toFixed(2)} GFLOP/s`;
  return `${(f / 1e12).toFixed(2)} TFLOP/s`;
}

export function formatBwGBs(bytesPerSec: number): string {
  if (bytesPerSec < 1024) return `${Number(bytesPerSec.toPrecision(4))} B/s`;
  if (bytesPerSec < 1024 ** 2) return `${(bytesPerSec / 1024).toFixed(2)} KiB/s`;
  if (bytesPerSec < 1024 ** 3) return `${(bytesPerSec / 1024 ** 2).toFixed(2)} MiB/s`;
  return `${(bytesPerSec / 1024 ** 3).toFixed(2)} GiB/s`;
}

export function formatRatio(r: number): string {
  return `${(r * 100).toFixed(1)}%`;
}

export const RESOURCE_LABEL: Record<string, string> = {
  hbm: "HBM",
  sram: "SRAM",
  compute: "Compute",
  link: "Link",
};
