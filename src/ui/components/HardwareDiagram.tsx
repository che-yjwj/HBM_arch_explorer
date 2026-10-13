import type { Metric, ResourceKind, Step, Tensor, Transfer } from "../../domain/types";
import { formatBytes } from "../format";

interface Props {
  resourceKinds: ResourceKind[]; // which of HBM/SRAM/Compute/Link this operator's model uses
  structureLabel: string;
  selectedStep: Step | null;
  transfers: Transfer[];
  tensors: Tensor[];
  metrics: Metric[];
  inspectorResourceId: string | null;
  onSelectResource: (id: string) => void;
  onSelectTransfer: (id: string) => void;
}

const ORDER: ResourceKind[] = ["hbm", "sram", "compute", "link"];
const LABELS: Record<ResourceKind, string> = { hbm: "HBM", sram: "SRAM", compute: "Compute", link: "Link" };
const RES_ID: Record<ResourceKind, string> = { hbm: "res-hbm", sram: "res-sram", compute: "res-compute", link: "res-link" };

const BOX_WIDTH = 140;
const SLOT_WIDTH = 200;
const MARGIN = 10;

/** Pure layout calculation, exported for regression testing (no overflow for any present-kind count 1..4). */
export function layoutFor(resourceKinds: ResourceKind[]) {
  const present = ORDER.filter((k) => resourceKinds.includes(k));
  const xFor = (kind: ResourceKind) => MARGIN + present.indexOf(kind) * SLOT_WIDTH;
  const width = MARGIN * 2 + Math.max(present.length, 1) * SLOT_WIDTH - (SLOT_WIDTH - BOX_WIDTH);
  return { present, xFor, width, boxWidth: BOX_WIDTH };
}

/**
 * Fixed schematic: slot order/spacing depends only on the operator's fixed resourceKinds set
 * (structural, same every render for a given operator), never on input magnitudes or byte values.
 * Each operator always gets the same layout: slot i sits at MARGIN + i*SLOT_WIDTH, in ORDER.
 */
export function HardwareDiagram({ resourceKinds, structureLabel, selectedStep, transfers, tensors, metrics, inspectorResourceId, onSelectResource, onSelectTransfer }: Props) {
  const related = new Set(selectedStep?.resourceIds ?? []);
  const { present, xFor, width } = layoutFor(resourceKinds);
  const activeTransfers = transfers.filter((transfer) => transfer.stepId === selectedStep?.id);
  const transferLabel = (transfer: Transfer) => {
    const tensor = tensors.find((item) => item.id === transfer.tensorId);
    const metric = metrics.find((item) => item.id === transfer.bytesMetricId);
    return `${tensor?.label ?? transfer.tensorId} · ${formatBytes(metric?.value ?? 0)}`;
  };

  const boxStyle = (resourceId: string, x: number): React.CSSProperties => ({
    position: "absolute",
    left: x,
    top: 70,
    width: BOX_WIDTH,
    height: 90,
    border: related.has(resourceId) ? "3px solid #1a7f37" : "2px solid #555",
    outline: inspectorResourceId === resourceId ? "2px solid #0969da" : "none",
    borderRadius: 8,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#fff",
    cursor: "pointer",
    fontWeight: 600,
  });

  return (
    <div style={{ overflowX: "auto", width: "100%" }}>
      <div style={{ marginBottom: 6, fontSize: 13, color: "#444" }}>{structureLabel}</div>
      <div
        role="group"
        aria-label={structureLabel}
        style={{ position: "relative", height: 240, width, minWidth: width, margin: "0 auto" }}
      >
        {present.map((kind) => (
          <button
            key={kind}
            type="button"
            style={boxStyle(RES_ID[kind], xFor(kind))}
            onClick={() => onSelectResource(RES_ID[kind])}
            aria-pressed={inspectorResourceId === RES_ID[kind]}
          >
            {LABELS[kind]}
          </button>
        ))}
        <svg width={width} height={240} style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}>
          <defs>
            <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill="#999" />
            </marker>
          </defs>
          {activeTransfers.map((transfer, i) => {
            const fromKind = ORDER.find((kind) => RES_ID[kind] === transfer.from);
            const toKind = ORDER.find((kind) => RES_ID[kind] === transfer.to);
            if (!fromKind || !toKind) return null;
            const fromX = xFor(fromKind);
            const toX = xFor(toKind);
            const y = i % 2 === 0 ? 66 : 166;
            const path = fromKind === toKind
              ? `M ${fromX + 45} 70 C ${fromX + 45} 25, ${fromX + 95} 25, ${fromX + 95} 70`
              : `M ${fromX + BOX_WIDTH / 2} 70 C ${(fromX + toX + BOX_WIDTH) / 2} ${y}, ${(fromX + toX + BOX_WIDTH) / 2} ${y}, ${toX + BOX_WIDTH / 2} 70`;
            const labelX = fromKind === toKind ? fromX + 70 : (fromX + toX + BOX_WIDTH) / 2;
            return (
              <g key={transfer.id}>
                <path d={path} fill="none" stroke="#0969da" strokeWidth={2.5} markerEnd="url(#arrow)" />
                <text x={labelX} y={y === 66 ? 48 : 198} textAnchor="middle" fontSize={11} fill="#084298">{transferLabel(transfer)}</text>
              </g>
            );
          })}
        </svg>
        {selectedStep && (
          <div style={{ position: "absolute", top: 180, left: 0, width: "100%", textAlign: "center", fontSize: 13, color: "#444" }}>
            {selectedStep.description}
          </div>
        )}
      </div>
      {selectedStep && activeTransfers.length > 0 && (
        <div aria-label="선택 단계 데이터 경로" style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 6, marginTop: 8 }}>
          {activeTransfers.map((transfer) => (
            <button key={transfer.id} type="button" onClick={() => onSelectTransfer(transfer.id)}>
              {transfer.trafficClass === "link" && transfer.from === transfer.to
                ? transfer.id === "transfer-combine" ? "peer Ranks → Rank 0 (논리 Link)" : "Rank 0 ↔ peer Ranks (논리 Link, 대칭 송수신)"
                : `${resourceLabel(transfer.from)} → ${resourceLabel(transfer.to)}`}: {transferLabel(transfer)}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function resourceLabel(id: string): string {
  const kind = ORDER.find((item) => RES_ID[item] === id);
  return kind ? LABELS[kind] : id;
}
