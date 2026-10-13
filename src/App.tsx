import { useMemo, useReducer, useState } from "react";
import { evaluate } from "./engine/evaluate";
import { compare } from "./engine/compare";
import {
  defaultAttentionWorkload,
  defaultCollectiveWorkload,
  defaultGemmWorkload,
  defaultHardware,
  defaultMoeWorkload,
} from "./presets/defaults";
import {
  attentionDraftFromDefaults,
  collectiveDraftFromDefaults,
  gemmDraftFromDefaults,
  hardwareDraftFromDefaults,
  moeDraftFromDefaults,
  uiReducer,
} from "./ui/state";
import type { UiState } from "./ui/state";
import { parseAttentionDraft, parseCollectiveDraft, parseGemmDraft, parseHardwareDraft, parseMoeDraft } from "./ui/parseInputs";
import { WorkloadForm } from "./ui/components/WorkloadForm";
import { AttentionForm } from "./ui/components/AttentionForm";
import { MoeForm } from "./ui/components/MoeForm";
import { CollectiveForm } from "./ui/components/CollectiveForm";
import { Summary } from "./ui/components/Summary";
import { HardwareDiagram } from "./ui/components/HardwareDiagram";
import { StepList } from "./ui/components/StepList";
import { Inspector } from "./ui/components/Inspector";
import { Comparison } from "./ui/components/Comparison";
import { SensitivityTable } from "./ui/components/SensitivityTable";
import type { BaselineSnapshot, ResourceKind, Workload } from "./domain/types";

const initialState: UiState = {
  operator: "gemm",
  gemmDraft: gemmDraftFromDefaults(defaultGemmWorkload()),
  attentionDraft: attentionDraftFromDefaults(defaultAttentionWorkload()),
  moeDraft: moeDraftFromDefaults(defaultMoeWorkload()),
  collectiveDraft: collectiveDraftFromDefaults(defaultCollectiveWorkload()),
  hardwareDraft: hardwareDraftFromDefaults(defaultHardware()),
  selectedStepId: null,
  inspectorTarget: null,
  baseline: null,
  targetSecondsDraft: "",
  inputPanelOpen: true,
};

// Which resource kinds each operator's model actually uses (drives the fixed diagram's visible slots).
const RESOURCE_KINDS_BY_OPERATOR: Record<string, ResourceKind[]> = {
  gemm: ["hbm", "sram", "compute"],
  attention: ["hbm", "compute"],
  moe: ["hbm", "compute", "link"],
  collective: ["link"],
};

function App() {
  const [state, dispatch] = useReducer(uiReducer, initialState);
  const [detailOpen, setDetailOpen] = useState(false);

  const gemmParsed = useMemo(() => parseGemmDraft(state.gemmDraft), [state.gemmDraft]);
  const attentionParsed = useMemo(() => parseAttentionDraft(state.attentionDraft), [state.attentionDraft]);
  const moeParsed = useMemo(() => parseMoeDraft(state.moeDraft), [state.moeDraft]);
  const collectiveParsed = useMemo(() => parseCollectiveDraft(state.collectiveDraft), [state.collectiveDraft]);
  const hwParsed = useMemo(() => parseHardwareDraft(state.hardwareDraft), [state.hardwareDraft]);

  const currentParsed = useMemo(() => {
    switch (state.operator) {
      case "gemm":
        return gemmParsed;
      case "attention":
        return attentionParsed;
      case "moe":
        return moeParsed;
      case "collective":
        return collectiveParsed;
    }
  }, [state.operator, gemmParsed, attentionParsed, moeParsed, collectiveParsed]);

  const workload: Workload | null = currentParsed.workload;

  const analysisResult = useMemo(() => {
    if (!workload || !hwParsed.hardware) return null;
    return evaluate(workload, hwParsed.hardware);
  }, [workload, hwParsed.hardware]);

  const core = analysisResult?.core ?? null;
  const steps = core?.valid ? core.steps : [];
  const selectedStep = steps.find((s) => s.id === state.selectedStepId) ?? steps[0] ?? null;

  const compareResult = useMemo(() => {
    if (!state.baseline) return null;
    if (!workload || !hwParsed.hardware) return null;
    const targetRaw = state.targetSecondsDraft.trim();
    const targetInvalid = targetRaw !== "" && (!/^(?:\d+\.?\d*|\.\d+)$/.test(targetRaw) || !Number.isFinite(Number(targetRaw)) || Number(targetRaw) <= 0);
    const validTarget = targetRaw === "" || targetInvalid ? null : Number(targetRaw);
    return compare(state.baseline, workload, hwParsed.hardware, core?.valid ? core : null, validTarget, targetInvalid);
  }, [state.baseline, state.targetSecondsDraft, workload, hwParsed.hardware, core]);

  const canSetBaseline = !!core?.valid;

  const setBaseline = () => {
    if (!core?.valid || !workload || !hwParsed.hardware) return;
    const snapshot: BaselineSnapshot = {
      workload: structuredClone(workload),
      hardware: structuredClone(hwParsed.hardware),
      modelVersion: core.modelVersion,
      result: structuredClone(core),
    };
    dispatch({ type: "SET_BASELINE", snapshot });
  };

  const setOperator = (operator: UiState["operator"]) => dispatch({ type: "SET_OPERATOR", operator });

  const rankIgnored = state.operator === "gemm" || state.operator === "attention";

  const restoreHardwareDefaults = () => dispatch({ type: "RESTORE_HARDWARE_DEFAULTS" });

  return (
    <div style={{ width: "100%", maxWidth: 1000, margin: "0 auto", padding: 16, boxSizing: "border-box", fontFamily: "system-ui, sans-serif" }}>
      <header>
        <h1 style={{ fontSize: 20 }}>AI Workload-Driven System Architecture Explorer (P0)</h1>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <label>
            Operator:{" "}
            <select value={state.operator} onChange={(e) => setOperator(e.target.value as UiState["operator"])}>
              <option value="gemm">GEMM/GEMV</option>
              <option value="attention">Attention</option>
              <option value="moe">MoE</option>
              <option value="collective">Collective</option>
            </select>
          </label>
          <label>
            Rank 수:{" "}
            <input
              id="hardware-rankCount"
              style={{ width: 80, padding: 4 }}
              value={state.hardwareDraft.rankCount}
              onChange={(e) => dispatch({ type: "SET_HARDWARE_FIELD", field: "rankCount", value: e.target.value })}
              aria-invalid={!!hwParsed.errors.find((e) => e.field === "rankCount")}
            />
          </label>
          {rankIgnored && (
            <span style={{ fontSize: 12, color: "#9a6700" }}>단일 Rank 분석: Rank 수 미적용</span>
          )}
          <button type="button" onClick={restoreHardwareDefaults}>
            Hardware 기본값 복원
          </button>
        </div>
        {hwParsed.errors.find((e) => e.field === "rankCount") && (
          <div style={{ color: "#cf222e", fontSize: 12 }}>{hwParsed.errors.find((e) => e.field === "rankCount")!.message}</div>
        )}
      </header>

      <section style={{ marginTop: 16 }}>
        {state.operator === "gemm" && (
          <WorkloadForm
            draft={state.gemmDraft}
            errors={gemmParsed.errors}
            onChange={(field, value) => dispatch({ type: "SET_GEMM_FIELD", field, value })}
            onRestoreDefaults={() => dispatch({ type: "RESTORE_WORKLOAD_DEFAULTS" })}
          />
        )}
        {state.operator === "attention" && (
          <AttentionForm
            draft={state.attentionDraft}
            errors={attentionParsed.errors}
            onChange={(field, value) => dispatch({ type: "SET_ATTENTION_FIELD", field, value })}
            onRestoreDefaults={() => dispatch({ type: "RESTORE_WORKLOAD_DEFAULTS" })}
          />
        )}
        {state.operator === "moe" && (
          <MoeForm
            draft={state.moeDraft}
            errors={moeParsed.errors}
            onChange={(field, value) => dispatch({ type: "SET_MOE_FIELD", field, value })}
            onRestoreDefaults={() => dispatch({ type: "RESTORE_WORKLOAD_DEFAULTS" })}
          />
        )}
        {state.operator === "collective" && (
          <CollectiveForm
            draft={state.collectiveDraft}
            errors={collectiveParsed.errors}
            onChangeOp={(op) => dispatch({ type: "SET_COLLECTIVE_FIELD", field: "op", value: op })}
            onChangeBytes={(value) => dispatch({ type: "SET_COLLECTIVE_FIELD", field: "sBytes", value })}
            onRestoreDefaults={() => dispatch({ type: "RESTORE_WORKLOAD_DEFAULTS" })}
          />
        )}
      </section>

      {!workload && (
        <section style={{ marginTop: 16 }} role="alert" aria-live="polite">
          <div style={{ padding: 12, background: "#fff8e6", border: "1px solid #e0a500", borderRadius: 6 }}>
            <strong>입력 확인 필요</strong>
            <p style={{ fontSize: 13, margin: "4px 0" }}>아래 필드를 수정하면 결과가 다시 계산됩니다. 이전 결과는 표시하지 않습니다.</p>
            <ul style={{ fontSize: 13, margin: 0 }}>
              {currentParsed.errors.map((e, i) => (
                <li key={i}>{e.message}</li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {workload && !hwParsed.hardware && (
        <section style={{ marginTop: 16 }} role="alert" aria-live="polite">
          <div style={{ padding: 12, background: "#fff8e6", border: "1px solid #e0a500", borderRadius: 6 }}>
            <strong>Hardware 입력 확인 필요</strong>
            <p style={{ fontSize: 13, margin: "4px 0" }}>Hardware 설정을 수정하면 결과가 다시 계산됩니다. 이전 결과는 표시하지 않습니다.</p>
            <ul style={{ fontSize: 13, margin: 0 }}>
              {hwParsed.errors.map((e, i) => (
                <li key={i}>{e.message}</li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {analysisResult && (
        <section style={{ marginTop: 16 }}>
          <Summary result={analysisResult} onOpenDetail={() => setDetailOpen(true)} onOpenMetric={(id) => dispatch({ type: "SET_INSPECTOR", target: { type: "metric", id } })} />
        </section>
      )}

      <section style={{ marginTop: 24 }}>
        <HardwareDiagram
          resourceKinds={RESOURCE_KINDS_BY_OPERATOR[state.operator]}
          structureLabel={rankIgnored
            ? "고정 하드웨어 구조 · 단일 Rank 모델"
            : `고정 하드웨어 구조 · Rank 수 P=${state.hardwareDraft.rankCount} · 논리 Link`}
          selectedStep={selectedStep}
          transfers={core?.valid ? core.transfers : []}
          tensors={core?.valid ? core.tensors : []}
          metrics={core?.valid ? core.metrics : []}
          inspectorResourceId={state.inspectorTarget?.type === "resource" ? state.inspectorTarget.id : null}
          onSelectResource={(id) => dispatch({ type: "SET_INSPECTOR", target: { type: "resource", id } })}
          onSelectTransfer={(id) => dispatch({ type: "SET_INSPECTOR", target: { type: "transfer", id } })}
        />
      </section>

      {core?.valid && (
        <section style={{ marginTop: 16 }}>
          <StepList
            steps={core.steps}
            metrics={core.metrics}
            selectedStepId={selectedStep?.id ?? null}
            onSelect={(id) => dispatch({ type: "SELECT_STEP", stepId: id })}
          />
        </section>
      )}

      <section style={{ marginTop: 16, border: "1px solid #ddd", borderRadius: 6, padding: 12, minHeight: 80 }}>
        {core && (
          <Inspector
            target={state.inspectorTarget}
            result={core}
            hardwareDraft={state.hardwareDraft}
            onHardwareFieldChange={(field, value) => dispatch({ type: "SET_HARDWARE_FIELD", field, value })}
            onSelectMetric={(metricId) => dispatch({ type: "SET_INSPECTOR", target: { type: "metric", id: metricId } })}
            onSelectStep={(stepId) => dispatch({ type: "SELECT_STEP", stepId })}
            onSelectResource={(resourceId) => dispatch({ type: "SET_INSPECTOR", target: { type: "resource", id: resourceId } })}
            onEditInput={(fieldId) => {
              const resourceByField: Record<string, string> = {
                "hardware-computeFlops": "res-compute",
                "hardware-computeEfficiency": "res-compute",
                "hardware-sramCapacityBytes": "res-sram",
                "hardware-sramBwBytes": "res-sram",
                "hardware-sramEfficiency": "res-sram",
                "hardware-hbmBwBytes": "res-hbm",
                "hardware-hbmEfficiency": "res-hbm",
                "hardware-linkBwBytes": "res-link",
                "hardware-linkEfficiency": "res-link",
                "hardware-linkLatencySeconds": "res-link",
              };
              const resourceId = resourceByField[fieldId];
              if (resourceId) {
                dispatch({ type: "SET_INSPECTOR", target: { type: "resource", id: resourceId } });
              }
              requestAnimationFrame(() => {
                const field = document.getElementById(fieldId) as HTMLInputElement | null;
                field?.scrollIntoView({ behavior: "smooth", block: "center" });
                field?.focus();
              });
            }}
            onClose={() => dispatch({ type: "SET_INSPECTOR", target: null })}
          />
        )}
      </section>

      {core?.valid && selectedStep && (
        <section style={{ marginTop: 8, fontSize: 13 }}>
          총시간 ›{" "}
          <button type="button" onClick={() => dispatch({ type: "SET_INSPECTOR", target: { type: "metric", id: selectedStep.timeMetricId } })}>
            {selectedStep.label} 시간 근거 보기
          </button>
        </section>
      )}

      {detailOpen && analysisResult && (
        <section style={{ marginTop: 24, borderTop: "1px solid #ddd", paddingTop: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <h2 style={{ fontSize: 16 }}>민감도 (자원 2배)와 비교</h2>
            <button type="button" onClick={() => setDetailOpen(false)}>
              단계로 돌아가기
            </button>
          </div>
          <SensitivityTable rows={analysisResult.sensitivity} />
          <div style={{ marginTop: 16 }}>
            <Comparison
              baseline={state.baseline}
              compareResult={compareResult}
              targetDraft={state.targetSecondsDraft}
              onTargetChange={(v) => dispatch({ type: "SET_TARGET", value: v })}
              onSetBaseline={setBaseline}
              onClearBaseline={() => dispatch({ type: "CLEAR_BASELINE" })}
              canSetBaseline={canSetBaseline}
            />
          </div>
        </section>
      )}

      <section style={{ marginTop: 16, fontSize: 12, color: "#777" }}>
        Hardware: Rank 수 {state.hardwareDraft.rankCount}
        {state.operator === "gemm" || state.operator === "attention" ? " (단일 Rank 분석: Rank 수 미적용)" : ""}
      </section>
    </div>
  );
}

export default App;
