# 분석 모델·계산식·가정

P0 설계 초안 · 2026-10-11

## 공통 규칙

이 문서의 모델은 구현 출발점이다. 실제 장치의 측정값으로 해석하지 않는다. 계산은 SI 단위 FLOP·byte·second로 수행하고 화면에서 TFLOP/s·GB/s·µs·MiB로 변환한다. 1 GB = 10⁹ byte, 1 MiB = 2²⁰ byte, FMA = 2 FLOPs다. 통신량은 기본적으로 Rank당 송신량이며 수신량과 전체 집계량은 별도 표기한다.

입력·정수 중간값의 안전 범위와 Tensor/Transfer 집계 규칙은 [03 데이터 구조·엔진](03-data-and-engine.md)을 따른다. 수학적으로 정의된 식이라도 구현의 지원 범위를 넘는 입력은 오류로 처리한다.

BF16 입력·최종 출력 크기를 q = 2 byte, FP32 누산·Attention 중간값 크기를 a = 4 byte로 둔다. 정밀도는 입력과 누산을 구분한다. P0 UI는 BF16/FP32 조합 하나로 시작할 수 있다.

유효 성능 = 명목 성능 × 해당 자원의 효율이다. 모든 효율은 0 초과 1 이하이며 기본값은 0.7이다. 유효 Compute 성능은 C_eff로 표기하고, P는 Rank 수에만 사용한다. P0의 Compute 입력은 BF16 입력·FP32 누산의 dense 성능이다. Sparse TOPS를 대입하지 않는다. Attention의 혼합 정밀도와 Softmax 시간 근사는 아래 별도 가정을 따른다.

- HBM 이동 시간: bytes / 유효 HBM BW.
- Compute 단계 시간: FLOPs / 유효 Compute 성능.
- GEMM Compute 단계는 SRAM 접근 하한을 추가해 max(Compute 시간, SRAM bytes / 유효 SRAM BW)로 추정한다.
- 통신 시간: 알고리즘 라운드 수 × link latency + Rank당 송신 bytes / 유효 Link BW.
- 전체 시간: 단계 시간의 합. 서로 다른 단계 사이의 중첩은 없다.

Compute 단계 내부 max 모델은 이상적인 연산·SRAM 접근 중첩 하한이다. 단계 사이 순차 가정과 함께 표시한다. SRAM 대역폭 모델을 정의하지 않은 Operator에는 이를 억지로 적용하지 않고 해당 민감도를 ‘모델 미지원’으로 표시한다.

## 제안 기본값

| 항목 | 기본값 |
|---|---|
| 가상 Hardware | 200 TFLOP/s, HBM 1,000 GB/s, SRAM 10,000 GB/s |
| SRAM 용량 | 32 MiB |
| 논리적 Rank 수 | 4 |
| Rank당 Link | 송신 100 GB/s, 라운드 지연 2 µs |
| GEMM | M=N=K=4096, 타일 128×128×128 |
| Attention | Batch=1, Heads=32, Sequence=1024, Head dimension=128 |
| MoE | Tokens=1024, Hidden=4096, FFN=14336, Experts=8, Top-k=2 |
| Collective | All-Reduce, Rank당 입력 256 MiB |

가상 Hardware는 HBM 용량 제한을 P0에서 모델링하지 않는다. SRAM 용량은 GEMM 타일에 적용한다. Link는 Rank당 독립 송신·수신이 가능한 full-duplex 경로로 가정하며 공용 fabric 경합은 없다.

Rank 수 P는 공통 Hardware의 rankCount로 보관하며 기본값은 4인 양의 정수다. Compute·HBM BW·SRAM 용량/BW·Link BW는 모두 Rank당 값이며, 모든 Rank는 동일한 사양을 사용한다. MoE와 Collective는 P를 사용한다. GEMM과 Attention은 단일 Rank 분석으로, 저장된 rankCount가 바뀌어도 결과가 변하지 않는다.

## GEMM/GEMV

A[M,K] × B[K,N] → C[M,N]. GEMV는 M=1 등 동일 GEMM 형식으로 표현한다.

- FLOPs = 2MNK.
- 유일한 입력·출력 크기 = q(MK + KN + MN).
- 타일 개수 I=ceil(M/tM), J=ceil(N/tN), L=ceil(K/tK).
- Output-stationary 순서: 각 출력 타일에 대해 K 타일을 순회하고 마지막에 C를 저장한다.
- HBM A 읽기 = qMKJ, B 읽기 = qKNI, C 쓰기 = qMN.
- HBM traffic Q = q(MKJ + KNI + MN).
- Arithmetic intensity = 2MNK / Q. 유일 데이터량으로 나눈 값과 혼용하지 않는다.
- SRAM traffic = q(MKJ + KNI) + aMN(2L−1).

SRAM 식은 입력을 연산기에 한 번 전달하고, 각 K 타일마다 누산값을 쓰며 첫 K 타일 이후 누산값을 읽는 추상 모델이다. 레지스터 내부 접근과 bank conflict는 제외한다. HBM→SRAM 쓰기는 HBM Load 단계에서 설명하고 별도의 SRAM 시간으로 중복 합산하지 않는다.

단계는 Load A/B → Compute → Store C의 집계 세 단계로 시작한다. 실제 타일 반복 수와 경계 타일은 근거에 표시한다. 세 단계가 전체 트래픽을 집계하므로 개별 타일의 실제 실행 순서 타임라인으로 해석하지 않는다.

필요 SRAM = q(tM′tK′ + tK′tN′) + a tM′tN′, tM′=min(M,tM) 등. Double buffering은 없다. 부족하면 계산 결과를 유효하지 않음으로 표시하고 필요한 용량을 안내한다. 자동 타일 변경이나 암묵적 spill은 하지 않는다. SRAM 용량 변경은 타일 적합 여부에 영향을 주며, 타일을 고정한 채 용량만 늘려도 재사용이 증가한다고 계산하지 않는다.

## Attention

P0 기본은 Q/K/V가 이미 만들어진 dense non-causal prefill이다. QKV projection은 제외한다. Causal·Decode·FlashAttention은 후속 범위다. 이는 앞선 계획의 causal 제안에서 마스킹과 실제 실행 연산량의 혼동을 줄이기 위해 구체화한 기본값이다.

B=Batch, H=Heads, S=Sequence, D=Head dimension.

- Q/K/V 각 shape = [B,H,S,D].
- QKᵀ와 PV FLOPs 합 = 4BHS²D.
- Softmax 비용 = 5BHS² FLOPs 상당의 근사치. exp/div 처리량은 실제 장치와 다르며 이 계수는 보정 대상이다.
- Q/K/V 읽기와 출력 쓰기 = 4BHSDq.
- Score와 Probability는 FP32로 HBM에 각각 쓰고 읽는다: 4BHS²a.
- 전체 HBM traffic = 4BHSDq + 4BHS²a.

단계: QKV Load → QKᵀ → Score Store/Load → Softmax → Probability Store/Load → PV → Output Store. Compute는 FLOPs/C_eff, 이동은 bytes/유효 BW로 계산한다. Q/K/V 재읽기·타일 재사용·SRAM BW/용량 제한은 이 모델에 포함하지 않는다. 따라서 GEMM의 메모리 모델과 정확도가 같다고 표시하지 않는다. 화면에 ‘중간 행렬을 HBM에 저장하는 단순 모델’을 표시한다.

Probability는 FP32 저장·읽기를 유지하며 PV는 FP32 Probability와 BF16 V의 혼합 정밀도 연산으로 표현한다. P0는 PV의 FLOPs와 Softmax의 FLOPs 상당량에도 QKᵀ와 동일한 C_eff를 적용하는 시간 근사를 사용한다. 이는 서로 다른 정밀도·연산의 실제 처리량이 같다는 주장이 아니다. 별도 BF16 변환 단계나 추가 HBM 이동은 삽입하지 않고, 변환·최종 출력 정밀도 변환의 비용은 모델에서 제외한다. 이 가정을 PV·Softmax 시간 산식과 Compute 민감도에 연결한다. 정밀도별·연산별 처리량을 도입할 때 모델 버전을 변경한다.

## MoE

P개의 Rank에 E개 Expert를 균등 배치하며 E는 P로 나누어져야 한다. T는 전체 Rank에 걸친 원본 토큰 수, D는 Hidden, F는 FFN, k는 Top-k다. Expert는 D→F→D의 두 선형층으로 시작한다. 균등 라우팅의 평균 토큰 수를 사용하므로 분수 평균이 가능하다.

- Router FLOPs = 2TDE. Rank당 2TDE/P.
- Expert 전체 FLOPs = 4TkDF. Rank당 4TkDF/P.
- Dispatch Rank당 송신량 = (TkDq/P) × (P−1)/P.
- Combine Rank당 송신량도 동일. 동일 Rank에 남는 토큰은 Link traffic에서 제외한다.
- Expert weights Rank당 읽기 = 2(E/P)DFq.
- Router weights는 각 Rank에 복제: DEq.
- 로컬 원본 토큰 읽기와 최종 출력 쓰기 = 각각 TDq/P.

단계: Input/Router Weight Load → Router → Dispatch → Expert Weight Load → Expert Compute → Combine → Output Store. Expert activation은 로컬 버퍼에 머문다고 가정하고 중간 HBM spill을 제외한다. Router 선택·정렬, activation 함수, combine 가중합 연산, 통신 metadata는 제외한다. 이 제외 항목과 균등 라우팅 가정을 표시한다. 버퍼 용량·SRAM 대역폭을 반영하려면 별도 모델이 필요하다.

전체 지연은 균등한 Rank 하나의 임계 경로로 계산한다. 전체 FLOPs를 Rank당 성능으로 나누지 않는다. Dispatch/Combine은 pairwise All-to-All 시간 모델을 사용한다. Top-k ≤ E, E mod P=0을 검증한다.

## Collective

P=Rank 수. 통신은 논리적 Rank 간 경로이며 물리적 NVLink·NIC·스위치 사양을 의미하지 않는다. P=1이면 통신량·시간·라운드는 모두 0이다.

| 연산 | S의 의미 | Rank당 송신량 | 라운드 |
|---|---|---|---|
| Ring All-Reduce | 각 Rank 입력 텐서 bytes | 2(P−1)S/P | 2(P−1) |
| Ring All-Gather | 각 Rank 초기 shard bytes | (P−1)S | P−1 |
| Ring Reduce-Scatter | 각 Rank 전체 입력 bytes | (P−1)S/P | P−1 |
| Pairwise All-to-All | 각 Rank의 전체 send buffer bytes, self 포함 | (P−1)S/P | P−1 |

All-Reduce는 Reduce-Scatter와 All-Gather 두 단계로 표시한다. 입력 S를 P개 shard로 나눈다. All-Gather의 최종 텐서는 PS bytes다. Reduce-Scatter의 출력은 S/P bytes다. All-to-All은 동일 크기의 P개 목적지 shard로 나눈다. 수신량은 대칭 가정에서 송신량과 같다. 전체 송신량은 Rank당 송신량 × P이며 송수신을 더해 이중 계산하지 않는다.

P0 Collective의 payload는 q=2 byte인 BF16 원소로 정의한다. S는 단위 변환 후 양의 정수 byte이고, All-Reduce·Reduce-Scatter·All-to-All은 S mod (Pq)=0이어야 한다. All-Gather는 이미 shard인 S를 입력받으므로 S mod q=0만 요구한다. 조건을 만족하지 않으면 입력 오류로 처리하고, 필요한 배수와 가장 가까운 양의 유효 크기를 안내한다. 자동 패딩·반올림은 하지 않는다. All-Reduce 내부 All-Gather의 입력은 원래 S가 아닌 S/P다. MoE의 Dispatch/Combine은 균등 라우팅의 평균 통신량이므로 이 정수 shard 검증을 적용하지 않는다.

Reduction 연산 비용과 로컬 메모리 복사 비용은 제외한다. HBM·Compute 민감도는 ‘모델 미지원’으로 표시한다. 작은 메시지에서는 latency가 지배할 수 있다.

## 병목·민감도·Trace

단계별 제한 자원은 해당 시간 모델을 결정하는 자원이다. GEMM Compute에서는 max의 지배 항을 표시하고 동률이면 공동 제한으로 표시한다. 전체 자원 기여도는 각 자원이 지배하는 단계 시간을 합산한 설명값이다. 이를 실제 자원 점유율이라고 부르지 않는다.

공동 제한 단계는 ‘Compute + SRAM 공동’과 같은 별도 항목에 단계 시간을 한 번만 합산하며, 개별 자원 항목에는 중복 배분하지 않는다. 단독·공동 항목의 시간 합은 전체 시간과 일치해야 한다. 양의 두 시간 x,y가 |x−y| ≤ 10⁻⁹ × max(x,y)이면 표시상 동률로 취급하되, 단계 시간은 원래 값의 max를 유지한다. 0초 단계는 기여도에 더하지 않으며, 전체 시간이 0이면 기여 비율과 병목은 ‘해당 없음’이다.

자원 r의 2배 민감도: 다른 입력은 고정하고 성능 또는 BW 하나만 2배로 바꿔 재계산한다. Speedup=Tbase/Tchanged, 절감률=1−Tchanged/Tbase. Link BW 2배 시 latency는 그대로다. 전역 병목 후보는 가장 큰 시간 절감을 주는 지원 자원으로 표시한다. 동률은 함께 표시하고 효과가 없으면 명시한다.

위 비율식은 Tbase와 Tchanged가 모두 양수일 때만 사용한다. 유효한 Collective P=1처럼 Tbase=Tchanged=0이면 절감 시간은 0초, Speedup과 절감률은 null로 반환하고 ‘해당 없음: 통신 없음’으로 표시한다. 정상 결과의 0% 개선이나 ‘모델 미지원’과 구분한다. 유효성 오류는 민감도 계산을 수행하지 않는다. 양의 기준 시간이 재계산 후 0이 되는 등 나머지 비정상 수치는 계산 오류로 진단하며 Infinity·NaN을 결과로 제공하지 않는다.

모든 핵심 결과에는 산식 ID, 실제 대입값, 단위, 입력 ID, 관련 단계·자원 ID, 모델 가정 ID를 연결한다. SRAM 부적합 같은 유효성 오류가 있으면 전체 시간과 민감도를 정상 결과처럼 제공하지 않는다.
