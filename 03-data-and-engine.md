# 데이터 구조·엔진 인터페이스

P0 설계 초안 · 2026-10-11

## 구조

React 화면과 계산 엔진을 분리한다. 엔진은 브라우저·React·React Flow에 의존하지 않는 TypeScript 모듈이다. 입력을 받아 결과를 반환하는 순수 함수로 구성하며 선택 상태는 UI에 둔다. P0는 별도 상태 관리 라이브러리 없이 시작한다.

구현 시 예상 구성: domain(타입), engine(공통 계산·검증), operators(네 모델), presets(가상 하드웨어), ui(입력·그림·단계·Trace). 이는 설계이며 현재 코드를 생성하지 않는다.

## 핵심 데이터 계약

| 객체 | 필수 정보 |
|---|---|
| Workload | Operator 구분, 정밀도, Operator별 입력, 알고리즘 ID |
| Hardware | ID, 명칭, 공통 rankCount, Rank당 Compute·SRAM 용량/BW·HBM BW·Link BW/latency, 각 효율 |
| Resource | ID, 종류, 소속 Rank, 후속 계층·연결 참조 |
| Tensor | ID, shape, precision, bytes, 전체/Rank당 범위 |
| Step | ID, 종류, 선행 단계 ID, 자원 ID, 텐서 ID, 시간 Metric ID |
| Transfer | ID, 출발·도착 자원, Tensor ID, bytes Metric ID, Step ID, trafficClass(hbm/sram/link) |
| Metric | ID, 값, 단위, 범위, Formula ID, 관련 Step·Resource ID |
| Formula | ID, 설명, 표현식, 입력/중간 Metric 참조, 대입값, 가정 ID |
| Assumption | ID, 설명, 적용 모델, 제한·제외 사항 |
| Diagnostic | 오류/주의 수준, 설명, 관련 입력 ID, 수정 안내 |
| SensitivityResult | 자원 ID, 상태(ok/no-work/unsupported/invalid), 기준·변경 시간, 절감 시간, nullable Speedup·절감률, 사유 |
| TimeContribution | 제한 자원 ID 집합, 관련 Step ID 목록, 합산 시간, nullable 전체 시간 대비 비율 |
| CoreAnalysisResult | 모델 버전, 자원·텐서·단계·전송·지표·산식·가정·진단, 총시간, 유효성 |
| AnalysisResult | CoreAnalysisResult와 자원별 민감도 |

Workload는 Operator별 구분 타입으로 정의한다. GEMM의 M/N/K와 Collective의 bytes 입력을 하나의 선택적 필드 묶음에 섞지 않는다. Metric의 범위는 global 또는 per-rank를 필수로 기록한다. 집계 Step임을 표시할 수 있어야 한다.

rankCount는 Hardware만 소유하고 Workload에 복제하지 않는다. MoE와 Collective의 입력·산식 참조는 hardware.rankCount를 사용한다. GEMM과 Attention은 단일 Rank를 분석하므로 해당 입력에 결과 의존성을 만들지 않는다. Operator 변경 시 Hardware와 rankCount를 유지하고 새 Workload와의 조합을 재검증한다.

SensitivityResult의 ok는 양의 시간에 대한 정상 비교, no-work는 기준·변경 시간이 모두 0인 유효한 비교다. no-work는 두 시간과 절감 시간을 0, 비율 두 개를 null로 반환한다. unsupported는 미지원 모델, invalid는 입력 또는 계산 오류이며 비교 수치를 null로 반환하고 사유를 제공한다. 비율이 null이면 일반 수치 Metric을 생성하지 않는다. 민감도 내부 재분석은 analyze를 호출하되 다시 sensitivity를 호출하지 않는다.

TimeContribution은 각 양의 시간 Step을 단독 또는 공동 제한 자원 집합 하나에만 귀속한다. 전체 시간이 0이면 기여 비율은 null이다. 이 합계와 자원별 2배 민감도의 결과는 별도로 보관한다.

## Tensor 크기와 전송량 집계

Tensor.bytes는 해당 Tensor의 논리적 크기다. Transfer의 bytes Metric은 관련 Step 전체에서 반복 접근을 합산한 단방향 전송량이며 Tensor.bytes와 같을 필요가 없다. 전송량의 범위는 Metric에 기록한다. 양방향 이동은 읽기·쓰기 Transfer로 나누고, 여러 Tensor를 함께 읽는 Step도 Tensor별 Transfer로 나눈다.

합산은 동일 집계 범위와 trafficClass 안에서 고유 Transfer ID별로 한 번만 수행한다. HBM·SRAM·Link 전송량을 단일 ‘데이터량’으로 합치지 않는다. HBM→SRAM Load는 hbm으로 집계하고 SRAM 쓰기로 다시 세지 않는다. SRAM 접근은 02의 Compute 모델에 포함된 접근만 집계한다. 송신과 수신은 같은 전송의 두 관점이며 Link 총량에 중복 가산하지 않는다.

UI에서 물리 자원의 표시 노드를 복제하거나 여러 화살표를 묶어도 엔진의 Transfer를 복제하지 않는다. Rank당 결과는 대표 Rank의 논리적 전송량으로 유지하고, 전체량은 명시적인 ×P 산식으로 만든다. Rank 묶음 그림과 예시 경로를 합산해 전체량을 만들지 않는다. 각 Step의 계층별 전송량 합은 해당 시간 산식이 참조하는 bytes와 일치해야 한다.

## 엔진 호출

1. validate(workload, hardware): 필드·조합·용량 진단 반환.
2. analyze(workload, hardware): 검증 후 CoreAnalysisResult 반환. 민감도를 호출하지 않는다.
3. sensitivity(workload, hardware, baseline): 같은 입력의 유효한 CoreAnalysisResult를 기준으로 지원 자원만 하나씩 변경해 analyze 재호출.
4. evaluate(workload, hardware): analyze를 한 번 호출하고, 유효하면 그 결과를 sensitivity에 전달해 AnalysisResult를 구성한다. 무효하면 정상 총시간과 비교 수치 없이 진단을 반환한다.

Operator 모델은 validate와 analyze를 같은 계약으로 제공한다. 민감도 지원 목록도 반환한다. 계산식 표현 문자열을 eval로 실행하지 않는다. 계산 함수와 설명용 산식이 동일한 이름의 입력값을 사용하고, 검증 사례로 수치 일치를 확인한다.

단위 변환은 입출력 경계에서만 한다. 원시 값은 반올림하지 않는다. NaN·Infinity·0 이하 성능과 용량, 잘못된 정수, 과도한 값으로 인한 유한수 범위 초과를 진단한다. 빈 입력 편집 중에는 이전 결과를 최신 결과로 표시하지 않는다.

## 수치 안전 범위와 계산 규모

P0의 정수 안전 상한은 2⁵³−1 = 9,007,199,254,740,991이다. Shape·타일·토큰·Expert·Top-k는 1부터 이 상한까지의 정수이며 조합 제약은 별도로 검증한다. Rank 수는 브라우저 구현의 지원 범위로 1~4,096으로 제한한다. Collective 입력과 SRAM 용량은 byte 변환 뒤 양의 안전 정수여야 한다. 이 범위는 제품의 물리 한계가 아니다.

입력 문자열의 정수 여부와 범위를 손실 없이 검증한 뒤 Number로 변환한다. byte 단위 변환도 원래 입력 문자열의 십진 값을 기준으로 수행하며, 소수 byte를 반올림해 통과시키지 않는다. 정수 덧셈·곱셈은 결과를 만들기 전에 상한 초과 여부를 검사한다. FLOPs·Tensor 크기·traffic의 정수 중간값도 같은 상한을 적용한다. MoE의 평균값은 나눗셈 뒤 소수를 허용하되 분자 계산의 안전성을 먼저 확인한다. 보수적인 중간값 제한 때문에 거부한 경우에도 관련 입력과 제한 사유를 표시한다.

성능·BW는 양의 유한수, 효율은 0 초과 1 이하, Link latency는 0 이상의 유한수다. 유효 성능과 시간 계산의 각 단계에서 overflow와 양의 값이 0이 되는 underflow를 검사한다. 지원 범위를 넘으면 오류를 반환하고 값을 자동 축소하지 않는다. 민감도의 2배 변경도 같은 검사에 포함한다.

Shape 크기의 실제 Tensor 배열, 타일별 Step, Rank 쌍별 Transfer를 생성하지 않는다. 집계 산식과 대표 Rank로 계산하고 그래프는 최대 8개 개별 Rank와 나머지 묶음으로 표시한다. 큰 Rank 수에서도 전체 P를 사용한 수치와 Trace는 유지한다.

## Trace 연결

P0는 선택한 Metric의 Formula와 입력/중간 Metric 참조를 따라 결과 → 산식 → 입력으로 탐색한다. ID 조회용 Map은 만들되 모든 입력에서 영향을 받는 결과를 찾는 역방향 의존성 인덱스는 후속 범위다. 총시간에서 개별 시간, bytes, Shape까지 필요한 근거를 단계적으로 조회한다.

입력·Step·Metric·Resource는 재계산해도 의미가 같으면 같은 ID를 유지한다. Operator 변경으로 ID가 사라지면 선택을 첫 단계로 초기화한다. Resource 선택은 해당 사양 설정과 현재 단계에 관련된 Metric을 보여준다. 관련 단계는 기존 Step의 자원 ID로 찾는다. UI 그래프 좌표는 엔진의 물리 의미와 분리한다.

## P0 비교 세션

UI 상태에는 편집 중 입력, 선택 단계 ID, 상세 대상 하나, 기준 스냅샷 하나, 선택적 목표 시간만 둔다. 현재 AnalysisResult와 비교 요약은 입력에서 계산한 파생값이다. 기준 스냅샷은 Workload·Hardware·모델 버전·AnalysisResult를 변경 불가능한 복사본으로 보관하며 현재 입력과 객체를 공유하지 않는다. 별도 선택안·선택 이유·선택 이력은 저장하지 않는다. Operator 변경·기본값 복원 때에도 기준을 자동 변경하지 않고 비교 호환 여부를 표시한다.

비교는 04의 Operator·알고리즘·정밀도·버전·단위·범위 조건을 확인하고, 변경된 입력과 계산 결과를 연결한다. 목표 시간의 유효성은 UI에서 검증하고, 잘못된 목표가 입력되면 달성 판정만 중단한다. 워크로드 분석 자체는 유효하다면 유지한다. 이 상태는 엔진 입력에 포함하지 않으며 P0 영구 저장 타입은 만들지 않는다.

## 확장 지점

공통 계약은 GEMM 구현 초기에 네 Operator의 대표 AnalysisResult 예시로 확인한다. 실제 모델을 모두 먼저 구현하지 않아도, 06 문서의 사례가 동일한 결과 표시·Trace·오류 처리 구조로 표현되어야 한다. 예시 결과는 개발 확인용이며 실제 입력에 대한 분석 결과로 사용자에게 제공하지 않는다.

- Hardware 계층에 die → package → device → node → rack을 추가한다.
- 연결에 bandwidth·latency·topology·출처를 추가한다.
- P0 순차 scheduler를 추후 dependency/resource scheduler로 교체한다. 시간 모델 버전을 구분한다.
- 모델 전체는 Operator 결과의 합성과 Tensor 연결로 확장한다.
- DSE는 기존 analyze 함수를 여러 설정으로 호출한다. 목표함수·제약·탐색은 별도 모듈로 둔다.
- 공개 사양, 추정값, 사용자 보정값의 provenance를 프리셋에 보존한다.
- 입력에서 결과로의 역탐색, 여러 비교 후보·별도 선택안·선택 이유·저장·공유는 후속 기능으로 추가한다.

미래 확장 타입은 필요할 때 추가한다. P0에서 Rack scheduler나 범용 수식 언어를 미리 구현하지 않는다.
