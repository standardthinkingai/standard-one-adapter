# JEV 형태의 판단 성능을 위한 추가 벤치 선택

확인일: 2026-09-21. **현재 비교의 중심은 KEV의 고정 평가 세트와 JevBench**다.
추가 공개 데이터는 아래 두 개를 추천한다. 이 문서는 선정 근거이며,
CLINC150과 ContractNLI의 변환·실행·추가 지표는 아직 통합하지 않았다.
모두 텍스트 입력에서 정해진 선택지의 확률만 읽어 평가할 수 있다.

| 우선순위 | 데이터 | 측정하려는 판단 | 공식 테스트 규모 | 라이선스 |
|---|---|---|---|---|
| 1 | CLINC150 / oos-eval | 요청 라우팅과 지원 범위 밖 요청 거절 | 지원 범위 4,500 + 범위 밖 1,000 = 5,500문항 | CC BY 3.0 |
| 2 | ContractNLI | 문서가 조건을 지지하는지, 반박하는지, 근거가 없는지 | 123개 계약 × 17개 명제 = 2,091판단 | CC BY 4.0 |

## CLINC150: 라우팅과 범위 밖 요청

150개 intent와 `oos` 한 개를 합쳐 **151개 선택지**를 사용한다.
공식 `data/data_full.json`은 intent마다 train 100개·validation 20개·test 30개이며,
별도 OOS 분할은 train 100개·validation 100개·test 1,000개다.
따라서 검증 세트는 총 3,100개, 테스트는 5,500개다.
영어 단일 intent 요청을 사람이 작성·패러프레이즈한 데이터다.
[공식 설명](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de)
[라이선스](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE)

권장 매핑은 `state=사용자 발화`, `choice=150개 intent + oos`다.
공식 intent 이름의 밑줄을 공백으로 바꾸는 정도의 결정적 설명을 쓰고,
테스트 발화나 정답을 이용해 선택지 설명을 만들지 않는다.
모델마다 동일한 후보 목록·순서를 사용하며, 후보 일부를 정답에 맞춰 제거하지 않는다.
151개 선택지는 기존 어댑터의 255개 한도 내에 있다.

통합할 때 추가할 지표:

- 지원 범위 150개에 대한 accuracy·macro F1, 전체 151개 macro F1.
- `p(oos)`를 점수, OOS를 양성으로 정한 AUROC·AUPRC.
- FPR@95% OOS TPR: OOS의 95%를 탐지할 때 정상 요청을 잘못 거절하는 비율.
- 고정 임계값에서 OOS 탐지율과 정상 요청 오거절률, 수락한 요청의 정확도·coverage.
- Brier·NLL·ECE와 요청별 p50/p95 지연. 후보 151개의 비용을 함께 명시.

운영 임계값을 선택한다면 validation만 사용하고 test에는 고정해서 적용한다.
가중치 학습 없이 비교하되, 임계값 조정 여부는 별도로 기록한다.
OOS는 **지원하지 않는 intent**라는 뜻이다. 입력 근거가 부족해서 답을 알 수 없는
KEV `unknowable`과는 다른 능력을 측정하므로 둘을 합산하지 않는다.

직접 검증한 원본: 커밋 `828f8093932c8fe6ca7936c3d2e52903b1c523de`,
[`data/data_full.json`](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json),
SHA256 `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`.
다른 `small`·`plus` 변형과 혼합하지 않는다.

## ContractNLI: 명시된 문서 근거에 따른 판단

607개의 NDA에 동일한 17개 명제를 사람이 주석했다.
분할은 train 423개·development 61개·test 123개 계약이다.
정답은 `Entailment`, `Contradiction`, `NotMentioned`의 세 종류다.
예외에 의한 부정, 문서에 없는 내용, 긴 문서의 근거를 찾는 판단을 점검하기 좋다.
[공식 데이터·스키마·라이선스](https://stanfordnlp.github.io/contract-nli/)
[원 논문의 분할 표](https://aclanthology.org/2021.findings-emnlp.164.pdf)

`state=전체 계약 텍스트`, 각 명제를 `choice` 질문으로 바꾼다.
출력은 세 선택지의 확률이며, 근거 문장 생성은 요구하지 않는다.
`NotMentioned`를 `false`로 합치면 반박과 근거 부재를 구분할 수 없으므로
기본 비교에는 `noul` 대신 3-way `choice`를 권장한다.
원본의 정답 evidence span은 입력에 넣지 않는다.

권장 지표는 3개 레이블 macro F1·accuracy, 명제별 성능, Brier·NLL·ECE,
`NotMentioned` precision/recall이다. 계약 단위로 신뢰구간을 계산해 한 문서의
17개 질문을 독립 표본으로 취급하지 않는다.
명제당 지연과 계약 전체 17개 질문의 지연을 함께 보고한다.

긴 계약을 임의로 잘라 평가하지 않는다. 각 모델의 tokenizer로 입력 길이를 먼저
측정하고, 공통 context 한도 밖의 문서는 실패·제외 수를 명시한다.
본문 전체 판단만 수행한 결과를 원 논문의 evidence identification까지 포함한
전체 과제 결과와 동일하다고 표현하지 않는다.

## 해석 범위

두 데이터 모두 공개된 지 오래되어 최신 Qwen·Mistral의 사전학습 오염을 배제할 수 없다.
학습하지 않은 모델을 평가한다는 것은 **이번 실험에서 추가 튜닝하지 않는다**는 뜻이다.
공개 테스트를 처음 접했다는 보장은 아니다. 영어 중심 결과를 한국어 운영 성능으로
일반화하지 않으며, 모델 비교에서는 동일 입력·동일 분할·동일 판단 형식을 유지한다.
