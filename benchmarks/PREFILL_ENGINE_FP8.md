# Prefill 확률 엔진의 FP8 경로

**아래 실측은 텍스트 입력 기준입니다.** 이후 추가한 [이미지 입력 경로](PREFILL_ENGINE_IMAGES.md)의 native 토큰·processor CPU 검증은 별도이며 FP8 이미지 forward·수치 패리티·성능은 GPU 미검증입니다.

**FP8 구현·설정 가드·벤치마크 profile을 로컬 canonical 코드에 반영했으며, 동일 SHA의 불변 격리 실행으로 H200의 3B·8B 첫 장시간 실행쌍을 검증했다.** 두 모델 모두 엄격한 동적 HTTP 패리티를 통과해 종료·보존했다. BF16 기본값·FP8 opt-in을 유지한다. 서버 설정만 FP8로 표시한 결과를 지원 근거로 사용하지 않는다. GPU 기동·원시 확률·HTTP 성능 결과와 남은 제약은 [실험 계획](../../outputs/h200-prefill-performance-20260928/FP8_PLAN.md)에 기록한다. 독립 반복 3회와 전체 범위는 미완료이며 MI325X는 미측정이다.

## 현재 검증 상태 (2026-09-28)

- canonical의 기존·신규 CPU pytest 210개와 readout 검사 11개, 별도 BF16/FP8 품질 도구 검사 11개가 통과했다. 로컬 full routing 테스트는 `sgl_kernel`·`triton` 부재로 실행하지 않았다. H200 실측은 같은 구현 SHA의 격리 소스를 사용한다.

### 3B 첫 실행쌍

- base 3B FP8에서 순정 SGLang wheel·외부 adapter 기준선과 fork 통합 candidate width8192를 같은 dense graph·입력·정밀도 조건으로 비교했다. C별 warmup 제외, 최소 30초, 오류 0이다. 아래 수치는 첫 실행쌍의 관찰이다.

| C | Stock FP8 → candidate FP8 width8192 p50 (ms) |
|---|---|
| 8 | 35.74 → 27.30 |
| 16 | 50.85 → 40.80 |
| 32 | 66.70 → 43.37 |
| 64 | 130.96 → 84.98 |
| 128 | 261.01 → 178.44 |

p50는 19.76~35.11% 감소했고 RPS는 28.65~61.68% 증가했다. p95/p99도 모든 C에서 낮았다. 실제·짧음·중간·김·혼합 5 workload의 raw-ID 총 11,020출력에서 최대 Δp 5.7725032e-8, 동적 HTTP 첫 pass에서 최대 Δp 5.6340368e-8이었다. argmax·표시 답 불일치 0으로 엄격한 패리티를 통과해 `status=comparable`, `single_pair_winner=candidate`다. **독립 1쌍이므로 `final_claim_eligible=false`**이며 최종 성능 우위는 아직 확정하지 않는다. 원본·설정·보존과 세부 지표는 [3B FP8 결과](../../outputs/h200-prefill-performance-20260928/THREE_B_FP8_RESULTS.md)에 있다.

이전 candidate BF16 width0 대비 이번 FP8 width8192의 C8/C16 p50는 각각 27.56%/28.49% 증가했다. 양자화와 projection 폭·실행 시점이 함께 바뀌어 원인을 분리할 수 없으며, **BF16을 기본값으로 유지하고 FP8은 명시적으로 선택**한다. 고정 base Ministral3 체크포인트의 결과이며 새 StandardOne v2.1 가중치·learned head·MI325X 결과가 아니다. 모델별 독립 반복·합성 workload HTTP 성능 sweep과 배치 한도 튜닝은 완료 처리하지 않는다.

### 8B 첫 실행쌍

같은 dense graph·입력·FP8 조건의 stock / candidate width8192 측정을 완료·종료·보존했다. 아래는 첫 실행쌍의 관찰이다.

| C | Stock FP8 → candidate FP8 width8192 p50 (ms) | Candidate RPS |
|---|---|---|
| 8 | 38.561 → 27.878 | 280.684 |
| 16 | 66.870 → 45.194 | 350.384 |
| 32 | 97.105 → 78.658 | 403.484 |
| 64 | 189.305 → 156.408 | 411.912 |
| 128 | 366.760 → 302.208 | 424.998 |

p50는 17.38~32.42% 감소했고 RPS는 21.55~44.73% 증가했다. 동적 HTTP 첫 pass 최대 Δp **4.768649486397436e-8**, argmax 불일치 0·표시 답 일치로 엄격한 패리티를 통과했다. `status=comparable`, `single_pair_winner=candidate`지만 모델별 독립 1쌍이므로 `final_claim_eligible=false`다. 표본 VRAM 최대는 stock **32,806MiB**, candidate **32,124MiB**였으며 특정 allocation의 절감량으로 해석하지 않는다. 원본·설정·보존과 세부 지표는 [8B FP8 결과](../../outputs/h200-prefill-performance-20260928/EIGHT_B_FP8_RESULTS.md)를 따른다.

완료된 stock 8B의 [BF16/FP8 별도 품질 재검증](../../outputs/h200-prefill-performance-20260928/quality-stock-8b-bf16-vs-fp8-trial1.json)은 각 C의 첫 pass clean 1,194문항에서 FP8 정확도가 0.335~0.419%p 감소한 관찰을 기록한다. 같은 FP8 엔진 패리티나 성능 승자 비교와 분리하며 반복 요청을 독립 품질 표본으로 합산하지 않는다.

두 모델의 완료분은 verified snapshot `20260928T010144.920728Z`(844파일·issues0, archive SHA `2dc04cd50191a3581a71b90f859dd751baf9a699920f1b17ed211333668e0ea5`)에 보존했다. 후속 3B BF16 width8192 C8/C16 대조도 완료했다. p50은21.11/32.42ms였으나 BF16 동적 확률 gate는 실패했다. 사전 입력·시간·HTTP pool16대128 차이도 있어 순수 양자화 효과를 확정하지 않는다. [대조 결과](../../outputs/h200-prefill-performance-20260928/BF16_LOW_C_CONTROL_RESULTS.md)를 따른다. 모든GPU실행을종료하고10:10 KST담당작업에반환했다. 최종대조까지보존한snapshot은20260928T010900.243747Z,884파일·issues0이다. 후속 대조·배치 튜닝·독립 반복의 순서는 [저지연 우선 설계](../../outputs/h200-prefill-performance-20260928/LATENCY_PRIORITY.md)를 따른다.

## 계산 범위

| 구간 | 정밀도 |
|---|---|
| Attention/MLP의 지원되는 linear 가중치·GEMM 입력 | native online FP8 W8A8 |
| Embedding, 모델 activation 경계, LM head, 선택 행 projection | BF16 |
| 선택 행 출력과 확률 후처리 | FP32 출력·기존 확률 처리 |

원본 BF16 체크포인트를 로드한 뒤 본체 가중치를 변환한다. readout은 같은 고정 체크포인트의 LM 행을 재사용한다. 토크나이저와 요청·응답 형식은 [기존 통합 엔진](PREFILL_ENGINE.md)과 같다. 신규 출력 토큰 생성 없이 마지막 prefill hidden state에서 확률을 추출한다.

H200의 지원되는 CUTLASS 경로는 소스상 가중치를 채널별로, activation을 토큰별로 양자화한다. `weight_block_size=None`만으로 전체 텐서 스케일 한 개를 사용한다고 해석하지 않는다. AMD에서 같은 scaling/커널이 선택된다고 가정하지 않는다.

초기 지원 범위는 다음 두 체크포인트로 제한한다.

| 모델 | 고정 revision |
|---|---|
| `mistralai/Ministral-3-3B-Instruct-2512-BF16` | `b6d637bef2393152b3da2b2fde72eecdee30557e` |
| `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `f6fae9795746f63c9be8344932f01275f3c63734` |

새 학습 체크포인트의 품질 검증을 대신하지 않는다. serialized FP8 체크포인트, MXFP8, block quantization, FP8 learned head, FP8와 KV-free의 조합은 초기 경로에서 거부한다. BF16 KV 풀은 그대로 존재한다.

## 실행 설정

기존 BF16 통합 서버 명령에 `--quantization fp8 --dtype bfloat16`을 지정한다. **이번 3B·8B 통과 실측의 candidate는 `SGLANG_DECISION_PROJECTION_WIDTH=8192`를 명시했다.** 같은 구성을 선택하려면 해당 서버를 시작할 shell에서 다음 값을 설정한 뒤 기존 명령의 모델·revision·readout·graph 설정과 FP8 플래그를 사용한다.

```bash
export SGLANG_DECISION_PROJECTION_WIDTH=8192
```

일반 기본값은 0이며 `--quantization fp8`만으로 8192가 선택되지는 않는다. 3B FP8 width0는 긴 synthetic/mixed 입력에서 패리티를 실패해 장시간 성능 후보로 사용하지 않았다. 고정 revision과 일치하는 readout artifact가 필요하다. TP/PP/DP=1, radix cache·decode graph 비활성, chunked prefill -1 등 기존 계약을 유지한다.

격리 실험 launcher는 양쪽 엔진에 같은 graph·token budget·최대 동시성 조건을 적용한다. `SGLANG_DECISION_PROJECTION_WIDTH=0|8192`는 시작 시 한 번 읽는다. 8192는 선택 행 GEMM의 열 수를 확장해 stock의 BF16 반올림과 맞추기 위한 옵션이며 FP8 정밀도를 올리는 기능은 아니다. 모델·출력 head별 실제 패리티를 확인한 값을 사용한다.

서버 시작 설정, 실제 ModelConfig, 첫 forward의 `Fp8Config`·BF16 head dtype·unquantized embedding method를 검사한다. 지원하지 않는 경로는 결과를 반환하기 전에 실패해야 한다. head 검사는 최초 한 번이며 이후 요청에 추가 GPU→CPU 동기화를 만들지 않는다.

## 저지연 우선 측정

우선순위는 **p50 → p95/p99 → 성공 처리량**이다. FP8은 명시적으로 선택하는 옵션이며 품질과 실제 p50를 확인하기 전 BF16 기본값을 바꾸지 않는다. 부하에 따라 정밀도를 자동 전환하면 같은 요청의 확률 의미도 달라질 수 있어 그런 자동 전환을 추가하지 않는다. 토크나이저·큐·GPU·응답을 포함하는 HTTP end-to-end 지연을 기준으로 한다. 서버 로딩·온라인 양자화 시간은 요청 p50와 별도로 기록한다.

- stock FP8와 전용 FP8를 같은 입력·양자화 정책·head 정밀도·자원·graph·배치 한도로 비교한다.
- 동시성 8/16/32/64/128 전체에서 warmup 제외, 최소 30초, 충분한 요청 수를 확보한다. 최종 반복 검증에는 독립 실행쌍 3회가 필요하다.
- 배치 요청 한도 `prefill_max_requests`와 token budget을 조절할 때 p50뿐 아니라 큐 대기, p95/p99, 오류율·RPS도 확인한다. 배치를 줄이면 항상 더 빨라진다고 가정하지 않는다.
- BF16 대비 FP8의 정확도·Brier·NLL 변화는 별도 품질 분석이다. 반복 요청은 독립 정확도 표본으로 중복 집계하지 않는다. 서로 다른 정밀도를 같은 조건 엔진 우위로 비교하지 않는다.
- 같은 FP8 정밀도 경로의 엄격한 수치 패리티를 유지한다. 정확성 실패가 있으면 성능 관찰 수치와 함께 공개하고 최종 승자로 판정하지 않는다.

NVIDIA와 AMD의 FP8 형식·커널은 다를 수 있으므로 CUDA에서 양자화한 가중치와 실측 결과를 MI325X 검증으로 재사용하지 않는다. [SGLang의 FP8 지원 문서](https://docs.sglang.io/docs/advanced_features/quantization)를 참고하되 장치별 실제 측정이 필요하다.
