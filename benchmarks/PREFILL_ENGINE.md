# 단일 서버 프리필 판단 엔진

이 모드는 이 저장소의 `sglang-jev` fork가 `POST /v1/systemone`을 직접 제공합니다. 기존 독립 `jev-adapter` 프로세스와 `--backend candidate-only`/`learned-head`는 HTTP를 사이에 둔 비교용 시제품입니다. 통합 모드는 SGLang이 이미 로드한 토크나이저를 재사용하고, 프리필 마지막 hidden state에 작은 GPU readout을 적용해 선택지 로짓만 반환합니다. 생성 토큰·디토큰화·hidden state의 HTTP 전송·어댑터→엔진 HTTP 호출은 없습니다. 서비스는 작은 라벨 수 `K`의 로짓을 CPU에서 softmax해 기존 choice/noul/score JSON을 만듭니다. 프리필 중 KV 풀 쓰기는 여전히 발생합니다.

## 준비와 실행

SGLang fork를 실행할 **같은 Python 환경**에 어댑터 패키지를 설치합니다. 통합 요청은 엔진의 토크나이저를 쓰므로 별도의 `native-tokenizer` 설치나 두 번째 토크나이저 로드가 필요하지 않습니다. 다음 명령은 저장소 루트에서 실행합니다.

```bash
python -m pip install -e './jev-adapter[candidate-readout]'
```

먼저 GPU용 `jev-gpu-readout-v1` 파일을 만듭니다. 기존 `jev-export-readout`이 만든 LM 행 파일은 다음처럼 변환합니다. 원본 LM 행 파일 생성 절차는 [어댑터 README](../README.md#선택형-candidate-only-백엔드)를 참고하세요.

```bash
python -m jev_adapter.export_gpu_readout \
  --candidate-readout /models/ministral3-3b-readout.npz \
  --output /models/ministral3-3b-gpu-readout.npz
```

검증된 3B 학습 head는 아카이브의 `linear-seed17/head.safetensors`에서 내보냅니다. 이때 `learned-head` 의존성이 필요합니다. **학습에 사용된 표시 라벨은 A–C뿐**이므로 실제 3지선다 실험에는 `--max-labels 3`을 권장합니다. 이 옵션을 빼면 원본의 모든 62개 행을 내보내지만 D 이후의 품질은 검증되지 않았습니다. LM 행 방식도 `--max-labels N`으로 선두 N개 라벨만 내보낼 수 있습니다.

```bash
python -m pip install -e './jev-adapter[candidate-readout,learned-head]'
python -m jev_adapter.export_gpu_readout \
  --learned-head /path/to/verified/results/linear-seed17/head.safetensors \
  --max-labels 3 \
  --output /models/ministral3-3b-learned-gpu-readout.npz
```

다음은 3B BF16 예시입니다. `--model-path`는 선택한 readout의 `metadata.engine_model_path`와 **정확히 같아야** 하며 `--revision`도 artifact의 고정 commit과 일치해야 합니다. 로컬 체크포인트 경로를 쓴다면 내보낼 때 `--engine-model-path`로 동일한 경로를 지정합니다. LM 행 방식과 학습 head 방식은 각각 별도 readout 파일로 서버를 실행합니다.

```bash
MODEL_PATH=mistralai/Ministral-3-3B-Instruct-2512-BF16
REVISION=b6d637bef2393152b3da2b2fde72eecdee30557e
READOUT=/models/ministral3-3b-gpu-readout.npz

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --revision "$REVISION" \
  --served-model-name decision-model \
  --decision-readout-path "$READOUT" \
  --dtype bfloat16 \
  --tp-size 1 --pp-size 1 --dp-size 1 \
  --context-length 8192 --max-running-requests 128 \
  --decision-deadline-seconds 110 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --disable-decode-cuda-graph \
  --disable-prefill-cuda-graph \
  --host 127.0.0.1 --port 30000
```

기존 `POST /v1/systemone` 요청 형식을 그대로 쓰되 `model`은 `decision-model`로 보냅니다. 텍스트 또는 `images`가 포함된 요청을 받습니다. 단일·다중 이미지의 입력 형식과 CPU 검증/GPU 미검증 범위는 [이미지 입력 안내](PREFILL_ENGINE_IMAGES.md)를 따릅니다. 비어 있지 않은 `options.assistant_prefix`와 `options.return_logprobs: true`는 거부합니다. 선택지 수는 artifact의 라벨 수를 넘을 수 없습니다. GPU readout의 고정 보정 온도는 엔진이 한 번 적용하고, 요청의 `options.temperature`가 있으면 서비스가 추가 적용합니다.

**요청 한도와 오류.** 결정 워커는 한 요청의 모든 평가(질문 × permutations)를 GPU 작업 전에 토큰화하고 검사한다.
- 평가 하나라도 컨텍스트를 넘으면 요청 전체를 GPU 작업 없이 422 `context_length_exceeded`로 거절한다. 텍스트만으로 넘으면 `field`는 `questions.<질문 키>`, 이미지를 더해 넘으면 `images`다. 한도는 `context_length - 예약 토큰 - 1`이며 자동 잘라내기(`--allow-auto-truncate`)는 시작 시 거부한다.
- `--decision-deadline-seconds`(기본 110초)를 넘기면 남은 GPU 작업을 중단하고 504 `engine_timeout`을 준다. Gateway는 120초까지 기다리므로 엔진이 먼저 답한다.
- `--decision-max-queued-evaluations`를 주면, `--max-running-requests`를 넘어 대기할 평가 수가 그 값을 넘는 요청을 작업 전에 503 `engine_overloaded`로 거절한다. 엔진이 비어 있을 때 온 요청은 크기와 관계없이 받는다. 기본값은 제한 없음이다. 스케줄러의 `--max-queued-requests`는 비워 두거나 `--max-running-requests`보다 크게 둔다(작으면 시작 시 거부).
- 스케줄러가 대기열 초과로 거절하면(HTTP 503) 503 `engine_overloaded`로 바꿔 준다.

시작 시 모델 경로·revision·Mistral3 아키텍처·hidden 크기·라벨 토큰 ID와 토크나이저의 `[/INST]` 경계를 검사합니다. 위 기본 전용 모드는 비양자화 BF16/FP16 또는 초기 지원 체크포인트의 native FP8 본체 + BF16 head를 지원합니다. `TP=PP=DP=1`을 요구합니다. 이미지 입력에는 로컬 비전 인코더·멀티모달 processor가 필요하며 KV-free 이미지 경로는 지원하지 않습니다. FP8 구현·설정 가드·벤치마크 precision profile은 로컬 canonical 코드에 반영됐으며, 아래 텍스트 GPU 실측은 당시 SHA의 불변 격리 소스로 수행했습니다. 이후 이미지 지원 변경은 CPU 검증 단계이며 별도 GPU 검증이 필요합니다. BF16을 기본값으로 유지하고 FP8은 명시적으로 선택합니다. 지원 revision·실행 조건·검증 범위는 [FP8 안내](PREFILL_ENGINE_FP8.md)와 [FP8 계획](../../outputs/h200-prefill-performance-20260928/FP8_PLAN.md)을 따릅니다. speculative decoding/MTP, chunked prefill, radix cache와 decode CUDA graph는 꺼야 합니다. 위 명령은 최초 동작 확인을 위해 prefill graph도 끕니다. H200에서는 3B graph4096의 캡처·재생, BF16·FP8 3B·8B dense graph16384의 첫 장시간 실행쌍을 확인했습니다. projection 폭을 늘리는 수치 호환성 옵션도 본 코드에 반영됐습니다. BF16 동적 확률 패리티와 독립 반복 성능 검증은 남아 있으며, 3B·8B FP8의 통과 범위는 아래 현재 상태를 따릅니다. MI325X graph는 미측정입니다. 학습 head는 현재 3B 모델용이며 8B 학습 head는 없습니다. 8B는 해당 체크포인트에서 내보낸 LM 행 readout으로만 시험할 수 있습니다.

## 비교할 때

서버를 한 개씩 교체하면서 측정하려면 [단일 엔드포인트 벤치마크](PREFILL_ENGINE_BENCH.md)의 `prefill_engine_bench.py run`과 `compare`를 사용합니다. 기본 동시성은 **8/16/32/64/128**입니다. 요청·응답 원본, 입력과 실행 설정을 보존하고 같은 조건의 저장 결과를 비교합니다. 서버 시작·종료는 이 도구가 수행하지 않습니다. 순정 SGLang 기준선과 포크 내부 비교의 구분은 [H200 성능 실험 계획](../../outputs/h200-prefill-performance-20260928/BASELINES.md)을 따릅니다.

기존 `benchmarks/prefill_compare.py` 안내는 **같은 엔진을 공유하는 외부 어댑터 두 개**를 순서대로 측정합니다. 전용 워커는 시작할 때 하나의 고정 GPU head를 로드하므로 같은 워커에 두 URL만 붙여 LM 행/학습 head 비교를 할 수 없습니다. LM 행 서버와 학습 head 서버를 각각 종료·시작하면서 같은 요청, 동시성, 모델 revision 조건으로 순차 측정하거나 별도 격리된 장치를 사용해야 합니다. H200 실험용 launcher·stage runner는 `outputs/h200-prefill-performance-20260928`에서 버전별로 관리하며 실행 중 소스를 교체하지 않습니다. 최종 성능 비교는 완료되지 않았습니다. MI325X는 장치 미확보로 미측정입니다. 기존 외부 시제품·공유 GPU·CPU offload 결과를 통합 full-GPU 성능으로 해석하지 마세요.

## 현재 검증 상태 (2026-09-28 · 3B·8B FP8 첫 실행쌍 완료)

- 06:22 KST 명시적 GPU 인계,06:23:43 직접 무점유 확인 후 새 `/opt/h200-prefill-performance-20260928T0624`에서 시작했습니다. 이전 학습·평가·백업 프로세스와 가상환경은 유지합니다.
- 8B base BF16의 같은 dense graph16384·aiohttp·고정 입력 조건에서 첫 30초 이상 실행쌍을 완료했습니다. 실험 후보 `src-wide8192`는 전 C8/16/32/64/128에서 p50 6.9~19.5% 감소, 성공 RPS 8.3~25.4% 증가를 관측했습니다. 1초 간격으로 수집한 최대 VRAM은 stock 35,894 MiB, 후보 35,212 MiB입니다. 새 StandardOne v2.1 가중치의 결과가 아닙니다.
- 같은 8B 후보의 전체 코퍼스 raw-ID 고정 배치 7,180출력은 최대 확률 차이 **4.024416155878896e-8**, argmax 불일치 0으로 통과했습니다. 그러나 동적 HTTP 확률 비교는 엄격한 0.01 게이트를 통과하지 못했고 독립 실행쌍 3회도 남아 있어 최종 승자 판정은 없습니다. stock 자체에서도 반복 응답 변동이 관측됐지만 이를 허용 오차 완화나 통과 근거로 쓰지 않습니다. [동적 패리티 기록](../../outputs/h200-prefill-performance-20260928/DYNAMIC_PARITY.md)을 따릅니다.
- 3B의 초기 HTTPX 탐색은 부하 도구 연결 풀 병목 때문에 최종 비교에 사용하지 않습니다. 이후 BF16 `stock-3b-dense-trial1` / `candidate-3b-corrected-dense-trial1`의 같은 aiohttp·dense graph 첫 실행쌍은 C8/16/32/64/128 모두 30초 이상·오류 0으로 완료·종료했습니다. candidate p50 18.08~43.59%, p95 46.98~49.87%, p99 41.06~64.11% 감소, RPS 31.35~70.25% 증가를 관측했고 성능 구간 VRAM 표본 최대치는 24,484 → 23,626MiB였습니다. 실제 개발셋 고정 raw-ID 최대 Δp 0.02406077, 동적 HTTP 최대 0.04624128·C32 결정 불일치 1건으로 **엄격한 확률 gate는 실패**, winner null입니다. 독립 3회 및 합성 workload HTTP 성능 sweep은 남았습니다. [3B 결과](../../outputs/h200-prefill-performance-20260928/THREE_B_DENSE_RESULTS.md)를 따릅니다.
- FP8의 로컬 canonical 통합은 완료했습니다. 관련 기존·신규 CPU pytest 210개와 readout 검사 11개, 별도 교차 정밀도 품질 도구 검사 11개가 통과했습니다. GPU 실행 소스와 구현 SHA가 일치하며, 로컬 full routing 테스트는 `sgl_kernel`·`triton` 부재로 실행하지 않았습니다.
- 3B base FP8의 stock / candidate width8192 첫 dense graph 실행쌍을 C8/16/32/64/128 모두 30초 이상·오류 0으로 완료·종료·보존했습니다. 같은 FP8 stock보다 candidate p50 **19.76~35.11% 감소**, RPS **28.65~61.68% 증가**를 관측했고 p95/p99도 모든 C에서 낮았습니다. 5 workload raw-ID 총 11,020출력의 최대 Δp **5.7725032e-8**, 동적 HTTP 최대 Δp **5.6340368e-8**, argmax·표시 답 불일치 0으로 엄격한 패리티를 통과했습니다. `status=comparable`, `single_pair_winner=candidate`이나 독립 1쌍이므로 `final_claim_eligible=false`입니다. [3B FP8 결과](../../outputs/h200-prefill-performance-20260928/THREE_B_FP8_RESULTS.md)를 따릅니다.
- 같은 3B candidate의 이전 BF16 width0보다 FP8 width8192의 C8/C16 p50는 **27.56%/28.49% 증가**했습니다. 양자화·projection 폭·실행 시점이 함께 달라 양자화만의 인과 효과로 해석하지 않습니다. BF16 기본값·FP8 opt-in을 유지하며 p50을 우선합니다. 이는 고정 base Ministral3 체크포인트의 결과이며 새 StandardOne v2.1 가중치의 결과가 아닙니다.
- 8B base FP8의 stock / candidate width8192 첫 dense graph 실행쌍도 C8/16/32/64/128 모두 완료·종료·보존했습니다. 같은 FP8 stock보다 candidate p50 **17.38~32.42% 감소**, RPS **21.55~44.73% 증가**를 관측했습니다. candidate p50는 **27.878/45.194/78.658/156.408/302.208ms**, RPS는 **280.684/350.384/403.484/411.912/424.998**입니다. 동적 HTTP 최대 Δp **4.768649486397436e-8**, argmax 불일치 0·표시 답 일치로 엄격한 패리티를 통과했습니다. 비교는 `comparable`, `single_pair_winner=candidate`이고 독립 1쌍이라 `final_claim_eligible=false`입니다. 표본 VRAM 최대는 stock **32,806MiB**, candidate **32,124MiB**였습니다. [8B FP8 결과](../../outputs/h200-prefill-performance-20260928/EIGHT_B_FP8_RESULTS.md)를 따릅니다.
- 3B BF16 width8192 C8/C16 대조도 완료했습니다. p50은21.11/32.42ms로 FP8 27.30/40.80ms보다 낮았지만, BF16 동적 확률 게이트는 실패했습니다. 사전 workload·시간·HTTP pool16대128 차이가 있어 순수 양자화 효과를 확정하지 않습니다. [대조 결과](../../outputs/h200-prefill-performance-20260928/BF16_LOW_C_CONTROL_RESULTS.md)와 [저지연 설계](../../outputs/h200-prefill-performance-20260928/LATENCY_PRIORITY.md)를 따릅니다. 모든 GPU 실행을 종료하고 담당 작업에 반환했으며 후속 실행에는 새 인계와 실제 상태 확인이 필요합니다.
- KV 풀 쓰기·할당은 현재 성능 경로에 남아 있습니다. 8B BF16 실험에서 K/V 합계 8.5 GiB를 할당했습니다. 별도 KV-free 실험 경로는 GPU 검증 전이므로 메모리·성능 개선으로 보고하지 않습니다. 독립 실행쌍 3회, 합성 workload HTTP 성능 sweep, 새 학습 체크포인트, learned head의 새 전체 GPU 비교 및 MI325X 실측도 남아 있습니다.
- 완료된 3B·8B FP8 양쪽 원본은 검증된 로컬 snapshot `outputs/h200-prefill-performance-20260928/remote-snapshots/20260928T010900.243747Z`(884파일·issues0, archive SHA `4d51d18bcc5d0f401fa345865e78a8c25deb8e0cc9b563dc75c55d248661d863`)에 보존했습니다. 3B·8B는 staged benchmark로 양쪽 raw 전체를 다시 비교한 결과가 원격 JSON과 경로 두 필드만 제외하고 일치했습니다. 파일 보존 검증이며 전체 범위 완료를 뜻하지 않습니다. 실행 tag·원시 로그·잔여 범위는 [현재 H200 실행 상태](../../outputs/h200-prefill-performance-20260928/ACTIVE_RUN.md)와 [GPU 인계 기록](../../outputs/h200-prefill-performance-20260928/COORDINATION.md)을 따릅니다. 전체 목표는 미달성 상태입니다.

## 이전 기능 검증 이력 (2026-09-27)

- 어댑터: pytest 233개, 하위 사례 80개 통과. 내부 토크나이저 재사용, 토큰화 횟수, 취소 처리, artifact 내보내기와 HF 캐시 링크 처리를 포함합니다.
- GPU head 연산의 CPU 검증: 5개 통과. GPU 커널 검증은 아닙니다.
- API·프로토콜·전용 모드 설정·가중치 갱신 차단: 75개, 하위 사례 64개 통과. SGLang의 무거운 패키지 초기화를 제외해 로드했으며, 워커를 구동한 통합 테스트는 아닙니다.
- HF 캐시 샤드 링크를 읽는 exporter 수정 관련 테스트 19개가 통과했습니다. 위 어댑터 총계에 포함됩니다. Ministral3의 디코더 중복 생성을 제거한 회귀 테스트는 2개와 하위 사례 2개가 통과했습니다.
- 토크나이저 고정 리비전을 HF 로더의 `revision` 인자로 전달하도록 수정했습니다. 실제 H200 서버의 Python 의존성 환경에서 리비전 회귀 테스트 4개가 통과했습니다.
- 사용자 승인 후 공유 H200에서 3B LM 라벨 head와 학습 head의 기능 검사 각 **7/7**을 통과했습니다. 고정된 5개 raw token ID 입력 한 건에서 LM 라벨 확률의 최대 절대 오차는 **0.0**, 학습 head logit의 최대 절대 오차는 **1.9669532775878906×10⁻⁶**였습니다. 디코더·토크나이저 수정 후에도 CPU offload 없이 3B LM 라벨 head의 **7/7** 통과와 확률 오차 **0.0**을 재확인했습니다. 학습 head는 수정 전 결과입니다. 수정 후 첫 재기동은 학습 메모리 사용 증가로 가용 메모리 2 GiB 가드가 테스트 서버를 중단했으며, 다음 재시도에서 통과했습니다.
- 8B LM 라벨 head는 위 수정 후 기능 검사 **7/7**을 통과했고, 같은 고정 입력 한 건의 확률 최대 절대 오차는 **0.0**이었습니다. **`--cpu-offload-gb 12`를 사용한 기능 검사**이므로 전체 가중치를 GPU에 올린 구성의 지연시간 근거로 사용할 수 없습니다. 8B 학습 head는 시험하지 않았습니다.
- [H200 실행 기록](../../outputs/h200-prefill-smoke-20260927T135703Z/REPORT.md)에 설정과 근거 파일을 정리했습니다. 기존 학습과 GPU를 공유하며 작은 KV 풀, 엔진 동시성 2, CUDA graph 비활성화로 검사했습니다. 당시 미완료였던 항목의 후속 진행은 위2026-09-28 현재 검증 상태를 따릅니다.
