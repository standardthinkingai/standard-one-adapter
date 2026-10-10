# H200 한 장에서 학습 전 판단 성능 비교

추가 학습·LoRA·확률 보정 없이 공식 모델을 실행해 KEV의 공개 평가 문항을 평가합니다. 모델과 SGLang은 서버에서 실행하고, 이 저장소의 어댑터와 벤치 클라이언트는 별도 Python 환경을 사용합니다. 기본 엔진을 수정하지 않습니다.

현재 상태: **H200 실측 완료.** 네 모델 각각 4,006개 요청·4,852개 질문을 오류 없이 처리했습니다. [결과 보고서](../../outputs/h200-baseline-benchmark-2026-09-21/RESULTS.md)에서 정확도와 HTTP 지연을 확인할 수 있습니다. 제공된 모델은 추가 학습 전 공식 지시학습 배포본이며, 사전학습 전용 `*-Base` 모델을 뜻하지 않습니다.

2026-09-21 로컬 검증: pytest 97개와 하위 사례 74개, H200 실행 도구 CPU 테스트 12개 통과. KEV 5종과 JevBench 공개 3종의 총 4,006개 요청·4,852개 질문을 준비했습니다. 합산 숫자는 실행 작업량이며 세트 간 중복을 제거한 독립 평가 문항 수가 아닙니다.

## 모델과 비교 조건

| 프로필 | 공식 모델 | 기본 정밀도 |
|---|---|---|
| `ministral3-3b-bf16` | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | BF16 |
| `qwen35-4b-bf16` | `Qwen/Qwen3.5-4B` | BF16 |
| `qwen36-27b-bf16` | `Qwen/Qwen3.6-27B` | BF16 |
| `qwen36-35b-a3b-bf16` | `Qwen/Qwen3.6-35B-A3B` | BF16 |

원래 다운로드했던 Ministral의 공식 FP8 배포본은 `ministral3-3b-fp8` 선택형 프로필입니다. 기본 비교는 BF16으로 맞추며, FP8 결과는 별도 행으로 비교합니다. 모델 리비전, 엔진 리비전과 실행 옵션은 [h200/models.json](h200/models.json)에 고정했습니다. H200 1장, TP=1, 컨텍스트 8,192, 입력 절단 없음, speculative decoding/MTP 없음, 비전 인코더 유지 조건입니다. 최대 동시 요청은 32이며 기본 클라이언트 동시성은 1입니다.

공식 지시학습 모델의 chat template에 `enable_thinking=False`를 전달하고, 선택지 라벨의 다음 토큰 확률을 읽습니다. 문장을 생성하지 않으며 `output_tokens=0`을 검증합니다. 질문마다 별도 프리필을 실행하므로 여러 질문을 하나의 특수 pointer head로 처리하는 KEV와 실행 구조는 다릅니다. 문항과 지표를 맞춘 **학습 전 어댑터 기준선**이며 KEV의 학습 결과나 기존 기본 모델 probe의 프롬프트를 그대로 복제한 실험은 아닙니다.

Ministral은 SGLang 0.5.20의 native chat 문자열 재인코딩 오류를 피하기 위해
어댑터에서 공식 토크나이저의 `tokenize=True` 결과를 전달합니다. 전체 4,852문항의
토큰 경계와 최대 입력 길이(4,037토큰)를 CPU에서 확인했습니다. Qwen은 엔진의
HTTP 토큰화 경로를 사용합니다. 따라서 모델 간 HTTP 지연에는 토큰화 위치와
호출 횟수 차이가 포함되며, 순수 GPU 프리필 시간 비교가 아닙니다.
초기 Ministral HTTP 점검은 `invalidated.json`으로 무효 표시하며 비교에서 제외합니다.

## 준비된 KEV 데이터

원본 커밋은 [`4f8110a3f8620cc3a182ae9a708e4398492c4b1a`](https://github.com/jaredpalmer/kev/tree/4f8110a3f8620cc3a182ae9a708e4398492c4b1a)입니다. 원본 manifest와 JSONL의 SHA256을 검증하고, 정답·출처·변형 정보를 모델 요청에서 제거합니다. 모든 질문을 유지하며 Banking77의 78개 선택지 변형도 포함합니다.

| 개발 세트 | 요청 수 | 전체 질문 | 기본 정확도에 포함되는 질문 |
|---|---:|---:|---:|
| `decision-v7` | 1,204 | 1,468 | 1,264 |
| `transfer-v4` | 764 | 764 | 656 |
| `transfer-v9` | 1,264 | 1,264 | 1,046 |
| `semif-v1` — 선택형 | 252 | 252 | 144 |
| `scienthoon-v1` — 선택형 | 291 | 873 | 873, 작업별 구분 필요 |

`decision-v7`은 KEV 학습에 사용된 출처의 평가 문항, `transfer-v4`는 KEV 미세조정에서 사용하지 않은 출처·정책 구조입니다. 이는 원래 Qwen/Mistral 사전학습에서 보지 않았다는 보장은 아닙니다. `transfer-v9`는 v4에 MMLU-Pro·긴 문맥·근거 제거 문항을 더한 것이므로 두 결과를 독립 데이터처럼 합산하지 않습니다.

모델카드의 정확도와 비교할 값은 `report.json`의 `suites.<suite>/development.clean.acc`입니다. 순서 변형과 none-of-the-above 문항은 `variants`, 조건 반전은 `paired_flip`, 선택지 순서 영향은 `permutation`에 기록합니다. v9의 근거 제거 110문항은 정확도에서 제외하고 `unknowable`의 과신 비율로 평가합니다.

SemIf는 원래 가족별 balanced accuracy를 사용하므로 여기의 clean micro accuracy를 원래 지표와 혼동하지 않습니다. Scienthoon의 원본 900개 질문 행은 KEV 변환 과정에서 291개 고유 상태·873개 질문이 됐습니다. priority 라벨은 입력에 없는 조직 규칙에 의존하므로 queue·angry와 분리해 `tasks`를 확인합니다. 두 외부 세트도 KEV가 이미 평가한 데이터입니다.

## JevBench 공개 데이터 — 기본 실행에 추가

[JevBench](https://github.com/fstandhartinger/jevbench/tree/fd51755eb0c0b546ca206d764faf3302feca913e)는 JEV 계열 판단 모델용 커뮤니티 벤치입니다. TypeSafe 공식 벤치는 아닙니다. 전체 결과표의 534문항 중 실제 공개된 231문항을 커밋과 SHA256으로 고정했습니다.

| 공개 세트 | 문항 | 주요 평가 |
|---|---:|---|
| `jevbench-original` | 72 | 정책, 라우팅, 의도, 등급, 추출, 답변 적합성 |
| `jevbench-easy` | 48 | 명확한 사실·의도·도구 선택 |
| `jevbench-hard` | 111 | 복합 규칙, 시간·수량, 확률, 모호성, 함정, 답변 판단 |

정답과 해설·생성 근거는 모델 입력에 넣지 않습니다. 데이터는 `public.jsonl`로 저장하며 테스트나 비공개 문항으로 부르지 않습니다. 확률을 정확히 계산할 수 있는 hard 10문항은 `reference_distribution`에서 확률 분포 오차를 별도로 측정합니다. 최고 확률 동률은 JevBench 원본처럼 라벨 사전순으로 처리합니다. 기존 KEV 지표와 공개 문항별 지표를 기록하며, JevBench의 전체 534문항 종합 점수나 네트워크 지연 보정 점수는 재현했다고 주장하지 않습니다. 확률 출력 방식도 `token-logprob-derived`로 구분합니다.

평가 선정 기준은 **정책·분류·근거 부족·다중 질문 판단과 확률 품질**이며 비전 여부는 조건이 아닙니다. 기본 준비 검사는 텍스트만 사용하고, `--probe-vision`을 추가했을 때만 이미지도 점검합니다. Typed Decisions는 작은 교사 모델과의 일치도를 측정하므로 보조 후보입니다. 추가 후보와 선정 근거는 [PUBLIC_DATASETS.md](PUBLIC_DATASETS.md), [DECISION_BENCHMARK_SELECTION.md](DECISION_BENCHMARK_SELECTION.md)에 있습니다.

## H200 서버에서 실행

이 저장소를 서버로 옮긴 뒤 저장소 루트에서 실행합니다. 모델 가중치는 실행 시 Hugging Face에서 받으며, 네 모델을 동시에 GPU에 올리지 않습니다. 전체 BF16 체크포인트 캐시를 위해 충분한 로컬 디스크 공간을 확보합니다. 엔진 설치 조건과 CUDA 13 환경은 [h200/README.md](h200/README.md)를 따릅니다.

어댑터 환경:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,native-tokenizer]'
```

평가 데이터만 다운로드합니다. 함께 전달된 `benchmarks/data`가 있다면 이 단계는 다시 실행해도 동일 파일인지 확인합니다.

```bash
.venv/bin/python -m jev_adapter.benchmarks.prepare \
  --output benchmarks/data \
  --suite decision-v7 --suite transfer-v4 --suite transfer-v9 \
  --suite jevbench-original --suite jevbench-easy --suite jevbench-hard \
  --suite semif-v1 --suite scienthoon-v1
```

오프라인에서는 같은 커밋의 KEV 체크아웃을 `--source-root /path/to/kev`, JevBench 체크아웃을 `--jevbench-source-root /path/to/jevbench`로 지정할 수 있습니다. 학습 데이터는 다운로드하지 않습니다.

먼저 네 모델의 실행 명령만 확인합니다. `--execute`가 없으면 모델을 받거나 실행하지 않습니다.

```bash
.venv/bin/python benchmarks/run_matrix.py \
  --engine-python .venv-sglang/bin/python \
  --output benchmarks/results/preview --include-external
```

서버 통합 점검은 소량 문항으로 진행합니다. 이 결과는 정확도 비교용으로 쓰지 않습니다.

```bash
.venv/bin/python benchmarks/run_matrix.py \
  --engine-python .venv-sglang/bin/python \
  --output benchmarks/results/smoke \
  --limit 8 --warmup 2 --execute
```

사전 점검이 통과하면 네 모델을 순서대로 평가합니다. 각 모델의 텍스트 입력, 토큰 경계, 0토큰 출력, 모델 리비전과 캐시 설정을 먼저 검증하고, 모델마다 어댑터를 다시 시작합니다. 기존 프로세스가 포트를 사용하면 중단합니다. JevBench만 실행할 때는 `--suite jevbench-original --suite jevbench-easy --suite jevbench-hard`를 지정하세요.

```bash
.venv/bin/python benchmarks/run_matrix.py \
  --engine-python .venv-sglang/bin/python \
  --output benchmarks/results/bf16-c1 \
  --include-external --execute
```

동시성 8에서의 처리량도 보려면 별도 출력 폴더로 실행합니다. 동시성 1의 지연시간과 구분해서 해석합니다.

```bash
.venv/bin/python benchmarks/run_matrix.py \
  --engine-python .venv-sglang/bin/python \
  --output benchmarks/results/bf16-c8 \
  --concurrency 8 --include-external --execute
```

`--profile qwen36-35b-a3b-bf16`처럼 모델 하나만 선택할 수 있습니다. `--repeats 3`은 지연 측정을 반복하고 정확도는 첫 반복만 사용합니다. 모든 결과 디렉터리는 새 경로여야 하며 기존 결과를 덮어쓰지 않습니다.

## Qwen3.5-4B 토크나이저·GDN 프리필 A/B

Qwen의 현재 기준선은 엔진 HTTP 토크나이저입니다. 로컬 토크나이저와 FlashInfer GDN
프리필은 **명시적으로 선택하는 실험 옵션**이며, 위 기본 행렬은 바뀌지 않습니다.
[오프라인 검증](../QWEN_TOKENIZER_VALIDATION.md)은 저장된 Qwen 프롬프트 4,852개의
로컬 토크나이저 해시·길이를 확인했습니다. 당시 엔진 HTTP 토큰 ID 원본은 보관되지
않았으므로, 두 경로의 **실제 ID 일치**는 아래 실서버 검사로 확인해야 합니다.

### 1. 토크나이저 경로만 비교

H200 한 장에서 [고정 Qwen 프로필](h200/models.json)의 엔진을 하나만 실행합니다.
엔진을 띄운 터미널은 계속 열어 두고, 다른 두 터미널에서 동일한 엔진을 바라보는
어댑터를 각각 시작합니다. 두 어댑터의 프롬프트 문구는 `served`, 온도는 1.0,
선택지 회전은 1회로 맞춥니다.

```bash
.venv/bin/python benchmarks/h200/runtime.py launch qwen35-4b-bf16 \
  --engine-python .venv-sglang/bin/python --gpu 0 \
  --output benchmarks/results/qwen-tokenizer-ab/engine-triton --execute
```

```bash
# 터미널 2: 기존 엔진 HTTP 토크나이저
.venv/bin/python -m jev_adapter --engine-url http://127.0.0.1:30000 \
  --model decision-model --port 30120 --max-concurrency 32 \
  --prompt-wording served
```

```bash
# 터미널 3: 같은 엔진, 고정 리비전의 로컬 토크나이저
.venv/bin/python -m jev_adapter --engine-url http://127.0.0.1:30000 \
  --model decision-model --port 30220 --max-concurrency 32 \
  --prompt-wording served \
  --tokenizer-model Qwen/Qwen3.5-4B \
  --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

네 번째 터미널에서 동일한 준비 데이터 목록을 사용합니다. 먼저 두 경로가
`/generate`에 넘길 **정확한 입력 토큰 ID**를 질문별로 비교합니다. 이 검사는
엔진의 채팅 토큰화 API와 로컬 공식 토크나이저를 직접 사용하며 지연 측정에
포함하지 않습니다. `status=matched`가 아니면 지연 비율을 해석하지 마세요.
엔진에 인증을 설정했다면 이 직접 검사는 `SGLANG_API_KEY`를 사용합니다.
어댑터 API 인증을 설정했다면 아래 응답 비교 클라이언트에는 별도로 `JEV_API_KEY`를
설정하세요.

```bash
DATA=(
  --data benchmarks/data/decision-v7/development.jsonl
  --data benchmarks/data/transfer-v4/development.jsonl
  --data benchmarks/data/transfer-v9/development.jsonl
  --data benchmarks/data/jevbench-original/public.jsonl
  --data benchmarks/data/jevbench-easy/public.jsonl
  --data benchmarks/data/jevbench-hard/public.jsonl
)
.venv/bin/python benchmarks/qwen_tokenizer_parity.py "${DATA[@]}" \
  --output benchmarks/results/qwen-tokenizer-ab/exact-ids.json
```

그다음 기존 [`prefill_compare.py`](prefill_compare.py)를 재사용합니다.
`sglang`은 HTTP 토크나이저 어댑터, `candidate`는 로컬 토크나이저 어댑터입니다.
같은 요청·순서를 보내며 엔진은 공유하고 두 측정 구간을 차례로 실행합니다.
`--require-parity`는 요청별 입력 토큰 수와 첫 반복의 선택 결과·선택지 조건부
확률을 검사합니다. 기본 확률 허용 오차는 0.01입니다. 이 검사는 앞의 정확한
토큰 ID 검사와 함께 통과해야 합니다.

```bash
.venv/bin/python benchmarks/prefill_compare.py "${DATA[@]}" \
  --sglang-url http://127.0.0.1:30120 \
  --candidate-url http://127.0.0.1:30220 \
  --model decision-model --concurrency 1 --warmup 32 --repeats 3 \
  --require-parity \
  --output benchmarks/results/qwen-tokenizer-ab/paired-c1
```

`paired-c1/report.json`의 `backends.*.request_latency_ms.p50/p95`,
`quality_parity`, `input_token_parity`를 함께 확인합니다. 동시성 8·32 처리량은
별도 새 출력 디렉터리에서 `--concurrency 8 --concurrency 32`로 실행하세요.
`exact-ids.json`의 `preparation_ms`는 HTTP 토큰화와 로컬 토큰화 준비 구간의
참고용 p50/p95입니다. 순차적으로 HTTP 경로를 먼저 재므로 성능 결론은 위의
교대 순서 전체 응답 비교에서 내리세요.
두 서비스의 입력 토큰 수와 품질이 일치하더라도, 응답 지연은 토큰화·HTTP·엔진
스케줄링·프리필의 합계입니다. `/generate`만의 시간이나 GPU 커널 시간으로
해석하지 않습니다.

### 2. 엔진 GDN 프리필 옵션만 비교

토크나이저 A/B가 통과한 뒤 두 어댑터와 엔진을 종료합니다. 다음 실행은 두 번
모두 **로컬 토크나이저**를 사용하고, 두 번째 실행에서만 SGLang의
`--linear-attn-prefill-backend flashinfer`를 켭니다. `run_matrix.py`가 엔진
실행 명령과 리비전을 각 결과의 `engine/launch.json`에 저장합니다.

```bash
.venv/bin/python benchmarks/run_matrix.py \
  --profile qwen35-4b-bf16 --qwen-local-tokenizer \
  --concurrency 1 --warmup 32 --repeats 3 \
  --output benchmarks/results/qwen-gdn-triton --execute

.venv/bin/python benchmarks/run_matrix.py \
  --profile qwen35-4b-bf16 --qwen-local-tokenizer \
  --linear-attn-prefill-backend flashinfer \
  --concurrency 1 --warmup 32 --repeats 3 \
  --output benchmarks/results/qwen-gdn-flashinfer --execute
```

FlashInfer 실행의 `engine/preflight.json`에서 실제
`linear_attn_prefill_backend=flashinfer`를 확인합니다. 두 실행의 각 `c1` 결과는
아래 명령으로 요청 SHA, 입력 토큰 수, 선택 결과, 확률 차이를 대조한 뒤
p50/p95를 비교하세요. `engine/launch.json`의 명령은 FlashInfer 플래그만
달라야 합니다.

```bash
.venv/bin/python benchmarks/compare_saved_runs.py \
  benchmarks/results/qwen-gdn-triton/qwen35-4b-bf16/c1 \
  benchmarks/results/qwen-gdn-flashinfer/qwen35-4b-bf16/c1 \
  --output benchmarks/results/qwen-gdn-parity.json
```

결과가 다르거나 FlashInfer 시작·준비 검사에 실패하면 기존 Triton 결과를
기준선으로 유지합니다. 이 단계는 엔진을 재시작하므로 토크나이저 A/B와 별도
실험으로 기록합니다.

## 결과와 측정 범위

### 기존 어댑터·전용 프리필·학습 헤드 비교

기존 어댑터와 전용 프리필 어댑터를 서로 다른 포트에서 띄우고, **같은 SGLang 엔진 프로세스와 체크포인트**를 바라보게 하세요. 두 서비스의 정밀도·프롬프트 형식·캐시 정책도 같아야 합니다. 준비된 JSONL을 두 `POST /v1/systemone` URL에 그대로 보내 비교합니다. H200과 MI325X에서 같은 명령을 각각 실행하고 장치별 결과 폴더를 사용하세요. 모델 로드와 서비스 시작은 이 도구가 수행하지 않습니다.

```bash
.venv/bin/python benchmarks/prefill_compare.py \
  --sglang-url http://127.0.0.1:30120 \
  --candidate-url http://127.0.0.1:30220 \
  --data benchmarks/data/jevbench-original/public.jsonl \
  --data benchmarks/data/jevbench-easy/public.jsonl \
  --data benchmarks/data/jevbench-hard/public.jsonl \
  --model decision-model \
  --output benchmarks/results/prefill-h200
```

기본 동시성은 `1, 8, 32`, 각 단계의 서비스별 준비 요청은 32회입니다. `--concurrency 1 --concurrency 8 --concurrency 32`, `--warmup`, `--repeats`, `--seed`, `--limit`으로 바꿀 수 있습니다. 동일한 입력·순서·요청 옵션을 사용하고 두 서비스를 차례로 측정합니다. 매 단계의 선행 서비스는 교대합니다. 준비 요청과 첫 번째 측정 이후 반복 요청은 품질 지표에서 제외합니다. 냉 프리필 비교에서는 엔진을 `--disable-radix-cache`로 띄우고 `/server_info`의 `disable_radix_cache=true`를 확인한 뒤, 다른 GPU 작업을 멈춘 상태에서 측정하세요. 벤치 클라이언트도 측정 요청의 `cached_tokens>0`을 발견하면 실패 처리합니다. 두 서비스의 프롬프트 템플릿·시스템 프롬프트와 모델 리비전도 실행 전에 직접 확인해야 합니다.

학습 헤드 어댑터까지 비교하려면 `--learned-url http://127.0.0.1:30320`을 추가합니다. 이 경우 세 서비스 모두 같은 선택 문항을 받으며, `c1-learned.jsonl`과 `report.json`의 `backends.learned`에 학습 헤드의 p50/p95·판단 수/초를 기록합니다. `learned_vs_sglang`에는 세트별 정확도/Brier/NLL과 기준선 대비 차이를 기록합니다. 학습으로 결과가 달라지는 것은 정상이며 `--require-parity`는 **기존 어댑터와 전용 프리필 경로에만** 적용합니다. 학습 헤드가 별도 엔진을 쓴다면 같은 GPU 모델과 동일한 부하 조건에서 차례로 측정하세요.

현재 3B 학습 헤드에서 실제 감독된 표시 라벨은 A–C뿐입니다. 4개 이상 선택지도 실험용 경로에서 계산하지만 D 이후의 품질은 보장하지 않습니다. 학습 범위 안에서 세 경로를 비교하려면 `--max-options 3`을 **명시적으로** 사용해 모든 질문이 3개 이하의 선택지를 가진 기록만 선택하세요. 제외된 기록 수는 manifest의 `excluded_records_by_max_options`에 남습니다. 이때 정확도는 선택한 부분 집합의 결과입니다. 필터 없이 실행하면 4개 이상 선택지의 품질 저하도 측정하며, 어떤 기록도 조용히 제외하지 않습니다.

`report.json`에 요청 왕복 p50/p95, 완료한 판단 수/초, 응답이 보고한 논리적 입력 토큰 수/초, 오류 수, 선택 결과 일치율, 확률 차이, 세트별 정확도·Brier·NLL을 기록합니다. `input_token_parity`는 반복·기록별 입력 토큰 수를 대조합니다. 하나라도 다르면 실행 상태가 `failed`, 품질 비교가 `incomparable`이며 상대 지연·처리량 비율을 출력하지 않습니다. 토큰 수가 같아도 프롬프트 내용까지 같다는 증거는 아니므로 서비스 설정 점검은 여전히 필요합니다. 기존 어댑터와 전용 프리필 경로의 비교 대상은 반환된 **선택지 조건부 확률**이며 전체 어휘 logprob의 일치 여부는 측정하지 않습니다. 요청은 모든 경로에서 `return_logprobs=false`로 고정합니다. `c1-sglang.jsonl` 같은 파일에 요청별 정규화 결과와 오류를 남깁니다. 오류가 있으면 해당 비교의 품질 지표는 `incomplete`입니다. 확률 허용 오차 기본값은 `0.01`이며 `--probability-tolerance`로 조정할 수 있습니다. 두 기본 URL의 인증 키는 각각 `JEV_SGLANG_API_KEY`, `JEV_CANDIDATE_API_KEY`, 학습 헤드 URL의 키는 `JEV_LEARNED_API_KEY` 환경 변수에서 읽고, 없으면 `JEV_API_KEY`를 사용합니다. 키 값은 결과에 저장하지 않습니다.

입력 토큰/초는 응답의 `usage.input_tokens` 합계이며 캐시 적용 전의 논리적 작업량일 수 있습니다. HTTP 지연에는 토큰화·스케줄링·프리필이 포함됩니다. GPU 커널만의 지연이나 실제 메모리 대역폭 수치는 아닙니다.

### 기존 H200 평가 행렬

각 모델 아래 `engine/launch.json`, `engine/preflight.json`, `engine/engine.log`, `c1/manifest.json`, `c1/predictions.jsonl`, `c1/report.json`이 생성됩니다. 전체 실행이 성공하면 `comparison.csv`에 모델·세트별 결과를 모읍니다.

- 정확도: clean 질문별 micro accuracy, 출처·작업별 정확도, 작업별 macro accuracy.
- 확률 품질: multiclass Brier, NLL, 10-bin ECE, 최대 확률 0.9 이상 오답 비율, 경험적 오류 예산별 coverage. 어댑터의 entropy confidence 대신 **최대 선택지 확률**로 ECE를 계산합니다.
- 순서·정책 반전: 라벨 정렬 후 flip rate, 최소 대조쌍 양쪽 정답률. 부분 집합의 누락된 짝은 별도 집계합니다.
- 시간: 요청별 HTTP 왕복 p50/p95/p99, 어댑터 내부 벽시계 시간, 전체 요청·질문 처리량, 입력 토큰 수. 모델 로딩과 20회 준비 요청은 제외합니다.

지연시간에는 HTTP, 토큰화, 엔진 스케줄링과 프리필이 포함됩니다. **순수 GPU 커널 시간이나 프리필 FLOPS 측정값이 아닙니다.** 질문 수와 입력 토큰 수도 함께 보세요. 어댑터는 질문마다 엔진 HTTP 호출 4회를 사용합니다. 정확도 평가를 위해 지정한 라벨 logprob를 반환받는 비용도 포함됩니다.

기본 실행은 radix cache와 이미지 전처리·비전 특징 재사용을 끕니다. 반복 입력의 캐시 효과가 섞이지 않는 조건입니다. 반대로 실제 서비스의 캐시 효과를 측정하려면 별도 엔진 설정과 `--cache-mode server-default`의 개별 runner 실행으로 기록해야 합니다.

오류·시간 초과·확률 누락·출력 토큰 발생 시 재시도나 문항 제외로 성공률을 높이지 않습니다. 실패한 실행은 `status=failed`로 저장하고 기본 정확도를 출력하지 않습니다. 데이터 SHA256·부분 집합 여부·코드 SHA256·실행 옵션·엔진/모델 정보를 기록합니다. 실패한 실행은 비교 CSV에 포함하지 않습니다.

개발 세트로 조건을 정한 뒤 최종 확인에만 test를 사용합니다. test 준비와 개별 실행은 모두 `--allow-test`를 명시해야 합니다. 행렬 실행기는 KEV 개발 세트와 JevBench 공개 세트를 사용합니다. 학습·프롬프트 선택·온도 보정 없이 현재 고정 조건을 그대로 비교하는 것이 이 기준선의 목적입니다.

```bash
.venv/bin/python -m pytest
.venv/bin/python -m unittest discover -s benchmarks/h200 -p 'test_*.py'
.venv/bin/ruff check .
```

지표 수식의 원본 대조 결과와 재현 방법은 [METRIC_VALIDATION.md](METRIC_VALIDATION.md)를 참고하세요. 공개 데이터마다 원래 라이선스가 다르며 변환 manifest가 해당 출처를 보존합니다. JEV 유료 API 호출이나 학습은 실행 과정에 포함되지 않습니다.
