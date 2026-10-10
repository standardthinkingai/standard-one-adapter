# 단일 엔드포인트 프리필 성능 측정

`prefill_engine_bench.py run`은 이미 실행 중인 `/v1/systemone` 서버 **하나**를 측정합니다. 서버를 시작하거나 종료하지 않습니다. 기본 동시성은 8·16·32·64·128이고, 각 수준에서 워밍업을 제외한 closed-loop 요청을 보냅니다. 클라이언트 지연은 작업자가 요청을 받은 직후부터 HTTP 응답 완료까지 측정합니다. HTTP 연결 풀은 최대 동시성 이상으로 설정합니다. 측정 요청 수와 워밍업 요청 수는 각각 최대 C 이상이어야 합니다(워밍업 0은 허용).

`--http-client httpx`가 기존 기본값입니다. `--http-client aiohttp`는 공용 `jev_adapter.http_client.AiohttpCompatClient`를 사용하며 응답을 전부 읽은 뒤 HTTPX 응답 형태로 반환하므로 기존 응답·확률·품질 검증을 유지합니다. `--http-max-connections`를 생략하면 기존처럼 최대 C를 사용합니다. 명시할 때는 1~128의 정수이고 선택한 최대 C 이상이어야 합니다. 예를 들어 전체 sweep과 C8/C16 대조 모두 `--http-max-connections 128`을 지정하면 같은 풀 크기를 유지합니다. aiohttp는 전체/호스트별 연결 한도를 모두 이 값으로, HTTPX는 전체/keepalive 한도를 모두 이 값으로 고정합니다. aiohttp는 환경 proxy를 사용하지 않고 `--timeout`을 전체 요청 시간에 적용합니다. HTTPX의 timeout은 기존처럼 네트워크 작업별 한도입니다. 기준선과 후보는 같은 client·버전·설정을 사용해야 합니다.

벤치가 생성한 client는 `manifest.http_pool_max_connections`와 `manifest.http_client.max_connections`에 같은 실제 연결 한도를 기록합니다. 저장 결과를 읽을 때 필드가 있으면 유효 범위와 owned client descriptor의 일치를 검사하고, 비교할 때 풀 크기가 다르면 거부합니다. 양쪽 이전 manifest에서 해당 필드가 없으면 기존 자료를 읽을 수 있습니다. 실제 client 종류·클래스·transport·라이브러리 버전·timeout 정책·helper 소스 SHA도 기록합니다. aiohttp helper SHA는 `client_code_sha256`에도 포함합니다. 테스트나 진단 코드가 client를 직접 주입하면 `external`과 실제 클래스명을 기록하고 내부 연결 설정·버전을 추정하지 않습니다. 이 경우 명시적 `http_max_connections` override는 실제 풀을 강제할 수 없어 거부합니다. 풀 크기를 같게 지정해도 서로 다른 C 배열·client source SHA를 같은 조건의 엔진 비교로 허용하지는 않습니다.

먼저 아래 형식의 설정 JSON을 서버 실행 기록에서 만듭니다. `head_mode=logprobs`이면 `head_artifact`는 `null`이고, `lm_rows` 또는 `learned`이면 실제 GPU readout 파일 경로와 그 파일의 SHA256을 적습니다. 벤치는 readout NPZ 안의 모드·라벨·토큰 ID·모델 경로·revision을 설정과 대조합니다. NPZ에는 dtype 필드가 없으므로 activation/head dtype은 실제 `/server_info.dtype`로 확인하고, 본체 양자화는 `/server_info.quantization`과 설정의 `precision_profile`로 별도 확인합니다. `labels`와 `token_ids`는 실행 모델에서 검증한 배열이어야 합니다. 예시의 토큰 ID와 해시는 설명용이므로 실제 감사 결과로 바꾸세요.

```json
{
  "schema_version": 1,
  "hardware": "H200",
  "model_path": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
  "model_revision": "b6d637bef2393152b3da2b2fde72eecdee30557e",
  "precision": "bfloat16",
  "prompt_contract": "served-v1",
  "label_contract": "single-token-A-C-v1",
  "labels": ["A", "B", "C"],
  "token_ids": [11, 12, 13],
  "head_mode": "lm_rows",
  "head_artifact": {
    "path": "/models/ministral3-3b-gpu-readout.npz",
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  }
}
```

실제 입력 토큰 **ID**가 같은지 검증한 외부 감사 파일을 `--token-audit`에 전달할 수 있습니다. 감사 도구는 각 요청의 모든 질문에 대한 입력 토큰 ID를 고정 순서로 직렬화해 해시해야 합니다. 이 파일이 없으면 비교 결과는 성능 비율을 보여도 승자를 표시하지 않습니다. 두 실행의 요청별 토큰 ID 해시가 다르면 비교를 오류로 중단합니다. 프롬프트 계약 이름이나 토큰 개수만으로 ID 일치를 주장하지 않습니다.

`prefill_token_audit.py`는 `CUDA_VISIBLE_DEVICES=''`에서 실제 고정 토크나이저와 두 백엔드의 입력 경로를 실행해 감사 파일을 만듭니다. 엔진 응답은 CPU 기록용 fake transport/manager가 제공하므로 GPU 패리티 검사는 아닙니다. H200 서버에서 3B·8B 실제 토크나이저로 생성한 고정 입력·감사 파일은 [CPU 준비 결과](../../outputs/h200-prefill-performance-20260928/CPU_PREP.md)에 있습니다. 측정과 감사의 `--data`·`--model`·`--limit` 선택은 같아야 합니다.

```json
{
  "schema_version": 1,
  "workload_sha256": "<run에서 선택한 요청 배열의 SHA256>",
  "model_revision": "b6d637bef2393152b3da2b2fde72eecdee30557e",
  "prompt_contract": "served-v1",
  "label_contract": "single-token-A-C-v1",
  "requests": [
    {
      "request_sha256": "<첫 요청의 정규화 JSON SHA256>",
      "input_token_ids_sha256": "<첫 요청의 실제 입력 토큰 ID 배열 SHA256>"
    }
  ]
}
```

```bash
.venv/bin/python benchmarks/prefill_engine_bench.py run \
  --base-url http://127.0.0.1:30000 --model decision-model \
  --settings-manifest /path/to/h200-lmrows-settings.json \
  --token-audit /path/to/fixture-token-audit.json \
  --data benchmarks/data/semif-v1/development.jsonl \
  --limit 252 --repeats 16 --warmup 128 \
  --output benchmarks/results/h200-lmrows-trial1
```

위 예시 데이터는 최대 3개 선택지만 사용합니다. 실제 최종 비교에는 동일한 입력 길이 구간으로 준비한 코퍼스와 각 실행의 측정 구간 30초 이상을 확인하세요.

실제 판단 정확도와 자연 입력 성능을 함께 살펴볼 때는 준비된 `../outputs/h200-prefill-performance-20260928/cpu-prep/workloads-frozen/real-development.jsonl`을 `--data`로 쓰고 `--limit 4096`을 지정해 선택된 1,436건 전체를 포함합니다. 기본 `--limit 512`는 일부만 선택합니다. 합성 길이 버킷은 성능·입력 모양 검사에 사용하며 일반 판단 정확도에 합산하지 않습니다.

서버를 별도로 교체한 뒤 동일한 데이터·옵션으로 새 출력 디렉터리에 다시 `run`을 실행합니다. `--engine-url`은 측정 URL과 `/model_info`·`/server_info`를 제공하는 엔진 URL이 다를 때 지정합니다. `--repeats`는 **한 실행 안에서** 같은 코퍼스를 반복하는 횟수입니다. 기본 `--min-measure-seconds 0`은 탐색용이며 최종 측정에는 `30`을 지정합니다. 각 C의 측정 구간이 30초보다 짧으면 `report.json`에 실제 초와 `MeasurementTooShort`가 기록되고 비교가 거부됩니다. 최종 성능 주장에는 각 30초 이상의 독립 실행 쌍 3개 이상이 필요하며, 이 도구의 단일 비교 결과는 최종 통계적 결론이 아닙니다.

처리량이 낮은 C의 실행 시간을 불필요하게 늘리지 않으려면 `--repeats-by-concurrency '{"8":10,"16":14,"32":20,"64":25,"128":25}'`로 각 C의 반복 횟수를 미리 정할 수 있습니다. 선택한 모든 C의 문자열 키를 정확히 한 번씩 지정하고 값은 양의 정수여야 합니다. 이 옵션이 없으면 기존 `--repeats`를 모든 C에 적용합니다. 서로 다른 C는 표본 수가 달라도 되지만 **같은 C의 기준선·후보와 독립 회차들은 동일한 반복 수·요청 순서**를 사용해야 합니다. 각 C가 최소 측정 시간을 충족해야 하며, 한 sweep의 측정 요청 합계는1,000,000개 이하여야 합니다. 이 합계와 C별 요청 수가 C 이상인지 작업 배열 생성 전에 검사합니다.

manifest의 `repeats_by_concurrency`와 `job_order_sha256_by_concurrency`가 실제 C별 반복 수·순서를 기록합니다. 기본 전역 반복을 사용하는 새 실행도 이 정보를 저장하며 기존 `repeats`와 단일 `job_order_sha256`는 유지합니다. C마다 순서가 다르면 단일 order 값은 `null`입니다. 이전 manifest는 기존 전역 값에서 C별 값을 복원해 읽습니다. 비교와 반복 요약은 실제 C별 값의 일치를 요구하며, C별 옵션을 쓴 경우 사용하지 않은 전역 fallback 값 차이는 결과 비교에 영향을 주지 않습니다.

128개 고정 버킷이라면 다음처럼 요청 순서를 사전에 고정합니다. 기본 C 5개에서 128 × 1,000 × 5 = 640,000 측정 요청으로, 총 1,000,000개 제한 안에 듭니다. 기준선과 후보에 **같은** `--repeats`를 사용하세요. 짧게 끝나면 양쪽 모두 반복 횟수를 늘려 새로 실행합니다. 벽시계에 따라 한쪽에만 추가 요청을 보내지 않습니다.

```bash
.venv/bin/python benchmarks/prefill_engine_bench.py run \
  --base-url http://127.0.0.1:30000 --model decision-model \
  --settings-manifest /path/to/settings.json \
  --token-audit /path/to/bucket-token-audit.json \
  --data /path/to/prepared-bucket-128.jsonl \
  --limit 128 --repeats 1000 --warmup 128 \
  --min-measure-seconds 30 \
  --output /path/to/trial1
```

```bash
.venv/bin/python benchmarks/prefill_engine_bench.py compare \
  benchmarks/results/h200-logprobs-trial1 \
  benchmarks/results/h200-lmrows-trial1 \
  --output benchmarks/results/h200-trial1-comparison.json
```

`run`은 모델 revision·정밀도·캐시 비활성화·TP/PP/DP=1·CPU offload=0을 엔진 정보로 확인하고, 모든 응답에서 유한한 확률, 출력 토큰 0, 캐시 토큰 0, 실제 서비스 한도 `metadata.service_max_concurrency >= C`를 요구합니다. 각 동시성의 `raw_outcomes.jsonl`에는 전체 요청·원문 응답·HTTP 상태·측정 시간과 POST 구간이 남습니다. POST의 실제 겹침 최고치가 C보다 작으면 실행이 실패합니다. 이 값은 클라이언트가 보낸 HTTP 동시 호출 수이며, GPU 내부 실행 배치 크기를 증명하지 않습니다.

`compare`는 동일한 C 배열, 데이터 SHA256, 요청 순서, 요청별 입력 토큰 수, 설정/라벨 배열, 엔진 리소스·graph·attention 설정을 확인한 뒤에만 비율을 계산합니다. 선택한 원본 항목과 각 원문 응답에서 품질 행을 다시 계산하고, 비학습 head끼리는 확률뿐 아니라 표시된 choice·score·confidence도 비교합니다. 학습 head의 정확도는 별도로 보고하며 그 결과만으로 성능 승자를 선언하지 않습니다.

각 C의 품질 지표는 반복 요청을 독립 정확도 표본으로 세지 않도록 **첫 측정 pass만** 사용합니다. `quality_overall`과 `quality_by_type`의 `choice`·`noul`·`score`는 기존 `jev_adapter.benchmarks.metrics.metrics`의 정확도(`acc`), NLL, Brier, ECE 등을 그대로 담고, `score`에는 기대 등급 MAE(`score_mae`)와 ranked probability score도 포함합니다. `quality_by_suite`도 함께 유지합니다. 주집계는 `variant=clean`, `source!=unknowable`이고 suite가 `synthetic-`으로 시작하지 않는 질문만 포함합니다. 변형·unknowable·합성 지연시간 문항은 이 정확도 집계에서 제외됩니다.

`compare`의 각 C에는 기준선과 후보의 실제 전체·질문 유형별·suite별 품질이 **두 head 방식 모두** 표시됩니다. LM 행과 logprobs는 별도의 확률/표시 답 패리티를 검사합니다. 학습 head의 품질은 기술 통계이며 이 개발 split의 단일 실행으로 품질 개선 승자를 선언하지 않습니다. 전체 데이터의 출처와 정답 가능 집계 범위는 [실제 판단 데이터 설명](../../outputs/h200-prefill-performance-20260928/BENCHMARK_DATA.md)을 참고하세요.
## 정밀도와 FP8

`precision`은 activation/head 경계 dtype이며 FP8도 `bfloat16`을 사용합니다. FP8 비교 설정은 `quantization: "fp8"`, `precision_profile: "fp8_w8a8_backbone_bf16_head"`를 함께 기록합니다. 기존 BF16 설정의 두 필드는 생략할 수 있으며 각각 `null`, `unquantized_bfloat16`으로 해석합니다. 실제 서버의 quantization과 일치해야 하며 BF16/FP8 간 성능 비교는 거부합니다. 프리필 요청 한도 `prefill_max_requests`가 다른 실행도 같은 조건 비교로 받아들이지 않습니다. [FP8 지원 범위](PREFILL_ENGINE_FP8.md)를 따릅니다.

### BF16/FP8 별도 품질 분석

`jev-adapter` 디렉터리에서 저장된 BF16·FP8 benchmark 디렉터리를 지정합니다. `--pair`를 반복해 여러 회차를 넣을 수 있으며 출력 파일은 새 경로여야 합니다.

```bash
.venv/bin/python -m benchmarks.prefill_precision_quality \
  --pair /path/to/bf16/benchmark /path/to/fp8/benchmark \
  --output /path/to/new-quality-report.json
```

이 도구는 모든 raw 응답과 저장 지표를 다시 검증한 뒤 같은 고정 checkpoint revision·head 방식·readout artifact SHA256·workload·라벨·temperature=1인 실행만 분석합니다. stock BF16↔stock FP8 또는 candidate BF16↔candidate FP8를 각각 지정하세요. 회차와 C별 `repeat=0`의 고유 질문만 품질 표본으로 사용하고 정확도·Brier·NLL, argmax 변경, 확률 절대 차이와 유형·benchmark별 집계를 기록합니다. 전체 development pass의 clean 품질 분모는 1,194이며 실제 분모를 출력합니다. 반복 요청·C·회차를 합쳐 독립 정확도 표본으로 계산하지 않고 성능 승자를 선언하지 않습니다. 기존 같은 정밀도 성능 비교의 검증 조건도 유지됩니다.

도구는 `SGLANG_DECISION_PROJECTION_WIDTH` 환경변수를 직접 추적하지 않습니다. candidate BF16의 폭 0과 FP8의 폭 8192를 비교하면 양자화와 head projection 모양이 함께 달라지므로 관측한 품질 변화를 양자화만의 영향으로 해석할 수 없습니다. 실행 source·환경 증거를 별도로 보존해야 하며 결과의 `causal_quantization_claim`은 항상 `false`입니다.
