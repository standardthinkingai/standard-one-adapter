> Part of the Standard One decision engine: this adapter is imported by the SGLang fork
> [standard-one-sglang](https://github.com/standardthinkingai/standard-one-sglang) (see its STANDARD_ONE.md for build and serving).

# Jev adapter

System One 판단 API에는 두 실행 방식이 있습니다. [새 단일 서버 통합 모드](benchmarks/PREFILL_ENGINE.md)는 이 저장소의 SGLang fork 안에서 토크나이저와 GPU readout을 사용해 `POST /v1/systemone`을 직접 제공합니다. 아래 설명은 기존 독립 어댑터 시제품입니다. 기존 기본 `logprobs` 경로는 별도 SGLang HTTP API를 호출하며, 어댑터 환경에 모델 가중치나 CUDA가 필요하지 않습니다.

```text
클라이언트 → jev-adapter :30120 → 기본 SGLang :30000 → GPU 모델
            질문/응답 변환       템플릿·토큰화·비전 처리·프리필
```

기본 백엔드에서는 모델이 선택지 라벨의 다음 토큰 logprob를 반환합니다. 선택형 `candidate-only` 백엔드는 마지막 hidden state와 작은 LM-head readout 파일을 사용하고, 실험용 `learned-head` 백엔드는 검증된 3B decision head 파일을 사용합니다. 어댑터가 확률 정규화와 최종 JSON 구성을 담당합니다. 엔진 내부 객체나 비공개 Python 함수를 호출하지 않습니다.

## 실행

엔진 환경에서 모델을 실행합니다. 어댑터와 같은 컴퓨터일 필요는 없습니다. 아래는 모델의 served name을 `decision-model`로 정한 예시입니다. `MODEL_PATH`는 엔진이 지원하는 실제 모델 경로 또는 Hugging Face ID로 설정합니다.

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name decision-model \
  --host 127.0.0.1 --port 30000
```

어댑터는 이 레포의 별도 가상 환경에서 실행합니다. 엔진을 다른 호스트에서 실행하면 `--engine-url`만 바꾸면 됩니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/jev-adapter \
  --engine-url http://127.0.0.1:30000 \
  --model decision-model \
  --port 30120
```

`python -m jev_adapter`도 같은 CLI입니다. 기본 listen 주소는 `127.0.0.1`이며, `--host`로 변경합니다. `--max-concurrency` 기본값은 32, 엔진 HTTP timeout은 `--timeout 120`입니다.

`--default-temperature T` (환경 변수 `JEV_DEFAULT_TEMPERATURE`)는 요청이 `options.temperature`를 생략했을 때만 적용되는 서버 기본 온도입니다. 요청이 온도를 명시하면 요청 값이 우선하고, `options.temperature_scaling: false`이면 기본값과 무관하게 스케일링하지 않습니다. 지정하지 않으면 기존과 같이 1.0입니다. 실제 적용된 온도는 응답의 `metadata.temperature`로 확인합니다.

`SGLANG_API_KEY`는 엔진에 보낼 키, `JEV_API_KEY`는 클라이언트가 어댑터에 보낼 키입니다. 설정했다면 각각 `Authorization: Bearer ...` 인증을 사용합니다. 키를 설정하지 않은 어댑터는 인증 없이 동작합니다.

시작할 때 엔진 정보를 확인하고, 엔진 토크나이저로 라벨을 검증합니다. 엔진을 먼저 실행해야 합니다. 모델/토크나이저를 교체하면 어댑터도 재시작합니다.

Ministral의 공식 `mistral-common` 토크나이저는 제어 토큰을 문자열로 변환했다가 다시
인코딩하면 토큰 ID가 달라질 수 있습니다. SGLang 0.5.20에서 이 문제가 확인되어,
이 모델의 텍스트 요청은 선택형 공식 토크나이저 경로를 사용합니다.

```bash
.venv/bin/python -m pip install -e '.[native-tokenizer]'
.venv/bin/jev-adapter --model decision-model \
  --tokenizer-model mistralai/Ministral-3-3B-Instruct-2512-BF16 \
  --tokenizer-revision b6d637bef2393152b3da2b2fde72eecdee30557e
```

이 경로는 CPU에서 토크나이저 파일만 로드하고 공식 `apply_chat_template(tokenize=True)`
결과를 엔진에 전달합니다. 각 assistant 라벨의 토큰 경계도 공식 continuation으로
검증합니다. 이미지 입력은 지원하지 않으며 기본 HTTP 경로로 자동 전환하지 않습니다.
토크나이저가 지원하면 3개 이상 라벨의 continuation을 일괄 렌더링하고, 지원하지
않으면 기존 순차 검증으로 돌아갑니다. 로컬 변환 Ministral 토크나이저에서 4개와
26개 라벨의 토큰 ID 일치를 확인했으며, 공식 GPU 배포에서는 아직 검증하지 않았습니다.

### 선택형 candidate-only 백엔드

이 경로는 **Ministral 3 3B/8B의 공식 비양자화 safetensors 체크포인트**에서
라벨에 필요한 LM-head 행만 내보냅니다. tied embedding 모델은 embedding 가중치,
untied 모델은 별도 LM-head 가중치의 해당 행을 읽습니다. 공식 배포본과
Transformers 저장본의 두 이름 체계를 체크포인트 인덱스에서 확인하며, 중복되면
실패합니다. 내보내기는 `get_slice`로 진행하며 전체 vocabulary 행렬을
메모리에 올리지 않습니다. 학습된 별도 decision head를 읽는 기능은 아닙니다.

```bash
.venv/bin/python -m pip install -e '.[native-tokenizer,candidate-readout,readout-export]'
.venv/bin/jev-export-readout \
  --checkpoint /models/Ministral-3-3B-Instruct-2512-BF16 \
  --model-id mistralai/Ministral-3-3B-Instruct-2512-BF16 \
  --revision b6d637bef2393152b3da2b2fde72eecdee30557e \
  --output /models/ministral3-3b-readout.npz
```

SGLang은 해당 모델 리비전(`--revision`에 위 commit SHA 지정)으로 실행하고
`--return-hidden-states-mode last`를
설정합니다. 엔진의 `/model_info.model_path`가 로컬 경로를 반환하면 내보내기에
`--engine-model-path`로 그 **정확한 경로**를 지정합니다. 기본값은 `--model-id`입니다.

```bash
.venv/bin/jev-adapter \
  --engine-url http://127.0.0.1:30000 --model decision-model \
  --backend candidate-only --readout /models/ministral3-3b-readout.npz \
  --tokenizer-model mistralai/Ministral-3-3B-Instruct-2512-BF16 \
  --tokenizer-revision b6d637bef2393152b3da2b2fde72eecdee30557e
```

시작할 때 artifact의 모델 ID·`/server_info.revision`·엔진 모델 경로·아키텍처·라벨 토큰 ID를
확인하고 hidden-state probe로 차원을 검사합니다. 요청에는
`max_new_tokens: 0`, `return_hidden_states: "last"`만 보내며
`return_logprob`를 요청하지 않습니다. CPU에서 선택한 행과 hidden state의 내적을
계산합니다. 선택 라벨 사이의 softmax에는 같은 결과를 주지만 전체 vocabulary의
정확한 logprob는 계산하지 않으므로 `options.return_logprobs: true`는 추론 전에
422로 거부합니다. 이미지 요청도 거부합니다.

이 경로에서 전체 vocabulary LM-head 연산을 실제로 생략하려면 이 저장소의
`sglang-jev`에 추가한 조건부 fast path가 필요합니다. 기본 SGLang은 hidden-only
요청에도 전체 head를 계산할 수 있습니다. 엔진은 BF16/FP16 비양자화 가중치로
실행해야 하며, 엔진 API에서 양자화 설정이 보이면 시작을 거부합니다. artifact의
revision과 토크나이저 pin을 비교합니다. `/server_info.revision`이 없거나 일치하지
않으면 시작하지 않습니다. 이 API의 revision은 엔진 설정값이므로 실제 가중치
파일의 해시까지 입증하지는 않습니다.
GPU 프로파일과 H200·MI325X에서의 p50/동시 처리량 측정은 아직 필요합니다.
CPU 토큰화, hidden state의 JSON 전송과 readout 계산도 지연에 포함됩니다.
현재 fork의 hidden-state relay는 pipeline parallelism `PP=1`에서만 사용합니다.

### 실험용 learned-head 백엔드 (3B)

검증된 아카이브의 `linear-seed17/head.safetensors`는 별도로 학습된 62×3072
선형 decision head입니다. 이 백엔드는 **Ministral 3 3B만** 지원하며 기본
`logprobs`나 위 `candidate-only` 경로를 변경하지 않습니다. `selection.json`의
선택 결과와 head SHA256, `provenance.json`의 `post_final_norm` 좌표·모델 pin·
라벨 순서, `temperature.json`의 보정값을 시작할 때 확인합니다. 네 파일은
아카이브 디렉터리 구조를 유지해야 합니다.

```bash
.venv/bin/python -m pip install -e '.[native-tokenizer,learned-head]'
.venv/bin/jev-adapter \
  --engine-url http://127.0.0.1:30000 --model decision-model \
  --backend learned-head \
  --readout ../jev-lora-data/runs/h200-trm-readout-20260922/verified/results/linear-seed17/head.safetensors \
  --tokenizer-model mistralai/Ministral-3-3B-Instruct-2512-BF16 \
  --tokenizer-revision b6d637bef2393152b3da2b2fde72eecdee30557e
```

SGLang은 같은 공식 3B BF16 모델 ID와 revision으로 실행하고
`--return-hidden-states-mode last`, `PP=1`을 지정합니다. 엔진의
`/model_info.model_path`가 공식 모델 ID이고 `/server_info.revision`이 위 SHA와
같아야 합니다. 양자화 엔진과 다른 아키텍처·dtype은 거부합니다. 마지막 hidden
state 차원을 startup probe로 확인하며, 요청에는 출력 토큰 0개와 마지막 hidden
state만 요구합니다. `options.return_logprobs: true`와 이미지는 지원하지 않습니다.

학습 아카이브에서 검증된 보정 온도 `0.8518692324650765`를 head 로짓에 먼저
적용합니다. API의 `options.temperature`가 있으면 서비스가 그 결과에 추가로
적용합니다. 즉 유효 온도는 두 값의 곱이며, API에서 temperature scaling을 꺼도
학습 head의 고정 보정은 유지됩니다.

학습 데이터의 **표시 라벨은 A–C뿐**입니다. 4개 이상 선택지는 비교 실험을 위해
계산하지만 D 이후 head 행은 지도학습 대상이 아니었으므로 실제 서빙 품질로
간주하면 안 됩니다. 8B용 학습 head도 아직 없습니다. 기존 프롬프트와 학습 캐시의
프롬프트·전처리가 정확히 같은지 확인되지 않았으므로, 같은 입력을 쓴 지연시간
비교는 가능하지만 이 경로의 온라인 정확도를 오프라인 평가 점수로 추정할 수
없습니다. H200·MI325X의 실측 p50 및 동시 처리량도 아직 없습니다.

### 답 라벨 체계 (`--label-scheme`)

`--label-scheme {upper,served}` (환경 변수 `JEV_LABEL_SCHEME`)는 선택지가 26개를
넘는 질문의 라벨을 정합니다. 선택지가 26개 이하이면 두 체계 모두 기존과 똑같이
`A`..`Z`를 씁니다.

- `upper` (기본값, v2.2부터의 계약): `A`..`Z` 다음 `AA`, `AB`, …, `ZZ`, 그다음
  `AAA`..`ZZZ` 순서의 대문자 후보 중에서, 토크나이저가 **특수 토큰이 아닌 서로 다른
  단일 토큰**으로 인코딩하고 그대로 디코딩되는 것만 순서대로 최대 255개 씁니다.
  Ministral-3 (`f6fae97…`)에서는 두 글자 대문자 676개 중 439개가 단일 토큰이라 세
  글자까지 가지 않고 255번째 라벨이 `MH`입니다(`AJ`, `AQ`, `AW` 등은 건너뜀).
- `served`: 이전 순서(`A`..`Z`, `a`..`z`, `0`..`9`, 그다음 `AA`, …). v2.2 이전의
  서빙 동작을 재현할 때와, 이 순서의 62개 행을 가진 3B learned head용으로 남겨 둡니다.

v2는 26개를 넘는 선택지를 학습한 적이 없습니다. v2.1 학습 입력(606,729행) 중
`display_labels`가 있는 593,092행의 최대 선택지 수는 26이고, 나머지 13,637행의
`criteria`도 26개 이하입니다. 따라서 v2의 26개 초과 동작은 두 체계 모두 학습되지 않은
동작이며, v2.2가 `upper`를 배우려면 학습 혼합에 26개를 넘는 행이 `labels[:n]`로
렌더링되어 들어가야 합니다.

알려진 한계(측정 전):

- 두 글자 라벨이 대문자 옵션 이름과 겹칠 수 있습니다. 예를 들어 미국 50개 주
  질문에서 2번째 줄은 `B: AK`(Alaska), 36번째 줄은 `AK: OK`(Oklahoma)입니다. 모델이
  라벨 대신 옵션 이름을 답하면 다른 옵션으로 채점됩니다. `served`의 27–52번은
  소문자라 이런 겹침이 없었습니다(26개 이하에서 `A`..`Z` 이름과 겹치는 문제는
  원래부터 있었습니다). 학습 데이터와 회귀 행에 이런 경우를 넣으세요.
- 첫 토큰만 채점하므로, `AB`를 `A` 다음 `B`로 쓰려는 모델의 확률은 1번 옵션 `A`에
  들어갑니다. Ministral-3에서는 쓰이는 두 글자 라벨이 모두 한 토큰이지만, 학습되지 않은
  모델은 1–26번 쪽으로 치우칠 수 있습니다.

운영 재빌드 시: 기본값이 `upper`이므로 모든 실행 스크립트에 `--label-scheme`
(decision 워커는 `--decision-label-scheme`, 또는 `JEV_LABEL_SCHEME`)를 명시하세요.
기존 v1 readout은 기본값에서 시작을 거부하므로 `--label-scheme upper`로 다시 내보내거나
`served`를 명시해야 합니다.

활성 체계는 `/v1/models`의 모델 항목과 응답 `metadata.label_scheme`에 표시됩니다.
규칙은 `jev_adapter/labels.py`의 `select_labels` 하나이며, 학습 데이터도 같은
목록을 써야 합니다(`tests/data/upper-labels-ministral3.json`).

GPU readout artifact(`jev-gpu-readout-v2`, candidate readout `jev-candidate-readout-v2`)는
`label_scheme`, `tokenizer_id`, `tokenizer_revision`을 기록합니다. v1 artifact는
`served`로 간주합니다. 통합 decision 서빙은 artifact의 체계가 설정된 체계
(`--decision-label-scheme` 서버 인자, 없으면 `JEV_LABEL_SCHEME`, 없으면 `upper`)와
다르거나, artifact 라벨이 그 체계로 엔진 토크나이저에서 뽑은 앞쪽 라벨과 정확히
같지 않으면 시작하지 않습니다. `jev-export-readout --label-scheme`으로 원하는 체계의
행을 내보냅니다. 실험용 learned-head(62개 `A-Z,a-z,0-9` 행)는 `--label-scheme served`가
필요합니다.

### 프롬프트 문구 (`--prompt-wording`)

`--prompt-wording {served,native}` (환경 변수 `JEV_PROMPT_WORDING`)는 어떤 문구로
프롬프트를 렌더링할지 정하는 서버 전역 설정입니다. 기본값 `served`는 기존과 동일하게
어댑터 자체 문구(`Context:/.../Options: A: name: description`)를 사용합니다.

`native`는 jevbench-hard 러너 자신의 오프라인 채점 문구(`run_suites_rotation.py`의
`State:/Question:/Options: A. name: description`)를 그대로 렌더링합니다. 러너 자체
ablation에서 이 문구 차이만으로 jevbench-hard 정확도가 약 +6점 높게 측정되었습니다
(native 61.3 대 served 55.0; 동일 토크나이저 계열, rotation 없는 canonical 순서 기준).
옵션 **순서** 자체는 바꾸지 않습니다 — 요청이 준 순서(`choice`/`score`는 criteria
순서, `noul`은 항상 true 다음 false)를 그대로 쓰고 문구만 바꿉니다. 그래서
`--tokenizer-model`/`--tokenizer-revision`이 가리키는 토크나이저의 앞쪽 라벨이
정확히 대문자 `A`..`Z`(최대 26개 선택지)가 아니면 해당 요청은 거부됩니다 — 대부분의
영문 어휘 토크나이저에서는 자동으로 성립합니다.

`--tokenizer-model`/`--tokenizer-revision`도 함께 주면, 어댑터는 같은 pin을
`fix_mistral_regex=True`로 별도 로드해 HF chat template이 자동 주입하는 모델
기본 시스템 프롬프트 텍스트를 추출하고, 이를 명시적 system 메시지로 앞에 붙입니다.
Ministral처럼 서빙 토크나이저가 `mistral-common` 백엔드로 해석되는 경우
(`native_tokenizer.py` 문서 참고) 이 백엔드는 기본 시스템 프롬프트를 자동으로
주입하지 않기 때문입니다 — 직접 추출한 텍스트가 없으면 시스템 메시지 없이 문구만
`native`로 서빙합니다(시작 시 경고 로그). `/opt/.../SYSTEM_PROMPT.txt` 같은 정적
문서 파일은 실제 chat template이 렌더링하는 기본 텍스트와 다를 수 있어(확인된 사례:
문장 하나가 중복/누락) 신뢰하지 않고, 항상 `apply_chat_template`을 직접 렌더링해서
추출합니다.

이 기본 시스템 프롬프트 자동 추출은 `--native-system-prompt {auto,none}`
(환경 변수 `JEV_NATIVE_SYSTEM_PROMPT`)로 끌 수 있습니다. 기본값 `auto`는 위에서
설명한 기존 동작 그대로이고, `none`은 추출 자체를 건너뛰고 시스템 메시지 없이
`native` 문구만 렌더링하며 누락 경고 로그도 남기지 않습니다. 이 옵션을 둔 이유는
Ministral 기본 시스템 프롬프트가 531토큰으로, 전형적인 요청 하나의 약 80%에 달할
만큼 커서 — 프리필 비용과 지연시간에 영향을 주는데, jevbench-hard 정확도 이득은
문구(`native` vs `served`) 자체에서 오지 시스템 프롬프트 유무에서 오는지는 별도
질문이기 때문입니다. `--prompt-wording served`에는 아무 영향이 없습니다.

```bash
.venv/bin/jev-adapter --model decision-model \
  --tokenizer-model mistralai/Ministral-3-8B-Instruct-2512-BF16 \
  --tokenizer-revision f6fae9795746f63c9be8344932f01275f3c63734 \
  --prompt-wording native --default-temperature 2.4
```

CPU 전용 검증(`jevbench-original` 40개 + `internal-eval-v1` 40개 전체, `jevbench-hard`
5개는 구조만) 결과, 요청 옵션 순서를 고정하면 어댑터가 렌더링한 `native` 문구와
러너의 `format_prompt()` 출력은 **텍스트와 토큰 ID가 모두 완전히 일치**했고
(`mistral-common`이 명시 system 메시지를 렌더링한 토큰 ID와 HF 템플릿의 자동 주입
토큰 ID도 이 표본에서는 차이가 없었습니다), 라벨 토큰 ID도 일치했습니다. 다만
`jevbench-original`의 40개 중 24개, 표본 `jevbench-hard` 5개 중 5개에서
데이터셋이 기록한 "정답 채점 순서"(`expected.labels`)가 어댑터가 실제로 쓰는 옵션
순서(criteria 순서 / noul 고정 순서)와 달랐습니다 — 이는 문구와 무관한, 옵션 **순서**
자체의 사전 존재 차이이며 어댑터가 요청 순서를 그대로 따르는 한 코드로 없앨 수
없습니다. 러너의 native 61.3점 자체도 이 순서로 측정된 값이라, jev-adapter로
`--prompt-wording native`를 서빙했을 때 실제로 회수되는 점수는 이 순서 효과만큼
61.3보다 낮을 수 있습니다 — 문구 효과 자체는 위에서 텍스트/토큰 단위로 확인했습니다.

```bash
.venv/bin/python examples/smoke.py
.venv/bin/python examples/smoke.py --image /path/to/screenshot.png
```

기존 평가 클라이언트에서도 base URL을 `http://127.0.0.1:30120`으로 바꾸면 됩니다. `/v1/models`는 실제 모델명을 반환하고, `/health`는 어댑터 프로세스의 생존 여부만 확인합니다.

## 요청과 응답

전체 예시는 [examples/request.json](examples/request.json)을 참고하세요. 모델에는 실제 served name 또는 기본 별칭 `jev-latest`를 사용할 수 있습니다.

```json
{
  "model": "jev-latest",
  "state": "중복 결제를 환불해 주세요.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "담당 부서는?",
      "criteria": {"billing": "결제 및 환불", "technical": "기술 지원"}
    },
    "refund": {"type": "noul", "instructions": "환불 요청인가?"},
    "urgency": {"type": "score", "instructions": "긴급도", "criteria": ["낮음", "보통", "높음"]}
  },
  "options": {"temperature": 1.0, "permutations": 1}
}
```

| 유형 | 결과 |
| --- | --- |
| `choice` | 선택지 확률, 최대 확률의 `choice`, 엔트로피 기반 `confidence` |
| `noul` | 참일 선택지의 확률인 `noul` |
| `score` | 0부터 시작하는 등급 인덱스의 기대값 `score`, 확률, `legend`, `confidence` |

`state`와 설명은 문자열 또는 JSON 객체/배열을 받을 수 있습니다. `images`는 HTTP(S) URL 또는 `data:image/...;base64,...` 목록이며, 이미지 바이트 처리와 비전 인코더 실행은 엔진이 담당합니다.

`options.permutations`는 선택지 순서를 회전한 뒤 원래 순서로 복원하여 확률을 평균합니다. **질문 수 × permutations만큼 프리필 평가**가 발생합니다. `return_logprobs: true`는 각 평가의 원래 vocabulary logprob를 제공합니다. `assistant_prefix`를 설정하면 실제 응답 경계에서 라벨이 단일 토큰인지 다시 검증합니다.

확률은 지정한 라벨 안에서 정규화한 값입니다. `confidence = 1 - entropy / log(선택지 수)`는 실제 정답률의 보정값이 아닙니다. 학습하지 않은 모델의 판단 정확도는 별도로 평가해야 합니다.

## 엔진 API와 지원 범위

사용하는 공개 API는 `/model_info`, `/server_info`, `/v1/tokenize`, `/v1/detokenize`, `/generate`입니다. SGLang upstream [`0024efa0de38794ee309ba10ab00ebb891a3d050`](https://github.com/sgl-project/sglang/tree/0024efa0de38794ee309ba10ab00ebb891a3d050)의 구현을 기준으로 연결 규약을 확인했습니다. 오래된 릴리스에 필요한 API가 없으면 엔진 버전을 올려야 합니다.

일반 텍스트는 템플릿을 적용한 token IDs를 엔진에 전달합니다. 이미지 요청은 Qwen VL/Qwen3.5, Mistral3/Pixtral 계열의 일반적인 텍스트 기반 멀티모달 경로를 대상으로 합니다. 이미지가 포함된 프롬프트의 특수 토큰과 원본 이미지를 함께 전달합니다. 토큰만으로 프롬프트를 표현하는 다른 VLM 계열은 명시적으로 거부합니다. 모델마다 이미지 경로의 GPU 통합 검증이 필요합니다.

기본 `logprobs` 경로의 `/generate`에는 `max_new_tokens: 0`, `return_logprob: true`, `token_ids_logprob: [...]`를 설정합니다. 모든 라벨의 유한한 logprob와 `completion_tokens == 0`을 확인합니다. 누락된 확률이나 토큰 사용량을 임의로 보완하지 않습니다. speculative decoding/MTP를 사용하지 않는 일반 generation 서버가 필요합니다.

현재의 엄격한 프롬프트 검증은 평가마다 템플릿 토큰화, detokenize, 라벨 경계 일괄 검증, generate의 **HTTP 호출 4회**를 사용합니다. 이 왕복과 CPU 처리 비용은 GPU 프리필 시간에 추가됩니다. `metadata.adapter_elapsed_ms`는 이 작업과 대기 시간을 포함하고, 외부 HTTP 응답 직렬화/전송 시간은 제외합니다. `smoke.py`는 클라이언트에서 전체 왕복 시간을 표시합니다. GPU 간 지연 비교 수치로 대신 사용할 수 없습니다.

선택형 공식 토크나이저 경로는 초기 라벨 검증 이후 평가마다 CPU 토큰화와 엔진
`/generate` 호출 1회를 사용합니다. 두 경로의 HTTP 지연에는 서로 다른 CPU 처리와
왕복 횟수가 포함되므로 순수 모델 연산 성능처럼 비교하면 안 됩니다.

클라이언트 연결이 끊어지면 어댑터의 대기 작업과 엔진 HTTP 요청을 취소합니다. 이후 정리는 기본 엔진의 disconnect 처리에 의존합니다. 이미 실행 중인 GPU 연산을 즉시 중단한다는 보장은 없습니다.

vLLM 백엔드는 아직 구현하지 않았습니다. 추가 시 `ScoringBackend` 인터페이스를 구현하면 질문/응답 규약은 그대로 재사용할 수 있습니다.

## 검증과 코드

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

테스트는 실제 프로토콜/서비스를 사용하고 추론 서버 HTTP 응답만 모의합니다. 확률 정규화, 순서 복원, 혼합 질문, 이미지 전달, 오류, 취소와 동시 실행 제한을 검증합니다. 이 테스트 통과는 NVIDIA GPU에서의 정확도·성능 또는 모델별 이미지 처리 성공을 뜻하지 않습니다.

2026-09-21 검증: 독립 Python 3.12 가상환경에서 **97개 테스트와 74개 하위 케이스 통과**, Ruff 통과. 기본 어댑터 환경에는 SGLang, vLLM, PyTorch, Transformers, MLX가 필요하지 않습니다. H200의 공식 SGLang 0.5.20 wheel로 네 모델 각각 4,006개 요청의 전체 벤치를 완료했으며, Ministral은 선택형 CPU 토크나이저 환경을 사용합니다.

2026-09-23 `--prompt-wording native` 추가 검증: 같은 가상환경에서 **136개 테스트와 80개 하위 케이스 통과**, Ruff 통과. 운영 서버(CPU 전용, GPU 미사용)에서 `jevbench-original`/`internal-eval-v1`/`jevbench-hard` 실제 레코드로 어댑터의 `native` 문구·토큰 ID 렌더링을 jevbench-hard 러너의 `run_suites_rotation.py` 계약과 직접 비교했습니다(위 "프롬프트 문구" 절 참고).

| 파일 | 역할 |
| --- | --- |
| `jev_adapter/protocol.py` | 요청 스키마, 프롬프트, 확률 및 응답 계산 |
| `jev_adapter/service.py` | 질문별 평가 실행, 병렬 수 제한, 결과 조합 |
| `jev_adapter/sglang.py` | 공개 SGLang HTTP API 클라이언트 |
| `jev_adapter/candidate_backend.py` | 마지막 hidden state를 읽는 요청 경로 |
| `jev_adapter/learned_head.py` | 검증된 3B decision head 로드·보정·채점 |
| `jev_adapter/server.py` | 독립 FastAPI 서버 및 인증 |
| `jev_adapter/backend.py` | 다른 추론 엔진을 연결할 최소 인터페이스 |

이 레포는 엔진 소스 트리나 형제 디렉터리를 import하지 않습니다. 기존 `sglang-jev` 실험 레포와 벤치마크 결과는 원래 위치에 남아 있습니다. 초기 프로토콜 코드의 출처는 [NOTICE](NOTICE)를 참고하세요.

## H200 학습 전 벤치마크

[benchmarks/README.md](benchmarks/README.md)에 H200 한 장에서 Ministral 3 3B, Qwen3.5-4B, Qwen3.6-27B, Qwen3.6-35B-A3B를 순서대로 평가하는 절차가 있습니다. 기본 비교는 BF16이며 KEV 공개 개발 세트의 정확도·확률 품질·HTTP 지연시간을 기록합니다. 데이터와 모델 리비전을 고정했고 H200 실측을 완료했습니다. KEV 외 공개 평가셋 조사는 [benchmarks/PUBLIC_DATASETS.md](benchmarks/PUBLIC_DATASETS.md)를 참고하세요.
