# H200 한 장에서 기본 모델 비교

추가 학습이나 LoRA 없이 공식 체크포인트를 기본 SGLang으로 실행하여 텍스트 판단
벤치마크를 진행합니다. 엔진과
`jev-adapter`는 별도 프로세스입니다. 이 디렉터리는 모델·엔진 실행 조건만 관리하며,
데이터 준비와 평가에는 `python -m jev_adapter.benchmarks.prepare`와
`python -m jev_adapter.benchmarks.run`을 사용합니다. 각 CLI의 `--help`에서 입력
파일·출력 디렉터리 옵션을 확인하세요.

현재 상태는 **모델 설정과 엔진 소스 확인, CPU 검사 완료 / H200 실측 미실행**입니다.
공식 체크포인트 리비전은 [models.json](models.json)에 고정했습니다.

| 프로필 | 공식 체크포인트 | 실행 정밀도 |
| --- | --- | --- |
| `ministral3-3b-bf16` | Ministral-3-3B-Instruct-2512-BF16 | 공식 BF16 체크포인트 |
| `qwen35-4b-bf16` | Qwen3.5-4B | BF16 |
| `qwen36-27b-bf16` | Qwen3.6-27B | BF16 |
| `qwen36-35b-a3b-bf16` | Qwen3.6-35B-A3B | BF16 |
| `ministral3-3b-fp8` (선택) | Ministral-3-3B-Instruct-2512 | 언어 모델 FP8; 비전·프로젝터 등 제외 모듈 BF16 |

기본 4종은 공식 **BF16 체크포인트**로 맞췄습니다. 모델별 토크나이저·비전 인코더·
프롬프트 템플릿 차이는 남으므로 순수한 아키텍처 비교는 아닙니다. 선택 프로필인
Ministral FP8와 Qwen BF16의 지연 차이를 아키텍처만의 효과로 해석하면 안 됩니다.
`--dtype bfloat16`만 지정해도 원본 Ministral FP8 가중치가 BF16 체크포인트로
바뀌지는 않습니다. 모든 모델에서 norm, 누산, 상태 저장 등 일부 연산은 별도
정밀도를 사용할 수 있습니다.

## 엔진 설치

H200가 있는 Linux 호스트에서 CUDA 13을 지원하는 NVIDIA 드라이버, Python 3.12,
공식 wheel에 맞는 x86_64/glibc 2.34 이상 환경을 준비합니다. 이 프로필은 별도의 엔진
가상환경을 사용하며, 기존 엔진 환경을 갱신하지 않습니다.

```bash
python3.12 -m venv .venv-sglang
.venv-sglang/bin/python -m pip install --upgrade pip
.venv-sglang/bin/python - <<'PY'
import json, subprocess, sys
engine = json.load(open('benchmarks/h200/models.json'))['engine']
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--pre',
                       '--only-binary=sglang', engine['install_spec']])
PY

python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,native-tokenizer]'
```

SGLang은 공식 0.5.20 wheel을 SHA256까지 고정해 설치합니다. 해당 [릴리스 소스 리비전](https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e)의
의존성은 Torch 2.13.0, Transformers 5.12.1과
CUDA 13 패키지입니다. [공식 설치 안내](https://docs.sglang.io/docs/get-started/install)와
다른 최신 릴리스의 의존성을 섞지 마세요. 모든 간접 의존성을 잠근 컨테이너 이미지는
아니므로 실행 시 실제 주요 패키지·CUDA·드라이버 버전을 기록합니다. 재현 실험에서는
이 기록을 함께 보관해야 합니다.

Ministral 프로필은 어댑터에서 고정 리비전의 공식 토크나이저를 사용합니다. 배포본의
chat 문자열 왕복 경로가 native 제어 토큰을 보존하지 않는 것을 실측 확인했기
때문입니다. 엔진은 수정하지 않습니다. 이 경로는 텍스트만 지원하며, Ministral에
`--probe-vision`을 지정하면 명확히 실패합니다. Qwen 프로필은 엔진의 HTTP 토큰화를
사용합니다. 토큰화 위치는 `models.json`과 각 실행의 모델 메타데이터에 기록됩니다.

## 실행

명령은 레포 루트에서 실행합니다. `--execute`를 생략하면 네트워크 연결, 모델
다운로드, GPU 초기화, 서버 실행 없이 최종 엔진 명령만 출력합니다.

```bash
.venv/bin/python benchmarks/h200/runtime.py list
.venv/bin/python benchmarks/h200/runtime.py launch qwen35-4b-bf16
```

H200에서 실제 실행할 때는 비어 있는 출력 경로를 지정합니다. 기본 GPU는 물리
인덱스 0이며 `--gpu GPU-...`로 NVIDIA UUID를 지정할 수도 있습니다.

```bash
.venv/bin/python benchmarks/h200/runtime.py launch qwen35-4b-bf16 \
  --execute --gpu 0 --output outputs/h200/qwen35-4b-bf16-runtime
```

실행기는 설치된 SGLang의 PEP 610 기록에서 소스 커밋을 확인하고, 선택한 GPU가
H200 한 장인지 검사합니다. 이어 모델 다운로드·엔진 초기화를 진행합니다. 엔진
종료, 이미 사용 중인 포트, 준비 시간 초과, 잘못된 모델·리비전·캐시·정밀도 설정은
즉시 실패로 처리합니다. 준비 제한은 기본 1,800초이며 `--timeout`으로 바꿀 수 있습니다.

`/v1/tokenize`의 chat template과 일괄 입력, `/v1/detokenize`, `/generate`의 지정
라벨 logprob·출력 0토큰 경로를 **텍스트로만 검사**합니다. 이미지 준비 검사는
`--probe-vision`을 지정했을 때만 224×224 PNG로 추가 실행합니다. 기본 텍스트
벤치마크는 이미지 처리 경로의 검사 결과에 의존하지 않습니다. 모델의 비전 인코더를
삭제하거나 비활성화하는 변경은 하지 않습니다. 성공 메시지를 확인한 뒤
다른 터미널에서 어댑터를 실행합니다.

```bash
.venv/bin/jev-adapter --engine-url http://127.0.0.1:30000 \
  --model decision-model --port 30120 --max-concurrency 32
.venv/bin/python examples/smoke.py
```

평가 러너에 `--engine-url http://127.0.0.1:30000`와
`--engine-manifest outputs/h200/qwen35-4b-bf16-runtime/launch.json`을 전달하면
실행 조건을 평가 결과와 연결할 수 있습니다. 한 모델을 마친 뒤 어댑터와 실행기를
각각 Ctrl-C로 종료하고 다음 프로필을 같은 GPU에서 실행하세요. 실행기는 자신이
시작한 엔진 프로세스 그룹만 정리합니다.

이미 실행 중인 엔진의 공개 API를 다시 검사할 수도 있습니다. 이 명령은 엔진의
환경 변수·GPU·설치 커밋을 새로 입증하지 않으므로 원래 `launch.json`을 보관하세요.

```bash
.venv/bin/python benchmarks/h200/runtime.py probe qwen35-4b-bf16 --timeout 60
```

나중에 이미지 입력도 검증하려면 `launch` 또는 `probe` 명령에 `--probe-vision`을
추가하세요. 이는 준비 검사 범위만 바꾸며 모델과 엔진 실행 조건은 그대로 유지합니다.

## 고정 조건과 기록

- H200 한 장, TP=1, 엔진의 최대 동시 실행 요청 32, 컨텍스트 최대 8,192토큰,
  chunked prefill 비활성. 단일 요청 지연은 평가 클라이언트 동시성을 1로 설정하고,
  처리량은 별도 동시성 조건으로 기록합니다.
- 학습·LoRA·MTP·speculative decoding 없음. 모델의 기본 비전 인코더 유지.
- `--disable-radix-cache`, `--mm-preprocess-cache-size-mb 0`,
  `SGLANG_VLM_CACHE_SIZE_MB=0`; prefix/global 멀티모달 캐시 활성화 옵션은 사용하지 않음.
- GPU의 모델 가중치와 커널/JIT 준비는 유지합니다. 각 요청은 prefix KV와 이미지
  임베딩을 재사용하지 않지만, OS 파일 캐시까지 초기화하는 cold-start 실험은 아닙니다.
- `SGLANG_MM_FEATURE_CACHE_MB`는 현재 엔진의 GPU feature **전송 풀** 예산입니다.
  비전 결과 캐시를 끄는 옵션으로 대신 사용하지 않습니다.
- 어댑터와 엔진은 `127.0.0.1`에만 바인딩합니다. 기본 실행과 준비 검사는 텍스트만
  사용합니다. 이미지 검사를 추가하더라도 준비 검사에는 메모리에서 만든 PNG를
  사용하므로 원격 이미지 다운로드가 필요하지 않습니다.

`launch.json`에는 모델/엔진 커밋, 실행 명령, 캐시 환경 변수, 주요 패키지,
GPU 이름·용량·compute capability, 드라이버 및 전력 제한이 들어갑니다.
`preflight.json`은 실제 엔진의 설정과 준비 검사 결과이고,
`engine.log`는 로딩과 실행 로그입니다. 검사에 성공하면 `ready.json`이 만들어지고
엔진 종료 시 제거됩니다. 자동 실행에서는 이 파일과 실행기 프로세스 상태를 함께
확인해야 합니다. `launch.json`, `preflight.json`, `ready.json`의 `probe_scope`에
요청한 검사 범위가 기록됩니다. 기본값은 `["text"]`이며 `--probe-vision`일 때만
`["text", "vision"]`입니다. 준비 검사 값은 정확도·성능 표에 넣지 않습니다.

모든 프로필이 H200의 모든 드라이버·커널 조합에서 실제로 성공했다고 확인한 것은
아닙니다. 소스에서 지원 아키텍처와 HTTP 규약을 확인했으며, 현장 준비 검사에 실패하면
평가를 중지하고 로그로 원인을 확인합니다. 임의로 엔진을 패치하거나 다른 모델로
바꾼 결과를 같은 프로필로 기록하지 마세요.

## CPU 검사

```bash
.venv/bin/python -m unittest discover -s benchmarks/h200 -p 'test_*.py'
.venv/bin/ruff check benchmarks/h200
```

모델별 정밀도 근거는 `models.json`의 고정된 `config_source`입니다. 캐시 제어는
[엔진의 멀티모달 설정](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/arg_groups/fields/mm.py)과
[환경 변수](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/environ.py)를
기준으로 확인했습니다.
