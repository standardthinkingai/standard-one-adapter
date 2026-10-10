# MI325X 공식 BF16 기본 모델 런타임

MI325X 서버에서 **GPU 한 장만 선택**해 모델 4종을 순서대로 실행합니다. 여러 GPU가
달린 서버에서도 TP=1을 유지합니다. 추가 학습·LoRA 없이 H200 실험과 같은 공식
체크포인트 리비전을 사용합니다. 여기서 기본 모델은 학습 전 랜덤 가중치가 아니라
**제조사가 공개한 체크포인트에 추가 학습을 하지 않은 상태**를 뜻합니다.

현재 상태는 **CPU 검사 완료, MI325X 하드웨어·모델 실행 검증 대기**입니다.
이미지 태그의 공개 여부와 AMD 지원 문서를 확인했지만, 실제 Docker digest와 드라이버
호환성, Qwen hybrid/MoE 커널, Ministral 텍스트 scoring은 서버에서 검사해야 합니다.
실패한 모델은 실패로 남기며 다른 모델·정밀도·엔진으로 자동 대체하지 않습니다.

| 프로필 | 공식 체크포인트 | 입력 처리 |
| --- | --- | --- |
| `qwen35-4b-bf16` | Qwen/Qwen3.5-4B | 엔진 HTTP chat tokenizer |
| `qwen36-27b-bf16` | Qwen/Qwen3.6-27B | 엔진 HTTP chat tokenizer |
| `qwen36-35b-a3b-bf16` | Qwen/Qwen3.6-35B-A3B | 엔진 HTTP chat tokenizer |
| `ministral3-3b-bf16` | mistralai/Ministral-3-3B-Instruct-2512-BF16 | 고정 리비전 공식 native text tokenizer |

정확한 리비전은 [models.json](models.json)에 있습니다. Ministral에는 H200에서
확인한 native 제어 토큰 보존 경로를 재사용합니다. 현재 Ministral 입력은 텍스트만
지원합니다. 텍스트 벤치에 비전 encoder를 제거하는 변경을 추가하지 않습니다.

같은 게임 상태의 텍스트 입력과 렌더링 이미지 입력을 비교할 때 Qwen 3종은
`launch ... --probe-vision`으로 이미지 준비 검사까지 통과해야 합니다. 모델별
`vision_validation`은 아직 `pending_runtime_probe`입니다. Ministral의 기존
HTTP 템플릿 왕복은 native 제어 토큰을 보존하지 못하므로 이미지 조건을
`unsupported_native_text_tokenizer`로 기록하고 비교에서 제외합니다. 텍스트 결과를
이미지 결과로 대신 쓰거나 다른 토크나이저로 조용히 전환하지 않습니다.

## 준비된 서버 확인

아래 명령은 **전체 프로젝트 루트**에서 실행합니다. 서버에 Docker와 AMD 호스트
드라이버가 이미 준비되어 있어야 합니다. 이 도구는 호스트 드라이버·Docker·패키지
저장소·BIOS·전력 제한을 설치하거나 변경하지 않습니다. Ubuntu LTS 또는 조직에서
지원하는 RHEL 계열 호스트의 실제 OS/커널/드라이버를 먼저 확인하세요. 패키지 설치가
필요하면 이 결과를 확인한 뒤 서버 운영 방식에 맞춰 준비합니다.

```bash
python3 jev-adapter/benchmarks/mi325x/runtime.py host-preflight \
  --execute --output outputs/mi325x/host-preflight.json
```

`/etc/os-release`, Docker 상태, 사용 가능한 AMD SMI/ROCm SMI, amdgpu 모듈,
`/dev/kfd`, render device와 PCI 매핑을 읽어 저장합니다. 이 inventory 자체가
드라이버 호환성 통과 판정은 아닙니다. `nvidia-smi`는 호출하지 않습니다.

## 공식 ROCm 이미지를 digest로 고정

기본 후보는 `lmsysorg/sglang:v0.5.20-rocm724-mi30x`입니다. [SGLang 공식 AMD 안내](https://docs.sglang.io/docs/hardware-platforms/amd_gpu)는
공식 ROCm Docker 이미지를 안내하며, [공식 Docker Hub 태그 목록](https://hub.docker.com/r/lmsysorg/sglang/tags?name=rocm)에
이 태그가 공개되어 있습니다. [AMD 사양표](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)에
따라 MI325X의 target은 `gfx942`, 메모리는 256 GiB입니다. `mi35x/gfx950` 이미지를
대신 사용하지 않습니다. H200용 CUDA wheel도 이 환경에 설치하지 않습니다.

```bash
python3 jev-adapter/benchmarks/mi325x/runtime.py lock-image \
  --execute --output outputs/mi325x/image-lock.json
```

이 명령은 큰 공식 이미지 다운로드를 수행합니다. 이어 태그와 immutable digest의
로컬 image ID가 같은지 검사하고 Linux/amd64 결과를 저장합니다. digest를 미리
추측하거나 `latest`를 사용하지 않습니다. 이후 같은 `image-lock.json`을 재사용하면
태그가 갱신돼도 기존 실험의 이미지 참조를 유지할 수 있습니다.

호스트 검사 결과에서 **실제로 사용할 MI325X의 render node를 선택**합니다.
아래 `/dev/dri/renderD128`은 예시이며 SMI의 PCI 주소와 `render_pci_devices`를
대조해 선택하세요. 전체 `/dev/dri`를 넘기지 않습니다.

```bash
export JEV_RENDER_DEVICE=/dev/dri/renderD128
JEV_IMAGE_REF=$(python3 -c 'import json; print(json.load(open("outputs/mi325x/image-lock.json"))["image_ref"])')
docker run --rm -it --init \
  --device=/dev/kfd --device="$JEV_RENDER_DEVICE" \
  --shm-size=32g --group-add=video \
  -e JEV_CONTAINER_IMAGE="$JEV_IMAGE_REF" \
  -e HF_TOKEN \
  -v "$PWD:/work" -w /work \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$JEV_IMAGE_REF" bash
```

엔진과 어댑터, 평가기는 모두 같은 컨테이너에서 localhost로 통신하므로 기본 예시는
호스트 network·공개 포트·privileged 권한이 필요하지 않습니다. 서버 정책 때문에
추가 권한이 필요하면 실제 오류를 확인해 조정합니다. JEV_CONTAINER_IMAGE는 위
digest 참조와 lock 파일의 일치 검사에 쓰는 실행 기록입니다. 컨테이너 내부에서
호스트 Docker daemon을 독립적으로 증명하는 원격 attestation은 아닙니다.

## 컨테이너 내부 실행

이미지의 엔진 Python을 먼저 기록하고 평가용 Python 환경을 별도로 만듭니다.
이미지의 Torch/SGLang/ROCm 패키지를 pip로 갱신하지 않습니다.

```bash
export JEV_ENGINE_PYTHON=$(command -v python3)
python3 -m venv /tmp/jev-mi325x-client
/tmp/jev-mi325x-client/bin/python -m pip install -e './jev-adapter[native-tokenizer]'

/tmp/jev-mi325x-client/bin/python jev-adapter/benchmarks/mi325x/runtime.py \
  launch qwen35-4b-bf16 --engine-python "$JEV_ENGINE_PYTHON" --gpu 0

/tmp/jev-mi325x-client/bin/python jev-adapter/benchmarks/mi325x/runtime.py \
  launch qwen35-4b-bf16 --engine-python "$JEV_ENGINE_PYTHON" --gpu 0 \
  --image-lock outputs/mi325x/image-lock.json \
  --output outputs/mi325x/qwen35-first/engine --execute
```

Docker가 선택한 GPU 한 장은 컨테이너 안에서 **인덱스 0**입니다. 호스트의 GPU
인덱스를 `--gpu`에 다시 넣지 않습니다. 런타임은 `ROCR_VISIBLE_DEVICES`로 한 장을
선택하고 이후 HIP 관점의 인덱스는 0으로 제한합니다. 기존 singleton ROCr allocation은
보존하며, 여러 GPU 또는 CUDA/HIP의 모호한 중복 mask가 있으면 명확히 실패합니다.
일반 환경 변수를 전부 지우지 않습니다. [AMD HIP visibility 문서](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.1/reference/env_variables.html)를
따르며 실제 torch HIP device count가 정확히 1인지 다시 검사합니다.

`--execute`가 없으면 이미지 다운로드·HTTP 요청·Torch import·GPU 초기화·출력 파일
생성 없이 명령만 출력합니다. `probe`, `host-preflight`, `lock-image`도 동일합니다.
실행 시에는 image lock과 컨테이너 marker를 먼저 확인한 뒤, HIP 빌드 여부, ROCm 7.2,
SGLang 0.5.20, MI325X/gfx942/메모리, 실행 포트와 모델 설정을 검증합니다. 메모리
partition이 잡힌 GPU는 이 초기 256 GiB 기준선에서 제외됩니다.

준비 검사는 기존 H200 모듈의 플랫폼과 무관한 HTTP helper만 재사용합니다.
H200 GPU 검사나 CUDA 설치 검사는 호출하지 않습니다. 엔진 공개 API의
`/health_generate`, `/server_info`, `/model_info`, `/v1/tokenize`, `/v1/detokenize`,
`/generate`를 사용해 **출력 0토큰·요청한 라벨 logprob** 경로를 실제로 확인합니다.
LoRA·speculative/MTP·chunked prefill·KV/멀티모달 재사용 설정도 검사합니다.
준비가 끝나면 `ready.json`을 생성하며 종료 시 삭제합니다. SIGTERM/Ctrl-C/실패 때
이 실행기가 시작한 엔진 process group만 정리합니다.

공개 엔진 재검사는 다음과 같습니다. 원래의 GPU·컨테이너 기록을 새로 증명하지
않으므로 최초 `launch.json`을 함께 보관합니다.

```bash
/tmp/jev-mi325x-client/bin/python jev-adapter/benchmarks/mi325x/runtime.py \
  probe qwen35-4b-bf16 --execute
```

## 고정 조건과 결과 해석

- 공식 BF16, TP=1, 추가 학습·LoRA·speculative decoding·MTP 없음.
- context 8192, 엔진 max_running_requests 32, mem_fraction_static 0.8,
  chunked prefill -1, radix cache와 vision embedding 재사용 끔.
- 모델 로딩과 첫 커널/JIT 준비, warmup은 결과 지연에서 제외합니다.
- 엔진·토크나이저·어댑터 포함 end-to-end 지연을 기록합니다. ROCm 커널만의 시간으로
  해석하지 않으며 H200와 비교할 때 GPU뿐 아니라 PyTorch/backend 차이도 함께 봅니다.
- 이미지 내 기본 AITER/ROCm 설정은 유지하고 `launch.json`에 기록합니다.
  호스트의 다른 프로세스 부하·GPU partition·전력 제한은 현장 inventory로 확인합니다.

`launch.json`의 `engine.revision`은 **OCI image digest**이며
`revision_kind=oci_image_digest`로 구분합니다. SGLang Git commit으로 위장하지
않습니다. 패키지·HIP·GPU 정보와 image lock도 같은 파일에 들어갑니다.
`preflight.json`은 텍스트 준비 검사 결과이고 `engine.log`는 엔진 로그입니다.
상위 multi-suite 실행은 [MI325X 평가 실행기](../../../jev-mi325x-bench/README.md)를
참조하세요.

## CPU 검사

```bash
python3 -m unittest discover -s jev-adapter/benchmarks/mi325x -p 'test_*.py'
jev-adapter/.venv/bin/ruff check jev-adapter/benchmarks/mi325x
```

CPU 검사는 dry-run의 무부작용, 모델 리비전/정밀도, AMD 한 장 선택, 잘못된 CUDA
빌드·다중 GPU·LoRA 거부, digest gate, 소유 프로세스 정리를 mock으로 검증합니다.
이 검사를 GPU 성능이나 ROCm 모델 지원을 확인한 결과로 표시하지 않습니다.
