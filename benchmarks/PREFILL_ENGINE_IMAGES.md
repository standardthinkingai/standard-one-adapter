# 이미지 입력으로 prefill 확률 추출

통합 `POST /v1/systemone` 경로에서 텍스트와 단일·다중 이미지를 함께 처리하도록 구현했다. 이미지 다운로드/디코딩·리사이즈·patch 생성은 기존 SGLang Pixtral processor, 비전 feature와 텍스트 embedding 결합은 기존 Mistral3 모델이 담당한다. 마지막 prefill hidden state에는 기존 작은 readout을 적용한다. 생성 토큰은0이며 이미지 설명을 생성한 뒤 다시 분류하는 단계는 없다.

## 요청

`images`에 base64 PNG·JPEG·WebP data URL(`data:image/png;base64,...`)을 입력 순서대로 넣는다. 결정 워커는 URL을 가져오거나 로컬 경로를 읽지 않으며, 그 밖의 문자열은 422 `invalid_image`로 거절한다. 서비스는 같은 이미지들을 각 질문·선택지 permutation의 평가에 전달한다. 질문 수 × permutations만큼 평가하므로 이미지 인코딩 재사용이나 추가 처리량을 보장하지 않는다.

저장소 루트에서 기존 smoke 도구를 사용할 수 있다. 이 명령은 이미 실행 중인 통합 서버에만 요청한다.

```bash
python jev-adapter/examples/smoke.py \
  --base-url http://127.0.0.1:30000 \
  --request jev-adapter/examples/request-image.json \
  --image /path/to/image.png
```

여러 이미지는 `--image`를 반복한다. 이 예제는 로컬 파일을 data URL로 변환하며 로컬 파일 경로 자체를 API로 보내지 않는다. 직접 JSON을 보낼 때는 다음처럼 기존 요청에 이미지만 추가한다.

```json
{
  "model": "decision-model",
  "state": "첫 번째 이미지의 내용에 답하세요.",
  "images": ["data:image/png;base64,iVBORw0KGgo..."],
  "questions": {
    "contains_person": {
      "type": "noul",
      "instructions": "사람이 보이나요?"
    }
  },
  "options": {"temperature": 1.0, "permutations": 1}
}
```

응답은 기존 `choice/noul/score`와 `usage.output_tokens=0` 형식이다. `usage.input_tokens`는 processor가 이미지 patch 토큰을 확장한 후의 prompt 길이를 사용한다. 이미지가 있는 요청은 그중 이미지 토큰 합계를 `usage.input_tokens_details.image_tokens`로 따로 보고한다. 엔진은 원래 텍스트 ID를 그대로 두고 이미지 marker만 patch 블록으로 바꾸므로, 평가마다 `prompt_tokens - 텍스트 토큰`이 정확한 이미지 토큰 수다. 이미지 토큰을 텍스트와 다른 단가로 청구할 때 이 값을 쓴다.

이미지 한 장의 토큰 수는 긴 변을 1540px 이하로 줄인 뒤 28px patch로 덮은 `행 × (열 + 1)`이다(행마다 `[IMG_BREAK]` 또는 `[IMG_END]` 하나). 이미지 한 장은 최대 3,080토큰이다. 워커는 GPU 작업 전에 이 값을 이미지 헤더에서 계산해 컨텍스트를 검사하며, 고정 3B·8B processor(Transformers 5.12.1)의 실제 출력과 17개 크기에서 일치함을 확인했다. catalog의 `max_image_tokens`는 3080으로 둔다. 이미지가 포함된 prompt가 context 한도를 넘으면 GPU 작업 전에 오류로 처리하며 일부 이미지를 자동으로 잘라 판단하지 않는다.

## 서버 설정과 지원 범위

- [통합 엔진 실행](PREFILL_ENGINE.md)의 모델·revision·readout을 사용한다. Mistral3 아키텍처의 Ministral3B·8B 경로를 대상으로 한다.
- 로컬 비전 인코더와 멀티모달 processor를 함께 로드한다. `--language-model-only`, encoder 분리 실행, KV-free prefill은 이미지 요청에서 지원하지 않는다. 기존 텍스트 요청은 `--language-model-only`에서도 유지한다.
- `TP=PP=DP=1`, decode 비활성, chunked prefill·radix cache 비활성 등 기존 전용 모드 조건은 유지한다. 이미지 전처리·encoder cache와 radix cache는 서로 다른 기능이며 이번 구현으로 캐시 성능을 주장하지 않는다.
- BF16 경로와 기존 고정 checkpoint의 [FP8 설정](PREFILL_ENGINE_FP8.md)을 사용할 수 있게 연결했다. 이미지의 실제 BF16/FP8 forward·정확도·VRAM·속도는 GPU 검증 전이다. 최초 GPU 검증은 prefill graph도 끈 조건에서 시작하고 graph 조건을 별도로 비교한다.
- 오디오·동영상·외부에서 계산한 embedding 입력은 지원하지 않는다. 기존 `return_logprobs=true`와 비어 있지 않은 `assistant_prefix` 제한도 유지한다.
- LM-row readout과 learned head의 출력 계산 코드는 그대로다. 기존3B learned head의 이미지 판단 품질은 검증되지 않았고8B learned head는 없다.

## 토큰과 이미지 순서 보존

MistralCommonBackend는 문자열 `[IMG]`를 일반 텍스트 조각으로 인코딩할 수 있다. 따라서 이미지 URL을 chat tokenizer에 넘기거나 chat 전체를 decode→encode하지 않는다. 검증된 native `[INST]` 뒤에 모델의 atomic image ID를 이미지 순서대로 삽입한다.

Pixtral processor는 이미지 marker만 있는 별도 입력으로 실제 patch 행·`[IMG_BREAK]`·`[IMG_END]`를 생성한다. 이 블록을 원래 native 토큰에 끼워 넣은 뒤 image offset을 계산하고 feature를 전달한다. 이미지 누락, 잘못된 patch 행, 원본에 이미 확장된 이미지 marker, 모호한 대화 경계, 최종 `[/INST]` 변경은 거부한다. 기존 텍스트-only 경로는 이미지 전처리를 호출하지 않는다.

## 검증 상태

CPU 계약·입력 전달·단일/다중 이미지 순서·오류·기존 텍스트/FP8 회귀를 검사한다. 고정 모델 assets와 실제 Transformers5.12.1·MistralCommonBackend·PixtralProcessor를 사용하는 CPU 감사에서 native 멀티모달 토큰과의 일치를 확인한다. 구체적인 파일 SHA·명령·결과는 [개발 기록](../../outputs/prefill-image-support-20260928/REPORT.md)에 보존한다.

H200 서버 종료 지시 이후 원격 실행·배포는 하지 않았다. 기존 H200 지연시간·처리량·VRAM·매출 수치는 텍스트 입력 실측이며 이미지 요청에 적용하지 않는다. GPU 이미지 forward, stock 대비 수치 패리티, 실제 이미지 벤치 정확도와 이미지 크기별 C8/16/32/64/128 성능은 후속 검증 대상이다. MI325X 실측도 없다.
