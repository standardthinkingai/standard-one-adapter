# Qwen3.5-4B 로컬 토크나이저 검증 (2026-09-27)

대상은 `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`의
텍스트 요청, `prompt_wording=served`, assistant prefix 없음이다. 공식 모델 메타데이터에
기록된 `tokenizer.json` SHA256은 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`,
`tokenizer_config.json` SHA256은 `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8`이다.
두 파일과 해시가 일치하는 복사본을 `../jev-lora-data/corpus-1m/batch-100k/release-exact-pilot-2000/assets/`
에서 읽었다. 모델 가중치는 로드하지 않았다.

현재 `jev_adapter.protocol`의 기본 `build_prompt`와 `NativeTokenizer.prepare`로
`benchmarks/data`의 8개 평가 세트 4,852개 질문을 다시 토큰화했다. 라벨은
당시의 `jev_adapter.sglang._label_candidates()` 순서(현재
`jev_adapter.labels.label_candidates("served")`)대로 단일 토큰 및 디코딩 왕복을
통과한 것만 선택했다(`A=32`, `B=33`). 각 입력 ID 배열에 대해
`sha256(json.dumps(ids, separators=(',', ':')).encode())`를 계산하여 저장된
H200 native 실행의 `input_sha256` 및 `input_tokens`와 질문별로 대조한 결과
**4,852/4,852개가 모두 일치**했다. 저장된 실행은 Transformers 5.12.1,
이번 오프라인 재검증은 5.17.0을 사용했다. 저장된 실행의 원본 코드는
`../outputs/h200-baseline-benchmark-2026-09-21/server/jev-adapter/`에 있고,
그 SHA256은 `../outputs/jev-full-performance-20260922/qwen-native/result.json`의
`source_sha256`과 일치한다. 질문별 기대 해시는 같은 디렉터리의
`predictions.jsonl`에서 `backend=torch` 행을 사용했다.

별도로 저장된 SGLang HTTP C1 실행의 단일 질문 요청 3,543개를 같은 질문의
native 기록과 연결했을 때 `input_tokens`가 **3,543/3,543개 일치**했다.
HTTP 실행은 실제 입력 ID 배열을 저장하지 않았으므로, 이 비교만으로 HTTP와
native의 **정확한 토큰 ID 일치**를 증명하지는 못한다. 두 경로의 정확한 ID는
같은 리비전의 실행 중인 SGLang 서버에서
`benchmarks/qwen_tokenizer_parity.py`로 확인해야 한다.

참고용 Mac CPU 측정(Transformers 5.17.0, 전체 질문에 균등 간격으로 선택한
100개)에서 `NativeTokenizer.prepare`의 중앙값은 0.559ms였다. 라벨 2개는
0.44ms, 드문 라벨 77~78개 요청은 약 17ms였다. 이 값에는 H200의 GPU 실행이나
HTTP 왕복이 포함되지 않으며, 서빙 지연 개선의 측정값으로 사용하지 않는다.
