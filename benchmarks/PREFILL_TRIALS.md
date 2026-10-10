# 저장된 프리필 실행 쌍 3회 이상 요약

`prefill_trials.py`는 이미 저장된 `prefill_engine_bench.py run` 결과를 읽습니다. 서버나 GPU에 연결하지 않습니다. 각 회차는 **기준선 실행 디렉터리 하나와 후보 실행 디렉터리 하나**로 지정합니다.

```bash
.venv/bin/python benchmarks/prefill_trials.py \
  --pair /path/to/trial1-baseline /path/to/trial1-candidate \
  --pair /path/to/trial2-baseline /path/to/trial2-candidate \
  --pair /path/to/trial3-baseline /path/to/trial3-candidate \
  --output /path/to/three-trial-summary.json
```

각 실행은 동일한 모델·revision·정밀도·GPU 종류·프롬프트/라벨 계약, 입력 corpus·순서·토큰 감사, 엔진 설정, 벤치 클라이언트 코드로 측정해야 합니다. 기준선과 후보의 head는 달라도 되지만 같은 역할의 head 방식과 artifact는 회차 사이에 바뀌면 안 됩니다. 이 도구는 각 쌍에 기존 `compare_saved()`를 호출해 원문 응답과 저장 요약을 재검증하고, C **8/16/32/64/128** 전체를 요구합니다. 같은 디렉터리를 반복 지정하거나 원문 실행 파일을 복사해 다른 회차로 사용하면 거부합니다. 디렉터리·타임스탬프·원문 해시가 다르다는 사실만으로 실행 환경의 독립성이 증명되지는 않습니다.

각 C의 양쪽 실제 측정 구간은 각각 **30초 이상**이어야 하며, 개별 `run`도 `--min-measure-seconds 30`으로 실행해야 합니다. 오류·캐시 토큰·측정 실패, 요청별 입력 토큰 수 및 외부 토큰 ID 감사 상태, LM 경로의 확률/표시 답 패리티를 회차별로 표시합니다. 학습 head의 품질은 별도로 표시하며 LM 확률 패리티를 요구하지 않습니다.

출력의 `trials`에는 각 회차·C별 p50/p95/p99 클라이언트 지연, RPS, decisions/s, input tokens/s, 후보÷기준선 비율, 실제 품질 점수와 검증 상태가 그대로 남습니다. `per_concurrency`는 **쌍별 비율**의 median/min/max와 각 지표가 퇴행한 회차 번호를 표시합니다. 요청들을 회차 사이에 합치거나 평균으로 회귀 회차를 숨기지 않습니다. 정확도 역시 각 실행의 첫 측정 pass 결과를 그대로 보존하며 성능 반복 요청을 추가 품질 표본으로 세지 않습니다.

`local_checks_passed`는 저장 파일로 확인 가능한 조건만 통과했다는 뜻입니다. 출력의 `final_claim_eligible`은 항상 `false`입니다. GPU 독점 사용과 다른 GPU 작업 유무, 순정 SGLang wheel의 출처 및 실제 실행 조건, 회차 사이 열·부하 독립성은 별도 증거로 확인해야 합니다. 이 도구는 성능 또는 품질 승자를 자동 선언하지 않습니다.
