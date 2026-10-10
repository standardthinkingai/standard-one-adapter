"""Bounded functional HTTP smoke for the integrated prefill decision server.

Seven requests at most: one mixed decision, four concurrent single decisions,
and two unsupported-option checks. Latencies are evidence from a shared GPU,
not a performance benchmark.
"""

import argparse
import asyncio
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "어느 팀이 이 요청을 처리해야 하나요?",
        "criteria": {
            "billing": "결제와 환불",
            "technical": "기술 지원",
            "general": "그 밖의 문의",
        },
    },
    "refund": {"type": "noul", "instructions": "환불을 요청했나요?"},
    "urgency": {
        "type": "score",
        "instructions": "처리 긴급도는 어느 정도인가요?",
        "criteria": ["낮음", "보통", "높음"],
    },
}


def _url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base URL must be HTTP(S) without credentials or query")
    return base_url.rstrip("/") + "/v1/systemone"


def _request(model: str, state: str, question_names: tuple[str, ...]) -> dict:
    return {
        "model": model,
        "state": state,
        "questions": {name: QUESTIONS[name] for name in question_names},
        "options": {"temperature": 1.0, "permutations": 1},
    }


def _finite_probability(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("probability must be finite and numeric")
    if not 0 <= value <= 1:
        raise ValueError("probability must lie in [0, 1]")
    return float(value)


def validate_success(body: object, request: dict, model: str) -> dict:
    if not isinstance(body, dict) or body.get("model") != model:
        raise ValueError("response model differs from the requested model")
    usage, metadata, answers = (
        body.get("usage"),
        body.get("metadata"),
        body.get("answers"),
    )
    if not all(isinstance(part, dict) for part in (usage, metadata, answers)):
        raise ValueError("response lacks usage, metadata, or answers")
    if (
        type(usage.get("input_tokens")) is not int
        or usage["input_tokens"] < 1
        or usage.get("output_tokens") != 0
        or type(usage["output_tokens"]) is not int
        or metadata.get("cached_tokens") != 0
        or type(metadata["cached_tokens"]) is not int
        or metadata.get("evaluations") != len(request["questions"])
        or type(metadata["evaluations"]) is not int
    ):
        raise ValueError(
            "expected positive input, zero output/cache, one evaluation per question"
        )
    if set(answers) != set(request["questions"]):
        raise ValueError("response answer names differ from request")

    for name, question in request["questions"].items():
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ValueError(f"answer {name} has the wrong type")
        if question["type"] == "noul":
            # The API exposes only P(true); P(false) is its complement.
            _finite_probability(answer.get("noul"))
            continue
        expected = (
            set(question["criteria"])
            if question["type"] == "choice"
            else {str(index) for index in range(len(question["criteria"]))}
        )
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != expected:
            raise ValueError(f"answer {name} has the wrong probability labels")
        values = [_finite_probability(value) for value in probabilities.values()]
        if not math.isclose(math.fsum(values), 1.0, abs_tol=1e-6):
            raise ValueError(f"answer {name} probabilities are not normalized")
        _finite_probability(answer.get("confidence"))
        if question["type"] == "choice":
            if answer.get("choice") not in expected:
                raise ValueError("choice is not among the probability labels")
        else:
            score = answer.get("score")
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not -1e-9 <= score <= len(expected) - 1 + 1e-9
            ):
                raise ValueError("score must be finite and within its legend")
    return {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cached_tokens": metadata["cached_tokens"],
        "evaluations": metadata["evaluations"],
        "answers": answers,
    }


def validate_rejection(body: object, status: int, expected_code: str) -> dict:
    error = body.get("error") if isinstance(body, dict) else None
    if (
        status != 422
        or not isinstance(error, dict)
        or error.get("code") != expected_code
        or "answers" in body
    ):
        raise ValueError(f"expected HTTP 422 with {expected_code}")
    return {"error_code": expected_code}


async def _case(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    name: str,
    request: dict,
    expected_error: str | None = None,
) -> dict:
    started = time.perf_counter()
    record = {"name": name, "passed": False}
    try:
        response = await client.post(url, json=request)
        record["http_status"] = response.status_code
        body = response.json()
        if expected_error is None:
            if response.status_code != 200:
                raise ValueError(f"expected HTTP 200, got {response.status_code}")
            record.update(validate_success(body, request, model))
        else:
            record.update(
                validate_rejection(body, response.status_code, expected_error)
            )
        record["passed"] = True
    except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError) as error:
        record["error"] = str(error)
    record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return record


async def smoke(
    base_url: str,
    model: str,
    *,
    timeout: float = 60.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    if not model.strip() or not 0 < timeout <= 120 or not math.isfinite(timeout):
        raise ValueError("model and timeout are invalid")
    url = _url(base_url)
    evidence = {
        "kind": "functional_smoke_on_shared_gpu",
        "performance_benchmark": False,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "endpoint": url,
        "model": model,
        "max_concurrent_http_requests": 4,
        "max_total_http_requests": 7,
        "cases": [],
    }
    key = os.environ.get("SGLANG_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(
        timeout=timeout, headers=headers, transport=transport
    ) as client:
        normal = _request(
            model,
            "결제가 두 번 되었습니다. 중복 결제 건을 환불해 주세요.",
            ("route", "refund", "urgency"),
        )
        first = await _case(client, url, model, "mixed_choice_noul_score", normal)
        evidence["cases"].append(first)
        if not first["passed"]:
            evidence["passed"] = False
            return evidence

        concurrent = [
            (
                f"concurrent_{index}_{name}_length_{repeat}",
                _request(
                    model,
                    "환불 관련 문의입니다. " + "추가 맥락입니다. " * repeat,
                    (name,),
                ),
            )
            for index, (name, repeat) in enumerate(
                (("route", 1), ("refund", 6), ("urgency", 18), ("route", 48)),
                start=1,
            )
        ]
        evidence["cases"].extend(
            await asyncio.gather(
                *(
                    _case(client, url, model, name, request)
                    for name, request in concurrent
                )
            )
        )

        prefix = _request(model, "환불 문의", ("route",))
        prefix["options"]["assistant_prefix"] = "답: "
        evidence["cases"].append(
            await _case(
                client,
                url,
                model,
                "reject_assistant_prefix",
                prefix,
                "assistant_prefix_not_supported",
            )
        )
        logprobs = _request(model, "환불 문의", ("route",))
        logprobs["options"]["return_logprobs"] = True
        evidence["cases"].append(
            await _case(
                client,
                url,
                model,
                "reject_return_logprobs",
                logprobs,
                "return_logprobs_unavailable",
            )
        )
    evidence["passed"] = all(case["passed"] for case in evidence["cases"])
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        result = asyncio.run(smoke(args.base_url, args.model, timeout=args.timeout))
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "passed": result["passed"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
