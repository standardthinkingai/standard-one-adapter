import asyncio
import json
import math

import httpx
import pytest

from benchmarks.prefill_engine_smoke import smoke, validate_success


@pytest.mark.asyncio
async def test_integrated_smoke_is_bounded_and_validates_all_response_types():
    calls = []
    active = peak = 0

    async def handler(request):
        nonlocal active, peak
        body = json.loads(request.content)
        calls.append(body)
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.005)
            options = body.get("options", {})
            if options.get("return_logprobs"):
                return httpx.Response(
                    422, json={"error": {"code": "return_logprobs_unavailable"}}
                )
            if options.get("assistant_prefix"):
                return httpx.Response(
                    422, json={"error": {"code": "assistant_prefix_not_supported"}}
                )
            answers = {}
            for name, question in body["questions"].items():
                kind = question["type"]
                if kind == "noul":
                    answers[name] = {"type": kind, "noul": 0.7}
                else:
                    names = (
                        list(question["criteria"])
                        if kind == "choice"
                        else [str(i) for i in range(len(question["criteria"]))]
                    )
                    probabilities = {name: 1 / len(names) for name in names}
                    answer = {
                        "type": kind,
                        "probabilities": probabilities,
                        "confidence": 0.0,
                    }
                    if kind == "choice":
                        answer["choice"] = names[0]
                    else:
                        answer["score"] = (len(names) - 1) / 2
                    answers[name] = answer
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "answers": answers,
                    "usage": {
                        "input_tokens": len(body["state"].split()) + 10,
                        "output_tokens": 0,
                    },
                    "metadata": {
                        "cached_tokens": 0,
                        "evaluations": len(body["questions"]),
                    },
                },
            )
        finally:
            active -= 1

    evidence = await smoke(
        "http://decision-server:30000",
        "decision-model",
        transport=httpx.MockTransport(handler),
    )
    assert evidence["passed"] is True
    assert evidence["performance_benchmark"] is False
    assert len(evidence["cases"]) == len(calls) == 7
    assert 1 < peak <= 4
    assert {case["name"] for case in evidence["cases"][-2:]} == {
        "reject_assistant_prefix",
        "reject_return_logprobs",
    }
    assert all(case["latency_ms"] >= 0 for case in evidence["cases"])
    assert all(case["output_tokens"] == 0 for case in evidence["cases"][:5])
    assert all(case["cached_tokens"] == 0 for case in evidence["cases"][:5])
    lengths = [len(call["state"]) for call in calls[1:5]]
    assert lengths == sorted(lengths) and len(set(lengths)) == 4


def test_validation_rejects_nonfinite_or_unnormalized_probabilities():
    request = {
        "questions": {
            "route": {
                "type": "choice",
                "criteria": {"billing": None, "technical": None},
            }
        }
    }
    body = {
        "model": "decision-model",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.2,
                "probabilities": {"billing": 0.6, "technical": 0.4},
            }
        },
        "usage": {"input_tokens": 12, "output_tokens": 0},
        "metadata": {"cached_tokens": 0, "evaluations": 1},
    }
    validate_success(body, request, "decision-model")
    body["answers"]["route"]["probabilities"]["technical"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        validate_success(body, request, "decision-model")
    body["answers"]["route"]["probabilities"]["technical"] = 0.2
    with pytest.raises(ValueError, match="normalized"):
        validate_success(body, request, "decision-model")
