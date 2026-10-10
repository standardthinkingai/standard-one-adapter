import argparse
import json

import httpx
import pytest

from benchmarks.prefill_compare import compare, validate_args
from jev_adapter.benchmarks.data import normalize_record


def fixture_item(record_id):
    return normalize_record(
        {
            "state": record_id,
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a team.",
                    "criteria": {"billing": None, "support": None},
                    "label": "billing",
                },
                "refund": {
                    "type": "noul",
                    "instructions": "Refundable?",
                    "label": True,
                },
            },
            "_meta": {"id": record_id, "source": "policy", "variant": "clean"},
        },
        "fixture",
        "development",
    )


def arguments(tmp_path, **changes):
    path = tmp_path / "fixture.jsonl"
    path.write_text(
        "".join(json.dumps(fixture_item(name)) + "\n" for name in ("one", "two"))
    )
    values = {
        "data": [path],
        "output": tmp_path / "result",
        "sglang_url": "http://sglang",
        "candidate_url": "http://candidate",
        "sglang_key_env": "JEV_SGLANG_API_KEY",
        "candidate_key_env": "JEV_CANDIDATE_API_KEY",
        "model": "decision-model",
        "concurrency": [1, 8, 32],
        "warmup": 2,
        "repeats": 2,
        "seed": 42,
        "limit": None,
        "timeout": 10.0,
        "probability_tolerance": 0.01,
        "assistant_prefix": None,
        "allow_test": False,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def response(payload, *, changed=False):
    labels = list(payload["questions"]["route"]["criteria"])
    route = {label: (0.8 if i == 0 else 0.2) for i, label in enumerate(labels)}
    if len(labels) > 2:
        route = {label: (0.7 if i == 0 else 0.1) for i, label in enumerate(labels)}
    if changed and payload["state"] == "two":
        route = {"billing": 0.2, "support": 0.8}
    return {
        "model": "decision-model",
        "answers": {
            "route": {"type": "choice", "probabilities": route},
            "refund": {"type": "noul", "noul": 0.9},
        },
        "usage": {"input_tokens": 120, "output_tokens": 0},
        "metadata": {
            "evaluations": 2,
            "cached_tokens": 0,
            "adapter_elapsed_ms": 1.0,
        },
    }


@pytest.mark.asyncio
async def test_paired_matrix_uses_same_requests_and_excludes_warmups(tmp_path):
    args = arguments(tmp_path)
    sent = {"sglang": [], "candidate": []}

    def transport(request):
        payload = json.loads(request.content)
        sent[request.url.host].append(payload)
        assert payload["options"]["return_logprobs"] is False
        assert "expected" not in payload and "metadata" not in payload
        assert all("label" not in q for q in payload["questions"].values())
        return httpx.Response(200, json=response(payload))

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        report = await compare(args, client)
    assert report["status"] == "complete"
    assert sent["sglang"] == sent["candidate"]
    assert len(sent["sglang"]) == 3 * (2 + 2 * 2)
    for c in (1, 8, 32):
        phase = report["concurrency"][f"c{c}"]
        assert phase["input_token_parity"]["candidate"]["status"] == "matched"
        assert phase["quality_parity"]["parity_passed"] is True
        assert phase["quality_parity"]["questions"] == 4
        assert phase["candidate_vs_sglang"]["p50_latency_ratio"] > 0
        assert phase["candidate_vs_sglang"]["decisions_per_second_ratio"] > 0
        assert (
            phase["quality_parity"]["quality_by_suite"]["sglang"][
                "fixture/development"
            ]["accuracy"]
            == 1
        )
        for backend in ("sglang", "candidate"):
            result = phase["backends"][backend]
            assert result["attempted_requests"] == 4
            assert result["successful_requests"] == 4
            assert result["errors"] == 0
            assert result["decisions"] == 8
            assert result["input_tokens"] == 480
            assert result["cached_tokens"] == 0
            assert result["decisions_per_second"] > 0
            assert result["input_tokens_per_second"] > 0
            assert result["request_latency_ms"]["p50"] is not None
            assert result["request_latency_ms"]["p95"] is not None
            assert (
                len((args.output / f"c{c}-{backend}.jsonl").read_text().splitlines())
                == 4
            )


@pytest.mark.asyncio
async def test_quality_mismatch_reported_without_transport_errors(tmp_path):
    args = arguments(tmp_path, concurrency=[1], warmup=0, repeats=1)

    def transport(request):
        payload = json.loads(request.content)
        return httpx.Response(
            200, json=response(payload, changed=request.url.host == "candidate")
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        report = await compare(args, client)
    phase = report["concurrency"]["c1"]
    assert report["status"] == "complete"
    assert phase["quality_parity"]["parity_passed"] is False
    assert phase["quality_parity"]["decision_disagreements"] == 1
    assert phase["quality_parity"]["probability_abs_delta"]["max"] == pytest.approx(0.6)
    quality = phase["quality_parity"]["quality_by_suite"]
    assert quality["candidate"]["fixture/development"]["accuracy"] == 0.75


@pytest.mark.asyncio
async def test_equal_aggregate_tokens_cannot_hide_per_request_workload_mismatch(tmp_path):
    args = arguments(tmp_path, concurrency=[1], warmup=0, repeats=1)

    def transport(request):
        payload = json.loads(request.content)
        body = response(payload)
        if request.url.host == "candidate":
            body["usage"]["input_tokens"] = (
                119 if payload["state"] == "one" else 121
            )
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        report = await compare(args, client)
    phase = report["concurrency"]["c1"]
    assert report["status"] == "failed"
    assert phase["backends"]["sglang"]["input_tokens"] == 240
    assert phase["backends"]["candidate"]["input_tokens"] == 240
    parity = phase["input_token_parity"]["candidate"]
    assert parity["status"] == "mismatch"
    assert parity["mismatched_requests"] == 2
    assert phase["quality_parity"]["status"] == "incomparable"
    assert phase["candidate_vs_sglang"]["p50_latency_ratio"] is None
    assert phase["candidate_vs_sglang"]["decisions_per_second_ratio"] is None


@pytest.mark.asyncio
async def test_optional_learned_endpoint_has_separate_quality_comparison(tmp_path):
    args = arguments(
        tmp_path,
        learned_url="http://learned",
        learned_key_env="JEV_LEARNED_API_KEY",
        concurrency=[1, 8, 32],
        warmup=1,
        repeats=1,
    )
    sent = {"sglang": [], "candidate": [], "learned": []}

    def transport(request):
        payload = json.loads(request.content)
        sent[request.url.host].append(payload)
        assert payload["options"]["return_logprobs"] is False
        return httpx.Response(
            200, json=response(payload, changed=request.url.host == "learned")
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        report = await compare(args, client)
    assert report["status"] == "complete"
    assert sent["sglang"] == sent["candidate"] == sent["learned"]
    assert len(sent["learned"]) == 3 * (1 + 2)
    for c in (1, 8, 32):
        phase = report["concurrency"][f"c{c}"]
        assert phase["quality_parity"]["parity_passed"] is True
        assert phase["learned_vs_sglang"]["status"] == "complete"
        assert (
            phase["learned_vs_sglang"]["learned_minus_sglang_by_suite"][
                "fixture/development"
            ]["accuracy"]
            == -0.25
        )
        assert phase["backends"]["learned"]["request_latency_ms"]["p95"] > 0
        assert phase["backends"]["learned"]["decisions_per_second"] > 0
        assert (args.output / f"c{c}-learned.jsonl").exists()


@pytest.mark.asyncio
async def test_learned_workload_mismatch_is_incomparable(tmp_path):
    args = arguments(
        tmp_path,
        learned_url="http://learned",
        learned_key_env="JEV_LEARNED_API_KEY",
        concurrency=[1],
        warmup=0,
        repeats=1,
    )

    def transport(request):
        payload = json.loads(request.content)
        body = response(payload)
        if request.url.host == "learned" and payload["state"] == "one":
            body["usage"]["input_tokens"] = 119
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        report = await compare(args, client)
    phase = report["concurrency"]["c1"]
    assert report["status"] == "failed"
    assert phase["input_token_parity"]["candidate"]["status"] == "matched"
    assert phase["quality_parity"]["status"] == "complete"
    assert phase["input_token_parity"]["learned"]["status"] == "mismatch"
    assert phase["learned_vs_sglang"]["status"] == "incomparable"
    assert phase["learned_vs_sglang"]["p50_latency_ratio"] is None
    assert phase["learned_vs_sglang"]["decisions_per_second_ratio"] is None


@pytest.mark.asyncio
async def test_cache_hit_invalidates_cold_prefill_comparison(tmp_path):
    args = arguments(tmp_path, concurrency=[1], warmup=0, repeats=1)

    def transport(request):
        payload = json.loads(request.content)
        body = response(payload)
        if request.url.host == "candidate" and payload["state"] == "one":
            body["metadata"]["cached_tokens"] = 10
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        report = await compare(args, client)
    phase = report["concurrency"]["c1"]
    assert report["status"] == "failed"
    assert phase["backends"]["candidate"]["cached_tokens"] == 10
    assert phase["backends"]["candidate"]["requests_with_cache_hits"] == 1
    assert phase["backends"]["candidate"]["error_samples"][0]["error_type"] == "CacheHit"
    assert phase["quality_parity"]["status"] == "incomplete"
    assert phase["candidate_vs_sglang"]["p50_latency_ratio"] is None


@pytest.mark.asyncio
async def test_learned_four_choice_failure_or_explicit_filter(tmp_path):
    args = arguments(
        tmp_path,
        learned_url="http://learned",
        learned_key_env="JEV_LEARNED_API_KEY",
        concurrency=[1],
        warmup=0,
        repeats=1,
    )
    many = fixture_item("many")
    names = ["billing", "support", "sales", "legal"]
    many["record"]["questions"]["route"]["criteria"] = dict.fromkeys(names)
    many["expected"]["route"].update(labels=names, target=[1, 0, 0, 0])
    with args.data[0].open("a") as handle:
        handle.write(json.dumps(many) + "\n")
    calls = {"sglang": [], "candidate": [], "learned": []}

    def transport(request):
        payload = json.loads(request.content)
        calls[request.url.host].append(payload["state"])
        if request.url.host == "learned" and payload["state"] == "many":
            return httpx.Response(422)
        return httpx.Response(200, json=response(payload))

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        unfiltered = await compare(args, client)
        args.output = tmp_path / "filtered"
        args.max_options = 3
        filtered = await compare(args, client)
    assert unfiltered["status"] == "failed"
    assert unfiltered["concurrency"]["c1"]["quality_parity"]["parity_passed"] is True
    assert unfiltered["concurrency"]["c1"]["backends"]["learned"]["errors"] == 1
    assert (
        unfiltered["concurrency"]["c1"]["learned_vs_sglang"]["status"] == "incomplete"
    )
    assert filtered["status"] == "complete"
    assert (
        json.loads((args.output / "manifest.json").read_text())[
            "excluded_records_by_max_options"
        ]
        == 1
    )
    assert calls["learned"].count("many") == 1
    assert (
        filtered["concurrency"]["c1"]["backends"]["learned"]["attempted_requests"] == 2
    )


@pytest.mark.asyncio
async def test_failed_request_counted_without_retry_or_quality_headline(tmp_path):
    args = arguments(tmp_path, concurrency=[1], warmup=0, repeats=1)
    calls = {"sglang": 0, "candidate": 0}

    def transport(request):
        calls[request.url.host] += 1
        payload = json.loads(request.content)
        if request.url.host == "candidate" and payload["state"] == "two":
            return httpx.Response(503)
        return httpx.Response(200, json=response(payload))

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        report = await compare(args, client)
    phase = report["concurrency"]["c1"]
    assert calls == {"sglang": 2, "candidate": 2}
    assert report["status"] == "failed"
    assert phase["backends"]["candidate"]["errors"] == 1
    assert phase["backends"]["candidate"]["error_samples"][0]["status_code"] == 503
    assert phase["quality_parity"]["status"] == "incomplete"


@pytest.mark.asyncio
async def test_malformed_response_is_counted_as_an_error(tmp_path):
    args = arguments(tmp_path, concurrency=[1], warmup=0, repeats=1)

    def transport(request):
        payload = json.loads(request.content)
        if request.url.host == "candidate" and payload["state"] == "one":
            return httpx.Response(200, json=["invalid"])
        return httpx.Response(200, json=response(payload))

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        report = await compare(args, client)
    result = report["concurrency"]["c1"]["backends"]["candidate"]
    assert result["errors"] == 1
    assert result["error_samples"][0]["error_type"] == "TypeError"


def test_rejects_unsafe_or_invalid_parameters(tmp_path):
    args = arguments(tmp_path, candidate_url="http://user:secret@candidate")
    with pytest.raises(ValueError, match="candidate URL"):
        validate_args(args)
    args.candidate_url = "http://candidate"
    args.concurrency = [1, 1]
    with pytest.raises(ValueError, match="invalid counts"):
        validate_args(args)
