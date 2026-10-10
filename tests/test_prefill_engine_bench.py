import argparse
import asyncio
import copy
import hashlib
import json

import httpx
import pytest
from benchmarks import prefill_engine_bench as bench
from benchmarks.prefill_compare import request_for
from benchmarks.prefill_engine_bench import (
    _same_workload,
    canonical_sha256,
    compare_saved,
    jobs_for,
    quality_summary,
    read_settings,
    run,
)
from jev_adapter import inprocess
from jev_adapter.benchmarks.data import normalize_record
from jev_adapter.benchmarks.run import load_data

REVISION = "a" * 40
MODEL_PATH = "mistralai/Ministral-3-3B-Instruct-2512-BF16"


def graph_comparison_manifest():
    return {
        "data": [],
        "max_options_in_workload": 2,
        "settings": {
            "hardware": "H200",
            "model_path": MODEL_PATH,
            "model_revision": REVISION,
            "precision": "bfloat16",
            "prompt_contract": "served-v1",
            "label_contract": "A-B-single-token-v1",
            "labels": ["A", "B"],
            "token_ids": [11, 12],
        },
        "engine": {
            "model_info": {"model_path": MODEL_PATH},
            "server_info": {
                "cuda_graph_config": {
                    phase: {"backend": "disabled", "max_bs": 128}
                    for phase in ("decode", "prefill")
                }
            },
        },
    }


def test_graph_comparison_accepts_missing_optional_nulls_without_mutating():
    stock = graph_comparison_manifest()
    candidate = copy.deepcopy(stock)
    for settings in candidate["engine"]["server_info"]["cuda_graph_config"].values():
        settings.update(max_context_size=None, max_seq_len=None)
    original = copy.deepcopy((stock, candidate))
    _same_workload(stock, candidate)
    _same_workload(candidate, stock)
    assert (stock, candidate) == original


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_context_size", 2048),
        ("max_seq_len", 2048),
        ("max_bs", 256),
        ("unknown_option", None),
    ],
)
def test_graph_comparison_rejects_nonnull_limits_and_other_changes(key, value):
    stock = graph_comparison_manifest()
    candidate = copy.deepcopy(stock)
    candidate["engine"]["server_info"]["cuda_graph_config"]["prefill"][key] = value
    with pytest.raises(
        ValueError, match="engine runtime setting cuda_graph_config differs"
    ):
        _same_workload(stock, candidate)


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_comparison_accepts_matching_fp32_head_flags(enabled):
    stock = graph_comparison_manifest()
    stock["engine"]["server_info"]["enable_fp32_lm_head"] = enabled
    candidate = copy.deepcopy(stock)
    _same_workload(stock, candidate)


@pytest.mark.parametrize(
    ("stock_flag", "candidate_flag"),
    [(False, True), (True, False), (None, True), (True, None)],
)
def test_runtime_comparison_rejects_fp32_head_flag_mismatch(stock_flag, candidate_flag):
    stock = graph_comparison_manifest()
    candidate = copy.deepcopy(stock)
    for manifest, flag in ((stock, stock_flag), (candidate, candidate_flag)):
        if flag is not None:
            manifest["engine"]["server_info"]["enable_fp32_lm_head"] = flag
    with pytest.raises(
        ValueError, match="engine runtime setting enable_fp32_lm_head differs"
    ):
        _same_workload(stock, candidate)


def fixture(record_id):
    return normalize_record(
        {
            "state": record_id,
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a team.",
                    "criteria": {"billing": None, "support": None},
                    "label": "billing",
                }
            },
            "_meta": {"id": record_id, "source": "policy", "variant": "clean"},
        },
        "fixture",
        "development",
    )


def files(tmp_path):
    data = tmp_path / "fixture.jsonl"
    data.write_text(
        "".join(json.dumps(fixture(name)) + "\n" for name in ("one", "two"))
    )
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hardware": "H200",
                "model_path": MODEL_PATH,
                "model_revision": REVISION,
                "precision": "bfloat16",
                "prompt_contract": "served-v1",
                "label_contract": "A-B-single-token-v1",
                "labels": ["A", "B"],
                "token_ids": [11, 12],
                "head_mode": "logprobs",
                "head_artifact": None,
            }
        )
    )
    items, _ = load_data([data], "decision-model")
    requests = [request_for(item, "decision-model", None) for item in items]
    audit = tmp_path / "token-audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workload_sha256": canonical_sha256(requests),
                "model_revision": REVISION,
                "prompt_contract": "served-v1",
                "label_contract": "A-B-single-token-v1",
                "requests": [
                    {
                        "request_sha256": canonical_sha256(request),
                        "input_token_ids_sha256": canonical_sha256(
                            [11, 12, request["state"]]
                        ),
                    }
                    for request in requests
                ],
            }
        )
    )
    return data, settings, audit


def args(tmp_path, **changes):
    data, settings, audit = files(tmp_path)
    values = {
        "base_url": "http://baseline",
        "engine_url": None,
        "model": "decision-model",
        "settings_manifest": settings,
        "token_audit": audit,
        "data": [data],
        "output": tmp_path / "baseline",
        "concurrency": [8, 16],
        "warmup": 16,
        "repeats": 8,
        "min_measure_seconds": 0.0,
        "limit": 2,
        "seed": 42,
        "timeout": 10.0,
        "key_env": "SGLANG_API_KEY",
        "allow_test": False,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def body(payload, service_capacity=128, wrong_choice=False):
    return {
        "model": "decision-model",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "support" if wrong_choice else "billing",
                "confidence": 0.8,
                "probabilities": {"billing": 0.8, "support": 0.2},
            }
        },
        "usage": {
            "input_tokens": 100 if payload["state"] == "one" else 120,
            "output_tokens": 0,
        },
        "metadata": {
            "evaluations": 1,
            "cached_tokens": 0,
            "adapter_elapsed_ms": 1.0,
            "service_max_concurrency": service_capacity,
        },
    }


def transport(
    *,
    cache_hit=False,
    output_token=False,
    revision=REVISION,
    service_capacity=128,
    delay=0.002,
    wrong_choice=False,
    readout_path=None,
    fp32_lm_head=False,
):
    async def respond(request):
        if request.url.path == "/model_info":
            return httpx.Response(
                200,
                json={"model_path": MODEL_PATH, "served_model_name": "decision-model"},
            )
        if request.url.path == "/server_info":
            return httpx.Response(
                200,
                json={
                    "model_path": MODEL_PATH,
                    "revision": revision,
                    "dtype": "bfloat16",
                    "enable_fp32_lm_head": fp32_lm_head,
                    "quantization": None,
                    "disable_radix_cache": True,
                    "mm_preprocess_cache_size_mb": 0,
                    "enable_prefix_mm_cache": False,
                    "enable_mm_global_cache": False,
                    "speculative_algorithm": None,
                    "max_running_requests": 128,
                    "decision_readout_path": readout_path,
                    "tp_size": 1,
                    "pp_size": 1,
                    "dp_size": 1,
                    "cpu_offload_gb": 0,
                    "context_length": 8192,
                    "max_total_tokens": 8192,
                    "max_prefill_tokens": 8192,
                    "mem_fraction_static": 0.7,
                    "disable_decode_cuda_graph": True,
                },
            )
        if delay:
            await asyncio.sleep(delay)
        payload = json.loads(request.content)
        result = body(payload, service_capacity, wrong_choice)
        if cache_hit:
            result["metadata"]["cached_tokens"] = 1
        if output_token:
            result["usage"]["output_tokens"] = 1
        return httpx.Response(200, json=result)

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_engine_snapshot_preserves_fp32_head_flag(tmp_path, enabled):
    arguments = args(tmp_path)
    settings, _ = read_settings(arguments.settings_manifest)
    async with httpx.AsyncClient(transport=transport(fp32_lm_head=enabled)) as client:
        snapshot = await bench.engine_snapshot(
            client, "http://engine", {}, settings, 128
        )
    assert snapshot["server_info"]["dtype"] == "bfloat16"
    assert snapshot["server_info"]["enable_fp32_lm_head"] is enabled


@pytest.mark.asyncio
async def test_closed_loop_single_endpoint_warmup_and_raw_evidence(tmp_path):
    arguments = args(tmp_path)
    async with httpx.AsyncClient(transport=transport()) as client:
        report = await run(arguments, client)
    assert report["status"] == "complete"
    manifest = json.loads((arguments.output / "manifest.json").read_text())
    assert manifest["http_client"] == {"kind": "external", "class": "httpx.AsyncClient"}
    assert manifest["http_pool_max_connections"] == 16
    assert manifest["request_options"]["return_logprobs"] is False
    assert manifest["token_id_audit"]["input_token_ids_sha256"]
    assert manifest["job_order_sha256"] == canonical_sha256(jobs_for([1, 2], 8, 42))
    assert manifest["repeats_by_concurrency"] == {"8": 8, "16": 8}
    assert manifest["job_order_sha256_by_concurrency"] == {
        "8": manifest["job_order_sha256"],
        "16": manifest["job_order_sha256"],
    }
    for c in (8, 16):
        phase = report["concurrency"][f"c{c}"]
        assert phase["attempted_requests"] == 16
        assert phase["successful_requests"] == 16
        assert phase["errors"] == 0
        assert phase["client_http_inflight_peak"] >= c
        assert phase["service_max_concurrency"] == 128
        assert phase["client_latency_ms"]["p50"] is not None
        assert phase["client_latency_ms"]["p95"] is not None
        assert phase["client_latency_ms"]["p99"] is not None
        assert phase["requests_per_second"] > 0
        assert phase["decisions_per_second"] > 0
        assert phase["input_tokens_per_second"] > 0
        assert phase["input_tokens"] == 1760
        assert phase["cached_tokens"] == 0
        assert phase["quality_overall"]["n"] == 2
        assert phase["quality_overall"]["acc"] == 1.0
        assert phase["quality_by_type"]["choice"]["n"] == 2
        assert phase["quality_by_type"]["noul"] is None
        assert phase["quality_by_type"]["score"] is None
        raw = [
            json.loads(line)
            for line in (arguments.output / f"c{c}" / "raw_outcomes.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(raw) == 16
        assert (
            len((arguments.output / f"c{c}" / "warmup.jsonl").read_text().splitlines())
            == 16
        )
        assert all(
            row["status"] == "success" and row["raw_response_text"] for row in raw
        )
        assert all(row["admitted_ns"] < row["completed_ns"] for row in raw)


@pytest.mark.asyncio
async def test_saved_pair_requires_same_token_ids_and_keeps_single_run_scope(tmp_path):
    arguments = args(tmp_path, concurrency=[8], repeats=4, warmup=0)
    async with httpx.AsyncClient(transport=transport()) as client:
        await run(arguments, client)
        baseline = arguments.output
        arguments.output = tmp_path / "candidate"
        arguments.base_url = "http://candidate"
        await run(arguments, client)
    result = compare_saved(baseline, arguments.output)
    assert result["status"] == "comparable"
    assert result["head_mode_array"] == ["logprobs", "logprobs"]
    assert result["token_id_audit_status"] == "matched_external_audits"
    assert result["comparison_scope"] == "single_run_pair"
    assert result["repeated_trials_required_for_final_claim"] is True
    assert result["final_claim_eligible"] is False
    assert result["per_concurrency"]["c8"]["quality_parity"]["parity_passed"]
    assert (
        result["per_concurrency"]["c8"]["displayed_answer_parity"]["status"]
        == "matched"
    )
    assert (
        result["per_concurrency"]["c8"]["quality"]["baseline"]["overall"]["acc"] == 1.0
    )
    assert (
        result["per_concurrency"]["c8"]["quality"]["candidate"]["by_type"]["choice"][
            "n"
        ]
        == 2
    )
    assert result["quality_winner"] is None
    candidate_manifest = arguments.output / "manifest.json"
    changed = json.loads(candidate_manifest.read_text())
    changed["token_id_audit"]["input_token_ids_sha256"][0] = "f" * 64
    candidate_manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="token ID hashes differ"):
        compare_saved(baseline, arguments.output)


@pytest.mark.asyncio
async def test_cache_output_and_revision_fail_closed(tmp_path):
    arguments = args(tmp_path, concurrency=[8], repeats=4, warmup=0)
    async with httpx.AsyncClient(transport=transport(revision="b" * 40)) as client:
        with pytest.raises(ValueError, match="revision"):
            await run(arguments, client)
    assert (
        json.loads((arguments.output / "manifest.json").read_text())["status"]
        == "failed"
    )
    arguments.output = tmp_path / "cache"
    async with httpx.AsyncClient(transport=transport(cache_hit=True)) as client:
        report = await run(arguments, client)
    assert report["status"] == "failed"
    assert report["concurrency"]["c8"]["errors"] == 8
    arguments.output = tmp_path / "output"
    async with httpx.AsyncClient(transport=transport(output_token=True)) as client:
        report = await run(arguments, client)
    assert report["status"] == "failed"
    assert report["concurrency"]["c8"]["errors"] == 8


def test_settings_and_request_bound(tmp_path):
    arguments = args(tmp_path)
    settings, digest = read_settings(arguments.settings_manifest)
    assert settings["head_mode"] == "logprobs"
    assert len(digest) == 64
    arguments.repeats = 1_000_001
    with pytest.raises(ValueError, match="invalid"):
        asyncio.run(run(arguments))
    arguments.repeats = 300_000
    with pytest.raises(ValueError, match="1000000-request bound"):
        asyncio.run(run(arguments))


def test_quality_summary_keeps_score_metrics_and_excludes_synthetic_unknown():
    score = {
        "suite": "real",
        "split": "development",
        "variant": "clean",
        "source": "policy",
        "type": "score",
        "p": [0.1, 0.2, 0.7],
        "label": 2,
        "keys": ["0", "1", "2"],
    }
    synthetic = {**score, "suite": "synthetic-latency-v1", "source": "generated"}
    unknowable = {**score, "source": "unknowable"}
    quality = quality_summary([score, synthetic, unknowable])
    assert quality["quality_overall"]["n"] == 1
    assert quality["quality_by_type"]["score"]["acc"] == 1.0
    assert quality["quality_by_type"]["score"]["score_mae"] == pytest.approx(0.4)
    assert "ranked_probability_score" in quality["quality_by_type"]["score"]
    assert quality["quality_by_type"]["choice"] is None
    assert list(quality["quality_by_suite"]) == ["real/development"]


@pytest.mark.asyncio
async def test_learned_head_comparison_reports_real_quality_without_winner(
    tmp_path, monkeypatch
):
    arguments = args(tmp_path, concurrency=[8], repeats=4, warmup=0)
    async with httpx.AsyncClient(transport=transport()) as client:
        await run(arguments, client)
    baseline = arguments.output

    artifact = tmp_path / "learned-readout.npz"
    artifact.write_bytes(b"fake-artifact-for-comparison-test")
    settings = json.loads(arguments.settings_manifest.read_text())
    settings["head_mode"] = "learned"
    settings["head_artifact"] = {
        "path": str(artifact),
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    arguments.settings_manifest.write_text(json.dumps(settings))
    monkeypatch.setattr(
        inprocess,
        "_load_metadata",
        lambda path: {
            "mode": "learned",
            "labels": settings["labels"],
            "token_ids": settings["token_ids"],
            "engine_model_path": settings["model_path"],
            "revision": settings["model_revision"],
        },
    )
    arguments.output = tmp_path / "learned"
    async with httpx.AsyncClient(
        transport=transport(readout_path=str(artifact))
    ) as client:
        await run(arguments, client)
    comparison = compare_saved(baseline, arguments.output)
    phase = comparison["per_concurrency"]["c8"]
    assert phase["quality"]["comparison"] == "separate_trained_head"
    assert phase["quality"]["baseline"]["overall"]["n"] == 2
    assert phase["quality"]["candidate"]["by_type"]["choice"]["acc"] == 1.0
    assert phase["quality"]["baseline"]["by_suite"]
    assert comparison["quality_winner"] is None
    assert comparison["single_pair_winner"] is None


@pytest.mark.asyncio
async def test_minimum_duration_is_recorded_and_refused_in_saved_comparison(tmp_path):
    arguments = args(
        tmp_path,
        concurrency=[8],
        repeats=4,
        warmup=0,
        min_measure_seconds=1.0,
    )
    async with httpx.AsyncClient(transport=transport()) as client:
        report = await run(arguments, client)
    phase = report["concurrency"]["c8"]
    assert report["status"] == "failed"
    assert 0 < phase["elapsed_s"] < 1.0
    assert phase["minimum_measure_seconds"] == 1.0
    assert phase["duration_requirement_met"] is False
    assert phase["measurement_failures"] == [
        {
            "error_type": "MeasurementTooShort",
            "observed_seconds": phase["elapsed_s"],
            "required_seconds": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="duration below minimum"):
        compare_saved(arguments.output, arguments.output)

    manifest_path = arguments.output / "manifest.json"
    report_path = arguments.output / "report.json"
    manifest = json.loads(manifest_path.read_text())
    saved_report = json.loads(report_path.read_text())
    manifest["status"] = saved_report["status"] = "complete"
    saved_report["concurrency"]["c8"]["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest))
    report_path.write_text(json.dumps(saved_report))
    with pytest.raises(ValueError, match="duration gate is invalid"):
        compare_saved(arguments.output, arguments.output)


@pytest.mark.asyncio
async def test_saved_pair_requires_same_minimum_duration(tmp_path):
    arguments = args(tmp_path, concurrency=[8], repeats=4, warmup=0)
    async with httpx.AsyncClient(transport=transport()) as client:
        await run(arguments, client)
        baseline = arguments.output
        arguments.output = tmp_path / "candidate"
        arguments.min_measure_seconds = 0.000001
        await run(arguments, client)
    with pytest.raises(ValueError, match="workload, concurrency"):
        compare_saved(baseline, arguments.output)


@pytest.mark.asyncio
async def test_jobs_peak_and_service_capacity_must_reach_c(tmp_path):
    arguments = args(tmp_path, concurrency=[8], repeats=3, warmup=0)
    with pytest.raises(ValueError, match="measured jobs"):
        await run(arguments)
    arguments.repeats = 4
    async with httpx.AsyncClient(transport=transport(delay=0)) as client:
        report = await run(arguments, client)
    phase = report["concurrency"]["c8"]
    assert phase["status"] == "failed"
    assert phase["client_http_inflight_peak"] < 8
    assert (
        phase["measurement_failures"][0]["error_type"] == "ClientConcurrencyUnderfill"
    )

    arguments.output = tmp_path / "capacity"
    async with httpx.AsyncClient(transport=transport(service_capacity=4)) as client:
        report = await run(arguments, client)
    assert report["concurrency"]["c8"]["errors"] == 8
    assert report["status"] == "failed"


def test_readout_metadata_is_bound_to_settings(tmp_path, monkeypatch):
    arguments = args(tmp_path)
    artifact = tmp_path / "gpu-readout.npz"
    artifact.write_bytes(b"fake-artifact-for-validation-test")
    value = json.loads(arguments.settings_manifest.read_text())
    value["head_mode"] = "lm_rows"
    value["head_artifact"] = {
        "path": str(artifact),
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    arguments.settings_manifest.write_text(json.dumps(value))
    metadata = {
        "mode": "learned",
        "labels": value["labels"],
        "token_ids": value["token_ids"],
        "engine_model_path": value["model_path"],
        "revision": value["model_revision"],
    }
    monkeypatch.setattr(inprocess, "_load_metadata", lambda path: metadata)
    with pytest.raises(ValueError, match="metadata mode"):
        read_settings(arguments.settings_manifest)
    metadata["mode"] = "lm_rows"
    metadata["token_ids"] = [13, 14]
    with pytest.raises(ValueError, match="metadata token_ids"):
        read_settings(arguments.settings_manifest)
    metadata["token_ids"] = value["token_ids"]
    assert read_settings(arguments.settings_manifest)[0]["head_mode"] == "lm_rows"


@pytest.mark.asyncio
async def test_saved_runtime_rows_and_displayed_answers_are_verified(tmp_path):
    arguments = args(tmp_path, concurrency=[8], repeats=4, warmup=0)
    async with httpx.AsyncClient(transport=transport()) as client:
        await run(arguments, client)
        baseline = arguments.output
        arguments.output = tmp_path / "candidate"
        await run(arguments, client)
    candidate = arguments.output

    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["engine"]["server_info"]["max_total_tokens"] = 4096
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="runtime setting max_total_tokens"):
        compare_saved(baseline, candidate)
    manifest["engine"]["server_info"]["max_total_tokens"] = 8192
    manifest_path.write_text(json.dumps(manifest))

    report_path = candidate / "report.json"
    saved_report = json.loads(report_path.read_text())
    saved_report["concurrency"]["c8"]["quality_overall"]["acc"] = 0.0
    report_path.write_text(json.dumps(saved_report))
    with pytest.raises(ValueError, match="aggregate metrics differ"):
        compare_saved(baseline, candidate)
    saved_report["concurrency"]["c8"]["quality_overall"]["acc"] = 1.0
    report_path.write_text(json.dumps(saved_report))

    raw_path = candidate / "c8" / "raw_outcomes.jsonl"
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    original_rows = json.loads(json.dumps(raw))
    raw[0]["rows"][0]["p"] = [0.2, 0.8]
    raw_path.write_text("\n".join(json.dumps(row) for row in raw) + "\n")
    with pytest.raises(ValueError, match="raw outcome contradicts"):
        compare_saved(baseline, candidate)

    raw = original_rows
    body = json.loads(raw[0]["raw_response_text"])
    body["answers"]["route"]["choice"] = "support"
    raw[0]["raw_response_text"] = json.dumps(body)
    raw_path.write_text("\n".join(json.dumps(row) for row in raw) + "\n")
    result = compare_saved(baseline, candidate)
    assert result["per_concurrency"]["c8"]["quality_parity"]["parity_passed"]
    assert (
        result["per_concurrency"]["c8"]["displayed_answer_parity"]["status"]
        == "mismatch"
    )
    assert result["single_pair_winner"] is None


def test_cli_selects_http_client_and_rejects_unknown(tmp_path, monkeypatch):
    selected = []
    mappings = []
    pools = []

    async def fake_run(arguments):
        selected.append(arguments.http_client)
        mappings.append(arguments.repeats_by_concurrency)
        pools.append(arguments.http_max_connections)
        return {"status": "complete"}

    monkeypatch.setattr(bench, "run", fake_run)
    options = [
        "run",
        "--base-url",
        "http://fixture",
        "--model",
        "decision-model",
        "--settings-manifest",
        str(tmp_path / "settings.json"),
        "--data",
        str(tmp_path / "data.jsonl"),
        "--output",
        str(tmp_path / "run"),
    ]
    assert bench.main(options) == 0
    assert bench.main([*options, "--http-client", "aiohttp"]) == 0
    assert selected == ["httpx", "aiohttp"]
    assert (
        bench.main(
            [*options, "--concurrency", "8", "--repeats-by-concurrency", '{"8":4}']
        )
        == 0
    )
    assert mappings == [None, None, '{"8":4}']
    assert pools == [None, None, None]
    assert bench.main([*options, "--concurrency", "8", "--http-max-connections", "128"]) == 0
    assert pools[-1] == 128
    with pytest.raises(SystemExit) as error:
        bench.main([*options, "--http-client", "unknown"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        bench.main([*options, "--http-max-connections", "1.5"])
    assert error.value.code == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("concurrency", "pool"), [([8], None), ([8], 128), ([8, 16], 128)]
)
async def test_owned_clients_record_transport_version_and_helper_sha(
    tmp_path, monkeypatch, concurrency, pool
):
    from jev_adapter import http_client

    arguments = args(
        tmp_path, concurrency=concurrency, repeats=8, warmup=0, http_max_connections=pool
    )
    expected_capacity = max(concurrency) if pool is None else pool
    default_client, descriptor = bench.make_http_client(arguments)
    try:
        assert descriptor["kind"] == "httpx"
        assert descriptor["transport"] == "httpx.AsyncHTTPTransport"
        assert descriptor["library_version"] == httpx.__version__
        assert descriptor["helper_source_sha256"] is None
        assert descriptor["max_connections"] == expected_capacity
        assert default_client._transport._pool._max_connections == expected_capacity
        assert default_client._transport._pool._max_keepalive_connections == expected_capacity
    finally:
        await default_client.aclose()

    created = []

    class FakeAiohttpClient:
        def __init__(self, *, concurrency, timeout):
            self.capacity, self.timeout = concurrency, timeout
            self.client = httpx.AsyncClient(transport=transport())
            self.closed = False
            created.append(self)

        async def get(self, *args, **kwargs):
            return await self.client.get(*args, **kwargs)

        async def post(self, *args, **kwargs):
            return await self.client.post(*args, **kwargs)

        async def aclose(self):
            self.closed = True
            await self.client.aclose()

    monkeypatch.setattr(http_client, "AiohttpCompatClient", FakeAiohttpClient)
    monkeypatch.setattr(bench, "package_version", lambda name: "3.14.3-mocked")
    arguments.http_client = "aiohttp"
    assert (await run(arguments))["status"] == "complete"
    assert created[0].capacity == expected_capacity
    assert created[0].timeout == arguments.timeout
    assert created[0].closed
    manifest = json.loads((arguments.output / "manifest.json").read_text())
    descriptor = manifest["http_client"]
    assert manifest["http_pool_max_connections"] == expected_capacity
    assert descriptor["max_connections"] == expected_capacity
    assert descriptor["kind"] == "aiohttp"
    assert descriptor["transport"] == "aiohttp.TCPConnector"
    assert descriptor["library_version"] == "3.14.3-mocked"
    assert descriptor["trust_env"] is False
    assert descriptor["timeout_scope"] == "total"
    assert descriptor["helper_source_sha256"] == bench.file_sha256(
        bench.Path(http_client.__file__)
    )
    assert (
        manifest["client_code_sha256"]["jev_adapter/http_client.py"]
        == descriptor["helper_source_sha256"]
    )
    bench.load_saved(arguments.output)


@pytest.mark.parametrize("pool", [0, -1, 7, 129, True, 8.0, "128"])
def test_http_pool_rejects_invalid_and_below_c_values(tmp_path, pool):
    arguments = args(tmp_path, concurrency=[8], http_max_connections=pool)
    with pytest.raises(ValueError, match="http_max_connections"):
        bench.validate_counts(arguments)


@pytest.mark.asyncio
async def test_explicit_http_pool_rejects_external_client_before_output(tmp_path):
    arguments = args(tmp_path, http_max_connections=128)
    async with httpx.AsyncClient(transport=transport()) as client:
        with pytest.raises(ValueError, match="external client"):
            await run(arguments, client)
    assert not arguments.output.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [None, [], ["8"], [[8]], [True], [8, 8]])
async def test_saved_concurrency_is_checked_before_pool(tmp_path, concurrency):
    arguments = args(tmp_path, concurrency=[8], repeats=4, warmup=0)
    async with httpx.AsyncClient(transport=transport()) as client:
        await run(arguments, client)
    path = arguments.output / "manifest.json"
    manifest = json.loads(path.read_text())
    if concurrency is None:
        del manifest["concurrency"]
    else:
        manifest["concurrency"] = concurrency
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="saved concurrency is invalid"):
        bench.load_saved(arguments.output)


@pytest.mark.asyncio
@pytest.mark.parametrize("pool", [None, 0, 7, 129, True, 8.0])
async def test_saved_http_pool_rejects_invalid_present_values(tmp_path, pool):
    arguments = args(tmp_path, concurrency=[8], repeats=4, warmup=0)
    async with httpx.AsyncClient(transport=transport()) as client:
        await run(arguments, client)
    path = arguments.output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["http_pool_max_connections"] = pool
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="saved HTTP pool capacity"):
        bench.load_saved(arguments.output)


@pytest.mark.asyncio
@pytest.mark.parametrize("descriptor_pool", [None, True, 7, 16, 129])
async def test_saved_http_pool_rejects_owned_descriptor_mismatch(
    tmp_path, descriptor_pool
):
    arguments = args(tmp_path, concurrency=[8], repeats=4, warmup=0)
    async with httpx.AsyncClient(transport=transport()) as client:
        await run(arguments, client)
    path = arguments.output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["http_client"] = {"kind": "aiohttp", "max_connections": descriptor_pool}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="HTTP pool"):
        bench.load_saved(arguments.output)


@pytest.mark.asyncio
async def test_saved_comparison_rejects_different_pools_and_accepts_legacy_missing(
    tmp_path,
):
    arguments = args(tmp_path, concurrency=[8], repeats=4, warmup=0)
    async with httpx.AsyncClient(transport=transport()) as client:
        await run(arguments, client)
        baseline = arguments.output
        arguments.output = tmp_path / "candidate"
        await run(arguments, client)
    path = arguments.output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["http_pool_max_connections"] = 16
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="workload, concurrency"):
        compare_saved(baseline, arguments.output)
    for directory in (baseline, arguments.output):
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["http_pool_max_connections"]
        path.write_text(json.dumps(manifest))
    assert compare_saved(baseline, arguments.output)["status"] == "comparable"


@pytest.mark.asyncio
async def test_saved_comparison_rejects_different_http_clients(tmp_path):
    arguments = args(tmp_path, concurrency=[8], repeats=4, warmup=0)
    async with httpx.AsyncClient(transport=transport()) as client:
        await run(arguments, client)
        baseline = arguments.output
        arguments.output = tmp_path / "candidate"
        await run(arguments, client)
    manifest_path = arguments.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["http_client"]["class"] = "diagnostic.AiohttpCompatClient"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="workload, concurrency"):
        compare_saved(baseline, arguments.output)


def test_repeat_mapping_requires_exact_unique_integer_counts(tmp_path):
    arguments = args(tmp_path)
    invalid = (
        '{"8":4}',
        '{"8":4,"16":8,"32":16}',
        '{"8":4,"8":5,"16":8}',
        '{"08":4,"16":8}',
        '{"8":true,"16":8}',
        '{"8":4.0,"16":8}',
        '{"8":0,"16":8}',
        '{"8":-1,"16":8}',
        '{"8":1000001,"16":8}',
        "[]",
        "null",
    )
    for value in invalid:
        arguments.repeats_by_concurrency = value
        with pytest.raises(ValueError):
            bench.validate_counts(arguments)
    arguments.repeats_by_concurrency = '{"16":9,"8":4}'
    bench.validate_counts(arguments)
    assert bench.repeat_counts([8, 16], 999, arguments.repeats_by_concurrency) == {
        "8": 4,
        "16": 9,
    }


def test_repeat_mapping_total_guard_precedes_job_allocation(tmp_path, monkeypatch):
    arguments = args(tmp_path, repeats_by_concurrency='{"8":300000,"16":300000}')

    def no_allocation(*_args):
        raise AssertionError("oversized workload must fail before jobs_for")

    monkeypatch.setattr(bench, "jobs_for", no_allocation)
    with pytest.raises(ValueError, match="1000000-request bound"):
        asyncio.run(run(arguments))
    arguments.repeats_by_concurrency = '{"8":4,"16":1}'
    with pytest.raises(ValueError, match="measured jobs"):
        asyncio.run(run(arguments))


@pytest.mark.asyncio
async def test_repeat_mapping_controls_each_phase_and_pair_compatibility(tmp_path):
    arguments = args(tmp_path, repeats=999999, repeats_by_concurrency='{"8":4,"16":9}')
    async with httpx.AsyncClient(transport=transport()) as client:
        report = await run(arguments, client)
        baseline = arguments.output
        arguments.output = tmp_path / "candidate"
        arguments.repeats = 1  # Unused fallback does not change the actual workload.
        await run(arguments, client)
        assert compare_saved(baseline, arguments.output)["status"] == "comparable"
        arguments.output = tmp_path / "different-map"
        arguments.repeats_by_concurrency = '{"8":5,"16":9}'
        await run(arguments, client)
    manifest = json.loads((baseline / "manifest.json").read_text())
    assert manifest["job_order_sha256"] is None
    for c, repeats in ((8, 4), (16, 9)):
        assert report["concurrency"][f"c{c}"]["attempted_requests"] == 2 * repeats
        assert report["concurrency"][f"c{c}"]["quality_overall"]["n"] == 2
        expected = canonical_sha256(jobs_for([1, 2], repeats, 42))
        assert manifest["job_order_sha256_by_concurrency"][str(c)] == expected
        assert report["concurrency"][f"c{c}"]["order_sha256"] == expected
    with pytest.raises(ValueError, match="repeats-by-concurrency differs"):
        compare_saved(baseline, arguments.output)


@pytest.mark.asyncio
async def test_legacy_global_repeat_manifests_remain_readable(tmp_path):
    arguments = args(tmp_path)
    async with httpx.AsyncClient(transport=transport()) as client:
        await run(arguments, client)
        baseline = arguments.output
        arguments.output = tmp_path / "legacy"
        await run(arguments, client)
    manifest_path = arguments.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["repeats_by_concurrency"]
    del manifest["job_order_sha256_by_concurrency"]
    manifest_path.write_text(json.dumps(manifest))
    assert compare_saved(baseline, arguments.output)["status"] == "comparable"
