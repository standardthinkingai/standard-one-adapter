"""Native online FP8 benchmark policy and legacy BF16 CPU regressions."""

import argparse
import asyncio
import copy
import json

import httpx
import numpy as np
import pytest
from benchmarks import prefill_engine_bench as bench
from jev_adapter import inprocess
from jev_adapter.benchmarks.data import normalize_record

MODEL = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"
FP8_PROFILE = "fp8_w8a8_backbone_bf16_head"


def settings(**changes):
    value = {
        "schema_version": 1,
        "hardware": "H200",
        "model_path": MODEL,
        "model_revision": REVISION,
        "precision": "bfloat16",
        "prompt_contract": "served-v1",
        "label_contract": "native-AB-v1",
        "head_mode": "logprobs",
        "head_artifact": None,
        "labels": ["A", "B"],
        "token_ids": [165, 166],
    }
    value.update(changes)
    return value


def write_settings(tmp_path, **changes):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings(**changes)))
    return path


def readout_metadata(**changes):
    value = {
        "format": inprocess.FORMAT,
        "mode": "lm_rows",
        "model_id": MODEL,
        "engine_model_path": MODEL,
        "revision": REVISION,
        "architecture": inprocess.ARCHITECTURE,
        "hidden_size": 2,
        "label_scheme": "upper",
        "tokenizer_id": MODEL,
        "tokenizer_revision": REVISION,
        "labels": ["A", "B"],
        "token_ids": [165, 166],
        "temperature": 1.0,
    }
    value.update(changes)
    return value


def test_old_bf16_defaults_preserve_input_and_hash(tmp_path):
    path = write_settings(tmp_path)
    raw = path.read_bytes()
    value, digest = bench.read_settings(path)
    assert value["quantization"] is None
    assert value["precision_profile"] == "unquantized_bfloat16"
    assert path.read_bytes() == raw
    assert digest == bench.sha256(raw)
    assert bench.precision_settings(settings(precision="float16")) == (
        None,
        "unquantized_float16",
    )


@pytest.mark.parametrize("mode", ["logprobs", "lm_rows"])
def test_fp8_profile_keeps_bf16_activation_and_artifact_rows(tmp_path, mode):
    artifact = None
    if mode == "lm_rows":
        path = tmp_path / "rows.npz"
        np.savez(
            path,
            metadata=np.bytes_(json.dumps(readout_metadata())),
            weights=np.zeros((2, 2), dtype=np.float32),
            bias=np.zeros(2, dtype=np.float32),
        )
        artifact = {"path": str(path), "sha256": bench.file_sha256(path)}
    path = write_settings(
        tmp_path, quantization="fp8", head_mode=mode, head_artifact=artifact
    )
    value, _ = bench.read_settings(path)
    assert value["precision"] == "bfloat16"
    assert value["precision_profile"] == FP8_PROFILE
    assert value["head_artifact"] == artifact


@pytest.mark.parametrize(
    "changes",
    [
        {"quantization": "mxfp8"},
        {"quantization": "fp8_serialized"},
        {"quantization": False},
        {"quantization": True},
        {"quantization": {}},
        {"quantization": "fp8", "head_mode": "learned"},
        {"quantization": "fp8", "precision": "float16"},
        {"quantization": "fp8", "precision_profile": "unquantized_bfloat16"},
        {"precision_profile": FP8_PROFILE},
        {"precision_profile": None},
    ],
)
def test_settings_reject_unsupported_or_mislabeled_precision(tmp_path, changes):
    with pytest.raises(ValueError):
        bench.read_settings(write_settings(tmp_path, **changes))


def mock_transport(quantization=None, prefill_max_requests=None):
    async def respond(request):
        if request.url.path == "/model_info":
            return httpx.Response(200, json={"model_path": MODEL})
        if request.url.path == "/server_info":
            return httpx.Response(
                200,
                json={
                    "model_path": MODEL,
                    "revision": REVISION,
                    "dtype": "bfloat16",
                    "quantization": quantization,
                    "disable_radix_cache": True,
                    "mm_preprocess_cache_size_mb": 0,
                    "enable_prefix_mm_cache": False,
                    "enable_mm_global_cache": False,
                    "speculative_algorithm": None,
                    "max_running_requests": 128,
                    "prefill_max_requests": prefill_max_requests,
                    "decision_readout_path": None,
                    "tp_size": 1,
                    "pp_size": 1,
                    "dp_size": 1,
                    "cpu_offload_gb": 0,
                },
            )
        await asyncio.sleep(0.002)
        return httpx.Response(
            200,
            json={
                "model": "decision-model",
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "billing",
                        "confidence": 0.8,
                        "probabilities": {"billing": 0.8, "support": 0.2},
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 0},
                "metadata": {
                    "evaluations": 1,
                    "cached_tokens": 0,
                    "adapter_elapsed_ms": 1.0,
                    "service_max_concurrency": 128,
                },
            },
        )

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
@pytest.mark.parametrize("quantization", [None, "fp8"])
async def test_snapshot_requires_exact_quantization_match(quantization):
    value = settings(quantization=quantization)
    async with httpx.AsyncClient(transport=mock_transport(quantization)) as client:
        result = await bench.engine_snapshot(client, "http://engine", {}, value, 128)
    assert result["server_info"]["quantization"] == quantization
    other = "fp8" if quantization is None else None
    async with httpx.AsyncClient(transport=mock_transport(other)) as client:
        with pytest.raises(ValueError, match="engine quantization"):
            await bench.engine_snapshot(client, "http://engine", {}, value, 128)


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [None, 1, 8, 128])
async def test_snapshot_records_prefill_request_cap(cap):
    async with httpx.AsyncClient(
        transport=mock_transport(prefill_max_requests=cap)
    ) as client:
        result = await bench.engine_snapshot(
            client, "http://engine", {}, settings(), 128
        )
    assert result["server_info"]["prefill_max_requests"] == cap


def graph_manifest(value):
    return {
        "data": [],
        "max_options_in_workload": 2,
        "settings": value,
        "engine": {"model_info": {"model_path": MODEL}, "server_info": {}},
    }


def test_comparison_rejects_mixed_profiles_and_preserves_legacy_defaults():
    left = graph_manifest(settings())
    right = copy.deepcopy(left)
    right["settings"].update(
        quantization=None, precision_profile="unquantized_bfloat16"
    )
    bench._same_workload(left, right)
    right["settings"].update(quantization="fp8", precision_profile=FP8_PROFILE)
    with pytest.raises(ValueError, match="separate comparison"):
        bench._same_workload(left, right)
    with pytest.raises(ValueError, match="separate comparison"):
        bench._same_workload(right, left)


def test_comparison_rejects_different_prefill_caps_preserving_legacy_none():
    left = graph_manifest(settings())
    right = copy.deepcopy(left)
    right["engine"]["server_info"]["prefill_max_requests"] = None
    bench._same_workload(left, right)
    left["engine"]["server_info"]["prefill_max_requests"] = 8
    right["engine"]["server_info"]["prefill_max_requests"] = 16
    with pytest.raises(ValueError, match="prefill_max_requests differs"):
        bench._same_workload(left, right)
    right["engine"]["server_info"]["prefill_max_requests"] = 8
    bench._same_workload(left, right)


@pytest.mark.asyncio
@pytest.mark.parametrize("quantization", [None, "fp8"])
async def test_saved_manifest_runtime_profile_integrity(tmp_path, quantization):
    data = tmp_path / "data.jsonl"
    items = [
        normalize_record(
            {
                "state": name,
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Choose a team.",
                        "criteria": {"billing": None, "support": None},
                        "label": "billing",
                    }
                },
                "_meta": {"id": name, "source": "fixture", "variant": "clean"},
            },
            "fixture",
            "development",
        )
        for name in ("one", "two")
    ]
    data.write_text("".join(json.dumps(item) + "\n" for item in items))
    args = argparse.Namespace(
        base_url="http://engine",
        engine_url=None,
        model="decision-model",
        data=[data],
        settings_manifest=write_settings(tmp_path, quantization=quantization),
        token_audit=None,
        output=tmp_path / "run",
        concurrency=[8],
        warmup=0,
        repeats=4,
        min_measure_seconds=0.0,
        limit=2,
        seed=42,
        timeout=10.0,
        key_env="SGLANG_API_KEY",
        allow_test=False,
    )
    async with httpx.AsyncClient(transport=mock_transport(quantization)) as client:
        assert (await bench.run(args, client))["status"] == "complete"
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["settings"]["precision_profile"] == (
        FP8_PROFILE if quantization else "unquantized_bfloat16"
    )
    bench.load_saved(args.output)
    if quantization is None:
        legacy = copy.deepcopy(manifest)
        legacy["settings"].pop("quantization")
        legacy["settings"].pop("precision_profile")
        manifest_path.write_text(json.dumps(legacy))
        bench.load_saved(args.output)
    manifest["settings"]["precision_profile"] = "unquantized_float16"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="precision_profile"):
        bench.load_saved(args.output)
    manifest["settings"]["precision_profile"] = (
        FP8_PROFILE if quantization else "unquantized_bfloat16"
    )
    manifest["engine"]["server_info"]["quantization"] = (
        "fp8" if quantization is None else None
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="saved engine quantization/dtype"):
        bench.load_saved(args.output)
