import argparse
import asyncio
import copy
import json

import httpx
import pytest
from benchmarks import prefill_precision_quality as quality
from benchmarks.prefill_engine_bench import load_saved, run
from jev_adapter.benchmarks.data import normalize_record

MODEL = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"


async def saved_run(tmp_path, name, quantization):
    data = tmp_path / "data.jsonl"
    if not data.exists():
        records = []
        for index, (kind, source, variant) in enumerate(
            [
                ("choice", "fixture", "clean"),
                ("noul", "fixture", "clean"),
                ("choice", "unknowable", "clean"),
                ("choice", "fixture", "permuted"),
            ]
        ):
            question = {"type": kind, "instructions": "Choose.", "label": True}
            if kind == "choice":
                question.update(
                    criteria={"billing": None, "support": None}, label="billing"
                )
            records.append(
                normalize_record(
                    {
                        "state": str(index),
                        "questions": {"route": question},
                        "_meta": {
                            "id": str(index),
                            "source": source,
                            "variant": variant,
                        },
                    },
                    "fixture",
                    "development",
                )
            )
        data.write_text("".join(json.dumps(row) + "\n" for row in records))
    config = tmp_path / f"{name}-settings.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hardware": "H200",
                "model_path": MODEL,
                "model_revision": REVISION,
                "precision": "bfloat16",
                "head_mode": "logprobs",
                "head_artifact": None,
                "prompt_contract": "served-v1",
                "label_contract": "AB-v1",
                "labels": ["A", "B"],
                "token_ids": [165, 166],
                "quantization": quantization,
            }
        )
    )
    args = argparse.Namespace(
        base_url="http://engine",
        engine_url=None,
        model="decision-model",
        data=[data],
        settings_manifest=config,
        token_audit=None,
        output=tmp_path / name,
        concurrency=[8, 16],
        warmup=0,
        repeats=4,
        min_measure_seconds=0.0,
        limit=4,
        seed=42,
        timeout=10.0,
        key_env="SGLANG_API_KEY",
        allow_test=False,
    )

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
                    "decision_readout_path": None,
                    "tp_size": 1,
                    "pp_size": 1,
                    "dp_size": 1,
                    "cpu_offload_gb": 0,
                },
            )
        await asyncio.sleep(0.002)
        payload = json.loads(request.content)
        if payload["questions"]["route"]["type"] == "noul":
            answer = {"type": "noul", "noul": 0.6 if quantization else 0.8}
        else:
            p = 0.4 if quantization else 0.8
            answer = {
                "type": "choice",
                "choice": "support" if quantization else "billing",
                "confidence": max(p, 1 - p),
                "probabilities": {"billing": p, "support": 1 - p},
            }
        return httpx.Response(
            200,
            json={
                "model": "decision-model",
                "answers": {"route": answer},
                "usage": {"input_tokens": 100, "output_tokens": 0},
                "metadata": {
                    "evaluations": 1,
                    "cached_tokens": 0,
                    "adapter_elapsed_ms": 1.0,
                    "service_max_concurrency": 128,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        assert (await run(args, client))["status"] == "complete"
    return args.output


@pytest.mark.asyncio
async def test_quality_first_pass_only_and_every_c_keeps_its_denominator(tmp_path):
    bf16 = await saved_run(tmp_path, "bf16", None)
    fp8 = await saved_run(tmp_path, "fp8", "fp8")
    report = quality.analyze_pairs([(bf16, fp8)])
    assert report["analysis_scope"] == "cross_precision_quality_only"
    assert report["performance_winner"] is None
    assert report["quality_winner"] is None
    assert report["causal_quantization_claim"] is False
    pair = report["pairs"][0]
    assert pair["head_mode"] == "logprobs"
    for phase in pair["per_concurrency"].values():
        assert phase["selected_requests_per_first_pass"] == 4
        assert phase["unique_questions_per_first_pass"] == 4
        assert phase["eligible_clean_knowable_non_synthetic_questions"] == 2
        assert phase["reference_clean1194_coverage"] is False
        assert phase["measured_http_responses_not_independent_quality_samples"] == {
            "bf16": 16,
            "fp8": 16,
        }
        assert (
            phase["saved_quality_verified_against_raw"]["bf16"]["quality_overall"]["n"]
            == 2
        )
        assert phase["clean_question_changes"]["argmax_changes"] == 1
        assert phase["all_question_changes"]["argmax_changes"] == 3
        assert phase["clean_question_changes"]["probability_abs_delta"][
            "max"
        ] == pytest.approx(0.4)
        assert phase["fp8_minus_bf16"]["acc"] == -0.5
        assert phase["fp8_minus_bf16"]["brier"] > 0
        assert phase["fp8_minus_bf16"]["nll"] > 0
        assert phase["clean_by_type"]["choice"]["bf16"]["n"] == 1
        assert phase["clean_by_type"]["noul"]["bf16"]["n"] == 1
        assert "fixture" in phase["clean_by_benchmark_task"]
    assert pair["runtime_differences"] == {"quantization": {"bf16": None, "fp8": "fp8"}}
    output = tmp_path / "cross-quality.json"
    assert quality.main(["--pair", str(bf16), str(fp8), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["pairs"][0] == pair
    with pytest.raises(FileExistsError):
        quality.main(["--pair", str(bf16), str(fp8), "--output", str(output)])


@pytest.mark.asyncio
async def test_reused_baseline_is_reported_not_pooled_and_duplicate_pair_rejected(
    tmp_path,
):
    bf16 = await saved_run(tmp_path, "bf16", None)
    fp8 = await saved_run(tmp_path, "fp8", "fp8")
    fp8_second = await saved_run(tmp_path, "fp8-second", "fp8")
    report = quality.analyze_pairs([(bf16, fp8), (bf16, fp8_second)])
    assert report["reused_source_runs"] == {str(bf16.resolve()): 2}
    assert [pair["pair_index"] for pair in report["pairs"]] == [1, 2]
    assert all(
        pair["per_concurrency"]["c8"]["eligible_clean_knowable_non_synthetic_questions"]
        == 2
        for pair in report["pairs"]
    )
    with pytest.raises(ValueError, match="duplicate saved quality pair"):
        quality.analyze_pairs([(bf16, fp8), (bf16, fp8)])
    with pytest.raises(ValueError, match="left run"):
        quality.analyze_pair(fp8, bf16)


@pytest.mark.asyncio
async def test_every_raw_response_validated_before_quality(tmp_path):
    bf16 = await saved_run(tmp_path, "bf16", None)
    fp8 = await saved_run(tmp_path, "fp8", "fp8")
    path = fp8 / "c8" / "raw_outcomes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    repeated = next(row for row in rows if row["repeat"] > 0)
    repeated["rows"][0]["p"][0] += 0.01  # A repeat>0 is still revalidated.
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="raw outcome contradicts saved metrics"):
        quality.analyze_pair(bf16, fp8)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,key,value",
    [
        ("settings", "model_revision", "a" * 40),
        ("settings", "labels", ["B", "A"]),
        ("settings", "head_mode", "lm_rows"),
        ("manifest", "workload_sha256", "b" * 64),
        ("manifest", "request_options", {"temperature": 0.5, "permutations": 1}),
    ],
)
async def test_contract_mismatches_rejected(tmp_path, scope, key, value):
    bf16 = await saved_run(tmp_path, "bf16", None)
    fp8 = await saved_run(tmp_path, "fp8", "fp8")
    old, _, _ = load_saved(bf16)
    new, _, _ = load_saved(fp8)
    target = new["settings"] if scope == "settings" else new
    target[key] = value
    with pytest.raises(ValueError):
        quality._contracts(old, new)


def test_duplicate_question_id_is_rejected_and_not_double_counted():
    row = {"suite": "test", "split": "development", "id": "one", "question": "route"}
    with pytest.raises(ValueError, match="duplicate first-pass question"):
        quality._unique_questions([row, copy.deepcopy(row)])


def test_synthetic_suite_never_counted_as_clean1194():
    assert not quality._eligible(
        {"variant": "clean", "source": "fixture", "suite": "synthetic-128"}
    )


@pytest.mark.asyncio
async def test_same_head_rows_artifact_required(tmp_path):
    bf16 = await saved_run(tmp_path, "bf16", None)
    fp8 = await saved_run(tmp_path, "fp8", "fp8")
    old, _, _ = load_saved(bf16)
    new, _, _ = load_saved(fp8)
    for manifest in (old, new):
        manifest["settings"].update(
            head_mode="lm_rows",
            head_artifact={"path": "/original.npz", "sha256": "a" * 64},
        )
    quality._contracts(old, new)
    new["settings"]["head_artifact"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="readout artifact differs"):
        quality._contracts(old, new)
    new["settings"]["head_artifact"]["sha256"] = "a" * 64
    for manifest in (old, new):
        manifest["settings"]["model_revision"] = "main"
    with pytest.raises(ValueError, match="pinned model revision"):
        quality._contracts(old, new)
