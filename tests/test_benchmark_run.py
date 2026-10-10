import argparse
import hashlib
import json

import httpx
import pytest

from jev_adapter.benchmarks.compare import collect
from jev_adapter.benchmarks.data import normalize_record
from jev_adapter.benchmarks.metrics import metrics, prediction_rows, summarize
from jev_adapter.benchmarks.run import evaluate, load_data, validate_launch


def item(record_id="one", variant="clean", source="policy"):
    return normalize_record(
        {
            "state": "The transaction is refundable.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a team.",
                    "criteria": {"billing": None, "support": "Technical support"},
                    "label": "billing",
                    "src": "route",
                },
                "refund": {
                    "type": "noul",
                    "instructions": "Refundable?",
                    "label": True,
                    "src": "refund",
                },
                "priority": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["low", "mid", "high"],
                    "label": 2,
                    "src": "priority",
                },
            },
            "_meta": {
                "id": record_id,
                "source": source,
                "variant": variant,
                "group_id": record_id,
            },
        },
        "fixture",
        "development",
    )


def response():
    return {
        "model": "decision-model",
        "answers": {
            "route": {
                "type": "choice",
                "probabilities": {"billing": 0.8, "support": 0.2},
            },
            "refund": {"type": "noul", "noul": 0.9},
            "priority": {
                "type": "score",
                "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
            },
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
        "metadata": {"adapter_elapsed_ms": 2.0, "evaluations": 3},
    }


def test_metrics_boolean_order_score_and_calibration():
    rows = prediction_rows(item(), response())
    assert rows[1]["keys"] == ["false", "true"]
    assert rows[1]["p"] == pytest.approx([0.1, 0.9])
    result = metrics(rows)
    assert result["n"] == 3
    assert result["acc"] == 1
    assert result["ece"] == pytest.approx(0.2)
    assert result["brier"] == pytest.approx((0.08 + 0.02 + 0.14) / 3)
    assert result["score_mae"] == pytest.approx(0.4)
    assert result["ranked_probability_score"] == pytest.approx(0.05)


def test_jevbench_probability_reference_and_native_tie_break():
    example = item()
    example["metadata"]["gold_policy"] = {"argmax_tie_break": "lexicographic_label"}
    example["expected"]["route"].update(
        labels=["support", "billing"],
        label=1,
        target=[0, 1],
        reference_probs=[0.5, 0.5],
    )
    body = response()
    body["answers"]["route"]["probabilities"] = {"support": 0.5, "billing": 0.5}
    rows = prediction_rows(example, body)
    assert metrics(rows)["acc"] == 1
    report = summarize(rows)
    assert report["reference_distribution"]["n"] == 1
    assert report["reference_distribution"]["squared_l2"] == 0
    assert report["reference_distribution"]["total_variation"] == 0
    assert report["reference_distribution"]["kl_reference_to_model"] == 0


def test_unknown_excluded_and_permutation_aligned():
    original = item()
    permuted = item("two", "permuted")
    permuted["metadata"]["group_id"] = "one"
    permuted["expected"]["route"].update(
        labels=["support", "billing"], label=1, target=[0, 1]
    )
    unknown = item("three", source="unknowable")
    rows = [
        row
        for i in [original, permuted, unknown]
        for row in prediction_rows(i, response())
    ]
    report = summarize(rows)
    assert report["clean"]["n"] == 3
    assert report["unknowable"]["n"] == 3
    assert report["permutation"]["n"] == 1
    assert report["permutation"]["flip_rate"] == 0
    assert "unknowable" not in report["sources"]


@pytest.mark.parametrize("change", ["missing", "nan", "output", "type"])
def test_invalid_predictions_rejected(change):
    body = response()
    if change == "missing":
        del body["answers"]["route"]["probabilities"]["support"]
    elif change == "nan":
        body["answers"]["refund"]["noul"] = float("nan")
    elif change == "type":
        body["answers"]["refund"]["type"] = "choice"
    else:
        body["answers"]["route"]["probabilities"] = {"billing": 0.9, "support": 0.9}
    with pytest.raises(ValueError):
        prediction_rows(item(), body)


def arguments(tmp_path, records=None):
    data = tmp_path / "data.jsonl"
    records = records or [item(), item("two")]
    data.write_text("".join(json.dumps(i) + "\n" for i in records))
    return argparse.Namespace(
        data=[data],
        model="decision-model",
        allow_test=False,
        limit=None,
        output=tmp_path / "result",
        concurrency=2,
        repeats=2,
        warmup=2,
        seed=42,
        cache_mode="full-prefill",
        timeout=10,
        engine_url="http://engine",
        base_url="http://adapter",
        engine_manifest=None,
        assistant_prefix=None,
    )


@pytest.mark.asyncio
async def test_runner_outputs_and_excludes_warmup_repeat_quality(tmp_path):
    args = arguments(tmp_path)
    seen = []

    def transport(request):
        if request.url.path == "/model_info":
            return httpx.Response(200, json={"served_model_name": "decision-model"})
        if request.url.path == "/server_info":
            return httpx.Response(
                200,
                json={
                    "disable_radix_cache": True,
                    "mm_preprocess_cache_size_mb": 0,
                    "api_key": "secret",
                },
            )
        payload = json.loads(request.content)
        seen.append(payload)
        assert "expected" not in payload and "metadata" not in payload
        assert all(
            "label" not in q and "src" not in q for q in payload["questions"].values()
        )
        return httpx.Response(200, json=response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await evaluate(args, client)
    assert len(seen) == 6
    assert result["status"] == "complete"
    assert result["successful_requests"] == 4
    assert result["suites"]["fixture/development"]["clean"]["n"] == 6
    assert result["latency_ms"]["n"] == 4
    assert "secret" not in (args.output / "manifest.json").read_text()
    assert len((args.output / "predictions.jsonl").read_text().splitlines()) == 4
    comparison = collect(args.output)
    assert len(comparison) == 1 and comparison[0]["clean_questions"] == 6
    assert comparison[0]["partial_dataset"] is True
    with pytest.raises(FileExistsError):
        await evaluate(args)
    (args.output / "invalidated.json").write_text(
        json.dumps({"reason": "Tokenization failed an independent canonical check"})
    )
    assert collect(args.output) == []


@pytest.mark.asyncio
async def test_runner_errors_do_not_produce_headline_or_retry(tmp_path):
    args = arguments(tmp_path)
    args.warmup = 0
    calls = 0

    def transport(request):
        nonlocal calls
        if request.url.path == "/model_info":
            return httpx.Response(200, json={"served_model_name": "decision-model"})
        if request.url.path == "/server_info":
            return httpx.Response(
                200,
                json={"disable_radix_cache": True, "mm_preprocess_cache_size_mb": 0},
            )
        calls += 1
        body = response()
        body["usage"]["output_tokens"] = 1
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await evaluate(args, client)
    assert result["status"] == "failed"
    assert result["suites"] == {}
    assert result["errors"] == 4 and calls == 4
    assert (args.output / "failures.json").exists()
    assert collect(args.output) == []


def test_locked_split_and_data_integrity(tmp_path):
    locked = item()
    locked["split"] = "test"
    args = arguments(tmp_path, [locked])
    with pytest.raises(ValueError, match="allow-test"):
        load_data(args.data, args.model)
    assert len(load_data(args.data, args.model, allow_test=True)[0]) == 1
    args.data[0].with_suffix(".manifest.json").write_text(
        json.dumps({"data_sha256": "bad"})
    )
    with pytest.raises(ValueError, match="checksum"):
        load_data(args.data, args.model, allow_test=True)


def test_prepared_subset_metadata_is_retained(tmp_path):
    args = arguments(tmp_path)
    raw = args.data[0].read_bytes()
    args.data[0].with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "data_sha256": hashlib.sha256(raw).hexdigest(),
                "selected": {"records": 2},
                "selection": {"is_full_partition": False},
            }
        )
    )
    _, sources = load_data(args.data, args.model)
    assert sources[0]["prepared_manifest"]["selection"]["is_full_partition"] is False


def test_launch_cannot_label_another_checkpoint():
    launch = {
        "model": {
            "repo_id": "Qwen/one",
            "revision": "abc",
            "dtype": "bfloat16",
            "quantization": None,
        },
        "profile": "one",
        "engine": {"revision": "def"},
    }
    snapshot = {
        "server_info": {
            "model_path": "Qwen/one",
            "revision": "abc",
            "dtype": "bfloat16",
            "quantization": None,
        },
        "model_info": {"model_path": "Qwen/one"},
    }
    validate_launch(snapshot, launch)
    snapshot["server_info"]["revision"] = "changed"
    with pytest.raises(ValueError, match="revision"):
        validate_launch(snapshot, launch)
