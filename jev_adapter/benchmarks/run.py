"""Evaluate frozen decisions against a running, untrained-model adapter.

No retries, generation, calibration, truncation, or training. Latency includes
adapter HTTP, tokenization, engine scheduling and forward work; it is NOT CUDA
kernel time. Model load and warmup are excluded. Failed runs have no headline.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from jev_adapter.protocol import SystemOneRequest

from .metrics import latency_summary, prediction_rows, summarize


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def load_data(paths, model, allow_test=False, limit=None):
    items, seen, sources = [], set(), []
    for path in paths:
        raw = path.read_bytes()
        loaded = [json.loads(line) for line in raw.splitlines() if line.strip()]
        source = {
            "path": str(path.resolve()),
            "sha256": digest(raw),
            "records": len(loaded),
        }
        manifest_path = path.with_suffix(".manifest.json")
        if manifest_path.exists():
            prepared = json.loads(manifest_path.read_text())
            if prepared.get("data_sha256") != digest(raw):
                raise ValueError(f"prepared dataset checksum mismatch: {path}")
            if prepared.get("selected", {}).get("records") != len(loaded):
                raise ValueError(f"prepared dataset count mismatch: {path}")
            source["prepared_manifest"] = prepared
        sources.append(source)
        for item in loaded:
            key = item["suite"], item["split"], item["id"]
            if key in seen:
                raise ValueError(f"duplicate input record: {key}")
            seen.add(key)
            if item["split"] == "test" and not allow_test:
                raise ValueError(
                    "test data requires --allow-test; use development first"
                )
            request = SystemOneRequest.model_validate(
                {**item["record"], "model": model}
            )
            if set(request.questions) != set(item["expected"]):
                raise ValueError(f"question/label mismatch: {key}")
            for qid, expected in item["expected"].items():
                labels, target, label = (
                    expected["labels"],
                    expected["target"],
                    expected["label"],
                )
                if (
                    len(labels) < 2
                    or len(labels) != len(set(labels))
                    or type(label) is not int
                    or not 0 <= label < len(labels)
                    or target != [int(i == label) for i in range(len(labels))]
                ):
                    raise ValueError(
                        "this runner requires valid one-hot objective labels"
                    )
                question = request.questions[qid]
                actual = (
                    list(question.criteria)
                    if question.type == "choice"
                    else ["false", "true"]
                    if question.type == "noul"
                    else [str(i) for i in range(len(question.criteria))]
                )
                if set(labels) != set(actual) or expected["type"] != question.type:
                    raise ValueError(
                        f"expected labels/type differ from request: {key}/{qid}"
                    )
                reference = expected.get("reference_probs")
                if reference is not None and (
                    not isinstance(reference, list)
                    or len(reference) != len(labels)
                    or any(
                        type(p) not in (int, float)
                        or not math.isfinite(p)
                        or not 0 <= p <= 1
                        for p in reference
                    )
                    or abs(math.fsum(reference) - 1) > 1e-8
                ):
                    raise ValueError("invalid reference probability distribution")
            items.append(item)
    if limit is not None:
        items = items[:limit]
    if not items:
        raise ValueError("no input records")
    return items, sources


async def engine_snapshot(client, base_url, cache_mode):
    headers = {}
    if key := os.environ.get("SGLANG_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    results = {}
    allowed = {
        "model_path",
        "model_type",
        "architectures",
        "served_model_name",
        "revision",
        "tokenizer_revision",
        "dtype",
        "quantization",
        "tp_size",
        "tp",
        "version",
        "context_length",
        "max_total_tokens",
        "max_running_requests",
        "chunked_prefill_size",
        "mem_fraction_static",
        "disable_radix_cache",
        "mm_preprocess_cache_size_mb",
        "enable_prefix_mm_cache",
        "enable_mm_global_cache",
        "speculative_algorithm",
        "is_generation",
        "has_image_understanding",
        "attention_backend",
        "moe_runner_backend",
    }
    for name in ("model_info", "server_info"):
        response = await client.get(base_url.rstrip("/") + "/" + name, headers=headers)
        if response.status_code == 404:
            response = await client.get(
                base_url.rstrip("/") + "/get_" + name, headers=headers
            )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise TypeError(f"invalid engine {name}")
        results[name] = {k: v for k, v in value.items() if k in allowed}
    config = results["server_info"]
    if config.get("speculative_algorithm") is not None:
        raise ValueError("disable speculative decoding for this benchmark")
    if cache_mode == "full-prefill":
        if config.get("disable_radix_cache") is not True:
            raise ValueError("full-prefill requires engine --disable-radix-cache")
        if config.get("mm_preprocess_cache_size_mb") != 0:
            raise ValueError("full-prefill requires --mm-preprocess-cache-size-mb 0")
        if config.get("enable_prefix_mm_cache") or config.get("enable_mm_global_cache"):
            raise ValueError("full-prefill requires multimodal feature caches disabled")
    return results


def checked_usage(response):
    usage, meta = response.get("usage", {}), response.get("metadata", {})
    if type(usage.get("output_tokens")) is not int or usage["output_tokens"] != 0:
        raise ValueError("engine must return exactly zero output tokens")
    if type(usage.get("input_tokens")) is not int or usage["input_tokens"] < 1:
        raise ValueError("missing or invalid input-token accounting")
    if type(meta.get("evaluations")) is not int or meta["evaluations"] < 1:
        raise ValueError("missing forward evaluation count")
    elapsed = meta.get("adapter_elapsed_ms")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("missing or invalid adapter elapsed time")
    if not isinstance(response.get("model"), str) or not response["model"]:
        raise ValueError("missing served model identity")
    return usage, meta


def validate_launch(snapshot, launch):
    model, engine = launch["model"], launch["engine"]
    server, info = snapshot["server_info"], snapshot["model_info"]
    for field, expected in {
        "model_path": model["repo_id"],
        "revision": model["revision"],
        "dtype": model["dtype"],
        "quantization": model["quantization"],
    }.items():
        if server.get(field) != expected:
            raise ValueError(f"launch manifest {field} differs from running engine")
    if info.get("model_path") != model["repo_id"]:
        raise ValueError("launch manifest model differs from running engine model_info")
    if not engine.get("revision") or not launch.get("profile"):
        raise ValueError("launch manifest lacks pinned engine/profile identity")


async def evaluate(args, client=None):
    items, inputs = load_data(args.data, args.model, args.allow_test, args.limit)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "started",
        "python": sys.version,
        "platform": platform.platform(),
        "requested_model": args.model,
        "base_url": args.base_url,
        "engine_url": args.engine_url,
        "data": inputs,
        "limit": args.limit,
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "cache_mode": args.cache_mode,
        "options": {"temperature": 1.0, "permutations": 1, "enable_thinking": False},
        "latency_scope": "client HTTP wall time, excluding queue before dispatch; NOT GPU kernel time",
        "quality_repeats": "first measured repetition only",
        "assistant_prefix": args.assistant_prefix,
        "code_sha256": {
            str(p.relative_to(Path(__file__).parents[1])): digest(p.read_bytes())
            for p in Path(__file__).parents[1].rglob("*.py")
        },
    }
    write_json(args.output / "manifest.json", manifest)
    owned = client is None
    if owned:
        client = httpx.AsyncClient(
            timeout=args.timeout, limits=httpx.Limits(max_connections=args.concurrency)
        )
    headers = {}
    if key := os.environ.get("JEV_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    results, failures = [], []
    try:
        manifest["engine"] = await engine_snapshot(
            client, args.engine_url, args.cache_mode
        )
        expected_served = manifest["engine"]["model_info"].get("served_model_name")
        if not expected_served:
            raise ValueError("engine did not identify its served model")
        if args.engine_manifest:
            manifest["launch_manifest"] = json.loads(args.engine_manifest.read_text())
            validate_launch(manifest["engine"], manifest["launch_manifest"])
        write_json(args.output / "manifest.json", manifest)

        async def call(item):
            request = {
                **item["record"],
                "model": args.model,
                "options": {
                    "temperature": 1.0,
                    "permutations": 1,
                    "return_logprobs": True,
                },
            }
            if args.assistant_prefix is not None:
                request["options"]["assistant_prefix"] = args.assistant_prefix
            started = time.perf_counter()
            response = await client.post(
                args.base_url.rstrip("/") + "/v1/systemone",
                headers=headers,
                json=request,
            )
            response.raise_for_status()
            body = response.json()
            elapsed = (time.perf_counter() - started) * 1000
            usage, meta = checked_usage(body)
            if body["model"] != expected_served:
                raise ValueError("adapter response model differs from engine snapshot")
            if meta["evaluations"] != len(item["record"]["questions"]):
                raise ValueError("expected one forward evaluation per question")
            rows = prediction_rows(item, body)
            return {
                "request_sha256": digest(json.dumps(request, sort_keys=True).encode()),
                "id": item["id"],
                "suite": item["suite"],
                "split": item["split"],
                "latency_ms": elapsed,
                "adapter_ms": meta["adapter_elapsed_ms"],
                "input_tokens": usage["input_tokens"],
                "evaluations": meta["evaluations"],
                "served_model": body["model"],
                "response": body,
                "rows": rows,
            }

        for i in range(args.warmup):
            # Span the full input so warmup includes more than the first task.
            item = items[(i * len(items) // max(args.warmup, 1)) % len(items)]
            print(
                f"Warmup {i + 1}/{args.warmup}: {item['suite']} {item['id']}",
                flush=True,
            )
            await call(item)
        jobs = [
            (i, repeat) for repeat in range(args.repeats) for i in range(len(items))
        ]
        random.Random(args.seed).shuffle(jobs)
        semaphore = asyncio.Semaphore(args.concurrency)
        started = time.perf_counter()
        with (args.output / "predictions.jsonl").open("w") as handle:

            async def job(index, repeat):
                async with semaphore:
                    try:
                        result = await call(items[index])
                        result.update(index=index, repeat=repeat)
                        results.append(result)
                        handle.write(
                            json.dumps(result, ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                        handle.flush()
                    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                        failures.append(
                            {
                                "index": index,
                                "repeat": repeat,
                                "id": items[index]["id"],
                                "suite": items[index]["suite"],
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                    done = len(results) + len(failures)
                    if done % 50 == 0 or done == len(jobs):
                        print(
                            f"{done}/{len(jobs)} requests; errors={len(failures)}",
                            flush=True,
                        )

            await asyncio.gather(*(job(index, repeat) for index, repeat in jobs))
        elapsed = time.perf_counter() - started
        served = {r["served_model"] for r in results}
        if len(served) > 1:
            failures.append({"error": "served model changed during run"})
        report = {
            "status": "failed" if failures else "complete",
            "requested_records": len(items),
            "requested_questions": sum(len(i["expected"]) for i in items),
            "attempted_requests": len(jobs),
            "successful_requests": len(results),
            "errors": len(failures),
            "served_models": sorted(served),
            "elapsed_measurement_s": elapsed,
            "requests_per_second": len(results) / elapsed,
            "questions_per_second": sum(r["evaluations"] for r in results) / elapsed,
            "latency_ms": latency_summary([r["latency_ms"] for r in results]),
            "adapter_ms": latency_summary([r["adapter_ms"] for r in results]),
            "input_tokens_per_request": latency_summary(
                [r["input_tokens"] for r in results]
            ),
            "suites": {},
            "latency_scope": manifest["latency_scope"],
            "partial_dataset": args.limit is not None
            or any(
                not source.get("prepared_manifest", {})
                .get("selection", {})
                .get("is_full_partition", False)
                for source in inputs
            ),
        }
        if not failures:
            grouped, times = defaultdict(list), defaultdict(list)
            for result in sorted(results, key=lambda r: (r["index"], r["repeat"])):
                key = result["suite"] + "/" + result["split"]
                times[key].append(result["latency_ms"])
                if result["repeat"] == 0:
                    grouped[key].extend(result["rows"])
            report["suites"] = {
                key: {**summarize(rows), "latency_ms": latency_summary(times[key])}
                for key, rows in grouped.items()
            }
        write_json(args.output / "report.json", report)
        if failures:
            write_json(args.output / "failures.json", failures)
        manifest.update(status=report["status"], served_models=sorted(served))
        write_json(args.output / "manifest.json", manifest)
        return report
    except BaseException as error:
        manifest.update(
            status="failed", failure_type=type(error).__name__, failure=str(error)
        )
        write_json(args.output / "manifest.json", manifest)
        raise
    finally:
        if owned:
            await client.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30120")
    parser.add_argument("--engine-url", default="http://127.0.0.1:30000")
    parser.add_argument("--engine-manifest", type=Path)
    parser.add_argument("--model", default="decision-model")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--assistant-prefix")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--cache-mode",
        choices=["full-prefill", "server-default"],
        default="full-prefill",
    )
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.repeats < 1
        or args.warmup < 0
        or (args.limit is not None and args.limit < 1)
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("counts and timeout must be positive; warmup may be zero")
    report = asyncio.run(evaluate(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
