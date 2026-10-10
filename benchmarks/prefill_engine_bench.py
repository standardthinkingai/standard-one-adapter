"""Bounded, sequential benchmark for one /v1/systemone endpoint at a time.

`run` records a closed-loop concurrency sweep without starting or stopping a
server. `compare` checks two saved sweeps before reporting performance ratios.
It never contacts a GPU during comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import time
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from jev_adapter.benchmarks.metrics import latency_summary, metrics, prediction_rows
from jev_adapter.benchmarks.run import checked_usage, load_data, write_json

if __package__:
    from .prefill_compare import (
        compare_input_tokens,
        compare_quality,
        first_pass_rows,
        quality_by_suite,
        request_for,
    )
    from .prefill_engine_smoke import validate_success
else:
    from prefill_compare import (
        compare_input_tokens,
        compare_quality,
        first_pass_rows,
        quality_by_suite,
        request_for,
    )
    from prefill_engine_smoke import validate_success

DEFAULT_CONCURRENCY = (8, 16, 32, 64, 128)
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
SERVER_FIELDS = (
    "model_path",
    "revision",
    "dtype",
    "enable_fp32_lm_head",
    "quantization",
    "disable_radix_cache",
    "mm_preprocess_cache_size_mb",
    "enable_prefix_mm_cache",
    "enable_mm_global_cache",
    "speculative_algorithm",
    "chunked_prefill_size",
    "context_length",
    "max_total_tokens",
    "max_prefill_tokens",
    "mem_fraction_static",
    "max_running_requests",
    "prefill_max_requests",
    "decision_readout_path",
    "tp_size",
    "pp_size",
    "dp_size",
    "cpu_offload_gb",
    "disable_prefill_cuda_graph",
    "disable_decode_cuda_graph",
    "disable_cuda_graph",
    "cuda_graph_config",
    "cuda_graph_backend_decode",
    "cuda_graph_backend_prefill",
    "cuda_graph_max_bs_decode",
    "cuda_graph_max_bs_prefill",
    "cuda_graph_max_seq_len_prefill",
    "attention_backend",
    "sampling_backend",
)
MODEL_FIELDS = ("model_path", "served_model_name", "architectures", "model_type")
RESOURCE_FIELDS = tuple(key for key in SERVER_FIELDS if key != "decision_readout_path")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    )


def clean_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("URL must be an HTTP(S) base URL without credentials or query")
    return value.rstrip("/")


def parse_repeats_by_concurrency(value):
    def unique_object(pairs):
        result = {}
        for key, count in pairs:
            if key in result:
                raise ValueError("repeats-by-concurrency contains duplicate keys")
            result[key] = count
        return result

    if isinstance(value, str):
        value = json.loads(value, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not re.fullmatch(r"[1-9][0-9]*", key)
        or type(count) is not int
        or not 1 <= count <= 1_000_000
        for key, count in value.items()
    ):
        raise ValueError(
            "repeats-by-concurrency requires C-string keys and positive integer counts"
        )
    return dict(value)


def repeat_counts(concurrency, repeats, override=None):
    if override is None:
        if type(repeats) is not int or not 1 <= repeats <= 1_000_000:
            raise ValueError("repeats must be a positive integer")
        return {str(c): repeats for c in concurrency}
    result = parse_repeats_by_concurrency(override)
    if set(result) != {str(c) for c in concurrency}:
        raise ValueError(
            "repeats-by-concurrency keys must exactly match selected concurrency"
        )
    return result


def manifest_repeat_counts(manifest):
    return manifest.get(
        "repeats_by_concurrency",
        {str(c): manifest.get("repeats") for c in manifest.get("concurrency", [])},
    )


def manifest_order_hashes(manifest):
    return manifest.get(
        "job_order_sha256_by_concurrency",
        {
            str(c): manifest.get("job_order_sha256")
            for c in manifest.get("concurrency", [])
        },
    )


def http_pool_capacity(concurrency, requested=None):
    capacity = max(concurrency) if requested is None else requested
    if type(capacity) is not int or not max(concurrency) <= capacity <= 128:
        raise ValueError("http_max_connections must be an integer from max(C) to 128")
    return capacity


def validate_counts(args) -> None:
    if (
        not args.concurrency
        or len(set(args.concurrency)) != len(args.concurrency)
        or any(type(c) is not int or not 1 <= c <= 128 for c in args.concurrency)
        or type(args.warmup) is not int
        or not 0 <= args.warmup <= 4096
        or type(args.repeats) is not int
        or not 1 <= args.repeats <= 1_000_000
        or type(args.limit) is not int
        or not 1 <= args.limit <= 4096
        or type(args.seed) is not int
        or not math.isfinite(args.min_measure_seconds)
        or not 0 <= args.min_measure_seconds <= 3600
        or not math.isfinite(args.timeout)
        or not 0 < args.timeout <= 300
    ):
        raise ValueError(
            "concurrency, warmup, repeats, limit, seed, min_measure_seconds, or timeout invalid"
        )
    if not args.model.strip():
        raise ValueError("model must be nonempty")
    if getattr(args, "http_client", "httpx") not in ("httpx", "aiohttp"):
        raise ValueError("http_client must be httpx or aiohttp")
    http_pool_capacity(args.concurrency, getattr(args, "http_max_connections", None))
    repeat_counts(
        args.concurrency, args.repeats, getattr(args, "repeats_by_concurrency", None)
    )
    clean_url(args.base_url)
    clean_url(args.engine_url or args.base_url)


def precision_settings(settings):
    """Resolve old BF16 manifests and require an explicit, truthful FP8 profile."""
    precision = settings.get("precision")
    if precision not in ("bfloat16", "float16"):
        raise ValueError("precision must be bfloat16 or float16")
    quantization = settings.get("quantization")
    if quantization is None:
        profile = f"unquantized_{precision}"
    elif quantization == "fp8":
        if precision != "bfloat16" or settings.get("head_mode") not in (
            "logprobs",
            "lm_rows",
        ):
            raise ValueError("fp8 requires BF16 activations and an unquantized LM head")
        profile = "fp8_w8a8_backbone_bf16_head"
    else:
        raise ValueError("quantization must be null or native online fp8")
    if settings.get("precision_profile", profile) != profile:
        raise ValueError(
            "precision_profile disagrees with quantization/activation dtype"
        )
    return quantization, profile


def read_settings(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("settings manifest must be a version 1 JSON object")
    for key in (
        "hardware",
        "model_path",
        "model_revision",
        "precision",
        "prompt_contract",
        "label_contract",
        "head_mode",
    ):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"settings manifest requires nonempty {key}")
    if not HEX40.fullmatch(value["model_revision"]):
        raise ValueError("model_revision must be a 40-digit lowercase git revision")
    if value["precision"] not in ("bfloat16", "float16"):
        raise ValueError("precision must be bfloat16 or float16")
    if value["head_mode"] not in ("logprobs", "lm_rows", "learned"):
        raise ValueError("head_mode must be logprobs, lm_rows, or learned")
    quantization, profile = precision_settings(value)
    value.setdefault("quantization", quantization)
    value.setdefault("precision_profile", profile)
    labels, token_ids = value.get("labels"), value.get("token_ids")
    if (
        not isinstance(labels, list)
        or not isinstance(token_ids, list)
        or not 2 <= len(labels) <= 255
        or len(labels) != len(token_ids)
        or any(not isinstance(label, str) or not label for label in labels)
        or any(type(token) is not int or token < 0 for token in token_ids)
        or len(set(labels)) != len(labels)
        or len(set(token_ids)) != len(token_ids)
    ):
        raise ValueError(
            "settings manifest needs matching unique labels/token_ids arrays"
        )
    artifact = value.get("head_artifact")
    if value["head_mode"] == "logprobs":
        if artifact is not None:
            raise ValueError("logprobs mode must have head_artifact=null")
    else:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("path"), str)
            or not artifact["path"]
            or not isinstance(artifact.get("sha256"), str)
            or not HEX64.fullmatch(artifact["sha256"])
        ):
            raise ValueError("readout modes require artifact path and SHA256")
        if file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("head artifact SHA256 differs from settings manifest")
        from jev_adapter.inprocess import _load_metadata

        metadata = _load_metadata(artifact["path"])
        expected_metadata = {
            "mode": value["head_mode"],
            "labels": labels,
            "token_ids": token_ids,
            "engine_model_path": value["model_path"],
            "revision": value["model_revision"],
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(f"head artifact metadata {key} differs from settings")
    return value, sha256(raw)


def read_token_audit(path, settings, workload_hash, requests):
    if path is None:
        return None
    raw = path.read_bytes()
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("workload_sha256") != workload_hash
        or value.get("model_revision") != settings["model_revision"]
        or value.get("prompt_contract") != settings["prompt_contract"]
        or value.get("label_contract") != settings["label_contract"]
    ):
        raise ValueError("token ID audit does not match the selected workload/contract")
    entries = value.get("requests")
    if not isinstance(entries, list) or len(entries) != len(requests):
        raise ValueError("token ID audit must cover every selected request")
    for request, entry in zip(requests, entries):
        if (
            not isinstance(entry, dict)
            or entry.get("request_sha256") != canonical_sha256(request)
            or not isinstance(entry.get("input_token_ids_sha256"), str)
            or not HEX64.fullmatch(entry["input_token_ids_sha256"])
        ):
            raise ValueError("token ID audit request/hash mismatch")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256(raw),
        "input_token_ids_sha256": [
            entry["input_token_ids_sha256"] for entry in entries
        ],
    }


async def _get_info(client, base_url, name, headers):
    response = await client.get(f"{base_url}/{name}", headers=headers)
    if response.status_code == 404:
        response = await client.get(f"{base_url}/get_{name}", headers=headers)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise TypeError(f"{name} must return a JSON object")
    return data


async def engine_snapshot(client, base_url, headers, settings, max_concurrency):
    quantization, _ = precision_settings(settings)
    model = await _get_info(client, base_url, "model_info", headers)
    server = await _get_info(client, base_url, "server_info", headers)
    expected = {
        "model_path": settings["model_path"],
        "revision": settings["model_revision"],
        "dtype": settings["precision"],
        "quantization": quantization,
        "disable_radix_cache": True,
        "mm_preprocess_cache_size_mb": 0,
    }
    for key, value in expected.items():
        if server.get(key) != value:
            raise ValueError(f"engine {key}={server.get(key)!r}; expected {value!r}")
    if model.get("model_path") != settings["model_path"]:
        raise ValueError("model_info model_path differs from settings manifest")
    if (
        server.get("enable_prefix_mm_cache")
        or server.get("enable_mm_global_cache")
        or server.get("speculative_algorithm") is not None
    ):
        raise ValueError("cache or speculative decoding is enabled")
    if any(server.get(key) != 1 for key in ("tp_size", "pp_size", "dp_size")):
        raise ValueError("final benchmark requires TP=PP=DP=1")
    if server.get("cpu_offload_gb") not in (0, 0.0):
        raise ValueError("final benchmark requires cpu_offload_gb=0")
    running = server.get("max_running_requests")
    if type(running) is not int or running < max_concurrency:
        raise ValueError("engine max_running_requests is below benchmark concurrency")
    path = server.get("decision_readout_path")
    artifact = settings.get("head_artifact")
    if settings["head_mode"] == "logprobs":
        if path is not None:
            raise ValueError("logprobs settings point to a decision-readout server")
    elif path != artifact["path"]:
        raise ValueError("running decision readout differs from head artifact path")
    return {
        "model_info": {key: model.get(key) for key in MODEL_FIELDS},
        "server_info": {key: server.get(key) for key in SERVER_FIELDS},
    }


def jobs_for(items, repeats, seed):
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    return [
        (ordinal, index, repeat)
        for ordinal, (index, repeat) in enumerate(
            (index, repeat) for repeat in range(repeats) for index in order
        )
    ]


def warmup_jobs(items, count):
    return [
        (ordinal, (ordinal * len(items) // count) % len(items), -1)
        for ordinal in range(count)
    ]


def max_options_in(items):
    return max(
        2 if question["type"] == "noul" else len(question["criteria"])
        for item in items
        for question in item["record"]["questions"].values()
    )


async def call_one(
    client, url, headers, item, request, ordinal, index, repeat, required_capacity
):
    started_ns = time.perf_counter_ns()  # after worker admission
    outcome = {
        "admitted_ns": started_ns,
        "ordinal": ordinal,
        "index": index,
        "repeat": repeat,
        "id": item["id"],
        "suite": item["suite"],
        "split": item["split"],
        "request_sha256": canonical_sha256(request),
        "request": request,
    }
    try:
        outcome["http_post_started_ns"] = time.perf_counter_ns()
        try:
            response = await client.post(url, headers=headers, json=request)
        finally:
            outcome["http_post_completed_ns"] = time.perf_counter_ns()
        completed_ns = time.perf_counter_ns()
        outcome["completed_ns"] = completed_ns
        outcome["client_latency_ms"] = (completed_ns - started_ns) / 1e6
        outcome["http_status"] = response.status_code
        outcome["raw_response_text"] = response.text
        response.raise_for_status()
        body = response.json()
        json.dumps(body, ensure_ascii=False, allow_nan=False)
        validated = validate_success(body, request, request["model"])
        checked_usage(body)
        capacity = body["metadata"].get("service_max_concurrency")
        if type(capacity) is not int or capacity < required_capacity:
            raise ValueError("service_max_concurrency is missing or below C")
        outcome.update(
            status="success",
            input_tokens=validated["input_tokens"],
            cached_tokens=validated["cached_tokens"],
            decisions=validated["evaluations"],
            service_max_concurrency=capacity,
            rows=prediction_rows(item, body),
        )
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as error:
        outcome.update(status="error", error_type=type(error).__name__)
        if isinstance(error, httpx.HTTPStatusError):
            outcome["status_code"] = error.response.status_code
        elif isinstance(error, (ValueError, TypeError, KeyError)):
            outcome["detail"] = str(error)
    finally:
        outcome.setdefault("completed_ns", time.perf_counter_ns())
        outcome.setdefault(
            "client_latency_ms", (outcome["completed_ns"] - started_ns) / 1e6
        )
    return outcome


async def closed_loop(client, url, headers, items, requests, jobs, concurrency):
    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    outcomes = []

    async def worker():
        while True:
            try:
                ordinal, index, repeat = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            outcome = await call_one(
                client,
                url,
                headers,
                items[index],
                requests[index],
                ordinal,
                index,
                repeat,
                concurrency,
            )
            outcomes.append(outcome)
            queue.task_done()

    started_ns = time.perf_counter_ns()
    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(jobs)))))
    completed_ns = time.perf_counter_ns()
    return (
        sorted(outcomes, key=lambda row: row["ordinal"]),
        (completed_ns - started_ns) / 1e9,
        started_ns,
        completed_ns,
    )


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def http_inflight_peak(outcomes):
    events = []
    for row in outcomes:
        start, end = row.get("http_post_started_ns"), row.get("http_post_completed_ns")
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError("HTTP POST timing is missing or invalid")
        events.extend(((start, 1), (end, -1)))
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


def quality_summary(rows):
    """Score only first-pass clean, knowable, non-synthetic questions."""
    eligible = [
        row
        for row in rows
        if row["variant"] == "clean"
        and row["source"] != "unknowable"
        and not row["suite"].startswith("synthetic-")
    ]
    return {
        "quality_overall": metrics(eligible),
        "quality_by_type": {
            kind: metrics([row for row in eligible if row["type"] == kind])
            for kind in ("choice", "noul", "score")
        },
        "quality_by_suite": quality_by_suite(eligible),
    }


def summarize(outcomes, elapsed, item_count, concurrency, min_measure_seconds):
    successes = [row for row in outcomes if row["status"] == "success"]
    failures = [row for row in outcomes if row["status"] != "success"]
    peak = http_inflight_peak(outcomes)
    measurement_failures = []
    if peak < concurrency:
        measurement_failures.append(
            {"error_type": "ClientConcurrencyUnderfill", "peak": peak}
        )
    if elapsed < min_measure_seconds:
        measurement_failures.append(
            {
                "error_type": "MeasurementTooShort",
                "observed_seconds": elapsed,
                "required_seconds": min_measure_seconds,
            }
        )
    capacities = {row["service_max_concurrency"] for row in successes}
    if successes and len(capacities) != 1:
        measurement_failures.append({"error_type": "InconsistentServiceCapacity"})
    cached = sum(row["cached_tokens"] for row in successes)
    if cached:
        failures.append({"error_type": "CacheHit", "cached_tokens": cached})
    decisions = sum(row["decisions"] for row in successes)
    tokens = sum(row["input_tokens"] for row in successes)
    result = {
        "status": "complete" if not failures and not measurement_failures else "failed",
        "attempted_requests": len(outcomes),
        "successful_requests": len(successes),
        "errors": len(failures),
        "error_samples": failures[:20],
        "measurement_failures": measurement_failures,
        "elapsed_s": elapsed,
        "minimum_measure_seconds": min_measure_seconds,
        "duration_requirement_met": elapsed >= min_measure_seconds,
        "closed_loop": True,
        "client_http_inflight_peak": peak,
        "service_max_concurrency": next(iter(capacities))
        if len(capacities) == 1
        else None,
        "latency_start": "after worker admission, immediately before HTTP POST",
        "client_latency_ms": latency_summary(
            [row["client_latency_ms"] for row in successes]
        ),
        "requests_per_second": len(successes) / elapsed if elapsed > 0 else None,
        "decisions_per_second": decisions / elapsed if elapsed > 0 else None,
        "input_tokens_per_second": tokens / elapsed if elapsed > 0 else None,
        "decisions": decisions,
        "input_tokens": tokens,
        "cached_tokens": cached,
        "quality_overall": None,
        "quality_by_type": {kind: None for kind in ("choice", "noul", "score")},
        "quality_by_suite": {},
    }
    if not failures:
        rows = first_pass_rows(outcomes, item_count)
        result.update(quality_summary(rows))
    return result


def make_http_client(args):
    """Create an owned client and record the transport actually selected."""
    kind = getattr(args, "http_client", "httpx")
    capacity = http_pool_capacity(
        args.concurrency, getattr(args, "http_max_connections", None)
    )
    helper_sha = None
    if kind == "aiohttp":
        from jev_adapter import http_client

        version = package_version("aiohttp")
        helper_sha = file_sha256(Path(http_client.__file__))
        client = http_client.AiohttpCompatClient(
            concurrency=capacity, timeout=args.timeout
        )
        transport = "aiohttp.TCPConnector"
    else:
        version = httpx.__version__
        client = httpx.AsyncClient(
            timeout=args.timeout,
            limits=httpx.Limits(
                max_connections=capacity, max_keepalive_connections=capacity
            ),
        )
        transport = "httpx.AsyncHTTPTransport"
    return client, {
        "kind": kind,
        "class": f"{type(client).__module__}.{type(client).__qualname__}",
        "transport": transport,
        "library_version": version,
        "httpx_response_version": httpx.__version__,
        "max_connections": capacity,
        "timeout_seconds": args.timeout,
        "timeout_scope": "total" if kind == "aiohttp" else "per-operation",
        "trust_env": kind == "httpx",
        "helper_source_sha256": helper_sha,
    }


async def run(args, client=None):
    validate_counts(args)
    capacity = http_pool_capacity(
        args.concurrency, getattr(args, "http_max_connections", None)
    )
    if client is not None and getattr(args, "http_max_connections", None) is not None:
        raise ValueError("http_max_connections cannot configure an external client")
    settings, settings_sha = read_settings(args.settings_manifest)
    items, sources = load_data(args.data, args.model, args.allow_test, args.limit)
    max_options = max_options_in(items)
    if max_options > len(settings["labels"]):
        raise ValueError("selected dataset exceeds configured head label count")
    requests = [request_for(item, args.model, None) for item in items]
    repeats = repeat_counts(
        args.concurrency, args.repeats, getattr(args, "repeats_by_concurrency", None)
    )
    if len(items) * sum(repeats.values()) > 1_000_000:
        raise ValueError("total measured requests exceed the 1000000-request bound")
    if any(len(items) * repeats[str(c)] < c for c in args.concurrency):
        raise ValueError("measured jobs must reach each selected concurrency C")
    orders_by_repeat = {
        count: jobs_for(items, count, args.seed) for count in set(repeats.values())
    }
    orders = {str(c): orders_by_repeat[repeats[str(c)]] for c in args.concurrency}
    order_hashes = {key: canonical_sha256(order) for key, order in orders.items()}
    if args.warmup and args.warmup < max(args.concurrency):
        raise ValueError("warmup must be zero or at least the largest concurrency C")
    workload_hash = canonical_sha256(requests)
    token_audit = read_token_audit(
        getattr(args, "token_audit", None), settings, workload_hash, requests
    )
    url = clean_url(args.base_url)
    engine_url = clean_url(args.engine_url or args.base_url)
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "selected_items.jsonl", items)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "started",
        "base_url": url,
        "engine_url": engine_url,
        "model": args.model,
        "settings": settings,
        "settings_manifest_sha256": settings_sha,
        "label_array_sha256": canonical_sha256(
            {"labels": settings["labels"], "token_ids": settings["token_ids"]}
        ),
        "data": sources,
        "selected_records": len(items),
        "selected_items_sha256": file_sha256(args.output / "selected_items.jsonl"),
        "selected_decisions": sum(len(item["expected"]) for item in items),
        "max_options_in_workload": max_options,
        "workload_sha256": workload_hash,
        "token_id_audit": token_audit,
        "job_order_sha256": next(iter(order_hashes.values()))
        if len(set(order_hashes.values())) == 1
        else None,
        "job_order_sha256_by_concurrency": order_hashes,
        "concurrency": args.concurrency,
        "warmup_requests_per_c": args.warmup,
        "repeats": args.repeats,
        "repeats_by_concurrency": repeats,
        "min_measure_seconds": args.min_measure_seconds,
        "seed": args.seed,
        "limit": args.limit,
        "request_options": requests[0]["options"],
        "http_pool_max_connections": capacity,
        "http_client": {
            "kind": "pending" if client is None else "external",
            "class": None
            if client is None
            else f"{type(client).__module__}.{type(client).__qualname__}",
        },
        "client_code_sha256": {
            name: file_sha256(Path(__file__).parent / name)
            for name in (
                "prefill_engine_bench.py",
                "prefill_compare.py",
                "prefill_engine_smoke.py",
            )
        },
    }
    write_json(args.output / "manifest.json", manifest)
    owned = client is None
    report = {"status": "started", "concurrency": {}}
    try:
        if owned:
            client, manifest["http_client"] = make_http_client(args)
            helper_sha = manifest["http_client"]["helper_source_sha256"]
            if helper_sha is not None:
                manifest["client_code_sha256"]["jev_adapter/http_client.py"] = (
                    helper_sha
                )
        write_json(args.output / "manifest.json", manifest)
        key = os.environ.get(args.key_env) or os.environ.get("JEV_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        manifest["engine"] = await engine_snapshot(
            client, engine_url, headers, settings, max(args.concurrency)
        )
        write_json(args.output / "manifest.json", manifest)
        for concurrency in args.concurrency:
            phase_dir = args.output / f"c{concurrency}"
            phase_dir.mkdir()
            warmup, _, _, _ = await closed_loop(
                client,
                url + "/v1/systemone",
                headers,
                items,
                requests,
                warmup_jobs(items, args.warmup),
                concurrency,
            )
            write_jsonl(phase_dir / "warmup.jsonl", warmup)
            failed_warmups = [row for row in warmup if row["status"] != "success"]
            if failed_warmups:
                report["status"] = "failed"
                report["concurrency"][f"c{concurrency}"] = {
                    "status": "warmup_failed",
                    "warmup_errors": len(failed_warmups),
                }
                write_json(args.output / "report.json", report)
                break
            outcomes, elapsed, started_ns, completed_ns = await closed_loop(
                client,
                url + "/v1/systemone",
                headers,
                items,
                requests,
                orders[str(concurrency)],
                concurrency,
            )
            write_jsonl(phase_dir / "raw_outcomes.jsonl", outcomes)
            phase = summarize(
                outcomes, elapsed, len(items), concurrency, args.min_measure_seconds
            )
            phase["concurrency"] = concurrency
            phase["measurement_started_ns"] = started_ns
            phase["measurement_completed_ns"] = completed_ns
            phase["order_sha256"] = canonical_sha256(
                [(row["ordinal"], row["index"], row["repeat"]) for row in outcomes]
            )
            phase["input_token_vector_sha256"] = canonical_sha256(
                [
                    (row["repeat"], row["index"], row.get("input_tokens"))
                    for row in outcomes
                ]
            )
            report["concurrency"][f"c{concurrency}"] = phase
            report["status"] = (
                "complete"
                if all(
                    row["status"] == "complete"
                    for row in report["concurrency"].values()
                )
                else "failed"
            )
            write_json(args.output / "report.json", report)
        manifest["status"] = report["status"]
        write_json(args.output / "manifest.json", manifest)
        return report
    except BaseException as error:
        manifest.update(
            status="failed", failure_type=type(error).__name__, failure=str(error)
        )
        write_json(args.output / "manifest.json", manifest)
        raise
    finally:
        if owned and client is not None:
            await client.aclose()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_saved(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if manifest.get("status") != "complete" or report.get("status") != "complete":
        short = [
            (key, phase.get("elapsed_s"), phase.get("minimum_measure_seconds"))
            for key, phase in report.get("concurrency", {}).items()
            if isinstance(phase, dict)
            and any(
                failure.get("error_type") == "MeasurementTooShort"
                for failure in phase.get("measurement_failures", [])
            )
        ]
        if short:
            raise ValueError(f"saved run duration below minimum: {short}")
        raise ValueError(f"saved run is incomplete: {directory}")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(manifest.get("settings"), dict)
        or not isinstance(manifest.get("engine"), dict)
        or not isinstance(manifest.get("client_code_sha256"), dict)
    ):
        raise ValueError(f"saved run lacks settings/runtime evidence: {directory}")
    concurrency = manifest.get("concurrency")
    if (
        not isinstance(concurrency, list)
        or not concurrency
        or any(type(c) is not int or not 1 <= c <= 128 for c in concurrency)
        or len(set(concurrency)) != len(concurrency)
    ):
        raise ValueError(f"saved concurrency is invalid: {directory}")
    maximum = max(concurrency)
    pool = manifest.get("http_pool_max_connections")
    if "http_pool_max_connections" in manifest and (
        type(pool) is not int or not maximum <= pool <= 128
    ):
        raise ValueError(f"saved HTTP pool capacity is invalid: {directory}")
    descriptor = manifest.get("http_client")
    if isinstance(descriptor, dict) and descriptor.get("kind") in ("httpx", "aiohttp"):
        actual = descriptor.get("max_connections")
        if "max_connections" in descriptor and (
            type(actual) is not int
            or not maximum <= actual <= 128
        ):
            raise ValueError(f"saved owned HTTP pool capacity is invalid: {directory}")
        if "http_pool_max_connections" in manifest and actual != pool:
            raise ValueError(f"saved HTTP pool and owned descriptor differ: {directory}")
    required_seconds = manifest.get("min_measure_seconds")
    if (
        type(required_seconds) not in (int, float)
        or not math.isfinite(required_seconds)
        or not 0 <= required_seconds <= 3600
    ):
        raise ValueError(f"saved run lacks valid minimum duration: {directory}")
    selected_path = directory / "selected_items.jsonl"
    if (
        not isinstance(manifest.get("selected_items_sha256"), str)
        or file_sha256(selected_path) != manifest["selected_items_sha256"]
    ):
        raise ValueError(f"selected item checksum differs: {directory}")
    selected_items = read_jsonl(selected_path)
    if len(selected_items) != manifest["selected_records"]:
        raise ValueError(f"selected item count differs: {directory}")
    expected_requests = [
        request_for(item, manifest["model"], None) for item in selected_items
    ]
    if canonical_sha256(expected_requests) != manifest["workload_sha256"]:
        raise ValueError(f"selected items differ from request workload: {directory}")
    settings = manifest["settings"]
    required = (
        "hardware",
        "model_path",
        "model_revision",
        "precision",
        "prompt_contract",
        "label_contract",
        "head_mode",
        "labels",
        "token_ids",
        "head_artifact",
    )
    if any(key not in settings for key in required):
        raise ValueError(f"saved run lacks head or contract configuration: {directory}")
    quantization, _ = precision_settings(settings)
    server = manifest["engine"].get("server_info")
    if (
        not isinstance(server, dict)
        or server.get("quantization") != quantization
        or server.get("dtype") != settings["precision"]
    ):
        raise ValueError(
            f"saved engine quantization/dtype differs from settings: {directory}"
        )
    if not isinstance(
        manifest.get("settings_manifest_sha256"), str
    ) or not HEX64.fullmatch(manifest["settings_manifest_sha256"]):
        raise ValueError(f"saved run lacks a settings manifest SHA256: {directory}")
    if not isinstance(manifest.get("label_array_sha256"), str) or not HEX64.fullmatch(
        manifest["label_array_sha256"]
    ):
        raise ValueError(f"saved run lacks a label array SHA256: {directory}")
    if (
        canonical_sha256(
            {"labels": settings["labels"], "token_ids": settings["token_ids"]}
        )
        != manifest["label_array_sha256"]
    ):
        raise ValueError(f"saved run label array checksum differs: {directory}")
    audit = manifest.get("token_id_audit")
    if audit is not None and (
        not isinstance(audit, dict)
        or not isinstance(audit.get("file_sha256"), str)
        or not HEX64.fullmatch(audit["file_sha256"])
        or not isinstance(audit.get("input_token_ids_sha256"), list)
        or len(audit["input_token_ids_sha256"]) != manifest["selected_records"]
        or any(
            not isinstance(value, str) or not HEX64.fullmatch(value)
            for value in audit["input_token_ids_sha256"]
        )
    ):
        raise ValueError(f"saved token ID audit is malformed: {directory}")
    repeats = repeat_counts(
        manifest["concurrency"],
        manifest["repeats"],
        manifest.get("repeats_by_concurrency"),
    )
    if len(selected_items) * sum(repeats.values()) > 1_000_000:
        raise ValueError("saved run exceeds the 1000000-request bound")
    order_hashes = manifest_order_hashes(manifest)
    if not isinstance(order_hashes, dict) or set(order_hashes) != set(repeats):
        raise ValueError("saved per-concurrency job orders are missing or malformed")
    saved = {}
    for concurrency in manifest["concurrency"]:
        key = f"c{concurrency}"
        summary = report["concurrency"].get(key)
        if not isinstance(summary, dict) or summary.get("status") != "complete":
            raise ValueError(f"missing complete {key} summary: {directory}")
        rows = read_jsonl(directory / key / "raw_outcomes.jsonl")
        identities = [(row["repeat"], row["index"]) for row in rows]
        if (
            len(rows) != summary["attempted_requests"]
            or len(rows) != len(selected_items) * repeats[str(concurrency)]
            or len(identities) != len(set(identities))
            or any(row.get("status") != "success" for row in rows)
        ):
            raise ValueError(
                f"missing, duplicate, or failed {key} outcomes: {directory}"
            )
        if (
            canonical_sha256(
                [(row["ordinal"], row["index"], row["repeat"]) for row in rows]
            )
            != summary["order_sha256"]
        ):
            raise ValueError(f"{key} saved job order checksum differs: {directory}")
        if (
            canonical_sha256(
                [(row["repeat"], row["index"], row["input_tokens"]) for row in rows]
            )
            != summary["input_token_vector_sha256"]
        ):
            raise ValueError(f"{key} input-token checksum differs: {directory}")
        peak = http_inflight_peak(rows)
        if peak < concurrency or summary.get("client_http_inflight_peak") != peak:
            raise ValueError(f"{key} client HTTP concurrency was below C: {directory}")
        expected_order = canonical_sha256(
            jobs_for(selected_items, repeats[str(concurrency)], manifest["seed"])
        )
        if (
            summary["order_sha256"] != order_hashes[str(concurrency)]
            or summary["order_sha256"] != expected_order
        ):
            raise ValueError(f"{key} order differs from run manifest: {directory}")
        started_ns = summary.get("measurement_started_ns")
        completed_ns = summary.get("measurement_completed_ns")
        if (
            type(started_ns) is not int
            or type(completed_ns) is not int
            or completed_ns <= started_ns
            or summary["elapsed_s"] != (completed_ns - started_ns) / 1e9
        ):
            raise ValueError(f"{key} measurement timing is invalid: {directory}")
        if (
            summary.get("minimum_measure_seconds") != required_seconds
            or summary.get("duration_requirement_met") is not True
            or summary["elapsed_s"] < required_seconds
            or summary.get("measurement_failures")
        ):
            raise ValueError(f"{key} measurement duration gate is invalid: {directory}")
        for row in rows:
            body = json.loads(row["raw_response_text"])
            json.dumps(body, allow_nan=False)
            checked = validate_success(body, row["request"], manifest["model"])
            checked_usage(body)
            capacity = body["metadata"].get("service_max_concurrency")
            if (
                type(row.get("index")) is not int
                or not 0 <= row["index"] < len(selected_items)
                or row["request"] != expected_requests[row["index"]]
                or row["request_sha256"] != canonical_sha256(row["request"])
                or row["cached_tokens"] != 0
                or row["input_tokens"] != checked["input_tokens"]
                or row["decisions"] != checked["evaluations"]
                or type(capacity) is not int
                or capacity < concurrency
                or row.get("service_max_concurrency") != capacity
                or type(row.get("admitted_ns")) is not int
                or type(row.get("completed_ns")) is not int
                or not started_ns
                <= row["admitted_ns"]
                < row["completed_ns"]
                <= completed_ns
                or not row["admitted_ns"]
                <= row["http_post_started_ns"]
                < row["http_post_completed_ns"]
                <= row["completed_ns"]
                or row["client_latency_ms"]
                != (row["completed_ns"] - row["admitted_ns"]) / 1e6
                or row["rows"] != prediction_rows(selected_items[row["index"]], body)
            ):
                raise ValueError(
                    f"{key} raw outcome contradicts saved metrics: {directory}"
                )
        capacities = {row["service_max_concurrency"] for row in rows}
        if (
            len(capacities) != 1
            or summary.get("service_max_concurrency") not in capacities
        ):
            raise ValueError(f"{key} service capacity differs across responses")
        computed_quality = quality_summary(first_pass_rows(rows, len(selected_items)))
        if (
            summary["client_latency_ms"]
            != latency_summary([row["client_latency_ms"] for row in rows])
            or summary["successful_requests"] != len(rows)
            or summary["input_tokens"] != sum(row["input_tokens"] for row in rows)
            or summary["decisions"] != sum(row["decisions"] for row in rows)
            or summary["requests_per_second"] != len(rows) / summary["elapsed_s"]
            or summary["decisions_per_second"]
            != summary["decisions"] / summary["elapsed_s"]
            or summary["input_tokens_per_second"]
            != summary["input_tokens"] / summary["elapsed_s"]
            or any(
                summary.get(name) != expected
                for name, expected in computed_quality.items()
            )
        ):
            raise ValueError(
                f"{key} aggregate metrics differ from raw outcomes: {directory}"
            )
        saved[key] = rows
    return manifest, report, saved


def canonical_graph_config(config):
    """Ignore only unset graph fields added after the pinned stock release.

    Keep manifests untouched, including unknown fields and non-null limits.
    PhaseConfig exposes these optional fields on both decode and prefill.
    """
    if not isinstance(config, dict):
        return config
    result = dict(config)
    for phase in ("decode", "prefill"):
        settings = config.get(phase)
        if isinstance(settings, dict):
            result[phase] = {
                key: value
                for key, value in settings.items()
                if key not in ("max_context_size", "max_seq_len") or value is not None
            }
    return result


def _same_workload(left, right):
    keys = (
        "model",
        "selected_records",
        "selected_items_sha256",
        "selected_decisions",
        "max_options_in_workload",
        "workload_sha256",
        "concurrency",
        "warmup_requests_per_c",
        "min_measure_seconds",
        "seed",
        "limit",
        "request_options",
        "client_code_sha256",
        "http_client",
        "http_pool_max_connections",
    )
    if any(left.get(key) != right.get(key) for key in keys):
        raise ValueError("workload, concurrency, seed/order, or client code differs")
    if manifest_repeat_counts(left) != manifest_repeat_counts(right):
        raise ValueError("repeats-by-concurrency differs")
    if manifest_order_hashes(left) != manifest_order_hashes(right):
        raise ValueError("per-concurrency job order differs")
    if [entry["sha256"] for entry in left["data"]] != [
        entry["sha256"] for entry in right["data"]
    ]:
        raise ValueError("corpus hashes differ")
    a, b = left["settings"], right["settings"]
    if precision_settings(a) != precision_settings(b):
        raise ValueError(
            "quantization/precision profiles differ; BF16-vs-FP8 quality needs a separate comparison"
        )
    for key in (
        "hardware",
        "model_path",
        "model_revision",
        "precision",
        "prompt_contract",
        "label_contract",
    ):
        if a[key] != b[key]:
            raise ValueError(f"settings {key} differs")
    needed = left["max_options_in_workload"]
    if (
        a["labels"][:needed] != b["labels"][:needed]
        or a["token_ids"][:needed] != b["token_ids"][:needed]
    ):
        raise ValueError("label/token arrays differ for used options")
    left_engine, right_engine = left["engine"], right["engine"]
    if left_engine["model_info"] != right_engine["model_info"]:
        raise ValueError("engine model_info differs")
    for key in RESOURCE_FIELDS:
        left_value = left_engine["server_info"].get(key)
        right_value = right_engine["server_info"].get(key)
        if key == "cuda_graph_config":
            left_value = canonical_graph_config(left_value)
            right_value = canonical_graph_config(right_value)
        if left_value != right_value:
            raise ValueError(f"engine runtime setting {key} differs")


def displayed_answer_parity(baseline_rows, candidate_rows, tolerance):
    differences = []
    checked_answers = 0
    for old, new in zip(baseline_rows, candidate_rows):
        a = json.loads(old["raw_response_text"])["answers"]
        b = json.loads(new["raw_response_text"])["answers"]
        for question in a:
            checked_answers += 1
            left, right = a[question], b[question]
            if left["type"] != right["type"]:
                differences.append((old["repeat"], old["index"], question, "type"))
                continue
            fields = (
                ("noul",)
                if left["type"] == "noul"
                else ("choice", "confidence")
                if left["type"] == "choice"
                else ("score", "confidence")
            )
            for field in fields:
                x, y = left.get(field), right.get(field)
                if (
                    type(x) in (int, float)
                    and type(y) in (int, float)
                    and math.isfinite(x)
                    and math.isfinite(y)
                ):
                    different = abs(x - y) > tolerance
                else:
                    different = x != y
                if different:
                    differences.append((old["repeat"], old["index"], question, field))
            if left["type"] != "noul" and any(
                abs(value - right["probabilities"][label]) > tolerance
                for label, value in left["probabilities"].items()
            ):
                differences.append(
                    (old["repeat"], old["index"], question, "probabilities")
                )
    return {
        "status": "matched" if not differences else "mismatch",
        "checked_answers": checked_answers,
        "mismatched_fields": len(differences),
        "samples": differences[:20],
    }


def compare_saved(baseline_dir, candidate_dir, tolerance=0.01):
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("probability tolerance must be between zero and one")
    baseline_manifest, baseline_report, baseline_rows = load_saved(baseline_dir)
    candidate_manifest, candidate_report, candidate_rows = load_saved(candidate_dir)
    _same_workload(baseline_manifest, candidate_manifest)
    modes = [
        baseline_manifest["settings"]["head_mode"],
        candidate_manifest["settings"]["head_mode"],
    ]
    learned = "learned" in modes
    old_audit = baseline_manifest.get("token_id_audit")
    new_audit = candidate_manifest.get("token_id_audit")
    if old_audit is not None and new_audit is not None:
        if old_audit.get("input_token_ids_sha256") != new_audit.get(
            "input_token_ids_sha256"
        ):
            raise ValueError("audited input token ID hashes differ")
        audit_status = "matched_external_audits"
    else:
        audit_status = "missing_external_audit"
    per_c = {}
    eligible = audit_status == "matched_external_audits"
    faster_everywhere = True
    for concurrency in baseline_manifest["concurrency"]:
        key = f"c{concurrency}"
        old_rows, new_rows = baseline_rows[key], candidate_rows[key]
        old_map = {(row["repeat"], row["index"]): row for row in old_rows}
        new_map = {(row["repeat"], row["index"]): row for row in new_rows}
        if old_map.keys() != new_map.keys():
            raise ValueError(f"{key} measured request keys differ")
        for identity, a in old_map.items():
            b = new_map[identity]
            if (
                a["request_sha256"] != b["request_sha256"]
                or a["id"] != b["id"]
                or a["suite"] != b["suite"]
                or a["split"] != b["split"]
                or a["decisions"] != b["decisions"]
            ):
                raise ValueError(f"{key} measured requests differ")
        token_match = compare_input_tokens(old_rows, new_rows, len(old_rows))
        if token_match["status"] != "matched":
            raise ValueError(f"{key} per-request input token counts differ")
        old = baseline_report["concurrency"][key]
        new = candidate_report["concurrency"][key]
        old_latency, new_latency = old["client_latency_ms"], new["client_latency_ms"]
        old_rate, new_rate = old["decisions_per_second"], new["decisions_per_second"]
        if not old_latency["p50"] or not old_rate:
            raise ValueError(f"{key} baseline has invalid latency or throughput")
        p50_ratio = new_latency["p50"] / old_latency["p50"]
        throughput_ratio = new_rate / old_rate
        faster_everywhere &= p50_ratio < 1 and throughput_ratio > 1
        phase = {
            "baseline": {
                "elapsed_s": old["elapsed_s"],
                "client_latency_ms": old_latency,
                "client_http_inflight_peak": old["client_http_inflight_peak"],
                "service_max_concurrency": old["service_max_concurrency"],
                "requests_per_second": old["requests_per_second"],
                "decisions_per_second": old_rate,
                "input_tokens_per_second": old["input_tokens_per_second"],
            },
            "candidate": {
                "elapsed_s": new["elapsed_s"],
                "client_latency_ms": new_latency,
                "client_http_inflight_peak": new["client_http_inflight_peak"],
                "service_max_concurrency": new["service_max_concurrency"],
                "requests_per_second": new["requests_per_second"],
                "decisions_per_second": new_rate,
                "input_tokens_per_second": new["input_tokens_per_second"],
            },
            "quality": {
                "sample": "first measured pass; clean, knowable, non-synthetic questions",
                "baseline": {
                    "overall": old["quality_overall"],
                    "by_type": old["quality_by_type"],
                    "by_suite": old["quality_by_suite"],
                },
                "candidate": {
                    "overall": new["quality_overall"],
                    "by_type": new["quality_by_type"],
                    "by_suite": new["quality_by_suite"],
                },
            },
            "p50_latency_ratio": p50_ratio,
            "decisions_per_second_ratio": throughput_ratio,
            "input_token_parity": token_match,
        }
        if learned:
            phase["quality"]["comparison"] = "separate_trained_head"
            eligible = False
        else:
            quality = compare_quality(
                old_rows,
                new_rows,
                baseline_manifest["selected_records"],
                tolerance,
            )
            phase["quality_parity"] = quality
            eligible &= quality.get("parity_passed") is True
            displayed = displayed_answer_parity(old_rows, new_rows, tolerance)
            phase["displayed_answer_parity"] = displayed
            eligible &= displayed["status"] == "matched"
        per_c[key] = phase
    return {
        "status": "comparable" if eligible else "no_winner_gate",
        "comparison_scope": "single_run_pair",
        "repeated_trials_required_for_final_claim": True,
        "final_claim_eligible": False,
        "baseline": str(baseline_dir.resolve()),
        "candidate": str(candidate_dir.resolve()),
        "concurrency": baseline_manifest["concurrency"],
        "min_measure_seconds": baseline_manifest["min_measure_seconds"],
        "head_mode_array": modes,
        "quantization_array": [
            precision_settings(manifest["settings"])[0]
            for manifest in (baseline_manifest, candidate_manifest)
        ],
        "precision_profile_array": [
            precision_settings(manifest["settings"])[1]
            for manifest in (baseline_manifest, candidate_manifest)
        ],
        "head_artifact_sha256_array": [
            manifest["settings"]["head_artifact"]["sha256"]
            if manifest["settings"]["head_artifact"]
            else None
            for manifest in (baseline_manifest, candidate_manifest)
        ],
        "settings_manifest_sha256_array": [
            baseline_manifest["settings_manifest_sha256"],
            candidate_manifest["settings_manifest_sha256"],
        ],
        "token_id_audit_status": audit_status,
        "token_id_audit_basis": "matching per-request hashes from external tokenization audits; endpoint HTTP responses do not expose token IDs",
        "quality_parity_required": not learned,
        "quality_winner": None,
        "quality_scope": "descriptive first-pass development metrics; no quality winner from this single run",
        "single_pair_winner": "candidate" if eligible and faster_everywhere else None,
        "winner_rule": "single-pair lower p50 and higher decisions/s at every same-C point, with matching inputs and quality; requires three independent >=30s pairs for a final claim",
        "per_concurrency": per_c,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="measure one already-running endpoint")
    runner.add_argument("--base-url", required=True)
    runner.add_argument("--engine-url")
    runner.add_argument("--model", required=True)
    runner.add_argument("--settings-manifest", type=Path, required=True)
    runner.add_argument("--token-audit", type=Path)
    runner.add_argument("--data", type=Path, action="append", required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--concurrency", type=int, action="append")
    runner.add_argument("--warmup", type=int, default=128)
    runner.add_argument("--repeats", type=int, default=1)
    runner.add_argument(
        "--repeats-by-concurrency",
        help='JSON counts for every selected C, e.g. {"8":10,"16":14}',
    )
    runner.add_argument("--min-measure-seconds", type=float, default=0.0)
    runner.add_argument("--limit", type=int, default=512)
    runner.add_argument("--seed", type=int, default=42)
    runner.add_argument("--timeout", type=float, default=120)
    runner.add_argument("--http-client", choices=("httpx", "aiohttp"), default="httpx")
    runner.add_argument(
        "--http-max-connections",
        type=int,
        help="connection pool size from max(C) to 128; defaults to max(C)",
    )
    runner.add_argument("--key-env", default="SGLANG_API_KEY")
    runner.add_argument("--allow-test", action="store_true")
    comparer = commands.add_parser("compare", help="compare two saved run directories")
    comparer.add_argument("baseline", type=Path)
    comparer.add_argument("candidate", type=Path)
    comparer.add_argument("--output", type=Path, required=True)
    comparer.add_argument("--probability-tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.command == "run":
        args.concurrency = args.concurrency or list(DEFAULT_CONCURRENCY)
        try:
            report = asyncio.run(run(args))
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps({"output": str(args.output), "status": report["status"]}))
        return 0 if report["status"] == "complete" else 1
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    try:
        result = compare_saved(
            args.baseline, args.candidate, args.probability_tolerance
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "single_pair_winner": result["single_pair_winner"],
            }
        )
    )
    return 0 if result["status"] == "comparable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
