"""Compare two or three System One services on identical prefill-only decisions.

Every endpoint receives the same prepared records and shuffled request order.
Warmups are excluded. Measured phases run sequentially, so the services
should be isolated from other GPU work when interpreting latency/throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from jev_adapter.benchmarks.metrics import (
    latency_summary,
    metrics,
    predicted_index,
    prediction_rows,
    quantile,
)
from jev_adapter.benchmarks.run import checked_usage, load_data, write_json

DEFAULT_CONCURRENCY = (1, 8, 32)
BACKENDS = ("sglang", "candidate")


def enabled_backends(args):
    return BACKENDS + (
        ("learned",) if getattr(args, "learned_url", None) is not None else ()
    )


def validate_args(args):
    if (
        not args.concurrency
        or len(set(args.concurrency)) != len(args.concurrency)
        or any(type(c) is not int or c < 1 for c in args.concurrency)
        or type(args.warmup) is not int
        or args.warmup < 0
        or type(args.repeats) is not int
        or args.repeats < 1
        or (args.limit is not None and (type(args.limit) is not int or args.limit < 1))
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.probability_tolerance)
        or not 0 <= args.probability_tolerance <= 1
        or (
            getattr(args, "max_options", None) is not None
            and (type(args.max_options) is not int or args.max_options < 2)
        )
    ):
        raise ValueError("invalid counts, timeout, or probability tolerance")
    for name in enabled_backends(args):
        url = getattr(args, f"{name}_url")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"{name} URL must be an HTTP(S) base URL without credentials or query"
            )


def filter_items(items, max_options, limit):
    if max_options is None:
        return items, 0
    selected = [
        item
        for item in items
        if all(
            (2 if question["type"] == "noul" else len(question["criteria"]))
            <= max_options
            for question in item["record"]["questions"].values()
        )
    ]
    excluded = len(items) - len(selected)
    selected = selected[:limit] if limit is not None else selected
    if not selected:
        raise ValueError("option filter selected no records")
    return selected, excluded


def request_for(item, model, assistant_prefix):
    request = {
        **item["record"],
        "model": model,
        "options": {
            "temperature": 1.0,
            "permutations": 1,
            "return_logprobs": False,
        },
    }
    if assistant_prefix is not None:
        request["options"]["assistant_prefix"] = assistant_prefix
    return request


def auth_headers(key_env):
    key = os.environ.get(key_env) or os.environ.get("JEV_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


async def request_once(client, semaphore, url, headers, item, request):
    async with semaphore:
        started = time.perf_counter()
        response = await client.post(
            url.rstrip("/") + "/v1/systemone", headers=headers, json=request
        )
        latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    body = response.json()
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("usage"), dict)
        or not isinstance(body.get("metadata"), dict)
    ):
        raise TypeError("response, usage, and metadata must be JSON objects")
    usage, metadata = checked_usage(body)
    if metadata["evaluations"] != len(item["record"]["questions"]):
        raise ValueError("expected exactly one forward evaluation per question")
    cached_tokens = metadata.get("cached_tokens")
    if (
        type(cached_tokens) is not int
        or not 0 <= cached_tokens <= usage["input_tokens"]
    ):
        raise ValueError("response must report valid cached_tokens")
    return {
        "latency_ms": latency_ms,
        "input_tokens": usage["input_tokens"],
        "cached_tokens": cached_tokens,
        "decisions": metadata["evaluations"],
        "served_model": body["model"],
        "rows": prediction_rows(item, body),
    }


def error_record(index, repeat, item, error):
    record = {
        "index": index,
        "repeat": repeat,
        "suite": item["suite"],
        "id": item["id"],
        "error_type": type(error).__name__,
    }
    if isinstance(error, httpx.HTTPStatusError):
        record["status_code"] = error.response.status_code
    elif isinstance(error, (ValueError, KeyError, TypeError)):
        record["detail"] = str(error)
    return record


async def dispatch(client, url, headers, items, requests, jobs, concurrency):
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index, repeat):
        try:
            result = await request_once(
                client, semaphore, url, headers, items[index], requests[index]
            )
            return {"index": index, "repeat": repeat, **result}
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            return error_record(index, repeat, items[index], error)

    started = time.perf_counter()
    outcomes = await asyncio.gather(*(one(index, repeat) for index, repeat in jobs))
    return outcomes, time.perf_counter() - started


def backend_summary(outcomes, elapsed, attempted):
    successes = [row for row in outcomes if "error_type" not in row]
    errors = [row for row in outcomes if "error_type" in row]
    served = sorted({row["served_model"] for row in successes})
    if len(served) > 1:
        errors.append({"error_type": "ModelChanged", "detail": str(served)})
    cache_hits = [row for row in successes if row["cached_tokens"] > 0]
    if cache_hits:
        errors.append(
            {
                "error_type": "CacheHit",
                "detail": f"{len(cache_hits)} measured requests reused prefix tokens",
            }
        )
    decisions = sum(row["decisions"] for row in successes)
    tokens = sum(row["input_tokens"] for row in successes)
    cached_tokens = sum(row["cached_tokens"] for row in successes)
    return {
        "status": "complete" if not errors else "failed",
        "attempted_requests": attempted,
        "successful_requests": len(successes),
        "errors": len(errors),
        "error_samples": errors[:20],
        "served_models": served,
        "elapsed_s": elapsed,
        "request_latency_ms": latency_summary([row["latency_ms"] for row in successes]),
        "decisions": decisions,
        "input_tokens": tokens,
        "cached_tokens": cached_tokens,
        "requests_with_cache_hits": len(cache_hits),
        "decisions_per_second": decisions / elapsed if elapsed > 0 else None,
        "input_tokens_per_second": tokens / elapsed if elapsed > 0 else None,
    }


def quality_by_suite(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["variant"] == "clean" and row["source"] != "unknowable":
            grouped[f"{row['suite']}/{row['split']}"].append(row)
    return {
        name: {
            "questions": score["n"],
            "accuracy": score["acc"],
            "brier": score["brier"],
            "nll": score["nll"],
        }
        for name, sample in sorted(grouped.items())
        if (score := metrics(sample)) is not None
    }


def first_pass_rows(outcomes, item_count):
    selected = {
        row["index"]: row
        for row in outcomes
        if row.get("repeat") == 0 and "error_type" not in row
    }
    if len(selected) != item_count:
        return None
    return [row for index in range(item_count) for row in selected[index]["rows"]]


def compare_input_tokens(baseline_outcomes, other_outcomes, attempted):
    """Check each measured request, not just the aggregate token total."""
    baseline = {
        (row["repeat"], row["index"]): row["input_tokens"]
        for row in baseline_outcomes
        if "error_type" not in row
    }
    other = {
        (row["repeat"], row["index"]): row["input_tokens"]
        for row in other_outcomes
        if "error_type" not in row
    }
    if (
        len(baseline) != attempted
        or len(other) != attempted
        or baseline.keys() != other.keys()
    ):
        return {
            "status": "incomplete",
            "reason": "missing or duplicate measured responses",
            "baseline_responses": len(baseline),
            "other_responses": len(other),
            "attempted_requests": attempted,
        }
    mismatches = [
        {
            "repeat": repeat,
            "index": index,
            "sglang_input_tokens": baseline[(repeat, index)],
            "other_input_tokens": other[(repeat, index)],
        }
        for repeat, index in sorted(baseline)
        if baseline[(repeat, index)] != other[(repeat, index)]
    ]
    return {
        "status": "matched" if not mismatches else "mismatch",
        "matched_requests": attempted - len(mismatches),
        "mismatched_requests": len(mismatches),
        "mismatch_samples": mismatches[:20],
    }


def compare_quality(sglang_outcomes, candidate_outcomes, item_count, tolerance):
    all_rows = {
        "sglang": first_pass_rows(sglang_outcomes, item_count),
        "candidate": first_pass_rows(candidate_outcomes, item_count),
    }
    if any(rows is None for rows in all_rows.values()):
        return {"status": "incomplete", "reason": "missing first-pass responses"}
    left, right = all_rows["sglang"], all_rows["candidate"]
    if len(left) != len(right):
        raise ValueError("backend decision counts differ")
    max_deltas = []
    agreement = 0
    for a, b in zip(left, right):
        identity = ("suite", "split", "id", "question", "keys")
        if any(a[key] != b[key] for key in identity):
            raise ValueError("backend decision identities differ")
        agreement += predicted_index(a) == predicted_index(b)
        max_deltas.append(max(abs(x - y) for x, y in zip(a["p"], b["p"])))
    within = sum(delta <= tolerance for delta in max_deltas)
    n = len(left)
    return {
        "status": "complete",
        "questions": n,
        "decision_agreement": agreement / n if n else None,
        "decision_disagreements": n - agreement,
        "probability_tolerance": tolerance,
        "questions_within_tolerance": within,
        "probability_abs_delta": {
            "mean_of_question_max": math.fsum(max_deltas) / n if n else None,
            "p95_of_question_max": quantile(max_deltas, 0.95),
            "max": max(max_deltas, default=None),
        },
        "parity_passed": agreement == n and within == n,
        "quality_by_suite": {
            name: quality_by_suite(rows) for name, rows in all_rows.items()
        },
    }


def compare_learned_quality(sglang_outcomes, learned_outcomes, item_count):
    baseline_rows = first_pass_rows(sglang_outcomes, item_count)
    learned_rows = first_pass_rows(learned_outcomes, item_count)
    if baseline_rows is None or learned_rows is None:
        return {"status": "incomplete", "reason": "missing first-pass responses"}
    baseline = quality_by_suite(baseline_rows)
    learned = quality_by_suite(learned_rows)
    if set(baseline) != set(learned):
        raise ValueError("learned and baseline quality suites differ")
    return {
        "status": "complete",
        "quality_by_suite": {"sglang": baseline, "learned": learned},
        "learned_minus_sglang_by_suite": {
            suite: {
                metric: learned[suite][metric] - baseline[suite][metric]
                for metric in ("accuracy", "brier", "nll")
            }
            for suite in baseline
        },
    }


async def compare(args, client=None):
    validate_args(args)
    backends = enabled_backends(args)
    max_options = getattr(args, "max_options", None)
    items, sources = load_data(
        args.data,
        args.model,
        args.allow_test,
        None if max_options is not None else args.limit,
    )
    items, excluded = filter_items(items, max_options, args.limit)
    requests = [request_for(item, args.model, args.assistant_prefix) for item in items]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "started",
        "data": sources,
        "selected_records": len(items),
        "selected_decisions": sum(len(item["expected"]) for item in items),
        "excluded_records_by_max_options": excluded,
        "max_options": max_options,
        "limit": args.limit,
        "model": args.model,
        "urls": {name: getattr(args, f"{name}_url") for name in backends},
        "concurrency": args.concurrency,
        "warmup_requests_per_backend_per_concurrency": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "probability_tolerance": args.probability_tolerance,
        "request_options": requests[0]["options"],
        "latency_scope": "successful client HTTP round trip, excluding semaphore wait",
        "throughput_scope": "completed decisions and reported logical input tokens / measured wall time",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(args.output / "manifest.json", manifest)
    owned = client is None
    if owned:
        max_connections = max(args.concurrency)
        client = httpx.AsyncClient(
            timeout=args.timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        )
    report = {"status": "started", "concurrency": {}}
    try:
        for concurrency in args.concurrency:
            order = list(range(len(items)))
            random.Random(args.seed).shuffle(order)
            warmup_indices = [
                (i * len(items) // args.warmup) % len(items) for i in range(args.warmup)
            ]
            measured_jobs = [
                (index, repeat) for repeat in range(args.repeats) for index in order
            ]
            phase = {"backends": {}}
            # Alternate first service to reduce fixed-order bias across levels.
            names = (
                backends
                if args.concurrency.index(concurrency) % 2 == 0
                else backends[::-1]
            )
            for name in names:
                url = getattr(args, f"{name}_url")
                headers = auth_headers(getattr(args, f"{name}_key_env"))
                warmup, _ = await dispatch(
                    client,
                    url,
                    headers,
                    items,
                    requests,
                    [(index, -1) for index in warmup_indices],
                    concurrency,
                )
                failures = [row for row in warmup if "error_type" in row]
                if failures:
                    raise RuntimeError(
                        f"{name} c{concurrency} warmup failed: {failures[:3]}"
                    )
                outcomes, elapsed = await dispatch(
                    client,
                    url,
                    headers,
                    items,
                    requests,
                    measured_jobs,
                    concurrency,
                )
                phase["backends"][name] = backend_summary(
                    outcomes, elapsed, len(measured_jobs)
                )
                phase[name + "_outcomes"] = outcomes
                with (args.output / f"c{concurrency}-{name}.jsonl").open("x") as handle:
                    for row in sorted(
                        outcomes, key=lambda r: (r["repeat"], r["index"])
                    ):
                        handle.write(
                            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                        )
            phase["input_token_parity"] = {
                name: compare_input_tokens(
                    phase["sglang_outcomes"],
                    phase[name + "_outcomes"],
                    len(measured_jobs),
                )
                for name in backends
                if name != "sglang"
            }
            candidate_workload = phase["input_token_parity"]["candidate"]["status"]
            if candidate_workload == "mismatch":
                phase["quality_parity"] = {
                    "status": "incomparable",
                    "reason": "per-request input token counts differ",
                }
            elif candidate_workload == "matched" and all(
                phase["backends"][name]["status"] == "complete" for name in BACKENDS
            ):
                phase["quality_parity"] = compare_quality(
                    phase["sglang_outcomes"],
                    phase["candidate_outcomes"],
                    len(items),
                    args.probability_tolerance,
                )
            else:
                phase["quality_parity"] = {
                    "status": "incomplete",
                    "reason": "one or both backends had errors",
                }
            old = phase["backends"]["sglang"]
            new = phase["backends"]["candidate"]
            old_p50 = old["request_latency_ms"]["p50"]
            new_p50 = new["request_latency_ms"]["p50"]
            old_rate = old["decisions_per_second"]
            new_rate = new["decisions_per_second"]
            candidate_comparable = (
                candidate_workload == "matched"
                and old["status"] == new["status"] == "complete"
            )
            phase["candidate_vs_sglang"] = {
                "p50_latency_ratio": (
                    new_p50 / old_p50 if candidate_comparable and old_p50 else None
                ),
                "decisions_per_second_ratio": (
                    new_rate / old_rate if candidate_comparable and old_rate else None
                ),
            }
            if "learned" in backends:
                learned = phase["backends"]["learned"]
                learned_workload = phase["input_token_parity"]["learned"]["status"]
                if learned_workload == "mismatch":
                    phase["learned_vs_sglang"] = {
                        "status": "incomparable",
                        "reason": "per-request input token counts differ",
                    }
                elif learned_workload == "matched" and old["status"] == learned["status"] == "complete":
                    phase["learned_vs_sglang"] = compare_learned_quality(
                        phase["sglang_outcomes"],
                        phase["learned_outcomes"],
                        len(items),
                    )
                else:
                    phase["learned_vs_sglang"] = {
                        "status": "incomplete",
                        "reason": "baseline or learned backend had errors",
                    }
                learned_p50 = learned["request_latency_ms"]["p50"]
                learned_rate = learned["decisions_per_second"]
                learned_comparable = (
                    learned_workload == "matched"
                    and old["status"] == learned["status"] == "complete"
                )
                phase["learned_vs_sglang"].update(
                    p50_latency_ratio=(
                        learned_p50 / old_p50 if learned_comparable and old_p50 else None
                    ),
                    decisions_per_second_ratio=(
                        learned_rate / old_rate if learned_comparable and old_rate else None
                    ),
                )
            for name in backends:
                del phase[name + "_outcomes"]
            report["concurrency"][f"c{concurrency}"] = phase
            report["status"] = (
                "complete"
                if all(
                    all(
                        group["backends"][name]["status"] == "complete"
                        for name in backends
                    )
                    and all(
                        parity["status"] == "matched"
                        for parity in group["input_token_parity"].values()
                    )
                    for group in report["concurrency"].values()
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
        if owned:
            await client.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--learned-url")
    parser.add_argument("--sglang-key-env", default="JEV_SGLANG_API_KEY")
    parser.add_argument("--candidate-key-env", default="JEV_CANDIDATE_API_KEY")
    parser.add_argument("--learned-key-env", default="JEV_LEARNED_API_KEY")
    parser.add_argument("--model", default="decision-model")
    parser.add_argument("--concurrency", type=int, action="append")
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-options", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--probability-tolerance", type=float, default=0.01)
    parser.add_argument("--assistant-prefix")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--require-parity", action="store_true")
    args = parser.parse_args()
    args.concurrency = args.concurrency or list(DEFAULT_CONCURRENCY)
    try:
        report = asyncio.run(compare(args))
    except ValueError as error:
        parser.error(str(error))
    for name, phase in report["concurrency"].items():
        for backend in enabled_backends(args):
            row = phase["backends"][backend]
            latency = row["request_latency_ms"]
            print(
                f"{name} {backend}: p50={latency['p50']}ms p95={latency['p95']}ms "
                f"decisions/s={row['decisions_per_second']} "
                f"input tokens/s={row['input_tokens_per_second']} errors={row['errors']}"
            )
        print(f"{name} parity: {phase['quality_parity']}")
        if "learned_vs_sglang" in phase:
            print(f"{name} learned quality: {phase['learned_vs_sglang']}")
    failed = report["status"] != "complete" or (
        args.require_parity
        and any(
            not phase["quality_parity"].get("parity_passed", False)
            for phase in report["concurrency"].values()
        )
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
