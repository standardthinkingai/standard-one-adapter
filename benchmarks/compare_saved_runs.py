"""Check workload and decision parity for two sequential single-GPU runs.

Use this after separate Triton/FlashInfer run_matrix.py runs. Both directories
must be cN result directories from jev_adapter.benchmarks.run. This is an
offline correctness gate; latency remains the original client HTTP measure.
"""

import argparse
import json
import math
from pathlib import Path

if __package__:
    from .prefill_compare import compare_input_tokens, compare_quality
else:
    from prefill_compare import compare_input_tokens, compare_quality

from jev_adapter.benchmarks.run import write_json


def load_run(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if manifest.get("status") != "complete" or report.get("status") != "complete":
        raise ValueError(f"incomplete benchmark run: {directory}")
    rows = [
        json.loads(line)
        for line in (directory / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    keys = [(row["repeat"], row["index"]) for row in rows]
    if len(rows) != report["attempted_requests"] or len(set(keys)) != len(keys):
        raise ValueError(f"missing or duplicate predictions: {directory}")
    return manifest, report, dict(zip(keys, rows))


def compare(left_dir, right_dir, tolerance):
    left_manifest, left_report, left = load_run(left_dir)
    right_manifest, right_report, right = load_run(right_dir)
    settings = (
        "limit",
        "concurrency",
        "warmup_requests",
        "repeats",
        "seed",
        "cache_mode",
        "options",
        "assistant_prefix",
        "code_sha256",
    )
    if any(left_manifest.get(key) != right_manifest.get(key) for key in settings):
        raise ValueError("benchmark request or code settings differ")
    if [x["sha256"] for x in left_manifest["data"]] != [
        x["sha256"] for x in right_manifest["data"]
    ]:
        raise ValueError("input dataset checksums differ")
    left_launch = left_manifest.get("launch_manifest", {})
    right_launch = right_manifest.get("launch_manifest", {})
    if not left_launch or not right_launch:
        raise ValueError("both runs need saved engine launch manifests")
    if any(
        left_launch.get(key) != right_launch.get(key)
        for key in ("profile", "model", "engine", "environment_overrides")
    ):
        raise ValueError("engine/model profile or environment differs")
    if left_report["requested_records"] != right_report["requested_records"]:
        raise ValueError("requested record counts differ")
    if left.keys() != right.keys():
        raise ValueError("measured request indexes differ")
    identity = ("request_sha256", "id", "suite", "split", "evaluations", "served_model")
    request_mismatches = [
        {"repeat": repeat, "index": index}
        for repeat, index in sorted(left)
        if any(left[repeat, index][field] != right[repeat, index][field] for field in identity)
    ]
    if request_mismatches:
        return {
            "status": "incomparable",
            "reason": "measured requests differ",
            "mismatched_requests": len(request_mismatches),
            "mismatch_samples": request_mismatches[:20],
        }
    token_parity = compare_input_tokens(
        list(left.values()), list(right.values()), len(left)
    )
    quality = compare_quality(
        list(left.values()),
        list(right.values()),
        left_report["requested_records"],
        tolerance,
    )
    passed = token_parity["status"] == "matched" and quality["parity_passed"]
    old_latency = left_report["latency_ms"]
    new_latency = right_report["latency_ms"]
    return {
        "status": "matched" if passed else "mismatch",
        "baseline": str(left_dir.resolve()),
        "candidate": str(right_dir.resolve()),
        "requests": len(left),
        "input_token_parity": token_parity,
        "quality_parity": quality,
        "latency_ms": {
            "baseline_p50": old_latency["p50"],
            "baseline_p95": old_latency["p95"],
            "candidate_p50": new_latency["p50"],
            "candidate_p95": new_latency["p95"],
            "p50_ratio": new_latency["p50"] / old_latency["p50"]
            if passed and old_latency["p50"]
            else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probability-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if (
        not math.isfinite(args.probability_tolerance)
        or not 0 <= args.probability_tolerance <= 1
    ):
        parser.error("probability tolerance must be between 0 and 1")
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    result = compare(args.baseline, args.candidate, args.probability_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"{result['status']}: {args.output}")
    raise SystemExit(0 if result["status"] == "matched" else 1)


if __name__ == "__main__":
    main()
