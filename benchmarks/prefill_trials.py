"""Summarize at least three saved, paired prefill benchmark trials offline.

Every pair is revalidated by prefill_engine_bench.compare_saved, which calls
load_saved on both runs. This command neither starts a server nor sends HTTP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from jev_adapter.benchmarks.run import write_json

if __package__:
    from .prefill_engine_bench import (
        _same_workload,
        compare_saved,
        file_sha256,
    )
else:
    from prefill_engine_bench import _same_workload, compare_saved, file_sha256


REQUIRED_CONCURRENCY = [8, 16, 32, 64, 128]
MIN_SECONDS = 30.0
METRIC_NAMES = (
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "requests_per_second",
    "decisions_per_second",
    "input_tokens_per_second",
)
LATENCY_METRICS = frozenset(METRIC_NAMES[:3])
EXTERNAL_CHECKS = (
    "GPU exclusivity and competing GPU work require external telemetry.",
    "Stock SGLang wheel provenance and launch settings require external evidence.",
    "Distinct saved raw runs do not prove independent thermal or load conditions.",
)


def _manifest(directory: Path) -> dict:
    value = json.loads((directory / "manifest.json").read_text())
    if not isinstance(value, dict):
        raise TypeError(f"manifest is not a JSON object: {directory}")
    return value


def _report(directory: Path) -> dict:
    value = json.loads((directory / "report.json").read_text())
    if not isinstance(value, dict):
        raise TypeError(f"report is not a JSON object: {directory}")
    return value


def _raw_fingerprint(directory: Path, concurrency: list[int]) -> str:
    digest = hashlib.sha256()
    for c in concurrency:
        path = directory / f"c{c}" / "raw_outcomes.jsonl"
        digest.update(f"c{c}:".encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def _metrics(side: dict) -> dict[str, float]:
    latency = side["client_latency_ms"]
    values = {
        "p50_latency_ms": latency["p50"],
        "p95_latency_ms": latency["p95"],
        "p99_latency_ms": latency["p99"],
        "requests_per_second": side["requests_per_second"],
        "decisions_per_second": side["decisions_per_second"],
        "input_tokens_per_second": side["input_tokens_per_second"],
    }
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in values.values()
    ):
        raise ValueError(
            "a saved trial has a missing or nonpositive performance metric"
        )
    return values


def _range(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _same_role(reference: dict, candidate: dict, role: str) -> None:
    _same_workload(reference, candidate)
    for key in ("settings_manifest_sha256", "settings", "engine"):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"{role} {key} changed between trials")
    if (reference.get("token_id_audit") or {}).get("input_token_ids_sha256") != (
        candidate.get("token_id_audit") or {}
    ).get("input_token_ids_sha256"):
        raise ValueError(f"{role} token ID audit changed between trials")


def summarize_pairs(
    pairs: list[tuple[Path, Path]], probability_tolerance: float = 0.01
) -> dict:
    if len(pairs) < 3:
        raise ValueError("at least three paired trials are required")
    if not math.isfinite(probability_tolerance) or not 0 <= probability_tolerance <= 1:
        raise ValueError("probability tolerance must be between zero and one")

    seen_paths: set[Path] = set()
    seen_raw: dict[str, str] = {}
    seen_created: set[str] = set()
    references: dict[str, dict] = {}
    trials = []
    by_c: dict[str, list[dict]] = {f"c{c}": [] for c in REQUIRED_CONCURRENCY}

    for number, (old_path, new_path) in enumerate(pairs, 1):
        directories = {"baseline": old_path.resolve(), "candidate": new_path.resolve()}
        for role, directory in directories.items():
            if directory in seen_paths:
                raise ValueError(f"a saved run directory was reused: {directory}")
            seen_paths.add(directory)

        comparison = compare_saved(
            directories["baseline"], directories["candidate"], probability_tolerance
        )
        if comparison["concurrency"] != REQUIRED_CONCURRENCY:
            raise ValueError("each trial must include C8, C16, C32, C64, C128")

        manifests = {role: _manifest(path) for role, path in directories.items()}
        reports = {role: _report(path) for role, path in directories.items()}
        fingerprints = {}
        for role, directory in directories.items():
            manifest = manifests[role]
            if role in references:
                _same_role(references[role], manifest, role)
            else:
                references[role] = manifest
            fingerprint = _raw_fingerprint(directory, REQUIRED_CONCURRENCY)
            if fingerprint in seen_raw:
                raise ValueError(
                    f"identical raw run reused: {directory} and {seen_raw[fingerprint]}"
                )
            seen_raw[fingerprint] = str(directory)
            fingerprints[role] = fingerprint
            created = manifest.get("created_at")
            if not isinstance(created, str) or created in seen_created:
                raise ValueError("a saved run timestamp is missing or reused")
            seen_created.add(created)

        trial = {
            "trial": number,
            "directories": {role: str(path) for role, path in directories.items()},
            "raw_sha256": fingerprints,
            "pair_status": comparison["status"],
            "head_mode_array": comparison["head_mode_array"],
            "token_id_audit_status": comparison["token_id_audit_status"],
            "minimum_measure_seconds": {
                role: manifests[role].get("min_measure_seconds") for role in directories
            },
            "per_concurrency": {},
        }
        for c in REQUIRED_CONCURRENCY:
            key = f"c{c}"
            pair_phase = comparison["per_concurrency"][key]
            baseline, candidate = (
                _metrics(pair_phase["baseline"]),
                _metrics(pair_phase["candidate"]),
            )
            ratios = {name: candidate[name] / baseline[name] for name in METRIC_NAMES}
            run_checks = {}
            for role in directories:
                phase = reports[role]["concurrency"][key]
                duration = phase["elapsed_s"]
                run_checks[role] = {
                    "elapsed_s": duration,
                    "configured_minimum_seconds": manifests[role].get(
                        "min_measure_seconds"
                    ),
                    "duration_at_least_30s": duration >= MIN_SECONDS,
                    "configured_minimum_at_least_30s": manifests[role].get(
                        "min_measure_seconds", 0
                    )
                    >= MIN_SECONDS,
                    "errors": phase["errors"],
                    "cached_tokens": phase["cached_tokens"],
                    "measurement_failures": phase["measurement_failures"],
                }
            probability_parity = (
                "not_applicable_learned_head"
                if not comparison["quality_parity_required"]
                else "matched"
                if pair_phase["quality_parity"]["parity_passed"]
                and pair_phase["displayed_answer_parity"]["status"] == "matched"
                else "mismatch"
            )
            gates = {
                "baseline": run_checks["baseline"],
                "candidate": run_checks["candidate"],
                "input_token_counts": pair_phase["input_token_parity"]["status"],
                "input_token_ids": comparison["token_id_audit_status"],
                "probability_parity": probability_parity,
            }
            trial_phase = {
                "baseline": baseline,
                "candidate": candidate,
                "candidate_over_baseline_ratio": ratios,
                "quality": pair_phase["quality"],
                "gates": gates,
            }
            trial["per_concurrency"][key] = trial_phase
            by_c[key].append(trial_phase)
        trials.append(trial)

    head_modes = trials[0]["head_mode_array"]
    learned = "learned" in head_modes
    local_failures = []
    for trial in trials:
        if trial["token_id_audit_status"] != "matched_external_audits":
            local_failures.append(f"trial {trial['trial']}: token ID audit missing")
        for role, configured in trial["minimum_measure_seconds"].items():
            if type(configured) not in (int, float) or configured < MIN_SECONDS:
                local_failures.append(
                    f"trial {trial['trial']} {role}: configured minimum below 30s"
                )
        for key, phase in trial["per_concurrency"].items():
            checks = phase["gates"]
            for role in ("baseline", "candidate"):
                result = checks[role]
                if not result["duration_at_least_30s"]:
                    local_failures.append(
                        f"trial {trial['trial']} {key} {role}: observed duration below 30s"
                    )
                if (
                    result["errors"]
                    or result["cached_tokens"]
                    or result["measurement_failures"]
                ):
                    local_failures.append(
                        f"trial {trial['trial']} {key} {role}: response or measurement errors"
                    )
            if checks["input_token_counts"] != "matched":
                local_failures.append(
                    f"trial {trial['trial']} {key}: token counts differ"
                )
            if not learned and checks["probability_parity"] != "matched":
                local_failures.append(
                    f"trial {trial['trial']} {key}: LM probability/display parity failed"
                )

    summaries = {}
    for c in REQUIRED_CONCURRENCY:
        key = f"c{c}"
        phases = by_c[key]
        summaries[key] = {
            "baseline": {
                name: _range([phase["baseline"][name] for phase in phases])
                for name in METRIC_NAMES
            },
            "candidate": {
                name: _range([phase["candidate"][name] for phase in phases])
                for name in METRIC_NAMES
            },
            "candidate_over_baseline_ratio": {
                name: _range(
                    [phase["candidate_over_baseline_ratio"][name] for phase in phases]
                )
                for name in METRIC_NAMES
            },
            "regression_trials": {
                name: [
                    index
                    for index, phase in enumerate(phases, 1)
                    if (
                        phase["candidate_over_baseline_ratio"][name] > 1
                        if name in LATENCY_METRICS
                        else phase["candidate_over_baseline_ratio"][name] < 1
                    )
                ]
                for name in METRIC_NAMES
            },
        }

    first = references["baseline"]
    return {
        "schema_version": 1,
        "status": "local_checks_passed"
        if not local_failures
        else "local_checks_failed",
        "trial_count": len(trials),
        "required_concurrency": REQUIRED_CONCURRENCY,
        "required_seconds_per_c": MIN_SECONDS,
        "model": first["model"],
        "model_revision": first["settings"]["model_revision"],
        "hardware": first["settings"]["hardware"],
        "head_mode_array": head_modes,
        "probability_tolerance": probability_tolerance,
        "local_gate_failures": local_failures,
        "final_claim_eligible": False,
        "external_checks_required": EXTERNAL_CHECKS,
        "quality_aggregation": "none; each run reports only its first measured pass",
        "quality_winner": None,
        "trials": trials,
        "per_concurrency": summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        type=Path,
        metavar=("BASELINE_DIR", "CANDIDATE_DIR"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probability-tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    try:
        result = summarize_pairs(args.pair, args.probability_tolerance)
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    return 0 if result["status"] == "local_checks_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
