"""Offline BF16-versus-FP8 quality observations; never a performance comparison.

Each --pair contains a saved BF16 benchmark and its matching FP8 benchmark.
load_saved validates complete saved reports against every raw HTTP outcome.
Quality uses repeat=0 only and remains separate for every pair and C.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from jev_adapter.benchmarks.metrics import metrics, predicted_index, quantile

if __package__:
    from .prefill_engine_bench import (
        HEX40,
        RESOURCE_FIELDS,
        file_sha256,
        first_pass_rows,
        load_saved,
        precision_settings,
    )
else:
    from prefill_engine_bench import (
        HEX40,
        RESOURCE_FIELDS,
        file_sha256,
        first_pass_rows,
        load_saved,
        precision_settings,
    )

BF16_PROFILE = "unquantized_bfloat16"
FP8_PROFILE = "fp8_w8a8_backbone_bf16_head"
REFERENCE_CLEAN_QUESTIONS = 1194
IDENTITY = ("suite", "split", "id", "question")
QUESTION_CONTRACT = (
    "keys",
    "label",
    "type",
    "task",
    "source",
    "variant",
    "argmax_tie_break",
)


def _identity(row):
    return tuple(row[key] for key in IDENTITY)


def _unique_questions(rows):
    result = {}
    for row in rows:
        key = _identity(row)
        if key in result:
            raise ValueError(f"duplicate first-pass question identity: {key}")
        result[key] = row
    return result


def _eligible(row):
    return (
        row["variant"] == "clean"
        and row["source"] != "unknowable"
        and not row["suite"].startswith("synthetic-")
    )


def _artifact_sha(settings):
    artifact = settings.get("head_artifact")
    if settings["head_mode"] == "logprobs":
        if artifact is not None:
            raise ValueError("logprobs profile must not contain a head artifact")
        return None
    if settings["head_mode"] != "lm_rows" or not isinstance(artifact, dict):
        raise ValueError("cross-precision quality requires original LM heads/rows")
    value = artifact.get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("saved readout artifact SHA256 is missing")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("saved readout artifact SHA256 is malformed") from error
    return value


def _contracts(bf16, fp8):
    a, b = bf16["settings"], fp8["settings"]
    if not isinstance(a.get("model_revision"), str) or not HEX40.fullmatch(
        a["model_revision"]
    ):
        raise ValueError("quality source checkpoint requires a pinned model revision")
    if precision_settings(a) != (None, BF16_PROFILE):
        raise ValueError("left run must be the unquantized BF16 profile")
    if precision_settings(b) != ("fp8", FP8_PROFILE):
        raise ValueError("right run must be native FP8 backbone + BF16 head")
    for key in (
        "hardware",
        "model_path",
        "model_revision",
        "precision",
        "prompt_contract",
        "label_contract",
        "head_mode",
        "labels",
        "token_ids",
    ):
        if a.get(key) != b.get(key):
            raise ValueError(f"quality settings {key} differs")
    if _artifact_sha(a) != _artifact_sha(b):
        raise ValueError("quality readout artifact differs")
    for key in (
        "model",
        "workload_sha256",
        "selected_items_sha256",
        "selected_records",
        "selected_decisions",
        "request_options",
        "label_array_sha256",
        "concurrency",
    ):
        if bf16.get(key) != fp8.get(key):
            raise ValueError(f"quality workload {key} differs")
    options = bf16.get("request_options")
    if (
        not isinstance(options, dict)
        or type(options.get("temperature")) not in (int, float)
        or not math.isfinite(options["temperature"])
        or options["temperature"] != 1.0
        or type(options.get("permutations")) is not int
        or options["permutations"] != 1
    ):
        raise ValueError("quality comparison requires temperature=1 and permutations=1")
    old_audit = bf16.get("token_id_audit")
    new_audit = fp8.get("token_id_audit")
    if (
        old_audit
        and new_audit
        and (old_audit["input_token_ids_sha256"] != new_audit["input_token_ids_sha256"])
    ):
        raise ValueError("cross-precision audited input token IDs differ")


def _small_metrics(rows):
    value = metrics(rows)
    if value is None:
        return None
    return {
        key: value[key]
        for key in ("n", "acc", "brier", "nll", "score_mae", "ranked_probability_score")
        if key in value
    }


def _differences(pairs):
    question_max, components = [], []
    changed = []
    for a, b in pairs:
        delta = [abs(x - y) for x, y in zip(a["p"], b["p"])]
        components.extend(delta)
        question_max.append(max(delta))
        if predicted_index(a) != predicted_index(b):
            changed.append(
                {
                    **dict(zip(IDENTITY, _identity(a))),
                    "bf16_prediction": a["keys"][predicted_index(a)],
                    "fp8_prediction": b["keys"][predicted_index(b)],
                    "gold": a["keys"][a["label"]],
                }
            )
    n = len(pairs)
    return {
        "questions": n,
        "argmax_changes": len(changed),
        "argmax_change_rate": len(changed) / n if n else None,
        "argmax_change_samples": changed[:20],
        "probability_abs_delta": {
            "components": len(components),
            "mean_over_components": math.fsum(components) / len(components)
            if components
            else None,
            "mean_of_question_max": math.fsum(question_max) / n if n else None,
            "p50_of_question_max": quantile(question_max, 0.50),
            "p95_of_question_max": quantile(question_max, 0.95),
            "max": max(question_max, default=None),
            "questions_over_0_01": sum(value > 0.01 for value in question_max),
        },
    }


def _grouped(pairs, field):
    groups = defaultdict(list)
    for a, b in pairs:
        groups[a[field]].append((a, b))
    return {
        str(key): {
            "bf16": _small_metrics([a for a, _ in sample]),
            "fp8": _small_metrics([b for _, b in sample]),
            "changes": _differences(sample),
        }
        for key, sample in sorted(groups.items())
    }


def analyze_pair(bf16_dir: Path, fp8_dir: Path):
    old, old_report, old_raw = load_saved(bf16_dir)
    new, new_report, new_raw = load_saved(fp8_dir)
    _contracts(old, new)
    per_c = {}
    for c in old["concurrency"]:
        key = f"c{c}"
        left = first_pass_rows(old_raw[key], old["selected_records"])
        right = first_pass_rows(new_raw[key], new["selected_records"])
        if left is None or right is None:
            raise ValueError(f"{key} lacks first-pass outcomes")
        a, b = _unique_questions(left), _unique_questions(right)
        if a.keys() != b.keys():
            raise ValueError(f"{key} first-pass question identities differ")
        pairs = []
        for identity, row in a.items():
            other = b[identity]
            if any(row.get(field) != other.get(field) for field in QUESTION_CONTRACT):
                raise ValueError(f"{key} question contract differs: {identity}")
            pairs.append((row, other))
        clean = [(a, b) for a, b in pairs if _eligible(a)]
        n = len(clean)
        reused_quality = {}
        for role, report in (("bf16", old_report), ("fp8", new_report)):
            phase = report["concurrency"][key]
            if (phase["quality_overall"] or {}).get("n", 0) != n:
                raise ValueError(f"{key} saved clean-quality denominator differs")
            reused_quality[role] = {
                name: phase[name]
                for name in ("quality_overall", "quality_by_type", "quality_by_suite")
            }
        per_c[key] = {
            "sample": "repeat=0 only; one row per unique suite/split/id/question",
            "selected_requests_per_first_pass": old["selected_records"],
            "unique_questions_per_first_pass": len(pairs),
            "eligible_clean_knowable_non_synthetic_questions": n,
            "reference_clean1194_coverage": n == REFERENCE_CLEAN_QUESTIONS,
            "excluded_questions": len(pairs) - n,
            "measured_http_responses_not_independent_quality_samples": {
                "bf16": len(old_raw[key]),
                "fp8": len(new_raw[key]),
            },
            "saved_quality_verified_against_raw": reused_quality,
            "fp8_minus_bf16": {
                metric: reused_quality["fp8"]["quality_overall"][metric]
                - reused_quality["bf16"]["quality_overall"][metric]
                for metric in ("acc", "brier", "nll")
            }
            if n
            else None,
            "all_question_changes": _differences(pairs),
            "clean_question_changes": _differences(clean),
            "clean_by_type": _grouped(clean, "type"),
            "clean_by_benchmark_task": _grouped(clean, "task"),
            "raw_sha256": {
                role: file_sha256(directory / key / "raw_outcomes.jsonl")
                for role, directory in (("bf16", bf16_dir), ("fp8", fp8_dir))
            },
        }
    old_server, new_server = old["engine"]["server_info"], new["engine"]["server_info"]
    return {
        "bf16": str(bf16_dir.resolve()),
        "fp8": str(fp8_dir.resolve()),
        "model_path": old["settings"]["model_path"],
        "model_revision": old["settings"]["model_revision"],
        "head_mode": old["settings"]["head_mode"],
        "head_artifact_sha256": _artifact_sha(old["settings"]),
        "precision_profiles": [BF16_PROFILE, FP8_PROFILE],
        "temperature": old["request_options"]["temperature"],
        "workload_sha256": old["workload_sha256"],
        "token_id_audit_status": "matched"
        if old.get("token_id_audit") and new.get("token_id_audit")
        else "missing_external_audit",
        "runtime_differences": {
            field: {"bf16": old_server.get(field), "fp8": new_server.get(field)}
            for field in RESOURCE_FIELDS
            if old_server.get(field) != new_server.get(field)
        },
        "source_sha256": {
            role: {
                name: file_sha256(directory / name)
                for name in ("manifest.json", "report.json")
            }
            for role, directory in (("bf16", bf16_dir), ("fp8", fp8_dir))
        },
        "per_concurrency": per_c,
    }


def analyze_pairs(pairs):
    if not pairs:
        raise ValueError("at least one BF16/FP8 pair is required")
    resolved = [(a.resolve(), b.resolve()) for a, b in pairs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate saved quality pair")
    use_counts = Counter(path for pair in resolved for path in pair)
    return {
        "schema_version": 1,
        "status": "complete",
        "analysis_scope": "cross_precision_quality_only",
        "raw_validation": "prefill_engine_bench.load_saved; every saved HTTP outcome and report revalidated",
        "sampling_policy": "repeat=0 per pair/C; no pooling across repeats, C, or trials; reused runs and repeated questions do not increase independent sample size",
        "probability_basis": "validated normalized API distributions, including noul [P(false),P(true)]; NLL floor 1e-9 and Brier definitions unchanged",
        "reference_clean_questions_per_full_development_pass": REFERENCE_CLEAN_QUESTIONS,
        "performance_winner": None,
        "quality_winner": None,
        "causal_quantization_claim": False,
        "reused_source_runs": {
            str(path): count for path, count in use_counts.items() if count > 1
        },
        "pairs": [
            {"pair_index": index, **analyze_pair(a, b)}
            for index, (a, b) in enumerate(resolved, 1)
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        type=Path,
        action="append",
        required=True,
        metavar=("BF16_BENCHMARK", "FP8_BENCHMARK"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze_pairs(args.pair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, ensure_ascii=False, allow_nan=False)
        target.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pairs": len(result["pairs"]),
                "analysis_scope": result["analysis_scope"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
