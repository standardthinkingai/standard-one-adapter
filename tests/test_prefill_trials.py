import json
from pathlib import Path

import pytest

from benchmarks import prefill_trials


def _manifest(created, minimum=30.0, mode="logprobs"):
    return {
        "created_at": created,
        "model": "decision-model",
        "selected_records": 128,
        "selected_items_sha256": "items",
        "selected_decisions": 128,
        "max_options_in_workload": 3,
        "workload_sha256": "workload",
        "job_order_sha256": "order",
        "concurrency": prefill_trials.REQUIRED_CONCURRENCY,
        "warmup_requests_per_c": 128,
        "repeats": 1000,
        "min_measure_seconds": minimum,
        "seed": 42,
        "limit": 128,
        "request_options": {"temperature": 1.0, "return_logprobs": False},
        "client_code_sha256": {"prefill_engine_bench.py": "code"},
        "data": [{"sha256": "corpus"}],
        "settings_manifest_sha256": f"settings-{mode}",
        "settings": {
            "hardware": "H200",
            "model_path": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
            "model_revision": "a" * 40,
            "precision": "bfloat16",
            "prompt_contract": "served-v1",
            "label_contract": "A-C-v1",
            "head_mode": mode,
            "labels": ["A", "B", "C"],
            "token_ids": [11, 12, 13],
            "head_artifact": None,
        },
        "engine": {
            "model_info": {"model_path": "model"},
            "server_info": {"max_running_requests": 128},
        },
        "token_id_audit": {"input_token_ids_sha256": ["ids"] * 128},
    }


def _make_pairs(tmp_path: Path, count=3, minimum=30.0, learned=False):
    pairs = []
    for number in range(1, count + 1):
        runs = []
        for role in ("baseline", "candidate"):
            directory = tmp_path / f"trial-{number}-{role}"
            directory.mkdir()
            mode = "learned" if learned and role == "candidate" else "logprobs"
            (directory / "manifest.json").write_text(
                json.dumps(
                    _manifest(f"2026-09-28T00:00:{number:02d}-{role}", minimum, mode)
                )
            )
            (directory / "report.json").write_text(
                json.dumps(
                    {
                        "concurrency": {
                            f"c{c}": {
                                "elapsed_s": 31.0,
                                "errors": 0,
                                "cached_tokens": 0,
                                "measurement_failures": [],
                            }
                            for c in prefill_trials.REQUIRED_CONCURRENCY
                        }
                    }
                )
            )
            for c in prefill_trials.REQUIRED_CONCURRENCY:
                phase = directory / f"c{c}"
                phase.mkdir()
                (phase / "raw_outcomes.jsonl").write_text(
                    f"trial={number},role={role},c={c}\n"
                )
            runs.append(directory)
        pairs.append(tuple(runs))
    return pairs


def _fake_compare(*, learned=False, audit="matched_external_audits", parity=True):
    def compare(baseline, candidate, tolerance):
        number = int(baseline.name.split("-")[1])
        phases = {}
        for c in prefill_trials.REQUIRED_CONCURRENCY:
            regress = number == 2 and c == 32
            old = {
                "client_latency_ms": {"p50": 100.0, "p95": 120.0, "p99": 140.0},
                "requests_per_second": 10.0,
                "decisions_per_second": 10.0,
                "input_tokens_per_second": 100.0,
            }
            new = {
                "client_latency_ms": {
                    "p50": 110.0 if regress else 90.0,
                    "p95": 130.0 if regress else 110.0,
                    "p99": 150.0 if regress else 130.0,
                },
                "requests_per_second": 9.0 if regress else 11.0,
                "decisions_per_second": 9.0 if regress else 11.0,
                "input_tokens_per_second": 90.0 if regress else 110.0,
            }
            phases[f"c{c}"] = {
                "baseline": old,
                "candidate": new,
                "input_token_parity": {"status": "matched"},
                "quality_parity": {"parity_passed": parity},
                "displayed_answer_parity": {
                    "status": "matched" if parity else "mismatch"
                },
                "quality": {
                    "baseline": {
                        "overall": {"n": 1194, "acc": 0.7},
                        "by_type": {"score": {"score_mae": 0.5}},
                        "by_suite": {"real/development": {"accuracy": 0.7}},
                    },
                    "candidate": {
                        "overall": {"n": 1194, "acc": 0.71},
                        "by_type": {"score": {"score_mae": 0.4}},
                        "by_suite": {"real/development": {"accuracy": 0.71}},
                    },
                },
            }
        return {
            "status": "no_winner_gate" if learned or not parity else "comparable",
            "concurrency": prefill_trials.REQUIRED_CONCURRENCY,
            "head_mode_array": ["logprobs", "learned" if learned else "logprobs"],
            "token_id_audit_status": audit,
            "quality_parity_required": not learned,
            "per_concurrency": phases,
        }

    return compare


def test_three_pairs_keep_each_regression_and_quality_separate(tmp_path, monkeypatch):
    pairs = _make_pairs(tmp_path)
    monkeypatch.setattr(prefill_trials, "compare_saved", _fake_compare())
    result = prefill_trials.summarize_pairs(pairs)
    assert result["status"] == "local_checks_passed"
    assert result["final_claim_eligible"] is False
    assert result["trial_count"] == 3
    assert result["quality_winner"] is None
    assert result["per_concurrency"]["c32"]["regression_trials"]["p50_latency_ms"] == [
        2
    ]
    assert result["per_concurrency"]["c32"]["regression_trials"][
        "decisions_per_second"
    ] == [2]
    assert result["per_concurrency"]["c32"]["candidate_over_baseline_ratio"][
        "p50_latency_ms"
    ] == {
        "median": 0.9,
        "min": 0.9,
        "max": 1.1,
    }
    assert (
        result["trials"][1]["per_concurrency"]["c32"]["candidate_over_baseline_ratio"][
            "p95_latency_ms"
        ]
        > 1
    )
    assert (
        result["trials"][0]["per_concurrency"]["c8"]["quality"]["baseline"]["overall"][
            "n"
        ]
        == 1194
    )
    assert (
        result["trials"][2]["per_concurrency"]["c8"]["quality"]["candidate"]["by_type"][
            "score"
        ]["score_mae"]
        == 0.4
    )


def test_requires_three_unique_pairs_and_raw_runs(tmp_path, monkeypatch):
    pairs = _make_pairs(tmp_path)
    monkeypatch.setattr(prefill_trials, "compare_saved", _fake_compare())
    with pytest.raises(ValueError, match="at least three"):
        prefill_trials.summarize_pairs(pairs[:2])
    with pytest.raises(ValueError, match="directory was reused"):
        prefill_trials.summarize_pairs([pairs[0], pairs[1], pairs[0]])
    for c in prefill_trials.REQUIRED_CONCURRENCY:
        first = pairs[0][0] / f"c{c}" / "raw_outcomes.jsonl"
        copied = pairs[1][0] / f"c{c}" / "raw_outcomes.jsonl"
        copied.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="identical raw run reused"):
        prefill_trials.summarize_pairs(pairs)


def test_cohort_settings_and_workload_must_stay_fixed(tmp_path, monkeypatch):
    pairs = _make_pairs(tmp_path)
    monkeypatch.setattr(prefill_trials, "compare_saved", _fake_compare())
    path = pairs[1][0] / "manifest.json"
    value = json.loads(path.read_text())
    value["settings"]["hardware"] = "MI325X"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="settings hardware differs"):
        prefill_trials.summarize_pairs(pairs)
    value["settings"]["hardware"] = "H200"
    value["client_code_sha256"] = {"prefill_engine_bench.py": "changed"}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="client code differs"):
        prefill_trials.summarize_pairs(pairs)


def test_short_duration_errors_and_missing_parity_are_visible(tmp_path, monkeypatch):
    pairs = _make_pairs(tmp_path, minimum=0.0)
    monkeypatch.setattr(
        prefill_trials,
        "compare_saved",
        _fake_compare(audit="missing_external_audit", parity=False),
    )
    path = pairs[0][0] / "report.json"
    report = json.loads(path.read_text())
    report["concurrency"]["c64"]["elapsed_s"] = 20.0
    report["concurrency"]["c64"]["errors"] = 1
    path.write_text(json.dumps(report))
    result = prefill_trials.summarize_pairs(pairs)
    assert result["status"] == "local_checks_failed"
    assert any(
        "configured minimum below 30s" in item for item in result["local_gate_failures"]
    )
    assert any(
        "observed duration below 30s" in item for item in result["local_gate_failures"]
    )
    assert any(
        "response or measurement errors" in item
        for item in result["local_gate_failures"]
    )
    assert any(
        "token ID audit missing" in item for item in result["local_gate_failures"]
    )
    assert any(
        "LM probability/display parity failed" in item
        for item in result["local_gate_failures"]
    )
    assert (
        result["trials"][0]["per_concurrency"]["c64"]["gates"]["baseline"]["elapsed_s"]
        == 20.0
    )


def test_learned_head_keeps_quality_separate_without_parity_claim(
    tmp_path, monkeypatch
):
    pairs = _make_pairs(tmp_path, learned=True)
    monkeypatch.setattr(prefill_trials, "compare_saved", _fake_compare(learned=True))
    result = prefill_trials.summarize_pairs(pairs)
    assert result["status"] == "local_checks_passed"
    assert result["head_mode_array"] == ["logprobs", "learned"]
    assert (
        result["trials"][0]["per_concurrency"]["c8"]["gates"]["probability_parity"]
        == "not_applicable_learned_head"
    )
    assert result["quality_aggregation"].startswith("none")
    assert result["final_claim_eligible"] is False
