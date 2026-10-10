import pytest

from benchmarks import prefill_gpu_telemetry as telemetry

GPU = "GPU-h200"
GPU_CSV = "GPU-other, Other, 100, 10, 90, 0\nGPU-h200, NVIDIA H200, 140000, 12000, 128000, 95\n"


def test_gpu_and_process_filters_do_not_mix_devices():
    gpu = telemetry.parse_gpu(GPU_CSV, GPU)
    assert gpu["used_mib"] == 12000
    assert gpu["utilization_percent"] == 95
    apps = telemetry.parse_apps(
        "GPU-other, 5, python, 10\nGPU-h200, 10, python, 11000\n", GPU
    )
    assert apps == [{"pid": 10, "process_name": "python", "used_mib": 11000}]
    assert telemetry.parse_apps("", GPU) == []
    with pytest.raises(ValueError):
        telemetry.parse_gpu(GPU_CSV.replace("12000", "[N/A]"), GPU)
    with pytest.raises(ValueError):
        telemetry.parse_gpu(GPU_CSV, "GPU-missing")


def test_linux_identity_handles_parentheses_in_process_name(tmp_path):
    proc = tmp_path / "10"
    proc.mkdir()
    fields = ["S"] + ["0"] * 21
    fields[11], fields[12], fields[19], fields[21] = "12", "3", "777", "900"
    (proc / "stat").write_text("10 (worker (pool)) " + " ".join(fields))
    assert telemetry.process_stats(10, tmp_path) == {
        "start_ticks": 777,
        "cpu_ticks": 15,
        "rss_pages": 900,
    }


def test_foreign_work_and_pid_reuse_invalidate_observation(monkeypatch):
    def query(field):
        return {
            "returncode": 0,
            "stderr": "",
            "stdout": GPU_CSV
            if field.startswith("--query-gpu=")
            else ("GPU-h200, 10, python, 11000\nGPU-h200, 99, training, 1000\n"),
        }

    monkeypatch.setattr(telemetry, "query", query)
    monkeypatch.setattr(
        telemetry,
        "process_stats",
        lambda pid: {
            "start_ticks": 999,
            "cpu_ticks": 5,
            "rss_pages": 100,
        },
    )
    sample = telemetry.sample(GPU, {10: 777})
    assert sample["issues"] == [
        {"type": "foreign_gpu_process", "pids": [99]},
        {"type": "pid_reused", "pid": 10},
    ]
    assert sample["completed_ns"] >= sample["started_ns"]
    assert "training" in sample["raw_apps"]["stdout"]
    assert not telemetry.summarize([sample], 1)["sampled_isolation_checks_passed"]


def test_missing_process_and_query_errors_are_not_idle_success(monkeypatch):
    monkeypatch.setattr(
        telemetry,
        "query",
        lambda field: {
            "returncode": 0,
            "stderr": "",
            "stdout": GPU_CSV if field.startswith("--query-gpu=") else "",
        },
    )
    monkeypatch.setattr(
        telemetry,
        "process_stats",
        lambda pid: {
            "start_ticks": 777,
            "cpu_ticks": 5,
            "rss_pages": 100,
        },
    )
    missing = telemetry.sample(GPU, {10: 777})
    assert missing["issues"] == [{"type": "missing_expected_gpu_process", "pids": [10]}]
    monkeypatch.setattr(
        telemetry,
        "query",
        lambda field: {
            "returncode": 1,
            "stderr": "driver error",
            "stdout": "",
        },
    )
    failed = telemetry.sample(GPU, {10: 777})
    assert failed["issues"][0]["type"] == "collection_error"
    assert failed["raw_gpu"]["stderr"] == "driver error"


def test_sampling_gaps_and_peaks_are_reported():
    gpu = telemetry.parse_gpu(GPU_CSV, GPU)
    samples = [
        {"started_ns": 0, "completed_ns": 100, "gpu": gpu, "issues": []},
        {
            "started_ns": 1_000_000_000,
            "completed_ns": 1_000_000_100,
            "gpu": {**gpu, "used_mib": 15000, "free_mib": 125000},
            "issues": [],
        },
    ]
    result = telemetry.summarize(samples, 1)
    assert result["sampled_isolation_checks_passed"]
    assert not result["exclusive_gpu_claim"]
    assert result["peak_used_mib"] == 15000
    assert result["min_free_mib"] == 125000
    samples[1]["started_ns"] = 5_000_000_000
    samples[1]["completed_ns"] = 5_000_000_100
    result = telemetry.summarize(samples, 1)
    assert not result["sampled_isolation_checks_passed"]
    assert result["issues"] == [{"type": "sampling_gap", "max_seconds": 5.0}]
