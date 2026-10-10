"""Read-only NVIDIA GPU sampling alongside a separately managed benchmark.

This tool neither starts nor stops an inference/training process. Explicit GPU
worker PIDs must be supplied after handoff. Sampling cannot detect a workload
that starts and ends entirely between samples; preserve the handoff separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

GPU_FIELDS = "uuid,name,memory.total,memory.used,memory.free,utilization.gpu"
APP_FIELDS = "gpu_uuid,pid,process_name,used_gpu_memory"


def rows(text, count):
    result = []
    for row in csv.reader(io.StringIO(text)):
        if not row or not any(field.strip() for field in row):
            continue
        if len(row) != count:
            raise ValueError("unexpected nvidia-smi CSV columns")
        result.append([field.strip() for field in row])
    return result


def number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("GPU counter is unavailable or invalid")
    return value


def parse_gpu(text, uuid):
    selected = [row for row in rows(text, 6) if row[0] == uuid]
    if len(selected) != 1:
        raise ValueError("selected GPU UUID is missing or duplicated")
    row = selected[0]
    result = dict(
        zip(
            ("total_mib", "used_mib", "free_mib", "utilization_percent"),
            map(number, row[2:]),
        )
    )
    if (
        result["total_mib"] == 0
        or result["used_mib"] > result["total_mib"]
        or result["free_mib"] > result["total_mib"]
        or result["utilization_percent"] > 100
    ):
        raise ValueError("GPU counters are out of range")
    return {"uuid": uuid, "name": row[1], **result}


def parse_apps(text, uuid):
    result = []
    for row in rows(text, 4):
        if row[0] != uuid:
            continue
        pid = int(row[1])
        if pid <= 0 or any(app["pid"] == pid for app in result):
            raise ValueError("invalid or duplicate GPU PID")
        result.append({"pid": pid, "process_name": row[2], "used_mib": number(row[3])})
    return result


def process_stats(pid, proc=Path("/proc")):
    # Split after the final ')' because Linux comm may contain spaces or ')'.
    fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "start_ticks": int(fields[19]),
        "cpu_ticks": int(fields[11]) + int(fields[12]),
        "rss_pages": int(fields[21]),
    }


def query(field):
    try:
        result = subprocess.run(
            ["nvidia-smi", field, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": str(error)}


def sample(uuid, identities):
    started = time.perf_counter_ns()
    gpu = query("--query-gpu=" + GPU_FIELDS)
    apps = query("--query-compute-apps=" + APP_FIELDS)
    result = {
        "started_ns": started,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "raw_gpu": gpu,
        "raw_apps": apps,
        "issues": [],
    }
    try:
        if gpu["returncode"] != 0 or apps["returncode"] != 0:
            raise ValueError("nvidia-smi query failed")
        result["gpu"] = parse_gpu(gpu["stdout"], uuid)
        result["apps"] = parse_apps(apps["stdout"], uuid)
        actual = {app["pid"] for app in result["apps"]}
        foreign = actual - set(identities)
        missing = set(identities) - actual
        if foreign:
            result["issues"].append(
                {"type": "foreign_gpu_process", "pids": sorted(foreign)}
            )
        if missing:
            result["issues"].append(
                {"type": "missing_expected_gpu_process", "pids": sorted(missing)}
            )
        result["processes"] = {}
        for pid, expected_birth in identities.items():
            stats = process_stats(pid)
            result["processes"][str(pid)] = stats
            if stats["start_ticks"] != expected_birth:
                result["issues"].append({"type": "pid_reused", "pid": pid})
    except (ValueError, OSError, IndexError) as error:
        result["issues"].append({"type": "collection_error", "detail": str(error)})
    result["completed_ns"] = time.perf_counter_ns()
    return result


def summarize(samples, interval):
    issues = [issue for row in samples for issue in row["issues"]]
    gaps = [
        (b["started_ns"] - a["started_ns"]) / 1e9 for a, b in pairwise(samples)
    ]
    if len(samples) < 2:
        issues.append({"type": "insufficient_samples"})
    if gaps and max(gaps) > interval * 3:
        issues.append({"type": "sampling_gap", "max_seconds": max(gaps)})
    gpu = [row["gpu"] for row in samples if "gpu" in row]
    return {
        "schema_version": 1,
        "sampled_isolation_checks_passed": not issues,
        "exclusive_gpu_claim": False,
        "sampling_limit": "short-lived work between samples may be missed",
        "samples": len(samples),
        "issues": issues,
        "first_sample_started_ns": samples[0]["started_ns"] if samples else None,
        "last_sample_completed_ns": samples[-1]["completed_ns"] if samples else None,
        "max_sample_gap_seconds": max(gaps) if gaps else None,
        "peak_used_mib": max(row["used_mib"] for row in gpu) if gpu else None,
        "min_free_mib": min(row["free_mib"] for row in gpu) if gpu else None,
        "mean_sampled_utilization_percent": (
            sum(row["utilization_percent"] for row in gpu) / len(gpu) if gpu else None
        ),
    }


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument(
        "--allowed-pid",
        type=int,
        action="append",
        required=True,
        help="GPU worker PID; repeat for each worker, not HTTP-only PIDs",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        not args.gpu_uuid.startswith("GPU-")
        or any(pid <= 0 for pid in args.allowed_pid)
        or len(set(args.allowed_pid)) != len(args.allowed_pid)
        or not 0.25 <= args.interval <= 60
        or not args.interval * 2 <= args.duration <= 3600
        or args.output.exists()
        or (args.stop_file and args.stop_file.exists())
    ):
        parser.error("invalid GPU/PID, interval/duration, or output/stop file exists")
    identities = {pid: process_stats(pid)["start_ticks"] for pid in args.allowed_pid}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": 1,
            "gpu_uuid": args.gpu_uuid,
            "allowed_pid_start_ticks": identities,
            "created_at": datetime.now(UTC).isoformat(),
            "interval_seconds": args.interval,
            "max_duration_seconds": args.duration,
            "stop_file": str(args.stop_file) if args.stop_file else None,
            "clock": "perf_counter_ns on benchmark host",
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "cpu_ticks_per_second": os.sysconf("SC_CLK_TCK"),
            "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    samples = []
    deadline = time.monotonic() + args.duration
    reason = "duration_limit"
    try:
        with (args.output / "samples.jsonl").open("x") as output:
            while True:
                row = sample(args.gpu_uuid, identities)
                samples.append(row)
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                if args.stop_file and args.stop_file.exists():
                    reason = "stop_file"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                query_seconds = (row["completed_ns"] - row["started_ns"]) / 1e9
                time.sleep(min(remaining, max(0, args.interval - query_seconds)))
    except KeyboardInterrupt:
        reason = "interrupted"
    report = summarize(samples, args.interval)
    report["stop_reason"] = reason
    if reason == "interrupted":
        report["issues"].append({"type": "interrupted"})
        report["sampled_isolation_checks_passed"] = False
    write_json(args.output / "summary.json", report)
    print(json.dumps(report))
    return 0 if report["sampled_isolation_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
