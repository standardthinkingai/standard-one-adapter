"""Collect complete benchmark reports into a comparison table, one row per suite."""

import argparse
import csv
import json
from pathlib import Path


def collect(root):
    rows = []
    for path in sorted(root.rglob("report.json")):
        if path.with_name("invalidated.json").exists():
            continue
        report = json.loads(path.read_text())
        if report.get("status") != "complete" or not report.get("suites"):
            continue
        manifest = json.loads(path.with_name("manifest.json").read_text())
        launch = manifest.get("launch_manifest", {})
        engine_model = (
            manifest.get("engine", {}).get("model_info", {}).get("model_path")
        )
        engine_revision = (
            manifest.get("engine", {}).get("server_info", {}).get("revision")
        )
        for suite, result in report["suites"].items():
            clean = result.get("clean")
            if clean is None:
                continue
            latency = result["latency_ms"]
            rows.append(
                {
                    "profile": launch.get("profile", manifest["requested_model"]),
                    "model_repo": launch.get("model", {}).get("repo_id", engine_model),
                    "model_revision": launch.get("model", {}).get(
                        "revision", engine_revision
                    ),
                    "suite": suite,
                    "concurrency": manifest["concurrency"],
                    "partial_dataset": report["partial_dataset"],
                    "clean_questions": clean["n"],
                    "accuracy": clean["acc"],
                    "brier": clean["brier"],
                    "ece": clean["ece"],
                    "confident_error_rate": clean["confident_error_rate"],
                    "p50_ms": latency["p50"],
                    "p95_ms": latency["p95"],
                    "p99_ms": latency["p99"],
                    "cache_mode": manifest["cache_mode"],
                    "report": str(path.resolve()),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root)
    if not rows:
        parser.error("no complete benchmark reports found; failed runs are not ranked")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} suite rows to {args.output}; latency is HTTP wall time.")


if __name__ == "__main__":
    main()
