#!/usr/bin/env python3
"""Run one pinned model at a time on one H200. Default is a command preview."""

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "benchmarks/h200/runtime.py"


def commands(args, manifest, profile):
    root = args.output.resolve() / profile
    engine = [
        sys.executable,
        str(RUNTIME),
        "launch",
        profile,
        "--engine-python",
        # Resolving a venv Python symlink loses pyvenv.cfg and its installed engine.
        str(args.engine_python.absolute()),
        "--gpu",
        args.gpu,
        "--output",
        str(root / "engine"),
        "--timeout",
        str(args.startup_timeout),
        "--execute",
    ]
    if args.probe_vision:
        engine.append("--probe-vision")
    if args.linear_attn_prefill_backend:
        engine += [
            "--linear-attn-prefill-backend",
            args.linear_attn_prefill_backend,
        ]
    adapter = [
        sys.executable,
        "-m",
        "jev_adapter",
        "--engine-url",
        "http://127.0.0.1:30000",
        "--model",
        "decision-model",
        "--port",
        "30120",
        "--max-concurrency",
        "32",
    ]
    model = manifest["models"][profile]
    if model.get("tokenization") == "native_official_text" or (
        args.qwen_local_tokenizer and profile == "qwen35-4b-bf16"
    ):
        adapter += [
            "--tokenizer-model",
            model["repo_id"],
            "--tokenizer-revision",
            model["revision"],
        ]
    suites = [
        "decision-v7",
        "transfer-v4",
        "transfer-v9",
        "jevbench-original",
        "jevbench-easy",
        "jevbench-hard",
    ]
    if args.suite:
        suites = args.suite
    if args.include_external:
        suites += [
            suite for suite in ["semif-v1", "scienthoon-v1"] if suite not in suites
        ]
    evaluations = []
    for concurrency in args.concurrency:
        run = [
            sys.executable,
            "-m",
            "jev_adapter.benchmarks.run",
            "--output",
            str(root / f"c{concurrency}"),
            "--engine-manifest",
            str(root / "engine/launch.json"),
            "--concurrency",
            str(concurrency),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
        ]
        for suite in suites:
            split = "public" if suite.startswith("jevbench-") else "development"
            run += [
                "--data",
                str(args.data_root.resolve() / suite / f"{split}.jsonl"),
            ]
        if args.limit is not None:
            run += ["--limit", str(args.limit)]
        evaluations.append(run)
    return {
        "profile": profile,
        "model": manifest["models"][profile],
        "engine": engine,
        "adapter": adapter,
        "evaluations": evaluations,
    }


def stop_owned(process):
    if process is not None and process.poll() is None:
        # SIGINT can be inherited as ignored under nohup; the runtime handles TERM.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=40)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Owned process {process.pid} did not stop; check its log."
            ) from exc


def wait_engine(process, ready_path, timeout):
    deadline = time.monotonic() + timeout
    while not ready_path.exists():
        if process.poll() is not None:
            raise RuntimeError(
                "Engine launcher exited; see launcher.log and engine/engine.log"
            )
        if time.monotonic() > deadline:
            raise TimeoutError("Engine startup deadline exceeded")
        time.sleep(1)


def wait_adapter(process, timeout=60):
    import httpx

    deadline = time.monotonic() + timeout
    headers = {}
    if key := os.environ.get("JEV_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(timeout=2) as client:
        while True:
            if process.poll() is not None:
                raise RuntimeError("Adapter startup failed; see adapter.log")
            try:
                response = client.get(
                    "http://127.0.0.1:30120/v1/models", headers=headers
                )
                if response.is_success:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Adapter startup deadline exceeded")
            time.sleep(1)


def main():
    manifest = json.loads((RUNTIME.parent / "models.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--engine-python", type=Path, default=ROOT / ".venv-sglang/bin/python"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "benchmarks/data")
    parser.add_argument("--profile", action="append", choices=list(manifest["models"]))
    parser.add_argument("--concurrency", type=int, action="append")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--include-external", action="store_true")
    parser.add_argument(
        "--suite",
        action="append",
        choices=[
            "decision-v7",
            "transfer-v4",
            "transfer-v9",
            "jevbench-original",
            "jevbench-easy",
            "jevbench-hard",
            "semif-v1",
            "scienthoon-v1",
        ],
    )
    parser.add_argument(
        "--probe-vision", action="store_true", help="optional image readiness probe"
    )
    parser.add_argument(
        "--linear-attn-prefill-backend",
        choices=("flashinfer",),
        help="Opt-in Qwen3.5-4B GDN prefill experiment; omission keeps the baseline",
    )
    parser.add_argument(
        "--qwen-local-tokenizer",
        action="store_true",
        help="Opt in to the pinned local Qwen3.5-4B tokenizer instead of engine HTTP tokenization",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.concurrency = args.concurrency or [1]
    if (
        any(c < 1 or c > 32 for c in args.concurrency)
        or len(set(args.concurrency)) != len(args.concurrency)
        or args.warmup < 0
        or args.repeats < 1
        or args.startup_timeout < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("invalid counts; concurrency must be unique values from 1 to 32")
    profiles = args.profile or manifest["default_matrix"]
    if len(set(profiles)) != len(profiles):
        parser.error("profiles must be unique")
    if args.linear_attn_prefill_backend and profiles != ["qwen35-4b-bf16"]:
        parser.error("--linear-attn-prefill-backend requires only --profile qwen35-4b-bf16")
    if args.qwen_local_tokenizer and profiles != ["qwen35-4b-bf16"]:
        parser.error("--qwen-local-tokenizer requires only --profile qwen35-4b-bf16")
    plans = [commands(args, manifest, profile) for profile in profiles]
    for plan in plans:
        print(f"\n{plan['profile']}")
        for command in [plan["engine"], plan["adapter"], *plan["evaluations"]]:
            print(shlex.join(command))
    if not args.execute:
        print("\nPreview only. Add --execute on the prepared H200 server.")
        return
    if not args.engine_python.is_file():
        parser.error("engine Python is missing; follow benchmarks/h200/README.md")
    for plan in plans:
        for command in plan["evaluations"]:
            for i, value in enumerate(command):
                if value == "--data" and not Path(command[i + 1]).is_file():
                    parser.error(f"missing prepared dataset: {command[i + 1]}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "matrix.json").write_text(json.dumps(plans, indent=2) + "\n")
    for plan in plans:
        # Never attach to or stop unrelated services using the benchmark ports.
        for port in (30000, 30120):
            with socket.socket() as listener:
                # Match server bind semantics: ignore TIME_WAIT, reject live listeners.
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", port))
        root = args.output.resolve() / plan["profile"]
        root.mkdir()
        engine, adapter = None, None
        with (
            (root / "launcher.log").open("x") as engine_log,
            (root / "adapter.log").open("x") as adapter_log,
        ):
            try:
                engine = subprocess.Popen(
                    plan["engine"],
                    cwd=ROOT,
                    stdout=engine_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                wait_engine(
                    engine, root / "engine/ready.json", args.startup_timeout + 90
                )
                adapter = subprocess.Popen(
                    plan["adapter"],
                    cwd=ROOT,
                    stdout=adapter_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                wait_adapter(adapter)
                for command in plan["evaluations"]:
                    subprocess.run(command, cwd=ROOT, check=True)
            finally:
                try:
                    stop_owned(adapter)
                finally:
                    stop_owned(engine)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jev_adapter.benchmarks.compare",
            str(args.output.resolve()),
            "--output",
            str(args.output.resolve() / "comparison.csv"),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
