import importlib.util
import os
import select
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def test_matrix_preview_uses_four_bf16_models_without_starting(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "does-not-exist"
    engine_python = tmp_path / "engine-venv-python"
    engine_python.symlink_to(sys.executable)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/run_matrix.py"),
            "--output",
            str(output),
            "--include-external",
            "--concurrency",
            "8",
            "--engine-python",
            str(engine_python),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Preview only" in result.stdout
    assert "ministral3-3b-bf16" in result.stdout
    assert "ministral3-3b-fp8" not in result.stdout
    assert "qwen36-27b-bf16" in result.stdout
    assert "qwen36-35b-a3b-bf16" in result.stdout
    assert "qwen35-4b-bf16" in result.stdout
    assert "scienthoon-v1" in result.stdout
    assert "jevbench-hard/public.jsonl" in result.stdout
    assert "--probe-vision" not in result.stdout
    assert f"--engine-python {engine_python}" in result.stdout
    assert not output.exists()


def test_stop_owned_cleans_runtime_that_inherits_ignored_sigint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "benchmark_matrix_under_test", root / "benchmarks/run_matrix.py"
    )
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    cleanup_marker = tmp_path / "runtime-cleaned-up"
    child_code = """
import signal
import sys
from pathlib import Path

# A background launcher can inherit this disposition from its parent shell.
signal.signal(signal.SIGINT, signal.SIG_IGN)

def terminate(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, terminate)
try:
    print("ready", flush=True)
    while True:
        signal.pause()
except KeyboardInterrupt:
    pass
finally:
    Path(sys.argv[1]).write_text("graceful SIGTERM cleanup")
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(cleanup_marker)],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        readable, _, _ = select.select([process.stdout], [], [], 5)
        assert readable, "child did not install its signal handlers"
        assert process.stdout.readline().strip() == "ready"
        # Bound regressions without replacing real OS signal delivery or wait().
        with ThreadPoolExecutor(max_workers=1) as executor:
            stopped = executor.submit(matrix.stop_owned, process)
            try:
                stopped.result(timeout=5)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        assert process.returncode == 0
        assert cleanup_marker.read_text() == "graceful SIGTERM cleanup"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()
