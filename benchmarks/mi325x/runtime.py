"""One MI325X, official BF16 checkpoints, stock ROCm SGLang; preview by default.

Only --execute permits Docker/network/runtime inspection or a model launch.
The H200 module supplies platform-neutral command and HTTP helpers only.
"""
# ruff: noqa: UP017 -- host inventory/image locking also supports Python 3.9/3.10

import argparse
import datetime
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

MANIFEST = Path(__file__).with_name("models.json")
_spec = importlib.util.spec_from_file_location(
    "jev_common_runtime", MANIFEST.parent.parent / "h200/runtime.py"
)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)


def load_manifest():
    return json.loads(MANIFEST.read_text())


def visibility_overrides(gpu, inherited):
    """ROCr selects a GPU; HIP subsequently sees it at index zero.

    An existing scheduler/container allocation is preserved, never widened.
    Ambiguous masks are rejected instead of silently clearing environment vars.
    """
    if not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", str(gpu)):
        raise ValueError("--gpu requires one nonnegative index or AMD GPU UUID")
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_OVERRIDE_GFX_VERSION",
    ):
        if name in inherited:
            raise ValueError(f"Ambiguous inherited {name}; use only a ROCr allocation")
    rocr = inherited.get("ROCR_VISIBLE_DEVICES")
    hip = inherited.get("HIP_VISIBLE_DEVICES")
    if rocr is not None:
        if not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", rocr) or gpu != "0":
            raise ValueError("Existing ROCr mask must contain one GPU; use --gpu 0")
        if hip not in (None, "0"):
            raise ValueError(
                "HIP mask must be zero inside the existing ROCr allocation"
            )
        return {"ROCR_VISIBLE_DEVICES": rocr, "HIP_VISIBLE_DEVICES": "0"}
    if hip is not None:
        raise ValueError(
            "Use an explicit ROCr allocation instead of an inherited HIP mask"
        )
    return {"ROCR_VISIBLE_DEVICES": gpu, "HIP_VISIBLE_DEVICES": "0"}


def launch_spec(manifest, key, python, gpu, port, inherited=None):
    model = manifest["models"][key]
    if model["dtype"] != "bfloat16" or model["quantization"] is not None:
        raise ValueError("Only official BF16 checkpoints are allowed in this baseline")
    allowed = {"--model-config-parser", "hf", "--load-format", "safetensors"}
    if any(arg not in allowed for arg in model["extra_engine_args"]):
        raise ValueError("Unexpected engine arguments in the baseline profile")
    command, _ = common.launch_spec(manifest, key, python, gpu, port)
    inherited = os.environ if inherited is None else inherited
    overrides = dict(manifest["defaults"]["environment"])
    overrides.update(visibility_overrides(gpu, inherited))
    overrides["PATH"] = (
        str(Path(python).absolute().parent) + os.pathsep + inherited.get("PATH", "")
    )
    return command, overrides


def validate_image_lock(lock, manifest):
    tag = manifest["engine"]["image_tag"]
    repository = tag.rsplit(":", 1)[0]
    if (
        lock.get("schema_version") != 1
        or lock.get("image_tag") != tag
        or not re.fullmatch(
            re.escape(repository) + r"@sha256:[a-f0-9]{64}",
            lock.get("image_ref", ""),
        )
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", lock.get("image_id", ""))
        or lock.get("os") != "linux"
        or lock.get("architecture") != "amd64"
        or lock.get("verification") != "docker_pull_and_digest_inspect"
    ):
        raise ValueError(
            "Invalid image lock; run lock-image on the prepared Docker host"
        )
    return lock


def lock_image(manifest):
    tag = manifest["engine"]["image_tag"]
    subprocess.run(["docker", "pull", "--platform", "linux/amd64", tag], check=True)
    info = json.loads(
        subprocess.check_output(["docker", "image", "inspect", tag], text=True)
    )[0]
    refs = [
        ref
        for ref in info.get("RepoDigests", [])
        if ref.startswith(tag.rsplit(":", 1)[0] + "@sha256:")
    ]
    if len(refs) != 1:
        raise ValueError("Docker did not return one official repository digest")
    pinned = json.loads(
        subprocess.check_output(["docker", "image", "inspect", refs[0]], text=True)
    )[0]
    if pinned["Id"] != info["Id"]:
        raise ValueError(
            "The pulled tag and immutable digest identify different images"
        )
    return validate_image_lock(
        {
            "schema_version": 1,
            "image_tag": tag,
            "image_ref": refs[0],
            "image_id": info["Id"],
            "os": info["Os"],
            "architecture": info["Architecture"],
            "verification": "docker_pull_and_digest_inspect",
            "resolved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        manifest,
    )


def verify_container(lock, manifest, environment):
    validate_image_lock(lock, manifest)
    if environment.get("JEV_CONTAINER_IMAGE") != lock["image_ref"]:
        raise ValueError("Launch the digest-pinned container with JEV_CONTAINER_IMAGE")
    if not Path("/.dockerenv").exists():
        raise ValueError("Run this profile inside the prepared official ROCm container")


def host_preflight():
    """Read-only host inventory: no package, driver, service or BIOS mutations."""
    report = {
        "platform": platform.platform(),
        "os_release": Path("/etc/os-release").read_text()
        if Path("/etc/os-release").exists()
        else None,
        "kfd_present": Path("/dev/kfd").exists(),
        "render_devices": sorted(str(p) for p in Path("/dev/dri").glob("renderD*")),
        "render_pci_devices": {
            p.name: str((p / "device").resolve())
            for p in Path("/sys/class/drm").glob("renderD*")
        },
        "commands": {},
    }
    for command in (
        ["docker", "version", "--format", "{{json .}}"],
        ["docker", "info", "--format", "{{json .DriverStatus}}"],
        ["amd-smi", "static", "--json"],
        ["rocm-smi", "--showproductname", "--showdriverversion", "--showuniqueid"],
        ["modinfo", "amdgpu"],
        ["systemctl", "--version"],
        ["apt-cache", "policy", "docker.io"],
        ["dnf", "--cacheonly", "info", "--installed", "docker-ce"],
    ):
        if not shutil.which(command[0]):
            continue
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        report["commands"][shlex.join(command)] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    return report


def validate_runtime(info, manifest):
    if not info.get("hip") or info.get("cuda") is not None:
        raise ValueError("Expected ROCm/HIP PyTorch, not a CUDA or CPU-only build")
    if not str(info["hip"]).startswith(manifest["engine"]["rocm_series"] + "."):
        raise ValueError("The selected image requires the ROCm 7.2 runtime series")
    if info.get("sglang_version") != manifest["engine"]["version"]:
        raise ValueError("Unexpected SGLang version; do not mix wheels into the image")
    gpus = info.get("gpus", [])
    if len(gpus) != 1 or "MI325X" not in gpus[0].get("name", "").upper():
        raise ValueError("Expected exactly one visible AMD Instinct MI325X")
    if gpus[0].get("gcn_arch", "").split(":")[0] != "gfx942":
        raise ValueError("Expected the MI325X gfx942 architecture")
    if gpus[0].get("memory_bytes", 0) < 240 * 1024**3:
        raise ValueError("Expected an unpartitioned MI325X with approximately 256 GiB")
    return info


def inspect_runtime(python, environment, manifest):
    code = """
import importlib.metadata as m, json, platform, torch
if not torch.version.hip or torch.version.cuda is not None:
    raise RuntimeError('Expected ROCm/HIP PyTorch before enumerating GPUs')
def version(name):
    try: return m.version(name)
    except m.PackageNotFoundError: return None
print(json.dumps({
 'python':platform.python_version(), 'platform':platform.platform(),
 'sglang_version':version('sglang'),
 'packages':{x:version(x) for x in ('torch','transformers','sglang-kernel',
                                   'triton','aiter','mistral-common')},
 'hip':torch.version.hip, 'cuda':torch.version.cuda,
 'gpus':[{'name':torch.cuda.get_device_name(i),
          'memory_bytes':torch.cuda.get_device_properties(i).total_memory,
          'gcn_arch':getattr(torch.cuda.get_device_properties(i),'gcnArchName',''),
          'uuid':str(getattr(torch.cuda.get_device_properties(i),'uuid',''))}
         for i in range(torch.cuda.device_count())]}))
"""
    result = subprocess.run(
        [python, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return validate_runtime(json.loads(result.stdout), manifest)


def validate_baseline(server):
    for field in ("enable_lora", "lora_paths", "speculative_algorithm"):
        if field not in server:
            raise ValueError(f"Server omitted baseline evidence: {field}")
    if server["enable_lora"] not in (False, None) or server["lora_paths"] not in (
        None,
        [],
        {},
    ):
        raise ValueError("LoRA is forbidden in the untrained baseline")
    if server["speculative_algorithm"] is not None or server.get(
        "speculative_draft_model_path"
    ):
        raise ValueError("Speculative decoding/MTP is forbidden in the baseline")
    if server.get("chunked_prefill_size") != -1:
        raise ValueError("Chunked prefill must be disabled")


def probe(base, manifest, key, timeout, probe_vision=False):
    server = common.request_json(base, "/server_info", timeout=timeout)
    validate_baseline(server)
    return common.probe(base, manifest, key, timeout, probe_vision=probe_vision)


def stop_owned(process):
    # A failed group leader can leave workers alive. Its group is still ours.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def launch(args, manifest, command, overrides, image_lock):
    environment = dict(os.environ, **overrides)
    verify_container(image_lock, manifest, environment)
    # Refuse to attach to an unrelated service using the selected benchmark port.
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", args.port))
    runtime = inspect_runtime(args.engine_python, environment, manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    scope = ["text", "vision"] if args.probe_vision else ["text"]
    record = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "profile": args.model,
        "probe_scope": scope,
        "model": manifest["models"][args.model],
        "engine": dict(
            manifest["engine"],
            revision=image_lock["image_ref"],
            revision_kind="oci_image_digest",
        ),
        "image_lock": image_lock,
        "command": command,
        "environment_overrides": overrides,
        "runtime": runtime,
        "image_environment": common.sanitize_metadata(
            {
                k: v
                for k, v in environment.items()
                if k.startswith(("SGLANG_", "HIP_", "ROCR_", "HSA_"))
            }
        ),
    }
    (args.output / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    base = f"http://127.0.0.1:{args.port}"
    with (args.output / "engine.log").open("x") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + args.timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Engine exited with status {process.returncode}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("Engine readiness deadline exceeded")
                try:
                    common.request_json(base, "/health_generate", timeout=5)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(min(5, max(0, deadline - time.monotonic())))
            checks = probe(base, manifest, args.model, 60, args.probe_vision)
            (args.output / "preflight.json").write_text(
                json.dumps(checks, indent=2) + "\n"
            )
            (args.output / "ready.json").write_text(
                json.dumps(
                    {
                        "profile": args.model,
                        "pid": process.pid,
                        "engine_url": base,
                        "model": manifest["defaults"]["served_model_name"],
                        "probe_scope": scope,
                        "ready_at": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                "MI325X text scoring ready; Ctrl-C stops the owned engine.", flush=True
            )
            return process.wait()
        finally:
            (args.output / "ready.json").unlink(missing_ok=True)
            stop_owned(process)


def main(argv=None):
    manifest = load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["list", "host-preflight", "lock-image", "launch", "probe"]
    )
    parser.add_argument("model", nargs="?", choices=manifest["models"])
    parser.add_argument("--engine-python", default=sys.executable)
    parser.add_argument(
        "--gpu", default="0", help="One ROCr GPU; zero inside a one-GPU allocation"
    )
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-lock", type=Path)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--probe-vision", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "list":
        print(json.dumps(manifest, indent=2))
        return 0
    if (
        not 1 <= args.port <= 65535
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("port and timeout must be positive and valid")
    if args.action in ("host-preflight", "lock-image"):
        if not args.execute:
            print(
                f"Preview: {args.action}; add --execute for read-only host checks or image pull"
            )
            return 0
        if not args.output or args.output.exists():
            parser.error("--execute requires --output pointing to a new JSON file")
        result = (
            host_preflight()
            if args.action == "host-preflight"
            else lock_image(manifest)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")
        return 0
    if not args.model:
        parser.error("model is required")
    if (
        args.probe_vision
        and manifest["models"][args.model]["tokenization"] == "native_official_text"
    ):
        parser.error("The native official Ministral tokenizer path supports text only")
    if args.action == "probe":
        if args.execute:
            print(
                json.dumps(
                    probe(
                        f"http://127.0.0.1:{args.port}",
                        manifest,
                        args.model,
                        min(args.timeout, 60),
                        args.probe_vision,
                    ),
                    indent=2,
                )
            )
        else:
            print(
                "Preview: public HTTP scoring probe; add --execute to contact the engine"
            )
        return 0
    command, overrides = launch_spec(
        manifest, args.model, args.engine_python, args.gpu, args.port
    )
    print(
        shlex.join(["env", *[f"{k}={v}" for k, v in overrides.items()], *command]),
        flush=True,
    )
    if not args.execute:
        return 0
    if not args.output or args.output.exists() or not args.image_lock:
        parser.error("--execute requires a new --output directory and --image-lock")
    image_lock = validate_image_lock(json.loads(args.image_lock.read_text()), manifest)
    return launch(args, manifest, command, overrides, image_lock)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, common.handle_termination)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"MI325X preflight failed: {error}", file=sys.stderr)
        raise SystemExit(1)
