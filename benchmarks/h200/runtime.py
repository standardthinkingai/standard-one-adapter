"""Launch one pinned stock SGLang worker, or probe its public HTTP contract.

The helper itself uses the standard library. The engine runs in its own Python
environment. Nothing downloads or launches unless --execute is supplied.
"""

import argparse
import base64
import datetime
import json
import math
import os
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

MANIFEST = Path(__file__).with_name("models.json")


def load_manifest():
    return json.loads(MANIFEST.read_text())


def launch_spec(manifest, key, python, gpu, port, linear_attn_prefill_backend=None):
    if linear_attn_prefill_backend is not None and key != "qwen35-4b-bf16":
        raise ValueError("the FlashInfer prefill experiment is for qwen35-4b-bf16")
    model = manifest["models"][key]
    defaults = manifest["defaults"]
    command = [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model["repo_id"],
        "--revision",
        model["revision"],
        "--served-model-name",
        defaults["served_model_name"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--model-impl",
        "sglang",
        "--dtype",
        model["dtype"],
        "--context-length",
        str(defaults["context_length"]),
        "--max-running-requests",
        str(defaults["max_running_requests"]),
        "--mem-fraction-static",
        str(defaults["mem_fraction_static"]),
        "--chunked-prefill-size",
        "-1",
        "--disable-radix-cache",
        "--mm-preprocess-cache-size-mb",
        "0",
    ]
    if model["quantization"]:
        command += ["--quantization", model["quantization"]]
    command += model["extra_engine_args"]
    if linear_attn_prefill_backend is not None:
        command += [
            "--linear-attn-prefill-backend",
            linear_attn_prefill_backend,
        ]
    environment = dict(defaults["environment"], CUDA_VISIBLE_DEVICES=str(gpu))
    # The engine launches helper executables (e.g. ninja) from its own venv.
    environment["PATH"] = (
        str(Path(python).absolute().parent) + os.pathsep + os.environ.get("PATH", "")
    )
    return command, environment


def validate_engine_install(info, engine):
    origin = info["sglang_origin"]
    hashes = origin.get("archive_info", {}).get("hashes", {})
    # uv verifies the install_spec URL hash but omits it from direct_url.json.
    recorded_hash = hashes.get("sha256")
    if (
        info["sglang_version"] != engine["version"]
        or origin.get("url") != engine["wheel_url"]
        or (recorded_hash is not None and recorded_hash != engine["wheel_sha256"])
    ):
        raise ValueError(
            "Install the official SGLang wheel with the pinned install_spec."
        )


def inspect_runtime(python, environment, manifest):
    code = """
import importlib.metadata as m, json, platform, torch
d=m.distribution('sglang')
print(json.dumps({
 'python':platform.python_version(), 'platform':platform.platform(),
 'sglang_version':d.version,
 'sglang_origin':json.loads(d.read_text('direct_url.json') or '{}'),
 'packages':{x:m.version(x) for x in ('torch','transformers','sglang-kernel','flashinfer-python','mistral-common')},
 'cuda':torch.version.cuda,
 'gpus':[{'name':torch.cuda.get_device_name(i),
          'memory_bytes':torch.cuda.get_device_properties(i).total_memory,
          'capability':list(torch.cuda.get_device_capability(i))}
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
    info = json.loads(result.stdout)
    validate_engine_install(info, manifest["engine"])
    if len(info["gpus"]) != 1 or "H200" not in info["gpus"][0]["name"]:
        raise ValueError("Select exactly one H200 with --gpu (CUDA_VISIBLE_DEVICES).")
    if info["gpus"][0]["capability"] != [9, 0]:
        raise ValueError("Expected H200 compute capability 9.0.")
    if not str(info["cuda"]).startswith(str(manifest["engine"]["cuda_major"]) + "."):
        raise ValueError("The pinned engine profile requires a CUDA 13 runtime.")
    for package in ("torch", "transformers"):
        if info["packages"][package].split("+")[0] != manifest["engine"][package]:
            raise ValueError(f"Unexpected {package} version in the engine environment.")
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version,power.limit",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    info["nvidia_smi"] = smi.stdout
    return info


def request_json(base, path, payload=None, timeout=60):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("SGLANG_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["SGLANG_API_KEY"]
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    value = json.loads(body) if body else {}
    if not isinstance(value, dict) or "error" in value:
        raise ValueError(f"Invalid response from {path}")
    return value


def validate_server(
    info,
    model_info,
    manifest,
    key,
    probe_vision=False,
    linear_attn_prefill_backend=None,
):
    model = manifest["models"][key]
    checks = {
        "model_path": model["repo_id"],
        "revision": model["revision"],
        "tp_size": 1,
        "disable_radix_cache": True,
        "mm_preprocess_cache_size_mb": 0,
        "enable_prefix_mm_cache": False,
        "enable_mm_global_cache": False,
        "speculative_algorithm": None,
        "quantization": model["quantization"],
        "dtype": model["dtype"],
        "max_running_requests": manifest["defaults"]["max_running_requests"],
        "context_length": manifest["defaults"]["context_length"],
    }
    for field, expected in checks.items():
        if field not in info or info[field] != expected:
            raise ValueError(
                f"Server {field}={info.get(field)!r}; expected {expected!r}"
            )
    if (
        linear_attn_prefill_backend is not None
        and info.get("linear_attn_prefill_backend") != linear_attn_prefill_backend
    ):
        raise ValueError(
            "Server linear_attn_prefill_backend="
            f"{info.get('linear_attn_prefill_backend')!r}; "
            f"expected {linear_attn_prefill_backend!r}"
        )
    if model_info.get("served_model_name") != manifest["defaults"]["served_model_name"]:
        raise ValueError("The engine serves a different benchmark profile.")
    if model_info.get("model_path") != model["repo_id"]:
        raise ValueError("The active model differs from the requested checkpoint.")
    if model_info.get("is_generation") is not True:
        raise ValueError("The engine must use generation mode.")
    if probe_vision and model_info.get("has_image_understanding") is not True:
        raise ValueError("The model's vision path is disabled.")
    if model["architecture"] not in model_info.get("architectures", []):
        raise ValueError("The engine reports a different model architecture.")


def probe_image():
    """Generate a 224x224 red PNG fixture without external image dependencies."""

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    pixels = (b"\x00" + b"\xff\x00\x00" * 224) * 224
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 224, 224, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def probe_score(base, key, image, timeout, native_tokenizer=None):
    prompt = "Choose exactly one label. A: red. B: blue. Answer:"
    if native_tokenizer is not None:
        if image is not None:
            raise ValueError(
                "Native official tokenizer probes currently accept text only."
            )
        batches = request_json(
            base,
            "/v1/tokenize",
            {"model": key, "prompt": ["A", "B"], "add_special_tokens": False},
            timeout,
        )["tokens"]
        if len(batches) != 2 or any(len(tokens) != 1 for tokens in batches):
            raise ValueError("Expected single-token A/B labels.")
        label_ids = [tokens[0] for tokens in batches]
        ids = native_tokenizer.prepare(prompt, ("A", "B"), tuple(label_ids), None)
        text = None
    else:
        content = (
            prompt
            if image is None
            else [
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": prompt},
            ]
        )
        rendered = request_json(
            base,
            "/v1/tokenize",
            {
                "model": key,
                "messages": [{"role": "user", "content": content}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout,
        )
        ids = rendered["tokens"]
        text = request_json(
            base,
            "/v1/detokenize",
            {
                "model": key,
                "tokens": ids,
                "skip_special_tokens": False,
            },
            timeout,
        )["text"]
        tokens = request_json(
            base,
            "/v1/tokenize",
            {
                "model": key,
                "prompt": [text, text + "A", text + "B"],
                "add_special_tokens": False,
            },
            timeout,
        )["tokens"]
        if tokens[0] != ids or any(t[:-1] != ids for t in tokens[1:]):
            raise ValueError("Chat roundtrip / single-token answer boundary failed.")
        label_ids = [t[-1] for t in tokens[1:]]
    if len(set(label_ids)) != 2:
        raise ValueError("Answer labels are not distinct.")
    payload = {
        "sampling_params": {
            "max_new_tokens": 0,
            "temperature": 1,
            "top_p": 1,
            "top_k": -1,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
        "token_ids_logprob": label_ids,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
        "stream": False,
    }
    payload.update(
        {"input_ids": ids}
        if image is None
        else {
            "text": text,
            "image_data": [image],
        }
    )
    response = request_json(base, "/generate", payload, timeout)
    meta = response.get("meta_info", {})
    if type(meta.get("completion_tokens")) is not int or meta["completion_tokens"] != 0:
        raise ValueError("The engine did not honor zero-output scoring.")
    if type(meta.get("prompt_tokens")) is not int or meta["prompt_tokens"] < 1:
        raise ValueError("The engine omitted prompt token usage.")
    if meta.get("cached_tokens", 0) != 0:
        raise ValueError("Unexpected prefix cache reuse in a full-prefill run.")
    positions = meta.get("output_token_ids_logprobs", [])
    if len(positions) != 1:
        raise ValueError("The engine omitted next-token logprobs.")
    values = {row[1]: row[0] for row in positions[0]}
    if any(t not in values or not math.isfinite(values[t]) for t in label_ids):
        raise ValueError("Missing or non-finite requested-label logprob.")
    return {
        "prompt_tokens": meta["prompt_tokens"],
        "completion_tokens": 0,
        "cached_tokens": meta.get("cached_tokens"),
        "labels": label_ids,
        "logprobs": [values[t] for t in label_ids],
    }


def probe(
    base,
    manifest,
    key,
    timeout,
    probe_vision=False,
    linear_attn_prefill_backend=None,
):
    request_json(base, "/health_generate", timeout=timeout)
    model = request_json(base, "/model_info", timeout=timeout)
    server = request_json(base, "/server_info", timeout=timeout)
    validate_server(
        server,
        model,
        manifest,
        key,
        probe_vision=probe_vision,
        linear_attn_prefill_backend=linear_attn_prefill_backend,
    )
    served = manifest["defaults"]["served_model_name"]
    profile = manifest["models"][key]
    native_tokenizer = None
    if profile.get("tokenization") == "native_official_text":
        from jev_adapter.native_tokenizer import NativeTokenizer

        native_tokenizer = NativeTokenizer.from_pretrained(
            profile["repo_id"], profile["revision"]
        )
    text = probe_score(base, served, None, timeout, native_tokenizer)
    vision = None
    if probe_vision:
        vision = probe_score(base, served, probe_image(), timeout, native_tokenizer)
        if vision["prompt_tokens"] <= text["prompt_tokens"]:
            raise ValueError(
                "The image request did not increase the prompt token count."
            )
    return {
        "probe_scope": ["text", "vision"] if probe_vision else ["text"],
        "tokenization": profile.get("tokenization", "engine_http_chat"),
        "linear_attn_prefill_backend": linear_attn_prefill_backend,
        "model_info": sanitize_metadata(model),
        "server_info": sanitize_metadata(server),
        "text_probe": text,
        "vision_probe": vision,
        "note": "Readiness only; probe outputs are not accuracy/latency results.",
    }


def sanitize_metadata(value):
    if isinstance(value, dict):
        return {
            key: (
                "[redacted]"
                if any(
                    s in key.lower()
                    for s in (
                        "api_key",
                        "password",
                        "secret",
                        "access_token",
                        "authorization",
                        "launch_command",
                    )
                )
                else sanitize_metadata(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_metadata(item) for item in value]
    return value


def main(argv=None):
    manifest = load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "launch", "probe"])
    parser.add_argument("model", nargs="?", choices=manifest["models"])
    parser.add_argument("--engine-python", default=".venv-sglang/bin/python")
    parser.add_argument("--gpu", default="0", help="One physical GPU index or UUID")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--output", type=Path, help="New directory for launch metadata")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--linear-attn-prefill-backend",
        choices=("flashinfer",),
        help="Opt-in Qwen3.5-4B GDN prefill experiment; omission keeps the baseline",
    )
    parser.add_argument(
        "--probe-vision",
        action="store_true",
        help="Also require the image scoring path to pass readiness checks",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch the engine; otherwise print commands only",
    )
    args = parser.parse_args(argv)
    if args.action == "list":
        print(json.dumps(manifest, indent=2))
        return 0
    if not args.model:
        parser.error("model is required")
    if (
        not 1 <= args.port <= 65535
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("port and timeout must be positive and valid")
    if "," in args.gpu or not args.gpu:
        parser.error("--gpu must select exactly one device")
    if args.linear_attn_prefill_backend and args.model != "qwen35-4b-bf16":
        parser.error("--linear-attn-prefill-backend requires qwen35-4b-bf16")
    base = f"http://127.0.0.1:{args.port}"
    if args.action == "probe":
        print(
            json.dumps(
                probe(
                    base,
                    manifest,
                    args.model,
                    min(args.timeout, 60),
                    probe_vision=args.probe_vision,
                    linear_attn_prefill_backend=args.linear_attn_prefill_backend,
                ),
                indent=2,
            )
        )
        return 0
    command, overrides = launch_spec(
        manifest,
        args.model,
        args.engine_python,
        args.gpu,
        args.port,
        linear_attn_prefill_backend=args.linear_attn_prefill_backend,
    )
    print(
        shlex.join(["env", *[f"{k}={v}" for k, v in overrides.items()], *command]),
        flush=True,
    )
    if not args.execute:
        return 0
    if not args.output:
        parser.error("--execute requires --output pointing to a new directory")
    # Refuse to probe an older process that happens to occupy the requested port.
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", args.port))
    environment = dict(os.environ, **overrides)
    runtime = inspect_runtime(args.engine_python, environment, manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    record = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "profile": args.model,
        "probe_scope": ["text", "vision"] if args.probe_vision else ["text"],
        "model": manifest["models"][args.model],
        "engine": manifest["engine"],
        "command": command,
        "environment_overrides": overrides,
        "runtime": runtime,
    }
    (args.output / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    with (args.output / "engine.log").open("x") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(
            f"Engine PID {process.pid}; log: {args.output / 'engine.log'}", flush=True
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
                    request_json(base, "/health_generate", timeout=5)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(min(5, max(0, deadline - time.monotonic())))
            checks = probe(
                base,
                manifest,
                args.model,
                60,
                probe_vision=args.probe_vision,
                linear_attn_prefill_backend=args.linear_attn_prefill_backend,
            )
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
                        "probe_scope": checks["probe_scope"],
                        "ready_at": datetime.datetime.now(datetime.UTC).isoformat(),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"{' and '.join(checks['probe_scope'])} zero-output probes passed. "
                "Engine is running; "
                "start the adapter in another terminal. Ctrl-C stops this engine.",
                flush=True,
            )
            return process.wait()
        finally:
            (args.output / "ready.json").unlink(missing_ok=True)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


def handle_termination(_signum, _frame):
    # A matrix runner may use Popen.terminate(); still release our child engine.
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_termination)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"H200 preflight failed: {error}", file=sys.stderr)
        raise SystemExit(1)
