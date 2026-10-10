"""Check exact Qwen prompt IDs for engine HTTP and local tokenizer paths.

Run against the same pinned SGLang engine used by both adapter services. This
is a correctness preflight, not a latency benchmark: it intentionally calls
the private preparation methods that the two serving paths use, without
calling /generate. Use prefill_compare.py for paired response/latency results.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

if __package__:
    from .prefill_compare import request_for
else:
    from prefill_compare import request_for

from jev_adapter.benchmarks.metrics import latency_summary
from jev_adapter.benchmarks.run import load_data, write_json
from jev_adapter.native_tokenizer import NativeTokenizer
from jev_adapter.protocol import (
    SystemOneRequest,
    build_prompt,
    plan_question,
    rotation_order,
)
from jev_adapter.sglang import SGLangBackend

PROFILE = json.loads(
    (Path(__file__).parent / "h200/models.json").read_text()
)["models"]["qwen35-4b-bf16"]


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def token_difference(left, right):
    return next(
        (index for index, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)) if len(left) != len(right) else None,
    )


async def compare(args):
    items, sources = load_data(
        args.data, args.model, allow_test=False, limit=args.limit
    )
    if any(item["record"].get("images") for item in items):
        raise ValueError("this exact-token check accepts text-only records")

    native_tokenizer = NativeTokenizer.from_pretrained(
        PROFILE["repo_id"], PROFILE["revision"]
    )
    native_system_prompt = (
        NativeTokenizer.native_default_system_prompt(
            PROFILE["repo_id"], PROFILE["revision"]
        )
        if args.prompt_wording == "native" and args.native_system_prompt == "auto"
        else None
    )
    base_kwargs = {
        "base_url": args.engine_url,
        "model": args.model,
        "api_key": os.environ.get("SGLANG_API_KEY"),
        "timeout": args.timeout,
        "prompt_wording": args.prompt_wording,
        "native_system_prompt": native_system_prompt,
    }
    http_backend = SGLangBackend(**base_kwargs)
    local_backend = SGLangBackend(
        **base_kwargs, native_tokenizer=native_tokenizer
    )
    try:
        await http_backend.start()
        await local_backend.start()
        info = http_backend._model_info
        server = http_backend._server_info
        if info.get("model_path") != PROFILE["repo_id"]:
            raise ValueError("engine model_path differs from pinned Qwen profile")
        if server.get("revision") != PROFILE["revision"]:
            raise ValueError("engine revision differs from pinned Qwen profile")
        if (http_backend._label_strings, http_backend._label_ids) != (
            local_backend._label_strings,
            local_backend._label_ids,
        ):
            raise ValueError("adapter paths selected different label tokens")

        questions = 0
        mismatches = []
        http_prep_ms = []
        local_prep_ms = []
        for item in items:
            request = SystemOneRequest.model_validate(
                request_for(item, args.model, args.assistant_prefix)
            )
            for qid, question in request.questions.items():
                plan = plan_question(question)
                labels, token_ids = http_backend.labels(len(plan.option_names))
                for rotation in range(request.options.permutations):
                    order = rotation_order(len(labels), rotation)
                    prompt = build_prompt(
                        request.state,
                        plan,
                        labels,
                        order,
                        wording=args.prompt_wording,
                    )
                    started = time.perf_counter()
                    http_ids, _ = await http_backend._prepare_http(
                        prompt,
                        request.images,
                        labels,
                        token_ids,
                        request.options.assistant_prefix,
                        native_system_prompt,
                    )
                    http_prep_ms.append((time.perf_counter() - started) * 1000)
                    started = time.perf_counter()
                    local_ids = await local_backend._prepare_native(
                        prompt,
                        labels,
                        token_ids,
                        request.options.assistant_prefix,
                    )
                    local_prep_ms.append((time.perf_counter() - started) * 1000)
                    questions += 1
                    difference = token_difference(http_ids, local_ids)
                    if difference is not None:
                        mismatches.append(
                            {
                                "suite": item["suite"],
                                "id": item["id"],
                                "question": qid,
                                "rotation": rotation,
                                "first_different_position": difference,
                                "http_length": len(http_ids),
                                "local_length": len(local_ids),
                                "http_sha256": token_digest(http_ids),
                                "local_sha256": token_digest(local_ids),
                            }
                        )
        return {
            "status": "matched" if not mismatches else "mismatch",
            "scope": "text prompts only; exact input token IDs before /generate",
            "model": PROFILE["repo_id"],
            "revision": PROFILE["revision"],
            "served_model": args.model,
            "engine_url": args.engine_url,
            "sources": sources,
            "selected_records": len(items),
            "checked_questions": questions,
            "assistant_prefix": args.assistant_prefix,
            "prompt_wording": args.prompt_wording,
            "native_system_prompt": args.native_system_prompt,
            "mismatched_questions": len(mismatches),
            "mismatch_samples": mismatches[:20],
            "preparation_ms": {
                "engine_http": latency_summary(http_prep_ms),
                "local": latency_summary(local_prep_ms),
                "scope": "sequential HTTP-first preparation only; excludes /generate",
            },
        }
    finally:
        await http_backend.close()
        await local_backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="decision-model")
    parser.add_argument("--assistant-prefix")
    parser.add_argument("--prompt-wording", choices=("served", "native"), default="served")
    parser.add_argument(
        "--native-system-prompt", choices=("auto", "none"), default="auto"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    report = asyncio.run(compare(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(
        f"{report['status']}: {report['checked_questions']} questions, "
        f"{report['mismatched_questions']} prompt ID mismatches; {args.output}"
    )
    raise SystemExit(0 if report["status"] == "matched" else 1)


if __name__ == "__main__":
    main()
