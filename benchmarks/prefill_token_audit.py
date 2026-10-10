"""CPU-only full-token audit for stock and integrated prefill decision paths.

Engine replies are fabricated only after each real backend has submitted its
input IDs to a recording transport. No model forward, GPU, or engine HTTP runs.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx

from jev_adapter.benchmarks.run import load_data, write_json
from jev_adapter.inprocess import (
    InProcessDecisionBackend,
    _load_metadata,
    _validate_metadata,
    artifact_label_scheme,
)
from jev_adapter.native_tokenizer import NativeTokenizer
from jev_adapter.protocol import (
    SystemOneRequest,
    build_prompt,
    plan_question,
    rotation_order,
)
from jev_adapter.sglang import SGLangBackend

if __package__:
    from .prefill_compare import request_for
    from .prefill_engine_bench import canonical_sha256, file_sha256, read_settings
else:
    from prefill_compare import request_for
    from prefill_engine_bench import canonical_sha256, file_sha256, read_settings


def _cpu_only() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("set CUDA_VISIBLE_DEVICES='' before this CPU-only audit")


def _class_source(tokenizer) -> dict:
    name = type(tokenizer)
    value = {"class": f"{name.__module__}.{name.__qualname__}"}
    try:
        source = inspect.getsourcefile(name)
    except (OSError, TypeError):
        source = None
    if source and Path(source).is_file():
        value["class_source_path"] = str(Path(source).resolve())
        value["class_source_sha256"] = file_sha256(Path(source))
    return value


def _loader_source(load_function) -> dict:
    path = Path(inspect.getsourcefile(load_function)).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def _load_tokenizers(model_id: str, engine_model_path: str, revision: str):
    _cpu_only()
    # Both loaders must resolve the pinned snapshot from the local cache.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    stock = NativeTokenizer.from_pretrained(model_id, revision)
    from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

    integrated = get_tokenizer(
        engine_model_path,
        tokenizer_revision=revision,
        local_files_only=True,
    )
    return stock, integrated, _loader_source(get_tokenizer)


class _AuditGenerateReqInput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _RecordingManager:
    is_generation = True

    def __init__(self, tokenizer, metadata: dict, model: str):
        self.tokenizer = tokenizer
        self.model_path = metadata["engine_model_path"]
        self.served_model_name = model
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="mistral3",
                architectures=[metadata["architecture"]],
                text_config=SimpleNamespace(hidden_size=metadata["hidden_size"]),
            )
        )
        self.submitted: list[list[int]] = []
        self.label_count = len(metadata["labels"])

    async def generate_request(self, obj, request=None):
        if request is not None:
            raise ValueError("audit expected a local manager request")
        ids = list(obj.input_ids)
        self.submitted.append(ids)
        yield {
            "meta_info": {
                "prompt_tokens": len(ids),
                "cached_tokens": 0,
                "completion_tokens": 0,
                # Readout logits, then the labels' vocabulary logprobs.
                "decision_logits": [0.0] * (2 * self.label_count),
            }
        }

    def abort_request(self, rid):
        raise RuntimeError(f"audit request unexpectedly aborted: {rid}")


def _recording_http(
    submitted: list[list[int]], tokenizer, metadata: dict, model: str
) -> httpx.MockTransport:
    def respond(request):
        if request.method == "GET" and request.url.path == "/model_info":
            return httpx.Response(
                200,
                json={
                    "is_generation": True,
                    "served_model_name": model,
                    "model_path": metadata["engine_model_path"],
                    "model_type": "mistral3",
                    "architectures": [metadata["architecture"]],
                },
            )
        if request.method == "GET" and request.url.path == "/server_info":
            return httpx.Response(
                200,
                json={"skip_tokenizer_init": False, "speculative_algorithm": None},
            )
        if request.method != "POST":
            raise ValueError("audit backend called an unexpected HTTP endpoint")
        payload = json.loads(request.content)
        if request.url.path == "/v1/tokenize":
            if (
                payload.get("model") != model
                or payload.get("add_special_tokens") is not False
            ):
                raise ValueError("stock startup changed its label tokenization")
            return httpx.Response(
                200,
                json={
                    "tokens": [
                        tokenizer.encode(label, add_special_tokens=False)
                        for label in payload["prompt"]
                    ]
                },
            )
        if request.url.path == "/v1/detokenize":
            if (
                payload.get("model") != model
                or payload.get("skip_special_tokens") is not True
            ):
                raise ValueError("stock startup changed its label detokenization")
            return httpx.Response(
                200,
                json={
                    "text": [
                        tokenizer.decode(ids, skip_special_tokens=True)
                        for ids in payload["tokens"]
                    ]
                },
            )
        if request.url.path != "/generate":
            raise ValueError("audit backend called an unexpected HTTP endpoint")
        ids = payload.get("input_ids")
        tokens = payload.get("token_ids_logprob")
        if (
            not isinstance(ids, list)
            or not ids
            or not isinstance(tokens, list)
            or payload.get("sampling_params", {}).get("max_new_tokens") != 0
        ):
            raise ValueError("stock backend did not submit a zero-output token request")
        submitted.append(list(ids))
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "prompt_tokens": len(ids),
                    "cached_tokens": 0,
                    "completion_tokens": 0,
                    "output_token_ids_logprobs": [
                        [[-1.0, token, None] for token in tokens]
                    ],
                }
            },
        )

    return httpx.MockTransport(respond)


def _audit_entry(
    request: dict, item: dict, question_names: list[str], ids: list[list[int]]
):
    if len(question_names) != len(ids):
        raise ValueError("backend submitted the wrong number of questions")
    return {
        "id": item["id"],
        "request_sha256": canonical_sha256(request),
        "input_token_ids_sha256": canonical_sha256(ids),
        "questions": [
            {
                "id": name,
                "input_tokens": len(tokens),
                "input_token_ids": tokens,
            }
            for name, tokens in zip(question_names, ids)
        ],
    }


async def audit(
    *,
    data: list[Path],
    model: str,
    model_id: str,
    revision: str,
    readout: Path,
    settings_manifest: Path,
    limit: int,
    allow_test: bool = False,
    stock_tokenizer: NativeTokenizer | None = None,
    integrated_tokenizer=None,
) -> tuple[dict, dict]:
    _cpu_only()
    if type(limit) is not int or not 1 <= limit <= 4096:
        raise ValueError("limit must be between 1 and 4096")
    settings, _ = read_settings(settings_manifest)
    metadata = _load_metadata(readout)
    if (
        settings["head_mode"] not in ("lm_rows", "learned")
        or settings["head_mode"] != metadata["mode"]
        or settings["prompt_contract"] != "served-v1"
        or settings["model_path"] != metadata["engine_model_path"]
        or settings["model_revision"] != revision
        or metadata["revision"] != revision
        or metadata["model_id"] != model_id
        or settings["labels"] != metadata["labels"]
        or settings["token_ids"] != metadata["token_ids"]
        or Path(settings["head_artifact"]["path"]).resolve() != readout.resolve()
    ):
        raise ValueError("settings, pinned model, and GPU readout disagree")
    if not model.strip():
        raise ValueError("model must be nonempty")
    items, sources = load_data(data, model, allow_test, limit)
    requests = [request_for(item, model, None) for item in items]
    loader_source = None
    if stock_tokenizer is None or integrated_tokenizer is None:
        stock_tokenizer, integrated_tokenizer, loader_source = _load_tokenizers(
            model_id, metadata["engine_model_path"], revision
        )
    if not isinstance(stock_tokenizer, NativeTokenizer):
        raise TypeError("stock tokenizer must use the native adapter wrapper")
    manager = _RecordingManager(integrated_tokenizer, metadata, model)
    # Audit the artifact under its own label scheme on both paths.
    label_scheme = artifact_label_scheme(metadata)
    server_args = SimpleNamespace(
        model_path=metadata["engine_model_path"],
        revision=revision,
        tokenizer_path=metadata["engine_model_path"],
        quantization=None,
        speculative_algorithm=None,
        decision_label_scheme=label_scheme,
    )
    boundary_id = _validate_metadata(metadata, manager, server_args)
    integrated = InProcessDecisionBackend(
        manager, metadata, _AuditGenerateReqInput, boundary_id
    )
    stock_submitted: list[list[int]] = []
    async with httpx.AsyncClient(
        transport=_recording_http(
            stock_submitted, integrated_tokenizer, metadata, model
        )
    ) as client:
        stock = SGLangBackend(
            "http://audit.invalid",
            model,
            client=client,
            native_tokenizer=stock_tokenizer,
            prompt_wording="served",
            label_scheme=label_scheme,
        )
        await stock.start()
        if stock.labels(len(metadata["labels"])) != (
            tuple(metadata["labels"]),
            tuple(metadata["token_ids"]),
        ):
            raise ValueError("readout label rows differ from stock engine label order")
        stock_entries, integrated_entries = [], []
        for item, request in zip(items, requests):
            checked = SystemOneRequest.model_validate(request)
            if (
                checked.images
                or checked.options.temperature != 1.0
                or checked.options.permutations != 1
                or checked.options.assistant_prefix
                or checked.options.return_logprobs
            ):
                raise ValueError(
                    "audit requires served text with one unrotated evaluation"
                )
            stock_start = len(stock_submitted)
            integrated_start = len(manager.submitted)
            question_names = []
            for name, question in checked.questions.items():
                plan = plan_question(question)
                labels, token_ids = integrated.labels(len(plan.option_names))
                if stock.labels(len(plan.option_names)) != (labels, token_ids):
                    raise ValueError("stock and integrated label selections differ")
                prompt = build_prompt(
                    checked.state,
                    plan,
                    labels,
                    rotation_order(len(labels), 0),
                    wording="served",
                )
                await stock.evaluate(prompt, [], labels, token_ids, None)
                await integrated.evaluate(prompt, [], labels, token_ids, None)
                question_names.append(name)
            stock_ids = stock_submitted[stock_start:]
            integrated_ids = manager.submitted[integrated_start:]
            stock_entries.append(_audit_entry(request, item, question_names, stock_ids))
            integrated_entries.append(
                _audit_entry(request, item, question_names, integrated_ids)
            )

    common = {
        "schema_version": 1,
        "workload_sha256": canonical_sha256(requests),
        "model_revision": revision,
        "prompt_contract": settings["prompt_contract"],
        "label_contract": settings["label_contract"],
        "model_id": model_id,
        "engine_model_path": metadata["engine_model_path"],
        "readout_sha256": file_sha256(readout),
        "data": sources,
        "selected_records": len(items),
        "cpu_tokenization_only": True,
        "gpu_parity_claim": False,
    }
    stock_audit = {
        **common,
        "path": "stock_native_adapter",
        "tokenizer": {
            **_class_source(stock_tokenizer.tokenizer),
            "native_loader_source": _loader_source(NativeTokenizer.from_pretrained),
            "backend_source": _loader_source(SGLangBackend.evaluate),
        },
        "requests": stock_entries,
    }
    integrated_audit = {
        **common,
        "path": "integrated_sglang",
        "tokenizer": {
            **_class_source(integrated_tokenizer),
            "sglang_loader_source": loader_source,
            "backend_source": _loader_source(InProcessDecisionBackend.evaluate),
        },
        "requests": integrated_entries,
    }
    return stock_audit, integrated_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--model", required=True, help="served request model name")
    parser.add_argument("--model-id", required=True, help="pinned tokenizer model ID")
    parser.add_argument("--revision", required=True, help="40-character model commit")
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--settings-manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    try:
        stock, integrated = asyncio.run(
            audit(
                data=args.data,
                model=args.model,
                model_id=args.model_id,
                revision=args.revision,
                readout=args.readout,
                settings_manifest=args.settings_manifest,
                limit=args.limit,
                allow_test=args.allow_test,
            )
        )
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stock_path = args.output_dir / "stock-token-audit.json"
    integrated_path = args.output_dir / "candidate-token-audit.json"
    write_json(stock_path, stock)
    write_json(integrated_path, integrated)
    mismatches = [
        index
        for index, (left, right) in enumerate(
            zip(stock["requests"], integrated["requests"])
        )
        if left["input_token_ids_sha256"] != right["input_token_ids_sha256"]
    ]
    print(
        json.dumps(
            {
                "stock_token_audit": str(stock_path),
                "candidate_token_audit": str(integrated_path),
                "selected_records": len(stock["requests"]),
                "mismatch_indices": mismatches[:20],
                "cpu_tokenization_only": True,
            }
        )
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
