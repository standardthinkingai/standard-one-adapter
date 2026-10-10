"""In-process System One bridge for the SGLang decision-readout fast path.

This module has no SGLang import at import time. The inference engine owns the
readout weights and returns only a compact decision-logit vector. This bridge
uses its already initialized tokenizer and scheduler, with no HTTP calls.
"""

import array
import asyncio
import json
import logging
import math
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .backend import AdapterError, Evaluation, ScoringResult
from .labels import resolve_label_scheme, select_labels, validate_label_scheme
from .native_tokenizer import NativeTokenizer
from .service import DATA_URL, SystemOneService

logger = logging.getLogger(__name__)

# v2 records its label scheme and tokenizer; v1 artifacts predate schemes and
# always carry the served order.
FORMAT = "jev-gpu-readout-v2"
LEGACY_FORMAT = "jev-gpu-readout-v1"
# The gateway waits 120 seconds for a decision; the engine gives up first so
# the caller gets 504 engine_timeout and the GPU work is aborted.
DEFAULT_DEADLINE_SECONDS = 110.0
ARCHITECTURE = "Mistral3ForConditionalGeneration"
_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_METADATA_KEYS = frozenset(
    {
        "format",
        "mode",
        "model_id",
        "engine_model_path",
        "revision",
        "architecture",
        "hidden_size",
        "labels",
        "token_ids",
        "temperature",
    }
)
_LABEL_METADATA_KEYS = frozenset(
    {"label_scheme", "tokenizer_id", "tokenizer_revision"}
)


def artifact_label_scheme(metadata: dict) -> str:
    """The label scheme a readout artifact was exported with."""
    if metadata.get("format") == LEGACY_FORMAT:
        return "served"
    try:
        return validate_label_scheme(metadata.get("label_scheme"))
    except ValueError as error:
        raise ValueError(f"decision readout {error}") from None


def _field(source: Any, key: str) -> Any:
    return source.get(key) if isinstance(source, dict) else getattr(source, key, None)


def _bad_response(message: str) -> AdapterError:
    return AdapterError("invalid_engine_response", message, status=502)


def _request_error(error: ValueError, images: bool) -> AdapterError | None:
    """Translate only known client-input failures from the engine manager.

    Manager/processor invariant failures also use ValueError. They must remain
    server errors, rather than being mislabeled as invalid caller input.
    Loader messages can contain image data, so never echo them.
    """
    message = str(error)
    if images and message.startswith(
        (
            "Error while loading data ",
            "An exception occurred while loading IMAGE data at index ",
            "An exception occurred while loading multimodal data: ",
        )
    ):
        return AdapterError("invalid_image", "An image could not be decoded.", "images")
    if images and re.fullmatch(
        r"Image count [0-9]+ exceeds limit [0-9]+ per request\.", message
    ):
        return AdapterError("image_limit_exceeded", message, "images")
    if (
        re.fullmatch(
            r"The input \([0-9]+ tokens\) is longer than the "
            r"model's context length \([0-9]+ tokens\)\.",
            message,
        )
        or message.startswith(
            "Requested token count exceeds the model's maximum context length of "
        )
        or (
            images
            and message == "decision images do not support automatic prompt truncation"
        )
    ):
        if images:
            return AdapterError(
                "context_length_exceeded",
                "The image and text input exceeds the model's context limit; "
                "reduce image size/count or text length.",
                "images",
            )
        return AdapterError(
            "context_length_exceeded",
            "The input exceeds the model's context limit; shorten the state or questions.",
            "questions",
        )
    return None


def _validate_readout_precision(metadata: dict, manager: Any, server_args: Any) -> None:
    """Permit only the isolated native online FP8 backbone with a BF16 LM head.

    The artifact still contains original BF16 checkpoint LM rows. The model
    worker separately validates its resolved Fp8Config and actual LM-head dtype
    and quantization method before the first readout.
    """
    quantization = _field(server_args, "quantization")
    if quantization in (None, False):
        return  # Preserve the original unquantized adapter behavior.
    if quantization != "fp8":
        raise ValueError(
            "decision readout supports only native online fp8 quantization"
        )
    model_config = _field(manager, "model_config")
    if (
        metadata.get("mode") != "lm_rows"
        or _field(server_args, "dtype") != "bfloat16"
        or str(_field(model_config, "dtype")) not in ("bfloat16", "torch.bfloat16")
        or _field(model_config, "quantization") != "fp8"
    ):
        raise ValueError(
            "online fp8 decision readout requires BF16 LM rows and activations"
        )
    if _field(server_args, "prefill_only_disable_kv_cache"):
        raise ValueError("online fp8 decision readout does not support KV-free prefill")
    if _field(server_args, "enable_fp32_lm_head"):
        raise ValueError("online fp8 decision readout requires a BF16 LM head")
    config = _field(model_config, "hf_config")
    for scope in (config, _field(config, "text_config")):
        for name in ("quantization_config", "compression_config"):
            if _field(scope, name) is not None:
                raise ValueError(
                    "online fp8 decision readout requires an unquantized checkpoint"
                )


def _load_metadata(path: str | Path) -> dict:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("The in-process decision backend requires NumPy.") from error
    try:
        # NpzFile lazily loads members. Only metadata.npy is read here; the
        # model runner separately owns weights and bias on the GPU.
        with np.load(path, allow_pickle=False) as artifact:
            if set(artifact.files) != {"metadata", "weights", "bias"}:
                raise ValueError("decision readout needs metadata, weights, and bias")
            value = artifact["metadata"]
            if value.ndim == 0 and value.dtype.kind == "S":
                raw = bytes(value.item())
            elif value.ndim == 1 and value.dtype == np.uint8:
                raw = value.tobytes()
            else:
                raise ValueError("decision readout metadata must be JSON bytes")
        metadata = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read decision readout metadata") from error
    if not isinstance(metadata, dict) or metadata.get("format") not in (
        FORMAT,
        LEGACY_FORMAT,
    ):
        raise ValueError("unsupported GPU decision readout format")
    required = _METADATA_KEYS | (
        _LABEL_METADATA_KEYS if metadata["format"] == FORMAT else frozenset()
    )
    if not required.issubset(metadata) or set(metadata) - required - {"source_sha256"}:
        raise ValueError("unsupported GPU decision readout format")
    return metadata


def _validate_metadata(metadata: dict, manager: Any, server_args: Any) -> int:
    if metadata.get("mode") not in {"lm_rows", "learned"}:
        raise ValueError("decision readout mode must be lm_rows or learned")
    for name in ("model_id", "engine_model_path", "revision"):
        if not isinstance(metadata.get(name), str) or not metadata[name]:
            raise ValueError(f"decision readout {name} is missing")
    if _REVISION_RE.fullmatch(metadata["revision"]) is None:
        raise ValueError("decision readout revision must be a pinned commit")
    if "source_sha256" in metadata and (
        not isinstance(metadata["source_sha256"], str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", metadata["source_sha256"]) is None
    ):
        raise ValueError("decision readout source_sha256 is invalid")
    if metadata.get("architecture") != ARCHITECTURE:
        raise ValueError("decision readout architecture is unsupported")
    hidden_size = metadata.get("hidden_size")
    if type(hidden_size) is not int or hidden_size < 1:
        raise ValueError("decision readout hidden_size is invalid")
    temperature = metadata.get("temperature")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("decision readout temperature is invalid")
    labels = metadata.get("labels")
    token_ids = metadata.get("token_ids")
    if (
        not isinstance(labels, list)
        or not isinstance(token_ids, list)
        or not 2 <= len(labels) <= 255
        or len(labels) != len(token_ids)
        or any(not isinstance(label, str) or not label for label in labels)
        or any(type(token) is not int or token < 0 for token in token_ids)
        or len(set(labels)) != len(labels)
        or len(set(token_ids)) != len(token_ids)
    ):
        raise ValueError("decision readout labels/token IDs are invalid")

    if _field(manager, "is_generation") is not True:
        raise ValueError("System One requires a generation model")
    model_path = _field(manager, "model_path")
    if (
        model_path != metadata["engine_model_path"]
        or _field(server_args, "model_path") != metadata["engine_model_path"]
    ):
        raise ValueError("decision readout model path differs from loaded model")
    if metadata["model_id"] != model_path and not Path(model_path).is_absolute():
        raise ValueError("decision readout model ID differs from loaded model")
    if _field(server_args, "revision") != metadata["revision"]:
        raise ValueError("decision readout revision differs from loaded model")
    scheme = artifact_label_scheme(metadata)
    configured = resolve_label_scheme(_field(server_args, "decision_label_scheme"))
    if scheme != configured:
        raise ValueError(
            f"decision readout labels use the {scheme!r} scheme but this server "
            f"is configured for {configured!r}; export a readout with "
            f"--label-scheme {configured} or start with --decision-label-scheme "
            f"{scheme} (env JEV_LABEL_SCHEME)"
        )
    if metadata["format"] == FORMAT and (
        metadata.get("tokenizer_id")
        not in (metadata["model_id"], metadata["engine_model_path"])
        or metadata.get("tokenizer_revision") != metadata["revision"]
    ):
        raise ValueError("decision readout tokenizer differs from the loaded model")
    tokenizer_path = _field(server_args, "tokenizer_path")
    if tokenizer_path not in (
        None,
        "",
        metadata["model_id"],
        metadata["engine_model_path"],
    ):
        raise ValueError("decision tokenizer path differs from readout model")
    _validate_readout_precision(metadata, manager, server_args)
    if _field(server_args, "speculative_algorithm") is not None:
        raise ValueError("decision readout does not support speculation or MTP")

    config = _field(_field(manager, "model_config"), "hf_config")
    architectures = _field(config, "architectures")
    text_config = _field(config, "text_config")
    if (
        _field(config, "model_type") != "mistral3"
        or not isinstance(architectures, (list, tuple))
        or ARCHITECTURE not in architectures
        or _field(text_config, "hidden_size") != hidden_size
    ):
        raise ValueError("decision readout disagrees with Mistral3 model config")

    tokenizer = _field(manager, "tokenizer")
    if tokenizer is None or not all(
        callable(getattr(tokenizer, name, None))
        for name in ("encode", "decode", "apply_chat_template")
    ):
        raise ValueError("the engine's native tokenizer is unavailable")
    special_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    for label, token in zip(labels, token_ids):
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if (
            not isinstance(encoded, list)
            or encoded != [token]
            or token in special_ids
            or tokenizer.decode([token], skip_special_tokens=True) != label
        ):
            raise ValueError("decision readout labels differ from engine tokenizer")
    # The rows must be exactly the scheme's leading labels for this tokenizer,
    # the same list training used; a reordered or gapped export is refused.
    if select_labels(tokenizer, scheme, limit=len(labels)) != (labels, token_ids):
        raise ValueError(
            f"decision readout labels are not the {scheme!r} scheme's labels "
            "for the engine tokenizer"
        )
    # The Mistral instruction boundary is an atomic special token. Once all
    # labels are proven to be one-token continuations of that boundary, a
    # request can render only its base chat and check the same terminal ID.
    # This removes the O(labels * prompt length) retokenization from the hot
    # path without trusting plain-text tokenization across control markers.
    native = NativeTokenizer(tokenizer)
    boundary_id = None
    for index, probe in enumerate(
        ("Decision tokenizer validation.", "A", "A trailing space ")
    ):
        count = len(labels) if index == 0 else 2
        prepared = native.prepare(
            probe,
            tuple(labels[:count]),
            tuple(token_ids[:count]),
            None,
        )
        terminal = prepared[-1]
        if (
            terminal not in special_ids
            or tokenizer.decode([terminal], skip_special_tokens=False) != "[/INST]"
            or (
                callable(getattr(tokenizer, "convert_ids_to_tokens", None))
                and tokenizer.convert_ids_to_tokens(terminal) != "[/INST]"
            )
        ):
            raise ValueError("native tokenizer lacks an atomic [/INST] boundary")
        if boundary_id is not None and terminal != boundary_id:
            raise ValueError("native tokenizer has an unstable assistant boundary")
        boundary_id = terminal
    return boundary_id


def _generate_input_type():
    # Keep this module importable in CPU-only adapter tests.
    from sglang.srt.managers.io_struct import GenerateReqInput

    return GenerateReqInput


@dataclass(frozen=True, slots=True)
class PreparedDecision:
    """One evaluation whose native prompt IDs passed every input check."""

    # Native IDs with one atomic image marker per image; 4 bytes per token.
    input_ids: array.array
    images: tuple[str, ...]
    token_ids: tuple[int, ...]
    # Prompt tokens other than image tokens.
    text_tokens: int
    # The prompt with images expanded, when the processor can count them.
    expected_tokens: int | None


class InProcessDecisionBackend:
    supports_return_logprobs = False

    def __init__(
        self,
        manager: Any,
        metadata: dict,
        request_type: type,
        boundary_id: int,
        *,
        max_prompt_tokens: int | None = None,
        max_images: int | None = None,
    ):
        self.manager = manager
        self.model = manager.served_model_name or manager.model_path
        self.mode = metadata["mode"]
        self.revision = metadata["revision"]
        self.label_scheme = artifact_label_scheme(metadata)
        self._labels = tuple(metadata["labels"])
        self._token_ids = tuple(metadata["token_ids"])
        self._token_to_position = {
            token: position for position, token in enumerate(self._token_ids)
        }
        self._native_tokenizer = NativeTokenizer(manager.tokenizer)
        self._request_type = request_type
        self._boundary_id = boundary_id
        self._image_placeholder_id: int | None = None
        self._instruction_start_id: int | None = None
        # The engine refuses a prompt of context_len - reserved tokens or more.
        # None leaves the context check to the engine (offline audits only).
        self._max_prompt_tokens = max_prompt_tokens
        self._max_images = max_images
        self._warned_image_count = False
        # LM-row readouts are the LM head's label rows, so the engine's
        # full-vocabulary label logprobs serve both the answer and
        # options.return_logprobs, exactly as the HTTP adapter's /generate.
        # A learned head's logits have no vocabulary normalization.
        self.supports_return_logprobs = self.mode == "lm_rows"
        # (options.assistant_prefix, label count) -> the prefix's native token
        # IDs after [/INST], checked once against that many leading labels.
        self._prefix_ids_cache: dict[tuple[str, int], tuple[int, ...]] = {}

    async def start(self) -> None:
        # The engine owns model/readout startup. No GPU request is sent here.
        return None

    async def close(self) -> None:
        # The engine owns its tokenizer manager lifecycle.
        return None

    def labels(self, count: int) -> tuple[tuple[str, ...], tuple[int, ...]]:
        if type(count) is not int or not 2 <= count <= len(self._labels):
            raise AdapterError(
                "labels_unavailable",
                f"This readout supports 2 to {len(self._labels)} answer labels.",
                "questions",
            )
        return self._labels[:count], self._token_ids[:count]

    def _image_placeholder(self) -> int:
        """Resolve the loaded model's atomic image marker without loading media."""
        if self._image_placeholder_id is not None:
            return self._image_placeholder_id
        token = _field(_field(self.manager, "model_config"), "image_token_id")
        tokenizer = self.manager.tokenizer
        try:
            valid = (
                type(token) is int
                and token >= 0
                and token in (getattr(tokenizer, "all_special_ids", ()) or ())
                and tokenizer.decode([token], skip_special_tokens=False) == "[IMG]"
            )
        except Exception as error:
            raise AdapterError(
                "unsupported_image_tokenizer",
                "The loaded tokenizer cannot represent the model's image marker.",
                "images",
            ) from error
        if not valid:
            raise AdapterError(
                "unsupported_image_tokenizer",
                "Image scoring requires the model's atomic [IMG] token.",
                "images",
            )
        self._image_placeholder_id = token
        return token

    def _insert_image_placeholders(self, input_ids: list[int], count: int) -> list[int]:
        image_token = self._image_placeholder()
        if image_token in input_ids:
            raise AdapterError(
                "invalid_image_prompt",
                "The scoring text must not contain reserved image markers.",
                "images",
            )
        if self._instruction_start_id is None:
            tokenizer = self.manager.tokenizer
            special_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
            candidates = {
                token
                for token in input_ids
                if token in special_ids
                and tokenizer.decode([token], skip_special_tokens=False) == "[INST]"
            }
            if len(candidates) == 1:
                self._instruction_start_id = candidates.pop()
        positions = [
            index
            for index, token in enumerate(input_ids)
            if token == self._instruction_start_id
        ]
        if len(positions) != 1 or input_ids.count(self._boundary_id) != 1:
            raise AdapterError(
                "invalid_image_boundary",
                "Image scoring requires one native [INST] ... [/INST] user turn.",
                "images",
            )
        # Put images first in the user turn, matching the native multimodal
        # template. Preserve any system prefix and every text/control token.
        # Literal '[IMG]' text is not a special token in mistral-common, and
        # URLs passed to its chat template can trigger eager media downloads.
        # The engine's existing processor owns image loading and expansion.
        insertion = positions[0] + 1
        return input_ids[:insertion] + [image_token] * count + input_ids[insertion:]

    def _check_images_supported(self, images: Sequence[str]) -> None:
        if images and (
            _field(_field(self.manager, "model_config"), "is_multimodal") is not True
            or not callable(
                getattr(_field(self.manager, "mm_processor"), "process_mm_data_async", None)
            )
        ):
            raise AdapterError(
                "image_processor_unavailable",
                "Image scoring requires the engine's loaded vision encoder and "
                "multimodal processor; restart without --language-model-only.",
                "images",
            )

    def _check_labels(self, labels: tuple[str, ...], token_ids: tuple[int, ...]) -> None:
        if (
            len(labels) < 2
            or len(labels) != len(token_ids)
            or any(
                self._token_to_position.get(token) != position
                or self._labels[position] != label
                for position, (label, token) in enumerate(zip(labels, token_ids))
            )
        ):
            raise AdapterError(
                "invalid_labels", "Labels must match the readout's leading rows."
            )

    def _native_ids(
        self, prompt: str, image_count: int, prefix: str = "", prefix_ids=()
    ) -> list[int]:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self._native_tokenizer._chat(messages, continuation=False)
        if input_ids[-1] != self._boundary_id:
            raise AdapterError(
                "invalid_label_boundary",
                "The native chat template changed its terminal instruction token.",
                "questions",
            )
        if prefix:
            # The whole conversation is rendered with the assistant prefix, as
            # NativeTokenizer.prepare does, and must extend the base prompt by
            # exactly the prefix tokens whose label continuations were checked.
            final_ids = self._native_tokenizer._chat(
                messages + [{"role": "assistant", "content": prefix}], continuation=True
            )
            if final_ids != input_ids + list(prefix_ids):
                raise AdapterError(
                    "invalid_label_boundary",
                    "The assistant prefix changes the native chat prompt boundary.",
                    "options.assistant_prefix",
                )
            input_ids = final_ids
        if image_count:
            input_ids = self._insert_image_placeholders(input_ids, image_count)
        return input_ids

    def _prefix_ids(self, prefix: str, label_count: int) -> tuple[int, ...]:
        """Native IDs of an assistant prefix after which each label is one token.

        Text after the atomic [/INST] is tokenized apart from the prompt, so a
        short probe proves the label boundary for every request; each request
        still re-renders its own prompt with the prefix (_native_ids).
        """
        key = (prefix, label_count)
        cached = self._prefix_ids_cache.get(key)
        if cached is not None:
            return cached
        probe = "Decision assistant-prefix validation."
        base = self._native_tokenizer._chat(
            [{"role": "user", "content": probe}], continuation=False
        )
        final = self._native_tokenizer.prepare(
            probe, self._labels[:label_count], self._token_ids[:label_count], prefix
        )
        if final[: len(base)] != base or len(final) == len(base):
            raise AdapterError(
                "invalid_label_boundary",
                "The assistant prefix changes the native chat prompt boundary.",
                "options.assistant_prefix",
            )
        prefix_ids = tuple(final[len(base) :])
        if len(self._prefix_ids_cache) >= 256:
            self._prefix_ids_cache.pop(next(iter(self._prefix_ids_cache)))
        self._prefix_ids_cache[key] = prefix_ids
        return prefix_ids

    def _image_tokens(self, images: Sequence[str]) -> int | None:
        """Prompt tokens the engine's processor will expand these images into.

        None when the loaded processor cannot count without processing the
        images; the engine still enforces the context limit per evaluation.
        """
        counter = getattr(
            _field(self.manager, "mm_processor"), "decision_image_tokens", None
        )
        if not callable(counter):
            return None
        total = 0
        for image in images:
            try:
                count = counter(image)
            except ValueError as error:
                raise AdapterError(
                    "invalid_image", "An image could not be decoded.", "images"
                ) from error
            if type(count) is not int or count < 1:
                raise RuntimeError("the image processor returned an invalid token count")
            total += count
        return total

    async def prepare(
        self,
        evaluations: Sequence[Evaluation],
        images: Sequence[str],
        assistant_prefix: str | None,
    ) -> list[PreparedDecision]:
        """Tokenize and check every evaluation before any GPU work (ENG-2).

        A request that could not complete is refused here as a whole, so the
        engine never spends GPU time on part of it.
        """
        self._check_images_supported(images)
        for evaluation in evaluations:
            self._check_labels(evaluation.labels, evaluation.token_ids)
        prefix = assistant_prefix or ""
        prefix_ids = (
            await asyncio.to_thread(
                self._prefix_ids,
                prefix,
                max(len(evaluation.labels) for evaluation in evaluations),
            )
            if prefix and evaluations
            else ()
        )
        images = tuple(images)
        if images and self._max_images is not None and len(images) > self._max_images:
            raise AdapterError(
                "image_limit_exceeded",
                f"This model accepts at most {self._max_images} images per request.",
                "images",
            )
        image_tokens = (
            await asyncio.to_thread(self._image_tokens, images) if images else 0
        )
        tokenized = await asyncio.gather(
            *(
                asyncio.to_thread(
                    self._native_ids, evaluation.prompt, len(images), prefix, prefix_ids
                )
                for evaluation in evaluations
            ),
            return_exceptions=True,
        )
        prepared = []
        limit = self._max_prompt_tokens
        for evaluation, input_ids in zip(evaluations, tokenized):
            if isinstance(input_ids, BaseException):
                raise input_ids
            text_tokens = len(input_ids) - len(images)
            if limit is not None and text_tokens > limit:
                raise AdapterError(
                    "context_length_exceeded",
                    f"Question {evaluation.question!r} needs {text_tokens} input tokens; "
                    f"the model accepts at most {limit}. Shorten the state or question.",
                    f"questions.{evaluation.question}",
                )
            expected = None if image_tokens is None else text_tokens + image_tokens
            if limit is not None and expected is not None and expected > limit:
                raise AdapterError(
                    "context_length_exceeded",
                    f"Question {evaluation.question!r} with its images needs {expected} "
                    f"input tokens; the model accepts at most {limit}. Send fewer or "
                    "smaller images, or shorten the text.",
                    "images",
                )
            prepared.append(
                PreparedDecision(
                    array.array("i", input_ids),
                    images,
                    evaluation.token_ids,
                    text_tokens,
                    expected,
                )
            )
        return prepared

    async def evaluate(
        self,
        prompt: str,
        images: list[str],
        labels: tuple[str, ...],
        token_ids: tuple[int, ...],
        assistant_prefix: str | None,
    ) -> ScoringResult:
        (prepared,) = await self.prepare(
            [Evaluation("", prompt, labels, token_ids)], images, assistant_prefix
        )
        return await self.evaluate_prepared(prepared)

    async def evaluate_prepared(self, prepared: PreparedDecision) -> ScoringResult:
        meta, prompt_tokens, cached_tokens = await self._generate(
            prepared.input_ids.tolist(), prepared.images
        )
        # Each row holds the readout logits, then the same labels'
        # full-vocabulary logprobs.
        row = meta.get("decision_logits")
        count = len(self._token_ids)
        if (
            not isinstance(row, list)
            or len(row) != 2 * count
            or any(
                type(logit) not in (int, float) or not math.isfinite(logit)
                for logit in row
            )
        ):
            raise _bad_response("The engine returned malformed decision logits.")
        logits = row[count:] if self.mode == "lm_rows" else row[:count]
        image_tokens = self._image_token_usage(
            prompt_tokens, prepared.text_tokens, prepared.images, prepared.expected_tokens
        )
        return ScoringResult(
            tuple(float(logits[self._token_to_position[token]]) for token in prepared.token_ids),
            prompt_tokens,
            cached_tokens,
            image_tokens,
        )

    def _image_token_usage(self, prompt_tokens, text_tokens, images, expected_tokens):
        # The engine keeps every native text ID and replaces each image marker
        # with its patch block, so the image tokens are exactly the remainder.
        if not images:
            return 0
        image_tokens = prompt_tokens - text_tokens
        if image_tokens < 2 * len(images):
            raise _bad_response("The engine's prompt usage omits image tokens.")
        if (
            expected_tokens is not None
            and prompt_tokens != expected_tokens
            and not self._warned_image_count
        ):
            self._warned_image_count = True
            logger.warning(
                "Decision image prompt had %d tokens; the processor predicted %d. "
                "The context pre-check may refuse or admit near the limit.",
                prompt_tokens,
                expected_tokens,
            )
        return image_tokens

    async def _generate(self, input_ids, images, **fields):
        """One zero-output prefill; returns (meta_info, prompt_tokens, cached_tokens)."""
        media = {"image_data": list(images)} if images else {}
        request = self._request_type(
            input_ids=input_ids,
            sampling_params={
                "max_new_tokens": 0,
                "temperature": 1,
                "top_p": 1,
                "top_k": -1,
            },
            return_logprob=False,
            return_hidden_states=False,
            stream=False,
            rid="systemone-" + uuid.uuid4().hex,
            **media,
            **fields,
        )
        iterator = self.manager.generate_request(request, request=None)
        finished = False
        try:
            try:
                result = await iterator.__anext__()
            except StopAsyncIteration as error:
                raise _bad_response(
                    "The engine returned no decision result."
                ) from error
            except ValueError as error:
                translated = _request_error(error, bool(images))
                if translated is not None:
                    raise translated from error
                raise
            except HTTPException as error:
                # The scheduler refuses work it cannot queue with 503 (ENG-4).
                if error.status_code == 503:
                    raise AdapterError(
                        "engine_overloaded",
                        "The decision engine is at capacity; retry shortly.",
                        status=503,
                    ) from error
                raise
            finished = True
        finally:
            if not finished:
                self.manager.abort_request(request.rid)
            await iterator.aclose()
        if isinstance(result, list):
            if len(result) != 1:
                raise _bad_response("The engine returned an unexpected batch.")
            result = result[0]
        if not isinstance(result, dict) or not isinstance(
            result.get("meta_info"), dict
        ):
            raise _bad_response("The engine omitted decision metadata.")
        meta = result["meta_info"]
        if (
            type(meta.get("completion_tokens")) is not int
            or meta["completion_tokens"] != 0
        ):
            raise _bad_response("The engine did not honor zero-output scoring.")
        prompt_tokens = meta.get("prompt_tokens")
        cached_tokens = meta.get("cached_tokens")
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise _bad_response("The engine returned invalid prompt-token usage.")
        if type(cached_tokens) is not int or not 0 <= cached_tokens <= prompt_tokens:
            raise _bad_response("The engine returned invalid cached-token usage.")
        return meta, prompt_tokens, cached_tokens


async def create_systemone_service(
    tokenizer_manager: Any, server_args: Any
) -> SystemOneService:
    """Validate the readout/tokenizer and create the existing decision service."""
    readout_path = _field(server_args, "decision_readout_path")
    if not isinstance(readout_path, str) or not readout_path:
        raise ValueError("--decision-readout-path is required")
    if _field(server_args, "allow_auto_truncate"):
        # A truncated prompt would be decided and billed as if it were whole.
        raise ValueError("decision serving requires allow_auto_truncate=False")
    if (Path(readout_path) / SCHEMA_HEAD_CONFIG).is_file():
        return await _create_schema_service(tokenizer_manager, server_args, readout_path)
    metadata = _load_metadata(readout_path)
    boundary_id = _validate_metadata(metadata, tokenizer_manager, server_args)
    context_len = _field(tokenizer_manager, "context_len")
    reserved = _field(tokenizer_manager, "num_reserved_tokens") or 0
    if type(context_len) is not int or type(reserved) is not int or context_len - reserved < 2:
        raise ValueError("the engine's context length is unavailable")
    max_concurrency = _field(server_args, "max_running_requests") or 32
    max_queued_requests = _field(server_args, "max_queued_requests")
    if type(max_queued_requests) is int and max_queued_requests <= max_concurrency:
        # The adapter keeps max_running_requests evaluations in the scheduler;
        # a smaller scheduler queue would refuse some midway through a request.
        raise ValueError(
            "decision serving bounds its queue with --decision-max-queued-evaluations; "
            "leave --max-queued-requests unset or above --max-running-requests"
        )
    deadline = _field(server_args, "decision_deadline_seconds")
    backend = InProcessDecisionBackend(
        tokenizer_manager,
        metadata,
        _generate_input_type(),
        boundary_id,
        max_prompt_tokens=context_len - reserved - 1,
        max_images=_image_limit(_field(server_args, "limit_mm_data_per_request")),
    )
    serving = serving_defaults_from_env()
    return SystemOneService(
        backend,
        alias=serving["alias"],
        default_temperature=serving["default_temperature"],
        temperature_by_type=serving["temperature_by_type"],
        prompt_wording=serving["prompt_wording"],
        engine_metadata=serving["engine_metadata"],
        max_concurrency=max_concurrency,
        deadline_seconds=DEFAULT_DEADLINE_SECONDS if deadline is None else deadline,
        max_queued_evaluations=_field(server_args, "decision_max_queued_evaluations"),
        remote_images=False,
    )


SCHEMA_HEAD_CONFIG = "schema_head_config.json"


def _engine_limits(tokenizer_manager: Any, server_args: Any) -> dict:
    context_len = _field(tokenizer_manager, "context_len")
    reserved = _field(tokenizer_manager, "num_reserved_tokens") or 0
    if type(context_len) is not int or type(reserved) is not int or context_len - reserved < 2:
        raise ValueError("the engine's context length is unavailable")
    max_concurrency = _field(server_args, "max_running_requests") or 32
    max_queued_requests = _field(server_args, "max_queued_requests")
    if type(max_queued_requests) is int and max_queued_requests <= max_concurrency:
        raise ValueError(
            "decision serving bounds its queue with --decision-max-queued-evaluations; "
            "leave --max-queued-requests unset or above --max-running-requests"
        )
    deadline = _field(server_args, "decision_deadline_seconds")
    return {
        "max_prompt_tokens": context_len - reserved - 1,
        "max_images": _image_limit(_field(server_args, "limit_mm_data_per_request")),
        "max_concurrency": max_concurrency,
        "deadline_seconds": DEFAULT_DEADLINE_SECONDS if deadline is None else deadline,
        "max_queued_evaluations": _field(server_args, "decision_max_queued_evaluations"),
    }


class InProcessSchemaBackend(InProcessDecisionBackend):
    """One prefill per request (and option rotation) through a joint schema head."""

    supports_return_logprobs = True

    def __init__(self, manager: Any, request_type: type, *, max_prompt_tokens, max_images):
        self.manager = manager
        self.model = manager.served_model_name or manager.model_path
        self._request_type = request_type
        self._max_prompt_tokens = max_prompt_tokens
        self._max_images = max_images
        self._warned_image_count = False

    def labels(self, count: int):
        raise AdapterError("labels_unavailable", "Schema models do not use answer labels.")

    async def image_tokens(self, images: Sequence[str]) -> list[int]:
        """Expanded token count of each image, before any GPU work."""
        self._check_images_supported(images)
        if self._max_images is not None and len(images) > self._max_images:
            raise AdapterError(
                "image_limit_exceeded",
                f"This model accepts at most {self._max_images} images per request.",
                "images",
            )
        if not images:
            return []
        counter = getattr(_field(self.manager, "mm_processor"), "decision_image_tokens", None)
        if not callable(counter):
            raise AdapterError(
                "image_processor_unavailable",
                "Schema image scoring needs the processor's image token count.",
                "images",
            )

        def count_all():
            counts = []
            for image in images:
                try:
                    counts.append(counter(image))
                except ValueError as error:
                    raise AdapterError(
                        "invalid_image", "An image could not be decoded.", "images"
                    ) from error
            return counts

        return await asyncio.to_thread(count_all)

    async def prepare(self, prompts, images, assistant_prefix):
        if assistant_prefix:
            raise AdapterError(
                "assistant_prefix_not_supported",
                "Schema models require an empty assistant prefix.",
                "options.assistant_prefix",
            )
        limit = self._max_prompt_tokens
        for prompt in prompts:
            if limit is not None and prompt.plan["n_tokens"] > limit:
                raise AdapterError(
                    "context_length_exceeded",
                    f"The request needs {prompt.plan['n_tokens']} input tokens; the model "
                    f"accepts at most {limit}. Shorten the state, questions or images.",
                    "images" if images else "state",
                )
        return [(prompt, tuple(images)) for prompt in prompts]

    async def evaluate_prepared(self, prepared) -> ScoringResult:
        prompt, images = prepared
        meta, prompt_tokens, cached_tokens = await self._generate(
            list(prompt.input_ids), images, decision_schema=prompt.plan
        )
        logits = meta.get("decision_logits")
        if (
            not isinstance(logits, list)
            or len(logits) != prompt.plan["n_values"]
            or any(
                type(logit) not in (int, float) or not math.isfinite(logit)
                for logit in logits
            )
        ):
            raise _bad_response("The engine returned malformed schema logits.")
        if prompt_tokens != prompt.plan["n_tokens"]:
            raise _bad_response("The engine's prompt differs from the schema plan.")
        image_tokens = self._image_token_usage(
            prompt_tokens, prompt.text_tokens, images, prompt.plan["n_tokens"]
        )
        return ScoringResult(tuple(float(v) for v in logits), prompt_tokens, cached_tokens, image_tokens)


class SchemaService(SystemOneService):
    """System One over a joint schema head: one engine prefill per option rotation."""

    def __init__(self, backend: InProcessSchemaBackend, renderer, **kwargs):
        super().__init__(backend, **kwargs)
        self.renderer = renderer

    async def score(self, request) -> dict:
        from .schema import answer_from_logits, rotation_orders, schema_questions, split_logits

        if request.model not in {self.backend.model, self.alias}:
            raise AdapterError("model_not_found", "Requested model is not configured.", "model", 404)
        if not self.remote_images and any(
            DATA_URL.fullmatch(image) is None for image in request.images
        ):
            raise AdapterError(
                "invalid_image", "Images must be base64 PNG, JPEG or WebP data URLs.", "images"
            )
        started = time.perf_counter()
        options = self.effective_options(request)
        questions = schema_questions(request)
        image_tokens = await self.backend.image_tokens(request.images)
        prompts = await asyncio.to_thread(
            lambda: [
                self.renderer.encode(
                    request.state, questions, rotation_orders(questions, rotation), image_tokens
                )
                for rotation in range(options.permutations)
            ]
        )
        self._admit(len(prompts))
        try:
            results = await self._run(prompts, request.images, options.assistant_prefix)
        finally:
            self._pending -= len(prompts)
        per_question = [split_logits(questions, result.logprobs) for result in results]
        answers = {}
        for i, question in enumerate(questions):
            question_options = self.question_options(request, options, question.type)
            temperature = (
                question_options.temperature if question_options.temperature_scaling else 1.0
            )
            answers[question.key] = answer_from_logits(
                question,
                [vectors[i] for vectors in per_question],
                temperature,
                options.return_logprobs,
            )
        usage = {"input_tokens": sum(r.input_tokens for r in results), "output_tokens": 0}
        if request.images:
            usage["input_tokens_details"] = {"image_tokens": sum(r.image_tokens for r in results)}
        return {
            "model": self.backend.model,
            "answers": answers,
            "usage": usage,
            "metadata": {
                "confidence_method": "1 - normalized_entropy",
                "temperature": options.temperature if options.temperature_scaling else 1.0,
                **(
                    {"temperature_by_type": dict(self.temperature_by_type)}
                    if self.temperature_by_type
                    and "temperature" not in request.options.model_fields_set
                    and options.temperature_scaling
                    else {}
                ),
                # Runners count one evaluation per question; the engine ran
                # one prefill per option rotation for the whole request.
                "evaluations": len(questions) * len(prompts),
                "prefills": len(prompts),
                "label_scheme": "schema-v1",
                **(
                    {
                        "service_max_concurrency": self.max_concurrency,
                        "cached_tokens": sum(r.cached_tokens or 0 for r in results),
                    }
                    if self.engine_metadata
                    else {}
                ),
                "adapter_elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        }


async def _create_schema_service(tokenizer_manager: Any, server_args: Any, head_dir: str):
    from .schema import TEMPLATE_ID, SchemaRenderer

    try:
        config = json.loads((Path(head_dir) / SCHEMA_HEAD_CONFIG).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {SCHEMA_HEAD_CONFIG}") from error
    head = config.get("head") or {}
    if head.get("template") != TEMPLATE_ID or config.get("preview", True) is not True:
        raise ValueError("this adapter renders only the schema-v1 template with question preview")
    if _field(server_args, "speculative_algorithm") is not None:
        raise ValueError("decision schema serving does not support speculation or MTP")
    limits = _engine_limits(tokenizer_manager, server_args)
    tokenizer_path = _field(server_args, "tokenizer_path") or _field(server_args, "model_path")
    renderer = await asyncio.to_thread(SchemaRenderer.from_pretrained, tokenizer_path)
    backend = InProcessSchemaBackend(
        tokenizer_manager,
        _generate_input_type(),
        max_prompt_tokens=limits["max_prompt_tokens"],
        max_images=limits["max_images"],
    )
    serving = serving_defaults_from_env()
    return SchemaService(
        backend,
        renderer,
        alias=serving["alias"],
        default_temperature=serving["default_temperature"],
        temperature_by_type=serving["temperature_by_type"],
        prompt_wording=serving["prompt_wording"],
        engine_metadata=serving["engine_metadata"],
        max_concurrency=limits["max_concurrency"],
        deadline_seconds=limits["deadline_seconds"],
        max_queued_evaluations=limits["max_queued_evaluations"],
        remote_images=False,
    )


def serving_defaults_from_env(environ: Any = None) -> dict:
    """Request defaults for the in-process service, from the standalone CLI's
    environment variables, so one engine can serve a released adapter config.

    JEV_DEFAULT_TEMPERATURE / JEV_TEMPERATURE_BY_TYPE apply only when a request
    omits options.temperature, exactly as the standalone --default-temperature /
    --temperature-by-type do; they multiply a GPU readout's own fixed
    temperature. JEV_PROMPT_WORDING selects served (default) or native text;
    native never adds a system message here (the standalone
    --native-system-prompt none behavior). JEV_MODEL_ALIAS names the accepted
    model alias besides the served model name. Unset values keep the previous
    fixed defaults: 1.0, served, decision-model.
    """
    import os

    env = os.environ if environ is None else environ
    raw = env.get("JEV_DEFAULT_TEMPERATURE", "").strip()
    default_temperature = 1.0
    if raw:
        try:
            default_temperature = float(raw)
        except ValueError:
            raise ValueError("JEV_DEFAULT_TEMPERATURE must be a number") from None
        if not math.isfinite(default_temperature) or default_temperature <= 0:
            raise ValueError("JEV_DEFAULT_TEMPERATURE must be finite and positive")
    temperature_by_type: dict[str, float] = {}
    for part in filter(
        None, (item.strip() for item in env.get("JEV_TEMPERATURE_BY_TYPE", "").split(","))
    ):
        key, sep, value = (item.strip() for item in part.partition("="))
        if not sep or key not in ("choice", "noul", "score") or key in temperature_by_type:
            raise ValueError(
                f"JEV_TEMPERATURE_BY_TYPE expects type=T pairs; got {part!r}"
            )
        try:
            temperature = float(value)
        except ValueError:
            raise ValueError(f"JEV_TEMPERATURE_BY_TYPE value is not a number: {part!r}") from None
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("JEV_TEMPERATURE_BY_TYPE values must be finite and positive")
        temperature_by_type[key] = temperature
    wording = env.get("JEV_PROMPT_WORDING", "").strip() or "served"
    if wording not in ("served", "native"):
        raise ValueError('JEV_PROMPT_WORDING must be "served" or "native"')
    alias = env.get("JEV_MODEL_ALIAS", "").strip() or "decision-model"
    engine_metadata = env.get("JEV_ENGINE_METADATA", "").strip()
    if engine_metadata not in ("", "0", "1"):
        raise ValueError('JEV_ENGINE_METADATA must be "0" or "1"')
    return {
        "alias": alias,
        "default_temperature": default_temperature,
        "temperature_by_type": temperature_by_type or None,
        "prompt_wording": wording,
        # Released adapters answer without service_max_concurrency/cached_tokens.
        "engine_metadata": engine_metadata == "1",
    }


def _image_limit(value: Any) -> int | None:
    """The per-request image count from --limit-mm-data-per-request, if any."""
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("--limit-mm-data-per-request must be a JSON object")
    limit = value.get("image")
    if limit is None:
        return None
    if type(limit) is not int or limit < 0:
        raise ValueError("--limit-mm-data-per-request image must be a nonnegative integer")
    return limit
