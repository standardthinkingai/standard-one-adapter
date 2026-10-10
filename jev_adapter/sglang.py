"""Prefill-only scoring through stock SGLang HTTP APIs.

Chat tokenization uses the inference server by default. An optional native CPU
tokenizer preserves control-token IDs for engines with a lossy string chat API.
No SGLang, CUDA, or model weights are imported into this process.
"""

import asyncio
import math
import uuid
from typing import Any

import httpx

from .backend import AdapterError, ScoringResult
from .labels import (
    DEFAULT_LABEL_SCHEME,
    MAX_LABELS,
    candidate_tiers,
    validate_label_scheme,
)


_IMAGE_MODEL_TYPES = {
    "mage_vl",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
    "mistral3",
    "pixtral",
}
_IMAGE_ARCHITECTURES = {
    "MageVLForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Mistral3ForConditionalGeneration",
    "PixtralForConditionalGeneration",
}


def _bad_response(message: str) -> AdapterError:
    return AdapterError("invalid_engine_response", message, status=502)


def _token_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        type(token) is int and token >= 0 for token in value
    )


class SGLangBackend:
    supports_return_logprobs = True

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120,
        max_connections: int = 32,
        client: httpx.AsyncClient | None = None,
        native_tokenizer: Any | None = None,
        prompt_wording: str = "served",
        native_system_prompt: str | None = None,
        http_transport: str = "httpx",
        label_scheme: str = DEFAULT_LABEL_SCHEME,
    ):
        if not math.isfinite(timeout) or timeout <= 0 or max_connections < 1:
            raise ValueError("timeout and max_connections must be positive")
        if prompt_wording not in ("served", "native"):
            raise ValueError('prompt_wording must be "served" or "native"')
        # Which candidate order answer labels come from (jev_adapter.labels).
        self.label_scheme = validate_label_scheme(label_scheme)
        if http_transport not in ("httpx", "aiohttp"):
            raise ValueError('http_transport must be "httpx" or "aiohttp"')
        url = httpx.URL(base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.host
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) origin or path without query or fragment"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._owns_client = client is None
        if client is not None:
            self._client = client
        elif http_transport == "aiohttp":
            from .http_client import AiohttpCompatClient

            self._client = AiohttpCompatClient(
                concurrency=max_connections, timeout=timeout
            )
        else:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=max_connections),
                timeout=timeout,
            )
        self._startup_lock = asyncio.Lock()
        self._label_strings: tuple[str, ...] = ()
        self._label_ids: tuple[int, ...] = ()
        self._image_capable = False
        self._image_format_supported = False
        self._ready = False
        self._model_info: dict = {}
        self._server_info: dict = {}
        self._native_tokenizer = native_tokenizer
        self._prompt_wording = prompt_wording
        # Only used when prompt_wording == "native": the runner's default
        # system-prompt text, prepended explicitly as a system message (see
        # native_tokenizer.extract_default_system_prompt/NativeTokenizer.prepare).
        # None means "wording changed, but no system message is added" --
        # either the wording is "served", or no such text was supplied/found.
        self._native_system_prompt = native_system_prompt if prompt_wording == "native" else None

    async def _request(
        self, method: str, path: str, payload: dict | None = None
    ) -> dict:
        try:
            response = await self._client.request(
                method,
                self.base_url + path,
                json=payload,
                headers=self._headers,
                timeout=self.timeout,
            )
        except httpx.TimeoutException as error:
            raise AdapterError(
                "engine_timeout", "The inference engine timed out.", status=504
            ) from error
        except httpx.HTTPError as error:
            raise AdapterError(
                "engine_unavailable",
                "Could not reach the inference engine.",
                status=502,
            ) from error
        if response.status_code == 404:
            raise AdapterError(
                "engine_endpoint_missing",
                f"The inference engine does not provide {path}.",
                status=502,
            )
        if response.status_code in (400, 422):
            raise AdapterError(
                "engine_rejected_request",
                "The inference engine rejected the scoring input.",
                status=422,
            )
        if not response.is_success:
            raise AdapterError(
                "engine_http_error",
                f"The inference engine returned HTTP {response.status_code}.",
                status=502,
            )
        try:
            result = response.json()
        except ValueError as error:
            raise _bad_response(
                "The inference engine returned invalid JSON."
            ) from error
        if not isinstance(result, dict) or "error" in result:
            raise _bad_response("The inference engine returned an unexpected response.")
        return result

    async def _information(self, path: str, legacy: str) -> dict:
        try:
            return await self._request("GET", path)
        except AdapterError as error:
            if error.code != "engine_endpoint_missing":
                raise
            return await self._request("GET", legacy)

    async def start(self) -> None:
        async with self._startup_lock:
            if self._ready:
                return
            info = await self._information("/model_info", "/get_model_info")
            if info.get("is_generation") is not True:
                raise AdapterError(
                    "unsupported_model", "A generation model is required."
                )
            served = info.get("served_model_name")
            if not isinstance(served, str) or not served:
                raise _bad_response(
                    "The inference engine did not identify its served model."
                )
            if self.model not in (served, info.get("model_path")):
                raise AdapterError(
                    "model_not_found",
                    "Configured model is not served by the inference engine.",
                    "model",
                    404,
                )
            config = await self._information("/server_info", "/get_server_info")
            if config.get("skip_tokenizer_init"):
                raise AdapterError(
                    "tokenizer_unavailable",
                    "The inference engine must initialize its tokenizer.",
                )
            if config.get("speculative_algorithm") is not None:
                raise AdapterError(
                    "speculation_not_supported",
                    "Disable speculative decoding and MTP for prefill-only scoring.",
                )
            self.model = served
            self._image_capable = info.get("has_image_understanding") is True
            architectures = info.get("architectures") or []
            self._image_format_supported = info.get(
                "model_type"
            ) in _IMAGE_MODEL_TYPES or (
                isinstance(architectures, list)
                and any(
                    arch in _IMAGE_ARCHITECTURES
                    for arch in architectures
                    if isinstance(arch, str)
                )
            )
            valid: list[tuple[str, int]] = []
            seen: set[int] = set()
            # A later candidate group is tokenized only while labels are short.
            for candidates in candidate_tiers(self.label_scheme):
                if len(valid) == MAX_LABELS:
                    break
                await self._select_tier(list(candidates), valid, seen)
            if len(valid) < 2:
                raise AdapterError(
                    "labels_unavailable",
                    "At least two distinct, non-special single-token labels are required.",
                )
            self._label_strings, self._label_ids = tuple(zip(*valid))
            self._model_info = info
            self._server_info = config
            self._ready = True

    async def _select_tier(
        self, candidates: list[str], valid: list[tuple[str, int]], seen: set[int]
    ) -> None:
        """Append this group's reversible single-token labels, in order."""
        tokenized = await self._request(
            "POST",
            "/v1/tokenize",
            {
                "model": self.model,
                "prompt": candidates,
                "add_special_tokens": False,
            },
        )
        batches = tokenized.get("tokens")
        if (
            not isinstance(batches, list)
            or len(batches) != len(candidates)
            or not all(_token_list(ids) for ids in batches)
        ):
            raise _bad_response(
                "The inference engine returned malformed label tokenization."
            )
        singles = [
            (label, ids[0]) for label, ids in zip(candidates, batches) if len(ids) == 1
        ]
        if not singles:
            return
        decoded = await self._request(
            "POST",
            "/v1/detokenize",
            {
                "model": self.model,
                "tokens": [[token] for _, token in singles],
                "skip_special_tokens": True,
            },
        )
        texts = decoded.get("text")
        if (
            not isinstance(texts, list)
            or len(texts) != len(singles)
            or not all(isinstance(item, str) for item in texts)
        ):
            raise _bad_response(
                "The inference engine returned malformed label detokenization."
            )
        for (label, token), decoded_label in zip(singles, texts):
            if label == decoded_label and token not in seen:
                valid.append((label, token))
                seen.add(token)
                if len(valid) == MAX_LABELS:
                    return

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def labels(self, count: int) -> tuple[tuple[str, ...], tuple[int, ...]]:
        if not self._ready:
            raise AdapterError(
                "engine_not_ready", "The scoring backend has not started.", status=503
            )
        if type(count) is not int or count < 2 or count > len(self._label_strings):
            raise AdapterError(
                "labels_unavailable",
                f"This tokenizer supports 2 to {len(self._label_strings)} answer labels.",
                "questions",
            )
        return self._label_strings[:count], self._label_ids[:count]

    async def evaluate(
        self,
        prompt: str,
        images: list[str],
        labels: tuple[str, ...],
        token_ids: tuple[int, ...],
        assistant_prefix: str | None,
    ) -> ScoringResult:
        self._validate_labels(labels, token_ids)
        if images and self._native_tokenizer is not None:
            raise AdapterError(
                "native_images_not_supported",
                "Native tokenizer scoring currently accepts text only.",
                "images",
            )
        if images and not self._image_capable:
            raise AdapterError(
                "images_not_supported",
                "The loaded model does not support images.",
                "images",
            )
        if images and not self._image_format_supported:
            raise AdapterError(
                "image_model_not_supported",
                "Image scoring currently supports Mage-VL, Qwen VL/Qwen3.5, and Mistral3/Pixtral chat formats.",
                "images",
            )
        if self._native_tokenizer is not None:
            final_ids = await self._prepare_native(
                prompt, labels, token_ids, assistant_prefix
            )
            final_text = None
        else:
            final_ids, final_text = await self._prepare_http(
                prompt,
                images,
                labels,
                token_ids,
                assistant_prefix,
                self._native_system_prompt,
            )
        payload = {
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 1,
                "top_p": 1,
                "top_k": -1,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
            "token_ids_logprob": list(token_ids),
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
            "stream": False,
            "rid": "jev-" + uuid.uuid4().hex,
        }
        if images:
            payload.update(text=final_text, image_data=images)
        else:
            payload["input_ids"] = final_ids
        result = await self._request("POST", "/generate", payload)
        meta = result.get("meta_info")
        if not isinstance(meta, dict):
            raise _bad_response("The inference engine omitted scoring metadata.")
        if (
            type(meta.get("completion_tokens")) is not int
            or meta["completion_tokens"] != 0
        ):
            raise _bad_response(
                "The inference engine did not honor zero-output scoring."
            )
        input_tokens = meta.get("prompt_tokens")
        if type(input_tokens) is not int or input_tokens < 1:
            raise _bad_response(
                "The inference engine returned invalid input-token usage."
            )
        cached_tokens = meta.get("cached_tokens")
        if (
            type(cached_tokens) is not int
            or not 0 <= cached_tokens <= input_tokens
        ):
            raise _bad_response("The inference engine returned invalid cache usage.")
        positions = meta.get("output_token_ids_logprobs")
        if (
            not isinstance(positions, list)
            or len(positions) != 1
            or not isinstance(positions[0], list)
        ):
            raise _bad_response(
                "The inference engine omitted the requested next-token logprobs."
            )
        values: dict[int, float] = {}
        for entry in positions[0]:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise _bad_response("The inference engine returned malformed logprobs.")
            value, token = entry[:2]
            if (
                type(token) is not int
                or type(value) not in (int, float)
                or not math.isfinite(value)
                or value > 0
                or token in values
            ):
                raise _bad_response("The inference engine returned invalid logprobs.")
            values[token] = float(value)
        if set(values) != set(token_ids):
            raise _bad_response(
                "The inference engine did not return every requested label logprob."
            )
        return ScoringResult(
            tuple(values[token] for token in token_ids), input_tokens, cached_tokens
        )

    def _validate_labels(
        self, labels: tuple[str, ...], token_ids: tuple[int, ...]
    ) -> None:
        if not self._ready:
            raise AdapterError(
                "engine_not_ready", "The scoring backend has not started.", status=503
            )
        if (
            len(labels) < 2
            or len(labels) != len(token_ids)
            or len(set(token_ids)) != len(token_ids)
        ):
            raise AdapterError(
                "invalid_labels", "Distinct label/token pairs are required."
            )
        known = dict(zip(self._label_strings, self._label_ids))
        if any(known.get(label) != token for label, token in zip(labels, token_ids)):
            raise AdapterError(
                "invalid_labels", "Labels must come from this backend's tokenizer."
            )

    async def _prepare_native(
        self,
        prompt: str,
        labels: tuple[str, ...],
        token_ids: tuple[int, ...],
        assistant_prefix: str | None,
    ) -> list[int]:
        final_ids = await asyncio.to_thread(
            self._native_tokenizer.prepare,
            prompt,
            labels,
            token_ids,
            assistant_prefix,
            self._native_system_prompt,
        )
        if not _token_list(final_ids) or not final_ids:
            raise _bad_response("The native tokenizer returned invalid prompt IDs.")
        return final_ids

    async def _prepare_http(
        self,
        prompt: str,
        images: list[str],
        labels: tuple[str, ...],
        token_ids: tuple[int, ...],
        assistant_prefix: str | None,
        system_prompt: str | None = None,
    ) -> tuple[list[int], str]:
        content: Any = prompt
        if images:
            content = [
                {"type": "image_url", "image_url": {"url": image}} for image in images
            ]
            content.append({"type": "text", "text": prompt})
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + [{"role": "user", "content": content}]
        rendered = await self._request(
            "POST",
            "/v1/tokenize",
            {
                "model": self.model,
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        prompt_ids = rendered.get("tokens")
        if not _token_list(prompt_ids) or not prompt_ids:
            raise _bad_response(
                "The inference engine returned malformed chat tokenization."
            )
        decoded = await self._request(
            "POST",
            "/v1/detokenize",
            {
                "model": self.model,
                "tokens": prompt_ids,
                "skip_special_tokens": False,
            },
        )
        rendered_text = decoded.get("text")
        if not isinstance(rendered_text, str) or not rendered_text:
            raise _bad_response(
                "The inference engine returned malformed chat detokenization."
            )
        final_text = rendered_text + (assistant_prefix or "")
        # Check the complete template roundtrip and the actual assistant boundary.
        # Checking labels in isolation can score the wrong tokens for BPE tokenizers.
        prompts = [rendered_text, final_text] + [final_text + label for label in labels]
        checked = await self._request(
            "POST",
            "/v1/tokenize",
            {
                "model": self.model,
                "prompt": prompts,
                "add_special_tokens": False,
            },
        )
        variants = checked.get("tokens")
        if (
            not isinstance(variants, list)
            or len(variants) != len(prompts)
            or not all(_token_list(ids) for ids in variants)
        ):
            raise _bad_response(
                "The inference engine returned malformed boundary tokenization."
            )
        if variants[0] != prompt_ids:
            raise AdapterError(
                "unsupported_tokenizer_roundtrip",
                "The rendered chat prompt cannot be losslessly tokenized through this HTTP adapter.",
            )
        final_ids = variants[1]
        if not final_ids:
            raise _bad_response("The formatted scoring prompt is empty.")
        if any(
            ids != final_ids + [token] for ids, token in zip(variants[2:], token_ids)
        ):
            raise AdapterError(
                "invalid_label_boundary",
                "Labels are not single tokens at the assistant boundary; configure a compatible assistant_prefix.",
                "options.assistant_prefix",
            )
        return final_ids, final_text
