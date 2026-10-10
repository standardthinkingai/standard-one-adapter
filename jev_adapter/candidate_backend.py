"""Optional selected-row readout over SGLang's last hidden state.

For Mistral3's SGLang Llama language-model path, the captured last state is
the normalized LM-head input. This backend deliberately accepts text only and
keeps the existing SGLang logprob backend as the default.
"""

import uuid

from .backend import AdapterError, ScoringResult
from .readout import CandidateReadout
from .sglang import SGLangBackend, _bad_response


class CandidateReadoutBackend(SGLangBackend):
    supports_return_logprobs = False

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        readout_path: str,
        tokenizer_model: str,
        tokenizer_revision: str,
        **kwargs,
    ):
        if kwargs.get("native_tokenizer") is None:
            raise ValueError("candidate readout requires the pinned native tokenizer")
        if not tokenizer_model or not tokenizer_revision:
            raise ValueError("candidate readout requires model ID and revision")
        super().__init__(base_url, model, **kwargs)
        self._readout = CandidateReadout.load(readout_path)
        if self._readout.label_scheme != self.label_scheme:
            raise ValueError(
                f"candidate readout uses the {self._readout.label_scheme!r} label "
                f"scheme, not the configured {self.label_scheme!r}"
            )
        self._tokenizer_model = tokenizer_model
        self._tokenizer_revision = tokenizer_revision

    async def start(self) -> None:
        if self._ready:
            return
        await super().start()
        try:
            self._readout.validate_engine(
                model_info=self._model_info,
                server_info=self._server_info,
                tokenizer_model=self._tokenizer_model,
                tokenizer_revision=self._tokenizer_revision,
                labels=self._label_strings,
                token_ids=self._label_ids,
            )
            # Check the server flag and the hidden dimension before accepting
            # traffic. The probe uses the same hidden-only hot path.
            await self.evaluate(
                "Readout startup probe.",
                [],
                self._label_strings[:2],
                self._label_ids[:2],
                None,
            )
        except (AdapterError, ValueError):
            self._ready = False
            raise

    async def evaluate(
        self,
        prompt: str,
        images: list[str],
        labels: tuple[str, ...],
        token_ids: tuple[int, ...],
        assistant_prefix: str | None,
    ) -> ScoringResult:
        self._validate_labels(labels, token_ids)
        if images:
            raise AdapterError(
                "images_not_supported",
                "Candidate readout scoring accepts text only.",
                "images",
            )
        final_ids = await self._prepare_native(
            prompt, labels, token_ids, assistant_prefix
        )
        result = await self._request(
            "POST",
            "/generate",
            {
                "input_ids": final_ids,
                "sampling_params": {
                    "max_new_tokens": 0,
                    "temperature": 1,
                    "top_p": 1,
                    "top_k": -1,
                },
                "return_hidden_states": "last",
                "stream": False,
                "rid": "jev-" + uuid.uuid4().hex,
            },
        )
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
        try:
            logits = self._readout.score(meta.get("hidden_states"), token_ids)
        except ValueError as error:
            raise _bad_response(str(error)) from error
        # These are selected raw logits. The service's selected-label softmax
        # is invariant to the unavailable full-vocabulary logsumexp.
        return ScoringResult(logits, input_tokens, cached_tokens)
