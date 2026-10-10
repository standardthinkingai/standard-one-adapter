"""Optional CPU chat tokenization that preserves native control-token IDs.

This path is useful when an engine's string-based chat API cannot roundtrip a
tokenizer's control tokens. It loads tokenizer assets, never model weights.
"""

from typing import Any

from .backend import AdapterError


class NativeTokenizer:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model: str, revision: str) -> "NativeTokenizer":
        if not model.strip() or not revision.strip():
            raise ValueError("Native tokenizer model and pinned revision are required")
        # Keep the default HTTP-only installation free of Transformers.
        from transformers import AutoTokenizer

        return cls(
            AutoTokenizer.from_pretrained(
                model,
                revision=revision,
                trust_remote_code=False,
            )
        )

    def _chat(self, messages: list[dict], *, continuation: bool) -> list[int]:
        try:
            tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=not continuation,
                continue_final_message=continuation,
                return_dict=False,
                enable_thinking=False,
            )
        except Exception as error:
            raise AdapterError(
                "unsupported_native_tokenizer",
                "The native tokenizer could not prepare an open assistant message.",
            ) from error
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise AdapterError(
                "unsupported_native_tokenizer",
                "The native tokenizer returned invalid chat token IDs.",
            )
        return tokens

    def _chat_many(
        self, conversations: list[list[dict]], *, continuation: bool
    ) -> list[list[int]]:
        """Render label continuations together when the tokenizer supports batches.

        A batch is only a tokenization optimization. Unsupported or malformed
        batch output falls back to the individually validated native path.
        """
        try:
            batches = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=not continuation,
                continue_final_message=continuation,
                return_dict=False,
                enable_thinking=False,
            )
        except (TypeError, ValueError, AttributeError, NotImplementedError, IndexError):
            batches = None
        if (
            isinstance(batches, list)
            and len(batches) == len(conversations)
            and all(
                isinstance(tokens, list)
                and tokens
                and all(type(token) is int and token >= 0 for token in tokens)
                for tokens in batches
            )
        ):
            return batches
        return [
            self._chat(messages, continuation=continuation)
            for messages in conversations
        ]

    def prepare(
        self,
        prompt: str,
        labels: tuple[str, ...],
        token_ids: tuple[int, ...],
        assistant_prefix: str | None,
        system_prompt: str | None = None,
    ) -> list[int]:
        """Return native prompt IDs only when every answer continues by one token.

        Full native conversations are tokenized for both the base and each
        continuation. Decoding and re-encoding control tokens is never used.
        Comparing against engine-selected IDs also detects tokenizer mismatch.

        ``system_prompt``, when given, is prepended as an explicit system
        message rather than relying on the tokenizer's own auto-injection: a
        mistral-common-backed tokenizer never auto-injects a default system
        prompt (unlike the HF Jinja chat template), so wording="native"
        serving supplies the runner's default system text explicitly here
        (see ``extract_default_system_prompt``/``native_default_system_prompt``).
        This can still render to different token IDs than the HF template's
        own auto-injection; that gap is measured, not assumed away.
        """
        if (
            len(labels) < 2
            or len(labels) != len(token_ids)
            or len(set(labels)) != len(labels)
            or len(set(token_ids)) != len(token_ids)
            or any(not isinstance(label, str) or not label for label in labels)
            or any(type(token) is not int or token < 0 for token in token_ids)
        ):
            raise AdapterError(
                "invalid_labels", "Distinct label/token pairs are required."
            )
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + [{"role": "user", "content": prompt}]
        base = self._chat(messages, continuation=False)
        prefix = assistant_prefix or ""
        final_ids = base
        if prefix:
            final_ids = self._chat(
                messages + [{"role": "assistant", "content": prefix}],
                continuation=True,
            )
            if final_ids[: len(base)] != base:
                raise AdapterError(
                    "invalid_label_boundary",
                    "The assistant prefix changes the native chat prompt boundary.",
                    "options.assistant_prefix",
                )
        conversations = [
            messages + [{"role": "assistant", "content": prefix + label}]
            for label in labels
        ]
        continuations = (
            self._chat_many(conversations, continuation=True)
            if len(conversations) >= 3
            else [self._chat(item, continuation=True) for item in conversations]
        )
        for continued, token_id in zip(continuations, token_ids):
            if continued != final_ids + [token_id]:
                raise AdapterError(
                    "invalid_label_boundary",
                    "A native assistant label does not continue by its engine token ID; "
                    "use matching tokenizers and a compatible assistant_prefix.",
                    "options.assistant_prefix",
                )
        return final_ids

    @classmethod
    def native_default_system_prompt(cls, model: str, revision: str) -> "str | None":
        """The jevbench-hard runner's default system-prompt text for ``model``
        @``revision``: a dedicated HF fast-tokenizer load with the runner's own
        ``fix_mistral_regex=True`` (for ``mistralai/*`` models), independent of
        whichever tokenizer class plain ``from_pretrained`` resolves to for
        serving -- for Ministral that is a mistral-common backend (see the
        module docstring and ``extract_default_system_prompt``), which never
        performs this auto-injection, so probing the serving tokenizer itself
        would incorrectly report "no default system prompt".

        Returns None when the model's template defines no default system
        message. Requires transformers; call only for wording="native".
        """
        if not model.strip() or not revision.strip():
            raise ValueError("Native tokenizer model and pinned revision are required")
        from transformers import AutoTokenizer

        kwargs: dict[str, Any] = {"revision": revision, "trust_remote_code": False}
        if model.startswith("mistralai/"):
            kwargs["fix_mistral_regex"] = True
        tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        return extract_default_system_prompt(tokenizer)


def extract_default_system_prompt(tokenizer: Any) -> "str | None":
    """Return the text a chat template auto-injects as a default system
    message, or None if rendering a system-free conversation injects none.

    Works by rendering one throwaway user turn and looking for the literal
    ``[SYSTEM_PROMPT]``/``[/SYSTEM_PROMPT]`` markers Mistral's HF Jinja
    template wraps its auto-injected default system message in. A
    mistral-common-backed tokenizer implements the same ``apply_chat_template``
    surface but never performs this injection (see run_suites_ablate.py's
    native_mc contract), so calling this on one correctly returns None --
    that is the documented tokenizer-path difference wording="native" serving
    has to work around by supplying the text explicitly instead of relying on
    auto-injection (see ``NativeTokenizer.prepare``'s ``system_prompt``).
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or "[SYSTEM_PROMPT]" not in rendered:
        return None
    return rendered.split("[SYSTEM_PROMPT]", 1)[1].split("[/SYSTEM_PROMPT]", 1)[0]
