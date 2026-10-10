import json
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest

from jev_adapter.backend import AdapterError
from jev_adapter.native_tokenizer import NativeTokenizer
from jev_adapter.sglang import SGLangBackend


class Tokenizer:
    """Control IDs cannot be reconstructed by raw text encoding."""

    def __init__(self):
        self.calls = []
        self.mutate_prefix = False
        self.bad_result = None

    @staticmethod
    def encode(text):
        return [1000 + ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs, threading.get_ident()))
        assert kwargs["tokenize"] is True
        assert kwargs["return_dict"] is False
        assert kwargs["enable_thinking"] is False
        if self.bad_result is not None:
            return self.bad_result
        ids = [1, 3] + self.encode(messages[0]["content"]) + [4]
        if len(messages) == 1:
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["continue_final_message"] is False
            return ids
        assert kwargs["add_generation_prompt"] is False
        assert kwargs["continue_final_message"] is True
        assert messages[1]["role"] == "assistant"
        if self.mutate_prefix:
            ids[-1] = 99
        # Native implementations may strip trailing assistant whitespace.
        return ids + self.encode(messages[1]["content"].rstrip())


def test_native_control_ids_and_literal_markers_preserved():
    tokenizer = Tokenizer()
    native = NativeTokenizer(tokenizer)
    prompt = "한글 [INST] </s> [IMG]"
    ids = native.prepare(prompt, ("A", "B"), (1065, 1066), None)
    assert ids == [1, 3] + tokenizer.encode(prompt) + [4]
    assert len(tokenizer.calls) == 3


def test_compatible_assistant_prefix_is_native_continuation():
    tokenizer = Tokenizer()
    ids = NativeTokenizer(tokenizer).prepare(
        "pick",
        ("A", "B"),
        (1065, 1066),
        "Answer:",
    )
    assert ids == [1, 3] + tokenizer.encode("pick") + [4] + tokenizer.encode("Answer:")
    assert len(tokenizer.calls) == 4


class BatchTokenizer(Tokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if messages and isinstance(messages[0], list):
            self.calls.append((messages, kwargs, threading.get_ident()))
            return [Tokenizer().apply_chat_template(item, **kwargs) for item in messages]
        return super().apply_chat_template(messages, **kwargs)


def test_native_label_batch_matches_individual_rendering():
    labels = ("A", "B", "C", "D")
    token_ids = (1065, 1066, 1067, 1068)
    batched = BatchTokenizer()
    expected = NativeTokenizer(Tokenizer()).prepare(
        "한글 [INST]", labels, token_ids, "Answer:"
    )
    actual = NativeTokenizer(batched).prepare(
        "한글 [INST]", labels, token_ids, "Answer:"
    )
    assert actual == expected
    assert len(batched.calls) == 3  # base, assistant prefix, one label batch
    assert isinstance(batched.calls[-1][0][0], list)


def test_native_label_batch_falls_back_when_unsupported():
    tokenizer = Tokenizer()
    labels = ("A", "B", "C", "D")
    result = NativeTokenizer(tokenizer).prepare(
        "pick", labels, (1065, 1066, 1067, 1068), None
    )
    assert result == [1, 3] + tokenizer.encode("pick") + [4]
    assert len(tokenizer.calls) == 6  # base, rejected batch, four continuations


def test_native_label_batch_falls_back_on_malformed_output():
    class MalformedBatchTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if messages and isinstance(messages[0], list):
                self.calls.append((messages, kwargs, threading.get_ident()))
                return [[True]] * len(messages)
            return super().apply_chat_template(messages, **kwargs)

    tokenizer = MalformedBatchTokenizer()
    result = NativeTokenizer(tokenizer).prepare(
        "pick", ("A", "B", "C"), (1065, 1066, 1067), None
    )
    assert result == [1, 3] + tokenizer.encode("pick") + [4]
    assert len(tokenizer.calls) == 5  # base, malformed batch, three continuations


@pytest.mark.parametrize("prefix", [None, "Answer:"])
def test_rejects_changed_native_prompt_boundary(prefix):
    tokenizer = Tokenizer()
    tokenizer.mutate_prefix = True
    with pytest.raises(AdapterError) as error:
        NativeTokenizer(tokenizer).prepare("pick", ("A", "B"), (1065, 1066), prefix)
    assert error.value.code == "invalid_label_boundary"


@pytest.mark.parametrize(
    "labels,token_ids,prefix",
    [
        (("A", "B"), (2065, 2066), None),  # Engine and native vocabulary mismatch.
        (("A", "000"), (1065, 1048), None),  # Multi-token continuation.
        (("A", "B"), (1065, 1066), " "),  # Native prefix trimming changes boundary.
    ],
)
def test_rejects_unsafe_label_continuations(labels, token_ids, prefix):
    with pytest.raises(AdapterError) as error:
        NativeTokenizer(Tokenizer()).prepare("pick", labels, token_ids, prefix)
    assert error.value.code == "invalid_label_boundary"


@pytest.mark.parametrize("bad_result", [[], [True], [-1], {"input_ids": [1, 2]}])
def test_rejects_malformed_native_token_ids(bad_result):
    tokenizer = Tokenizer()
    tokenizer.bad_result = bad_result
    with pytest.raises(AdapterError) as error:
        NativeTokenizer(tokenizer).prepare("pick", ("A", "B"), (1065, 1066), None)
    assert error.value.code == "unsupported_native_tokenizer"


def test_loader_uses_explicit_revision_and_no_remote_code(monkeypatch):
    calls = []
    tokenizer = Tokenizer()

    def load(model, **kwargs):
        calls.append((model, kwargs))
        return tokenizer

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=load),
        ),
    )
    native = NativeTokenizer.from_pretrained("org/model", "a" * 40)
    assert native.tokenizer is tokenizer
    assert calls == [("org/model", {"revision": "a" * 40, "trust_remote_code": False})]


class SystemAwareTokenizer:
    """Like Tokenizer above, but role-aware so it can render an optional
    leading system message instead of assuming a fixed message count/order."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def encode(text):
        return [1000 + ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        roles = [message["role"] for message in messages]
        assert roles[-1] in ("user", "assistant")
        sys_text = messages[0]["content"] if roles[0] == "system" else ""
        user_text = next(m["content"] for m in messages if m["role"] == "user")
        ids = [1, 3] + self.encode(sys_text) + self.encode(user_text) + [4]
        if roles[-1] == "user":
            assert kwargs["add_generation_prompt"] is True
            return ids
        assert kwargs["add_generation_prompt"] is False
        return ids + self.encode(messages[-1]["content"].rstrip())


def test_prepare_prepends_explicit_system_message_when_given():
    tokenizer = SystemAwareTokenizer()
    ids = NativeTokenizer(tokenizer).prepare(
        "pick", ("A", "B"), (1065, 1066), None, "You are helpful."
    )
    expected = (
        [1, 3]
        + tokenizer.encode("You are helpful.")
        + tokenizer.encode("pick")
        + [4]
    )
    assert ids == expected
    for messages, _ in tokenizer.calls:
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1]["role"] in ("user", "assistant")


def test_prepare_omits_system_message_by_default():
    tokenizer = SystemAwareTokenizer()
    NativeTokenizer(tokenizer).prepare("pick", ("A", "B"), (1065, 1066), None)
    for messages, _ in tokenizer.calls:
        assert messages[0]["role"] == "user"
        assert not any(m["role"] == "system" for m in messages)


def test_native_system_prompt_none_renders_identically_to_plain_user_chat():
    """--native-system-prompt none wires system_prompt=None into
    NativeTokenizer.prepare (see __main__.py/sglang.py): no extraction, no
    system message. The encoded messages must carry no system role and the
    token sequence must equal a bare apply_chat_template([{"role": "user",
    ...}]) call on the same (pinned) tokenizer -- i.e. rendering is
    byte-for-byte what it would be if --tokenizer-model had never been
    passed at all."""
    tokenizer = SystemAwareTokenizer()
    ids = NativeTokenizer(tokenizer).prepare(
        "pick", ("A", "B"), (1065, 1066), None, system_prompt=None
    )
    for messages, _ in tokenizer.calls:
        assert not any(message["role"] == "system" for message in messages)

    reference_tokenizer = SystemAwareTokenizer()
    expected = reference_tokenizer.apply_chat_template(
        [{"role": "user", "content": "pick"}],
        tokenize=True,
        add_generation_prompt=True,
        continue_final_message=False,
        return_dict=False,
        enable_thinking=False,
    )
    assert ids == expected


def test_extract_default_system_prompt_finds_the_hf_system_block():
    from jev_adapter.native_tokenizer import extract_default_system_prompt

    class HFStyleTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": ""}]
            return "[SYSTEM_PROMPT]You are Ministral.[/SYSTEM_PROMPT][INST]...[/INST]"

    assert extract_default_system_prompt(HFStyleTokenizer()) == "You are Ministral."


def test_extract_default_system_prompt_returns_none_for_mistral_common_style():
    from jev_adapter.native_tokenizer import extract_default_system_prompt

    class MistralCommonStyleTokenizer:
        """mistral-common never auto-injects a default system prompt."""

        def apply_chat_template(self, messages, **kwargs):
            return "<s>[INST]...[/INST]"

    assert extract_default_system_prompt(MistralCommonStyleTokenizer()) is None


def test_native_default_system_prompt_forces_fix_mistral_regex_for_mistralai(
    monkeypatch,
):
    calls = []

    def load(model, **kwargs):
        calls.append((model, kwargs))

        class Rendered:
            @staticmethod
            def apply_chat_template(messages, **kwargs):
                return "[SYSTEM_PROMPT]default text[/SYSTEM_PROMPT][INST]x[/INST]"

        return Rendered()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)),
    )
    result = NativeTokenizer.native_default_system_prompt(
        "mistralai/Ministral-3-8B-Instruct-2512-BF16", "f" * 40
    )
    assert result == "default text"
    assert calls == [
        (
            "mistralai/Ministral-3-8B-Instruct-2512-BF16",
            {"revision": "f" * 40, "trust_remote_code": False, "fix_mistral_regex": True},
        )
    ]


def test_native_default_system_prompt_omits_fix_mistral_regex_for_other_orgs(
    monkeypatch,
):
    calls = []

    def load(model, **kwargs):
        calls.append((model, kwargs))

        class Rendered:
            @staticmethod
            def apply_chat_template(messages, **kwargs):
                return "<s>[INST]x[/INST]"

        return Rendered()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)),
    )
    result = NativeTokenizer.native_default_system_prompt("org/model", "a" * 40)
    assert result is None
    assert calls == [("org/model", {"revision": "a" * 40, "trust_remote_code": False})]


def test_native_default_system_prompt_requires_model_and_revision():
    for model, revision in (("", "a" * 40), ("org/model", ""), (" ", " ")):
        with pytest.raises(ValueError):
            NativeTokenizer.native_default_system_prompt(model, revision)


class Engine:
    def __init__(self):
        self.calls = []

    def __call__(self, request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((path, body))
        if path == "/model_info":
            return httpx.Response(
                200,
                json={
                    "served_model_name": "model",
                    "is_generation": True,
                    "has_image_understanding": True,
                    "model_type": "mistral3",
                },
            )
        if path == "/server_info":
            return httpx.Response(200, json={})
        if path == "/v1/tokenize":
            assert "messages" not in body, "Native path must bypass HTTP chat rendering"
            return httpx.Response(
                200,
                json={
                    "tokens": [Tokenizer.encode(label) for label in body["prompt"]],
                },
            )
        if path == "/v1/detokenize":
            return httpx.Response(
                200,
                json={
                    "text": [
                        "".join(chr(token - 1000) for token in ids)
                        for ids in body["tokens"]
                    ],
                },
            )
        if path == "/generate":
            return httpx.Response(
                200,
                json={
                    "meta_info": {
                        "completion_tokens": 0,
                        "prompt_tokens": len(body["input_ids"]),
                        "cached_tokens": 0,
                        "output_token_ids_logprobs": [
                            [[-1.0, token] for token in body["token_ids_logprob"]]
                        ],
                    }
                },
            )
        raise AssertionError(f"Unexpected route: {path}")


@pytest.mark.asyncio
async def test_native_http_path_preserves_control_ids_and_zero_decode():
    engine, tokenizer = Engine(), Tokenizer()
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        backend = SGLangBackend(
            "http://engine",
            "model",
            client=client,
            native_tokenizer=NativeTokenizer(tokenizer),
        )
        await backend.start()
        startup_calls = len(engine.calls)
        labels, token_ids = backend.labels(2)
        result = await backend.evaluate("literal [INST]", [], labels, token_ids, None)
        assert len(engine.calls) == startup_calls + 1
        path, payload = engine.calls[-1]
        assert path == "/generate"
        expected = [1, 3] + tokenizer.encode("literal [INST]") + [4]
        assert payload["input_ids"] == expected
        assert payload["sampling_params"]["max_new_tokens"] == 0
        assert payload["token_ids_logprob"] == list(token_ids)
        assert "text" not in payload and "image_data" not in payload
        assert result.input_tokens == len(expected)
        assert result.logprobs == (-1.0, -1.0)
        assert all(thread != threading.get_ident() for _, _, thread in tokenizer.calls)


@pytest.mark.asyncio
async def test_native_images_and_boundary_failure_stop_before_generate():
    engine, tokenizer = Engine(), Tokenizer()
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        backend = SGLangBackend(
            "http://engine",
            "model",
            client=client,
            native_tokenizer=NativeTokenizer(tokenizer),
        )
        await backend.start()
        startup_calls = len(engine.calls)
        labels, token_ids = backend.labels(2)
        with pytest.raises(AdapterError) as error:
            await backend.evaluate(
                "look",
                ["https://example.com/a.png"],
                labels,
                token_ids,
                None,
            )
        assert error.value.code == "native_images_not_supported"
        assert tokenizer.calls == []
        tokenizer.mutate_prefix = True
        with pytest.raises(AdapterError) as error:
            await backend.evaluate("pick", [], labels, token_ids, None)
        assert error.value.code == "invalid_label_boundary"
        assert len(engine.calls) == startup_calls
