import copy
import json

import httpx
import pytest

from jev_adapter.backend import AdapterError
from jev_adapter.sglang import SGLangBackend


class Engine:
    def __init__(self):
        self.calls = []
        self.info = {
            "served_model_name": "qwen",
            "model_path": "org/qwen",
            "is_generation": True,
            "has_image_understanding": True,
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        }
        self.config = {"speculative_algorithm": None, "skip_tokenizer_init": False}
        self.mutate_result = lambda result: result
        self.invalid_boundary = False
        self.lossy_roundtrip = False
        self.special_label = None
        self.fail_path = None
        self.failure = None

    @staticmethod
    def encode(text):
        return [ord(char) + 1000 for char in text]

    @staticmethod
    def decode(tokens):
        return "".join(chr(token - 1000) for token in tokens)

    def __call__(self, request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((path, body, request))
        if path == self.fail_path:
            if isinstance(self.failure, Exception):
                raise self.failure
            return self.failure
        if path == "/model_info":
            return httpx.Response(200, json=self.info)
        if path == "/server_info":
            return httpx.Response(200, json=self.config)
        if path == "/v1/tokenize":
            if "messages" in body:
                content = body["messages"][0]["content"]
                if isinstance(content, list):
                    content = "<image>" * (len(content) - 1) + content[-1]["text"]
                tokens = self.encode("<user>" + content + "</user><assistant>")
            else:
                assert body["add_special_tokens"] is False
                tokens = [self.encode(text) for text in body["prompt"]]
                if body["prompt"][0].startswith("<user>"):
                    if self.invalid_boundary:
                        tokens[-1][-1] += 1
                    if self.lossy_roundtrip:
                        tokens[0][-1] += 1
            return httpx.Response(200, json={"tokens": tokens})
        if path == "/v1/detokenize":
            tokens = body["tokens"]
            if isinstance(tokens[0], list):
                assert body["skip_special_tokens"] is True
                text = [self.decode(ids) for ids in tokens]
                text = ["" if value == self.special_label else value for value in text]
            else:
                assert body["skip_special_tokens"] is False
                text = self.decode(tokens)
            return httpx.Response(200, json={"text": text})
        if path == "/generate":
            result = {
                "text": "",
                "meta_info": {
                    "completion_tokens": 0,
                    "prompt_tokens": 71,
                    "cached_tokens": 0,
                    # Return labels out of order to ensure alignment by token ID.
                    "output_token_ids_logprobs": [
                        [
                            [-index - 0.5, token, None]
                            for index, token in reversed(
                                list(enumerate(body["token_ids_logprob"]))
                            )
                        ]
                    ],
                },
            }
            return httpx.Response(200, json=self.mutate_result(result))
        raise AssertionError(f"Unexpected route: {path}")


async def make_backend(engine=None, **kwargs):
    engine = engine or Engine()
    client = httpx.AsyncClient(transport=httpx.MockTransport(engine))
    backend = SGLangBackend("http://engine", "qwen", client=client, **kwargs)
    await backend.start()
    return backend, engine, client


@pytest.mark.asyncio
async def test_public_http_contract_and_zero_decode():
    backend, engine, client = await make_backend(api_key="private-key")
    try:
        labels, tokens = backend.labels(3)
        result = await backend.evaluate("pick an answer", [], labels, tokens, None)
        assert result.logprobs == (-0.5, -1.5, -2.5)
        assert result.input_tokens == 71
        assert result.cached_tokens == 0
        path, body, _ = engine.calls[-1]
        assert path == "/generate"
        assert "text" not in body and "image_data" not in body
        assert body["input_ids"] == engine.encode(
            "<user>pick an answer</user><assistant>"
        )
        assert body["sampling_params"] == {
            "max_new_tokens": 0,
            "temperature": 1,
            "top_p": 1,
            "top_k": -1,
        }
        assert body["token_ids_logprob"] == list(tokens)
        assert body["logprob_start_len"] == -1
        assert body["return_logprob"] is True
        assert body["top_logprobs_num"] == 0
        assert body["return_text_in_logprobs"] is False
        assert body["stream"] is False
        assert body["rid"].startswith("jev-")
        assert all(
            request.headers["Authorization"] == "Bearer private-key"
            for _, _, request in engine.calls
        )
        assert engine.calls[-4][1]["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        await client.aclose()


def test_prompt_wording_rejects_unknown_value():
    with pytest.raises(ValueError):
        SGLangBackend("http://engine", "qwen", prompt_wording="bogus")


@pytest.mark.asyncio
async def test_native_wording_prepends_system_message_over_http():
    backend, engine, client = await make_backend(
        prompt_wording="native", native_system_prompt="Default system text."
    )
    try:
        labels, tokens = backend.labels(3)
        await backend.evaluate("pick an answer", [], labels, tokens, None)
        # The messages-based /v1/tokenize call, three calls before /generate
        # (detokenize, boundary-check tokenize, generate follow it).
        path, body, _ = engine.calls[-4]
        assert path == "/v1/tokenize"
        assert body["messages"] == [
            {"role": "system", "content": "Default system text."},
            {"role": "user", "content": "pick an answer"},
        ]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_served_wording_never_sends_a_system_message_even_if_configured():
    # native_system_prompt is only honored when prompt_wording == "native";
    # the default ("served") must stay byte-identical to today's behaviour.
    backend, engine, client = await make_backend(
        prompt_wording="served", native_system_prompt="Should be ignored."
    )
    try:
        labels, tokens = backend.labels(2)
        await backend.evaluate("look", [], labels, tokens, None)
        path, body, _ = engine.calls[-4]
        assert path == "/v1/tokenize"
        assert body["messages"] == [{"role": "user", "content": "look"}]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_native_wording_passes_system_prompt_to_native_tokenizer():
    class RecordingNativeTokenizer:
        def __init__(self):
            self.calls = []

        def prepare(self, prompt, labels, token_ids, assistant_prefix, system_prompt=None):
            self.calls.append((prompt, labels, token_ids, assistant_prefix, system_prompt))
            return [1, 2, 3]

    native_tokenizer = RecordingNativeTokenizer()
    engine = Engine()
    client = httpx.AsyncClient(transport=httpx.MockTransport(engine))
    backend = SGLangBackend(
        "http://engine",
        "qwen",
        client=client,
        native_tokenizer=native_tokenizer,
        prompt_wording="native",
        native_system_prompt="Default system text.",
    )
    try:
        await backend.start()
        labels, tokens = backend.labels(2)
        await backend.evaluate("pick", [], labels, tokens, "Answer:")
        assert native_tokenizer.calls == [
            ("pick", labels, tokens, "Answer:", "Default system text.")
        ]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_images_remain_in_native_request_with_template_and_prefix():
    backend, engine, client = await make_backend()
    try:
        images = ["data:image/png;base64,YQ==", "https://example.com/image.png"]
        labels, tokens = backend.labels(2)
        await backend.evaluate("look", images, labels, tokens, "Answer: ")
        body = engine.calls[-1][1]
        assert body["image_data"] == images
        assert body["text"] == "<user><image><image>look</user><assistant>Answer: "
        assert "input_ids" not in body
        content = engine.calls[-4][1]["messages"][0]["content"]
        assert content == [
            {"type": "image_url", "image_url": {"url": images[0]}},
            {"type": "image_url", "image_url": {"url": images[1]}},
            {"type": "text", "text": "look"},
        ]
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_type,architectures",
    [
        ("mage_vl", []),
        ("unknown", ["MageVLForConditionalGeneration"]),
    ],
)
async def test_mage_vl_image_capability_and_message_format(model_type, architectures):
    engine = Engine()
    engine.info.update(model_type=model_type, architectures=architectures)
    backend, engine, client = await make_backend(engine)
    try:
        image = "data:image/png;base64,YQ=="
        labels, tokens = backend.labels(2)
        await backend.evaluate("describe", [image], labels, tokens, None)
        assert engine.calls[-4][1]["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        assert engine.calls[-1][1]["image_data"] == [image]
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value,code",
    [
        ("has_image_understanding", False, "images_not_supported"),
        ("model_type", "kimi_k3", "image_model_not_supported"),
    ],
)
async def test_image_capability_rejections(field, value, code):
    engine = Engine()
    engine.info[field] = value
    engine.info["architectures"] = []
    backend, engine, client = await make_backend(engine)
    try:
        labels, tokens = backend.labels(2)
        with pytest.raises(AdapterError, match="image|Image") as error:
            await backend.evaluate(
                "look", ["https://example.com/a.png"], labels, tokens, None
            )
        assert error.value.code == code
        assert not any(path == "/generate" for path, _, _ in engine.calls)
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag,code",
    [
        ("invalid_boundary", "invalid_label_boundary"),
        ("lossy_roundtrip", "unsupported_tokenizer_roundtrip"),
    ],
)
async def test_rejects_unsafe_tokenization_before_inference(flag, code):
    backend, engine, client = await make_backend()
    try:
        setattr(engine, flag, True)
        labels, tokens = backend.labels(2)
        with pytest.raises(AdapterError) as error:
            await backend.evaluate("look", [], labels, tokens, None)
        assert error.value.code == code
        assert not any(path == "/generate" for path, _, _ in engine.calls)
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "decoded",
        "missing_label",
        "duplicate_label",
        "nan",
        "positive",
        "missing_usage",
        "bad_usage",
        "bad_cache",
        "empty_positions",
        "multiple_positions",
    ],
)
async def test_rejects_unusable_engine_results(mutation):
    backend, engine, client = await make_backend()

    def mutate(original):
        result = copy.deepcopy(original)
        meta = result["meta_info"]
        entries = meta["output_token_ids_logprobs"][0]
        if mutation == "decoded":
            meta["completion_tokens"] = 1
        elif mutation == "missing_label":
            entries.pop()
        elif mutation == "duplicate_label":
            entries.append(entries[0])
        elif mutation == "nan":
            # JSON null is also rejected; transport JSON forbids NaN values.
            entries[0][0] = None
        elif mutation == "positive":
            entries[0][0] = 1.0
        elif mutation == "missing_usage":
            del meta["completion_tokens"]
        elif mutation == "bad_usage":
            meta["prompt_tokens"] = True
        elif mutation == "bad_cache":
            meta["cached_tokens"] = True
        elif mutation == "empty_positions":
            meta["output_token_ids_logprobs"] = []
        elif mutation == "multiple_positions":
            meta["output_token_ids_logprobs"].append(entries)
        return result

    engine.mutate_result = mutate
    try:
        labels, tokens = backend.labels(2)
        with pytest.raises(AdapterError) as error:
            await backend.evaluate("look", [], labels, tokens, None)
        assert error.value.status == 502
        assert error.value.code == "invalid_engine_response"
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,status,code",
    [
        (httpx.ReadTimeout("secret upstream location"), 504, "engine_timeout"),
        (httpx.ConnectError("secret upstream location"), 502, "engine_unavailable"),
        (httpx.Response(401, text="secret-key"), 502, "engine_http_error"),
        (httpx.Response(422, text="sensitive prompt"), 422, "engine_rejected_request"),
        (httpx.Response(200, text="not-json secret"), 502, "invalid_engine_response"),
    ],
)
async def test_transport_failures_are_sanitized(failure, status, code):
    backend, engine, client = await make_backend()
    engine.fail_path, engine.failure = "/generate", failure
    try:
        labels, tokens = backend.labels(2)
        with pytest.raises(AdapterError) as error:
            await backend.evaluate("look", [], labels, tokens, None)
        assert error.value.status == status
        assert error.value.code == code
        assert "secret" not in str(error.value) and "sensitive" not in str(error.value)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_special_labels_removed_and_start_idempotent():
    engine = Engine()
    engine.special_label = "A"
    backend, engine, client = await make_backend(engine)
    try:
        labels, _ = backend.labels(2)
        assert labels == ("B", "C")
        count = len(engine.calls)
        await backend.start()
        assert len(engine.calls) == count
        with pytest.raises(AdapterError):
            backend.labels(255)
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config,code",
    [
        ({"speculative_algorithm": "EAGLE"}, "speculation_not_supported"),
        ({"skip_tokenizer_init": True}, "tokenizer_unavailable"),
    ],
)
async def test_startup_validates_engine_configuration(config, code):
    engine = Engine()
    engine.config.update(config)
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        backend = SGLangBackend("http://engine", "qwen", client=client)
        with pytest.raises(AdapterError) as error:
            await backend.start()
        assert error.value.code == code


@pytest.mark.asyncio
async def test_nonfinite_raw_json_logprob_rejected():
    backend, engine, client = await make_backend()
    engine.fail_path = "/generate"
    engine.failure = httpx.Response(
        200,
        text='{"meta_info":{"completion_tokens":0,"prompt_tokens":1,"output_token_ids_logprobs":[[[NaN,1065],[-1.0,1066]]]}}',
    )
    try:
        labels, tokens = backend.labels(2)
        with pytest.raises(AdapterError) as error:
            await backend.evaluate("look", [], labels, tokens, None)
        assert error.value.code == "invalid_engine_response"
    finally:
        await client.aclose()
