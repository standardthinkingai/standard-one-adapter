import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from jev_adapter import inprocess
from jev_adapter.backend import AdapterError
from jev_adapter.protocol import SystemOneRequest

REVISION = "a" * 40
# Integrated serving accepts only base64 data URLs; the processor fakes below
# never decode them.
IMAGE_A = "data:image/png;base64,aW1hZ2UtYQ=="
IMAGE_B = "data:image/png;base64,aW1hZ2UtYg=="


def make_artifact(path, *, metadata_as_vector=False, legacy=False, **changes):
    metadata = {
        "format": inprocess.LEGACY_FORMAT if legacy else inprocess.FORMAT,
        "mode": "lm_rows",
        "model_id": "org/ministral",
        "engine_model_path": "org/ministral",
        "revision": REVISION,
        "architecture": inprocess.ARCHITECTURE,
        "hidden_size": 2,
        "labels": ["A", "B", "C"],
        "token_ids": [165, 166, 167],
        "temperature": 2.4,
    }
    if not legacy:
        metadata.update(
            label_scheme="upper",
            tokenizer_id="org/ministral",
            tokenizer_revision=REVISION,
        )
    metadata.update(changes)
    # Object dtype proves that the bridge never materializes weights. NumPy
    # would reject loading this member with allow_pickle=False.
    raw_metadata = json.dumps(metadata).encode("utf-8")
    with path.open("wb") as file:
        np.savez(
            file,
            metadata=(
                np.frombuffer(raw_metadata, dtype=np.uint8)
                if metadata_as_vector
                else np.bytes_(raw_metadata)
            ),
            weights=np.asarray([object()], dtype=object),
            bias=np.asarray([object()], dtype=object),
        )
    return path


class Tokenizer:
    all_special_ids = (1, 2, 3, 4)

    def __init__(self):
        self.template_calls = 0
        self.changed_tail = False

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 100 for char in text]

    def decode(self, ids, skip_special_tokens=True):
        if ids == [1]:
            return "" if skip_special_tokens else "<s>"
        if ids == [2]:
            return "" if skip_special_tokens else "[/INST]"
        if ids == [3]:
            return "" if skip_special_tokens else "[IMG]"
        if ids == [4]:
            return "" if skip_special_tokens else "[INST]"
        return "".join(chr(token - 100) for token in ids)

    def convert_ids_to_tokens(self, token):
        return "[/INST]" if token == 2 else str(token)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        continue_final_message,
        return_dict,
        enable_thinking,
    ):
        self.template_calls += 1
        if isinstance(messages[0], list):
            return [
                self.apply_chat_template(
                    item,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    return_dict=return_dict,
                    enable_thinking=enable_thinking,
                )
                for item in messages
            ]
        user = next(message for message in messages if message["role"] == "user")
        terminal = (
            9 if self.changed_tail and "changed boundary" in user["content"] else 2
        )
        base = [1, 4] + self.encode(user["content"]) + [terminal]
        if messages[-1]["role"] == "assistant":
            return base + self.encode(messages[-1]["content"])
        return base


class GenerateReqInput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Manager:
    is_generation = True
    model_path = "org/ministral"
    served_model_name = "decision-model"
    context_len = 8192
    num_reserved_tokens = 0

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model_config = SimpleNamespace(
            is_multimodal=False,
            image_token_id=3,
            hf_config=SimpleNamespace(
                model_type="mistral3",
                architectures=[inprocess.ARCHITECTURE],
                text_config=SimpleNamespace(hidden_size=2),
            )
        )
        self.calls = []
        self.abort_calls = []
        self.closed = 0
        self.response = None
        self.block = None
        self.empty = False
        self.mm_processor = None
        self.failure = None
        # The processor replaces each image marker with this many tokens.
        self.image_block = 4

    async def generate_request(self, obj, request=None):
        self.calls.append((obj, request))
        try:
            if self.block is not None:
                await self.block.wait()
            if self.failure is not None:
                raise self.failure
            if self.empty:
                return
            yield self.response or {
                "meta_info": {
                    "completion_tokens": 0,
                    "prompt_tokens": len(obj.input_ids)
                    + (self.image_block - 1) * len(getattr(obj, "image_data", None) or []),
                    "cached_tokens": 1,
                    # Readout logits, then the labels' vocabulary logprobs
                    # (the same distribution over the three labels).
                    "decision_logits": [1.0, 2.0, 3.0, -5.0, -4.0, -3.0],
                }
            }
        finally:
            self.closed += 1

    def abort_request(self, rid):
        self.abort_calls.append(rid)


def enable_images(manager):
    async def process_mm_data_async(*args, **kwargs):
        raise AssertionError("Only the engine may invoke its image processor")

    manager.model_config.is_multimodal = True
    manager.mm_processor = SimpleNamespace(process_mm_data_async=process_mm_data_async)
    return manager


def make_args(path):
    return SimpleNamespace(
        decision_readout_path=str(path),
        model_path="org/ministral",
        tokenizer_path="org/ministral",
        revision=REVISION,
        quantization=None,
        speculative_algorithm=None,
        max_running_requests=8,
    )


async def make_service(monkeypatch, path, manager=None, args=None):
    monkeypatch.setattr(inprocess, "_generate_input_type", lambda: GenerateReqInput)
    manager = manager or Manager()
    args = args or make_args(path)
    return await inprocess.create_systemone_service(manager, args), manager


def request(*, return_logprobs=False, images=None, assistant_prefix=None):
    return SystemOneRequest.model_validate(
        {
            "model": "decision-model",
            "state": "Help me",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Route?",
                    "criteria": {"one": "first", "two": "second"},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Urgency?",
                    "criteria": ["low", "medium", "high"],
                },
                "refund": {"type": "noul", "instructions": "Refund?"},
            },
            "options": {
                "return_logprobs": return_logprobs,
                "assistant_prefix": assistant_prefix,
            },
            "images": images or [],
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["lm_rows", "learned"])
async def test_internal_scheduler_path_and_existing_service(
    monkeypatch, tmp_path, mode
):
    path = make_artifact(tmp_path / "head.npz", mode=mode)
    service, manager = await make_service(monkeypatch, path)
    assert service.backend.mode == mode
    assert service.default_temperature == 1.0  # GPU already applied 2.4
    assert manager.calls == []  # no startup GPU request
    template_calls_before = manager.tokenizer.template_calls
    result = await service.score(request())
    assert service.backend.label_scheme == "upper"
    assert result["metadata"]["label_scheme"] == "upper"
    assert set(result["answers"]) == {"route", "urgency", "refund"}
    assert result["answers"]["route"]["choice"] == "two"
    assert result["answers"]["urgency"]["score"] > 1
    assert result["answers"]["refund"]["noul"] < 0.5
    # Released adapters' metadata: the gateway passes it to /v1/judgments callers.
    assert set(result["metadata"]) == {
        "confidence_method", "temperature", "evaluations", "label_scheme", "adapter_elapsed_ms"
    }
    assert result["usage"]["output_tokens"] == 0
    assert len(manager.calls) == 3
    assert manager.closed == 3
    assert manager.tokenizer.template_calls == template_calls_before + 3
    for call, raw_request in manager.calls:
        assert raw_request is None
        assert call.input_ids and isinstance(call.input_ids, list)
        assert call.sampling_params["max_new_tokens"] == 0
        assert call.return_logprob is False
        assert call.return_hidden_states is False
        assert call.stream is False


@pytest.mark.asyncio
async def test_unavailable_logprobs_and_images_fail_before_gpu(monkeypatch, tmp_path):
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz", mode="learned")
    )
    # A learned head's logits are not vocabulary-normalized.
    with pytest.raises(AdapterError) as error:
        await service.score(request(return_logprobs=True))
    assert error.value.code == "return_logprobs_unavailable"
    assert not manager.calls
    with pytest.raises(AdapterError) as error:
        await service.score(request(images=[IMAGE_A]))
    assert error.value.code == "image_processor_unavailable"
    assert not manager.calls


@pytest.mark.asyncio
async def test_lm_rows_answers_and_logprobs_use_vocabulary_logprobs(monkeypatch, tmp_path):
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz")
    )
    manager.response = {
        "meta_info": {
            "completion_tokens": 0,
            "prompt_tokens": 10,
            "cached_tokens": 0,
            # The readout half disagrees on purpose: only the vocabulary half counts.
            "decision_logits": [9.0, 0.0, 0.0, -0.5, -1.5, -3.0],
        }
    }
    result = await service.score(request(return_logprobs=True))
    for answer in result["answers"].values():
        assert answer["logprobs"]
        for row in answer["logprobs"]:
            assert set(row.values()) <= {-0.5, -1.5, -3.0}
    choice = next(a for a in result["answers"].values() if a["type"] == "choice")
    assert max(choice["probabilities"], key=choice["probabilities"].get) == choice["choice"]
    assert choice["logprobs"][0][choice["choice"]] == -0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("images", [[], [IMAGE_A]])
async def test_assistant_prefix_extends_the_native_prompt(monkeypatch, tmp_path, images):
    manager = enable_images(Manager())
    manager.mm_processor.decision_image_tokens = lambda image: manager.image_block
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    labels, token_ids = service.backend.labels(2)
    prefix = "Answer: "
    for _ in range(2):
        await service.backend.evaluate("choose", images, labels, token_ids, prefix)
    tokenizer = manager.tokenizer
    base = [1, 4] + tokenizer.encode("choose") + [2]
    if images:
        base = [1, 4, 3] + base[2:]
    for call, _ in manager.calls:
        assert call.input_ids == base + tokenizer.encode(prefix)
    assert list(service.backend._prefix_ids_cache) == [(prefix, 2)]


@pytest.mark.asyncio
async def test_assistant_prefix_that_moves_the_boundary_is_refused(monkeypatch, tmp_path):
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz")
    )
    native = service.backend._native_tokenizer
    original = native._chat

    def drifting(messages, *, continuation):
        ids = original(messages, continuation=continuation)
        return ids + [7] if continuation and messages[0]["content"] == "choose" else ids

    monkeypatch.setattr(native, "_chat", drifting)
    labels, token_ids = service.backend.labels(2)
    with pytest.raises(AdapterError) as error:
        await service.backend.evaluate("choose", [], labels, token_ids, "Answer: ")
    assert error.value.code == "invalid_label_boundary"
    assert not manager.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["lm_rows", "learned"])
@pytest.mark.parametrize(
    "images",
    [
        [IMAGE_A],
        ["data:image/png;base64,aW1hZ2U="],
        [IMAGE_A, "data:image/png;base64,c2Vjb25k"],
    ],
)
async def test_images_preserve_native_text_ids_and_order_without_fetching(
    monkeypatch, tmp_path, mode, images
):
    manager = enable_images(Manager())
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz", mode=mode), manager
    )
    native = service.backend._native_tokenizer
    base = native._chat(
        [{"role": "user", "content": "Compare the images."}], continuation=False
    )
    before = manager.tokenizer.template_calls
    labels, token_ids = service.backend.labels(2)
    result = await service.backend.evaluate(
        "Compare the images.", images, labels, token_ids, None
    )
    call, raw_request = manager.calls[0]
    assert raw_request is None
    assert manager.tokenizer.template_calls == before + 1
    assert call.input_ids == base[:2] + [3] * len(images) + base[2:]
    assert call.image_data == images
    assert call.image_data is not images
    assert not hasattr(call, "text")
    assert call.sampling_params["max_new_tokens"] == 0
    assert call.return_logprob is False
    assert call.return_hidden_states is False
    assert call.stream is False
    # LM rows answer from the vocabulary half of the row, learned heads from the readout.
    assert result.logprobs == ((-5.0, -4.0) if mode == "lm_rows" else (1.0, 2.0))
    assert manager.closed == 1


@pytest.mark.asyncio
async def test_each_question_receives_images_in_original_order(monkeypatch, tmp_path):
    manager = enable_images(Manager())
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    images = [IMAGE_A, IMAGE_B]
    result = await service.score(request(images=images))
    assert len(manager.calls) == 3
    assert set(result["answers"]) == {"route", "urgency", "refund"}
    assert result["usage"]["output_tokens"] == 0
    for call, _ in manager.calls:
        assert call.image_data == images
        assert call.input_ids[:4] == [1, 4, 3, 3]
        assert call.input_ids[-1] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", ["missing_processor", "text_model"])
async def test_missing_vision_capability_fails_before_tokenization_and_gpu(
    monkeypatch, tmp_path, capability
):
    manager = enable_images(Manager())
    if capability == "missing_processor":
        manager.mm_processor = None
    else:
        manager.model_config.is_multimodal = False
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    before = manager.tokenizer.template_calls
    with pytest.raises(AdapterError) as error:
        await service.score(request(images=[IMAGE_A]))
    assert error.value.code == "image_processor_unavailable"
    assert manager.tokenizer.template_calls == before
    assert not manager.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("image_token", [None, True, -1, 4, 2])
async def test_missing_or_mismatched_atomic_image_token_fails_before_gpu(
    monkeypatch, tmp_path, image_token
):
    manager = enable_images(Manager())
    manager.model_config.image_token_id = image_token
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    with pytest.raises(AdapterError) as error:
        await service.score(request(images=[IMAGE_A]))
    assert error.value.code == "unsupported_image_tokenizer"
    assert not manager.calls


@pytest.mark.asyncio
async def test_reserved_image_marker_in_scoring_text_is_rejected(
    monkeypatch, tmp_path
):
    manager = enable_images(Manager())
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    monkeypatch.setattr(
        service.backend._native_tokenizer, "_chat", lambda *a, **kw: [1, 3, 2]
    )
    labels, token_ids = service.backend.labels(2)
    with pytest.raises(AdapterError) as error:
        await service.backend.evaluate(
            "reserved marker", [IMAGE_A], labels, token_ids, None
        )
    assert error.value.code == "invalid_image_prompt"
    assert not manager.calls


@pytest.mark.asyncio
async def test_images_keep_system_prefix_and_all_native_text_ids(monkeypatch, tmp_path):
    manager = enable_images(Manager())
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    native_ids = [1, 101, 102, 4, 103, 104, 2]
    monkeypatch.setattr(
        service.backend._native_tokenizer, "_chat", lambda *a, **kw: list(native_ids)
    )
    labels, token_ids = service.backend.labels(2)
    await service.backend.evaluate(
        "text", [IMAGE_A], labels, token_ids, None
    )
    assert manager.calls[0][0].input_ids == [1, 101, 102, 4, 3, 103, 104, 2]
    assert native_ids == [1, 101, 102, 4, 103, 104, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("native_ids", [[1, 101, 2], [1, 4, 101, 4, 2], [1, 4, 2, 101, 2]])
async def test_images_reject_missing_or_ambiguous_native_turn(
    monkeypatch, tmp_path, native_ids
):
    manager = enable_images(Manager())
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    monkeypatch.setattr(
        service.backend._native_tokenizer, "_chat", lambda *a, **kw: list(native_ids)
    )
    labels, token_ids = service.backend.labels(2)
    with pytest.raises(AdapterError) as error:
        await service.backend.evaluate(
            "text", [IMAGE_A], labels, token_ids, None
        )
    assert error.value.code == "invalid_image_boundary"
    assert not manager.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("with_images", [False, True])
async def test_changed_terminal_control_token_rejected_before_gpu(
    monkeypatch, tmp_path, with_images
):
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), enable_images(Manager())
    )
    manager.tokenizer.changed_tail = True
    labels, token_ids = service.backend.labels(2)
    with pytest.raises(AdapterError) as error:
        await service.backend.evaluate(
            "changed boundary",
            [IMAGE_A] if with_images else [],
            labels,
            token_ids,
            None,
        )
    assert error.value.code == "invalid_label_boundary"
    assert not manager.calls


@pytest.mark.asyncio
async def test_uint8_metadata_and_empty_engine_result(monkeypatch, tmp_path):
    service, manager = await make_service(
        monkeypatch,
        make_artifact(tmp_path / "head.npz", metadata_as_vector=True),
    )
    manager.empty = True
    with pytest.raises(AdapterError) as error:
        await service.score(request())
    assert error.value.code == "invalid_engine_response"
    assert set(manager.abort_calls) == {call.rid for call, _ in manager.calls}


@pytest.mark.asyncio
async def test_missing_bias_member_rejected_before_gpu(monkeypatch, tmp_path):
    path = tmp_path / "head.npz"
    metadata = {
        "format": inprocess.FORMAT,
        "mode": "lm_rows",
        "model_id": "org/ministral",
        "engine_model_path": "org/ministral",
        "revision": REVISION,
        "architecture": inprocess.ARCHITECTURE,
        "hidden_size": 2,
        "labels": ["A", "B"],
        "token_ids": [165, 166],
        "temperature": 1.0,
    }
    np.savez(path, metadata=np.bytes_(json.dumps(metadata)), weights=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="weights, and bias"):
        await make_service(monkeypatch, path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", ["revision", "model", "architecture", "hidden", "labels", "quantization"]
)
async def test_startup_rejects_mismatch(monkeypatch, tmp_path, bad):
    manager = Manager()
    args = make_args(tmp_path / "head.npz")
    changes = {}
    if bad == "revision":
        args.revision = "b" * 40
    elif bad == "model":
        manager.model_path = "other/model"
    elif bad == "architecture":
        manager.model_config.hf_config.architectures = ["Other"]
    elif bad == "hidden":
        changes["hidden_size"] = 3
    elif bad == "labels":
        changes["token_ids"] = [165, 166, 999]
    elif bad == "quantization":
        args.quantization = "fp8"
    path = make_artifact(tmp_path / "head.npz", **changes)
    with pytest.raises(ValueError):
        await make_service(monkeypatch, path, manager, args)
    assert not manager.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy,artifact_scheme,flag,env,accepted",
    [
        (False, "upper", None, None, True),  # the default is upper
        (False, "upper", "upper", "served", True),  # the server flag wins
        (False, "served", "served", None, True),
        (False, "served", None, "served", True),
        (True, None, None, "served", True),  # v1 carries the served order
        (True, None, "served", None, True),
        (True, None, None, None, False),
        (False, "served", None, None, False),
        (False, "upper", "served", None, False),
        (False, "upper", None, "served", False),
    ],
)
async def test_readout_and_server_label_schemes_must_agree(
    monkeypatch, tmp_path, legacy, artifact_scheme, flag, env, accepted
):
    changes = {} if legacy else {"label_scheme": artifact_scheme}
    path = make_artifact(tmp_path / "head.npz", legacy=legacy, **changes)
    args = make_args(path)
    if flag is not None:
        args.decision_label_scheme = flag
    if env is not None:
        monkeypatch.setenv("JEV_LABEL_SCHEME", env)
    if not accepted:
        manager = Manager()
        with pytest.raises(ValueError, match="scheme"):
            await make_service(monkeypatch, path, manager, args)
        assert not manager.calls
        return
    service, _ = await make_service(monkeypatch, path, args=args)
    expected = "served" if legacy else artifact_scheme
    assert service.backend.label_scheme == expected
    result = await service.score(request())
    assert result["metadata"]["label_scheme"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes,match",
    [
        # A gap: C is a single token, so the upper list cannot skip it.
        ({"labels": ["A", "B", "D"], "token_ids": [165, 166, 168]}, "scheme's labels"),
        # Lowercase is never an upper label.
        ({"labels": ["A", "B", "a"], "token_ids": [165, 166, 197]}, "scheme's labels"),
        ({"tokenizer_revision": "b" * 40}, "tokenizer"),
        ({"tokenizer_id": "other/tokenizer"}, "tokenizer"),
        ({"label_scheme": "lower"}, "label scheme"),
    ],
)
async def test_startup_refuses_labels_that_are_not_the_scheme_list(
    monkeypatch, tmp_path, changes, match
):
    path = make_artifact(tmp_path / "head.npz", **changes)
    with pytest.raises(ValueError, match=match):
        await make_service(monkeypatch, path)


@pytest.mark.asyncio
async def test_v2_artifact_requires_label_metadata(monkeypatch, tmp_path):
    path = tmp_path / "head.npz"
    make_artifact(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(bytes(data["metadata"].item()))
    del metadata["label_scheme"]
    np.savez(
        path,
        metadata=np.bytes_(json.dumps(metadata)),
        weights=np.asarray([object()], dtype=object),
        bias=np.asarray([object()], dtype=object),
    )
    with pytest.raises(ValueError, match="format"):
        await make_service(monkeypatch, path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "meta",
    [
        {"completion_tokens": 1},
        {"prompt_tokens": 0},
        {"cached_tokens": -1},
        {"decision_logits": [1.0, 2.0, 3.0]},
        {"decision_logits": [1.0, 2.0, 3.0, -5.0, float("nan"), -3.0]},
    ],
)
async def test_malformed_engine_result_rejected(monkeypatch, tmp_path, meta):
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz")
    )
    baseline = {
        "completion_tokens": 0,
        "prompt_tokens": 10,
        "cached_tokens": 1,
        "decision_logits": [1.0, 2.0, 3.0, -5.0, -4.0, -3.0],
    }
    manager.response = {"meta_info": {**baseline, **meta}}
    with pytest.raises(AdapterError) as error:
        await service.score(request())
    assert error.value.code == "invalid_engine_response"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_images", [False, True])
async def test_cancel_aborts_internal_request(monkeypatch, tmp_path, with_images):
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), enable_images(Manager())
    )
    manager.block = asyncio.Event()
    labels, token_ids = service.backend.labels(2)
    task = asyncio.create_task(
        service.backend.evaluate(
            "choose",
            [IMAGE_A] if with_images else [],
            labels,
            token_ids,
            None,
        )
    )
    while not manager.calls:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.abort_calls == [manager.calls[0][0].rid]
    assert manager.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message, code",
    [
        ("Error while loading data data:image/png;base64,secret: invalid base64", "invalid_image"),
        ("An exception occurred while loading IMAGE data at index 0: secret", "invalid_image"),
        ("An exception occurred while loading multimodal data: secret", "invalid_image"),
        ("Image count 3 exceeds limit 2 per request.", "image_limit_exceeded"),
        ("The input (9000 tokens) is longer than the model's context length (8192 tokens).", "context_length_exceeded"),
        ("Requested token count exceeds the model's maximum context length of 8192 tokens.", "context_length_exceeded"),
        ("decision images do not support automatic prompt truncation", "context_length_exceeded"),
    ],
)
async def test_known_image_input_failure_is_translated_and_cleans_up(
    monkeypatch, tmp_path, message, code
):
    manager = enable_images(Manager())
    manager.failure = ValueError(message)
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    labels, token_ids = service.backend.labels(2)
    with pytest.raises(AdapterError) as error:
        await service.backend.evaluate(
            "choose", [IMAGE_A], labels, token_ids, None
        )
    assert error.value.code == code
    assert error.value.status == 422
    assert error.value.field == "images"
    assert "secret" not in str(error.value)
    assert error.value.__cause__ is manager.failure
    assert manager.abort_calls == [manager.calls[0][0].rid]
    assert manager.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, with_images",
    [
        (ValueError("unexpected worker invariant"), True),
        (ValueError("decision image processor changed the [/INST] boundary"), True),
        (RuntimeError("CUDA out of memory"), True),
        (ValueError("Error while loading data secret: invalid base64"), False),
    ],
)
async def test_internal_failures_and_text_path_are_not_mislabeled_as_image_input(
    monkeypatch, tmp_path, failure, with_images
):
    manager = enable_images(Manager())
    manager.failure = failure
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    labels, token_ids = service.backend.labels(2)
    with pytest.raises(type(failure)) as error:
        await service.backend.evaluate(
            "choose",
            [IMAGE_A] if with_images else [],
            labels,
            token_ids,
            None,
        )
    assert error.value is failure
    assert manager.abort_calls == [manager.calls[0][0].rid]
    assert manager.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, status, code, message, field",
    [
        (
            "An exception occurred while loading IMAGE data at index 0: private-url",
            422,
            "invalid_image",
            "An image could not be decoded.",
            "images",
        ),
        (
            "decision image processor changed the [/INST] boundary",
            500,
            "scoring_failed",
            "Could not compute a complete decision.",
            "",
        ),
    ],
)
async def test_integrated_http_handler_distinguishes_image_input_and_engine_failure(
    monkeypatch, tmp_path, failure, status, code, message, field
):
    from fastapi.responses import JSONResponse

    handler_path = (
        Path(__file__).resolve().parents[2]
        / "sglang-jev/python/sglang/srt/entrypoints/systemone.py"
    )
    spec = importlib.util.spec_from_file_location("decision_image_http_test", handler_path)
    handler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handler)
    # Exercise the actual route/service error handling using the standard
    # serializer: the adapter's CPU-only install does not require orjson.
    monkeypatch.setattr(handler, "ORJSONResponse", JSONResponse)
    manager = enable_images(Manager())
    manager.failure = ValueError(failure)
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )

    async def read_json():
        return request(images=[IMAGE_A]).model_dump()

    async def is_disconnected():
        return False

    response = await handler.handle_systemone(
        SimpleNamespace(json=read_json, is_disconnected=is_disconnected), service
    )
    assert response.status_code == status
    assert json.loads(response.body) == {
        "error": {
            "code": code,
            "message": message,
            "field": field,
        }
    }
    assert manager.closed == len(manager.calls)
    assert set(manager.abort_calls) == {call.rid for call, _ in manager.calls}


def decision_request(questions, *, images=None):
    return SystemOneRequest.model_validate(
        {
            "model": "decision-model",
            "state": "Help me",
            "questions": questions,
            "images": images or [],
        }
    )


def count_images(manager, tokens_per_image):
    manager.mm_processor.decision_image_tokens = lambda image: tokens_per_image
    return manager


@pytest.mark.asyncio
async def test_an_over_long_question_refuses_the_whole_request_before_gpu_work(
    monkeypatch, tmp_path
):
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz")
    )
    before = manager.tokenizer.template_calls
    with pytest.raises(AdapterError) as error:
        await service.score(
            decision_request(
                {
                    "short": {"type": "noul", "instructions": "Refund?"},
                    "long": {"type": "noul", "instructions": "x" * 9000},
                }
            )
        )
    assert (error.value.code, error.value.status) == ("context_length_exceeded", 422)
    assert error.value.field == "questions.long"
    assert "8191" in str(error.value)
    # Every evaluation was tokenized and none reached the scheduler, not even
    # the question that fits.
    assert manager.tokenizer.template_calls == before + 2
    assert manager.calls == []
    assert service._pending == 0


@pytest.mark.asyncio
async def test_reserved_tokens_shrink_the_prompt_limit(monkeypatch, tmp_path):
    manager = Manager()
    manager.context_len = 300
    manager.num_reserved_tokens = 100
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    with pytest.raises(AdapterError) as error:
        await service.score(
            decision_request({"q": {"type": "noul", "instructions": "y" * 150}})
        )
    assert error.value.code == "context_length_exceeded"
    assert "199" in str(error.value)
    assert manager.calls == []


@pytest.mark.asyncio
async def test_images_that_overflow_the_context_are_refused_before_gpu_work(
    monkeypatch, tmp_path
):
    manager = count_images(enable_images(Manager()), 5000)
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    with pytest.raises(AdapterError) as error:
        await service.score(request(images=[IMAGE_A, IMAGE_B]))
    assert (error.value.code, error.value.field) == ("context_length_exceeded", "images")
    assert manager.calls == []


@pytest.mark.asyncio
async def test_undecodable_image_is_refused_before_gpu_work(monkeypatch, tmp_path):
    manager = enable_images(Manager())

    def counter(image):
        raise ValueError("cannot identify image file")

    manager.mm_processor.decision_image_tokens = counter
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    with pytest.raises(AdapterError) as error:
        await service.score(request(images=[IMAGE_A]))
    assert (error.value.code, error.value.field) == ("invalid_image", "images")
    assert manager.calls == []


@pytest.mark.asyncio
async def test_usage_reports_exact_image_tokens(monkeypatch, tmp_path, caplog):
    manager = count_images(enable_images(Manager()), 4)
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    result = await service.score(request(images=[IMAGE_A, IMAGE_B]))
    # Three evaluations, each with two images of four tokens.
    assert result["usage"]["input_tokens_details"] == {"image_tokens": 24}
    assert result["usage"]["input_tokens"] == sum(
        len(call.input_ids) + 3 * 2 for call, _ in manager.calls
    )
    assert "predicted" not in caplog.text

    text_only = await service.score(request())
    assert "input_tokens_details" not in text_only["usage"]


@pytest.mark.asyncio
async def test_prediction_drift_is_logged_once_and_usage_follows_the_engine(
    monkeypatch, tmp_path, caplog
):
    manager = count_images(enable_images(Manager()), 3)
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    result = await service.score(request(images=[IMAGE_A]))
    assert result["usage"]["input_tokens_details"] == {"image_tokens": 12}
    assert caplog.text.count("the processor predicted") == 1


@pytest.mark.asyncio
async def test_engine_usage_without_image_tokens_is_an_engine_defect(
    monkeypatch, tmp_path
):
    manager = enable_images(Manager())
    manager.image_block = 1
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    with pytest.raises(AdapterError) as error:
        await service.score(request(images=[IMAGE_A]))
    assert (error.value.code, error.value.status) == ("invalid_engine_response", 502)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image",
    [
        "https://example.test/a.png",
        "file:///etc/passwd.png",
        "/etc/passwd.png",
        "data:image/gif;base64,R0lGODlh",
        "data:image/png;base64,not base64",
    ],
)
async def test_only_base64_image_data_urls_reach_the_engine(
    monkeypatch, tmp_path, image
):
    manager = enable_images(Manager())
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager
    )
    before = manager.tokenizer.template_calls
    body = request().model_dump()
    body["images"] = [image]
    parsed = (
        SystemOneRequest.model_validate(body)
        if image.startswith(("https://", "data:"))
        else SystemOneRequest.model_construct(**{**request().__dict__, "images": [image]})
    )
    with pytest.raises(AdapterError) as error:
        await service.score(parsed)
    assert (error.value.code, error.value.status, error.value.field) == (
        "invalid_image",
        422,
        "images",
    )
    assert manager.tokenizer.template_calls == before
    assert manager.calls == []


@pytest.mark.asyncio
async def test_image_count_limit_is_checked_before_tokenization(monkeypatch, tmp_path):
    manager = enable_images(Manager())
    args = make_args(tmp_path / "head.npz")
    args.limit_mm_data_per_request = '{"image": 1}'
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), manager, args
    )
    before = manager.tokenizer.template_calls
    with pytest.raises(AdapterError) as error:
        await service.score(request(images=[IMAGE_A, IMAGE_B]))
    assert (error.value.code, error.value.field) == ("image_limit_exceeded", "images")
    assert manager.tokenizer.template_calls == before
    assert manager.calls == []


@pytest.mark.asyncio
async def test_deadline_aborts_engine_work_and_answers_504(monkeypatch, tmp_path):
    args = make_args(tmp_path / "head.npz")
    args.decision_deadline_seconds = 0.05
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), args=args
    )
    manager.block = asyncio.Event()
    with pytest.raises(AdapterError) as error:
        await service.score(request())
    assert (error.value.code, error.value.status) == ("engine_timeout", 504)
    assert len(manager.calls) == 3
    assert set(manager.abort_calls) == {call.rid for call, _ in manager.calls}
    assert manager.closed == 3
    assert service._pending == 0


@pytest.mark.asyncio
async def test_integrated_default_deadline_is_below_the_gateway_timeout(
    monkeypatch, tmp_path
):
    service, _ = await make_service(monkeypatch, make_artifact(tmp_path / "head.npz"))
    assert service.deadline_seconds == inprocess.DEFAULT_DEADLINE_SECONDS == 110.0
    assert service.remote_images is False


@pytest.mark.asyncio
async def test_queue_overflow_is_refused_before_any_work(monkeypatch, tmp_path):
    args = make_args(tmp_path / "head.npz")
    args.max_running_requests = 2
    args.decision_max_queued_evaluations = 3
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), args=args
    )
    manager.block = asyncio.Event()
    first = asyncio.create_task(service.score(request()))
    while len(manager.calls) < 2:
        await asyncio.sleep(0)
    # Three evaluations pending; three more would exceed 2 running + 3 queued.
    before = manager.tokenizer.template_calls
    with pytest.raises(AdapterError) as error:
        await service.score(request())
    assert (error.value.code, error.value.status) == ("engine_overloaded", 503)
    assert manager.tokenizer.template_calls == before
    assert len(manager.calls) == 2
    manager.block.set()
    assert set((await first)["answers"]) == {"route", "urgency", "refund"}
    assert service._pending == 0


@pytest.mark.asyncio
async def test_an_idle_engine_admits_a_request_larger_than_its_queue(
    monkeypatch, tmp_path
):
    args = make_args(tmp_path / "head.npz")
    args.max_running_requests = 1
    args.decision_max_queued_evaluations = 0
    service, _ = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz"), args=args
    )
    result = await service.score(request())
    assert result["metadata"]["evaluations"] == 3


@pytest.mark.asyncio
async def test_scheduler_refusal_is_reported_as_overload(monkeypatch, tmp_path):
    from fastapi import HTTPException

    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz")
    )
    manager.failure = HTTPException(status_code=503, detail="The request queue is full.")
    with pytest.raises(AdapterError) as error:
        await service.score(request())
    assert (error.value.code, error.value.status) == ("engine_overloaded", 503)
    assert set(manager.abort_calls) == {call.rid for call, _ in manager.calls}

    manager.failure = HTTPException(status_code=500, detail="scheduler crashed")
    with pytest.raises(HTTPException):
        await service.score(request())


@pytest.mark.asyncio
async def test_engine_context_refusal_on_the_text_path_is_a_caller_error(
    monkeypatch, tmp_path
):
    service, manager = await make_service(
        monkeypatch, make_artifact(tmp_path / "head.npz")
    )
    manager.failure = ValueError(
        "The input (9000 tokens) is longer than the model's context length (8192 tokens)."
    )
    with pytest.raises(AdapterError) as error:
        await service.score(request())
    assert (error.value.code, error.value.status, error.value.field) == (
        "context_length_exceeded",
        422,
        "questions",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change, message",
    [
        ({"allow_auto_truncate": True}, "allow_auto_truncate"),
        ({"max_queued_requests": 8}, "--max-queued-requests"),
        ({"limit_mm_data_per_request": '["image"]'}, "JSON object"),
        ({"limit_mm_data_per_request": {"image": -1}}, "nonnegative"),
        ({"decision_deadline_seconds": 0}, "deadline_seconds"),
        ({"decision_max_queued_evaluations": -1}, "max_queued_evaluations"),
    ],
)
async def test_startup_rejects_unsafe_decision_settings(
    monkeypatch, tmp_path, change, message
):
    args = make_args(tmp_path / "head.npz")
    for name, value in change.items():
        setattr(args, name, value)
    with pytest.raises((ValueError, TypeError), match=message):
        await make_service(
            monkeypatch, make_artifact(tmp_path / "head.npz"), args=args
        )


@pytest.mark.asyncio
async def test_startup_requires_the_engine_context_length(monkeypatch, tmp_path):
    manager = Manager()
    manager.context_len = None
    with pytest.raises(ValueError, match="context length"):
        await make_service(monkeypatch, make_artifact(tmp_path / "head.npz"), manager)


def test_serving_defaults_from_env_match_the_standalone_cli():
    assert inprocess.serving_defaults_from_env({}) == {
        "alias": "decision-model",
        "default_temperature": 1.0,
        "temperature_by_type": None,
        "prompt_wording": "served",
        "engine_metadata": False,
    }
    assert inprocess.serving_defaults_from_env(
        {
            "JEV_DEFAULT_TEMPERATURE": "1.65",
            "JEV_TEMPERATURE_BY_TYPE": "choice=0.8, score=0.95",
            "JEV_PROMPT_WORDING": "native",
            "JEV_MODEL_ALIAS": "jev-latest",
            "JEV_ENGINE_METADATA": "1",
        }
    ) == {
        "alias": "jev-latest",
        "default_temperature": 1.65,
        "temperature_by_type": {"choice": 0.8, "score": 0.95},
        "prompt_wording": "native",
        "engine_metadata": True,
    }
    for bad in (
        {"JEV_DEFAULT_TEMPERATURE": "0"},
        {"JEV_DEFAULT_TEMPERATURE": "nan"},
        {"JEV_DEFAULT_TEMPERATURE": "warm"},
        {"JEV_TEMPERATURE_BY_TYPE": "choice"},
        {"JEV_TEMPERATURE_BY_TYPE": "other=1"},
        {"JEV_TEMPERATURE_BY_TYPE": "choice=1,choice=2"},
        {"JEV_TEMPERATURE_BY_TYPE": "noul=-1"},
        {"JEV_PROMPT_WORDING": "other"},
        {"JEV_ENGINE_METADATA": "yes"},
    ):
        with pytest.raises(ValueError):
            inprocess.serving_defaults_from_env(bad)


@pytest.mark.asyncio
async def test_env_defaults_reach_the_in_process_service(monkeypatch, tmp_path):
    monkeypatch.setenv("JEV_DEFAULT_TEMPERATURE", "1.65")
    monkeypatch.setenv("JEV_MODEL_ALIAS", "jev-latest")
    service, _ = await make_service(monkeypatch, make_artifact(tmp_path / "head.npz"))
    assert service.default_temperature == 1.65
    assert service.alias == "jev-latest"
    result = await service.score(request())
    assert result["metadata"]["temperature"] == 1.65
