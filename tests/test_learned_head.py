import json
import math
import shutil
from pathlib import Path

import httpx
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("safetensors")

from jev_adapter.backend import AdapterError
from jev_adapter.learned_head import (
    CALIBRATION_TEMPERATURE,
    HIDDEN_SIZE,
    LABELS,
    MODEL_ID,
    REVISION,
    LearnedHeadBackend,
    LearnedHeadReadout,
)
from jev_adapter.protocol import SystemOneRequest
from jev_adapter.service import SystemOneService

ARCHIVE = (
    Path(__file__).resolve().parents[2]
    / "jev-lora-data/runs/h200-trm-readout-20260922/verified/results"
)
HEAD = ARCHIVE / "linear-seed17/head.safetensors"


class Native:
    def prepare(self, prompt, labels, ids, prefix, system_prompt):
        return [1, 2, 3]


class Engine:
    def __init__(self):
        self.calls = []
        self.hidden = [0.0] * HIDDEN_SIZE
        self.revision = REVISION
        self.model_path = MODEL_ID
        self.mode = "last"
        self.dtype = "bfloat16"
        self.first_token = 1065

    def __call__(self, request):
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.url.path, body))
        path = request.url.path
        if path == "/model_info":
            return httpx.Response(
                200,
                json={
                    "is_generation": True,
                    "served_model_name": "decision-model",
                    "model_path": self.model_path,
                    "model_type": "mistral3",
                    "architectures": ["Mistral3ForConditionalGeneration"],
                },
            )
        if path == "/server_info":
            return httpx.Response(
                200,
                json={
                    "revision": self.revision,
                    "return_hidden_states_mode": self.mode,
                    "dtype": self.dtype,
                    "quantization": None,
                    "speculative_algorithm": None,
                    "skip_tokenizer_init": False,
                },
            )
        if path == "/v1/tokenize":
            def token(label):
                if len(label) != 1 or label not in LABELS:
                    return []
                if label in "ABC":
                    return [self.first_token + ord(label) - ord("A")]
                return [2000 + ord(label)]

            return httpx.Response(
                200,
                json={"tokens": [token(label) for label in body["prompt"]]},
            )
        if path == "/v1/detokenize":
            def label(token):
                if token in (self.first_token, self.first_token + 1, self.first_token + 2):
                    return chr(ord("A") + token - self.first_token)
                return chr(token - 2000)

            return httpx.Response(
                200, json={"text": [label(ids[0]) for ids in body["tokens"]]}
            )
        if path == "/generate":
            return httpx.Response(
                200,
                json={
                    "meta_info": {
                        "completion_tokens": 0,
                        "prompt_tokens": 3,
                        "cached_tokens": 0,
                        "hidden_states": self.hidden,
                    }
                },
            )
        raise AssertionError(path)


def make_backend(engine=None):
    engine = engine or Engine()
    client = httpx.AsyncClient(transport=httpx.MockTransport(engine))
    backend = LearnedHeadBackend(
        "http://engine",
        "decision-model",
        readout_path=str(HEAD),
        tokenizer_model=MODEL_ID,
        tokenizer_revision=REVISION,
        native_tokenizer=Native(),
        client=client,
    )
    return backend, engine, client


def test_learned_head_uses_only_the_served_label_order():
    # The head's 62 rows are A-Z, a-z, 0-9; the upper order would misalign them.
    with pytest.raises(ValueError, match="label-scheme served"):
        LearnedHeadBackend(
            "http://engine",
            "decision-model",
            readout_path=str(HEAD),
            tokenizer_model=MODEL_ID,
            tokenizer_revision=REVISION,
            native_tokenizer=Native(),
            label_scheme="upper",
        )


def test_archive_loads_selected_head_and_frozen_calibration():
    if not HEAD.exists():
        pytest.skip("verified training archive is not checked out")
    readout = LearnedHeadReadout.load(HEAD)
    assert readout.weights.shape == (62, 3072)
    assert readout.bias.shape == (62,)
    assert readout.weights.flags.writeable is False
    readout._engine_token_ids = (1065, 1066, 1067, 2068)
    scores = readout.score([0.0] * HIDDEN_SIZE, (1065, 1066, 1067, 2068))
    assert scores == pytest.approx(
        tuple(float(x) / CALIBRATION_TEMPERATURE for x in readout.bias[:4])
    )


@pytest.mark.asyncio
async def test_backend_uses_hidden_only_path_and_allows_experimental_fourth_label():
    if not HEAD.exists():
        pytest.skip("verified training archive is not checked out")
    backend, engine, client = make_backend()
    try:
        await backend.start()
        labels, token_ids = backend.labels(4)
        assert labels == ("A", "B", "C", "D")
        result = await backend.evaluate("Choose", [], labels, token_ids, None)
        assert result.logprobs == pytest.approx(
            tuple(float(x) / CALIBRATION_TEMPERATURE for x in backend._readout.bias[:4])
        )
        assert result.input_tokens == 3
        requests = [body for path, body in engine.calls if path == "/generate"]
        assert len(requests) == 2  # startup probe plus evaluation
        assert all(body["sampling_params"]["max_new_tokens"] == 0 for body in requests)
        assert all(body["return_hidden_states"] == "last" for body in requests)
        assert all("return_logprob" not in body for body in requests)

        request = SystemOneRequest.model_validate(
            {
                "model": "jev-latest",
                "state": "Need to route a request",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Which route?",
                        "criteria": {"one": "one", "two": "two"},
                    }
                },
            }
        )
        answer = (await SystemOneService(backend).score(request))["answers"]["route"]
        first = float(backend._readout.bias[0]) / CALIBRATION_TEMPERATURE
        second = float(backend._readout.bias[1]) / CALIBRATION_TEMPERATURE
        expected = 1 / (1 + math.exp(second - first))
        assert answer["probabilities"]["one"] == pytest.approx(expected)
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["revision", "model_path", "mode", "dtype", "tokens", "hidden"])
async def test_incompatible_engine_fails_startup(bad):
    if not HEAD.exists():
        pytest.skip("verified training archive is not checked out")
    engine = Engine()
    if bad == "revision":
        engine.revision = None
    elif bad == "model_path":
        engine.model_path = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
    elif bad == "mode":
        engine.mode = None
    elif bad == "dtype":
        engine.dtype = "float16"
    elif bad == "tokens":
        engine.first_token = 999
    elif bad == "hidden":
        engine.hidden = [0.0]
    backend, _, client = make_backend(engine)
    try:
        with pytest.raises((AdapterError, ValueError)):
            await backend.start()
        assert not backend._ready
    finally:
        await client.aclose()


def test_modified_checkpoint_or_provenance_is_rejected(tmp_path):
    if not HEAD.exists():
        pytest.skip("verified training archive is not checked out")
    results = tmp_path / "results"
    predictor = results / "linear-seed17"
    predictor.mkdir(parents=True)
    for name in ("selection.json", "provenance.json"):
        shutil.copy2(ARCHIVE / name, results / name)
    for name in ("head.safetensors", "temperature.json"):
        shutil.copy2(ARCHIVE / "linear-seed17" / name, predictor / name)

    head = predictor / "head.safetensors"
    with head.open("r+b") as file:
        file.seek(-1, 2)
        value = file.read(1)
        file.seek(-1, 2)
        file.write(bytes((value[0] ^ 1,)))
    with pytest.raises(ValueError, match="SHA256"):
        LearnedHeadReadout.load(head)

    shutil.copy2(HEAD, head)
    provenance = results / "provenance.json"
    data = json.loads(provenance.read_text())
    data["training"]["manifest"]["hidden_coordinate"] = "pre_final_norm"
    provenance.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance"):
        LearnedHeadReadout.load(head)


@pytest.mark.asyncio
async def test_full_vocabulary_logprobs_rejected_before_inference():
    if not HEAD.exists():
        pytest.skip("verified training archive is not checked out")
    backend, engine, client = make_backend()
    try:
        await backend.start()
        before = len(engine.calls)
        request = SystemOneRequest.model_validate(
            {
                "model": "jev-latest",
                "state": "Need to route a request",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Which route?",
                        "criteria": {"one": "one", "two": "two"},
                    }
                },
                "options": {"return_logprobs": True},
            }
        )
        with pytest.raises(AdapterError) as error:
            await SystemOneService(backend).score(request)
        assert error.value.code == "return_logprobs_unavailable"
        assert len(engine.calls) == before
    finally:
        await client.aclose()
