import json
import math
from pathlib import Path

import httpx
import pytest

np = pytest.importorskip("numpy")

from jev_adapter.backend import AdapterError
from jev_adapter.candidate_backend import CandidateReadoutBackend
from jev_adapter.export_readout import _source_file, export_readout
from jev_adapter.protocol import SystemOneRequest
from jev_adapter.readout import ARCHITECTURE, FORMAT, LEGACY_FORMAT, CandidateReadout
from jev_adapter.service import SystemOneService

REVISION = "a" * 40


def artifact(
    path,
    *,
    token_ids=(165, 166, 167),
    revision=REVISION,
    label_scheme="upper",
    tokenizer=None,
):
    metadata = {
        "format": LEGACY_FORMAT if label_scheme is None else FORMAT,
        "mode": "lm_rows",
        "model_id": "org/ministral",
        "revision": revision,
        "engine_model_path": "org/ministral",
        "architecture": ARCHITECTURE,
        "source_weight_key": "lm_head.weight",
        "source_dtype": "BF16",
        "config_sha256": "0" * 64,
        "hidden_size": 2,
        "labels": ["A", "B", "C"],
        "token_ids": list(token_ids),
    }
    if label_scheme is not None:
        metadata.update(
            label_scheme=label_scheme,
            tokenizer_id="org/ministral",
            tokenizer_revision=revision,
        )
        metadata.update(tokenizer or {})
    weights = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    with path.open("wb") as file:
        np.savez(file, metadata=np.bytes_(json.dumps(metadata)), weights=weights)
    return path


class Native:
    def prepare(self, prompt, labels, ids, prefix, system_prompt):
        return [1, 2, 3]


class Engine:
    def __init__(self):
        self.calls = []
        self.hidden = [1.0, 2.0]
        self.model_path = "org/ministral"
        self.revision = REVISION
        self.quantization = None

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
                    "architectures": [ARCHITECTURE],
                },
            )
        if path == "/server_info":
            return httpx.Response(
                200,
                json={
                    "return_hidden_states_mode": "last",
                    "revision": self.revision,
                    "quantization": self.quantization,
                    "speculative_algorithm": None,
                    "skip_tokenizer_init": False,
                },
            )
        if path == "/v1/tokenize":
            return httpx.Response(
                200,
                json={
                    "tokens": [
                        [ord(label) + 100] if label in ("A", "B", "C") else []
                        for label in body["prompt"]
                    ]
                },
            )
        if path == "/v1/detokenize":
            return httpx.Response(
                200,
                json={"text": [chr(ids[0] - 100) for ids in body["tokens"]]},
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


async def make_backend(path, engine=None, revision=REVISION, **kwargs):
    engine = engine or Engine()
    client = httpx.AsyncClient(transport=httpx.MockTransport(engine))
    backend = CandidateReadoutBackend(
        "http://engine",
        "decision-model",
        readout_path=str(path),
        tokenizer_model="org/ministral",
        tokenizer_revision=revision,
        native_tokenizer=Native(),
        client=client,
        **kwargs,
    )
    return backend, engine, client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "artifact_scheme,backend_scheme,accepted",
    [
        ("upper", "upper", True),
        ("served", "served", True),
        (None, "served", True),  # a v1 artifact carries the served order
        (None, "upper", False),
        ("served", "upper", False),
        ("upper", "served", False),
    ],
)
async def test_readout_label_scheme_must_match_the_backend(
    tmp_path, artifact_scheme, backend_scheme, accepted
):
    path = artifact(tmp_path / "head.npz", label_scheme=artifact_scheme)
    if not accepted:
        with pytest.raises(ValueError, match="label scheme"):
            await make_backend(path, label_scheme=backend_scheme)
        return
    backend, _, client = await make_backend(path, label_scheme=backend_scheme)
    try:
        await backend.start()
        assert backend.label_scheme == backend_scheme
        assert backend.labels(3)[0] == ("A", "B", "C")
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "tokenizer",
    [
        {"tokenizer_id": "other/tokenizer"},
        {"tokenizer_revision": "b" * 40},
    ],
)
def test_v2_readout_tokenizer_must_be_the_readout_models(tmp_path, tokenizer):
    path = artifact(tmp_path / "head.npz", tokenizer=tokenizer)
    with pytest.raises(ValueError, match="tokenizer differs"):
        CandidateReadout.load(path)


def test_v2_readout_tokenizer_may_name_the_engine_model_path(tmp_path):
    path = artifact(tmp_path / "head.npz")
    loaded = CandidateReadout.load(path)
    assert loaded.label_scheme == "upper"
    assert loaded.metadata["tokenizer_id"] == loaded.metadata["engine_model_path"]


@pytest.mark.asyncio
async def test_hidden_only_request_scores_selected_rows(tmp_path):
    backend, engine, client = await make_backend(artifact(tmp_path / "head.npz"))
    try:
        await backend.start()
        labels, token_ids = backend.labels(3)
        result = await backend.evaluate("choose", [], labels, token_ids, None)
        assert result.logprobs == (1.0, 2.0, 3.0)
        assert result.input_tokens == 3
        generations = [body for path, body in engine.calls if path == "/generate"]
        assert len(generations) == 2  # startup probe, then the real evaluation
        assert all(
            body["sampling_params"]["max_new_tokens"] == 0 for body in generations
        )
        assert all(body["return_hidden_states"] == "last" for body in generations)
        assert all("return_logprob" not in body for body in generations)
        assert all("token_ids_logprob" not in body for body in generations)
        request = SystemOneRequest.model_validate(
            {
                "model": "jev-latest",
                "state": "Need help",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Route?",
                        "criteria": {"one": "first", "two": "second"},
                    }
                },
            }
        )
        answer = (await SystemOneService(backend).score(request))["answers"]["route"]
        assert answer["probabilities"]["one"] == pytest.approx(1 / (1 + math.e))
        assert answer["probabilities"]["two"] == pytest.approx(math.e / (1 + math.e))
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    ["revision", "engine_revision", "model_path", "tokens", "hidden", "quantization"],
)
async def test_rejects_mismatch_before_serving(tmp_path, bad):
    path = artifact(
        tmp_path / "head.npz",
        token_ids=(165, 166, 999) if bad == "tokens" else (165, 166, 167),
        revision="b" * 40 if bad == "revision" else REVISION,
    )
    engine = Engine()
    if bad == "model_path":
        engine.model_path = "other/checkpoint"
    if bad == "engine_revision":
        engine.revision = None
    if bad == "hidden":
        engine.hidden = [1.0]
    if bad == "quantization":
        engine.quantization = "fp8"
    backend, _, client = await make_backend(path, engine)
    try:
        with pytest.raises((ValueError, AdapterError)):
            await backend.start()
        assert not backend._ready
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_exact_vocab_logprobs_rejected_before_inference(tmp_path):
    backend, engine, client = await make_backend(artifact(tmp_path / "head.npz"))
    try:
        await backend.start()
        before = len(engine.calls)
        request = SystemOneRequest.model_validate(
            {
                "model": "jev-latest",
                "state": "Need help",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Route?",
                        "criteria": {"one": "first", "two": "second"},
                    }
                },
                "options": {"return_logprobs": True},
            }
        )
        with pytest.raises(AdapterError) as error:
            await SystemOneService(backend).score(request)
        assert error.value.code == "return_logprobs_unavailable"
        assert error.value.field == "options.return_logprobs"
        assert len(engine.calls) == before
    finally:
        await client.aclose()


class Tokenizer:
    def encode(self, label, add_special_tokens=False):
        return [ord(label) + 100] if label in ("A", "B", "C") else []

    def decode(self, ids, skip_special_tokens=True):
        return chr(ids[0] - 100)


class Slice:
    def __init__(self, weight):
        self.weight = weight

    def get_shape(self):
        return list(self.weight.shape)

    def get_dtype(self):
        return "BF16"

    def __getitem__(self, token):
        class Row:
            def __init__(self, values):
                self.values = values

            def float(self):
                return self

            def numpy(self):
                return self.values

        return Row(self.weight[token])


class SafeOpen:
    def __init__(self, key):
        self.key = key
        self.get_slice_calls = []
        self.weight = np.zeros((300, 2), dtype=np.float32)
        self.weight[165] = [1, 0]
        self.weight[166] = [0, 1]
        self.weight[167] = [1, 1]

    def __call__(self, path, framework, device):
        assert framework == "pt" and device == "cpu"
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def keys(self):
        return [self.key]

    def get_slice(self, key):
        self.get_slice_calls.append(key)
        return Slice(self.weight)

    def get_tensor(self, key):
        raise AssertionError("full tensor must never load")


@pytest.mark.parametrize(
    ("tied", "tie_location", "weight_key"),
    [
        (True, "text", "model.language_model.embed_tokens.weight"),
        (True, "top", "language_model.model.embed_tokens.weight"),
        (False, "text", "lm_head.weight"),
        (False, "top", "language_model.lm_head.weight"),
    ],
)
def test_export_reads_only_selected_rows(tmp_path, tied, tie_location, weight_key):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {
        "model_type": "mistral3",
        "architectures": [ARCHITECTURE],
        "text_config": {"hidden_size": 2, "vocab_size": 300},
    }
    if tie_location == "text":
        config["text_config"]["tie_word_embeddings"] = tied
    else:
        config["tie_word_embeddings"] = tied
    (checkpoint / "config.json").write_text(json.dumps(config))
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_key: "model.safetensors"}})
    )
    (checkpoint / "model.safetensors").touch()
    safe_open = SafeOpen(weight_key)
    output = tmp_path / "readout.npz"
    metadata = export_readout(
        checkpoint,
        model_id="org/ministral",
        revision=REVISION,
        output=output,
        tokenizer=Tokenizer(),
        safe_open=safe_open,
    )
    assert metadata["source_weight_key"] == weight_key
    assert metadata["format"] == FORMAT
    assert metadata["labels"] == ["A", "B", "C"]
    assert metadata["label_scheme"] == "upper"
    assert metadata["tokenizer_id"] == "org/ministral"
    assert metadata["tokenizer_revision"] == REVISION
    assert safe_open.get_slice_calls == [weight_key]
    assert CandidateReadout.load(output).score([1.0, 2.0], (165, 166, 167)) == (
        1.0,
        2.0,
        3.0,
    )


@pytest.mark.parametrize(
    "scheme,env,expected",
    [
        ("upper", None, ["A", "B", "C", "AA"]),
        ("served", None, ["A", "B", "C", "a", "b", "AA"]),
        (None, None, ["A", "B", "C", "AA"]),
        (None, "served", ["A", "B", "C", "a", "b", "AA"]),
    ],
)
def test_export_selects_rows_in_the_label_scheme_order(
    tmp_path, monkeypatch, scheme, env, expected
):
    vocabulary = {"A": 165, "B": 166, "C": 167, "a": 197, "b": 198, "AA": 250}

    class SchemeTokenizer:
        def encode(self, label, add_special_tokens=False):
            return [vocabulary[label]] if label in vocabulary else [1, 2]

        def decode(self, ids, skip_special_tokens=True):
            return {token: label for label, token in vocabulary.items()}[ids[0]]

    if env:
        monkeypatch.setenv("JEV_LABEL_SCHEME", env)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mistral3",
                "architectures": [ARCHITECTURE],
                "text_config": {
                    "hidden_size": 2,
                    "vocab_size": 300,
                    "tie_word_embeddings": False,
                },
            }
        )
    )
    (checkpoint / "model.safetensors").touch()
    metadata = export_readout(
        checkpoint,
        model_id="org/ministral",
        revision=REVISION,
        output=tmp_path / "readout.npz",
        tokenizer=SchemeTokenizer(),
        safe_open=SafeOpen("lm_head.weight"),
        label_scheme=scheme,
    )
    assert metadata["labels"] == expected
    assert metadata["token_ids"] == [vocabulary[label] for label in expected]
    assert metadata["label_scheme"] == (scheme or env or "upper")
    loaded = CandidateReadout.load(tmp_path / "readout.npz")
    assert loaded.label_scheme == metadata["label_scheme"]


@pytest.mark.parametrize("shared_blob", [False, True])
def test_export_reads_standard_hf_cache_blob_symlink(tmp_path, shared_blob):
    model_cache = tmp_path / "hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16"
    snapshot = model_cache / "snapshots" / REVISION
    blobs = model_cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = tmp_path / "hub/blobs/a3" / ("a" * 64) if shared_blob else blobs / ("b" * 64)
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.touch()
    if shared_blob:
        model_blob = blobs / ("b" * 64)
        model_blob.symlink_to(Path("../../blobs/a3") / blob.name)
    shard = snapshot / "model-00001-of-00002.safetensors"
    shard.symlink_to(Path("../../blobs") / ("b" * 64))
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mistral3",
                "architectures": [ARCHITECTURE],
                "text_config": {
                    "hidden_size": 2,
                    "vocab_size": 300,
                    "tie_word_embeddings": False,
                },
            }
        )
    )
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": shard.name}})
    )

    class CheckingSafeOpen(SafeOpen):
        def __call__(self, path, framework, device):
            assert Path(path) == shard
            assert Path(path).resolve() == blob
            return super().__call__(path, framework, device)

    output = tmp_path / "cached-readout.npz"
    metadata = export_readout(
        snapshot,
        model_id="mistralai/Ministral-3-3B-Instruct-2512-BF16",
        revision=REVISION,
        output=output,
        tokenizer=Tokenizer(),
        safe_open=CheckingSafeOpen("lm_head.weight"),
    )
    assert metadata["source_weight_key"] == "lm_head.weight"
    assert CandidateReadout.load(output).score([1.0, 2.0], (165, 166, 167)) == (
        1.0,
        2.0,
        3.0,
    )


@pytest.mark.parametrize(
    "index_name",
    [
        "../../blobs/direct.safetensors",
        "../outside.safetensors",
        "..\\outside.safetensors",
    ],
)
def test_export_rejects_index_filename_traversal(tmp_path, index_name):
    snapshot = tmp_path / "models--org--model/snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": index_name}})
    )
    with pytest.raises(ValueError, match="invalid weight file"):
        _source_file(snapshot, ("lm_head.weight",), SafeOpen("lm_head.weight"))


def test_export_rejects_shard_symlink_outside_model_cache(tmp_path):
    snapshot = tmp_path / "hub/models--org--model/snapshots" / REVISION
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.touch()
    (snapshot / "model.safetensors").symlink_to(outside)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": "model.safetensors"}})
    )
    with pytest.raises(ValueError, match="invalid weight file"):
        _source_file(snapshot, ("lm_head.weight",), SafeOpen("lm_head.weight"))


def test_export_rejects_ambiguous_readout_weights(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mistral3",
                "architectures": [ARCHITECTURE],
                "text_config": {
                    "hidden_size": 2,
                    "vocab_size": 300,
                    "tie_word_embeddings": False,
                },
            }
        )
    )
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "lm_head.weight": "model.safetensors",
                    "language_model.lm_head.weight": "model.safetensors",
                }
            }
        )
    )
    with pytest.raises(ValueError, match="exactly one supported readout tensor"):
        export_readout(
            checkpoint,
            model_id="org/ministral",
            revision=REVISION,
            output=tmp_path / "readout.npz",
            tokenizer=Tokenizer(),
            safe_open=SafeOpen("lm_head.weight"),
        )
