import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from jev_adapter.export_gpu_readout import (
    FORMAT,
    _learned_token_ids,
    export_gpu_readout,
    main,
)
from jev_adapter.learned_head import (
    CALIBRATION_TEMPERATURE,
    HEAD_SHA256,
    LABELS,
    MODEL_ID,
    REVISION,
    LearnedHeadReadout,
)
from jev_adapter.readout import ARCHITECTURE
from jev_adapter.readout import FORMAT as CANDIDATE_FORMAT
from jev_adapter.readout import LEGACY_FORMAT as LEGACY_CANDIDATE_FORMAT

HEAD = (
    Path(__file__).resolve().parents[2]
    / "jev-lora-data/runs/h200-trm-readout-20260922/verified/results"
    / "linear-seed17/head.safetensors"
)
REQUIRED = {
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
    "label_scheme",
    "tokenizer_id",
    "tokenizer_revision",
}


def candidate_artifact(path, *, legacy=False, **changes):
    metadata = {
        "format": LEGACY_CANDIDATE_FORMAT if legacy else CANDIDATE_FORMAT,
        "mode": "lm_rows",
        "model_id": "org/ministral",
        "revision": "a" * 40,
        "engine_model_path": "org/ministral",
        "architecture": ARCHITECTURE,
        "source_weight_key": "lm_head.weight",
        "source_dtype": "BF16",
        "config_sha256": "0" * 64,
        "hidden_size": 2,
        "labels": ["A", "B", "C"],
        "token_ids": [101, 102, 103],
    }
    if not legacy:
        metadata.update(
            label_scheme="upper",
            tokenizer_id="org/ministral",
            tokenizer_revision="a" * 40,
        )
    metadata.update(changes)
    weights = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    with path.open("wb") as file:
        np.savez(file, metadata=np.bytes_(json.dumps(metadata)), weights=weights)
    return path


def load_gpu(path):
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"metadata", "weights", "bias"}
        assert isinstance(data["metadata"].item(), bytes)
        return (
            json.loads(data["metadata"].item()),
            np.array(data["weights"], copy=True),
            np.array(data["bias"], copy=True),
        )


class PinnedTokenizer:
    def encode(self, label, add_special_tokens=False):
        assert add_special_tokens is False
        if label in "ABC":
            return [1065 + ord(label) - ord("A")]
        return [2000 + ord(label)]

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        token = ids[0]
        if 1065 <= token <= 1067:
            return chr(ord("A") + token - 1065)
        return chr(token - 2000)


def test_candidate_export_preserves_validated_rows_and_adds_zero_bias(tmp_path):
    source = candidate_artifact(tmp_path / "candidate.npz")
    destination = tmp_path / "gpu.npz"
    returned = export_gpu_readout(
        output=destination,
        candidate_readout=source,
        engine_model_path="/models/3b",
    )
    metadata, weights, bias = load_gpu(destination)
    assert metadata == returned
    assert REQUIRED <= metadata.keys()
    assert metadata["format"] == FORMAT
    assert metadata["mode"] == "lm_rows"
    assert metadata["model_id"] == "org/ministral"
    assert metadata["engine_model_path"] == "/models/3b"
    assert metadata["revision"] == "a" * 40
    assert metadata["architecture"] == ARCHITECTURE
    assert metadata["labels"] == ["A", "B", "C"]
    assert metadata["token_ids"] == [101, 102, 103]
    assert metadata["temperature"] == 1.0
    assert metadata["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert metadata["label_scheme"] == "upper"
    assert metadata["tokenizer_id"] == "org/ministral"
    assert metadata["tokenizer_revision"] == "a" * 40
    np.testing.assert_array_equal(
        weights, np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(bias, np.zeros(3, dtype=np.float32))


def test_legacy_candidate_export_records_the_served_scheme(tmp_path):
    source = candidate_artifact(tmp_path / "candidate.npz", legacy=True)
    metadata = export_gpu_readout(output=tmp_path / "gpu.npz", candidate_readout=source)
    assert metadata["format"] == FORMAT
    assert metadata["label_scheme"] == "served"
    assert metadata["tokenizer_id"] == "org/ministral"
    assert metadata["tokenizer_revision"] == "a" * 40


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"labels": ["A", "B", "a"]}, "upper scheme"),
        ({"labels": ["B", "A", "C"]}, "upper scheme"),
        ({"label_scheme": "lower"}, "label scheme"),
        ({"tokenizer_revision": "short"}, "tokenizer"),
    ],
)
def test_candidate_export_refuses_labels_outside_their_scheme(tmp_path, changes, match):
    source = candidate_artifact(tmp_path / "candidate.npz", **changes)
    destination = tmp_path / "gpu.npz"
    with pytest.raises(ValueError, match=match):
        export_gpu_readout(output=destination, candidate_readout=source)
    assert not destination.exists()


def test_candidate_export_rejects_invalid_source_and_does_not_create_output(tmp_path):
    source = candidate_artifact(tmp_path / "candidate.npz")
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
        weights = np.array(data["weights"], copy=True)
    metadata["token_ids"] = [101, 101, 103]
    with source.open("wb") as file:
        np.savez(file, metadata=np.bytes_(json.dumps(metadata)), weights=weights)
    destination = tmp_path / "gpu.npz"
    with pytest.raises(ValueError, match="labels/token IDs"):
        export_gpu_readout(output=destination, candidate_readout=source)
    assert not destination.exists()


def test_max_labels_cli_slices_candidate_rows_in_label_order(tmp_path, capsys):
    source = candidate_artifact(tmp_path / "candidate.npz")
    destination = tmp_path / "gpu.npz"
    main(
        [
            "--candidate-readout",
            str(source),
            "--max-labels",
            "2",
            "--output",
            str(destination),
        ]
    )
    metadata, weights, bias = load_gpu(destination)
    assert metadata["labels"] == ["A", "B"]
    assert metadata["token_ids"] == [101, 102]
    assert metadata["temperature"] == 1.0
    assert weights.shape == (2, 2)
    assert bias.shape == (2,)
    np.testing.assert_array_equal(weights, [[1, 0], [0, 1]])
    printed = json.loads(capsys.readouterr().out)
    assert printed["labels"] == 2
    assert printed["label_scheme"] == "upper"


def test_learned_export_copies_verified_head_and_temperature(tmp_path):
    pytest.importorskip("safetensors")
    if not HEAD.exists():
        pytest.skip("verified training archive is not checked out")
    destination = tmp_path / "gpu.npz"
    returned = export_gpu_readout(
        output=destination,
        learned_head=HEAD,
        tokenizer=PinnedTokenizer(),
    )
    metadata, weights, bias = load_gpu(destination)
    source = LearnedHeadReadout.load(HEAD)
    assert metadata == returned
    assert REQUIRED <= metadata.keys()
    assert metadata["mode"] == "learned"
    assert metadata["model_id"] == MODEL_ID
    assert metadata["engine_model_path"] == MODEL_ID
    assert metadata["revision"] == REVISION
    assert metadata["architecture"] == ARCHITECTURE
    assert metadata["hidden_size"] == 3072
    assert metadata["labels"] == list(LABELS)
    assert metadata["token_ids"][:3] == [1065, 1066, 1067]
    assert metadata["temperature"] == CALIBRATION_TEMPERATURE
    assert metadata["source_sha256"] == HEAD_SHA256
    assert metadata["label_scheme"] == "served"
    assert metadata["tokenizer_id"] == MODEL_ID
    assert metadata["tokenizer_revision"] == REVISION
    np.testing.assert_array_equal(weights, source.weights)
    np.testing.assert_array_equal(bias, source.bias)
    assert weights.dtype == np.float32 and bias.dtype == np.float32
    # The engine will divide once by metadata.temperature after the affine head.
    hidden = np.arange(3072, dtype=np.float32) / 3072
    expected = (source.weights[:3] @ hidden + source.bias[:3]) / CALIBRATION_TEMPERATURE
    actual = (weights[:3] @ hidden + bias[:3]) / metadata["temperature"]
    np.testing.assert_array_equal(actual, expected)


def test_max_labels_slices_learned_head_without_changing_calibration(tmp_path):
    pytest.importorskip("safetensors")
    if not HEAD.exists():
        pytest.skip("verified training archive is not checked out")
    destination = tmp_path / "gpu.npz"
    export_gpu_readout(
        output=destination,
        learned_head=HEAD,
        tokenizer=PinnedTokenizer(),
        max_labels=3,
    )
    metadata, weights, bias = load_gpu(destination)
    source = LearnedHeadReadout.load(HEAD)
    assert metadata["labels"] == ["A", "B", "C"]
    assert metadata["token_ids"] == [1065, 1066, 1067]
    assert metadata["temperature"] == CALIBRATION_TEMPERATURE
    assert weights.shape == (3, 3072)
    assert bias.shape == (3,)
    np.testing.assert_array_equal(weights, source.weights[:3])
    np.testing.assert_array_equal(bias, source.bias[:3])


@pytest.mark.parametrize("count", [1, 4, True])
def test_max_labels_rejects_out_of_range_counts_without_writing(tmp_path, count):
    source = candidate_artifact(tmp_path / "candidate.npz")
    destination = tmp_path / "gpu.npz"
    with pytest.raises(ValueError, match="max_labels"):
        export_gpu_readout(
            output=destination,
            candidate_readout=source,
            max_labels=count,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "bad", ["split_label", "bad_decode", "wrong_first", "duplicate"]
)
def test_learned_tokenizer_requires_all_62_exact_unique_tokens(bad):
    class BadTokenizer(PinnedTokenizer):
        duplicate_seen = 0

        def encode(self, label, add_special_tokens=False):
            if bad == "split_label" and label == "z":
                return [1, 2]
            if bad == "wrong_first" and label == "A":
                return [999]
            if bad == "duplicate" and label == "D":
                return [1065]
            return super().encode(label, add_special_tokens=add_special_tokens)

        def decode(self, ids, skip_special_tokens=True):
            if bad == "bad_decode" and ids == [2000 + ord("z")]:
                return "other"
            if bad == "wrong_first" and ids == [999]:
                return "A"
            if bad == "duplicate" and ids == [1065]:
                self.duplicate_seen += 1
                return "A" if self.duplicate_seen == 1 else "D"
            return super().decode(ids, skip_special_tokens=skip_special_tokens)

    tokenizer = BadTokenizer()
    with pytest.raises(ValueError, match="duplicate" if bad == "duplicate" else None):
        _learned_token_ids(tokenizer)


def test_default_learned_tokenizer_loader_uses_exact_cached_pin(monkeypatch, tmp_path):
    pytest.importorskip("safetensors")
    if not HEAD.exists():
        pytest.skip("verified training archive is not checked out")
    calls = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return PinnedTokenizer()

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer)
    )
    export_gpu_readout(output=tmp_path / "gpu.npz", learned_head=HEAD)
    assert calls == [
        (
            MODEL_ID,
            {
                "revision": REVISION,
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]


def test_cli_rejects_two_sources_and_output_is_not_overwritten(tmp_path):
    source = candidate_artifact(tmp_path / "candidate.npz")
    destination = tmp_path / "gpu.npz"
    destination.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        main(
            [
                "--candidate-readout",
                str(source),
                "--learned-head",
                str(HEAD),
                "--output",
                str(destination),
            ]
        )
    with pytest.raises(FileExistsError):
        export_gpu_readout(output=destination, candidate_readout=source)
    assert destination.read_bytes() == b"keep"
