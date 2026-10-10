"""Experimental frozen 3B decision head over SGLang's last hidden state.

Only the selected, verified linear-seed17 readout is accepted. The training
archive pins a post-final-norm 3B backbone and canonical 62 output labels.
"""

import hashlib
import json
import math
import string
from pathlib import Path
from typing import Any

from .candidate_backend import CandidateReadoutBackend
from .readout import ARCHITECTURE, _numpy
from .sglang import SGLangBackend

MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"
PREDICTOR = "linear-seed17"
HEAD_SHA256 = "9fef1ef2768a815ab4310eded8883638101a3c853216dfff7d1ea4e537c44552"
TEMPERATURE_SHA256 = "8ddb250b7c006f7a9c100c8bc618f811f09139c5b84ea5907c4095d64dba2638"
CALIBRATION_TEMPERATURE = 0.8518692324650765
LABELS = string.ascii_uppercase + string.ascii_lowercase + string.digits
HIDDEN_SIZE = 3072
TRAINED_LABELS = "ABC"
TRAINED_TOKEN_IDS = {"A": 1065, "B": 1066, "C": 1067}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read learned-head provenance: {path.name}") from error
    if not isinstance(value, dict):
        raise TypeError(f"malformed learned-head provenance: {path.name}")
    return value


class LearnedHeadReadout:
    """A verified [62, 3072] linear head and its frozen calibration."""

    def __init__(self, weights: Any, bias: Any):
        self.weights = weights
        self.bias = bias
        self._engine_token_ids: tuple[int, ...] = ()

    @classmethod
    def load(cls, path: str | Path, *, safe_open: Any | None = None):
        np = _numpy()
        path = Path(path)
        if path.name != "head.safetensors" or path.parent.name != PREDICTOR:
            raise ValueError("learned-head requires the selected linear-seed17 archive")
        selection = _json(path.parent.parent / "selection.json")
        checkpoint_hashes = selection.get("checkpoint_sha256")
        if (
            selection.get("selected_predictor") != PREDICTOR
            or selection.get("selected_kind") != "linear"
            or not isinstance(checkpoint_hashes, dict)
            or checkpoint_hashes.get(PREDICTOR) != HEAD_SHA256
        ):
            raise ValueError("learned-head selection does not match pinned head")
        try:
            head_sha = _sha256(path)
        except OSError as error:
            raise ValueError("cannot read learned-head checkpoint") from error
        if head_sha != HEAD_SHA256:
            raise ValueError("learned-head checkpoint SHA256 mismatch")

        provenance = _json(path.parent.parent / "provenance.json")
        training = provenance.get("training")
        manifest = training.get("manifest") if isinstance(training, dict) else None
        source = manifest.get("source_model") if isinstance(manifest, dict) else None
        if (
            not isinstance(manifest, dict)
            or manifest.get("status") != "complete"
            or manifest.get("source_profile") != "ministral3_3b"
            or manifest.get("hidden_coordinate") != "post_final_norm"
            or manifest.get("norm_module") != "model.language_model.norm"
            or manifest.get("hidden_size") != HIDDEN_SIZE
            or manifest.get("backbone_frozen") is not True
            or manifest.get("label_order") != list(LABELS)
            or manifest.get("train_supported_display_labels") != list(TRAINED_LABELS)
            or manifest.get("source_label_token_ids") != TRAINED_TOKEN_IDS
            or not isinstance(source, dict)
            or source.get("model_path") != MODEL_ID
            or source.get("base_revision") != REVISION
            or source.get("model_class") != ARCHITECTURE
            or source.get("tokenizer_mode") != "native_chat"
        ):
            raise ValueError("learned-head feature provenance is incompatible")

        temperature_path = path.parent / "temperature.json"
        try:
            temperature_sha = _sha256(temperature_path)
        except OSError as error:
            raise ValueError("cannot read learned-head calibration") from error
        temperature = _json(temperature_path)
        if (
            temperature_sha != TEMPERATURE_SHA256
            or temperature.get("fit_split") != "calibration"
            or temperature.get("fit_performed") is not True
            or temperature.get("temperature") != CALIBRATION_TEMPERATURE
        ):
            raise ValueError("learned-head calibration does not match selected head")

        if safe_open is None:
            try:
                from safetensors import safe_open
            except ImportError as error:
                raise RuntimeError(
                    "Learned head requires safetensors; install jev-adapter[learned-head]."
                ) from error
        try:
            with safe_open(str(path), framework="np", device="cpu") as file:
                if set(file.keys()) != {"readout.weight", "readout.bias"}:
                    raise ValueError("learned-head checkpoint has unexpected tensors")
                weights = np.array(file.get_tensor("readout.weight"), copy=True)
                bias = np.array(file.get_tensor("readout.bias"), copy=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("cannot load learned-head safetensors") from error
        if (
            weights.shape != (len(LABELS), HIDDEN_SIZE)
            or bias.shape != (len(LABELS),)
            or weights.dtype != np.dtype("float32")
            or bias.dtype != np.dtype("float32")
            or not np.isfinite(weights).all()
            or not np.isfinite(bias).all()
        ):
            raise ValueError("learned-head tensors must be finite FP32 [62,3072]/[62]")
        weights.flags.writeable = False
        bias.flags.writeable = False
        return cls(weights, bias)

    def validate_engine(
        self,
        *,
        model_info: dict,
        server_info: dict,
        tokenizer_model: str,
        tokenizer_revision: str,
        labels: tuple[str, ...],
        token_ids: tuple[int, ...],
    ) -> None:
        if tokenizer_model != MODEL_ID or tokenizer_revision != REVISION:
            raise ValueError("learned-head requires the pinned 3B native tokenizer")
        if model_info.get("model_path") != MODEL_ID:
            raise ValueError("learned-head engine model path differs from training base")
        if server_info.get("revision") != REVISION:
            raise ValueError("learned-head engine revision differs from training base")
        if (
            model_info.get("model_type") != "mistral3"
            or not isinstance(model_info.get("architectures"), list)
            or ARCHITECTURE not in model_info["architectures"]
        ):
            raise ValueError("learned-head requires SGLang Mistral3 architecture")
        if server_info.get("return_hidden_states_mode") != "last":
            raise ValueError("learned-head requires --return-hidden-states-mode last")
        if server_info.get("dtype") not in ("bfloat16", "bf16"):
            raise ValueError("learned-head requires the BF16 training backbone")
        if any(
            source.get(key) not in (None, False)
            for source in (model_info, server_info)
            for key in ("quantization", "quantization_config")
        ):
            raise ValueError("learned-head requires an unquantized engine model")
        if tuple(labels[: len(LABELS)]) != tuple(LABELS) or len(labels) < len(LABELS):
            raise ValueError("learned-head requires canonical A-Z,a-z,0-9 labels")
        if len(token_ids) < len(LABELS) or dict(zip(labels[:3], token_ids[:3])) != TRAINED_TOKEN_IDS:
            raise ValueError("learned-head tokenizer differs from training token IDs")
        self._engine_token_ids = token_ids[: len(LABELS)]

    def score(
        self, hidden_states: Any, token_ids: tuple[int, ...]
    ) -> tuple[float, ...]:
        np = _numpy()
        if (
            not isinstance(hidden_states, list)
            or len(hidden_states) != HIDDEN_SIZE
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in hidden_states
            )
        ):
            raise ValueError("engine returned malformed post-final-norm hidden state")
        if (
            not self._engine_token_ids
            or not 2 <= len(token_ids) <= len(LABELS)
            or token_ids != self._engine_token_ids[: len(token_ids)]
        ):
            raise ValueError("requested labels differ from validated learned-head order")
        vector = np.asarray(hidden_states, dtype=np.float32)
        count = len(token_ids)
        logits = (self.weights[:count] @ vector + self.bias[:count]) / CALIBRATION_TEMPERATURE
        if not np.isfinite(logits).all():
            raise ValueError("learned head produced non-finite logits")
        return tuple(float(value) for value in logits)


class LearnedHeadBackend(CandidateReadoutBackend):
    """Use the hidden-only request path with the selected experimental head."""

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
            raise ValueError("learned-head requires the pinned native tokenizer")
        if tokenizer_model != MODEL_ID or tokenizer_revision != REVISION:
            raise ValueError("learned-head requires the pinned 3B model ID and revision")
        # The head's rows are A-Z, a-z, 0-9: the served order.
        kwargs.setdefault("label_scheme", "served")
        if kwargs["label_scheme"] != "served":
            raise ValueError("learned-head requires --label-scheme served")
        SGLangBackend.__init__(self, base_url, model, **kwargs)
        self._readout = LearnedHeadReadout.load(readout_path)
        self._tokenizer_model = tokenizer_model
        self._tokenizer_revision = tokenizer_revision
