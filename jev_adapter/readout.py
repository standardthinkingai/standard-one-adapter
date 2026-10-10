"""Selected LM-head rows for the optional, text-only hidden-state backend.

The artifact contains only candidate rows, never the full vocabulary matrix.
It is tied to a model revision and to the engine's exact label token IDs.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .labels import follows_scheme_order, validate_label_scheme

# v2 records the label scheme and tokenizer; v1 artifacts carry the served order.
FORMAT = "jev-candidate-readout-v2"
LEGACY_FORMAT = "jev-candidate-readout-v1"
ARCHITECTURE = "Mistral3ForConditionalGeneration"


def _numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Candidate readout requires NumPy; install jev-adapter[candidate-readout]."
        ) from error
    return np


@dataclass(frozen=True)
class CandidateReadout:
    metadata: dict[str, Any]
    weights: Any

    @classmethod
    def load(cls, path: str | Path) -> "CandidateReadout":
        np = _numpy()
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"metadata", "weights"}:
                    raise ValueError(
                        "readout artifact must contain metadata and weights"
                    )
                raw = data["metadata"].item()
                if not isinstance(raw, (bytes, str)):
                    raise TypeError("readout metadata is malformed")
                metadata = json.loads(raw)
                weights = np.array(data["weights"], copy=True)
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("cannot load candidate readout artifact") from error
        cls.validate_data(metadata, weights)
        weights.flags.writeable = False
        return cls(metadata, weights)

    @staticmethod
    def validate_data(metadata: dict, weights: Any) -> None:
        np = _numpy()
        if not isinstance(metadata, dict) or metadata.get("format") not in (
            FORMAT,
            LEGACY_FORMAT,
        ):
            raise ValueError("unsupported candidate readout format")
        if metadata.get("mode") != "lm_rows":
            raise ValueError("only selected LM-head rows are supported")
        for field in (
            "model_id",
            "revision",
            "engine_model_path",
            "source_weight_key",
            "source_dtype",
            "config_sha256",
        ):
            if not isinstance(metadata.get(field), str) or not metadata[field]:
                raise ValueError(f"readout {field} is missing")
        if not re.fullmatch(r"[0-9a-f]{40}", metadata["revision"]):
            raise ValueError("readout revision must be a 40-character commit SHA")
        if not re.fullmatch(r"[0-9a-f]{64}", metadata["config_sha256"]):
            raise ValueError("readout config SHA256 is invalid")
        if metadata["source_weight_key"] not in {
            "model.language_model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "lm_head.weight",
            "language_model.lm_head.weight",
        } or metadata["source_dtype"] not in {"BF16", "F16", "F32"}:
            raise ValueError("readout source tensor is unsupported")
        if metadata.get("architecture") != ARCHITECTURE:
            raise ValueError("readout architecture is unsupported")
        hidden_size = metadata.get("hidden_size")
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("readout hidden_size is invalid")
        labels = metadata.get("labels")
        token_ids = metadata.get("token_ids")
        if (
            not isinstance(labels, list)
            or not isinstance(token_ids, list)
            or not 2 <= len(labels) <= 255
            or len(labels) != len(token_ids)
            or any(not isinstance(label, str) or not label for label in labels)
            or any(type(token) is not int or token < 0 for token in token_ids)
            or len(set(labels)) != len(labels)
            or len(set(token_ids)) != len(token_ids)
        ):
            raise ValueError("readout labels/token IDs are invalid")
        if metadata["format"] == FORMAT:
            try:
                scheme = validate_label_scheme(metadata.get("label_scheme"))
            except ValueError as error:
                raise ValueError(f"readout {error}") from None
            if (
                not isinstance(metadata.get("tokenizer_id"), str)
                or not metadata["tokenizer_id"]
                or not isinstance(metadata.get("tokenizer_revision"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", metadata["tokenizer_revision"])
            ):
                raise ValueError("readout tokenizer ID/revision is missing")
            # The labels were chosen with the pinned model's own tokenizer.
            if metadata["tokenizer_id"] not in (
                metadata["model_id"],
                metadata["engine_model_path"],
            ) or metadata["tokenizer_revision"] != metadata["revision"]:
                raise ValueError("readout tokenizer differs from the readout model")
            if not follows_scheme_order(scheme, labels):
                raise ValueError(f"readout labels do not follow the {scheme} scheme")
        if (
            not isinstance(weights, np.ndarray)
            or weights.dtype != np.dtype("float32")
            or weights.shape != (len(labels), hidden_size)
            or not np.isfinite(weights).all()
        ):
            raise ValueError("readout weights must be finite float32 candidate rows")

    @property
    def label_scheme(self) -> str:
        if self.metadata["format"] == LEGACY_FORMAT:
            return "served"
        return self.metadata["label_scheme"]

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
        meta = self.metadata
        if (
            meta["model_id"] != tokenizer_model
            or meta["revision"] != tokenizer_revision
        ):
            raise ValueError("readout model ID/revision differs from pinned tokenizer")
        if model_info.get("model_path") != meta["engine_model_path"]:
            raise ValueError("readout engine model path differs from served checkpoint")
        if server_info.get("revision") != meta["revision"]:
            raise ValueError("readout revision differs from engine revision")
        if model_info.get(
            "model_type"
        ) != "mistral3" or ARCHITECTURE not in model_info.get("architectures", []):
            raise ValueError("candidate readout requires SGLang Mistral3 architecture")
        if any(
            source.get(key) not in (None, False)
            for source in (model_info, server_info)
            for key in ("quantization", "quantization_config")
        ):
            raise ValueError("candidate readout requires an unquantized engine model")
        mode = server_info.get("return_hidden_states_mode")
        if mode not in (None, "last", "full"):
            raise ValueError("SGLang must enable --return-hidden-states-mode last")
        if tuple(meta["labels"]) != labels or tuple(meta["token_ids"]) != token_ids:
            raise ValueError("readout labels/token IDs differ from engine tokenizer")

    def score(
        self, hidden_states: Any, token_ids: tuple[int, ...]
    ) -> tuple[float, ...]:
        np = _numpy()
        hidden_size = self.metadata["hidden_size"]
        if (
            not isinstance(hidden_states, list)
            or len(hidden_states) != hidden_size
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in hidden_states
            )
        ):
            raise ValueError("engine returned malformed last hidden state")
        all_ids = self.metadata["token_ids"]
        index = {token: position for position, token in enumerate(all_ids)}
        try:
            rows = [index[token] for token in token_ids]
        except KeyError as error:
            raise ValueError(
                "requested label is absent from candidate readout"
            ) from error
        vector = np.asarray(hidden_states, dtype=np.float32)
        logits = self.weights[rows] @ vector
        if not np.isfinite(logits).all():
            raise ValueError("candidate readout produced non-finite logits")
        return tuple(float(value) for value in logits)
