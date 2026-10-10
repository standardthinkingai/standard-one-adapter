"""Export a small, engine-integrated readout without loading the base model.

Usage: python -m jev_adapter.export_gpu_readout --candidate-readout rows.npz
       --output gpu-readout.npz
   or: python -m jev_adapter.export_gpu_readout --learned-head head.safetensors
       --output gpu-readout.npz

The learned-head path reads only the verified frozen linear head. The tokenizer
is loaded from the pinned 3B model revision in the local Hugging Face cache.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .labels import follows_scheme_order, validate_label_scheme
from .learned_head import (
    CALIBRATION_TEMPERATURE,
    HEAD_SHA256,
    LABELS,
    MODEL_ID,
    REVISION,
    TRAINED_TOKEN_IDS,
    LearnedHeadReadout,
)
from .readout import ARCHITECTURE, CandidateReadout, _numpy

# v2 adds label_scheme, tokenizer_id and tokenizer_revision to v1.
FORMAT = "jev-gpu-readout-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Learned-head export requires transformers and a cached pinned tokenizer."
        ) from error
    return AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )


def _learned_token_ids(tokenizer: Any) -> list[int]:
    token_ids: list[int] = []
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if (
            not isinstance(ids, list)
            or len(ids) != 1
            or type(ids[0]) is not int
            or ids[0] < 0
            or tokenizer.decode([ids[0]], skip_special_tokens=True) != label
        ):
            raise ValueError(f"pinned tokenizer does not round-trip label {label!r}")
        token_ids.append(ids[0])
    if len(set(token_ids)) != len(LABELS):
        raise ValueError("pinned tokenizer has duplicate canonical label token IDs")
    if dict(zip(LABELS[:3], token_ids[:3])) != TRAINED_TOKEN_IDS:
        raise ValueError("pinned tokenizer A-C token IDs differ from trained head")
    return token_ids


def _validate(metadata: dict, weights: Any, bias: Any) -> None:
    np = _numpy()
    if metadata.get("format") != FORMAT or metadata.get("mode") not in {
        "lm_rows",
        "learned",
    }:
        raise ValueError("invalid GPU readout format or mode")
    if (
        metadata.get("architecture") != ARCHITECTURE
        or not isinstance(metadata.get("model_id"), str)
        or not metadata["model_id"].strip()
        or not isinstance(metadata.get("engine_model_path"), str)
        or not metadata["engine_model_path"].strip()
        or not isinstance(metadata.get("revision"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", metadata["revision"])
        or not isinstance(metadata.get("source_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", metadata["source_sha256"])
    ):
        raise ValueError("GPU readout model identity is invalid")
    labels, token_ids = metadata.get("labels"), metadata.get("token_ids")
    hidden_size = metadata.get("hidden_size")
    temperature = metadata.get("temperature")
    if (
        not isinstance(labels, list)
        or not isinstance(token_ids, list)
        or not 2 <= len(labels) <= 255
        or len(token_ids) != len(labels)
        or any(not isinstance(label, str) or not label for label in labels)
        or any(type(token) is not int or token < 0 for token in token_ids)
        or len(set(labels)) != len(labels)
        or len(set(token_ids)) != len(token_ids)
        or type(hidden_size) is not int
        or hidden_size < 1
        or type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("GPU readout labels, shape, or temperature are invalid")
    try:
        scheme = validate_label_scheme(metadata.get("label_scheme"))
    except ValueError as error:
        raise ValueError(f"GPU readout {error}") from None
    if (
        metadata.get("tokenizer_id")
        not in (metadata["model_id"], metadata["engine_model_path"])
        or metadata.get("tokenizer_revision") != metadata["revision"]
    ):
        raise ValueError("GPU readout tokenizer must be the pinned model tokenizer")
    if not follows_scheme_order(scheme, labels):
        raise ValueError(f"GPU readout labels do not follow the {scheme} scheme")
    if (
        not isinstance(weights, np.ndarray)
        or not isinstance(bias, np.ndarray)
        or weights.dtype != np.dtype("float32")
        or bias.dtype != np.dtype("float32")
        or weights.shape != (len(labels), hidden_size)
        or bias.shape != (len(labels),)
        or not np.isfinite(weights).all()
        or not np.isfinite(bias).all()
    ):
        raise ValueError("GPU readout tensors must be finite float32 [K,D] and [K]")


def export_gpu_readout(
    *,
    output: str | Path,
    candidate_readout: str | Path | None = None,
    learned_head: str | Path | None = None,
    engine_model_path: str | None = None,
    tokenizer: Any | None = None,
    max_labels: int | None = None,
) -> dict:
    """Create the common GPU artifact from exactly one validated source."""
    np = _numpy()
    if (candidate_readout is None) == (learned_head is None):
        raise ValueError("provide exactly one candidate readout or learned head")
    if engine_model_path is not None and not engine_model_path.strip():
        raise ValueError("engine_model_path must be nonempty")
    if candidate_readout is not None and tokenizer is not None:
        raise ValueError("tokenizer injection is only valid for a learned head")
    if max_labels is not None and (type(max_labels) is not int or max_labels < 2):
        raise ValueError("max_labels must be an integer of at least 2")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)

    if candidate_readout is not None:
        source = Path(candidate_readout)
        readout = CandidateReadout.load(source)
        old = readout.metadata
        metadata = {
            "format": FORMAT,
            "mode": "lm_rows",
            "model_id": old["model_id"],
            "engine_model_path": engine_model_path or old["engine_model_path"],
            "revision": old["revision"],
            "architecture": old["architecture"],
            "hidden_size": old["hidden_size"],
            "label_scheme": readout.label_scheme,
            "tokenizer_id": old.get("tokenizer_id", old["model_id"]),
            "tokenizer_revision": old.get("tokenizer_revision", old["revision"]),
            "labels": list(old["labels"]),
            "token_ids": list(old["token_ids"]),
            "temperature": 1.0,
            "source_sha256": _sha256(source),
        }
        weights = np.array(readout.weights, dtype=np.float32, copy=True)
        bias = np.zeros(len(metadata["labels"]), dtype=np.float32)
    else:
        source = Path(learned_head)
        readout = LearnedHeadReadout.load(source)
        token_ids = _learned_token_ids(
            tokenizer if tokenizer is not None else _pinned_tokenizer()
        )
        metadata = {
            "format": FORMAT,
            "mode": "learned",
            "model_id": MODEL_ID,
            "engine_model_path": engine_model_path or MODEL_ID,
            "revision": REVISION,
            "architecture": ARCHITECTURE,
            "hidden_size": readout.weights.shape[1],
            # The head was trained on A-Z, a-z, 0-9: the served order.
            "label_scheme": "served",
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": REVISION,
            "labels": list(LABELS),
            "token_ids": token_ids,
            "temperature": CALIBRATION_TEMPERATURE,
            "source_sha256": HEAD_SHA256,
        }
        weights = np.array(readout.weights, dtype=np.float32, copy=True)
        bias = np.array(readout.bias, dtype=np.float32, copy=True)

    if max_labels is not None:
        available = len(metadata["labels"])
        if max_labels > available:
            raise ValueError(
                f"max_labels={max_labels} exceeds the source's {available} labels"
            )
        metadata["labels"] = metadata["labels"][:max_labels]
        metadata["token_ids"] = metadata["token_ids"][:max_labels]
        weights = weights[:max_labels].copy()
        bias = bias[:max_labels].copy()

    _validate(metadata, weights, bias)
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with destination.open("xb") as file:
            created = True
            np.savez(
                file,
                metadata=np.bytes_(json.dumps(metadata, separators=(",", ":"))),
                weights=weights,
                bias=bias,
            )
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--candidate-readout", help="Validated candidate LM-row .npz")
    sources.add_argument(
        "--learned-head", help="Verified linear-seed17/head.safetensors"
    )
    parser.add_argument("--output", required=True, help="New GPU readout .npz path")
    parser.add_argument(
        "--max-labels",
        type=int,
        help="Export only the leading N labels (2 to the source label count)",
    )
    parser.add_argument(
        "--engine-model-path",
        help="Exact /model_info.model_path; defaults to source model ID/path",
    )
    args = parser.parse_args(argv)
    metadata = export_gpu_readout(
        output=args.output,
        candidate_readout=args.candidate_readout,
        learned_head=args.learned_head,
        engine_model_path=args.engine_model_path,
        max_labels=args.max_labels,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "mode": metadata["mode"],
                "labels": len(metadata["labels"]),
                "label_scheme": metadata["label_scheme"],
                "hidden_size": metadata["hidden_size"],
            }
        )
    )


if __name__ == "__main__":
    main()
