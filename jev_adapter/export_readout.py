"""Export only the candidate LM-head rows from a local Ministral3 checkpoint.

Uses safetensors.get_slice; it never materializes the full vocabulary matrix.
The output is an ordinary NumPy .npz file used by --backend candidate-only.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .labels import LABEL_SCHEMES, resolve_label_scheme, select_labels
from .readout import ARCHITECTURE, FORMAT, CandidateReadout, _numpy


def _select_labels(tokenizer: Any, scheme: str) -> tuple[list[str], list[int]]:
    labels, token_ids = select_labels(tokenizer, scheme)
    if len(labels) < 2:
        raise ValueError("tokenizer has fewer than two usable single-token labels")
    return labels, token_ids


def _source_file(
    checkpoint: Path, keys: tuple[str, ...], safe_open: Any
) -> tuple[Path, str]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        matches = [key for key in keys if key in weight_map]
        if len(matches) != 1:
            raise ValueError(
                "checkpoint must contain exactly one supported readout tensor"
            )
        key = matches[0]
        name = weight_map[key]
        if not isinstance(name, str):
            raise ValueError(f"checkpoint index has no {key}")
        # Validate the index entry itself before following any symlink. Hugging
        # Face snapshots link shard names to ../../blobs/<hash>, whose target
        # deliberately has no .safetensors suffix and lies outside snapshots/.
        parts = name.split("/")
        if (
            not name
            or "\\" in name
            or Path(name).is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or Path(name).suffix != ".safetensors"
        ):
            raise ValueError("checkpoint index names an invalid weight file")
        source = checkpoint.joinpath(*parts)
        resolved = source.resolve()
        checkpoint_root = checkpoint.resolve()
        if not resolved.is_relative_to(checkpoint_root):
            # Cached shards may resolve through models--*/blobs/<hash> to the
            # shared hub/blobs/<prefix>/<hash> store. No other external target
            # is accepted from a checkpoint index.
            model_cache = checkpoint_root.parent.parent
            model_blobs = model_cache / "blobs"
            shared_blobs = model_cache.parent / "blobs"
            if (
                checkpoint_root.parent.name != "snapshots"
                or not model_cache.name.startswith("models--")
                or not (
                    resolved.is_relative_to(model_blobs.resolve())
                    or (
                        model_cache.parent.name == "hub"
                        and resolved.is_relative_to(shared_blobs.resolve())
                    )
                )
            ):
                raise ValueError("checkpoint index names an invalid weight file")
        if not source.is_file():
            raise ValueError(f"checkpoint weight file is missing: {source}")
        return source, key
    matches: list[tuple[Path, str]] = []
    for source in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            present = set(handle.keys())
            matches.extend((source, key) for key in keys if key in present)
    if len(matches) != 1:
        raise ValueError("checkpoint must contain exactly one supported readout tensor")
    return matches[0]


def export_readout(
    checkpoint: str | Path,
    *,
    model_id: str,
    revision: str,
    output: str | Path,
    engine_model_path: str | None = None,
    tokenizer: Any | None = None,
    safe_open: Any | None = None,
    label_scheme: str | None = None,
) -> dict:
    """Create a small, revision-bound candidate-row artifact from local files.

    ``label_scheme`` None means JEV_LABEL_SCHEME, else the default (upper).
    """
    np = _numpy()
    label_scheme = resolve_label_scheme(label_scheme)
    root = Path(checkpoint).resolve()
    destination = Path(output)
    if not root.is_dir():
        raise ValueError("checkpoint must be a local directory")
    if not model_id.strip() or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model_id and 40-character commit revision are required")
    if destination.exists():
        raise FileExistsError(destination)
    config_path = root / "config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    text_config = config.get("text_config")
    if (
        config.get("model_type") != "mistral3"
        or config.get("architectures") != [ARCHITECTURE]
        or not isinstance(text_config, dict)
    ):
        raise ValueError(
            "only Mistral3 conditional-generation checkpoints are supported"
        )
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("quantized checkpoints are unsupported; export BF16/FP16 rows")
    hidden_size = text_config.get("hidden_size")
    vocab_size = text_config.get("vocab_size")
    if (
        type(hidden_size) is not int
        or hidden_size < 1
        or type(vocab_size) is not int
        or vocab_size < 2
    ):
        raise ValueError("checkpoint hidden/vocabulary dimensions are invalid")
    text_tie = text_config.get("tie_word_embeddings")
    model_tie = config.get("tie_word_embeddings")
    ties = [value for value in (text_tie, model_tie) if type(value) is bool]
    if not ties or len(set(ties)) != 1:
        raise ValueError("checkpoint tie_word_embeddings is missing or inconsistent")
    keys = (
        (
            "model.language_model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
        )
        if ties[0]
        else ("lm_head.weight", "language_model.lm_head.weight")
    )
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=False)
    labels, token_ids = _select_labels(tokenizer, label_scheme)
    if safe_open is None:
        from safetensors import safe_open as safe_open_impl

        safe_open = safe_open_impl
    source, key = _source_file(root, keys, safe_open)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        if key not in set(handle.keys()):
            raise ValueError(f"checkpoint weight file lacks {key}")
        sliced = handle.get_slice(key)
        shape = sliced.get_shape()
        dtype = sliced.get_dtype()
        if shape != [vocab_size, hidden_size]:
            raise ValueError("checkpoint readout matrix disagrees with text_config")
        if dtype not in {"BF16", "F16", "F32"}:
            raise ValueError("checkpoint readout matrix must be BF16/FP16/FP32")
        if max(token_ids) >= vocab_size:
            raise ValueError("tokenizer label ID exceeds checkpoint vocabulary")
        # One narrow row read per candidate. Never call get_tensor(key).
        weights = np.stack(
            [sliced[token].float().numpy() for token in token_ids]
        ).astype(np.float32, copy=False)
    metadata = {
        "format": FORMAT,
        "mode": "lm_rows",
        "model_id": model_id,
        "revision": revision,
        "engine_model_path": engine_model_path or model_id,
        "architecture": ARCHITECTURE,
        "source_weight_key": key,
        "source_dtype": dtype,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "hidden_size": hidden_size,
        "label_scheme": label_scheme,
        # The tokenizer is read from the same pinned checkpoint.
        "tokenizer_id": model_id,
        "tokenizer_revision": revision,
        "labels": labels,
        "token_ids": token_ids,
    }
    CandidateReadout.validate_data(metadata, weights)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        np.savez(file, metadata=np.bytes_(json.dumps(metadata)), weights=weights)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Local safetensors directory"
    )
    parser.add_argument(
        "--model-id", required=True, help="Pinned Hugging Face model ID"
    )
    parser.add_argument(
        "--revision", required=True, help="40-character model commit SHA"
    )
    parser.add_argument(
        "--engine-model-path",
        help="Exact /model_info.model_path; defaults to --model-id",
    )
    parser.add_argument("--output", required=True, help="New .npz artifact path")
    parser.add_argument(
        "--label-scheme",
        choices=LABEL_SCHEMES,
        default=None,
        help="Answer-label order of the exported rows: 'upper' (A-Z, then AA, "
        "AB, ...) or 'served' (A-Z, a-z, 0-9, then AA, ...). Must match the "
        "serving label scheme. Env: JEV_LABEL_SCHEME. Default: upper.",
    )
    args = parser.parse_args()
    metadata = export_readout(
        args.checkpoint,
        model_id=args.model_id,
        revision=args.revision,
        output=args.output,
        engine_model_path=args.engine_model_path,
        label_scheme=args.label_scheme,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "model_id": metadata["model_id"],
                "revision": metadata["revision"],
                "labels": len(metadata["labels"]),
                "label_scheme": metadata["label_scheme"],
                "hidden_size": metadata["hidden_size"],
            }
        )
    )


if __name__ == "__main__":
    main()
