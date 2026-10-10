"""CPU validation of native online FP8 backbone with original BF16 LM rows."""

from types import SimpleNamespace

import pytest
from jev_adapter import inprocess

MODEL = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"


def readout_metadata(**changes):
    value = {
        "format": inprocess.FORMAT,
        "mode": "lm_rows",
        "model_id": MODEL,
        "engine_model_path": MODEL,
        "revision": REVISION,
        "architecture": inprocess.ARCHITECTURE,
        "hidden_size": 2,
        "label_scheme": "upper",
        "tokenizer_id": MODEL,
        "tokenizer_revision": REVISION,
        "labels": ["A", "B"],
        "token_ids": [165, 166],
        "temperature": 1.0,
    }
    value.update(changes)
    return value


def adapter_context():
    config = SimpleNamespace(
        model_type="mistral3",
        architectures=[inprocess.ARCHITECTURE],
        text_config=SimpleNamespace(hidden_size=2),
    )
    manager = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config, dtype="torch.bfloat16", quantization="fp8"
        )
    )
    args = SimpleNamespace(
        quantization="fp8",
        dtype="bfloat16",
        enable_fp32_lm_head=False,
        prefill_only_disable_kv_cache=False,
    )
    return readout_metadata(), manager, args


def test_adapter_bf16_default_unchanged_and_native_fp8_allowed():
    metadata, manager, args = adapter_context()
    inprocess._validate_readout_precision(metadata, manager, args)
    for quantization in (None, False):
        inprocess._validate_readout_precision(
            {"mode": "learned"}, None, SimpleNamespace(quantization=quantization)
        )


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("args", "quantization", "mxfp8"),
        ("args", "quantization", "quark"),
        ("args", "dtype", "auto"),
        ("args", "dtype", "float16"),
        ("args", "enable_fp32_lm_head", True),
        ("args", "prefill_only_disable_kv_cache", True),
        ("model_config", "dtype", "torch.float16"),
        ("model_config", "quantization", "mxfp8"),
        ("metadata", "mode", "learned"),
        ("config", "quantization_config", {"quant_method": "fp8"}),
        ("config", "compression_config", {}),
        ("text_config", "quantization_config", {}),
        ("text_config", "compression_config", {"format": "float-quantized"}),
    ],
)
def test_adapter_rejects_unsupported_fp8_modes(target, key, value):
    metadata, manager, args = adapter_context()
    config = manager.model_config.hf_config
    targets = {
        "args": args,
        "metadata": metadata,
        "model_config": manager.model_config,
        "config": config,
        "text_config": config.text_config,
    }
    if target == "metadata":
        targets[target][key] = value
    else:
        setattr(targets[target], key, value)
    with pytest.raises(ValueError):
        inprocess._validate_readout_precision(metadata, manager, args)
