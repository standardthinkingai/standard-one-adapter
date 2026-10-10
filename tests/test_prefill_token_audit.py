import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from benchmarks import prefill_token_audit as token_audit
from benchmarks.prefill_compare import request_for
from benchmarks.prefill_engine_bench import (
    canonical_sha256,
    file_sha256,
    read_settings,
    read_token_audit,
)
from jev_adapter.benchmarks.data import normalize_record
from jev_adapter.benchmarks.run import load_data
from jev_adapter.native_tokenizer import NativeTokenizer

REVISION = "a" * 40
MODEL_ID = "org/model"


class FakeTokenizer:
    all_special_ids = (1, 2, 9)

    def __init__(self, start=1):
        self.start = start

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 100 for char in text]

    def decode(self, ids, skip_special_tokens=True):
        if ids == [2]:
            return "" if skip_special_tokens else "[/INST]"
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
        base = [self.start] + self.encode(user["content"]) + [2]
        if messages[-1]["role"] == "assistant":
            return base + self.encode(messages[-1]["content"])
        return base


def inputs(tmp_path):
    raw = {
        "state": "중복 결제를 환불해 주세요.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "담당 부서",
                "criteria": {"billing": "결제", "technical": "기술", "general": "기타"},
                "label": "billing",
            },
            "refund": {"type": "noul", "instructions": "환불 요청?", "label": True},
            "urgency": {
                "type": "score",
                "instructions": "긴급도",
                "criteria": ["낮음", "보통", "높음"],
                "label": 1,
            },
        },
        "_meta": {"id": "one", "source": "policy", "variant": "clean"},
    }
    data = tmp_path / "prepared.jsonl"
    data.write_text(json.dumps(normalize_record(raw, "fixture", "development")) + "\n")
    metadata = {
        "format": "jev-gpu-readout-v1",
        "mode": "lm_rows",
        "model_id": MODEL_ID,
        "engine_model_path": MODEL_ID,
        "revision": REVISION,
        "architecture": "Mistral3ForConditionalGeneration",
        "hidden_size": 2,
        "labels": ["A", "B", "C"],
        "token_ids": [165, 166, 167],
        "temperature": 1.0,
    }
    readout = tmp_path / "head.npz"
    np.savez(
        readout,
        metadata=np.bytes_(json.dumps(metadata)),
        weights=np.zeros((3, 2), dtype=np.float32),
        bias=np.zeros(3, dtype=np.float32),
    )
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hardware": "H200",
                "model_path": MODEL_ID,
                "model_revision": REVISION,
                "precision": "bfloat16",
                "prompt_contract": "served-v1",
                "label_contract": "single-token-A-C-v1",
                "labels": metadata["labels"],
                "token_ids": metadata["token_ids"],
                "head_mode": "lm_rows",
                "head_artifact": {"path": str(readout), "sha256": file_sha256(readout)},
            }
        )
    )
    return data, settings, readout


@pytest.mark.asyncio
async def test_records_actual_backend_submissions_and_benchmark_hashes(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    data, settings, readout = inputs(tmp_path)
    stock, candidate = await token_audit.audit(
        data=[data],
        model="decision-model",
        model_id=MODEL_ID,
        revision=REVISION,
        readout=readout,
        settings_manifest=settings,
        limit=1,
        stock_tokenizer=NativeTokenizer(FakeTokenizer()),
        integrated_tokenizer=FakeTokenizer(),
    )
    items, _ = load_data([data], "decision-model", limit=1)
    request = request_for(items[0], "decision-model", None)
    assert (
        stock["workload_sha256"]
        == candidate["workload_sha256"]
        == canonical_sha256([request])
    )
    for audit in (stock, candidate):
        assert audit["gpu_parity_claim"] is False
        entry = audit["requests"][0]
        assert entry["request_sha256"] == canonical_sha256(request)
        assert entry["input_token_ids_sha256"] == canonical_sha256(
            [question["input_token_ids"] for question in entry["questions"]]
        )
        assert [question["id"] for question in entry["questions"]] == [
            "route",
            "refund",
            "urgency",
        ]
        assert all(
            len(question["input_token_ids"]) == question["input_tokens"] > 0
            for question in entry["questions"]
        )
        assert audit["tokenizer"]["class_source_sha256"]
    assert (
        stock["requests"][0]["input_token_ids_sha256"]
        == candidate["requests"][0]["input_token_ids_sha256"]
    )
    settings_value, _ = read_settings(settings)
    for name, result in (("stock", stock), ("candidate", candidate)):
        path = tmp_path / f"{name}-audit.json"
        path.write_text(json.dumps(result))
        checked = read_token_audit(
            path, settings_value, canonical_sha256([request]), [request]
        )
        assert checked["input_token_ids_sha256"] == [
            result["requests"][0]["input_token_ids_sha256"]
        ]


@pytest.mark.asyncio
async def test_different_full_chat_ids_are_detected(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    data, settings, readout = inputs(tmp_path)
    stock, candidate = await token_audit.audit(
        data=[data],
        model="decision-model",
        model_id=MODEL_ID,
        revision=REVISION,
        readout=readout,
        settings_manifest=settings,
        limit=1,
        stock_tokenizer=NativeTokenizer(FakeTokenizer(start=1)),
        integrated_tokenizer=FakeTokenizer(start=9),
    )
    assert (
        stock["requests"][0]["input_token_ids_sha256"]
        != candidate["requests"][0]["input_token_ids_sha256"]
    )


def test_cli_writes_both_audits_then_exits_nonzero_on_mismatch(monkeypatch, tmp_path):
    async def fake_audit(**kwargs):
        return (
            {"requests": [{"input_token_ids_sha256": "a" * 64}]},
            {"requests": [{"input_token_ids_sha256": "b" * 64}]},
        )

    monkeypatch.setattr(token_audit, "audit", fake_audit)
    output = tmp_path / "audit"
    result = token_audit.main(
        [
            "--data",
            str(Path("data.jsonl")),
            "--model",
            "decision-model",
            "--model-id",
            MODEL_ID,
            "--revision",
            REVISION,
            "--readout",
            "head.npz",
            "--settings-manifest",
            "settings.json",
            "--output-dir",
            str(output),
        ]
    )
    assert result == 1
    assert (output / "stock-token-audit.json").is_file()
    assert (output / "candidate-token-audit.json").is_file()


def test_cpu_only_guard_rejects_visible_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        token_audit._cpu_only()
