"""CPU checks for exact-token parity preflight; no engine or model download."""

import argparse
import asyncio
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from benchmarks import qwen_tokenizer_parity as parity
from jev_adapter.benchmarks.data import normalize_record


class FakeBackend:
    instances: ClassVar[list] = []
    local_difference = False

    def __init__(self, **kwargs):
        self.local = kwargs.get("native_tokenizer") is not None
        self._model_info = {"model_path": parity.PROFILE["repo_id"]}
        self._server_info = {"revision": parity.PROFILE["revision"]}
        self._label_strings = ("A", "B")
        self._label_ids = (11, 12)
        self.prompts = []
        self.closed = False
        self.instances.append(self)

    async def start(self):
        pass

    async def close(self):
        self.closed = True

    def labels(self, count):
        assert count == 2
        return self._label_strings, self._label_ids

    async def _prepare_http(self, prompt, images, labels, token_ids, prefix, system):
        self.prompts.append((prompt, images, labels, token_ids, prefix, system))
        return [1, 2, 3], "rendered"

    async def _prepare_native(self, prompt, labels, token_ids, prefix):
        self.prompts.append((prompt, [], labels, token_ids, prefix, None))
        return [1, 9, 3] if self.local_difference else [1, 2, 3]


class QwenParityTests(unittest.TestCase):
    def setUp(self):
        self.item = normalize_record(
            {
                "state": "A short state",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Choose a route.",
                        "criteria": {"fast": None, "safe": None},
                        "label": "fast",
                    }
                },
                "_meta": {"id": "sample", "source": "policy", "variant": "clean"},
            },
            "fixture",
            "development",
        )
        self.args = argparse.Namespace(
            data=[Path("fixture.jsonl")],
            model="decision-model",
            limit=None,
            engine_url="http://localhost:30000",
            timeout=10,
            assistant_prefix=" ",
            prompt_wording="served",
            native_system_prompt="auto",
        )

    def run_comparison(self, different):
        FakeBackend.instances = []
        FakeBackend.local_difference = different
        with (
            patch.object(parity, "load_data", return_value=([self.item], [])),
            patch.object(parity, "SGLangBackend", FakeBackend),
            patch.object(parity.NativeTokenizer, "from_pretrained", return_value=object()),
        ):
            return asyncio.run(parity.compare(self.args))

    def test_same_request_prompt_and_ids_pass(self):
        report = self.run_comparison(False)
        self.assertEqual(report["status"], "matched")
        self.assertEqual(report["checked_questions"], 1)
        http, local = FakeBackend.instances
        self.assertEqual(http.prompts, local.prompts)
        self.assertIn("Context:\nA short state", http.prompts[0][0])
        self.assertEqual(http.prompts[0][4], " ")
        self.assertTrue(http.closed and local.closed)

    def test_equal_length_with_different_id_fails(self):
        report = self.run_comparison(True)
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(report["mismatched_questions"], 1)
        self.assertEqual(
            report["mismatch_samples"][0]["first_different_position"], 1
        )
        self.assertNotEqual(
            report["mismatch_samples"][0]["http_sha256"],
            report["mismatch_samples"][0]["local_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
