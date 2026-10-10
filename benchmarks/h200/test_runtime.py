"""CPU contract checks; no CUDA, engine installation, or network required."""

import base64
import contextlib
import importlib.util
import io
import json
import math
import struct
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "h200_runtime", Path(__file__).with_name("runtime.py")
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.manifest = runtime.load_manifest()
        self.key = "qwen35-4b-bf16"
        self.model = self.manifest["models"][self.key]
        self.server = {
            "model_path": self.model["repo_id"],
            "revision": self.model["revision"],
            "tp_size": 1,
            "disable_radix_cache": True,
            "mm_preprocess_cache_size_mb": 0,
            "enable_prefix_mm_cache": False,
            "enable_mm_global_cache": False,
            "speculative_algorithm": None,
            "quantization": None,
            "dtype": "bfloat16",
            "max_running_requests": 32,
            "context_length": 8192,
        }
        self.info = {
            "model_path": self.model["repo_id"],
            "served_model_name": "decision-model",
            "is_generation": True,
            "has_image_understanding": True,
            "architectures": [self.model["architecture"]],
        }

    def test_default_comparison_has_matching_precision_and_immutable_revisions(self):
        for key in self.manifest["default_matrix"]:
            model = self.manifest["models"][key]
            self.assertIsNone(model["quantization"])
            self.assertEqual(model["dtype"], "bfloat16")
            self.assertRegex(model["revision"], r"^[a-f0-9]{40}$")
        self.assertEqual(
            self.manifest["models"]["ministral3-3b-fp8"]["quantization"], "fp8"
        )

    def test_dry_run_does_not_inspect_gpu_or_contact_network(self):
        with (
            patch.object(runtime, "inspect_runtime") as inspect,
            patch.object(runtime, "request_json") as request,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            code = runtime.main(["launch", self.key])
        self.assertEqual(code, 0)
        inspect.assert_not_called()
        request.assert_not_called()
        self.assertIn(self.model["revision"], output.getvalue())
        self.assertIn("SGLANG_VLM_CACHE_SIZE_MB=0", output.getvalue())

    def test_flashinfer_prefill_is_opt_in_and_checked_by_preflight(self):
        baseline, _ = runtime.launch_spec(
            self.manifest, self.key, "python", "0", 30000
        )
        experimental, _ = runtime.launch_spec(
            self.manifest,
            self.key,
            "python",
            "0",
            30000,
            linear_attn_prefill_backend="flashinfer",
        )
        self.assertNotIn("--linear-attn-prefill-backend", baseline)
        self.assertEqual(
            experimental[-2:], ["--linear-attn-prefill-backend", "flashinfer"]
        )
        self.assertEqual(experimental[:-2], baseline)
        with self.assertRaisesRegex(ValueError, "linear_attn_prefill_backend"):
            runtime.validate_server(
                self.server,
                self.info,
                self.manifest,
                self.key,
                linear_attn_prefill_backend="flashinfer",
            )
        runtime.validate_server(
            dict(self.server, linear_attn_prefill_backend="flashinfer"),
            self.info,
            self.manifest,
            self.key,
            linear_attn_prefill_backend="flashinfer",
        )
        with self.assertRaisesRegex(ValueError, "qwen35-4b-bf16"):
            runtime.launch_spec(
                self.manifest,
                "ministral3-3b-bf16",
                "python",
                "0",
                30000,
                linear_attn_prefill_backend="flashinfer",
            )

    def test_engine_wheel_requires_exact_version_origin_and_matching_recorded_hash(
        self,
    ):
        engine = self.manifest["engine"]
        good = {
            "sglang_version": engine["version"],
            "sglang_origin": {
                "url": engine["wheel_url"],
                "archive_info": {"hashes": {"sha256": engine["wheel_sha256"]}},
            },
        }
        runtime.validate_engine_install(good, engine)
        uv_origin = {**good["sglang_origin"], "archive_info": {}}
        runtime.validate_engine_install(dict(good, sglang_origin=uv_origin), engine)
        for wrong in (
            dict(good, sglang_version="0.0.0"),
            dict(good, sglang_origin={}),
            dict(
                good,
                sglang_origin={
                    **good["sglang_origin"],
                    "archive_info": {"hashes": {"sha256": "0" * 64}},
                },
            ),
        ):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, "wheel"):
                runtime.validate_engine_install(wrong, engine)

    def test_reject_unsafe_or_misleading_live_configuration(self):
        runtime.validate_server(self.server, self.info, self.manifest, self.key)
        wrong = {
            "model_path": "Qwen/other-model",
            "revision": "main",
            "tp_size": 2,
            "disable_radix_cache": False,
            "mm_preprocess_cache_size_mb": 100,
            "enable_prefix_mm_cache": True,
            "enable_mm_global_cache": True,
            "speculative_algorithm": "NEXTN",
            "quantization": "fp8",
            "max_running_requests": 1,
        }
        for field, value in wrong.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime.validate_server(
                    dict(self.server, **{field: value}),
                    self.info,
                    self.manifest,
                    self.key,
                )

    def test_vision_capability_is_required_only_when_requested(self):
        runtime.validate_server(
            self.server,
            dict(self.info, has_image_understanding=False),
            self.manifest,
            self.key,
        )
        with self.assertRaisesRegex(ValueError, "vision"):
            runtime.validate_server(
                self.server,
                dict(self.info, has_image_understanding=False),
                self.manifest,
                self.key,
                probe_vision=True,
            )

    def test_reject_model_swapped_after_startup(self):
        with self.assertRaisesRegex(ValueError, "active model"):
            runtime.validate_server(
                self.server,
                dict(self.info, model_path="Qwen/other-model"),
                self.manifest,
                self.key,
            )

    def test_png_fixture_is_valid_and_has_expected_dimensions(self):
        data = base64.b64decode(runtime.probe_image().split(",", 1)[1])
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", data[16:24]), (224, 224))
        offset = 8
        pixels = b""
        while offset < len(data):
            size = struct.unpack(">I", data[offset : offset + 4])[0]
            kind = data[offset + 4 : offset + 8]
            content = data[offset + 8 : offset + 8 + size]
            crc = struct.unpack(">I", data[offset + 8 + size : offset + 12 + size])[0]
            self.assertEqual(crc, zlib.crc32(kind + content) & 0xFFFFFFFF)
            if kind == b"IDAT":
                pixels += zlib.decompress(content)
            offset += 12 + size
        self.assertEqual(len(pixels), 224 * (1 + 224 * 3))

    def fake_api(self, changes=None):
        self.calls = []

        def call(base, path, payload=None, timeout=60):
            self.calls.append((path, payload))
            if path == "/health_generate":
                return {}
            if path == "/model_info":
                return self.info
            if path == "/server_info":
                return self.server
            if path == "/v1/tokenize":
                if "messages" in payload:
                    return {"tokens": [10, 11]}
                return {"tokens": [[10, 11], [10, 11, 65], [10, 11, 66]]}
            if path == "/v1/detokenize":
                return {"text": "<user>test</user><assistant>"}
            if path == "/generate":
                self.assertEqual(payload["sampling_params"]["max_new_tokens"], 0)
                self.assertEqual(payload["token_ids_logprob"], [65, 66])
                meta = {
                    "completion_tokens": 0,
                    "prompt_tokens": 80 if "image_data" in payload else 40,
                    "cached_tokens": 0,
                    "output_token_ids_logprobs": [[[-1.0, 65], [-2.0, 66]]],
                }
                return {"meta_info": dict(meta, **(changes or {}))}
            self.fail(path)

        return call

    def test_default_probe_is_text_only_even_without_vision_capability(self):
        self.info["has_image_understanding"] = False
        with (
            patch.object(runtime, "request_json", side_effect=self.fake_api()),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(runtime.main(["probe", self.key]), 0)
        result = json.loads(output.getvalue())
        generates = [payload for path, payload in self.calls if path == "/generate"]
        self.assertEqual(len(generates), 1)
        self.assertNotIn("image_data", generates[0])
        self.assertIn("input_ids", generates[0])
        self.assertEqual(result["probe_scope"], ["text"])
        self.assertIsNone(result["vision_probe"])

    def test_probe_vision_flag_exercises_image_path_and_records_scope(self):
        with (
            patch.object(runtime, "request_json", side_effect=self.fake_api()),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(runtime.main(["probe", self.key, "--probe-vision"]), 0)
        result = json.loads(output.getvalue())
        generates = [payload for path, payload in self.calls if path == "/generate"]
        self.assertEqual(len(generates), 2)
        self.assertNotIn("image_data", generates[0])
        self.assertEqual(generates[1]["image_data"], [runtime.probe_image()])
        self.assertEqual(result["probe_scope"], ["text", "vision"])
        self.assertGreater(
            result["vision_probe"]["prompt_tokens"],
            result["text_probe"]["prompt_tokens"],
        )

    def test_vision_probe_transmits_actual_image_bytes(self):
        image = runtime.probe_image()
        with patch.object(runtime, "request_json", side_effect=self.fake_api()):
            result = runtime.probe_score(
                "http://127.0.0.1:30000", "decision-model", image, 5
            )
        payload = self.calls[-1][1]
        self.assertEqual(payload["image_data"], [image])
        self.assertNotIn("input_ids", payload)
        self.assertEqual(result["completion_tokens"], 0)
        self.assertEqual(len(self.calls), 4)

    def test_missing_scoring_data_and_cache_reuse_fail(self):
        changes = [
            {"completion_tokens": 1},
            {"cached_tokens": 5},
            {"output_token_ids_logprobs": []},
            {"output_token_ids_logprobs": [[[-1.0, 65]]]},
            {"output_token_ids_logprobs": [[[math.nan, 65], [-2.0, 66]]]},
        ]
        for change in changes:
            with (
                self.subTest(change=change),
                self.assertRaises(ValueError),
                patch.object(
                    runtime, "request_json", side_effect=self.fake_api(change)
                ),
            ):
                runtime.probe_score("http://127.0.0.1:30000", "decision-model", None, 5)

    def test_metadata_does_not_publish_keys_or_auth_command(self):
        clean = runtime.sanitize_metadata(
            {
                "api_key": "secret",
                "launch_command": ["--api-key", "secret"],
                "nested": {"access_token": "secret"},
                "revision": "abc",
            }
        )
        self.assertNotIn("secret", json.dumps(clean))
        self.assertEqual(clean["revision"], "abc")


if __name__ == "__main__":
    unittest.main()
