import contextlib
import copy
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "mi325x_runtime", Path(__file__).with_name("runtime.py")
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.manifest = runtime.load_manifest()
        self.key = "qwen35-4b-bf16"
        self.lock = {
            "schema_version": 1,
            "image_tag": self.manifest["engine"]["image_tag"],
            "image_ref": "lmsysorg/sglang@sha256:" + "a" * 64,
            "image_id": "sha256:" + "b" * 64,
            "os": "linux",
            "architecture": "amd64",
            "verification": "docker_pull_and_digest_inspect",
        }
        self.info = {
            "hip": "7.2.53210",
            "cuda": None,
            "sglang_version": "0.5.20",
            "gpus": [
                {
                    "name": "AMD Instinct MI325X",
                    "gcn_arch": "gfx942:sramecc+:xnack-",
                    "memory_bytes": 256 * 1024**3,
                }
            ],
        }

    def test_same_four_official_bf16_revisions_as_h200(self):
        reference = runtime.common.load_manifest()
        self.assertEqual(set(self.manifest["models"]), set(reference["default_matrix"]))
        for key, model in self.manifest["models"].items():
            for field in (
                "repo_id",
                "revision",
                "dtype",
                "quantization",
                "tokenization",
            ):
                self.assertEqual(model[field], reference["models"][key][field])
            self.assertEqual(model["hardware_validation"], "pending")
        self.assertIsNone(self.manifest["engine"]["image_digest"])

    def test_command_has_rocm_single_gpu_bf16_and_cache_policy(self):
        command, env = runtime.launch_spec(
            self.manifest, self.key, "/engine/bin/python", "3", 30000, {}
        )
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
        self.assertEqual(command[command.index("--tp-size") + 1], "1")
        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
        self.assertIn("--disable-radix-cache", command)
        self.assertEqual(command[command.index("--chunked-prefill-size") + 1], "-1")
        self.assertFalse(any("lora" in x or "speculative" in x for x in command))

    def test_preserves_existing_allocation_and_environment(self):
        inherited = {
            "ROCR_VISIBLE_DEVICES": "GPU-abcdef",
            "PATH": "/tools",
            "HF_TOKEN": "secret",
        }
        _, env = runtime.launch_spec(
            self.manifest, self.key, "/engine/bin/python", "0", 30000, inherited
        )
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "GPU-abcdef")
        self.assertEqual(dict(inherited, **env)["HF_TOKEN"], "secret")
        self.assertEqual(inherited["PATH"], "/tools")

    def test_rejects_multiple_or_conflicting_gpu_masks(self):
        for gpu, inherited in [
            ("0,1", {}),
            ("-1", {}),
            ("all", {}),
            ("0", {"ROCR_VISIBLE_DEVICES": "0,1"}),
            ("3", {"ROCR_VISIBLE_DEVICES": "3"}),
            ("0", {"HIP_VISIBLE_DEVICES": "0"}),
            ("0", {"ROCR_VISIBLE_DEVICES": "3", "HIP_VISIBLE_DEVICES": "3"}),
            ("0", {"CUDA_VISIBLE_DEVICES": "0"}),
            ("0", {"GPU_DEVICE_ORDINAL": "1"}),
        ]:
            with (
                self.subTest(gpu=gpu, inherited=inherited),
                self.assertRaises(ValueError),
            ):
                runtime.visibility_overrides(gpu, inherited)

    def test_rejects_modified_precision_and_extra_lora_args(self):
        for field, value in [
            ("quantization", "fp8"),
            ("dtype", "float16"),
            ("extra_engine_args", ["--enable-lora"]),
        ]:
            manifest = copy.deepcopy(self.manifest)
            manifest["models"][self.key][field] = value
            with self.assertRaises(ValueError):
                runtime.launch_spec(manifest, self.key, "python", "0", 30000, {})

    def test_valid_amd_runtime(self):
        self.assertEqual(runtime.validate_runtime(self.info, self.manifest), self.info)

    def test_rejects_cuda_cpu_other_hardware_and_multiple_devices(self):
        for changes in [
            {"hip": None, "cuda": "13.0"},
            {"hip": None},
            {"hip": "6.4.0"},
            {"sglang_version": "0.5.17"},
            {"gpus": []},
            {"gpus": self.info["gpus"] * 2},
            {"gpus": [dict(self.info["gpus"][0], name="MI300X")]},
            {"gpus": [dict(self.info["gpus"][0], gcn_arch="gfx950")]},
            {"gpus": [dict(self.info["gpus"][0], memory_bytes=128 * 1024**3)]},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runtime.validate_runtime(dict(self.info, **changes), self.manifest)

    def test_inspector_uses_hip_not_h200_or_nvidia_smi(self):
        result = Mock(stdout=json.dumps(self.info))
        with (
            patch.object(runtime.subprocess, "run", return_value=result) as run,
            patch.object(
                runtime.common,
                "inspect_runtime",
                side_effect=AssertionError("H200 inspector called"),
            ),
        ):
            runtime.inspect_runtime("python", {}, self.manifest)
        self.assertEqual(run.call_count, 1)
        self.assertIn("torch.version.hip", run.call_args.args[0][2])
        self.assertNotIn("nvidia-smi", str(run.call_args))

    def test_digest_lock_and_running_container_marker(self):
        runtime.validate_image_lock(self.lock, self.manifest)
        with patch.object(runtime.Path, "exists", return_value=True):
            runtime.verify_container(
                self.lock,
                self.manifest,
                {"JEV_CONTAINER_IMAGE": self.lock["image_ref"]},
            )
            with self.assertRaises(ValueError):
                runtime.verify_container(self.lock, self.manifest, {})
        for field, value in [
            ("image_ref", self.lock["image_tag"]),
            ("image_tag", "latest"),
            ("os", "windows"),
            ("architecture", "arm64"),
            ("verification", "unchecked"),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime.validate_image_lock(
                    dict(self.lock, **{field: value}), self.manifest
                )

    def test_lock_resolves_and_checks_local_digest_without_gpu(self):
        inspect = {
            "RepoDigests": [self.lock["image_ref"]],
            "Id": self.lock["image_id"],
            "Os": "linux",
            "Architecture": "amd64",
        }
        with (
            patch.object(runtime.subprocess, "run") as run,
            patch.object(
                runtime.subprocess, "check_output", return_value=json.dumps([inspect])
            ) as check,
        ):
            lock = runtime.lock_image(self.manifest)
        self.assertEqual(lock["image_ref"], self.lock["image_ref"])
        self.assertEqual(run.call_args.args[0][:2], ["docker", "pull"])
        self.assertEqual(check.call_count, 2)

    def test_baseline_disallows_lora_speculative_mtp_and_chunks(self):
        server = {
            "enable_lora": None,
            "lora_paths": None,
            "speculative_algorithm": None,
            "chunked_prefill_size": -1,
        }
        runtime.validate_baseline(server)
        for field, value in [
            ("enable_lora", True),
            ("lora_paths", ["finetuned"]),
            ("speculative_algorithm", "EAGLE"),
            ("speculative_draft_model_path", "mtp"),
            ("chunked_prefill_size", 4096),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime.validate_baseline(dict(server, **{field: value}))
        with self.assertRaises(ValueError):
            runtime.validate_baseline({})

    def test_vision_scope_delegates_to_public_image_probe(self):
        server = {
            "enable_lora": None,
            "lora_paths": None,
            "speculative_algorithm": None,
            "chunked_prefill_size": -1,
        }
        with (
            patch.object(runtime.common, "request_json", return_value=server),
            patch.object(
                runtime.common,
                "probe",
                return_value={"probe_scope": ["text", "vision"]},
            ) as probe,
        ):
            result = runtime.probe(
                "http://localhost:30000", self.manifest, self.key, 60, True
            )
        self.assertEqual(result["probe_scope"], ["text", "vision"])
        self.assertTrue(probe.call_args.kwargs["probe_vision"])
        self.assertEqual(
            self.manifest["models"][self.key]["vision_validation"],
            "pending_runtime_probe",
        )

    def test_native_ministral_rejects_vision_before_launch(self):
        key = "ministral3-3b-bf16"
        self.assertEqual(
            self.manifest["models"][key]["input_modes_to_validate"], ["text"]
        )
        with (
            contextlib.redirect_stderr(io.StringIO()),
            patch.object(runtime, "launch") as launch,
        ):
            with self.assertRaises(SystemExit):
                runtime.main(["launch", key, "--probe-vision", "--execute"])
            launch.assert_not_called()

    def test_preview_performs_no_subprocess_http_or_gpu_work(self):
        with (
            patch.dict(runtime.os.environ, {}, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(
                runtime.subprocess,
                "run",
                side_effect=AssertionError("subprocess on preview"),
            ),
            patch.object(
                runtime.subprocess,
                "Popen",
                side_effect=AssertionError("launch on preview"),
            ),
            patch.object(
                runtime.common,
                "request_json",
                side_effect=AssertionError("HTTP on preview"),
            ),
            patch.object(
                runtime, "inspect_runtime", side_effect=AssertionError("GPU on preview")
            ),
        ):
            for argv in [
                ["launch", self.key],
                ["probe", self.key],
                ["host-preflight"],
                ["lock-image"],
                ["list"],
            ]:
                self.assertEqual(runtime.main(argv), 0)

    def test_execute_requires_lock_before_gpu(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(runtime.os.environ, {}, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            patch.object(runtime, "inspect_runtime") as inspect,
        ):
            with self.assertRaises(SystemExit):
                runtime.main(
                    [
                        "launch",
                        self.key,
                        "--execute",
                        "--output",
                        str(Path(directory) / "run"),
                    ]
                )
            inspect.assert_not_called()

    def test_stops_only_owned_process_group(self):
        process = Mock(pid=1234)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("engine", 15), 0]
        with patch.object(runtime.os, "killpg") as kill:
            runtime.stop_owned(process)
        self.assertEqual([call.args[0] for call in kill.call_args_list], [1234, 1234])

    def test_launch_failure_cleans_owned_group_and_keeps_pinned_metadata(self):
        process = Mock(pid=3210, returncode=1)
        process.poll.return_value = 1
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                model=self.key,
                port=30000,
                output=Path(directory) / "engine",
                engine_python="python",
                probe_vision=False,
                timeout=30,
            )
            with (
                patch.object(runtime, "verify_container"),
                patch.object(runtime.socket, "socket"),
                patch.object(runtime, "inspect_runtime", return_value=self.info),
                patch.object(runtime.subprocess, "Popen", return_value=process),
                patch.object(runtime, "stop_owned") as stop,
                self.assertRaises(RuntimeError),
            ):
                runtime.launch(
                    args,
                    self.manifest,
                    ["python", "-m", "sglang.launch_server"],
                    {},
                    self.lock,
                )
            stop.assert_called_once_with(process)
            self.assertFalse((args.output / "ready.json").exists())
            record = json.loads((args.output / "launch.json").read_text())
            self.assertEqual(record["engine"]["revision"], self.lock["image_ref"])
            self.assertEqual(record["engine"]["revision_kind"], "oci_image_digest")


if __name__ == "__main__":
    unittest.main()
