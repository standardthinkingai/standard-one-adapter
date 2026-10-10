"""CLI wiring tests for --prompt-wording (argparse choices/env default, and
plumbing into SGLangBackend + create_app). uvicorn.run is monkeypatched so
main() never actually binds a socket; SGLangBackend/create_app/NativeTokenizer
are monkeypatched to capture their constructor arguments instead of talking to
a real engine or loading real tokenizer files (transformers is an optional,
not-installed-here dependency for this default venv)."""

import sys

import pytest

import jev_adapter.__main__ as main_module
import jev_adapter.candidate_backend as candidate_backend_module
import jev_adapter.learned_head as learned_head_module
import jev_adapter.native_tokenizer as native_tokenizer_module
import jev_adapter.server as server_module
import jev_adapter.sglang as sglang_module


class Capture:
    def __init__(self):
        self.backend_kwargs = None
        self.app_kwargs = None

    def fake_backend(self, *args, **kwargs):
        self.backend_kwargs = kwargs
        return object()

    def fake_create_app(self, backend, **kwargs):
        self.app_kwargs = kwargs
        return object()


@pytest.fixture
def capture(monkeypatch):
    capture = Capture()
    monkeypatch.setattr(sglang_module, "SGLangBackend", capture.fake_backend)
    monkeypatch.setattr(server_module, "create_app", capture.fake_create_app)
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: None)
    return capture


def run_main(monkeypatch, argv, env=None):
    monkeypatch.setattr(sys, "argv", ["jev-adapter", *argv])
    for key in (
        "JEV_PROMPT_WORDING",
        "JEV_DEFAULT_TEMPERATURE",
        "JEV_NATIVE_SYSTEM_PROMPT",
        "JEV_LABEL_SCHEME",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    main_module.main()


def test_prompt_wording_defaults_to_served(monkeypatch, capture):
    run_main(monkeypatch, ["--model", "decision-model"])
    assert capture.backend_kwargs["prompt_wording"] == "served"
    assert capture.backend_kwargs["native_system_prompt"] is None
    assert capture.app_kwargs["prompt_wording"] == "served"
    assert capture.backend_kwargs["http_transport"] == "httpx"


@pytest.mark.parametrize("backend", ["logprobs", "candidate-only", "learned-head"])
def test_http_transport_is_forwarded_for_all_backends(monkeypatch, capture, backend):
    monkeypatch.setattr(
        candidate_backend_module, "CandidateReadoutBackend", capture.fake_backend
    )
    monkeypatch.setattr(
        learned_head_module, "LearnedHeadBackend", capture.fake_backend
    )
    monkeypatch.setattr(
        native_tokenizer_module.NativeTokenizer,
        "from_pretrained", lambda model, revision: object(),
    )
    options = ["--model", "decision-model", "--backend", backend,
               "--http-transport", "aiohttp", "--max-concurrency", "128"]
    if backend != "logprobs":
        options += ["--readout", "/tmp/readout.npz", "--tokenizer-model", "org/model",
                    "--tokenizer-revision", "a" * 40]
    if backend == "learned-head":
        options += ["--label-scheme", "served"]
    run_main(monkeypatch, options)
    assert capture.backend_kwargs["http_transport"] == "aiohttp"
    assert capture.backend_kwargs["max_connections"] == 128
    assert capture.app_kwargs["max_concurrency"] == 128


def test_candidate_backend_selects_readout_and_native_tokenizer(monkeypatch, capture):
    monkeypatch.setattr(
        candidate_backend_module,
        "CandidateReadoutBackend",
        capture.fake_backend,
    )
    monkeypatch.setattr(
        native_tokenizer_module.NativeTokenizer,
        "from_pretrained",
        lambda model, revision: object(),
    )
    run_main(
        monkeypatch,
        [
            "--model", "decision-model",
            "--backend", "candidate-only",
            "--readout", "/tmp/readout.npz",
            "--tokenizer-model", "org/ministral",
            "--tokenizer-revision", "a" * 40,
        ],
    )
    assert capture.backend_kwargs["readout_path"] == "/tmp/readout.npz"
    assert capture.backend_kwargs["tokenizer_model"] == "org/ministral"
    assert capture.backend_kwargs["tokenizer_revision"] == "a" * 40
    assert capture.backend_kwargs["native_tokenizer"] is not None


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "m", "--backend", "candidate-only"],
        ["--model", "m", "--readout", "/tmp/readout.npz"],
    ],
)
def test_candidate_backend_requires_readout_and_pin(monkeypatch, capture, args):
    monkeypatch.setattr(sys, "argv", ["jev-adapter", *args])
    with pytest.raises(SystemExit):
        main_module.main()


def test_prompt_wording_flag_selects_native(monkeypatch, capture):
    run_main(monkeypatch, ["--model", "decision-model", "--prompt-wording", "native"])
    assert capture.backend_kwargs["prompt_wording"] == "native"
    assert capture.app_kwargs["prompt_wording"] == "native"


def test_prompt_wording_env_var_sets_the_default(monkeypatch, capture):
    run_main(
        monkeypatch,
        ["--model", "decision-model"],
        env={"JEV_PROMPT_WORDING": "native"},
    )
    assert capture.backend_kwargs["prompt_wording"] == "native"
    assert capture.app_kwargs["prompt_wording"] == "native"


def test_explicit_flag_overrides_env_var(monkeypatch, capture):
    run_main(
        monkeypatch,
        ["--model", "decision-model", "--prompt-wording", "served"],
        env={"JEV_PROMPT_WORDING": "native"},
    )
    assert capture.backend_kwargs["prompt_wording"] == "served"


def test_invalid_prompt_wording_choice_rejected(monkeypatch, capture):
    monkeypatch.setattr(sys, "argv", ["jev-adapter", "--model", "m", "--prompt-wording", "bogus"])
    with pytest.raises(SystemExit):
        main_module.main()


def test_native_wording_without_tokenizer_model_serves_text_only_and_warns(
    monkeypatch, capture, caplog
):
    with caplog.at_level("WARNING"):
        run_main(monkeypatch, ["--model", "decision-model", "--prompt-wording", "native"])
    assert capture.backend_kwargs["native_system_prompt"] is None
    assert any("native" in record.message for record in caplog.records)


def test_native_wording_with_tokenizer_model_extracts_the_system_prompt(
    monkeypatch, capture
):
    calls = []

    class FakeNativeTokenizer:
        @classmethod
        def from_pretrained(cls, model, revision):
            calls.append(("from_pretrained", model, revision))
            return object()

        @classmethod
        def native_default_system_prompt(cls, model, revision):
            calls.append(("native_default_system_prompt", model, revision))
            return "Default system text."

    monkeypatch.setattr(native_tokenizer_module, "NativeTokenizer", FakeNativeTokenizer)
    run_main(
        monkeypatch,
        [
            "--model",
            "decision-model",
            "--prompt-wording",
            "native",
            "--tokenizer-model",
            "mistralai/Ministral-3-8B-Instruct-2512-BF16",
            "--tokenizer-revision",
            "f" * 40,
        ],
    )
    assert calls == [
        ("from_pretrained", "mistralai/Ministral-3-8B-Instruct-2512-BF16", "f" * 40),
        (
            "native_default_system_prompt",
            "mistralai/Ministral-3-8B-Instruct-2512-BF16",
            "f" * 40,
        ),
    ]
    assert capture.backend_kwargs["native_system_prompt"] == "Default system text."


def test_served_wording_with_tokenizer_model_never_extracts_a_system_prompt(
    monkeypatch, capture
):
    calls = []

    class FakeNativeTokenizer:
        @classmethod
        def from_pretrained(cls, model, revision):
            calls.append("from_pretrained")
            return object()

        @classmethod
        def native_default_system_prompt(cls, model, revision):
            calls.append("native_default_system_prompt")
            return "should not be reached"

    monkeypatch.setattr(native_tokenizer_module, "NativeTokenizer", FakeNativeTokenizer)
    run_main(
        monkeypatch,
        [
            "--model",
            "decision-model",
            "--tokenizer-model",
            "org/model",
            "--tokenizer-revision",
            "a" * 40,
        ],
    )
    assert calls == ["from_pretrained"]
    assert capture.backend_kwargs["prompt_wording"] == "served"
    assert capture.backend_kwargs["native_system_prompt"] is None


def test_native_system_prompt_defaults_to_auto(monkeypatch, capture):
    calls = []

    class FakeNativeTokenizer:
        @classmethod
        def from_pretrained(cls, model, revision):
            return object()

        @classmethod
        def native_default_system_prompt(cls, model, revision):
            calls.append((model, revision))
            return "Default system text."

    monkeypatch.setattr(native_tokenizer_module, "NativeTokenizer", FakeNativeTokenizer)
    run_main(
        monkeypatch,
        [
            "--model",
            "decision-model",
            "--prompt-wording",
            "native",
            "--tokenizer-model",
            "org/model",
            "--tokenizer-revision",
            "a" * 40,
        ],
    )
    assert calls == [("org/model", "a" * 40)]
    assert capture.backend_kwargs["native_system_prompt"] == "Default system text."


def test_native_system_prompt_none_skips_extraction_and_warning(
    monkeypatch, capture, caplog
):
    calls = []

    class FakeNativeTokenizer:
        @classmethod
        def from_pretrained(cls, model, revision):
            return object()

        @classmethod
        def native_default_system_prompt(cls, model, revision):
            calls.append((model, revision))
            return "should not be reached"

    monkeypatch.setattr(native_tokenizer_module, "NativeTokenizer", FakeNativeTokenizer)
    with caplog.at_level("WARNING"):
        run_main(
            monkeypatch,
            [
                "--model",
                "decision-model",
                "--prompt-wording",
                "native",
                "--tokenizer-model",
                "org/model",
                "--tokenizer-revision",
                "a" * 40,
                "--native-system-prompt",
                "none",
            ],
        )
    assert calls == []
    assert capture.backend_kwargs["native_system_prompt"] is None
    assert not any("native" in record.message for record in caplog.records)


def test_native_system_prompt_none_without_tokenizer_never_warns(
    monkeypatch, capture, caplog
):
    with caplog.at_level("WARNING"):
        run_main(
            monkeypatch,
            [
                "--model",
                "decision-model",
                "--prompt-wording",
                "native",
                "--native-system-prompt",
                "none",
            ],
        )
    assert capture.backend_kwargs["native_system_prompt"] is None
    assert caplog.records == []


def test_native_system_prompt_env_var_sets_the_default(monkeypatch, capture, caplog):
    with caplog.at_level("WARNING"):
        run_main(
            monkeypatch,
            ["--model", "decision-model", "--prompt-wording", "native"],
            env={"JEV_NATIVE_SYSTEM_PROMPT": "none"},
        )
    assert capture.backend_kwargs["native_system_prompt"] is None
    assert caplog.records == []


def test_native_system_prompt_explicit_flag_overrides_env_var(monkeypatch, capture):
    calls = []

    class FakeNativeTokenizer:
        @classmethod
        def from_pretrained(cls, model, revision):
            return object()

        @classmethod
        def native_default_system_prompt(cls, model, revision):
            calls.append((model, revision))
            return "Default system text."

    monkeypatch.setattr(native_tokenizer_module, "NativeTokenizer", FakeNativeTokenizer)
    run_main(
        monkeypatch,
        [
            "--model",
            "decision-model",
            "--prompt-wording",
            "native",
            "--tokenizer-model",
            "org/model",
            "--tokenizer-revision",
            "a" * 40,
            "--native-system-prompt",
            "auto",
        ],
        env={"JEV_NATIVE_SYSTEM_PROMPT": "none"},
    )
    assert calls == [("org/model", "a" * 40)]
    assert capture.backend_kwargs["native_system_prompt"] == "Default system text."


def test_native_system_prompt_none_has_no_effect_on_served_wording(
    monkeypatch, capture
):
    calls = []

    class FakeNativeTokenizer:
        @classmethod
        def from_pretrained(cls, model, revision):
            return object()

        @classmethod
        def native_default_system_prompt(cls, model, revision):
            calls.append((model, revision))
            return "should not be reached"

    monkeypatch.setattr(native_tokenizer_module, "NativeTokenizer", FakeNativeTokenizer)
    run_main(
        monkeypatch,
        [
            "--model",
            "decision-model",
            "--tokenizer-model",
            "org/model",
            "--tokenizer-revision",
            "a" * 40,
            "--native-system-prompt",
            "auto",
        ],
    )
    assert calls == []
    assert capture.backend_kwargs["prompt_wording"] == "served"
    assert capture.backend_kwargs["native_system_prompt"] is None


def test_invalid_native_system_prompt_choice_rejected(monkeypatch, capture):
    monkeypatch.setattr(
        sys,
        "argv",
        ["jev-adapter", "--model", "m", "--native-system-prompt", "bogus"],
    )
    with pytest.raises(SystemExit):
        main_module.main()


def test_native_system_prompt_startup_log_line(monkeypatch, capture, caplog):
    with caplog.at_level("INFO"):
        run_main(
            monkeypatch,
            [
                "--model",
                "decision-model",
                "--prompt-wording",
                "native",
                "--native-system-prompt",
                "none",
            ],
        )
    info_records = [r for r in caplog.records if r.levelname == "INFO"]
    assert len(info_records) == 1
    message = info_records[0].getMessage()
    assert "prompt_wording=native" in message
    assert "native_system_prompt=none" in message
    assert "label_scheme=upper" in message


def test_label_scheme_defaults_to_upper(monkeypatch, capture):
    run_main(monkeypatch, ["--model", "decision-model"])
    assert capture.backend_kwargs["label_scheme"] == "upper"


def test_label_scheme_flag_and_env_var(monkeypatch, capture):
    run_main(monkeypatch, ["--model", "m"], env={"JEV_LABEL_SCHEME": "served"})
    assert capture.backend_kwargs["label_scheme"] == "served"
    run_main(
        monkeypatch,
        ["--model", "m", "--label-scheme", "upper"],
        env={"JEV_LABEL_SCHEME": "served"},
    )
    assert capture.backend_kwargs["label_scheme"] == "upper"


@pytest.mark.parametrize(
    "argv,env",
    [
        (["--label-scheme", "lower"], {}),
        ([], {"JEV_LABEL_SCHEME": "lower"}),
    ],
)
def test_invalid_label_scheme_rejected(monkeypatch, capture, argv, env):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, ["--model", "m", *argv], env=env)
    assert capture.backend_kwargs is None


def test_learned_head_requires_the_served_scheme(monkeypatch, capture):
    monkeypatch.setattr(learned_head_module, "LearnedHeadBackend", capture.fake_backend)
    with pytest.raises(SystemExit):
        run_main(
            monkeypatch,
            [
                "--model", "m", "--backend", "learned-head",
                "--readout", "/tmp/head.safetensors",
                "--tokenizer-model", "org/model", "--tokenizer-revision", "a" * 40,
            ],
        )
    assert capture.backend_kwargs is None
