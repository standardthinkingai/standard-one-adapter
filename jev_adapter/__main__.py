"""CLI for the independent adapter process."""

import argparse
import logging
import math
import os

from .labels import DEFAULT_LABEL_SCHEME, LABEL_SCHEME_ENV, LABEL_SCHEMES

logger = logging.getLogger(__name__)


def positive_temperature(text: str) -> float:
    """argparse type for --default-temperature: a finite number above zero."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def temperature_map(text: str) -> dict[str, float]:
    """argparse type for --temperature-by-type: 'choice=1.6,noul=1.2,score=1.8'."""
    result = {}
    for part in filter(None, (item.strip() for item in text.split(","))):
        key, sep, value = part.partition("=")
        key = key.strip()
        if not sep or key not in ("choice", "noul", "score") or key in result:
            raise argparse.ArgumentTypeError(
                f"expected type=T pairs for choice, noul, score; got {part!r}"
            )
        result[key] = positive_temperature(value.strip())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True, help="Engine's served model name")
    parser.add_argument("--alias", default="jev-latest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30120)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument(
        "--http-transport",
        choices=("httpx", "aiohttp"),
        default="httpx",
        help="Engine HTTP transport. aiohttp requires the performance extra.",
    )
    parser.add_argument(
        "--backend",
        choices=("logprobs", "candidate-only", "learned-head"),
        default="logprobs",
        help="Scoring backend: stock SGLang label logprobs (default) or "
        "last-hidden-state selected LM-head rows / experimental learned head on CPU.",
    )
    parser.add_argument(
        "--readout",
        help="Selected-row .npz for candidate-only or verified head.safetensors "
        "for learned-head",
    )
    parser.add_argument(
        "--tokenizer-model", help="Optional official text tokenizer repo"
    )
    parser.add_argument("--tokenizer-revision", help="Immutable tokenizer revision")
    parser.add_argument(
        "--default-temperature",
        type=positive_temperature,
        default=os.environ.get("JEV_DEFAULT_TEMPERATURE", "1.0"),
        help="Temperature applied when a request omits options.temperature "
        "(label logits are divided by T before softmax; an explicit request "
        "value wins). Env: JEV_DEFAULT_TEMPERATURE. Default 1.0.",
    )
    parser.add_argument(
        "--temperature-by-type",
        type=temperature_map,
        default=os.environ.get("JEV_TEMPERATURE_BY_TYPE", ""),
        help="Per-answer-type defaults used instead of --default-temperature when "
        "a request omits options.temperature, e.g. 'choice=1.6,noul=1.2,score=1.8'. "
        "Types not listed use --default-temperature. Env: JEV_TEMPERATURE_BY_TYPE.",
    )
    parser.add_argument(
        "--prompt-wording",
        choices=("served", "native"),
        default=os.environ.get("JEV_PROMPT_WORDING", "served"),
        help="'served' (default, unchanged behaviour): this adapter's own "
        "prompt text (Context:/.../Options: A: name: description). 'native': "
        "the jevbench-hard runner's own contract instead (run_suites_rotation.py: "
        "State:/Question:/Options: A. name: description), which the runner's "
        "own ablation measured as worth about +6 points on jevbench-hard by "
        "wording alone. Requires the engine tokenizer's first labels to be "
        "canonical single-token A-Z (true for essentially all "
        "English-vocabulary tokenizers; jev_adapter.protocol.build_prompt "
        "raises per-request otherwise). With --tokenizer-model/"
        "--tokenizer-revision also set, the adapter additionally extracts the "
        "model's default system-prompt text from a dedicated HF-template load "
        "of that same pin and prepends it explicitly, since a mistral-common "
        "native tokenizer (the default for Ministral) never auto-injects it. "
        "See --native-system-prompt to disable this. Env: JEV_PROMPT_WORDING.",
    )
    parser.add_argument(
        "--native-system-prompt",
        choices=("auto", "none"),
        default=os.environ.get("JEV_NATIVE_SYSTEM_PROMPT", "auto"),
        help="Only takes effect with --prompt-wording native; ignored "
        "otherwise. 'auto' (default, unchanged behaviour): with "
        "--tokenizer-model/--tokenizer-revision also set, extract the "
        "model's default system-prompt text from the pinned tokenizer's chat "
        "template (see --prompt-wording) and prepend it as an explicit "
        "system message; warns at startup if no tokenizer pin is given or "
        "the template defines no default system prompt. 'none': render the "
        "native wording with no system message at all -- skips the "
        "extraction entirely and never warns about a missing prompt (useful "
        "when the extra system-prompt tokens are not wanted, e.g. the "
        "Ministral default is 531 tokens, about 80%% of a typical request). "
        "Env: JEV_NATIVE_SYSTEM_PROMPT.",
    )
    parser.add_argument(
        "--label-scheme",
        choices=LABEL_SCHEMES,
        default=os.environ.get(LABEL_SCHEME_ENV) or DEFAULT_LABEL_SCHEME,
        help="Answer labels for questions with more than 26 options. 'upper' "
        "(default): A-Z, then AA, AB, ... (uppercase only; labels that are not "
        "single tokens are skipped). 'served': the original A-Z, a-z, 0-9, "
        "then AA, ... order (pre-v2.2 serving; the 3B learned head). Both give A-Z to "
        "questions with at most 26 options. Reported in /v1/models and in "
        "response metadata. Env: JEV_LABEL_SCHEME.",
    )
    # argparse runs the type converter on the string default too, so a bad
    # JEV_DEFAULT_TEMPERATURE fails at startup with the same message.
    args = parser.parse_args()
    if args.label_scheme not in LABEL_SCHEMES:
        parser.error(
            f"JEV_LABEL_SCHEME must be one of {', '.join(LABEL_SCHEMES)}; "
            f"got {args.label_scheme!r}"
        )
    if args.backend == "learned-head" and args.label_scheme != "served":
        parser.error("learned-head requires --label-scheme served")
    if args.timeout <= 0 or args.max_concurrency < 1:
        parser.error("timeout and max-concurrency must be positive")
    if bool(args.tokenizer_model) != bool(args.tokenizer_revision):
        parser.error("tokenizer-model and tokenizer-revision must be supplied together")
    if args.backend in ("candidate-only", "learned-head"):
        if not args.readout or not args.tokenizer_model:
            parser.error(
                f"{args.backend} requires --readout, --tokenizer-model, "
                "and --tokenizer-revision"
            )
    elif args.readout:
        parser.error("--readout requires --backend candidate-only or learned-head")

    import uvicorn

    from .server import create_app
    from .sglang import SGLangBackend

    native_tokenizer = None
    native_system_prompt = None
    if args.tokenizer_model:
        from .native_tokenizer import NativeTokenizer

        native_tokenizer = NativeTokenizer.from_pretrained(
            args.tokenizer_model, args.tokenizer_revision
        )
        if args.prompt_wording == "native" and args.native_system_prompt == "auto":
            native_system_prompt = NativeTokenizer.native_default_system_prompt(
                args.tokenizer_model, args.tokenizer_revision
            )
    if (
        args.prompt_wording == "native"
        and args.native_system_prompt == "auto"
        and native_system_prompt is None
    ):
        logger.warning(
            "prompt-wording=native without a default system prompt (pass "
            "--tokenizer-model/--tokenizer-revision to extract and prepend the "
            "model's own default system message, or the model's template "
            "defines none); serving the native wording text without it."
        )
    logger.info(
        "prompt_wording=%s native_system_prompt=%s (%s) label_scheme=%s",
        args.prompt_wording,
        args.native_system_prompt,
        "system message attached"
        if native_system_prompt
        else "no system message attached",
        args.label_scheme,
    )
    backend_kwargs = {
        "api_key": os.environ.get("SGLANG_API_KEY"),
        "timeout": args.timeout,
        "max_connections": args.max_concurrency,
        "http_transport": args.http_transport,
        "native_tokenizer": native_tokenizer,
        "prompt_wording": args.prompt_wording,
        "native_system_prompt": native_system_prompt,
        "label_scheme": args.label_scheme,
    }
    if args.backend == "candidate-only":
        from .candidate_backend import CandidateReadoutBackend

        backend = CandidateReadoutBackend(
            args.engine_url,
            args.model,
            readout_path=args.readout,
            tokenizer_model=args.tokenizer_model,
            tokenizer_revision=args.tokenizer_revision,
            **backend_kwargs,
        )
    elif args.backend == "learned-head":
        from .learned_head import LearnedHeadBackend

        backend = LearnedHeadBackend(
            args.engine_url,
            args.model,
            readout_path=args.readout,
            tokenizer_model=args.tokenizer_model,
            tokenizer_revision=args.tokenizer_revision,
            **backend_kwargs,
        )
    else:
        backend = SGLangBackend(
            args.engine_url,
            args.model,
            **backend_kwargs,
        )
    app = create_app(
        backend,
        api_key=os.environ.get("JEV_API_KEY"),
        max_concurrency=args.max_concurrency,
        alias=args.alias,
        default_temperature=args.default_temperature,
        temperature_by_type=args.temperature_by_type,
        prompt_wording=args.prompt_wording,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
