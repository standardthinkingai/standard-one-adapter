"""Build questions and construct responses independently of the inference engine."""

import asyncio
import math
import re
import time
from collections.abc import Mapping, Sequence
from functools import partial
from numbers import Real

from .backend import AdapterError, Evaluation, ScoringBackend, ScoringResult
from .protocol import (
    SystemOneRequest,
    build_prompt,
    plan_question,
    reduce_probabilities,
    rotation_order,
)

QUESTION_TYPES = ("choice", "noul", "score")
# The only images a service that refuses remote images accepts. An engine
# given a URL or a path would otherwise fetch or read it itself.
DATA_URL = re.compile(r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}")


def validate_temperature_by_type(values: Mapping[str, float] | None) -> dict[str, float]:
    """Check a {answer type: temperature} map; returns a plain dict (empty if None)."""
    if values is None:
        return {}
    result = {}
    for key, value in values.items():
        if key not in QUESTION_TYPES:
            raise ValueError(f"unknown answer type for temperature_by_type: {key!r}")
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("temperature_by_type values must be finite and greater than zero")
        result[key] = float(value)
    return result


class SystemOneService:
    def __init__(
        self,
        backend: ScoringBackend,
        *,
        max_concurrency: int = 32,
        alias: str = "jev-latest",
        default_temperature: float = 1.0,
        temperature_by_type: Mapping[str, float] | None = None,
        prompt_wording: str = "served",
        deadline_seconds: float | None = None,
        max_queued_evaluations: int | None = None,
        remote_images: bool = True,
        engine_metadata: bool = True,
    ):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if deadline_seconds is not None and (
            isinstance(deadline_seconds, bool)
            or not isinstance(deadline_seconds, Real)
            or not math.isfinite(deadline_seconds)
            or deadline_seconds <= 0
        ):
            raise ValueError("deadline_seconds must be finite and greater than zero")
        if max_queued_evaluations is not None and (
            type(max_queued_evaluations) is not int or max_queued_evaluations < 0
        ):
            raise ValueError("max_queued_evaluations must be a nonnegative integer")
        if (
            isinstance(default_temperature, bool)
            or not isinstance(default_temperature, Real)
            or not math.isfinite(default_temperature)
            or default_temperature <= 0
        ):
            raise ValueError("default_temperature must be finite and greater than zero")
        if prompt_wording not in ("served", "native"):
            raise ValueError('prompt_wording must be "served" or "native"')
        self.backend, self.alias = backend, alias
        # Applied only when a request omits options.temperature; an explicit
        # request value always wins, and temperature_scaling=false still
        # disables scaling. Without this setting the request default (1.0) holds.
        self.default_temperature = float(default_temperature)
        # Optional per-answer-type defaults (choice / noul / score); a type not
        # listed falls back to default_temperature. Also applied only when the
        # request omits options.temperature.
        self.temperature_by_type = validate_temperature_by_type(temperature_by_type)
        # Server-wide, not per-request: which text build_prompt() renders.
        # "native" additionally requires the backend's labels to be canonical
        # A-Z (build_prompt raises otherwise) and relies on the backend to
        # prepend a matching system message where feasible (see sglang.py).
        self.prompt_wording = prompt_wording
        self.max_concurrency = max_concurrency
        self.capacity = asyncio.Semaphore(self.max_concurrency)
        # A request still running after this many seconds is stopped, its
        # engine work aborted, and answered 504, so a caller with its own
        # deadline gets an answer rather than a timeout.
        self.deadline_seconds = (
            None if deadline_seconds is None else float(deadline_seconds)
        )
        # Evaluations may wait for a slot beyond max_concurrency up to this
        # many. A request that would exceed it is refused 503 before any work;
        # one that arrives while nothing is pending is always admitted.
        self.max_queued_evaluations = max_queued_evaluations
        self.remote_images = remote_images
        # service_max_concurrency and cached_tokens in metadata. The gateway
        # passes metadata through to /v1/judgments callers, so the integrated
        # engine omits them unless asked (JEV_ENGINE_METADATA=1, benchmarks).
        self.engine_metadata = engine_metadata
        self._pending = 0

    def effective_options(self, request: SystemOneRequest):
        """Request options with the server default filled into an omitted temperature."""
        options = request.options
        if "temperature" in options.model_fields_set:
            return options
        return options.model_copy(update={"temperature": self.default_temperature})

    def question_options(self, request: SystemOneRequest, options, question_type: str):
        """Options for one question: the per-type default when the request left
        temperature unset and a default exists for this answer type."""
        if (
            "temperature" in request.options.model_fields_set
            or question_type not in self.temperature_by_type
        ):
            return options
        return options.model_copy(
            update={"temperature": self.temperature_by_type[question_type]}
        )

    async def score(self, request: SystemOneRequest) -> dict:
        if request.model not in {self.backend.model, self.alias}:
            raise AdapterError(
                "model_not_found",
                "Requested model is not configured.",
                "model",
                404,
            )
        if request.options.return_logprobs and not getattr(
            self.backend, "supports_return_logprobs", True
        ):
            raise AdapterError(
                "return_logprobs_unavailable",
                "This backend cannot return vocabulary-normalized logprobs.",
                "options.return_logprobs",
            )
        if not self.remote_images and any(
            DATA_URL.fullmatch(image) is None for image in request.images
        ):
            raise AdapterError(
                "invalid_image",
                "Images must be base64 PNG, JPEG or WebP data URLs.",
                "images",
            )
        started = time.perf_counter()
        options = self.effective_options(request)
        evaluations, orders, plans = [], [], {}
        # Validate and prepare all questions before scheduling any inference.
        for key, question in request.questions.items():
            plan = plans[key] = plan_question(question)
            labels, token_ids = self.backend.labels(len(plan.option_names))
            for rotation in range(options.permutations):
                order = rotation_order(len(labels), rotation)
                prompt = build_prompt(
                    request.state, plan, labels, order, wording=self.prompt_wording
                )
                evaluations.append(Evaluation(key, prompt, labels, token_ids))
                orders.append(order)

        self._admit(len(evaluations))
        try:
            results = await self._run(
                evaluations, request.images, options.assistant_prefix
            )
        finally:
            self._pending -= len(evaluations)
        grouped = {key: ([], []) for key in plans}
        input_tokens = cached_tokens = image_tokens = 0
        for evaluation, order, result in zip(evaluations, orders, results):
            vectors, question_orders = grouped[evaluation.question]
            vectors.append(result.logprobs)
            question_orders.append(order)
            input_tokens += result.input_tokens
            if result.cached_tokens is not None:
                cached_tokens += result.cached_tokens
            if result.image_tokens is not None:
                image_tokens += result.image_tokens
        usage = {"input_tokens": input_tokens, "output_tokens": 0}
        # Image tokens are part of input_tokens; a separate count lets a
        # caller check a bill that prices them apart from text.
        if request.images and all(result.image_tokens is not None for result in results):
            usage["input_tokens_details"] = {"image_tokens": image_tokens}
        return {
            "model": self.backend.model,
            "answers": {
                key: reduce_probabilities(
                    plan, *grouped[key], self.question_options(request, options, plan.type)
                )
                for key, plan in plans.items()
            },
            "usage": usage,
            "metadata": {
                "confidence_method": "1 - normalized_entropy",
                "temperature": (
                    options.temperature if options.temperature_scaling else 1.0
                ),
                **(
                    {"temperature_by_type": dict(self.temperature_by_type)}
                    if self.temperature_by_type
                    and "temperature" not in request.options.model_fields_set
                    and options.temperature_scaling
                    else {}
                ),
                "evaluations": len(evaluations),
                **(
                    {"service_max_concurrency": self.max_concurrency}
                    if self.engine_metadata
                    else {}
                ),
                **(
                    {"label_scheme": self.backend.label_scheme}
                    if getattr(self.backend, "label_scheme", None)
                    else {}
                ),
                **(
                    {"cached_tokens": cached_tokens}
                    if self.engine_metadata
                    and all(result.cached_tokens is not None for result in results)
                    else {}
                ),
                "adapter_elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        }

    def _admit(self, count: int) -> None:
        limit = self.max_queued_evaluations
        if (
            limit is not None
            and self._pending
            and self._pending + count > self.max_concurrency + limit
        ):
            raise AdapterError(
                "engine_overloaded",
                "The decision engine is at capacity; retry shortly.",
                status=503,
            )
        self._pending += count

    async def _run(
        self,
        evaluations: Sequence[Evaluation],
        images: list[str],
        assistant_prefix: str | None,
    ) -> list[ScoringResult]:
        if not hasattr(asyncio, "timeout"):
            # Python 3.10 (the SGLang ROCm images) has no asyncio.timeout.
            return await self._run_py310(evaluations, images, assistant_prefix)
        deadline = asyncio.timeout(self.deadline_seconds)
        try:
            async with deadline:
                return await self._evaluate_all(evaluations, images, assistant_prefix)
        except TimeoutError:
            if not deadline.expired():
                raise
            raise self._timeout_error() from None

    def _timeout_error(self) -> AdapterError:
        return AdapterError(
            "engine_timeout",
            f"The decision did not finish within {self.deadline_seconds:g} seconds.",
            status=504,
        )

    async def _run_py310(
        self,
        evaluations: Sequence[Evaluation],
        images: list[str],
        assistant_prefix: str | None,
    ) -> list[ScoringResult]:
        """Same deadline contract as _run without asyncio.timeout.

        Only the deadline itself becomes engine_timeout; a TimeoutError raised
        inside the evaluation propagates unchanged. Cancelling the caller or
        reaching the deadline cancels the evaluation, whose own cleanup aborts
        every engine request, before this returns.
        """
        task = asyncio.ensure_future(
            self._evaluate_all(evaluations, images, assistant_prefix)
        )
        try:
            done, _ = await asyncio.wait({task}, timeout=self.deadline_seconds)
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        if task not in done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise self._timeout_error()
        return task.result()

    async def _evaluate_all(
        self,
        evaluations: Sequence[Evaluation],
        images: list[str],
        assistant_prefix: str | None,
    ) -> list[ScoringResult]:
        prepare = getattr(self.backend, "prepare", None)
        if prepare is not None:
            # Every caller-input check finishes before any inference starts.
            prepared = await prepare(evaluations, images, assistant_prefix)
            if len(prepared) != len(evaluations):
                raise RuntimeError("the backend prepared a different number of evaluations")
            calls = [partial(self.backend.evaluate_prepared, item) for item in prepared]
        else:
            calls = [
                partial(
                    self.backend.evaluate,
                    evaluation.prompt,
                    images,
                    evaluation.labels,
                    evaluation.token_ids,
                    assistant_prefix,
                )
                for evaluation in evaluations
            ]

        async def evaluate(call):
            async with self.capacity:
                return await call()

        tasks = [asyncio.create_task(evaluate(call)) for call in calls]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
