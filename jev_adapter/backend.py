"""The small contract between response formatting and inference transport."""

from dataclasses import dataclass
from typing import Protocol


class AdapterError(Exception):
    def __init__(self, code: str, message: str, field: str = "", status: int = 422):
        super().__init__(message)
        self.code, self.field, self.status = code, field, status


@dataclass(frozen=True)
class ScoringResult:
    # The default backend returns vocabulary logprobs. Candidate-only returns
    # selected raw logits; the service uses only their relative differences
    # unless options.return_logprobs is requested (which candidate-only rejects).
    logprobs: tuple[float, ...]
    input_tokens: int
    cached_tokens: int | None = None
    # Prompt tokens that encode images, included in input_tokens. None when
    # the backend cannot count them exactly; 0 for a text-only evaluation.
    image_tokens: int | None = None


@dataclass(frozen=True)
class Evaluation:
    """One question rendered under one option rotation."""

    question: str
    prompt: str
    labels: tuple[str, ...]
    token_ids: tuple[int, ...]


class ScoringBackend(Protocol):
    """Scores one prompt at a time.

    A backend may also define ``prepare(evaluations, images, assistant_prefix)``
    and ``evaluate_prepared(prepared)``. The service then calls ``prepare`` once
    per request, before any inference: it must finish every caller-input check
    (tokenization, context length, images) for all evaluations, raising
    AdapterError, and return one prepared value per evaluation.
    """

    model: str

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    def labels(self, count: int) -> tuple[tuple[str, ...], tuple[int, ...]]: ...

    async def evaluate(
        self,
        prompt: str,
        images: list[str],
        labels: tuple[str, ...],
        token_ids: tuple[int, ...],
        assistant_prefix: str | None,
    ) -> ScoringResult: ...
