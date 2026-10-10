"""Validated requests and pure scoring helpers for the System One compatible API.

This is next-token classification with an existing language model, not the
proprietary Jev model. Confidence follows the llama.cpp fork's formula,
``1 - entropy / log(number of options)``; it is not a calibration guarantee.
"""

import json
import math
import string
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

StructuredText = StrictStr | dict[StrictStr, JsonValue] | list[JsonValue]
Description = StructuredText | None
QuestionType = Literal["choice", "noul", "score"]
PromptWording = Literal["served", "native"]

# The jevbench-hard runner's own contract (run_suites_rotation.py: PREFIX,
# format_prompt, CONTRACT), copied verbatim. Rendering ``build_prompt`` with
# wording="native" reproduces this text exactly; the runner measured this
# wording alone as worth about +6 points on jevbench-hard over the served
# wording below (native 61.3 vs served 55.0 Hard accuracy, same tokenizer
# family). See run_suites_ablate.py for the full effect decomposition.
NATIVE_PROMPT_PREFIX = (
    "Read the state and question. Choose the single best option using the "
    "supplied information. Respond with exactly one option letter and no "
    "other text."
)
NATIVE_PROMPT_CONTRACT = "jev-public-hard-native-letter-v1"
_NATIVE_DISPLAY = string.ascii_uppercase


class _SystemOneModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_finite_json(self):
        # JsonValue permits floats; JSON descriptions must not contain NaN/Inf.
        try:
            json.dumps(self.model_dump(), allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("all structured values must be finite JSON") from exc
        return self


class ChoiceQuestion(_SystemOneModel):
    type: Literal["choice"]
    instructions: StructuredText
    criteria: dict[StrictStr, Description] = Field(min_length=2, max_length=255)

    @field_validator("criteria")
    @classmethod
    def validate_names(cls, criteria):
        if any(not name.strip() for name in criteria):
            raise ValueError("option names must not be empty")
        return criteria


class NoulQuestion(_SystemOneModel):
    type: Literal["noul"]
    instructions: StructuredText
    criteria: dict[StrictStr, Description] | None = None

    @field_validator("criteria")
    @classmethod
    def validate_boolean_criteria(cls, criteria):
        # An empty map means "no descriptions", same as omitting the field.
        if criteria == {}:
            return None
        if criteria is not None and set(criteria) != {"true", "false"}:
            raise ValueError('noul criteria must contain exactly "true" and "false"')
        return criteria


class ScoreQuestion(_SystemOneModel):
    type: Literal["score"]
    instructions: StructuredText
    criteria: list[Description] = Field(min_length=2, max_length=10)


SystemOneQuestion = Annotated[
    ChoiceQuestion | NoulQuestion | ScoreQuestion, Field(discriminator="type")
]


class SystemOneOptions(_SystemOneModel):
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False, strict=True)
    temperature_scaling: StrictBool = True
    permutations: int = Field(default=1, ge=1, le=16, strict=True)
    return_logprobs: StrictBool = False
    assistant_prefix: StrictStr | None = None


class SystemOneRequest(_SystemOneModel):
    model: StrictStr = Field(min_length=1)
    state: StructuredText
    questions: dict[StrictStr, SystemOneQuestion] = Field(min_length=1, max_length=256)
    options: SystemOneOptions = Field(default_factory=SystemOneOptions)
    images: list[StrictStr] = Field(default_factory=list)

    @field_validator("model")
    @classmethod
    def validate_model(cls, model):
        if not model.strip():
            raise ValueError("model must not be empty")
        return model

    @field_validator("questions")
    @classmethod
    def validate_question_names(cls, questions):
        if any(not name.strip() for name in questions):
            raise ValueError("question names must not be empty")
        return questions

    @field_validator("images")
    @classmethod
    def validate_images(cls, images):
        for image in images:
            if image.startswith("data:image/"):
                if "," not in image or not image.split(",", 1)[1]:
                    raise ValueError("image data URLs must have a nonempty payload")
            else:
                parsed = urlsplit(image)
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    raise ValueError("images must be image data URLs or HTTP(S) URLs")
        return images


@dataclass(frozen=True)
class QuestionPlan:
    type: QuestionType
    instructions: StructuredText
    option_names: tuple[str, ...]
    descriptions: tuple[Description, ...]


def plan_question(question: SystemOneQuestion) -> QuestionPlan:
    """Keep caller option order; boolean options always map to true, false."""
    if isinstance(question, ChoiceQuestion):
        names = tuple(question.criteria)
        descriptions = tuple(question.criteria.values())
    elif isinstance(question, NoulQuestion):
        names = ("true", "false")
        criteria = question.criteria or {}
        descriptions = tuple(criteria.get(name) for name in names)
    elif isinstance(question, ScoreQuestion):
        names = tuple(str(index) for index in range(len(question.criteria)))
        descriptions = tuple(question.criteria)
    else:
        raise TypeError("question must be a validated System One question")
    return QuestionPlan(question.type, question.instructions, names, descriptions)


def render_json_text(value: StructuredText) -> str:
    """Render structured input deterministically without changing plain text."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def render_native_state_text(value: StructuredText) -> str:
    """Render ``state`` exactly as the jevbench-hard runner's ``format_prompt``
    does: plain strings verbatim, everything else as a sorted, indented JSON
    dump. Only ``state`` gets this treatment in the native contract -- unlike
    the served wording, the runner interpolates ``instructions`` and option
    descriptions directly (``str()``), which ``_build_native_prompt`` mirrors."""
    if isinstance(value, str):
        return value
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
    )


def rotation_order(option_count: int, rotation: int) -> tuple[int, ...]:
    """Map each displayed label position to its original option index."""
    if option_count < 2 or rotation < 0:
        raise ValueError("at least two options and a nonnegative rotation are required")
    return tuple(
        (position + rotation) % option_count for position in range(option_count)
    )


def _validate_order(order: Sequence[int], option_count: int) -> None:
    if (
        len(order) != option_count
        or any(isinstance(index, bool) or not isinstance(index, int) for index in order)
        or set(order) != set(range(option_count))
    ):
        raise ValueError("option order must be a permutation of all original indices")


def build_prompt(
    state: StructuredText,
    plan: QuestionPlan,
    labels: Sequence[str],
    order: Sequence[int] | None = None,
    *,
    wording: PromptWording = "served",
) -> str:
    """Build user content; the serving layer applies a non-thinking chat template.

    Labels must already have been verified as distinct single-token symbols in
    the assistant context by the serving layer. The shared state is first to
    make prefix reuse possible across questions and option rotations.

    ``wording="served"`` (default, unchanged) renders this adapter's own
    contract (``Context:/.../Options: A: name: description``).

    ``wording="native"`` renders the jevbench-hard runner's own contract
    instead (run_suites_rotation.py: ``PREFIX``/``State:``/``Question:``/
    ``Options: A. name: description``), which the runner's own ablation
    measured as worth about +6 points on jevbench-hard by wording alone. It
    requires canonical, single-token ``A``..``Z`` labels (at most 26 options)
    in the given order, because that is the letter-token contract the native
    runner assumes; a backend whose tokenizer's first labels are not exactly
    those letters cannot serve this wording and ``build_prompt`` raises.
    """
    option_count = len(plan.option_names)
    if len(labels) != option_count or len(set(labels)) != option_count:
        raise ValueError("exactly one distinct label per option is required")
    if any(
        not isinstance(label, str) or not label or "\n" in label for label in labels
    ):
        raise ValueError("labels must be nonempty single-line strings")
    order = tuple(range(option_count)) if order is None else order
    _validate_order(order, option_count)
    if wording == "native":
        return _build_native_prompt(state, plan, labels, order)
    if wording != "served":
        raise ValueError('wording must be "served" or "native"')
    lines = []
    for label, index in zip(labels, order):
        name = plan.option_names[index]
        if plan.type == "noul":
            name = "yes" if name == "true" else "no"
        description = plan.descriptions[index]
        text = (
            name if description is None else f"{name}: {render_json_text(description)}"
        )
        lines.append(f"{label}: {text}")
    return (
        f"Context:\n{render_json_text(state)}\n\n"
        "Answer the question with only the label of the best option "
        "(the complete label before the colon), nothing else.\n"
        f"Question: {render_json_text(plan.instructions)}\nOptions:\n"
        + "\n".join(lines)
    )


def _build_native_prompt(
    state: StructuredText,
    plan: QuestionPlan,
    labels: Sequence[str],
    order: Sequence[int],
) -> str:
    """``build_prompt(..., wording="native")``'s body: a byte-for-byte port of
    run_suites_rotation.py's ``option_texts`` + ``format_prompt`` for the
    normalized KEV question schema this adapter already validates into
    (``plan.type``/``plan.option_names``/``plan.descriptions``).

    Ported quirks, kept intentionally rather than "fixed", because parity with
    the runner's actual output is the point: only ``state`` gets a JSON dump
    when non-string (``instructions`` and descriptions are interpolated with
    plain ``str()``, exactly like the runner's f-strings), and a *falsy*
    description (not just ``None``) is omitted, matching ``if description``
    in the runner's ``format_prompt``.
    """
    option_count = len(plan.option_names)
    if tuple(labels) != tuple(_NATIVE_DISPLAY[:option_count]):
        raise ValueError(
            "native prompt wording requires canonical single-token A-Z labels "
            "in order (at most 26 options); this backend's labels are "
            f"{tuple(labels)!r}"
        )
    lines = []
    for position, index in enumerate(order):
        name = plan.option_names[index]
        if plan.type == "noul":
            name = "yes" if name == "true" else "no"
        description = plan.descriptions[index]
        line = f"{_NATIVE_DISPLAY[position]}. {name}"
        if description:
            line += f": {description}"
        lines.append(line)
    return (
        f"{NATIVE_PROMPT_PREFIX}\n\n"
        f"State:\n{render_native_state_text(state)}\n\n"
        f"Question:\n{plan.instructions}\n\nOptions:\n" + "\n".join(lines)
    )


def probabilities_from_logprobs(
    logprobs: Sequence[float], temperature: float = 1.0
) -> list[float]:
    """Normalize selected vocabulary logprobs, equivalent to label-logit softmax.

    Subtract the maximum *before* scaling to stay stable even with tiny positive
    temperatures or very negative logprobs. Missing/nonfinite engine values are
    errors, rather than fabricated probabilities.
    """
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and greater than zero")
    if len(logprobs) < 2:
        raise ValueError("at least two label logprobs are required")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        for value in logprobs
    ):
        raise ValueError("every label must have a finite numeric logprob")
    maximum = max(logprobs)
    weights = [math.exp((value - maximum) / temperature) for value in logprobs]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def reduce_probabilities(
    plan: QuestionPlan,
    logprob_vectors: Sequence[Sequence[float]],
    orders: Sequence[Sequence[int]],
    options: SystemOneOptions,
) -> dict:
    """Scale, undo rotations, then average probabilities across evaluations.

    Returned ``logprobs`` (when requested) is a list of canonical option maps,
    one per evaluation. These are unmodified vocabulary-normalized logprobs,
    not raw logits and not the final label distribution. Keeping each evaluation
    separately permits fitting temperature before permutation averaging.
    """
    if not logprob_vectors or len(logprob_vectors) != len(orders):
        raise ValueError("each logprob vector must have a matching option order")
    option_count = len(plan.option_names)
    temperature = options.temperature if options.temperature_scaling else 1.0
    columns: list[list[float]] = [[] for _ in range(option_count)]
    raw_rows = []
    for vector, order in zip(logprob_vectors, orders):
        if len(vector) != option_count:
            raise ValueError("engine must return one logprob for every label")
        _validate_order(order, option_count)
        probabilities = probabilities_from_logprobs(vector, temperature)
        raw = {}
        for position, original in enumerate(order):
            columns[original].append(probabilities[position])
            raw[plan.option_names[original]] = float(vector[position])
        raw_rows.append({name: raw[name] for name in plan.option_names})
    averaged = [math.fsum(column) / len(logprob_vectors) for column in columns]
    # Account for floating-point summation drift before entropy and expectation.
    total = math.fsum(averaged)
    averaged = [probability / total for probability in averaged]
    answer = {"type": plan.type}
    if plan.type == "noul":
        answer["noul"] = averaged[0]
    else:
        answer["probabilities"] = dict(zip(plan.option_names, averaged))
        entropy = -math.fsum(p * math.log(p) for p in averaged if p > 0)
        answer["confidence"] = min(
            1.0, max(0.0, 1.0 - entropy / math.log(option_count))
        )
        if plan.type == "choice":
            answer["choice"] = plan.option_names[
                max(range(option_count), key=averaged.__getitem__)
            ]
        else:
            answer["score"] = math.fsum(index * p for index, p in enumerate(averaged))
            answer["legend"] = dict(zip(plan.option_names, plan.descriptions))
    if options.return_logprobs:
        answer["logprobs"] = raw_rows
    return answer
