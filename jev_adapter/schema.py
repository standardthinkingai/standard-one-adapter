"""Joint schema head serving: render one prompt per request and read every answer from it.

A schema-head engine (``--decision-readout-path`` naming a schema head directory)
scores all options of all questions in one prefill. This module renders the
request with the head's training template (``schema-v1``, canonical wording W0,
question preview), locates every question/option/state token span, and turns the
engine's per-option logits into the System One response.

Spans are half-open token ranges in the prompt as the engine's image processor
expands it; options are always in canonical order (choice: caller order, noul:
true then false, score: levels 0..K-1) whatever order they are displayed in.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .backend import AdapterError

TEMPLATE_ID = "schema-v1"
MAX_OPTIONS = 255
MAX_SCORE_LEVELS = 10
# Wording W0, the only one used outside training augmentation.
PREAMBLE = "Read the state and answer every question. Each question lists its possible answers."
PREVIEW_HEADER = "Questions to answer:"
STATE_HEADER = "State:"
QUESTIONS_HEADER = "Questions:"
TYPE_TAGS = {"choice": "(choose one)", "noul": "(true or false)", "score": "(score 0 to {top})"}
BULLET = "- "


def render_value(value) -> str:
    """Plain strings verbatim; structured values as sorted, indented JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)


@dataclass(frozen=True)
class SchemaQuestion:
    key: str
    type: str
    instructions: object
    # (name, description) in canonical order
    options: tuple[tuple[str, object], ...]


def schema_questions(request) -> list[SchemaQuestion]:
    """System One questions in canonical option order, checked against the template."""
    questions = []
    for key, question in request.questions.items():
        if not key.strip() or any(c in key for c in "\n[]"):
            raise AdapterError(
                "invalid_question_key",
                "Schema models need question keys without brackets or line breaks.",
                f"questions.{key}",
            )
        criteria = question.criteria
        if question.type == "choice":
            options = tuple(criteria.items())
        elif question.type == "noul":
            criteria = criteria or {}
            options = (("true", criteria.get("true")), ("false", criteria.get("false")))
        else:
            options = tuple((str(level), text) for level, text in enumerate(criteria))
        names = [name for name, _ in options]
        if any(not name.strip() or "\n" in name for name in names):
            raise AdapterError(
                "invalid_option_name",
                "Schema models need option names without line breaks.",
                f"questions.{key}",
            )
        high = MAX_OPTIONS if question.type == "choice" else MAX_SCORE_LEVELS
        if not 2 <= len(options) <= high:
            raise AdapterError(
                "invalid_option_count",
                f"A {question.type} question needs 2 to {high} options.",
                f"questions.{key}",
            )
        questions.append(SchemaQuestion(key, question.type, question.instructions, options))
    return questions


def render(state, questions: Sequence[SchemaQuestion], option_orders: dict[str, list[int]]):
    """User text and character segments: state, preview/question per key, options per canonical index."""
    parts, position = [], 0
    segments = {"preview": {}, "question": {}, "option": {}}

    def emit(text):
        nonlocal position
        start = position
        parts.append(text)
        position += len(text)
        return start, position

    emit(PREAMBLE + "\n\n" + PREVIEW_HEADER + "\n")
    for question in questions:
        segments["preview"][question.key] = emit(
            f"[{question.key}] {render_value(question.instructions)}"
        )
        emit("\n")
    emit("\n")
    state_start, _ = emit(STATE_HEADER + "\n")
    _, state_end = emit(render_value(state))
    segments["state"] = (state_start, state_end)
    emit("\n\n" + QUESTIONS_HEADER)
    for question in questions:
        count = len(question.options)
        tag = TYPE_TAGS[question.type].format(top=count - 1)
        emit("\n")
        segments["question"][question.key] = emit(
            f"[{question.key}] {tag} {render_value(question.instructions)}"
        )
        spans = [None] * count
        for canonical in option_orders.get(question.key, range(count)):
            name, description = question.options[canonical]
            description = render_value(description)
            emit("\n" + BULLET)
            spans[canonical] = emit(f"{name}: {description}" if description else name)
        segments["option"][question.key] = spans
    return "".join(parts), segments


def rotation_orders(questions: Sequence[SchemaQuestion], rotation: int) -> dict[str, list[int]]:
    """Display order for one permutation: choice and noul rotate, score levels never move."""
    orders = {}
    for question in questions:
        count = len(question.options)
        if question.type != "score" and rotation:
            orders[question.key] = [(p + rotation) % count for p in range(count)]
    return orders


@dataclass(frozen=True)
class SchemaPrompt:
    """One prefill: native prompt IDs with one [IMG] per image, plus the engine plan."""

    input_ids: tuple[int, ...]
    plan: dict
    text_tokens: int


class SchemaRenderer:
    """Template schema-v1 with the head's own tokenizer and chat template.

    tokenizer: the transformers tokenizer the head was trained with (chat
    template + round trip); fast: ``tokenizers.Tokenizer`` of the same
    tokenizer.json (character offsets); processor: only for image prompts,
    whose chat text comes from the processor's template as in training.
    """

    def __init__(self, tokenizer, fast, processor=None, *, image_token_id=None):
        self.tokenizer, self.fast, self.processor = tokenizer, fast, processor
        self.image_token_id = image_token_id
        self.special_ids = set(tokenizer.all_special_ids)

    @classmethod
    def from_pretrained(cls, path: str) -> SchemaRenderer:
        import transformers
        from tokenizers import Tokenizer

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, fix_mistral_regex=True
        )
        fast = Tokenizer.from_file(f"{path}/tokenizer.json")
        processor = transformers.AutoProcessor.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, fix_mistral_regex=True
        )
        image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
        return cls(tokenizer, fast, processor, image_token_id=image_token_id)

    def _chat(self, content: str, image_count: int) -> str:
        if image_count:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}] * image_count
                    + [{"type": "text", "text": content}],
                }
            ]
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def encode(
        self,
        state,
        questions: Sequence[SchemaQuestion],
        option_orders: dict[str, list[int]],
        image_tokens: Sequence[int],
    ) -> SchemaPrompt:
        """image_tokens: the expanded token count of each image, in request order."""
        content, segments = render(state, questions, option_orders)
        chat = self._chat(content, len(image_tokens))
        if chat.count(content) != 1:
            raise RuntimeError("the chat template repeated or dropped the schema content")
        c0 = chat.index(content)
        encoded = self.fast.encode(chat, add_special_tokens=False)
        ids, offsets = list(encoded.ids), list(encoded.offsets)
        if not image_tokens:
            reference = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            if list(reference) != ids:
                raise RuntimeError("schema prompt does not round-trip through the chat template")
        positions, image_span = self._expand(ids, offsets, c0, image_tokens)
        plan = self._plan(ids, offsets, positions, chat, c0, content, segments, questions, image_span)
        return SchemaPrompt(tuple(ids), plan, len(ids) - len(image_tokens))

    def _expand(self, ids, offsets, c0, image_tokens):
        """Expanded token position of every native token; the image block's span."""
        if not image_tokens:
            return list(range(len(ids))), None
        markers = [i for i, token in enumerate(ids) if token == self.image_token_id]
        first = markers[0] if markers else -1
        if (
            len(markers) != len(image_tokens)
            or markers != list(range(first, first + len(markers)))
            or offsets[markers[-1]][1] > c0
        ):
            raise RuntimeError("schema images must be consecutive before the text")
        block = sum(image_tokens)
        shift = block - len(markers)
        positions = [i if i < first else i + shift for i in range(len(ids))]
        return positions, (first, first + block)

    def _plan(self, ids, offsets, positions, chat, c0, content, segments, questions, image_span):
        n_tokens = positions[-1] + 1
        names, ranges = [], []

        def add(name, span):
            names.append(name)
            ranges.append((span[0] + c0, span[1] + c0))
            return len(names) - 1

        state_id = add(("state",), segments["state"])
        ids_of = {}
        for question in questions:
            key = question.key
            ids_of[key] = (
                add(("preview", key), segments["preview"][key]),
                add(("question", key), segments["question"][key]),
                [add(("option", key, j), span) for j, span in enumerate(segments["option"][key])],
            )
        order = sorted(range(len(ranges)), key=lambda i: ranges[i][0])
        starts = [ranges[i][0] for i in order]

        def segment_of(char):
            k = bisect.bisect_right(starts, char) - 1
            if k < 0:
                return -1
            sid = order[k]
            return sid if ranges[sid][0] <= char < ranges[sid][1] else -1

        content_end = c0 + len(content)
        spans: dict[int, list[int]] = {}
        for t, (a, b) in enumerate(offsets):
            if b <= c0 or a >= content_end or ids[t] == self.image_token_id:
                continue
            if ids[t] in self.special_ids:
                raise AdapterError(
                    "special_token_in_content",
                    "The request text contains a reserved control token.",
                    "questions",
                )
            char = a
            while char < b and chat[char].isspace():
                char += 1
            sid = segment_of(char if char < b else a)
            if sid < 0:
                continue
            span = spans.get(sid)
            if span is not None and span[1] != t:
                raise RuntimeError(f"schema span {names[sid]} is not contiguous")
            spans[sid] = [t, t + 1] if span is None else [span[0], t + 1]
        for sid, (ca, cb) in enumerate(ranges):
            if sid not in spans:
                raise RuntimeError(f"schema span {names[sid]} is empty")
            a, b = spans[sid]
            first, last = ca, cb
            while first < cb and chat[first].isspace():
                first += 1
            while last > first and chat[last - 1].isspace():
                last -= 1
            if offsets[a][0] > first or offsets[b - 1][1] < last:
                raise RuntimeError(f"schema span {names[sid]} straddles its tokens")

        def expanded(sid):
            a, b = spans[sid]
            return [positions[a], positions[b - 1] + 1]

        state_spans = [expanded(state_id)]
        if image_span is not None:
            state_spans.insert(0, list(image_span))
        o_span = [[expanded(s) for s in ids_of[q.key][2]] for q in questions]
        plan = {
            "template": TEMPLATE_ID,
            "n_tokens": n_tokens,
            "qtype": [q.type for q in questions],
            "p_span": [expanded(ids_of[q.key][0]) for q in questions],
            "q_span": [expanded(ids_of[q.key][1]) for q in questions],
            "o_span": o_span,
            "state_spans": state_spans,
            "image_span": list(image_span) if image_span is not None else None,
            "n_values": sum(len(spans) for spans in o_span),
        }
        return plan


def answer_from_logits(
    question: SchemaQuestion,
    vectors: Sequence[Sequence[float]],
    temperature: float,
    return_logprobs: bool,
) -> dict:
    """Average per-permutation softmax(z / T) over canonical options into a System One answer."""
    names = [name for name, _ in question.options]
    totals = [0.0] * len(names)
    raw = []
    for z in vectors:
        peak = max(z)
        log_norm = peak + math.log(math.fsum(math.exp(v - peak) for v in z))
        raw.append({name: v - log_norm for name, v in zip(names, z)})
        scaled = [v / temperature for v in z]
        peak = max(scaled)
        weights = [math.exp(v - peak) for v in scaled]
        total = math.fsum(weights)
        for j, w in enumerate(weights):
            totals[j] += w / total
    norm = math.fsum(totals)
    probs = [v / norm for v in totals]
    answer: dict = {"type": question.type}
    if question.type == "noul":
        answer["noul"] = probs[0]
    else:
        answer["probabilities"] = dict(zip(names, probs))
        entropy = -math.fsum(p * math.log(p) for p in probs if p > 0)
        answer["confidence"] = min(1.0, max(0.0, 1.0 - entropy / math.log(len(probs))))
        if question.type == "choice":
            answer["choice"] = names[max(range(len(names)), key=probs.__getitem__)]
        else:
            answer["score"] = math.fsum(i * p for i, p in enumerate(probs))
            answer["legend"] = {name: description for name, description in question.options}
    if return_logprobs:
        answer["logprobs"] = raw
    return answer


def split_logits(questions: Sequence[SchemaQuestion], logits: Sequence[float]) -> list[list[float]]:
    """The engine's flat per-request vector back into one vector per question."""
    out, start = [], 0
    for question in questions:
        out.append(list(logits[start : start + len(question.options)]))
        start += len(question.options)
    if start != len(logits):
        raise RuntimeError("schema logits do not match the request's options")
    return out


ImageCounter = Callable[[str], int]
