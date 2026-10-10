"""Answer-label schemes shared by serving, readout export, and training data.

A scheme is an ordered list of candidate label strings. A model's labels are
the candidates that its tokenizer encodes as one distinct, non-special token
which decodes back to exactly the candidate, taken in candidate order, at most
``MAX_LABELS``. A question with ``n`` options always uses the first ``n``.

``upper`` (the default): ``A``..``Z``, then two-letter ``AA``, ``AB``, ...,
``ZZ``, then three-letter ``AAA``..``ZZZ``; uppercase ASCII only, ordered by
length and then alphabetically. Three-letter candidates are only consulted
when the one- and two-letter single tokens number fewer than ``MAX_LABELS``.

``served`` (the original order, kept to reproduce pre-v2.2 serving and for the
3B learned head, whose rows are A-Z, a-z, 0-9): ``A``..``Z``, ``a``..``z``,
``0``..``9``, ``000``..``999``, ``AA``..``ZZ``, ``aa``..``zz``. No v2 training
row had more than 26 options, so beyond Z neither order is what v2 learned.

Both schemes give ``A``..``Z`` to questions with at most 26 options whenever
the tokenizer has those 26 single-token letters.
"""

import itertools
import os
import re
import string
from collections.abc import Mapping, Sequence
from typing import Any

LABEL_SCHEMES = ("served", "upper")
DEFAULT_LABEL_SCHEME = "upper"
LABEL_SCHEME_ENV = "JEV_LABEL_SCHEME"
MAX_LABELS = 255
UPPER_MAX_LENGTH = 3
_UPPER_LABEL = re.compile(r"[A-Z]{1,3}")


def validate_label_scheme(value: Any) -> str:
    if value not in LABEL_SCHEMES:
        raise ValueError(
            f"label scheme must be one of {', '.join(LABEL_SCHEMES)}; got {value!r}"
        )
    return value


def resolve_label_scheme(
    value: str | None = None, environ: Mapping[str, str] | None = None
) -> str:
    """An explicit value, else ``JEV_LABEL_SCHEME``, else the default (upper)."""
    if value is None or value == "":
        environ = os.environ if environ is None else environ
        value = environ.get(LABEL_SCHEME_ENV) or DEFAULT_LABEL_SCHEME
    return validate_label_scheme(value)


def _served_candidates() -> tuple[str, ...]:
    letters = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return tuple(
        list(letters)
        + [f"{value:03d}" for value in range(1000)]
        + [
            first + second
            for alphabet in (string.ascii_uppercase, string.ascii_lowercase)
            for first in alphabet
            for second in alphabet
        ]
    )


def _upper_length(length: int) -> tuple[str, ...]:
    return tuple(
        "".join(chars)
        for chars in itertools.product(string.ascii_uppercase, repeat=length)
    )


def candidate_tiers(scheme: str) -> tuple[tuple[str, ...], ...]:
    """Candidates in order, grouped so a caller may stop after an early group.

    A later group is consulted only while fewer than MAX_LABELS labels were
    found; the concatenated order is the scheme's order either way.
    """
    validate_label_scheme(scheme)
    if scheme == "served":
        return (_served_candidates(),)
    return (
        _upper_length(1) + _upper_length(2),
        _upper_length(3),
    )


def label_candidates(scheme: str) -> list[str]:
    return [label for tier in candidate_tiers(scheme) for label in tier]


def select_labels(
    tokenizer: Any, scheme: str, *, limit: int = MAX_LABELS
) -> tuple[list[str], list[int]]:
    """The scheme's labels for a tokenizer with ``encode`` and ``decode``.

    A candidate is kept when it encodes (without special tokens) to exactly
    one token that is not special, not already taken, and decodes back to the
    candidate. This is the rule every exporter, backend, and training-data
    writer must share; it needs only the tokenizer files.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_LABELS:
        raise ValueError(f"limit must be an integer from 1 to {MAX_LABELS}")
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())
    labels: list[str] = []
    token_ids: list[int] = []
    seen: set[int] = set()
    for tier in candidate_tiers(scheme):
        for label in tier:
            ids = tokenizer.encode(label, add_special_tokens=False)
            if not isinstance(ids, list) or len(ids) != 1 or type(ids[0]) is not int:
                continue
            token = ids[0]
            if token < 0 or token in seen or token in special:
                continue
            if tokenizer.decode([token], skip_special_tokens=True) != label:
                continue
            labels.append(label)
            token_ids.append(token)
            seen.add(token)
            if len(labels) == limit:
                return labels, token_ids
    return labels, token_ids


def follows_scheme_order(scheme: str, labels: Sequence[str]) -> bool:
    """Whether labels are distinct candidates of the scheme in scheme order.

    A structural check that needs no tokenizer: it cannot tell whether a
    skipped candidate was a single token, only that nothing is out of order.
    """
    validate_label_scheme(scheme)
    if not labels or any(not isinstance(label, str) for label in labels):
        return False
    if scheme == "upper":
        if any(_UPPER_LABEL.fullmatch(label) is None for label in labels):
            return False
        ranks = [(len(label), label) for label in labels]
    else:
        index = {label: position for position, label in enumerate(_served_candidates())}
        if any(label not in index for label in labels):
            return False
        ranks = [index[label] for label in labels]
    return all(before < after for before, after in itertools.pairwise(ranks))
