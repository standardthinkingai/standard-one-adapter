"""Answer-label schemes: exact lists, the <=26 guarantee, and end-to-end mapping."""

import hashlib
import json
import re
import string
from pathlib import Path

import httpx
import pytest

from jev_adapter.labels import (
    DEFAULT_LABEL_SCHEME,
    MAX_LABELS,
    follows_scheme_order,
    label_candidates,
    resolve_label_scheme,
    select_labels,
)
from jev_adapter.server import create_app
from jev_adapter.sglang import SGLangBackend

GOLDEN_PATH = Path(__file__).parent / "data" / "upper-labels-ministral3.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text())
UPPER = GOLDEN["labels"]
COUNTS = (2, 26, 27, 77, 151, 255)


class VocabularyTokenizer:
    """Every character is one token; listed multi-letter strings are one token.

    With the golden vocabulary this has exactly the pinned Ministral-3
    single-token pattern for the upper labels (e.g. AJ and AQ are split).
    """

    all_special_ids = (0,)

    def __init__(self, words):
        self.word_ids = {word: 100_000 + index for index, word in enumerate(words)}
        self.words = {token: word for word, token in self.word_ids.items()}

    def encode(self, text, add_special_tokens=False):
        ids, position = [], 0
        while position < len(text):
            for length in (3, 2):
                piece = text[position : position + length]
                if len(piece) == length and piece in self.word_ids:
                    ids.append(self.word_ids[piece])
                    position += length
                    break
            else:
                ids.append(ord(text[position]) + 1000)
                position += 1
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(
            self.words[token] if token in self.words else chr(token - 1000)
            for token in ids
        )


def ministral_like():
    return VocabularyTokenizer([label for label in UPPER if len(label) > 1])


def test_default_and_resolution():
    assert DEFAULT_LABEL_SCHEME == "upper"
    assert resolve_label_scheme(environ={}) == "upper"
    assert resolve_label_scheme(environ={"JEV_LABEL_SCHEME": "served"}) == "served"
    assert resolve_label_scheme("upper", {"JEV_LABEL_SCHEME": "served"}) == "upper"
    assert resolve_label_scheme("", {"JEV_LABEL_SCHEME": ""}) == "upper"
    for bad in ("lower", "UPPER", 3):
        with pytest.raises(ValueError, match="label scheme"):
            resolve_label_scheme(bad, {})
    with pytest.raises(ValueError, match="label scheme"):
        resolve_label_scheme(None, {"JEV_LABEL_SCHEME": "lower"})


def test_candidate_orders():
    upper = label_candidates("upper")
    assert upper[:28] == [*string.ascii_uppercase, "AA", "AB"]
    assert upper[26 + 675] == "ZZ" and upper[26 + 676] == "AAA"
    assert upper[-1] == "ZZZ" and len(upper) == 26 + 26**2 + 26**3
    assert all(re.fullmatch(r"[A-Z]{1,3}", label) for label in upper)
    served = label_candidates("served")
    # The served order is unchanged from before the scheme option existed.
    assert served[:62] == list(
        string.ascii_uppercase + string.ascii_lowercase + string.digits
    )
    assert served[62:64] == ["000", "001"] and served[1062:1064] == ["AA", "AB"]
    assert served[-1] == "zz" and len(served) == 62 + 1000 + 2 * 26**2


@pytest.mark.parametrize("count", COUNTS)
def test_upper_lists_for_each_option_count(count):
    labels, token_ids = select_labels(ministral_like(), "upper", limit=count)
    assert labels == UPPER[:count]
    assert len(set(labels)) == len(set(token_ids)) == count
    assert all(re.fullmatch(r"[A-Z]{1,2}", label) for label in labels)
    assert labels[: min(count, 26)] == list(string.ascii_uppercase[:count])
    if count > 26:
        assert labels[26:29] == ["AA", "AB", "AC"][: count - 26]
        # AJ, AQ and AW are not single Ministral-3 tokens, so they are skipped.
        assert not {"AJ", "AQ", "AW"} & set(labels)


def test_exact_boundary_labels():
    labels, _ = select_labels(ministral_like(), "upper")
    assert len(labels) == MAX_LABELS
    assert labels[26] == "AA"
    assert labels[76] == "CG"  # the 77th label: the last of BANKING77's options
    assert labels[150] == "FT"  # the 151st: the last of CLINC150 with oos
    assert labels[254] == "MH"


@pytest.mark.parametrize("count", range(2, 27))
def test_at_most_26_options_keep_a_to_z_in_both_schemes(count):
    tokenizer = ministral_like()
    upper = select_labels(tokenizer, "upper", limit=count)
    served = select_labels(tokenizer, "served", limit=count)
    assert upper == served
    assert upper[0] == list(string.ascii_uppercase[:count])


def test_served_scheme_still_uses_lowercase_after_z():
    labels, _ = select_labels(ministral_like(), "served")
    assert labels[26:30] == ["a", "b", "c", "d"]
    assert labels[52:62] == list(string.digits)
    assert labels[62] == "AA"


@pytest.mark.parametrize("count", COUNTS)
def test_label_index_round_trip(count):
    labels, token_ids = select_labels(ministral_like(), "upper", limit=count)
    index = {label: position for position, label in enumerate(labels)}
    token_index = {token: position for position, token in enumerate(token_ids)}
    assert len(index) == len(token_index) == count
    for position, (label, token) in enumerate(zip(labels, token_ids)):
        assert index[label] == token_index[token] == position
        assert labels[index[label]] == label
    # Every label is also a prefix-free answer at the list level: a longer label
    # is one token, never the token of a shorter label followed by another.
    tokenizer = ministral_like()
    assert all(
        tokenizer.encode(label) == [token] for label, token in zip(labels, token_ids)
    )


def test_three_letter_continuation_when_two_letters_run_out():
    # Only AA..AJ are two-letter tokens; the list continues with AAA, AAB, ...
    two = [f"A{letter}" for letter in "ABCDEFGHIJ"]
    three = ["AAA", "AAB", "ABA", "ZZZ"]
    tokenizer = VocabularyTokenizer(two + three)
    labels, _ = select_labels(tokenizer, "upper")
    assert labels == list(string.ascii_uppercase) + two + three


def test_special_duplicate_and_lossy_tokens_are_skipped():
    class Tricky(VocabularyTokenizer):
        all_special_ids = (ord("B") + 1000,)

        def decode(self, ids, skip_special_tokens=True):
            text = super().decode(ids, skip_special_tokens)
            return "ab" if text == "AB" else text

        def encode(self, text, add_special_tokens=False):
            return [ord("C") + 1000] if text == "AC" else super().encode(text)

    labels, _ = select_labels(Tricky(["AA", "AB", "AC", "AD"]), "upper")
    assert labels[:3] == ["A", "C", "D"] and "B" not in labels
    assert labels[25:] == ["AA", "AD"]


def test_scheme_order_check():
    assert follows_scheme_order("upper", ["A", "B", "AA", "AB", "MH", "AAA"])
    assert not follows_scheme_order("upper", ["A", "a"])
    assert not follows_scheme_order("upper", ["B", "A"])
    assert not follows_scheme_order("upper", ["AA", "Z"])
    assert follows_scheme_order("served", ["A", "Z", "a", "0", "AA", "aa"])
    assert not follows_scheme_order("served", ["a", "A"])
    assert not follows_scheme_order("served", ["A", "AAA"])


def test_pinned_ministral3_tokenizer_matches_the_golden_list():
    transformers = pytest.importorskip("transformers")
    tokenizer_info = GOLDEN["tokenizer"]
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_info["id"],
            revision=tokenizer_info["revision"],
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001 - absent from this machine's HF cache
        pytest.skip("pinned Ministral-3 tokenizer is not cached locally")
    labels, token_ids = select_labels(tokenizer, "upper")
    assert labels == GOLDEN["labels"]
    assert token_ids == GOLDEN["token_ids"]
    # The recorded probe below is what this tokenizer really does.
    for candidate, ids in PROBE["encode"].items():
        assert tokenizer.encode(candidate, add_special_tokens=False) == ids
    for token, text in PROBE["decode"].items():
        assert tokenizer.decode([int(token)], skip_special_tokens=True) == text
    assert set(tokenizer.all_special_ids) <= ProbeTokenizer().all_special_ids
    from jev_adapter.native_tokenizer import NativeTokenizer

    # Every label continues the native chat prompt by exactly its token.
    NativeTokenizer(tokenizer).prepare("Choose.", tuple(labels), tuple(token_ids), None)


# Raw encode/decode results of the pinned Ministral-3 tokenizer for every one-
# and two-letter uppercase candidate, recorded on the GPU host with
# AutoTokenizer and cross-checked against tekken.json (mistral_common) and the
# release tokenizer.json. Unlike ministral_like(), which is built from the
# golden list, this table is tokenizer data only, so the tests below check the
# selection rule against the real tokenizer on every CI run.
PROBE_PATH = Path(__file__).parent / "data" / "ministral3-label-probe.json"
PROBE = json.loads(PROBE_PATH.read_text())


class ProbeTokenizer:
    """Replays the recorded tokenizer; a string outside the table is an error."""

    def __init__(self):
        self.all_special_ids = {
            token
            for first, last in PROBE["special_id_ranges"]
            for token in range(first, last + 1)
        }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(PROBE["encode"][text])

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "".join(PROBE["decode"][str(token)] for token in ids)


def test_probe_records_the_golden_lists_tokenizer():
    assert PROBE["schema"] == "jev-tokenizer-probe-v1"
    assert PROBE["tokenizer"] == GOLDEN["tokenizer"]
    expected = PROBE["expected_upper"]
    assert expected["labels"] == GOLDEN["labels"]
    assert expected["token_ids"] == GOLDEN["token_ids"]
    assert expected["source_sha256"] == hashlib.sha256(
        GOLDEN_PATH.read_bytes()
    ).hexdigest()
    candidates = [*string.ascii_uppercase] + [
        first + second
        for first in string.ascii_uppercase
        for second in string.ascii_uppercase
    ]
    assert list(PROBE["encode"]) == candidates
    singles = [label for label, ids in PROBE["encode"].items() if len(ids) == 1]
    counts = GOLDEN["single_token_counts"]
    assert sum(len(label) == 1 for label in singles) == counts["one_letter"] == 26
    assert sum(len(label) == 2 for label in singles) == counts["two_letter"] == 439


def test_rule_on_the_recorded_ministral3_tokenizer_gives_the_golden_list():
    # Three-letter candidates are absent from the table, so consulting one
    # would raise: 255 labels must come from one and two letters.
    labels, token_ids = select_labels(ProbeTokenizer(), "upper")
    assert labels == GOLDEN["labels"]
    assert token_ids == GOLDEN["token_ids"]
    for count in COUNTS:
        assert select_labels(ProbeTokenizer(), "upper", limit=count) == (
            GOLDEN["labels"][:count],
            GOLDEN["token_ids"][:count],
        )


@pytest.mark.asyncio
async def test_http_backend_picks_the_golden_list_from_the_recorded_tokenizer():
    engine = LabelEngine(ProbeTokenizer(), "A")
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as upstream:
        backend = SGLangBackend(
            "http://engine", "standard-one", client=upstream, label_scheme="upper"
        )
        await backend.start()
        try:
            for count in COUNTS:
                assert backend.labels(count) == (
                    tuple(GOLDEN["labels"][:count]),
                    tuple(GOLDEN["token_ids"][:count]),
                )
        finally:
            await backend.close()


class LabelEngine:
    """A remote engine whose tokenizer has two-letter tokens and prefers one label."""

    def __init__(self, tokenizer, preferred):
        self.tokenizer = tokenizer
        self.preferred = preferred
        self.prompts = []

    def __call__(self, request):
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        if path == "/model_info":
            return httpx.Response(
                200,
                json={
                    "served_model_name": "standard-one",
                    "model_path": "org/standard-one",
                    "is_generation": True,
                },
            )
        if path == "/server_info":
            return httpx.Response(200, json={"speculative_algorithm": None})
        encode = self.tokenizer.encode
        if path == "/v1/tokenize":
            if "messages" in body:
                content = body["messages"][-1]["content"]
                self.prompts.append(content)
                return httpx.Response(
                    200, json={"tokens": encode(f"<u>{content}</u><a>")}
                )
            return httpx.Response(
                200, json={"tokens": [encode(text) for text in body["prompt"]]}
            )
        if path == "/v1/detokenize":
            tokens = body["tokens"]
            decode = self.tokenizer.decode
            text = (
                [decode(ids) for ids in tokens]
                if isinstance(tokens[0], list)
                else decode(tokens)
            )
            return httpx.Response(200, json={"text": text})
        if path == "/generate":
            preferred = encode(self.preferred)[0]
            entries = [
                [-0.1 if token == preferred else -9.0, token, None]
                for token in body["token_ids_logprob"]
            ]
            return httpx.Response(
                200,
                json={
                    "meta_info": {
                        "completion_tokens": 0,
                        "prompt_tokens": len(body["input_ids"]),
                        "cached_tokens": 0,
                        "output_token_ids_logprobs": [entries],
                    }
                },
            )
        raise AssertionError(path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheme,count,preferred,expected_option",
    [
        ("upper", 77, "CG", 76),
        ("upper", 77, "AA", 26),
        ("upper", 151, "FT", 150),
        ("upper", 26, "Z", 25),
        ("served", 77, "a", 26),
        ("served", 26, "Z", 25),
    ],
)
async def test_label_maps_to_its_option_through_the_http_stack(
    scheme, count, preferred, expected_option
):
    engine = LabelEngine(ministral_like(), preferred)
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as upstream:
        backend = SGLangBackend(
            "http://engine", "standard-one", client=upstream, label_scheme=scheme
        )
        app = create_app(backend)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://adapter"
            ) as client,
        ):
            models = (await client.get("/v1/models")).json()
            assert models["data"][0]["label_scheme"] == scheme
            options = {f"intent-{index}": None for index in range(count)}
            response = await client.post(
                "/v1/systemone",
                json={
                    "model": "standard-one",
                    "state": "I lost my card.",
                    "questions": {
                        "intent": {
                            "type": "choice",
                            "instructions": "Which intent?",
                            "criteria": options,
                        }
                    },
                },
            )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["metadata"]["label_scheme"] == scheme
    assert result["answers"]["intent"]["choice"] == f"intent-{expected_option}"
    lines = engine.prompts[-1].split("Options:\n")[1].splitlines()
    labels = [line.split(":", 1)[0] for line in lines]
    expected, _ = select_labels(ministral_like(), scheme, limit=count)
    assert labels == expected
    assert lines[expected_option] == f"{preferred}: intent-{expected_option}"
