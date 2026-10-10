"""CPU tests of request validation, prompt mapping, and decision probabilities."""

import json
import math
import string
import unittest

from pydantic import ValidationError

from jev_adapter.protocol import (
    NATIVE_PROMPT_PREFIX,
    SystemOneOptions,
    SystemOneRequest,
    build_prompt,
    plan_question,
    probabilities_from_logprobs,
    reduce_probabilities,
    rotation_order,
)

CANONICAL_LETTERS = tuple(string.ascii_uppercase)


def runner_format_prompt(state, instructions, ordered):
    """Independent, verbatim port of run_suites_rotation.py's format_prompt/
    option_texts, used as the parity oracle for wording="native" -- kept
    separate from jev_adapter.protocol so a bug in the adapter's own port
    cannot also hide in the test."""
    text = (
        state
        if isinstance(state, str)
        else json.dumps(state, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
    )
    lines = []
    for position, (name, description) in enumerate(ordered):
        lines.append(
            f"{CANONICAL_LETTERS[position]}. {name}"
            + (f": {description}" if description else "")
        )
    return (
        f"{NATIVE_PROMPT_PREFIX}\n\nState:\n{text}\n\nQuestion:\n{instructions}\n\nOptions:\n"
        + "\n".join(lines)
    )


def request_for(question, **overrides):
    return SystemOneRequest.model_validate(
        {
            "model": "test-model",
            "state": "Customer needs help.",
            "questions": {"routing": question},
            **overrides,
        }
    )


def choice_question(**overrides):
    return {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Payment issue", "technical": None},
        **overrides,
    }


class TestSystemOneValidation(unittest.TestCase):
    def test_preserves_question_and_option_order_and_structured_values(self):
        request = request_for(
            choice_question(criteria={"z": ["last", {"nested": True}], "a": None}),
            state={"ticket": [7, "안녕하세요", None]},
            questions={
                "second": choice_question(criteria={"z": {"reason": 1}, "a": None}),
                "first": {"type": "noul", "instructions": ["Is urgent?"]},
            },
        )
        self.assertEqual(list(request.questions), ["second", "first"])
        plan = plan_question(request.questions["second"])
        self.assertEqual(plan.option_names, ("z", "a"))
        self.assertEqual(plan.descriptions, ({"reason": 1}, None))
        self.assertEqual(request.state["ticket"], [7, "안녕하세요", None])

    def test_choice_accepts_255_but_rejects_256(self):
        criteria = {str(index): None for index in range(255)}
        request = request_for(choice_question(criteria=criteria))
        self.assertEqual(
            len(plan_question(request.questions["routing"]).option_names), 255
        )
        with self.assertRaises(ValidationError):
            request_for(choice_question(criteria={**criteria, "255": None}))

    def test_invalid_question_schemas(self):
        cases = [
            choice_question(type="arbitrary_json"),
            choice_question(criteria={"only": None}),
            choice_question(criteria={"": None, "b": None}),
            choice_question(criteria={"a": 42, "b": None}),
            choice_question(instructions=None),
            choice_question(instructions=False),
            choice_question(unrecognized=True),
            {"type": "noul", "instructions": "Yes?", "criteria": {"true": "yes"}},
            {
                "type": "noul",
                "instructions": "Yes?",
                "criteria": {"yes": None, "no": None},
            },
            {"type": "score", "instructions": "Rate", "criteria": ["only"]},
            {"type": "score", "instructions": "Rate", "criteria": [None] * 11},
            {
                "type": "score",
                "instructions": "Rate",
                "criteria": {"0": "low", "1": "high"},
            },
        ]
        for question in cases:
            with self.subTest(question=question), self.assertRaises(ValidationError):
                request_for(question)

    def test_empty_noul_criteria_mean_no_descriptions(self):
        # {} is the same as omitting criteria (the released server's fix).
        empty = request_for({"type": "noul", "instructions": "Urgent?", "criteria": {}})
        omitted = request_for({"type": "noul", "instructions": "Urgent?"})
        self.assertIsNone(empty.questions["routing"].criteria)
        plan = plan_question(empty.questions["routing"])
        self.assertEqual(plan, plan_question(omitted.questions["routing"]))
        self.assertEqual(plan.descriptions, (None, None))
        self.assertTrue(build_prompt("s", plan, ["A", "B"]).endswith("A: yes\nB: no"))
        # Only an empty map is relaxed; a partial or foreign map still fails.
        for criteria in ({"true": None}, {"yes": None, "no": None}):
            with self.subTest(criteria=criteria), self.assertRaises(ValidationError):
                request_for(
                    {"type": "noul", "instructions": "Urgent?", "criteria": criteria}
                )

    def test_required_fields_and_request_limits(self):
        for updates in (
            {"model": " "},
            {"model": 3},
            {"state": 42},
            {"state": None},
            {"questions": {}},
            {"questions": {" ": choice_question()}},
            {"questions": {str(i): choice_question() for i in range(257)}},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                request_for(choice_question(), **updates)
        for field in ("model", "state", "questions"):
            payload = request_for(choice_question()).model_dump()
            del payload[field]
            with self.subTest(missing=field), self.assertRaises(ValidationError):
                SystemOneRequest.model_validate(payload)

    def test_rejects_nonfinite_structured_json(self):
        for number in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(number=number), self.assertRaises(ValidationError):
                request_for(choice_question(), state={"nested": [number]})

    def test_option_bounds_and_types(self):
        for options in (
            {"temperature": 0},
            {"temperature": -1},
            {"temperature": float("inf")},
            {"temperature": float("nan")},
            {"temperature": True},
            {"temperature": "1"},
            {"permutations": 0},
            {"permutations": 17},
            {"permutations": 1.5},
            {"permutations": True},
            {"return_logprobs": "true"},
            {"return_logits": True},
            {"assistant_prefix": 3},
        ):
            with self.subTest(options=options), self.assertRaises(ValidationError):
                request_for(choice_question(), options=options)

    def test_image_sources(self):
        request = request_for(
            choice_question(),
            images=["https://example.com/frame.png", "data:image/png;base64,YQ=="],
        )
        self.assertEqual(len(request.images), 2)
        for image in ("file:///tmp/frame.png", "data:image/png;base64,", "https://", 7):
            with self.subTest(image=image), self.assertRaises(ValidationError):
                request_for(choice_question(), images=[image])


class TestSystemOnePrompts(unittest.TestCase):
    def test_multicharacter_token_labels_request_the_complete_label(self):
        plan = plan_question(request_for(choice_question()).questions["routing"])
        prompt = build_prompt("state", plan, ["AA", "000"])
        self.assertIn("the complete label before the colon", prompt)
        self.assertTrue(prompt.endswith("AA: billing: Payment issue\n000: technical"))

    def test_shared_prefix_and_rotation_preserve_option_meaning(self):
        request = request_for(choice_question(), state={"b": 2, "a": "안녕"})
        plan = plan_question(request.questions["routing"])
        first = build_prompt(request.state, plan, ["A", "B"])
        rotated = build_prompt(request.state, plan, ["A", "B"], rotation_order(2, 1))
        self.assertTrue(first.startswith('Context:\n{"a": "안녕", "b": 2}\n\n'))
        self.assertEqual(first.split("Options:\n")[0], rotated.split("Options:\n")[0])
        self.assertTrue(first.endswith("A: billing: Payment issue\nB: technical"))
        self.assertTrue(rotated.endswith("A: technical\nB: billing: Payment issue"))

    def test_boolean_default_and_explicit_criteria(self):
        request = request_for({"type": "noul", "instructions": "Urgent?"})
        plan = plan_question(request.questions["routing"])
        self.assertTrue(
            build_prompt(request.state, plan, ["A", "B"]).endswith("A: yes\nB: no")
        )
        request = request_for(
            {
                "type": "noul",
                "instructions": "Urgent?",
                "criteria": {"false": "Can wait", "true": "Now"},
            }
        )
        plan = plan_question(request.questions["routing"])
        self.assertEqual(plan.option_names, ("true", "false"))
        self.assertTrue(
            build_prompt(request.state, plan, ["A", "B"]).endswith(
                "A: yes: Now\nB: no: Can wait"
            )
        )

    def test_invalid_label_and_option_mappings_fail(self):
        plan = plan_question(request_for(choice_question()).questions["routing"])
        for labels, order in (
            (["A"], None),
            (["A", "A"], None),
            (["A", "B"], [0, 0]),
            (["A", "B"], [0, True]),
        ):
            with (
                self.subTest(labels=labels, order=order),
                self.assertRaises(ValueError),
            ):
                build_prompt("state", plan, labels, order)


class TestNativePromptWording(unittest.TestCase):
    """wording="native" must match the jevbench-hard runner's own contract
    (run_suites_rotation.py: PREFIX/format_prompt/option_texts) byte-for-byte;
    ``runner_format_prompt`` above is an independent port used as the oracle."""

    def test_matches_runner_for_choice_across_rotations_with_json_state(self):
        request = request_for(
            choice_question(
                criteria={"billing": "Payment issue", "technical": None, "sales": ""}
            ),
            state={"ticket": [7, "안녕하세요", None], "z": 1},
        )
        plan = plan_question(request.questions["routing"])
        labels = CANONICAL_LETTERS[:3]
        raw_ordered = [
            ("billing", "Payment issue"),
            ("technical", None),
            ("sales", ""),
        ]
        for rotation in range(3):
            order = rotation_order(3, rotation)
            mine = build_prompt(request.state, plan, labels, order, wording="native")
            theirs = runner_format_prompt(
                request.state,
                plan.instructions,
                [raw_ordered[i] for i in order],
            )
            with self.subTest(rotation=rotation):
                self.assertEqual(mine, theirs)
        self.assertTrue(mine.startswith(NATIVE_PROMPT_PREFIX + "\n\nState:\n"))
        self.assertIn("\n\nQuestion:\n", mine)

    def test_matches_runner_for_noul_and_score_with_string_state(self):
        noul_plan = plan_question(
            request_for(
                {
                    "type": "noul",
                    "instructions": "Urgent?",
                    "criteria": {"true": "Now", "false": "Can wait"},
                }
            ).questions["routing"]
        )
        mine = build_prompt(
            "plain state text", noul_plan, ["A", "B"], None, wording="native"
        )
        theirs = runner_format_prompt(
            "plain state text", "Urgent?", [("yes", "Now"), ("no", "Can wait")]
        )
        self.assertEqual(mine, theirs)

        score_plan = plan_question(
            request_for(
                {"type": "score", "instructions": "Rate", "criteria": ["low", "medium", "high"]}
            ).questions["routing"]
        )
        mine = build_prompt("s", score_plan, ["A", "B", "C"], None, wording="native")
        theirs = runner_format_prompt(
            "s", "Rate", [("0", "low"), ("1", "medium"), ("2", "high")]
        )
        self.assertEqual(mine, theirs)

    def test_falsy_description_is_omitted_like_the_runner(self):
        # The runner's `f": {description}" if description else ""` treats an
        # empty string the same as no description at all (a truthiness check,
        # not `is not None`) -- ported deliberately, not "fixed".
        plan = plan_question(
            request_for(choice_question(criteria={"a": "", "b": None})).questions[
                "routing"
            ]
        )
        prompt = build_prompt("s", plan, ["A", "B"], None, wording="native")
        self.assertTrue(prompt.endswith("A. a\nB. b"))

    def test_rejects_noncanonical_or_out_of_order_labels(self):
        plan = plan_question(request_for(choice_question()).questions["routing"])
        for labels in (["X", "Y"], ["B", "A"], ["A", "B", "C"]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                build_prompt("s", plan, labels, None, wording="native")

    def test_rejects_more_than_26_options(self):
        criteria = {f"opt{i}": None for i in range(27)}
        plan = plan_question(
            request_for(choice_question(criteria=criteria)).questions["routing"]
        )
        with self.assertRaises(ValueError):
            build_prompt("s", plan, CANONICAL_LETTERS[:26] + ("AA",), None, wording="native")

    def test_unknown_wording_rejected(self):
        plan = plan_question(request_for(choice_question()).questions["routing"])
        with self.assertRaises(ValueError):
            build_prompt("s", plan, ["A", "B"], None, wording="bogus")

    def test_served_wording_is_unchanged_default(self):
        plan = plan_question(request_for(choice_question()).questions["routing"])
        explicit = build_prompt("s", plan, ["A", "B"], None, wording="served")
        implicit = build_prompt("s", plan, ["A", "B"], None)
        self.assertEqual(explicit, implicit)
        self.assertTrue(implicit.startswith("Context:\n"))


class TestSystemOneProbabilityReduction(unittest.TestCase):
    def setUp(self):
        self.plan = plan_question(request_for(choice_question()).questions["routing"])

    def test_softmax_is_shift_invariant_and_extreme_safe(self):
        expected = probabilities_from_logprobs([-1, -3], temperature=2)
        actual = probabilities_from_logprobs([-1001, -1003], temperature=2)
        for left, right in zip(expected, actual):
            self.assertAlmostEqual(left, right)
        self.assertEqual(probabilities_from_logprobs([-1e308, 1e308], 1e-300), [0, 1])

    def test_temperature_scaling_can_be_disabled(self):
        vector = [-1, -3]
        normal = reduce_probabilities(self.plan, [vector], [[0, 1]], SystemOneOptions())
        disabled = reduce_probabilities(
            self.plan,
            [vector],
            [[0, 1]],
            SystemOneOptions(temperature=100, temperature_scaling=False),
        )
        softened = probabilities_from_logprobs(vector, temperature=100)
        self.assertEqual(normal, disabled)
        self.assertLess(softened[0], normal["probabilities"]["billing"])

    def test_rotation_is_undone_before_probability_average(self):
        answer = reduce_probabilities(
            self.plan,
            [[math.log(0.8), math.log(0.2)], [math.log(0.7), math.log(0.3)]],
            [[0, 1], [1, 0]],
            SystemOneOptions(permutations=2, return_logprobs=True),
        )
        self.assertAlmostEqual(answer["probabilities"]["billing"], 0.55)
        self.assertAlmostEqual(answer["probabilities"]["technical"], 0.45)
        self.assertEqual(answer["choice"], "billing")
        self.assertEqual(
            answer["logprobs"][1],
            {"billing": math.log(0.3), "technical": math.log(0.7)},
        )

    def test_confidence_uniform_and_certain(self):
        uniform = reduce_probabilities(
            self.plan, [[-5, -5]], [[0, 1]], SystemOneOptions()
        )
        certain = reduce_probabilities(
            self.plan, [[0, -1000]], [[0, 1]], SystemOneOptions()
        )
        self.assertAlmostEqual(uniform["confidence"], 0)
        self.assertEqual(certain["confidence"], 1)
        self.assertEqual(uniform["choice"], "billing")  # Stable input-order tie break.

    def test_score_expectation_structured_legend_and_entropy(self):
        criteria = [{"severity": "low", "examples": ["A"]}, ["medium"], None]
        plan = plan_question(
            request_for(
                {
                    "type": "score",
                    "instructions": {"task": "rate"},
                    "criteria": criteria,
                }
            ).questions["routing"]
        )
        answer = reduce_probabilities(
            plan,
            [[math.log(0.25), math.log(0.25), math.log(0.5)]],
            [[0, 1, 2]],
            SystemOneOptions(),
        )
        self.assertAlmostEqual(answer["score"], 1.25)
        self.assertEqual(
            answer["legend"], {"0": criteria[0], "1": criteria[1], "2": None}
        )
        expected_entropy = -(0.5 * math.log(0.25) + 0.5 * math.log(0.5))
        self.assertAlmostEqual(answer["confidence"], 1 - expected_entropy / math.log(3))

    def test_noul_returns_probability_of_true_after_rotation(self):
        plan = plan_question(
            request_for({"type": "noul", "instructions": "Yes?"}).questions["routing"]
        )
        answer = reduce_probabilities(
            plan, [[math.log(0.1), math.log(0.9)]], [[1, 0]], SystemOneOptions()
        )
        self.assertAlmostEqual(answer["noul"], 0.9)
        self.assertEqual(set(answer), {"type", "noul"})

    def test_missing_nonfinite_or_invalid_engine_output_fails(self):
        for values in (
            [None, -1],
            [float("nan"), -1],
            [float("inf"), -1],
            [float("-inf"), -1],
            ["0", -1],
            [True, -1],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                reduce_probabilities(self.plan, [values], [[0, 1]], SystemOneOptions())
        for vectors, orders in (
            ([], []),
            ([[-1]], [[0, 1]]),
            ([[-1, -2]], []),
            ([[-1, -2]], [[0, 0]]),
        ):
            with (
                self.subTest(vectors=vectors, orders=orders),
                self.assertRaises(ValueError),
            ):
                reduce_probabilities(self.plan, vectors, orders, SystemOneOptions())


if __name__ == "__main__":
    unittest.main()
