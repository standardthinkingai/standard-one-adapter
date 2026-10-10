"""Check JevBench conversion, reference alignment and frozen source integrity."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from jev_adapter.benchmarks.data import sha256
from jev_adapter.benchmarks.jevbench import (
    JEVBENCH_COMMIT,
    JEVBENCH_SUITES,
    JevBenchSpec,
    normalize_jevbench_record,
    parse_jevbench,
    prepare_jevbench,
)


def fixture(kind="choice", name="sample"):
    questions = {
        "choice": {
            "type": "choice",
            "instructions": "Choose a queue.",
            "criteria": {"support": "Support", "billing": "Billing"},
        },
        "noul": {
            "type": "noul",
            "instructions": "Is a refund requested?",
            "criteria": {"true": "Refund requested", "false": "No refund request"},
        },
        "score": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["low", "medium", "high"],
        },
    }
    return {
        "id": name,
        "family": "routing",
        "group": None,
        "split": "public",
        "state": {"message": "Please refund my bill."},
        "question": questions[kind],
        "labels": {
            "choice": ["billing", "support"],
            "noul": ["no", "yes"],
            "score": ["0", "1", "2"],
        }[kind],
        "expected": {"choice": "billing", "noul": "yes", "score": 2}[kind],
        "provenance": {
            "source": "Unit test fixture",
            "license": "MIT",
            "rationale": "SECRET_RATIONALE",
            "surface_answer": "SECRET_SURFACE_ANSWER",
            "label_basis": "Authored rubric",
        },
    }


def source_fixture(records):
    payload = "".join(json.dumps(row) + "\n" for row in records).encode()
    notices = {"LICENSE": b"Fixture MIT notice", "THIRD-PARTY.md": b"Fixture notice"}
    spec = JevBenchSpec(
        "datasets/public/fixture.jsonl", sha256(payload), len(records), "Fixture"
    )
    return (
        spec,
        {spec.path: payload, **notices},
        {path: sha256(content) for path, content in notices.items()},
    )


class TestJevBenchNormalization(unittest.TestCase):
    def test_native_question_and_canonical_label_orders_preserved_without_gold(self):
        raw = fixture()
        raw["provenance"]["gold_probs"] = {"support": 0.25, "billing": 0.75}
        original = copy.deepcopy(raw)
        row = normalize_jevbench_record(raw, "jevbench-hard")
        self.assertEqual(raw, original)
        self.assertEqual(
            row["record"],
            {"state": raw["state"], "questions": {"decision": raw["question"]}},
        )
        expected = row["expected"]["decision"]
        self.assertEqual(expected["labels"], ["billing", "support"])
        self.assertEqual(
            list(row["record"]["questions"]["decision"]["criteria"]),
            ["support", "billing"],
        )
        self.assertEqual(expected["target"], [1, 0])
        self.assertEqual(expected["reference_probs"], [0.75, 0.25])
        request_text = json.dumps(row["record"])
        for secret in (
            "SECRET_",
            "gold_probs",
            "expected",
            "provenance",
            "label_basis",
        ):
            self.assertNotIn(secret, request_text)
        self.assertEqual(row["metadata"]["provenance"], raw["provenance"])
        self.assertEqual(row["metadata"]["group_id"], raw["id"])
        self.assertEqual(row["split"], "public")

    def test_noul_reference_mapping_and_score_hard_label(self):
        raw = fixture("noul")
        raw["provenance"]["gold_probs"] = {"yes": 0.8, "no": 0.2}
        noul = normalize_jevbench_record(raw, "jevbench-hard")["expected"]["decision"]
        self.assertEqual(noul["labels"], ["false", "true"])
        self.assertEqual(noul["target"], [0, 1])
        self.assertEqual(noul["reference_probs"], [0.2, 0.8])
        score = normalize_jevbench_record(fixture("score"), "jevbench-original")
        self.assertEqual(score["expected"]["decision"]["label"], 2)
        self.assertNotIn("reference_probs", score["expected"]["decision"])

    def test_paraphrase_group_preserved_and_null_groups_independent(self):
        first, second = fixture(name="one"), fixture(name="two")
        a = normalize_jevbench_record(first, "jevbench-easy")
        b = normalize_jevbench_record(second, "jevbench-easy")
        self.assertNotEqual(a["metadata"]["group_id"], b["metadata"]["group_id"])
        first["group"] = second["group"] = "pair"
        self.assertEqual(
            normalize_jevbench_record(first, "jevbench-original")["metadata"][
                "group_id"
            ],
            normalize_jevbench_record(second, "jevbench-original")["metadata"][
                "group_id"
            ],
        )

    def test_bad_labels_private_rows_and_malformed_references_rejected(self):
        changes = [
            {"split": "private"},
            {"labels": ["billing", "absent"]},
            {"expected": "absent"},
            {"provenance": {"exclude_reason": "ambiguous"}},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_jevbench_record({**fixture(), **change}, "jevbench-hard")
        for probs in (
            {"billing": 0.5},
            {"billing": 0.8, "support": 0.3},
            {"billing": float("nan"), "support": 0.3},
            {"billing": True, "support": 0.0},
        ):
            raw = fixture()
            raw["provenance"]["gold_probs"] = probs
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                normalize_jevbench_record(raw, "jevbench-hard")
        for kind, gold in (("score", True), ("score", 3), ("noul", True)):
            raw = fixture(kind)
            raw["expected"] = gold
            with self.subTest(kind=kind, gold=gold), self.assertRaises(ValueError):
                normalize_jevbench_record(raw, "jevbench-hard")

    def test_duplicate_and_blank_rows_rejected(self):
        row = json.dumps(fixture()) + "\n"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_jevbench((row * 2).encode(), "jevbench-original")
        with self.assertRaisesRegex(ValueError, "blank"):
            parse_jevbench((row + "\n").encode(), "jevbench-original")


class TestJevBenchPreparation(unittest.TestCase):
    def test_pinned_download_preserves_notices_and_is_idempotent(self):
        spec, sources, notices = source_fixture(
            [fixture(name="one"), fixture(name="two")]
        )
        requests = []

        def transport(request):
            requests.append(request.url.path)
            prefix = f"/fstandhartinger/jevbench/{JEVBENCH_COMMIT}/"
            self.assertTrue(request.url.path.startswith(prefix))
            return httpx.Response(
                200, content=sources[request.url.path.removeprefix(prefix)]
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(JEVBENCH_SUITES, {"jevbench-fixture": spec}),
            patch("jev_adapter.benchmarks.jevbench.JEVBENCH_NOTICES", notices),
            httpx.Client(transport=httpx.MockTransport(transport)) as client,
        ):
            output = Path(tmp)
            result = prepare_jevbench(
                "jevbench-fixture", output, client=client, limit=1
            )
            self.assertEqual(
                result,
                prepare_jevbench("jevbench-fixture", output, client=client, limit=1),
            )
            self.assertEqual(result["full_partition"]["questions"], 2)
            self.assertEqual(result["selected"]["questions"], 1)
            self.assertFalse(result["selection"]["is_full_partition"])
            directory = output / "jevbench-fixture"
            self.assertEqual(
                result["data_sha256"], sha256((directory / "public.jsonl").read_bytes())
            )
            for name in notices:
                self.assertEqual((directory / name).read_bytes(), sources[name])
            self.assertTrue(
                all("private" not in path and "train" not in path for path in requests)
            )
            with self.assertRaises(FileExistsError):
                prepare_jevbench("jevbench-fixture", output, client=client)
            self.assertEqual(
                result["data_sha256"], sha256((directory / "public.jsonl").read_bytes())
            )

    def test_bad_data_notice_or_count_fails_before_writing(self):
        spec, sources, notices = source_fixture([fixture()])
        for corrupt in (spec.path, "LICENSE", "THIRD-PARTY.md", "count"):
            with (
                self.subTest(corrupt=corrupt),
                tempfile.TemporaryDirectory() as tmp,
                patch.dict(
                    JEVBENCH_SUITES,
                    {
                        "jevbench-fixture": replace(spec, records=2)
                        if corrupt == "count"
                        else spec
                    },
                ),
                patch("jev_adapter.benchmarks.jevbench.JEVBENCH_NOTICES", notices),
            ):
                root, output = Path(tmp) / "source", Path(tmp) / "out"
                for name, content in sources.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content + (b" " if name == corrupt else b""))
                with self.assertRaisesRegex(
                    ValueError, "SHA256 mismatch|count mismatch"
                ):
                    prepare_jevbench(
                        "jevbench-fixture", output, source_root=root, limit=1
                    )
                self.assertFalse(output.exists())

    def test_changed_notice_is_not_partially_overwritten(self):
        spec, sources, notices = source_fixture([fixture()])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(JEVBENCH_SUITES, {"jevbench-fixture": spec}),
            patch("jev_adapter.benchmarks.jevbench.JEVBENCH_NOTICES", notices),
        ):
            root, output = Path(tmp) / "source", Path(tmp) / "out"
            for name, content in sources.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            directory = output / "jevbench-fixture"
            directory.mkdir(parents=True)
            (directory / "LICENSE").write_bytes(b"Existing different license")
            with self.assertRaises(FileExistsError):
                prepare_jevbench("jevbench-fixture", output, source_root=root)
            self.assertFalse((directory / "public.jsonl").exists())
            self.assertEqual(
                (directory / "LICENSE").read_bytes(), b"Existing different license"
            )


if __name__ == "__main__":
    unittest.main()
