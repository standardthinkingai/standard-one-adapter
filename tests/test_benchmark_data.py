"""Check benchmark integrity and prevent labels from entering inference requests."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from jev_adapter.benchmarks.data import (
    SuiteSpec,
    normalize_record,
    parse_partition,
    population_counts,
    sha256,
)
from jev_adapter.benchmarks.prepare import prepare_suite


def fixture_record(name="example/1"):
    return {
        "state": {"message": "Needs a refund", "count": 4},
        "questions": {
            "queue": {
                "type": "choice",
                "instructions": "Choose the queue.",
                "criteria": {"other": None, "billing": "Refunds"},
                "label": "billing",
                "src": "routing",
            },
            "urgent": {
                "type": "noul",
                "instructions": "Is the message urgent?",
                "label": False,
                "src": "urgency",
            },
            "priority": {
                "type": "score",
                "instructions": "Select priority.",
                "criteria": ["low", "medium", "high"],
                "label": 2,
                "src": "priority",
            },
        },
        "_meta": {
            "id": name,
            "source": "example",
            "variant": "clean",
            "group_id": "pair-1",
            "pair_id": "pair-1",
            "sibling": "a",
        },
    }


def fake_source(records):
    payload = ("".join(json.dumps(r) + "\n" for r in records)).encode()
    manifest = json.dumps(
        {
            "files": {
                name: {
                    "sha256": sha256(payload),
                    "records": len(records),
                    "questions": sum(len(r["questions"]) for r in records),
                }
                for name in ("development.jsonl", "test.jsonl")
            },
            "dataset_revisions": {},
        }
    ).encode()
    spec = SuiteSpec("evals/example", sha256(manifest), "Test fixture")
    return spec, manifest, payload


class TestBenchmarkNormalization(unittest.TestCase):
    def test_all_question_types_preserve_order_and_hide_gold(self):
        original = fixture_record()
        before = copy.deepcopy(original)
        row = normalize_record(original, "example", "development")
        self.assertEqual(original, before)
        self.assertEqual(
            list(row["record"]["questions"]), ["queue", "urgent", "priority"]
        )
        self.assertNotIn("model", row["record"])
        self.assertNotIn("_meta", row["record"])
        for q in row["record"]["questions"].values():
            self.assertNotIn("label", q)
            self.assertNotIn("src", q)
        self.assertEqual(row["expected"]["queue"]["labels"], ["other", "billing"])
        self.assertEqual(row["expected"]["queue"]["target"], [0, 1])
        self.assertEqual(row["expected"]["urgent"]["labels"], ["false", "true"])
        self.assertEqual(row["expected"]["urgent"]["target"], [1, 0])
        self.assertEqual(row["expected"]["priority"]["target"], [0, 0, 1])
        self.assertIsNone(row["record"]["questions"]["queue"]["criteria"]["other"])
        self.assertEqual(row["metadata"]["pair_id"], "pair-1")

    def test_large_choice_space_is_not_truncated(self):
        raw = fixture_record()
        raw["questions"] = {
            "intent": {
                "type": "choice",
                "instructions": "Which intent?",
                "criteria": {f"intent_{i}": None for i in range(78)},
                "label": "intent_77",
            }
        }
        row = normalize_record(raw, "example", "development")
        self.assertEqual(len(row["expected"]["intent"]["labels"]), 78)
        self.assertEqual(row["expected"]["intent"]["label"], 77)

    def test_duplicate_ids_and_malformed_targets_fail(self):
        raw = fixture_record()
        payload = (json.dumps(raw) + "\n") * 2
        with self.assertRaisesRegex(ValueError, "duplicate record"):
            parse_partition(payload.encode(), "example", "development")
        for question, label in (
            ("queue", "absent"),
            ("urgent", "false"),
            ("priority", 9),
        ):
            modified = copy.deepcopy(raw)
            modified["questions"][question]["label"] = label
            with self.subTest(question=question), self.assertRaises(ValueError):
                normalize_record(modified, "example", "development")

    def test_population_separates_perturbations_and_unknowable(self):
        raws = [fixture_record(str(i)) for i in range(3)]
        raws[1]["_meta"]["variant"] = "permuted"
        raws[2]["_meta"]["source"] = "unknowable"
        counts = population_counts(
            [normalize_record(raw, "example", "development") for raw in raws]
        )
        self.assertEqual(counts["records"], 3)
        self.assertEqual(counts["questions"], 9)
        self.assertEqual(counts["clean_questions"], 6)
        self.assertEqual(counts["headline_questions"], 3)


class TestBenchmarkPreparation(unittest.TestCase):
    def test_verified_download_subset_and_idempotency(self):
        spec, manifest, payload = fake_source(
            [fixture_record("one"), fixture_record("two")]
        )
        fetched = []

        def transport(request):
            fetched.append(request.url.path)
            return httpx.Response(
                200,
                content=manifest
                if request.url.path.endswith("manifest.json")
                else payload,
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict("jev_adapter.benchmarks.prepare.SUITES", {"example": spec}),
            httpx.Client(transport=httpx.MockTransport(transport)) as client,
        ):
            out = Path(tmp)
            result = prepare_suite(
                "example", "development", out, client=client, limit=1
            )
            again = prepare_suite("example", "development", out, client=client, limit=1)
            self.assertEqual(result, again)
            self.assertFalse(result["selection"]["is_full_partition"])
            self.assertEqual(result["full_partition"]["questions"], 6)
            self.assertEqual(result["selected"]["questions"], 3)
            self.assertEqual(
                result["data_sha256"],
                sha256((out / "example/development.jsonl").read_bytes()),
            )
            self.assertTrue(all("train" not in path for path in fetched))
            with self.assertRaises(FileExistsError):
                prepare_suite("example", "development", out, client=client)

    def test_tampered_partition_rejected_without_writing(self):
        spec, manifest, payload = fake_source([fixture_record()])

        def transport(request):
            return httpx.Response(
                200,
                content=manifest
                if request.url.path.endswith("manifest.json")
                else payload + b" ",
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict("jev_adapter.benchmarks.prepare.SUITES", {"example": spec}),
            httpx.Client(transport=httpx.MockTransport(transport)) as client,
        ):
            out = Path(tmp)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                prepare_suite("example", "development", out, client=client)
            self.assertFalse((out / "example").exists())

    def test_offline_source_has_same_integrity_and_test_gate(self):
        spec, manifest, payload = fake_source([fixture_record()])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict("jev_adapter.benchmarks.prepare.SUITES", {"example": spec}),
        ):
            root = Path(tmp) / "source"
            directory = root / "evals/example"
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_bytes(manifest)
            (directory / "test.jsonl").write_bytes(payload)
            out = Path(tmp) / "out"
            with self.assertRaisesRegex(ValueError, "locked test"):
                prepare_suite("example", "test", out, source_root=root)
            result = prepare_suite(
                "example", "test", out, source_root=root, allow_test=True
            )
            self.assertTrue(result["protocol"]["locked_test"])
            (directory / "manifest.json").write_bytes(manifest + b" ")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                prepare_suite("example", "test", out, source_root=root, allow_test=True)
