"""Normalize pinned, public KEV evaluation records without changing their tasks.

Only the request is sent to a model. Labels and source metadata remain outside
that request, in a separate expected/metadata envelope used by the evaluator.
This module is an independent format conversion, not imported KEV model code.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

KEV_COMMIT = "4f8110a3f8620cc3a182ae9a708e4398492c4b1a"
KEV_REPOSITORY = "https://github.com/jaredpalmer/kev"
KEV_RAW = f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_COMMIT}"
DEFAULT_SUITES = ("decision-v7", "transfer-v4", "transfer-v9")


@dataclass(frozen=True)
class SuiteSpec:
    path: str
    manifest_sha256: str
    description: str
    notes: tuple[str, ...] = ()


SUITES = {
    "decision-v7": SuiteSpec(
        "evals/v7/decision-v7",
        "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2",
        "Ten public sources and generated policies; KEV trained-source evaluation.",
        ("Includes up to 78 choices with none-of-the-above variants.",),
    ),
    "transfer-v4": SuiteSpec(
        "evals/v4/transfer-v4",
        "31677c2256b406222e7d94ffdc0a02a70ce05746b9efe307876024c4e77291d1",
        "Six sources unseen in KEV fine-tuning and held-out policy structures.",
        ("Unseen means unseen in KEV fine-tuning, not in base-model pretraining.",),
    ),
    "transfer-v9": SuiteSpec(
        "evals/v9/transfer-v9",
        "3c4f0be94509a3612678bfd3a30fd99a8d0ca3c47ddfe7318075d95b2fa365e4",
        "Transfer-v4 plus MMLU-Pro, buried evidence and unknowable/control pairs.",
        (
            "Contains transfer-v4 records; do not pool both suites as independent data.",
            "Source 'unknowable' is evaluated for confidence, not accuracy.",
        ),
    ),
    "semif-v1": SuiteSpec(
        "evals/external/semif-v1",
        "0de05eac16b0ddeeb2719c50a94a9148d6ae195f66303aec74aa103a3845ad11",
        "SemIf's 144 authored choices plus 108 perturbations, frozen by KEV.",
        (
            "Already evaluated by KEV; this is not an additional independent suite.",
            "SemIf's own headline is mean family balanced accuracy.",
        ),
    ),
    "scienthoon-v1": SuiteSpec(
        "evals/external/scienthoon-v1",
        "ef31183425bf9d3c2d8ac5d245a14d30fa44d1e531c945e4405edcb1f75ae0d5",
        "KEV's frozen conversion of scienthoon's synthetic support tickets.",
        (
            "Already evaluated by KEV; this is not an additional independent suite.",
            "Original 900 question rows become 291 unique states / 873 questions in this conversion.",
            "Priority labels depend on an organizational rule absent from the state; report separately.",
        ),
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_sha256(data: bytes, expected: str, name: str) -> None:
    actual = sha256(data)
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for {name}: expected {expected}, got {actual}"
        )


def normalize_record(raw: dict[str, Any], suite: str, split: str) -> dict[str, Any]:
    """Retain option order, all sibling questions and KEV's original provenance."""
    meta = copy.deepcopy(raw["_meta"])
    if not isinstance(meta.get("id"), str) or not meta["id"]:
        raise ValueError("record requires a nonempty _meta.id")
    questions, expected = {}, {}
    for qid, question in raw["questions"].items():
        kind = question["type"]
        if kind == "choice":
            labels = list(question["criteria"])
            try:
                label = labels.index(question["label"])
            except ValueError as exc:
                raise ValueError(f"{meta['id']}/{qid}: label is not an option") from exc
        elif kind == "noul":
            if type(question["label"]) is not bool:
                raise ValueError(f"{meta['id']}/{qid}: noul label must be a boolean")
            labels, label = ["false", "true"], int(question["label"])
        elif kind == "score":
            labels = [str(i) for i in range(len(question["criteria"]))]
            label = question["label"]
            if type(label) is not int or not 0 <= label < len(labels):
                raise ValueError(f"{meta['id']}/{qid}: score label is out of range")
        else:
            raise ValueError(f"unsupported question type: {kind}")
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError(f"{meta['id']}/{qid}: invalid option labels")
        questions[qid] = {
            key: copy.deepcopy(question[key])
            for key in ("type", "instructions", "criteria")
            if key in question
        }
        expected[qid] = {
            "labels": labels,
            "target": [float(i == label) for i in range(len(labels))],
            "label": label,
            "type": kind,
            "task": question.get("src", meta["source"]),
        }
    if not questions:
        raise ValueError(f"{meta['id']}: no questions")
    record = {"state": copy.deepcopy(raw["state"]), "questions": questions}
    for key in ("images", "options"):
        if key in raw:
            record[key] = copy.deepcopy(raw[key])
    return {
        "id": meta["id"],
        "suite": suite,
        "split": split,
        "source": meta["source"],
        "variant": meta.get("variant", "clean"),
        "record": record,
        "expected": expected,
        "metadata": meta,
    }


def parse_partition(data: bytes, suite: str, split: str) -> list[dict[str, Any]]:
    records, seen = [], set()
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{suite}/{split}:{number}: blank JSONL record")
        row = normalize_record(json.loads(line), suite, split)
        if row["id"] in seen:
            raise ValueError(f"duplicate record id: {row['id']}")
        seen.add(row["id"])
        records.append(row)
    return records


def population_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [row for row in records if row["variant"] == "clean"]
    return {
        "records": len(records),
        "questions": sum(len(row["expected"]) for row in records),
        "clean_records": len(clean),
        "clean_questions": sum(len(row["expected"]) for row in clean),
        "headline_questions": sum(
            len(row["expected"]) for row in clean if row["source"] != "unknowable"
        ),
        "variants": dict(Counter(row["variant"] for row in records)),
        "clean_sources": dict(Counter(row["source"] for row in clean)),
        "question_types": dict(
            Counter(q["type"] for row in records for q in row["expected"].values())
        ),
        "maximum_options": max(
            (len(q["labels"]) for row in records for q in row["expected"].values()),
            default=0,
        ),
    }


def source_provenance(
    records: list[dict[str, Any]], upstream_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Point to underlying dataset licenses; do not relicense mixed source data."""
    datasets = {}
    for row in records:
        meta = row["metadata"]
        repo = meta.get("repo")
        if not repo:
            continue
        revision = meta.get("revision")
        key = (repo, revision)
        external = upstream_manifest.get("external", {})
        is_external = external.get("repo", "").endswith("/" + repo)
        datasets[key] = {
            "repository": repo,
            "revision": revision,
            "source_url": (
                external["repo"]
                if is_external
                else f"https://huggingface.co/datasets/{repo}"
            ),
            "license": external.get("license")
            if is_external
            else "see upstream dataset",
        }
    return {
        "kev_repository_license": "Apache-2.0",
        "kev_license_url": f"{KEV_REPOSITORY}/blob/{KEV_COMMIT}/LICENSE",
        "dataset_notice": (
            "Public and downloadable does not mean all source datasets share Apache-2.0. "
            "Their individual licenses and attribution terms continue to apply."
        ),
        "datasets": list(datasets.values()),
        "external": upstream_manifest.get("external"),
        "dataset_revisions_from_manifest": upstream_manifest.get(
            "dataset_revisions", {}
        ),
    }
