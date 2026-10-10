"""Prepare JevBench's three pinned public tiers without exposing answer metadata.

This is an independent format conversion. The public 231 questions are not the
534-question leaderboard, whose remaining questions are private or untracked.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from jev_adapter.protocol import SystemOneRequest

from .data import normalize_record, population_counts, require_sha256, sha256

JEVBENCH_COMMIT = "fd51755eb0c0b546ca206d764faf3302feca913e"
JEVBENCH_REPOSITORY = "https://github.com/fstandhartinger/jevbench"
JEVBENCH_RAW = (
    f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{JEVBENCH_COMMIT}"
)
JEVBENCH_NOTICES = {
    "LICENSE": "3e5beed774bb0bcbfb2fcf24ba9554212c6ae112937d308040989465ab0c5784",
    "THIRD-PARTY.md": "396422e29055bba4073f4a9e5a7163cc724f13e970a34ef58e8ea1fb439b9b38",
}


@dataclass(frozen=True)
class JevBenchSpec:
    path: str
    sha256: str
    records: int
    description: str


JEVBENCH_SUITES = {
    "jevbench-original": JevBenchSpec(
        "datasets/public/original.jsonl",
        "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
        72,
        "Public original tier: 72 decisions in 36 paraphrase pairs.",
    ),
    "jevbench-easy": JevBenchSpec(
        "datasets/public/easy.jsonl",
        "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
        48,
        "Public easy tier: 48 explicit facts, intents and tool selections.",
    ),
    "jevbench-hard": JevBenchSpec(
        "datasets/public/hard.jsonl",
        "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
        111,
        "Public hard tier: 111 decisions; ten supply probability references.",
    ),
}


def normalize_jevbench_record(raw: dict[str, Any], suite: str) -> dict[str, Any]:
    """Keep native question/option order and canonical scoring label order."""
    if raw.get("split") != "public":
        raise ValueError("JevBench converter accepts only the public subset")
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("JevBench record requires a nonempty id")
    labels = raw["labels"]
    if (
        not isinstance(labels, list)
        or len(labels) < 2
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(f"{raw['id']}: invalid canonical labels")
    question = raw["question"]
    kind, gold = question["type"], raw["expected"]
    if kind == "choice":
        if not isinstance(question["criteria"], dict) or set(labels) != set(
            question["criteria"]
        ):
            raise ValueError(f"{raw['id']}: choice criteria/labels mismatch")
        if not isinstance(gold, str) or gold not in labels:
            raise ValueError(f"{raw['id']}: invalid choice expected label")
        canonical_labels, canonical_gold = list(labels), gold
    elif kind == "noul":
        if labels != ["no", "yes"] or gold not in ("no", "yes"):
            raise ValueError(f"{raw['id']}: noul requires no/yes canonical labels")
        canonical_labels = ["false", "true"]
        canonical_gold, gold = ("true" if gold == "yes" else "false"), gold == "yes"
    elif kind == "score":
        if not isinstance(question["criteria"], list) or labels != [
            str(i) for i in range(len(question["criteria"]))
        ]:
            raise ValueError(f"{raw['id']}: score criteria/labels mismatch")
        if type(gold) is not int or not 0 <= gold < len(labels):
            raise ValueError(f"{raw['id']}: invalid score expected label")
        canonical_labels, canonical_gold = list(labels), str(gold)
    else:
        raise ValueError(f"unsupported JevBench question type: {kind}")
    provenance = copy.deepcopy(raw["provenance"])
    reference = provenance.get("gold_probs")
    if provenance.get("exclude_reason"):
        raise ValueError(f"{raw['id']}: excluded upstream item is not evaluable")
    family = raw["family"]
    if not isinstance(family, str) or not family:
        raise ValueError(f"{raw['id']}: missing task family")
    # Only these three native question fields are allowed into the request.
    clean_question = {
        key: copy.deepcopy(question[key])
        for key in ("type", "instructions", "criteria")
        if key in question
    }
    clean_question["label"] = gold
    row = normalize_record(
        {
            "state": raw["state"],
            "questions": {"decision": clean_question},
            "_meta": {
                "id": raw["id"],
                "source": family,
                "variant": "clean",
                "group_id": raw.get("group") or raw["id"],
                "upstream_group": raw.get("group"),
                "canonical_labels": list(labels),
                "provenance": provenance,
                "gold_policy": {
                    "hard_label": "authored_reviewed_rubric",
                    "reference_probs": (
                        "countable_mathematical_probability"
                        if reference is not None
                        else None
                    ),
                    "argmax_tie_break": "lexicographic_label",
                },
            },
        },
        suite,
        "public",
    )
    expected = row["expected"]["decision"]
    expected["labels"] = canonical_labels
    expected["label"] = canonical_labels.index(canonical_gold)
    expected["target"] = [float(label == canonical_gold) for label in canonical_labels]
    if reference is not None:
        if not isinstance(reference, dict) or set(reference) != set(labels):
            raise ValueError(f"{raw['id']}: reference probability labels mismatch")
        probabilities = [reference[label] for label in labels]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in probabilities
        ) or not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{raw['id']}: invalid reference probabilities")
        expected["reference_probs"] = [float(value) for value in probabilities]
    SystemOneRequest.model_validate({"model": "validation-only", **row["record"]})
    return row


def parse_jevbench(data: bytes, suite: str) -> list[dict[str, Any]]:
    records, seen = [], set()
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{suite}/public:{number}: blank JSONL record")
        row = normalize_jevbench_record(json.loads(line), suite)
        if row["id"] in seen:
            raise ValueError(f"duplicate record id: {row['id']}")
        seen.add(row["id"])
        records.append(row)
    return records


def prepare_jevbench(
    suite: str,
    output: Path,
    *,
    client: httpx.Client | None = None,
    source_root: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Verify all source bytes before writing an immutable prepared public tier."""
    if suite not in JEVBENCH_SUITES:
        raise ValueError(f"unknown JevBench suite: {suite}")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("--limit must be positive")

    def read_source(path: str) -> bytes:
        if source_root is not None:
            return (source_root / path).read_bytes()
        if client is None:
            raise ValueError("an HTTP client or --source-root is required")
        response = client.get(f"{JEVBENCH_RAW}/{path}")
        response.raise_for_status()
        return response.content

    spec = JEVBENCH_SUITES[suite]
    raw_bytes = read_source(spec.path)
    require_sha256(raw_bytes, spec.sha256, spec.path)
    notices = {}
    for path, digest in JEVBENCH_NOTICES.items():
        content = read_source(path)
        require_sha256(content, digest, path)
        notices[path] = content
    records = parse_jevbench(raw_bytes, suite)
    counts = population_counts(records)
    if counts["records"] != spec.records or counts["questions"] != spec.records:
        raise ValueError(f"upstream record/question count mismatch: {suite}/public")
    selected = records if limit is None else records[:limit]
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in selected
    ).encode("utf-8")
    full = len(selected) == len(records)
    manifest = {
        "schema_version": 1,
        "suite": suite,
        "split": "public",
        "description": spec.description,
        "data_file": "public.jsonl",
        "data_sha256": sha256(serialized),
        "upstream": {
            "repository": JEVBENCH_REPOSITORY,
            "commit": JEVBENCH_COMMIT,
            "path": spec.path,
            "url": f"{JEVBENCH_RAW}/{spec.path}",
            "sha256": spec.sha256,
            "notice_sha256": dict(JEVBENCH_NOTICES),
        },
        "full_partition": counts,
        "selected": population_counts(selected),
        "selection": {
            "method": "full" if full else "prefix",
            "requested_limit": limit,
            "is_full_partition": full,
            "note": (
                "Full frozen public tier; not the complete published leaderboard."
                if full
                else "Smoke subset only; not a full-tier result. Pairs may be incomplete."
            ),
        },
        "protocol": {
            "calibration_applied": False,
            "training_data_downloaded": False,
            "gold_labels_sent_to_model": False,
            "headline_variant": "clean",
            "exclude_from_headline_sources": [],
            "locked_test": False,
            "notes": [
                "Community benchmark; not an official TypeSafe dataset release.",
                "Public original/easy/hard have 72/48/111 questions; report separately.",
                "The 534-item leaderboard contains private/untracked data not downloaded here.",
                "Choice request criteria retain their native order; scoring labels retain canonical order.",
                "Native accuracy uses argmax with lexicographic label tie-break, including Score.",
                "Score also supports expected-value MAE; original has 36 paraphrase groups.",
                "One-hot labels and ten explicit mathematical probability references are separate targets.",
                "Provenance, rationale and gold probabilities are never sent in requests.",
            ],
        },
        "provenance": {
            "dataset_license": "MIT",
            "license_url": f"{JEVBENCH_REPOSITORY}/blob/{JEVBENCH_COMMIT}/LICENSE",
            "notices": list(JEVBENCH_NOTICES),
            "reference_probability_questions": sum(
                "reference_probs" in row["expected"]["decision"] for row in records
            ),
            "gold_policy": (
                "Authored rubric labels reviewed before inference; hard items retain author "
                "and review metadata. Explicit mathematical distributions are not teacher "
                "model confidence or population frequency estimates."
            ),
        },
    }
    artifacts = {
        "public.jsonl": serialized,
        "public.manifest.json": (
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8"),
        **notices,
    }
    directory = output / suite
    for name, content in artifacts.items():
        path = directory / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(
                f"refusing to overwrite a different prepared artifact: {path}"
            )
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (directory / name).write_bytes(content)
    return manifest
