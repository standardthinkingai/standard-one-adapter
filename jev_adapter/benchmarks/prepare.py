"""Download pinned KEV and JevBench evaluation data, without training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from jev_adapter.protocol import SystemOneRequest

from .data import (
    DEFAULT_SUITES,
    KEV_COMMIT,
    KEV_RAW,
    KEV_REPOSITORY,
    SUITES,
    parse_partition,
    population_counts,
    require_sha256,
    sha256,
    source_provenance,
)
from .jevbench import JEVBENCH_SUITES, prepare_jevbench


def read_source(
    path: str, client: httpx.Client | None, source_root: Path | None
) -> bytes:
    if source_root is not None:
        return (source_root / path).read_bytes()
    if client is None:
        raise ValueError("an HTTP client or --source-root is required")
    response = client.get(f"{KEV_RAW}/{path}")
    response.raise_for_status()
    return response.content


def prepare_suite(
    suite: str,
    split: str,
    output: Path,
    *,
    client: httpx.Client | None = None,
    source_root: Path | None = None,
    allow_test: bool = False,
    limit: int | None = None,
) -> dict:
    if suite not in SUITES:
        raise ValueError(f"unknown suite: {suite}")
    if split not in ("development", "test"):
        raise ValueError("only development and test are evaluation partitions")
    if split == "test" and not allow_test:
        raise ValueError(
            "locked test requires --allow-test; use development for iteration"
        )
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    spec = SUITES[suite]
    manifest_bytes = read_source(f"{spec.path}/manifest.json", client, source_root)
    require_sha256(manifest_bytes, spec.manifest_sha256, f"{suite}/manifest.json")
    upstream = json.loads(manifest_bytes)
    expected_file = upstream["files"][f"{split}.jsonl"]
    raw_bytes = read_source(f"{spec.path}/{split}.jsonl", client, source_root)
    require_sha256(raw_bytes, expected_file["sha256"], f"{suite}/{split}.jsonl")
    records = parse_partition(raw_bytes, suite, split)
    counts = population_counts(records)
    if counts["records"] != expected_file["records"]:
        raise ValueError(f"upstream record count mismatch: {suite}/{split}")
    if (
        "questions" in expected_file
        and counts["questions"] != expected_file["questions"]
    ):
        raise ValueError(f"upstream question count mismatch: {suite}/{split}")
    if not records:
        raise ValueError(
            f"{suite}/{split} is empty; this suite may be development-only"
        )
    selected = records if limit is None else records[:limit]
    for row in selected:
        SystemOneRequest.model_validate({"model": "validation-only", **row["record"]})
    serialized = (
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in selected
        )
    ).encode("utf-8")
    directory = output / suite
    data_path = directory / f"{split}.jsonl"
    manifest_path = directory / f"{split}.manifest.json"
    upstream_path = directory / "upstream.manifest.json"
    manifest = {
        "schema_version": 1,
        "suite": suite,
        "split": split,
        "description": spec.description,
        "data_file": data_path.name,
        "data_sha256": sha256(serialized),
        "upstream": {
            "repository": KEV_REPOSITORY,
            "commit": KEV_COMMIT,
            "path": f"{spec.path}/{split}.jsonl",
            "url": f"{KEV_RAW}/{spec.path}/{split}.jsonl",
            "sha256": sha256(raw_bytes),
            "manifest_sha256": spec.manifest_sha256,
        },
        "full_partition": counts,
        "selected": population_counts(selected),
        "selection": {
            "method": "full" if len(selected) == len(records) else "prefix",
            "requested_limit": limit,
            "is_full_partition": len(selected) == len(records),
            "note": (
                "Full frozen partition."
                if len(selected) == len(records)
                else "Smoke subset only; not a full-suite result. Pairs may be incomplete."
            ),
        },
        "protocol": {
            "calibration_applied": False,
            "training_data_downloaded": False,
            "gold_labels_sent_to_model": False,
            "headline_variant": "clean",
            "exclude_from_headline_sources": ["unknowable"],
            "locked_test": split == "test",
            "notes": list(spec.notes),
            "upstream_context_policy": upstream.get("context"),
        },
        "provenance": source_provenance(records, upstream),
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode()
    # Idempotent preparation is safe; refuse to replace any changed frozen artifact.
    for path, content in (
        (data_path, serialized),
        (manifest_path, manifest_payload),
        (upstream_path, manifest_bytes),
    ):
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(
                f"refusing to overwrite a different prepared artifact: {path}"
            )
    directory.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(serialized)
    manifest_path.write_bytes(manifest_payload)
    upstream_path.write_bytes(manifest_bytes)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite", action="append", choices=tuple(SUITES) + tuple(JEVBENCH_SUITES)
    )
    parser.add_argument(
        "--split", choices=("development", "test"), default="development"
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--limit", type=int, help="deterministic prefix smoke subset per suite"
    )
    parser.add_argument(
        "--source-root", type=Path, help="offline KEV checkout; same SHA256 checks"
    )
    parser.add_argument(
        "--jevbench-source-root", type=Path, help="offline pinned JevBench checkout"
    )
    args = parser.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        parser.error("--split test requires --allow-test")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    suites = args.suite or (
        DEFAULT_SUITES if args.split == "test" else (*DEFAULT_SUITES, *JEVBENCH_SUITES)
    )
    if args.split == "test" and any(suite in JEVBENCH_SUITES for suite in suites):
        parser.error("JevBench supplies a public subset, not a locked test split")
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for suite in dict.fromkeys(suites):
            if suite in JEVBENCH_SUITES:
                manifest = prepare_jevbench(
                    suite,
                    args.output,
                    client=client,
                    source_root=args.jevbench_source_root,
                    limit=args.limit,
                )
                print(
                    json.dumps(
                        {
                            "suite": suite,
                            "split": "public",
                            "selected": manifest["selected"],
                            "is_full_partition": manifest["selection"][
                                "is_full_partition"
                            ],
                        }
                    )
                )
                continue
            manifest = prepare_suite(
                suite,
                args.split,
                args.output,
                client=client,
                source_root=args.source_root,
                allow_test=args.allow_test,
                limit=args.limit,
            )
            print(
                json.dumps(
                    {
                        "suite": suite,
                        "split": args.split,
                        "data": str(args.output / suite / f"{args.split}.jsonl"),
                        "selected": manifest["selected"],
                        "is_full_partition": manifest["selection"]["is_full_partition"],
                    }
                )
            )


if __name__ == "__main__":
    main()
