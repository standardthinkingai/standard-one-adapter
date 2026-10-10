"""Audit metric parity against verified KEV source, without importing KEV or torch.

Requires NumPy only in the optional audit environment, not in jev-adapter.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_adapter.benchmarks import metrics as adapter_metrics
from jev_adapter.benchmarks.data import KEV_COMMIT

REFERENCE_FILES = {
    "kev/evaluate.py": (
        "1b20e3f9edf417aa8dae924b1526e52f74b710cadf7213c5ec68334f6e7f8fe1",
        {"ece"},
    ),
    "kev/benchmark.py": (
        "4192ec3b26b065452f84bde38a091e6854a28fe185d2e0f39a0c91b2efe69df7",
        {"coverage_at_error", "metrics"},
    ),
    "kev/contrastive.py": (
        "cbb979aa5d40265ad0e64695f94b281d91751fa405811ddfa8212fede111edcf",
        {"paired_flip"},
    ),
}


def extract_reference(root, numpy):
    scope = {"np": numpy, "math": math, "EPSILON": 1e-9}
    for relative, (expected_hash, names) in REFERENCE_FILES.items():
        payload = (root / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(
                f"reference file differs from KEV {KEV_COMMIT}: {relative}"
            )
        tree = ast.parse(payload, filename=relative)
        tree.body = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        if {node.name for node in tree.body} != names:
            raise ValueError(f"missing reference functions in {relative}")
        # Execute only the hash-verified function definitions. Top-level imports,
        # model loading, decorators elsewhere and benchmark entry points are absent.
        exec(compile(tree, relative, "exec"), scope)  # noqa: S102
    return scope


def audit(root, numpy):
    reference = extract_reference(root, numpy)
    rng = numpy.random.default_rng(483)
    maximum_delta = {}
    tolerance = 1e-12
    for batch in range(100):
        rows = []
        for index in range(50):
            kind = ["choice", "noul", "score"][index % 3]
            options = 2 if kind == "noul" else int(rng.integers(2, 11))
            probabilities = rng.dirichlet(numpy.ones(options)).tolist()
            label = int(rng.integers(options))
            if index < 10:
                probabilities = [index / 10, 1 - index / 10]
                label %= 2
                kind = "noul"
            rows.append({"p": probabilities, "label": label, "type": kind})
        expected = reference["metrics"](rows)
        actual = adapter_metrics.metrics(rows)
        for key, value in actual.items():
            if value is None or not isinstance(value, (int, float)):
                continue
            delta = abs(value - expected[key])
            maximum_delta[key] = max(maximum_delta.get(key, 0), delta)
            if delta > tolerance:
                raise ValueError(
                    f"batch {batch}, {key}: adapter={value}, KEV={expected[key]}"
                )
    rows = []
    for index in range(10):
        for sibling, label in (("a", 0), ("b", index % 2)):
            rows.append(
                {
                    "pair_id": str(index),
                    "question": "q",
                    "keys": ["x", "y"],
                    "label": label,
                    "p": [0.7, 0.3] if index % 3 else [0.4, 0.6],
                    "sibling": sibling,
                }
            )
    expected = reference["paired_flip"](rows)
    actual = adapter_metrics.paired_flip(rows)
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(f"pair metric differs: {key}: {actual[key]} != {value}")
    return {
        "status": "passed",
        "kev_commit": KEV_COMMIT,
        "reference_sha256": {path: entry[0] for path, entry in REFERENCE_FILES.items()},
        "adapter_metrics_sha256": hashlib.sha256(
            Path(adapter_metrics.__file__).read_bytes()
        ).hexdigest(),
        "numpy_version": numpy.__version__,
        "seed": 483,
        "batches": 100,
        "rows_per_batch": 50,
        "total_metric_rows": 5000,
        "absolute_tolerance": tolerance,
        "maximum_absolute_delta": maximum_delta,
        "complete_pair_metrics_exact": expected,
        "limitations": [
            "Numerical metric parity, not GPU/model inference validation.",
            "Only common returned scalar metrics and complete pair metrics compared.",
            "Partial-pair handling and unknowable reporting intentionally differ.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kev-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        import numpy
    except ImportError:
        parser.error("NumPy is required in this optional audit environment")
    print(json.dumps(audit(args.kev_root, numpy), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
