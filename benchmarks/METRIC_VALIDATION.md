# KEV metric parity validation

On 2026-09-21, the adapter's scalar quality metrics were compared numerically
with the original KEV implementations at commit
`4f8110a3f8620cc3a182ae9a708e4398492c4b1a`.
All common scalar metrics matched within `1e-12` absolute error. The largest
observed difference was `8.881784197001252e-16` (score MAE). Complete contrastive
pair metrics matched exactly.

This validates the metric calculations. It does not validate H200 execution,
inference probabilities, model accuracy, or serving latency.

## Method

The optional [verify_metrics.py](verify_metrics.py) script reads these original
files from a local KEV checkout and verifies their full SHA256 before execution:

| File | Extracted functions | SHA256 |
|---|---|---|
| `kev/evaluate.py` | `ece` | `1b20e3f9edf417aa8dae924b1526e52f74b710cadf7213c5ec68334f6e7f8fe1` |
| `kev/benchmark.py` | `coverage_at_error`, `metrics` | `4192ec3b26b065452f84bde38a091e6854a28fe185d2e0f39a0c91b2efe69df7` |
| `kev/contrastive.py` | `paired_flip` | `cbb979aa5d40265ad0e64695f94b281d91751fa405811ddfa8212fede111edcf` |

Python AST extraction keeps only those function definitions. KEV's package,
PyTorch, Transformers and model-loading code are not imported. The audit uses
NumPy in its own optional environment; NumPy is not an adapter dependency.
The original successful audit used NumPy `2.3.5`.

Input generation uses `numpy.random.default_rng(483)` for 100 batches of 50
rows, totaling 5,000 rows. Rows mix choice, boolean and ordinal score questions,
with 2–10 options. Probabilities come from Dirichlet distributions. The first
10 rows in each batch additionally cover binary probability endpoints and
decimal boundaries, including zero and one. Both implementations receive the
same rows in the same order, including confidence ties.

Compared metrics include accuracy, NLL, multiclass Brier score, ten-bin ECE,
mean confidence, confidence bias, confident errors, coverage/accuracy at 0.9,
coverage at 1%/5% empirical error, score MAE and ranked probability score.
Ten complete two-sibling pairs additionally check relevant and invariant pair
metrics. Five pairs have changed gold labels and five have unchanged labels.

## Reproduce

Use a Python environment that already has NumPy installed:

```bash
git clone https://github.com/jaredpalmer/kev.git /tmp/kev-reference
git -C /tmp/kev-reference checkout 4f8110a3f8620cc3a182ae9a708e4398492c4b1a
python benchmarks/verify_metrics.py --kev-root /tmp/kev-reference
```

The JSON output includes the reference hashes, current adapter metric-code hash,
NumPy version, sample counts, per-metric maximum differences and pair results.
A source mismatch or numerical discrepancy exits unsuccessfully. Reference
files are checked even if the local checkout has uncommitted changes.

## Scope differences

- Accuracy headlines use clean question rows, not all submitted records.
  Source `unknowable` is excluded from accuracy and scored for confidence.
- Incomplete pairs are counted explicitly for smoke subsets; KEV's original
  pair function rejects incomplete pairs.
- The adapter omits meaningless unknowable accuracy from grouped reports;
  KEV's original code includes it in some diagnostic subreports.
- This audit compares common scalar metrics and complete pairs. It does not
  certify every report field, input conversion, HTTP behavior, or SemIf's
  separate family-balanced aggregation.

Original definitions: [KEV benchmark.py](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/kev/benchmark.py),
[evaluate.py](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/kev/evaluate.py),
[contrastive.py](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/kev/contrastive.py).
