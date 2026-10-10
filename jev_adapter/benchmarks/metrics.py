"""Dependency-free metrics following KEV's frozen benchmark definitions.

Reimplemented against jaredpalmer/kev commit
4f8110a3f8620cc3a182ae9a708e4398492c4b1a, kev/benchmark.py,
kev/evaluate.py and kev/contrastive.py (Apache-2.0). Differences are explicit:
partial pairs are counted, and unknowable items never receive accuracy metrics.
"""

import math
from collections import defaultdict
from statistics import fmean


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def predicted_index(row):
    if row.get("argmax_tie_break") == "lexicographic_label":
        return min(range(len(row["p"])), key=lambda i: (-row["p"][i], row["keys"][i]))
    return argmax(row["p"])


def distribution(raw, labels):
    if not isinstance(raw, dict) or set(raw) != set(labels):
        raise ValueError("probability keys differ from requested labels")
    p = [raw[key] for key in labels]
    if any(
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in p
    ):
        raise ValueError("invalid probability value")
    total = math.fsum(p)
    if total <= 0 or abs(total - 1) > max(1e-5, len(labels) * 0.005 + 1e-8):
        raise ValueError("invalid probability sum")
    return [value / total for value in p], total


def prediction_rows(item, response):
    expected = item["expected"]
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(expected):
        raise ValueError("response question IDs differ from request")
    rows = []
    meta = item["metadata"]
    for qid, target in expected.items():
        answer = answers[qid]
        kind = target["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ValueError("response answer type differs from request")
        if kind == "noul":
            value = answer.get("noul")
            if type(value) not in (int, float):
                raise ValueError("invalid noul probability")
            raw = {"true": value, "false": 1 - value}
        else:
            raw = answer.get("probabilities")
        labels = target["labels"]
        p, total = distribution(raw, labels)
        rows.append(
            {
                "id": item["id"],
                "suite": item["suite"],
                "split": item["split"],
                "question": qid,
                "source": item["source"],
                "task": target["task"],
                "type": kind,
                "variant": item["variant"],
                "keys": labels,
                "label": target["label"],
                "p": p,
                "group": meta.get("group_id", item["id"]),
                "pair_id": meta.get("pair_id"),
                "sibling": meta.get("sibling"),
                "control_id": meta.get("control_id"),
                "parent": meta.get("parent_id")
                or (item["id"] if item["variant"] == "clean" else meta.get("group_id")),
                "raw_probability_sum": total,
                "zero_count": sum(value == 0 for value in p),
                "reference_probs": target.get("reference_probs"),
                "argmax_tie_break": meta.get("gold_policy", {}).get("argmax_tie_break"),
            }
        )
    return rows


def coverage_at_error(confidence, correct, budget):
    # Stable ties deliberately match KEV. This empirical, retrospective cutoff
    # is not a risk guarantee and must not be deployed as a fitted threshold.
    order = sorted(range(len(confidence)), key=lambda i: -confidence[i])
    wrong, accepted = 0, 0
    for rank, index in enumerate(order, 1):
        wrong += not correct[index]
        if wrong <= budget * rank:
            accepted = rank
    return accepted / len(order)


def metrics(rows):
    if not rows:
        return None
    nll, correct, confidence, brier, mae, rps = [], [], [], [], [], []
    for row in rows:
        p, y = row["p"], row["label"]
        nll.append(-math.log(max(p[y], 1e-9)))
        correct.append(predicted_index(row) == y)
        confidence.append(max(p))
        brier.append(math.fsum((value - (i == y)) ** 2 for i, value in enumerate(p)))
        if row["type"] == "score":
            mae.append(abs(math.fsum(i * value for i, value in enumerate(p)) - y))
            cumulative = 0.0
            errors = []
            for i, value in enumerate(p[:-1]):
                cumulative += value
                errors.append((cumulative - (i >= y)) ** 2)
            rps.append(fmean(errors))
    ece = 0.0
    for index in range(10):
        # Match numpy.linspace(0, 1, 11), including floating-point bin edges.
        lo, hi = index * 0.1, (index + 1) * 0.1
        bucket = [
            i
            for i, c in enumerate(confidence)
            if lo <= c and (c < hi if index < 9 else c <= 1)
        ]
        if bucket:
            ece += (
                len(bucket)
                / len(rows)
                * abs(
                    fmean(correct[i] for i in bucket)
                    - fmean(confidence[i] for i in bucket)
                )
            )
    high = [i for i, value in enumerate(confidence) if value >= 0.9]
    result = {
        "n": len(rows),
        "acc": fmean(correct),
        "nll": fmean(nll),
        "brier": fmean(brier),
        "ece": ece,
        "mean_conf": fmean(confidence),
        "confidence_bias": fmean(confidence) - fmean(correct),
        "confident_error_rate": sum(not correct[i] for i in high) / len(rows),
        "coverage_at_0_9": len(high) / len(rows),
        "accuracy_at_0_9": fmean(correct[i] for i in high) if high else None,
        "coverage_at_5pct_error": coverage_at_error(confidence, correct, 0.05),
        "coverage_at_1pct_error": coverage_at_error(confidence, correct, 0.01),
    }
    if mae:
        result.update(score_mae=fmean(mae), ranked_probability_score=fmean(rps))
    return result


def grouped_metrics(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        if row["source"] != "unknowable":
            grouped[row[key]].append(row)
    return {name: metrics(group) for name, group in sorted(grouped.items())}


def paired_flip(rows):
    pairs = defaultdict(dict)
    for row in rows:
        if row.get("pair_id"):
            pair = pairs[row["pair_id"], row["question"]]
            if row["sibling"] in pair:
                raise ValueError("duplicate contrastive sibling")
            pair[row["sibling"]] = row
    complete = [pair for pair in pairs.values() if set(pair) == {"a", "b"}]

    def truth(row):
        return row["keys"][row["label"]]

    def prediction(row):
        return row["keys"][predicted_index(row)]

    relevant = [p for p in complete if truth(p["a"]) != truth(p["b"])]
    invariant = [p for p in complete if truth(p["a"]) == truth(p["b"])]

    def both(group):
        return (
            fmean(all(prediction(r) == truth(r) for r in p.values()) for p in group)
            if group
            else None
        )

    return {
        "pairs": len(relevant),
        "incomplete_pairs": len(pairs) - len(complete),
        "flip_rate": fmean(prediction(p["a"]) != prediction(p["b"]) for p in relevant)
        if relevant
        else None,
        "both_correct_rate": both(relevant),
        "invariant_pairs": len(invariant),
        "invariant_both_correct_rate": both(invariant),
        "invariance_rate": fmean(
            prediction(p["a"]) == prediction(p["b"]) for p in invariant
        )
        if invariant
        else None,
    }


def summarize(rows):
    clean = [r for r in rows if r["variant"] == "clean"]
    knowable = [r for r in clean if r["source"] != "unknowable"]
    controls = {r["id"]: r for r in clean if r["source"] == "unknowable_control"}
    unknown = [r for r in clean if r["source"] == "unknowable"]
    unknowable = None
    if unknown:
        paired = [
            (max(r["p"]), max(controls[r["control_id"]]["p"]))
            for r in unknown
            if r.get("control_id") in controls
        ]
        unknowable = {
            "n": len(unknown),
            "mean_max_p": fmean(max(r["p"]) for r in unknown),
            "share_at_0_9": fmean(max(r["p"]) >= 0.9 for r in unknown),
            "control_acc": (
                fmean(predicted_index(r) == r["label"] for r in controls.values())
                if controls
                else None
            ),
            "control_mean_max_p": (
                fmean(max(r["p"]) for r in controls.values()) if controls else None
            ),
            "control_share_at_0_9": (
                fmean(max(r["p"]) >= 0.9 for r in controls.values())
                if controls
                else None
            ),
            "paired_confidence_drop": fmean(c - u for u, c in paired)
            if paired
            else None,
            "share_less_confident_than_control": (
                fmean(u < c for u, c in paired) if paired else None
            ),
        }
    originals = {(r["id"], r["question"]): r for r in clean}
    deltas, flips, missing = [], [], 0
    for row in rows:
        if row["variant"] != "permuted" or row["type"] != "choice":
            continue
        original = originals.get((row["parent"], row["question"]))
        if original is None:
            missing += 1
            continue
        aligned = [row["p"][row["keys"].index(key)] for key in original["keys"]]
        deltas.append(max(abs(a - b) for a, b in zip(aligned, original["p"])))
        flips.append(argmax(aligned) != argmax(original["p"]))
    tasks = grouped_metrics(clean, "task")
    return {
        "clean": metrics(knowable),
        "tasks": tasks,
        "sources": grouped_metrics(clean, "source"),
        "variants": grouped_metrics(rows, "variant"),
        "macro_task_acc": fmean(t["acc"] for t in tasks.values()) if tasks else None,
        "objective": -fmean(t["nll"] for t in tasks.values()) if tasks else None,
        "paired_flip": paired_flip(clean),
        "reference_distribution": reference_metrics(clean),
        "unknowable": unknowable,
        "permutation": {
            "n": len(flips),
            "missing_parents": missing,
            "flip_rate": fmean(flips) if flips else None,
            "mean_max_delta": fmean(deltas) if deltas else None,
        },
        "metric_policy": {
            "nll_floor": 1e-9,
            "ece_bins": 10,
            "confidence_for_calibration": "maximum label probability",
            "calibration_applied": False,
            "unknown_accuracy_excluded": True,
            "coverage_error_budget_is_retrospective": True,
            "raw_sums_outside_1e_5": sum(
                abs(r["raw_probability_sum"] - 1) > 1e-5 for r in rows
            ),
            "returned_zeros": sum(r["zero_count"] for r in rows),
        },
    }


def reference_metrics(rows):
    selected = [row for row in rows if row.get("reference_probs") is not None]
    if not selected:
        return None
    squared, variation, kl = [], [], []
    for row in selected:
        p, q = row["p"], row["reference_probs"]
        squared.append(math.fsum((a - b) ** 2 for a, b in zip(p, q)))
        variation.append(0.5 * math.fsum(abs(a - b) for a, b in zip(p, q)))
        kl.append(
            math.fsum(b * math.log(b / max(a, 1e-9)) for a, b in zip(p, q) if b > 0)
        )
    return {
        "n": len(selected),
        "squared_l2": fmean(squared),
        "total_variation": fmean(variation),
        "kl_reference_to_model": fmean(kl),
        "note": "Exact reference-distribution fidelity, separate from hard-label Brier/ECE.",
    }


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def latency_summary(values):
    return {
        "n": len(values),
        "mean": fmean(values) if values else None,
        "p50": quantile(values, 0.5),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
    }
