#!/usr/bin/env python3
"""Fit held-out temperatures and audit Jev-style typed decision logits.

Prediction format: one JSON object per line: {"id": row_id, "logits": {key: number}}.
Each map must have exactly the keys accepted by that row's question. Logits must
be pre-softmax, not already-normalized probabilities.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keys_for(question: dict):
    kind = question["type"]
    if kind == "choice":
        return list(question["criteria"])
    if kind == "noul":
        return ["false", "true"]
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    raise ValueError(f"unsupported question type {kind}")


def load_pairs(dataset: Path, predictions: Path):
    expected = {x["id"]: x for x in read_jsonl(dataset)}
    predicted = {}
    for item in read_jsonl(predictions):
        id_ = item["id"]
        if id_ in predicted:
            raise ValueError(f"duplicate prediction {id_}")
        predicted[id_] = item["logits"]
    if set(expected) != set(predicted):
        raise ValueError(f"prediction IDs differ: missing {len(set(expected)-set(predicted))}, extra {len(set(predicted)-set(expected))}")
    pairs = []
    for id_, row in expected.items():
        keys = keys_for(row["question"])
        logits = predicted[id_]
        if set(logits) != set(keys) or any(not isinstance(logits[k], (int, float)) or not math.isfinite(logits[k]) for k in keys):
            raise ValueError(f"invalid logit map for {id_}")
        pairs.append((row, [float(logits[k]) for k in keys], keys.index(row["gold"]["key"])))
    return pairs


def softmax(logits: list[float], temperature: float):
    scaled = [x / temperature for x in logits]
    maximum = max(scaled)
    exps = [math.exp(x - maximum) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]


def nll(pairs, temperature: float):
    if not pairs:
        raise ValueError("empty calibration subset")
    return sum(-math.log(max(1e-12, softmax(logits, temperature)[gold])) for _, logits, gold in pairs) / len(pairs)


def fit_temperature(pairs):
    # One scalar per question type. The range is deliberately bounded; report
    # a boundary fit as a signal to obtain more calibration data.
    grid = [math.exp(math.log(0.25) + i * (math.log(4.0) - math.log(0.25)) / 160) for i in range(161)]
    return min(grid, key=lambda t: nll(pairs, t))


def reliability(confidences, correctness, bins: int = 5):
    result = []
    for bin_id in range(bins):
        indices = [i for i, p in enumerate(confidences)
                   if min(bins - 1, int(p * bins)) == bin_id]
        if indices:
            result.append({"range": [bin_id / bins, (bin_id + 1) / bins], "n": len(indices),
                           "mean_confidence": sum(confidences[i] for i in indices) / len(indices),
                           "accuracy": sum(correctness[i] for i in indices) / len(indices)})
    return result


def evaluate(pairs, temperature: float):
    losses, briers, correct, confidences, outcomes = [], [], [], [], []
    true_probs, true_outcomes, absolute_errors, ranked_scores = [], [], [], []
    for row, logits, gold in pairs:
        probs = softmax(logits, temperature)
        winner = max(range(len(probs)), key=lambda i: probs[i])
        losses.append(-math.log(max(1e-12, probs[gold])))
        briers.append(sum((p - (i == gold)) ** 2 for i, p in enumerate(probs)))
        correct.append(int(winner == gold))
        confidences.append(probs[winner])
        outcomes.append(int(winner == gold))
        if row["question"]["type"] == "noul":
            true_probs.append(probs[1])
            true_outcomes.append(int(gold == 1))
        if row["question"]["type"] == "score":
            expected = sum(i * p for i, p in enumerate(probs))
            absolute_errors.append(abs(expected - gold))
            ranked_scores.append(sum((sum(probs[:k+1]) - int(gold <= k)) ** 2
                                     for k in range(len(probs) - 1)) / (len(probs) - 1))
    result = {
        "n": len(pairs), "accuracy": sum(correct) / len(pairs),
        "log_loss": sum(losses) / len(pairs), "multiclass_brier": sum(briers) / len(pairs),
        "top_answer_reliability": reliability(confidences, outcomes),
    }
    if true_probs:
        result["true_event_reliability"] = reliability(true_probs, true_outcomes)
    if absolute_errors:
        result["score_mae"] = sum(absolute_errors) / len(absolute_errors)
        result["ranked_probability_score"] = sum(ranked_scores) / len(ranked_scores)
    return result


def main():
    if os.environ.get("VL_JEV_MODAL") != "1":
        raise RuntimeError("Calibration evaluation must run inside the Modal function")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "pilot_v1")
    parser.add_argument("--calibration-logits", type=Path, required=True)
    parser.add_argument("--test-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    calibration = load_pairs(args.dataset_dir / "calibration.jsonl", args.calibration_logits)
    test = load_pairs(args.dataset_dir / "test.jsonl", args.test_logits)
    report = {}
    for kind in ["choice", "noul", "score"]:
        fit_rows = [x for x in calibration if x[0]["question"]["type"] == kind]
        test_rows = [x for x in test if x[0]["question"]["type"] == kind]
        temperature = fit_temperature(fit_rows)
        report[kind] = {
            "calibration_n": len(fit_rows), "test_n": len(test_rows),
            "temperature": temperature,
            "boundary_fit": temperature <= 0.251 or temperature >= 3.99,
            "test_before": evaluate(test_rows, 1.0),
            "test_after": evaluate(test_rows, temperature),
            "test_by_modality_after": {modality: evaluate([x for x in test_rows if x[0]["modality"] == modality], temperature)
                                       for modality in sorted({x[0]["modality"] for x in test_rows})},
        }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
