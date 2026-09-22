"""Temperature fitting on a separate partition; no model or accelerator imports."""

import math

import numpy as np


def temperature_scale(probs: list[float], temperature: float) -> list[float]:
    if (
        isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive")
    if (
        not probs
        or any(not math.isfinite(p) or p < 0 for p in probs)
        or sum(probs) <= 0
    ):
        raise ValueError("invalid distribution")
    logits = (
        np.log(np.maximum(np.asarray(probs, dtype=np.float64), 1e-12)) / temperature
    )
    values = np.exp(logits - np.max(logits))
    return (values / values.sum()).tolist()


def fit_temperature(rows: list[dict], lower: float = 0.05, upper: float = 20) -> dict:
    if not rows or not 0 < lower < upper:
        raise ValueError("nonempty data and positive temperature bounds required")
    for row in rows:
        if (
            len(row["probabilities"]) != len(row["target"])
            or row["target"].count(1.0) != 1
            or any(p not in (0, 1) for p in row["target"])
        ):
            raise ValueError("temperature fitting requires one-hot outcome targets")
        temperature_scale(row["probabilities"], 1)

    def objective(log_t):
        t = math.exp(log_t)
        return sum(
            -math.log(
                max(
                    temperature_scale(r["probabilities"], t)[r["target"].index(1.0)],
                    1e-12,
                )
            )
            for r in rows
        ) / len(rows)

    left, right = math.log(lower), math.log(upper)
    ratio = (math.sqrt(5) - 1) / 2
    x = right - ratio * (right - left)
    y = left + ratio * (right - left)
    fx, fy = objective(x), objective(y)
    for _ in range(64):
        if fx < fy:
            right, y, fy = y, x, fx
            x = right - ratio * (right - left)
            fx = objective(x)
        else:
            left, x, fx = x, y, fy
            y = left + ratio * (right - left)
            fy = objective(y)
    candidates = [left, right, (left + right) / 2]
    if math.log(lower) <= 0.0 <= math.log(upper):
        candidates.append(0.0)  # t=1 considered only when the bounds admit it
    log_t = min(candidates, key=objective)
    t = math.exp(log_t)
    return {
        "temperature": t,
        "n": len(rows),
        "nll_before": objective(0),
        "nll_after": objective(log_t),
        "at_bound": min(abs(t - lower), abs(t - upper)) < 1e-4,
        "bounds": [lower, upper],
    }


def prediction_config(report: dict) -> dict:
    args = report["arguments"]
    return {
        "renderer_version": report["renderer_version"],
        "readout_version": report.get("readout_version", "letters-v1"),
        "precision": args.get("precision", "native"),
        "execution_mode": args.get("execution_mode", "independent"),
        "max_batch_size": args.get("max_batch_size", 4),
        "contextual_correction": args.get("calibrate", False),
        "backbone_files": report["backbone"]["files"],
        "adapter_sha256": report["adapter_sha256"],
    }


WORKLOAD_METHOD = "per-workload-temperature-v1"
# Guidance thresholds for the fit diagnostic (decision 31 outcome), not gates:
# a temperature within [1/1.25, 1.25] is immaterial, and a near-1 temperature
# with high residual calibration error means the miscalibration is structural.
MATERIAL_LOG_TEMPERATURE = math.log(1.25)
STRUCTURAL_RESIDUAL_ECE = 0.10


def label_to_target_index(question, label) -> int:
    """Map a gold label to its answer_keys index; strict types, fail closed."""
    if question.type == "noul":
        if not isinstance(label, bool):
            raise ValueError("noul labels must be true or false")
        return int(label)  # answer order is [no, yes]
    if question.type == "score":
        if isinstance(label, bool) or type(label) is not int:
            raise ValueError("score labels must be integer level indices")
        if not 0 <= label < len(question.criteria):
            raise ValueError("score label outside the declared levels")
        return label
    if not isinstance(label, str) or label not in question.criteria:
        raise ValueError("choice labels must name one of the question's options")
    return list(question.criteria).index(label)


def top1_ece(rows: list[dict], temperature: float, bins: int = 15) -> float:
    """Binned ECE of top-1 confidence vs correctness after scaling; pure Python."""
    pairs = []
    for row in rows:
        probs = temperature_scale(row["probabilities"], temperature)
        top = max(range(len(probs)), key=probs.__getitem__)
        pairs.append((probs[top], float(row["target"][top] == 1.0)))
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, ok) for c, ok in pairs if (c > lo or b == 0) and c <= hi]
        if bucket:
            confidence = sum(c for c, _ in bucket) / len(bucket)
            accuracy = sum(ok for _, ok in bucket) / len(bucket)
            total += (len(bucket) / len(pairs)) * abs(accuracy - confidence)
    return total


def build_workload_fits(
    rows_by_primitive: dict[str, list[dict]], min_examples: int
) -> tuple[dict, dict]:
    """Fit per-primitive temperatures for one workload sample, with diagnostics.

    A primitive below min_examples, or whose fit is degenerate (an optimizer
    bound hit, or the zero-NLL plateau of an all-correct sample), is diagnosed
    but never applied. Diagnostics are in-sample and therefore optimistic; the
    resampled evidence for the method is EXPERIMENTS section 15.
    """
    if type(min_examples) is not int or min_examples < 2:
        raise ValueError("min_examples must be an integer >= 2")
    fits: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    for primitive in sorted(rows_by_primitive):
        rows = rows_by_primitive[primitive]
        if len(rows) < min_examples:
            diagnostics[primitive] = {
                "n": len(rows),
                "verdict": "insufficient_examples",
                "minimum": min_examples,
            }
            continue
        fit = fit_temperature(rows)
        ece_before = top1_ece(rows, 1.0)
        ece_after = top1_ece(rows, fit["temperature"])
        # An all-correct sample makes the objective a zero plateau below some
        # temperature, so the optimizer stops near-but-not-at the bound with a
        # sharpen-to-certainty fit; treat any vanishing-NLL solution as
        # unidentifiable, exactly like a bound hit.
        degenerate = fit["at_bound"] or fit["nll_after"] < 1e-6
        if degenerate:
            verdict = "degenerate"  # bound hit or zero-NLL plateau; never applied
        elif abs(math.log(fit["temperature"])) >= MATERIAL_LOG_TEMPERATURE:
            verdict = "apply"
        elif ece_after > STRUCTURAL_RESIDUAL_ECE:
            verdict = "structural_warning"  # not a scalar problem; widen escalation
        else:
            verdict = "neutral"
        diagnostics[primitive] = {
            **fit,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "verdict": verdict,
        }
        if not degenerate:
            fits[primitive] = fit
    if not fits:
        raise ValueError(
            "no primitive produced a usable fit; supply at least "
            f"{min_examples} labeled examples of a primitive, with a "
            "non-degenerate mix of outcomes"
        )
    return fits, diagnostics
