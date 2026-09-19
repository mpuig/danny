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
    log_t = min([0.0, left, right, (left + right) / 2], key=objective)
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
