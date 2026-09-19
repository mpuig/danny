"""Versioned confidence statistics, not probabilities of being correct.

Reference formulas: typesafe-ai/system-one-adapter-python, commit
fb52b1030b7fc1f4f1cf39910afa5da54f9835e3, _utils/confidence_metrics.py.
This pins the public adapter semantics; it does not verify a live Jev deployment.
"""

import math

ENTROPY = "entropy-v0"
ADAPTER = "typesafe-fb52b10"
SCHEMES = (ENTROPY, ADAPTER)


def confidence(probs: list[float], primitive: str, scheme: str = ADAPTER) -> float:
    if not probs or any(not math.isfinite(p) or p < 0 for p in probs):
        raise ValueError("invalid confidence distribution")
    if scheme not in SCHEMES:
        raise ValueError("unknown confidence scheme")
    k = len(probs)
    total = math.fsum(probs)
    p = [v / total for v in probs] if total else [1 / k] * k
    if k == 1:
        return 1.0
    if scheme == ENTROPY:
        return max(
            0.0, 1 + math.fsum(v * math.log(v) for v in p if v > 0) / math.log(k)
        )
    if primitive == "choice":
        return max(0.0, (max(p) - 1 / k) / (1 - 1 / k))
    if primitive != "score":
        raise ValueError("confidence is defined for Choice and Score, not Noul")
    mode = max(range(k), key=p.__getitem__)
    distance = math.fsum(v * abs(i - mode) for i, v in enumerate(p))
    center = (k - 1) / 2
    uniform_distance = math.fsum(abs(i - center) for i in range(k)) / k
    return max(0.0, 1 - distance / uniform_distance)
