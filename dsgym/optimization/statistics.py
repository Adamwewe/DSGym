from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class BootstrapDeltaResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    p_value_two_sided: float
    prob_delta_gt_zero: float
    prob_delta_lt_zero: float
    n_resamples: int


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _resample_with_replacement(values: Sequence[float], rng: random.Random) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    return [values[rng.randrange(n)] for _ in range(n)]


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])

    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def bootstrap_mean_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
    paired: bool = False,
) -> BootstrapDeltaResult:
    """Bootstrap CI for mean delta = mean(candidate) - mean(baseline)."""

    b = list(baseline)
    c = list(candidate)

    if not b or not c:
        raise ValueError("baseline and candidate must both be non-empty")

    if paired and len(b) != len(c):
        raise ValueError("paired bootstrap requires equal-length baseline and candidate")

    if n_resamples < 100:
        raise ValueError("n_resamples should be >= 100 for stable intervals")

    if confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be in (0, 1)")

    rng = random.Random(seed)
    point_estimate = _mean(c) - _mean(b)

    deltas: List[float] = []
    if paired:
        n = len(b)
        for _ in range(n_resamples):
            idx = [rng.randrange(n) for _ in range(n)]
            sampled_delta = _mean([c[i] - b[i] for i in idx])
            deltas.append(sampled_delta)
    else:
        for _ in range(n_resamples):
            b_sample = _resample_with_replacement(b, rng)
            c_sample = _resample_with_replacement(c, rng)
            deltas.append(_mean(c_sample) - _mean(b_sample))

    deltas.sort()
    alpha = (1.0 - confidence) / 2.0
    ci_lower = _quantile(deltas, alpha)
    ci_upper = _quantile(deltas, 1.0 - alpha)

    gt_zero = sum(1 for value in deltas if value > 0.0) / len(deltas)
    lt_zero = sum(1 for value in deltas if value < 0.0) / len(deltas)

    # Two-sided bootstrap p-value around zero effect.
    p_value_two_sided = min(1.0, 2.0 * min(1.0 - gt_zero, 1.0 - lt_zero))

    return BootstrapDeltaResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value_two_sided=p_value_two_sided,
        prob_delta_gt_zero=gt_zero,
        prob_delta_lt_zero=lt_zero,
        n_resamples=n_resamples,
    )


def summarize_direction(result: BootstrapDeltaResult, *, objective: str) -> Dict[str, bool]:
    """Summarize significance for objective direction.

    objective: maximize or minimize
    """

    normalized = objective.strip().lower()
    if normalized not in {"maximize", "minimize"}:
        raise ValueError("objective must be maximize or minimize")

    if normalized == "maximize":
        return {
            "significant_improvement": result.ci_lower > 0.0,
            "significant_regression": result.ci_upper < 0.0,
        }

    # minimize objective improves when delta < 0
    return {
        "significant_improvement": result.ci_upper < 0.0,
        "significant_regression": result.ci_lower > 0.0,
    }
