"""Pure scoring functions for calibration metrics.

All functions take a list of (probability, outcome) tuples where
probability is in [0, 1] and outcome is 0 or 1.
No I/O — easy to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BinStats:
    """Statistics for a single calibration bin."""

    bin_start: float
    bin_end: float
    n: int  # count of predictions in this bin
    mean_prob: float  # average predicted probability
    observed_freq: float  # observed frequency of YES outcomes


@dataclass
class MurphyDecomposition:
    """Murphy decomposition of Brier score."""

    reliability: float  # lower is better
    resolution: float  # higher is better
    uncertainty: float  # base rate variance
    brier_check: float  # reliability - resolution + uncertainty (should ≈ brier)


def brier_score(predictions: list[tuple[float, int]]) -> float:
    """Brier score = mean((p - o)^2). Lower is better, range [0, 1]."""
    if not predictions:
        return 0.0
    return sum((p - o) ** 2 for p, o in predictions) / len(predictions)


def log_loss(predictions: list[tuple[float, int]]) -> float:
    """Log loss = -mean(o*ln(p) + (1-o)*ln(1-p)). Lower is better."""
    if not predictions:
        return 0.0
    eps = 1e-15
    total = 0.0
    for p, o in predictions:
        p_clipped = max(eps, min(1 - eps, p))
        total += o * math.log(p_clipped) + (1 - o) * math.log(1 - p_clipped)
    return -total / len(predictions)


def calibration_bins(
    predictions: list[tuple[float, int]], num_bins: int = 10
) -> list[BinStats]:
    """Split predictions into equal-width bins and compute stats."""
    if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins < 1:
        raise ValueError("num_bins must be a positive integer")
    for p, o in predictions:
        if not math.isfinite(p) or not 0 <= p <= 1 or o not in (0, 1):
            raise ValueError("predictions require finite probabilities in [0, 1] and binary outcomes")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(num_bins)]
    bin_width = 1.0 / num_bins

    for p, o in predictions:
        idx = min(int(p * num_bins), num_bins - 1)
        bins[idx].append((p, o))

    result = []
    for i, bin_preds in enumerate(bins):
        bin_start = i * bin_width
        bin_end = (i + 1) * bin_width
        if bin_preds:
            mean_prob = sum(p for p, _ in bin_preds) / len(bin_preds)
            observed_freq = sum(o for _, o in bin_preds) / len(bin_preds)
            result.append(
                BinStats(
                    bin_start=bin_start,
                    bin_end=bin_end,
                    n=len(bin_preds),
                    mean_prob=mean_prob,
                    observed_freq=observed_freq,
                )
            )
        else:
            result.append(
                BinStats(
                    bin_start=bin_start,
                    bin_end=bin_end,
                    n=0,
                    mean_prob=(bin_start + bin_end) / 2,
                    observed_freq=0.0,
                )
            )
    return result


def murphy_decomposition(
    predictions: list[tuple[float, int]],
) -> MurphyDecomposition:
    """Murphy decomposition: Brier = Reliability - Resolution + Uncertainty.

    Groups by *unique forecast values* so the identity holds exactly.
    (The 10-bin grid is only used for the calibration plot.)
    """
    if not predictions:
        return MurphyDecomposition(0.0, 0.0, 0.0, 0.0)

    n_total = len(predictions)
    base_rate_val = sum(o for _, o in predictions) / n_total

    # Group by unique predicted probability
    from collections import defaultdict

    groups: dict[float, list[int]] = defaultdict(list)
    for p, o in predictions:
        groups[p].append(o)

    reliability = 0.0
    resolution = 0.0
    for p_val, outcomes in groups.items():
        n_k = len(outcomes)
        observed = sum(outcomes) / n_k
        reliability += n_k * (p_val - observed) ** 2
        resolution += n_k * (observed - base_rate_val) ** 2

    reliability /= n_total
    resolution /= n_total
    uncertainty = base_rate_val * (1 - base_rate_val)

    return MurphyDecomposition(
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        brier_check=reliability - resolution + uncertainty,
    )


def base_rate(predictions: list[tuple[float, int]]) -> float:
    """Overall frequency of YES outcomes."""
    if not predictions:
        return 0.0
    return sum(o for _, o in predictions) / len(predictions)
