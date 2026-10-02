"""Calibration curve plot generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from calibrate.scoring import calibration_bins


def plot_calibration_curve(
    predictions: list[tuple[float, int]],
    output_path: str = "assets/calibration.png",
    num_bins: int = 10,
) -> str:
    """Generate a calibration curve PNG and return the output path."""
    bins = calibration_bins(predictions, num_bins)

    non_empty = [b for b in bins if b.n > 0]
    mean_probs = [b.mean_prob for b in non_empty]
    observed_freqs = [b.observed_freq for b in non_empty]
    counts = [b.n for b in non_empty]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    # Calibration curve
    ax.plot(
        mean_probs,
        observed_freqs,
        "o-",
        color="#2563eb",
        linewidth=2,
        markersize=8,
        label="Your calibration",
    )

    # Annotate bin counts
    for mp, of, n in zip(mean_probs, observed_freqs, counts):
        ax.annotate(
            f"n={n}",
            (mp, of),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8,
            color="#6b7280",
        )

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Observed frequency", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return str(out)
