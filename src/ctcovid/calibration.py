"""Model calibration assessment and reliability diagrams.

Evaluates whether the model's confidence corresponds to empirical probability,
computing Expected Calibration Error (ECE) and generating reliability diagrams.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def compute_ece(
    confidences: Sequence[float],
    predictions: Sequence[int],
    ground_truth: Sequence[int],
    n_bins: int = 10,
) -> dict:
    """Compute Expected Calibration Error (ECE) across confidence bins.

    ECE = sum_b (N_b / N) * |acc(b) - conf(b)|
    """
    confs = np.asarray(confidences, dtype=float)
    preds = np.asarray(predictions, dtype=int)
    trues = np.asarray(ground_truth, dtype=int)
    n = len(confs)

    if n == 0:
        return {"ece": 0.0, "bins": []}

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bins = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        # In-bin mask
        in_bin = (confs > lower) & (confs <= upper) if lower > 0 else (confs >= lower) & (confs <= upper)
        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            bin_acc = float(np.mean(preds[in_bin] == trues[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = float((lower + upper) / 2.0)

        bins.append({
            "lower": float(lower),
            "upper": float(upper),
            "count": bin_size,
            "accuracy": bin_acc,
            "confidence": bin_conf,
        })

    return {
        "ece": float(ece),
        "n_samples": n,
        "n_bins": n_bins,
        "bins": bins,
    }


def plot_reliability_diagram(ece_result: dict, out_path: str | Path) -> None:
    """Plot reliability diagram with gap bars indicating under/overconfidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = ece_result["bins"]
    ece = ece_result["ece"]

    bin_middles = [(b["lower"] + b["upper"]) / 2.0 for b in bins]
    bin_accs = [b["accuracy"] for b in bins]
    bin_confs = [b["confidence"] for b in bins]
    bin_counts = [b["count"] for b in bins]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Reliability curve
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax1.bar(
        bin_middles,
        bin_accs,
        width=1.0 / len(bins),
        alpha=0.6,
        color="#2563eb",
        edgecolor="#1d4ed8",
        label="Observed Accuracy",
    )
    ax1.plot(bin_confs, bin_accs, "o-", color="#dc2626", label="Confidence vs Acc")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"Reliability Diagram (ECE = {ece:.3f})")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Sample distribution histogram
    ax2.bar(bin_middles, bin_counts, width=1.0 / len(bins), color="#4b5563", alpha=0.7)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1.0)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
