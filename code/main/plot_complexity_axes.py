"""Regenerate the timing figure from the realized-k complexity ledger."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "exp" / "complexity.csv"
OUTPUT = ROOT / "paper" / "figures" / "paper_figures" / "reviewer3_runtime_axes.pdf"


def main():
    data = pd.read_csv(DATA)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15))
    labels = {"n": "samples $n$", "K": "requested clusters $K$", "d": "dimensions $d$"}
    for ax, axis in zip(axes, ("n", "K", "d")):
        subset = data[data["axis"] == axis]
        summary = subset.groupby("value")["time_s"].agg(["mean", "std"]).reset_index()
        ax.errorbar(summary["value"], summary["mean"], yerr=summary["std"],
                    marker="o", linewidth=1.4, capsize=3, color="#276678")
        ax.set_xlabel(labels[axis])
        ax.set_ylabel("seconds")
        ax.grid(True, alpha=0.24)
        ax.tick_params(labelsize=8)
    fig.suptitle("Independent Runtime Diagnostics for MBC (three seeds per point)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
