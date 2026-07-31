"""
Parameter sensitivity analysis for Morphogenetic Buds Clustering.

The analysis perturbs one parameter at a time around the paper default and
evaluates representative datasets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from morphogenetic_buds_clustering import MorphogeneticBudsClustering
from sklearn.datasets import load_digits, load_wine, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


SEEDS = [3, 7, 11]


def datasets(seed: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    wine = load_wine()
    digits = load_digits()
    mask = digits.target < 5
    moons_x, moons_y = make_moons(n_samples=500, noise=0.07, random_state=seed)
    circles_x, circles_y = make_circles(n_samples=500, factor=0.42, noise=0.045, random_state=seed)
    return [
        ("two_moons", moons_x, moons_y),
        ("circles", circles_x, circles_y),
        ("wine", wine.data, wine.target),
        ("digits_0_4", digits.data[mask], digits.target[mask]),
    ]


def preprocess(x: np.ndarray) -> np.ndarray:
    x = StandardScaler().fit_transform(x)
    if x.shape[1] > 10:
        x = PCA(n_components=min(10, x.shape[1], x.shape[0] - 1), random_state=0).fit_transform(x)
    return x


def default_params(n_clusters: int) -> dict:
    return {
        "initial_buds": max(12, n_clusters * 6),
        "steps": 140,
        "growth_rate": 0.15,
        "inhibition": 0.045,
        "birth_quantile": 0.88,
        "organ_merge_radius": 3.6,
        "valley_ratio": 0.18,
        "tangent_alignment": 0.35,
        "max_bridge_void": 0.46,
        "target_clusters": n_clusters,
        "compact_refinement_steps": 18,
        "compact_refinement_restarts": 10,
        "compact_refinement_dim_threshold": 3,
    }


def run_sensitivity() -> pd.DataFrame:
    grid = {
        "organ_merge_radius": [2.8, 3.2, 3.6, 4.0, 4.4],
        "tangent_alignment": [0.0, 0.2, 0.35, 0.5, 0.65],
        "compact_refinement_steps": [0, 5, 10, 18, 30],
        "compact_refinement_restarts": [1, 3, 5, 10, 15],
    }
    rows = []
    for seed in SEEDS:
        for dataset_name, raw_x, y in datasets(seed):
            x = preprocess(raw_x)
            n_clusters = len(np.unique(y))
            for param, values in grid.items():
                for value in values:
                    params = default_params(n_clusters)
                    params[param] = value
                    params["apoptosis_mass"] = max(2.5, len(x) * 0.0055)
                    params["random_state"] = seed
                    labels = MorphogeneticBudsClustering(**params).fit_predict(x)
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "parameter": param,
                            "value": value,
                            "ari": adjusted_rand_score(y, labels),
                        }
                    )
    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "mbc_sensitivity_raw.csv", index=False, encoding="utf-8-sig")

    summary = (
        df.groupby(["parameter", "value"])["ari"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["parameter", "value"])
    )
    summary.to_csv(out_dir / "mbc_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=160)
    axes = axes.ravel()
    for ax, (param, group) in zip(axes, summary.groupby("parameter")):
        ax.errorbar(group["value"], group["mean"], yerr=group["std"], marker="o", capsize=3)
        ax.set_title(param)
        ax.set_xlabel("value")
        ax.set_ylabel("ARI")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "mbc_sensitivity.png")
    plt.close(fig)

    lines = [
        "# MBC Parameter Sensitivity",
        "",
        "Mean ARI over two nonconvex datasets and two real feature datasets with three seeds.",
        "",
        summary.round(4).to_markdown(index=False),
    ]
    (out_dir / "mbc_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results" / "paper_results"
    df = run_sensitivity()
    save_outputs(df, out_dir)
    print(df.groupby("parameter")["ari"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
