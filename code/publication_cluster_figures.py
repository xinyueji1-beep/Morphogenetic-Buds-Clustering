"""
Create publication-style clustering figures for the MBC manuscript.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from external_real_benchmark_mbc import prepare_features
from morphogenetic_buds_clustering import MorphogeneticBudsClustering
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import fetch_openml, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PACKAGE_ROOT / "results" / "paper_results"
FIG_DIR = PACKAGE_ROOT / "figures" / "paper_figures"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )


def mbc_model(k: int, seed: int) -> MorphogeneticBudsClustering:
    return MorphogeneticBudsClustering(
        initial_buds=max(12, k * 6),
        steps=150,
        growth_rate=0.15,
        inhibition=0.045,
        apoptosis_mass=3.0,
        birth_quantile=0.88,
        organ_merge_radius=3.6,
        valley_ratio=0.18,
        tangent_alignment=0.35,
        max_bridge_void=0.46,
        target_clusters=k,
        compact_refinement_steps=0,
        random_state=seed,
    )


def scatter(ax, x: np.ndarray, labels: np.ndarray, title: str) -> None:
    ax.scatter(x[:, 0], x[:, 1], c=labels, cmap="tab10", s=16, alpha=0.88, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(alpha=0.18)


def make_nonconvex_showcase() -> None:
    datasets = [
        ("Two moons", *make_moons(n_samples=700, noise=0.065, random_state=9)),
        ("Concentric circles", *make_circles(n_samples=700, factor=0.42, noise=0.04, random_state=9)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 7.2), dpi=180)
    summary = []

    for row, (name, raw_x, y) in enumerate(datasets):
        x = StandardScaler().fit_transform(raw_x)
        k = len(np.unique(y))
        labels_mbc = mbc_model(k, 9).fit_predict(x)
        labels_spec = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=12,
            random_state=9,
        ).fit_predict(x)
        labels_km = KMeans(n_clusters=k, n_init=30, random_state=9).fit_predict(x)

        panels = [
            (y, f"{name}: truth"),
            (labels_mbc, f"MBC, ARI={adjusted_rand_score(y, labels_mbc):.3f}"),
            (labels_spec, f"Spectral, ARI={adjusted_rand_score(y, labels_spec):.3f}"),
            (labels_km, f"KMeans, ARI={adjusted_rand_score(y, labels_km):.3f}"),
        ]
        for col, (labels, title) in enumerate(panels):
            scatter(axes[row, col], x, labels, title)

        summary.append(
            {
                "dataset": name,
                "MBC": adjusted_rand_score(y, labels_mbc),
                "Spectral": adjusted_rand_score(y, labels_spec),
                "KMeans": adjusted_rand_score(y, labels_km),
            }
        )

    fig.suptitle("Nonconvex morphology: why the original MBC core is retained", y=1.01, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "publication_nonconvex_showcase.png")
    pd.DataFrame(summary).to_csv(OUT_DIR / "publication_nonconvex_showcase_scores.csv", index=False)


def make_morphogenesis_diagram() -> None:
    x, y = make_moons(n_samples=620, noise=0.065, random_state=13)
    x = StandardScaler().fit_transform(x)
    model = mbc_model(2, 13)
    labels = model.fit_predict(x)
    centers = model.centers_
    organs = model.organ_centers_

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), dpi=180)
    axes[0].scatter(x[:, 0], x[:, 1], c="#6b7280", s=16, alpha=0.72, linewidths=0)
    axes[0].set_title("Tissue cells emit morphogens")

    axes[1].scatter(x[:, 0], x[:, 1], c="#d1d5db", s=14, alpha=0.65, linewidths=0)
    axes[1].scatter(centers[:, 0], centers[:, 1], c="#111827", s=145, marker="*", edgecolors="white", linewidths=0.8, label="micro-buds")
    if organs is not None:
        axes[1].scatter(organs[:, 0], organs[:, 1], c="#ef4444", s=70, marker="o", edgecolors="white", linewidths=0.9, label="organs")
    axes[1].set_title("Buds grow, inhibit, and fuse")
    axes[1].legend(frameon=False)

    scatter(axes[2], x, labels, f"Organ-level clusters, ARI={adjusted_rand_score(y, labels):.3f}")

    for ax in axes:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.18)
    fig.tight_layout()
    save_figure(fig, "publication_morphogenesis_diagram.png")


def make_bio_pca_polished() -> None:
    datasets = [("Leukemia-3", 45091), ("Lymphoma-3", 45094)]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), dpi=180)
    rows = []
    for row, (name, data_id) in enumerate(datasets):
        try:
            bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        except Exception as exc:
            print(f"[warn] skipped publication biological figure {name}: {exc}")
            for ax in axes[row]:
                ax.text(0.5, 0.5, f"{name}\\nunavailable", ha="center", va="center")
                ax.set_axis_off()
            continue
        x = prepare_features(bunch.data)
        y = LabelEncoder().fit_transform(bunch.target.astype(str))
        labels = mbc_model(len(np.unique(y)), 7).fit_predict(x)
        coords = PCA(n_components=2, random_state=0).fit_transform(x)
        scatter(axes[row, 0], coords, y, f"{name}: biological labels")
        scatter(axes[row, 1], coords, labels, f"MBC clusters, ARI={adjusted_rand_score(y, labels):.3f}")
        rows.append({"dataset": name, "ari": adjusted_rand_score(y, labels)})

    fig.suptitle("Biological expression data: PCA view", y=1.01, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "publication_bio_expression_pca.png")
    pd.DataFrame(rows).to_csv(OUT_DIR / "publication_bio_expression_pca_scores.csv", index=False)


def save_figure(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / name
    fig_path = FIG_DIR / name
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(fig_path, bbox_inches="tight")
    if fig_path.suffix.lower() != ".pdf":
        fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    make_nonconvex_showcase()
    make_morphogenesis_diagram()
    make_bio_pca_polished()
    print("Saved publication figures to", OUT_DIR)


if __name__ == "__main__":
    main()
