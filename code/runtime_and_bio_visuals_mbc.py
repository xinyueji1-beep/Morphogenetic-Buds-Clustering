"""
Runtime analysis and biological-data visualization for MBC.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from external_real_benchmark_mbc import prepare_features, run_algorithm
from morphogenetic_buds_clustering import MorphogeneticBudsClustering
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.datasets import fetch_openml, make_blobs, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


SEEDS = [3, 7, 11]
ALGORITHMS = ["MBC", "KMeans", "GMM", "Agglomerative", "Spectral", "DBSCAN"]


def run_runtime_analysis(out_dir: Path) -> pd.DataFrame:
    rows = []
    sizes = [200, 500, 1000, 2000]
    for n_samples in sizes:
        for seed in SEEDS:
            x, y = make_moons(n_samples=n_samples, noise=0.07, random_state=seed)
            x = StandardScaler().fit_transform(x)
            n_clusters = len(np.unique(y))
            for algorithm in ALGORITHMS:
                start = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    labels = run_runtime_algorithm(algorithm, x, n_clusters, seed)
                rows.append(
                    {
                        "dataset": "two_moons",
                        "n_samples": n_samples,
                        "algorithm": algorithm,
                        "seed": seed,
                        "seconds": time.perf_counter() - start,
                        "ari": adjusted_rand_score(y, labels),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "runtime_scaling_raw.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby(["n_samples", "algorithm"])[["seconds", "ari"]].agg(["mean", "std"]).reset_index()
    summary.to_csv(out_dir / "runtime_scaling_summary.csv", index=False, encoding="utf-8-sig")

    pivot = df.groupby(["n_samples", "algorithm"])["seconds"].mean().reset_index().pivot(
        index="n_samples", columns="algorithm", values="seconds"
    )
    ax = pivot.plot(marker="o", figsize=(9, 5.4))
    ax.set_title("Runtime Scaling on Two Moons")
    ax.set_xlabel("number of samples")
    ax.set_ylabel("seconds")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_scaling.png", dpi=180)
    plt.close()
    return df


def run_runtime_algorithm(name: str, x: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if name == "MBC":
        return MorphogeneticBudsClustering(
            initial_buds=max(12, n_clusters * 6),
            steps=140,
            growth_rate=0.15,
            inhibition=0.045,
            apoptosis_mass=max(2.5, len(x) * 0.0055),
            birth_quantile=0.88,
            organ_merge_radius=3.6,
            valley_ratio=0.18,
            tangent_alignment=0.35,
            max_bridge_void=0.46,
            target_clusters=n_clusters,
            compact_refinement_steps=18,
            compact_refinement_restarts=10,
            compact_refinement_dim_threshold=3,
            random_state=seed,
        ).fit_predict(x)
    if name == "KMeans":
        return KMeans(n_clusters=n_clusters, n_init=30, random_state=seed).fit_predict(x)
    if name == "GMM":
        return GaussianMixture(n_components=n_clusters, random_state=seed).fit_predict(x)
    if name == "Agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(x)
    if name == "Spectral":
        return SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=min(12, len(x) - 1),
            random_state=seed,
        ).fit_predict(x)
    if name == "DBSCAN":
        k = min(6, len(x))
        distances = NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)[0][:, -1]
        eps = float(np.quantile(distances, 0.72))
        return DBSCAN(eps=eps, min_samples=max(4, x.shape[1] + 1)).fit_predict(x)
    raise ValueError(name)


def make_bio_visuals(out_dir: Path) -> None:
    datasets = [
        ("Leukemia_3class", 45091),
        ("Lymphoma_3class", 45094),
    ]
    rows = []
    for name, data_id in datasets:
        try:
            bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        except Exception as exc:
            print(f"[warn] skipped biological visualization {name}: {exc}")
            continue
        x = prepare_features(bunch.data)
        y = LabelEncoder().fit_transform(bunch.target.astype(str))
        labels = run_algorithm("MBC", x, len(np.unique(y)), seed=7)
        coords = PCA(n_components=2, random_state=0).fit_transform(x)
        ari = adjusted_rand_score(y, labels)
        rows.append({"dataset": name, "ari": ari, "n_samples": len(x), "n_features_after_preprocess": x.shape[1]})

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=180)
        axes[0].scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab10", s=34, alpha=0.86, linewidths=0)
        axes[0].set_title(f"{name}: biological labels")
        axes[1].scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=34, alpha=0.86, linewidths=0)
        axes[1].set_title(f"MBC clusters, ARI={ari:.3f}")
        for ax in axes:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / f"bio_visual_{name}.png")
        plt.close(fig)

    pd.DataFrame(rows).to_csv(out_dir / "bio_visual_summary.csv", index=False, encoding="utf-8-sig")


def save_report(out_dir: Path, runtime_df: pd.DataFrame) -> None:
    runtime_summary = runtime_df.groupby(["n_samples", "algorithm"])["seconds"].mean().reset_index().pivot(
        index="n_samples", columns="algorithm", values="seconds"
    )
    ari_summary = runtime_df.groupby(["n_samples", "algorithm"])["ari"].mean().reset_index().pivot(
        index="n_samples", columns="algorithm", values="ari"
    )
    bio_summary = pd.read_csv(out_dir / "bio_visual_summary.csv")
    lines = [
        "# Runtime and Biological Visualization Report",
        "",
        "## Runtime Scaling",
        "",
        runtime_summary.round(4).to_markdown(),
        "",
        "## ARI During Runtime Scaling",
        "",
        ari_summary.round(4).to_markdown(),
        "",
        "## Biological PCA Visualizations",
        "",
        bio_summary.round(4).to_markdown(index=False),
        "",
        "## Figures",
        "",
        "- `runtime_scaling.png`",
        "- `bio_visual_Leukemia_3class.png`",
        "- `bio_visual_Lymphoma_3class.png`",
    ]
    (out_dir / "runtime_and_bio_visuals_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results" / "paper_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_df = run_runtime_analysis(out_dir)
    make_bio_visuals(out_dir)
    save_report(out_dir, runtime_df)
    print((out_dir / "runtime_and_bio_visuals_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
