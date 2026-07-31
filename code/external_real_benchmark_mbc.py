"""
External real-data benchmark for Morphogenetic Buds Clustering.

This script adds OpenML datasets, including the biologically motivated
MiceProtein dataset, to strengthen the paper evidence.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from morphogenetic_buds_clustering import (
    MorphogeneticBudsClustering,
    kmeanspp_centers,
    refine_centers,
)
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


SEEDS = [3, 7, 11]
ALGORITHMS = [
    "MBC",
    "KMeans",
    "GMM",
    "Agglomerative",
    "Spectral",
    "DBSCAN",
    "CompactRefineOnly",
]

OPENML_DATASETS = [
    {"name": "MiceProtein_8class", "id": 40966, "family": "biological_protein_expression", "target_mode": "class"},
    {"name": "MiceProtein_genotype", "id": 40966, "family": "biological_factor", "target_mode": "prefix"},
    {"name": "MiceProtein_behavior", "id": 40966, "family": "biological_factor", "target_mode": "middle"},
    {"name": "MiceProtein_treatment", "id": 40966, "family": "biological_factor", "target_mode": "suffix"},
    {"name": "Leukemia_2class", "id": 45090, "family": "biological_gene_expression"},
    {"name": "Leukemia_3class", "id": 45091, "family": "biological_gene_expression"},
    {"name": "Lymphoma_3class", "id": 45094, "family": "biological_gene_expression"},
    {"name": "credit-g", "id": 31, "family": "real_mixed_tabular"},
    {"name": "diabetes", "id": 37, "family": "real_medical_tabular"},
    {"name": "vehicle", "id": 54, "family": "real_shape_features"},
    {"name": "ionosphere", "id": 59, "family": "real_signal_features"},
    {"name": "blood-transfusion", "id": 1464, "family": "real_medical_tabular"},
]


def load_openml_dataset(spec: dict) -> tuple[str, str, np.ndarray, np.ndarray]:
    bunch = fetch_openml(data_id=spec["id"], as_frame=True, parser="auto")
    x = prepare_features(bunch.data)
    target = bunch.target.astype(str)
    if spec.get("target_mode") == "prefix":
        target = target.str.split("-").str[0]
    elif spec.get("target_mode") == "middle":
        target = target.str.split("-").str[1]
    elif spec.get("target_mode") == "suffix":
        target = target.str.split("-").str[2]
    y = LabelEncoder().fit_transform(target)
    return spec["name"], spec["family"], x, y


def prepare_features(frame: pd.DataFrame) -> np.ndarray:
    frame = frame.copy()
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in frame.columns if col not in numeric_cols]

    parts = []
    if numeric_cols:
        numeric = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(numeric.median(numeric_only=True))
        parts.append(numeric)

    if categorical_cols:
        categorical = frame[categorical_cols].astype("object")
        categorical = categorical.fillna(categorical.mode(dropna=True).iloc[0])
        parts.append(pd.get_dummies(categorical, drop_first=False, dtype=float))

    x = pd.concat(parts, axis=1).to_numpy(dtype=float)
    x = StandardScaler().fit_transform(x)
    if x.shape[1] > 12:
        n_components = min(12, x.shape[1], x.shape[0] - 1)
        x = PCA(n_components=n_components, random_state=0).fit_transform(x)
    return x


def run_algorithm(name: str, x: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if name == "MBC":
        model = MorphogeneticBudsClustering(
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
        )
        return model.fit_predict(x)

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

    if name == "CompactRefineOnly":
        rng = np.random.default_rng(seed)
        best_labels = None
        best_inertia = np.inf
        for _ in range(10):
            centers = kmeanspp_centers(x, n_clusters, rng)
            labels, _, inertia = refine_centers(x, centers, 18)
            if inertia < best_inertia:
                best_labels = labels
                best_inertia = inertia
        return best_labels

    raise ValueError(name)


def valid_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    if -1 in labels:
        mask = labels != -1
        x = x[mask]
        labels = labels[mask]
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return float("nan")
    return float(silhouette_score(x, labels))


def benchmark() -> pd.DataFrame:
    rows = []
    for spec in OPENML_DATASETS:
        try:
            dataset_name, family, x, y = load_openml_dataset(spec)
        except Exception as exc:
            print(f"[warn] skipped OpenML dataset {spec['name']}: {exc}")
            continue
        n_clusters = len(np.unique(y))
        for seed in SEEDS:
            for algorithm in ALGORITHMS:
                start = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    labels = run_algorithm(algorithm, x, n_clusters, seed)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "family": family,
                        "algorithm": algorithm,
                        "seed": seed,
                        "n_samples": len(x),
                        "n_features_after_preprocess": x.shape[1],
                        "true_clusters": n_clusters,
                        "clusters_found": len(set(labels)) - (1 if -1 in labels else 0),
                        "ari": adjusted_rand_score(y, labels),
                        "nmi": normalized_mutual_info_score(y, labels),
                        "silhouette": valid_silhouette(x, labels),
                        "seconds": time.perf_counter() - start,
                    }
                )
    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "external_real_benchmark_raw.csv"
    summary_path = out_dir / "external_real_benchmark_summary.csv"
    report_path = out_dir / "external_real_benchmark_report.md"
    fig_path = out_dir / "external_real_benchmark_ari.png"

    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary = (
        df.groupby(["dataset", "family", "algorithm"])["ari"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["dataset", "mean"], ascending=[True, False])
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    ranking = df.groupby("algorithm")["ari"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    biological = df[df["family"] == "biological_protein_expression"].groupby("algorithm")["ari"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    family = df.groupby(["family", "algorithm"])["ari"].mean().reset_index().pivot(index="family", columns="algorithm", values="ari")
    gene_expression = df[df["family"] == "biological_gene_expression"].groupby("algorithm")["ari"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    pivot = df.groupby(["dataset", "algorithm"])["ari"].mean().reset_index().pivot(index="dataset", columns="algorithm", values="ari")

    ax = pivot.plot(kind="bar", figsize=(12, 5.8), width=0.82)
    ax.set_title("External Real-Data Benchmark")
    ax.set_ylabel("ARI")
    ax.set_xlabel("")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    lines = [
        "# External Real-Data Benchmark",
        "",
        "## Overall Mean ARI",
        "",
        ranking.round(4).to_markdown(),
        "",
        "## Biological Dataset: MiceProtein",
        "",
        biological.round(4).to_markdown(),
        "",
        "## Biological Gene Expression Datasets",
        "",
        gene_expression.round(4).to_markdown(),
        "",
        "## Mean ARI by Family",
        "",
        family.round(4).to_markdown(),
        "",
        "## ARI by Dataset",
        "",
        pivot.round(4).to_markdown(),
        "",
        "## Files",
        "",
        f"- Raw results: `{raw_path.name}`",
        f"- Summary: `{summary_path.name}`",
        f"- Figure: `{fig_path.name}`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"raw": str(raw_path), "summary": str(summary_path), "report": str(report_path), "figure": str(fig_path)}, indent=2))
    print(ranking.round(4).to_string())


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results" / "paper_results"
    df = benchmark()
    save_outputs(df, out_dir)


if __name__ == "__main__":
    main()
