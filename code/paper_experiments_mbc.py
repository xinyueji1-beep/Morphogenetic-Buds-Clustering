"""
Paper-grade experiments for Morphogenetic Buds Clustering.

This script creates reproducible evidence for a clustering paper:

- multiple benchmark datasets;
- multiple random seeds;
- classical baselines;
- mean/std metrics;
- average algorithm ranks;
- paired Wilcoxon tests against MBC;
- ablation variants of MBC;
- publication-friendly CSV, LaTeX and PNG outputs.

Run:
    python paper_experiments_mbc.py
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
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
from scipy.stats import f, rankdata, wilcoxon, friedmanchisquare
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


SEEDS = [3, 7, 11, 19, 23]
BASELINES = ["KMeans", "GMM", "Agglomerative", "Spectral", "DBSCAN", "CompactRefineOnly"]
MBC_VARIANTS = [
    "MBC",
    "MBC_no_compact",
    "MBC_no_polarity",
    "MBC_no_capillary",
    "MBC_micro_only",
]
ALGORITHMS = ["MBC", *BASELINES]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    x: np.ndarray
    y: np.ndarray
    family: str


def make_datasets(seed: int) -> list[DatasetSpec]:
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    digits = load_digits()

    moons_x, moons_y = make_moons(n_samples=500, noise=0.07, random_state=seed)
    circles_x, circles_y = make_circles(
        n_samples=500,
        factor=0.42,
        noise=0.045,
        random_state=seed,
    )
    varied_x, varied_y = make_varied_density(seed)
    blobs_x, blobs_y = make_blobs(
        n_samples=500,
        centers=4,
        cluster_std=[0.55, 0.8, 1.15, 0.45],
        random_state=seed,
    )
    noisy_moons_x, noisy_moons_y = make_moons(n_samples=500, noise=0.13, random_state=seed + 1)
    noisy_circles_x, noisy_circles_y = make_circles(
        n_samples=500, factor=0.5, noise=0.06, random_state=seed + 1
    )

    mask = digits.target < 5
    specs = [
        DatasetSpec("iris", iris.data, iris.target, "real_low_dim"),
        DatasetSpec("wine", wine.data, wine.target, "real_tabular"),
        DatasetSpec("breast_cancer", cancer.data, cancer.target, "real_tabular"),
        DatasetSpec("digits_0_4", digits.data[mask], digits.target[mask], "real_image_features"),
        DatasetSpec("two_moons", moons_x, moons_y, "nonconvex"),
        DatasetSpec("circles", circles_x, circles_y, "nonconvex"),
        DatasetSpec("noisy_moons", noisy_moons_x, noisy_moons_y, "nonconvex"),
        DatasetSpec("noisy_circles", noisy_circles_x, noisy_circles_y, "nonconvex"),
        DatasetSpec("varied_density", varied_x, varied_y, "synthetic_density"),
        DatasetSpec("anisotropic_blobs", blobs_x @ np.array([[0.6, -0.7], [1.4, 0.35]]), blobs_y, "synthetic_gaussian"),
    ]

    openml_candidates = [
        ("ecoli", "real_tabular"),
        ("glass", "real_tabular"),
        ("vehicle", "real_tabular"),
        ("cmc", "real_tabular"),
        ("wdbc", "real_tabular"),
    ]
    for name, family in openml_candidates:
        loaded = load_openml_clean(name)
        if loaded is None:
            continue
        ox, oy = loaded
        specs.append(DatasetSpec(name, ox, oy, family))

    return specs


def make_varied_density(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=[-1.8, 0.0], scale=[0.18, 0.55], size=(140, 2))
    middle = rng.normal(loc=[0.0, 0.2], scale=[0.62, 0.2], size=(220, 2))
    right = rng.normal(loc=[1.9, -0.1], scale=[0.28, 0.32], size=(140, 2))
    x = np.vstack([left, middle, right])
    y = np.r_[np.zeros(len(left), dtype=int), np.ones(len(middle), dtype=int), np.full(len(right), 2)]
    return x, y


def load_openml_clean(name: str, version: int = 1) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a small OpenML classification dataset as (x, y) with integer labels.

    Returns None when the dataset is unavailable, non-numeric, or too small so
    that the benchmark stays runnable even without network access.
    """
    try:
        bunch = fetch_openml(name=name, version=version, as_frame=False, parser="auto")
        x = np.asarray(bunch.data, dtype=float)
        y_raw = np.asarray(bunch.target)
        if x.ndim != 2 or x.shape[0] != y_raw.shape[0]:
            return None
        if np.isnan(x).any():
            return None
        if y_raw.dtype.kind in ("U", "S", "O"):
            y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(int)
        else:
            y = np.asarray(y_raw, dtype=float)
            if np.isnan(y).any():
                return None
            y = y.astype(int)
        if x.shape[0] < 60 or len(np.unique(y)) < 2:
            return None
        return x, y
    except Exception as exc:  # pragma: no cover - network/format guard
        print(f"[warn] skipped OpenML dataset {name}: {exc}")
        return None


def preprocess(x: np.ndarray) -> np.ndarray:
    x = StandardScaler().fit_transform(x)
    if x.shape[1] > 10:
        n_components = min(10, x.shape[0] - 1, x.shape[1])
        x = PCA(n_components=n_components, random_state=0).fit_transform(x)
    return x


def run_algorithm(name: str, x: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if name == "MBC":
        return run_mbc(x, n_clusters, seed).fit_predict(x)

    if name == "MBC_no_polarity":
        model = run_mbc(x, n_clusters, seed)
        model.tangent_alignment = 0.0
        return model.fit_predict(x)

    if name == "MBC_no_capillary":
        model = run_mbc(x, n_clusters, seed)
        model.max_bridge_void = 1.0
        return model.fit_predict(x)

    if name == "MBC_no_compact":
        model = run_mbc(x, n_clusters, seed)
        model.compact_refinement_steps = 0
        return model.fit_predict(x)

    if name == "MBC_micro_only":
        model = run_mbc(x, n_clusters, seed)
        model.fit(x)
        return model.bud_labels_

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


def run_mbc(x: np.ndarray, n_clusters: int, seed: int) -> MorphogeneticBudsClustering:
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
    )


def evaluate(x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    result = {
        "ari": adjusted_rand_score(y, labels),
        "nmi": normalized_mutual_info_score(y, labels),
        "clusters_found": float(len(set(labels)) - (1 if -1 in labels else 0)),
    }
    valid_x, valid_labels = remove_noise_for_internal_metrics(x, labels)
    if len(set(valid_labels)) >= 2 and len(set(valid_labels)) < len(valid_labels):
        result["silhouette"] = silhouette_score(valid_x, valid_labels)
        result["davies_bouldin"] = davies_bouldin_score(valid_x, valid_labels)
        result["calinski_harabasz"] = calinski_harabasz_score(valid_x, valid_labels)
    else:
        result["silhouette"] = np.nan
        result["davies_bouldin"] = np.nan
        result["calinski_harabasz"] = np.nan
    return result


def remove_noise_for_internal_metrics(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if -1 not in labels:
        return x, labels
    mask = labels != -1
    return x[mask], labels[mask]


def run_full_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_rows = []
    ablation_rows = []
    for seed in SEEDS:
        for spec in make_datasets(seed):
            x = preprocess(spec.x)
            y = spec.y
            n_clusters = len(np.unique(y))

            for algorithm in ALGORITHMS:
                row = run_one(spec, x, y, n_clusters, algorithm, seed)
                benchmark_rows.append(row)

            for variant in MBC_VARIANTS:
                row = run_one(spec, x, y, n_clusters, variant, seed)
                ablation_rows.append(row)

    return pd.DataFrame(benchmark_rows), pd.DataFrame(ablation_rows)


def run_one(
    spec: DatasetSpec,
    x: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    algorithm: str,
    seed: int,
) -> dict:
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = run_algorithm(algorithm, x, n_clusters, seed)
    seconds = time.perf_counter() - start
    row = {
        "dataset": spec.name,
        "family": spec.family,
        "algorithm": algorithm,
        "seed": seed,
        "n_samples": len(x),
        "n_features": x.shape[1],
        "true_clusters": n_clusters,
        "seconds": seconds,
    }
    row.update(evaluate(x, y, labels))
    return row


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["ari", "nmi", "silhouette", "davies_bouldin", "calinski_harabasz", "seconds", "clusters_found"]
    return (
        df.groupby(["dataset", "family", "algorithm"], as_index=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )


def flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["_".join(str(part) for part in col if part) for col in df.columns.to_flat_index()]
    return df


def mean_ari_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("algorithm")["ari"].agg(["mean", "std"]).sort_values("mean", ascending=False)


def family_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["family", "algorithm"])["ari"].mean().reset_index().pivot(
        index="family", columns="algorithm", values="ari"
    )


def rank_table(df: pd.DataFrame) -> pd.DataFrame:
    per_dataset = df.groupby(["dataset", "algorithm"])["ari"].mean().reset_index()
    rows = []
    for dataset, group in per_dataset.groupby("dataset"):
        ranks = rankdata(-group["ari"].to_numpy(), method="average")
        for algorithm, rank in zip(group["algorithm"], ranks):
            rows.append({"dataset": dataset, "algorithm": algorithm, "rank": rank})
    return pd.DataFrame(rows).groupby("algorithm")["rank"].agg(["mean", "std"]).sort_values("mean")


def wilcoxon_against_mbc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mean_scores = df.groupby(["dataset", "algorithm"])["ari"].mean().reset_index()
    mbc = mean_scores[mean_scores["algorithm"] == "MBC"].set_index("dataset")["ari"]
    for algorithm in BASELINES:
        other = mean_scores[mean_scores["algorithm"] == algorithm].set_index("dataset")["ari"]
        common = mbc.index.intersection(other.index)
        if len(common) < 2:
            continue
        statistic, p_value = wilcoxon(mbc.loc[common], other.loc[common], zero_method="wilcox")
        rows.append(
            {
                "comparison": f"MBC vs {algorithm}",
                "mbc_mean": float(mbc.loc[common].mean()),
                "other_mean": float(other.loc[common].mean()),
                "mean_difference": float((mbc.loc[common] - other.loc[common]).mean()),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


# Nemenyi critical q-values (two-sided, alpha=0.05) for k=2..20 algorithms.
NEMENYI_Q = {
    2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.312, 14: 3.352,
    15: 3.389, 16: 3.422, 17: 3.452, 18: 3.480, 19: 3.506, 20: 3.530,
}


def friedman_nemenyi(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Friedman omnibus test (Iman-Davenport F correction) + Nemenyi CD.

    Returns a scalar dict and a per-algorithm rank table with significance flags
    relative to the best-ranked algorithm.
    """
    per_dataset = df.groupby(["dataset", "algorithm"])["ari"].mean().reset_index()
    algorithms = sorted(per_dataset["algorithm"].unique())
    k = len(algorithms)
    datasets = sorted(per_dataset["dataset"].unique())
    n = len(datasets)

    # Per-dataset average ranks (used for the Nemenyi post-hoc comparison)
    rank_sum = {a: 0.0 for a in algorithms}
    for ds in datasets:
        sub = per_dataset[per_dataset["dataset"] == ds]
        r = rankdata(-sub["ari"].to_numpy(), method="average")
        for a, rank in zip(sub["algorithm"], r):
            rank_sum[a] += rank
    avg_rank = {a: rank_sum[a] / n for a in algorithms}

    # Friedman omnibus test via scipy (canonical computation with tie correction)
    wide = per_dataset.pivot(index="dataset", columns="algorithm", values="ari")
    wide = wide[algorithms]
    cols = [wide[a].to_numpy(dtype=float) for a in algorithms]
    chi2, p_value = friedmanchisquare(*cols)
    df1 = k - 1
    df2 = (k - 1) * (n - 1)
    if n * df1 - chi2 <= 0:
        F = float("inf")
    else:
        F = (n - 1) * chi2 / (n * df1 - chi2)

    # Nemenyi critical difference for pairwise post-hoc comparison
    q = NEMENYI_Q.get(k, 3.530)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))

    best_rank = min(avg_rank.values())
    rows = []
    for a in sorted(algorithms, key=lambda x: avg_rank[x]):
        diff = avg_rank[a] - best_rank
        rows.append(
            {
                "algorithm": a,
                "average_rank": float(avg_rank[a]),
                "diff_from_best": float(diff),
                "significant_vs_best": bool(diff > cd),
            }
        )
    ranks_df = pd.DataFrame(rows)

    scalars = {
        "n_datasets": n,
        "n_algorithms": k,
        "friedman_chi2": float(chi2),
        "iman_davenport_F": float(F),
        "p_value": float(p_value),
        "nemenyi_cd": float(cd),
    }
    return scalars, ranks_df


def save_outputs(benchmark_df: pd.DataFrame, ablation_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_df.to_csv(out_dir / "paper_benchmark_raw.csv", index=False, encoding="utf-8-sig")
    ablation_df.to_csv(out_dir / "paper_ablation_raw.csv", index=False, encoding="utf-8-sig")

    benchmark_summary = flat_columns(aggregate(benchmark_df))
    ablation_summary = flat_columns(aggregate(ablation_df))
    benchmark_summary.to_csv(out_dir / "paper_benchmark_summary.csv", index=False, encoding="utf-8-sig")
    ablation_summary.to_csv(out_dir / "paper_ablation_summary.csv", index=False, encoding="utf-8-sig")

    mean_ari = mean_ari_table(benchmark_df)
    families = family_table(benchmark_df)
    ranks = rank_table(benchmark_df)
    tests = wilcoxon_against_mbc(benchmark_df)
    friedman_scalars, friedman_ranks = friedman_nemenyi(benchmark_df)

    mean_ari.to_csv(out_dir / "paper_mean_ari_ranking.csv", encoding="utf-8-sig")
    families.to_csv(out_dir / "paper_family_ari.csv", encoding="utf-8-sig")
    ranks.to_csv(out_dir / "paper_average_ranks.csv", encoding="utf-8-sig")
    tests.to_csv(out_dir / "paper_wilcoxon_vs_mbc.csv", index=False, encoding="utf-8-sig")
    friedman_ranks.to_csv(out_dir / "paper_friedman_nemenyi.csv", index=False, encoding="utf-8-sig")
    (out_dir / "paper_friedman_scalars.json").write_text(
        json.dumps(friedman_scalars, indent=2), encoding="utf-8"
    )

    save_latex_tables(out_dir, mean_ari, families, ranks, tests, friedman_scalars, friedman_ranks)
    save_figures(out_dir, benchmark_df, ablation_df)
    save_markdown_report(
        out_dir, mean_ari, families, ranks, tests, benchmark_df, ablation_df, friedman_scalars, friedman_ranks
    )


def save_latex_tables(
    out_dir: Path,
    mean_ari: pd.DataFrame,
    families: pd.DataFrame,
    ranks: pd.DataFrame,
    tests: pd.DataFrame,
    friedman_scalars: dict,
    friedman_ranks: pd.DataFrame,
) -> None:
    latex = {
        "mean_ari": mean_ari.round(4).to_latex(float_format="%.4f"),
        "family_ari": families.round(4).to_latex(float_format="%.4f"),
        "average_ranks": ranks.round(4).to_latex(float_format="%.4f"),
        "wilcoxon": tests.round(4).to_latex(index=False, float_format="%.4f"),
        "friedman_ranks": friedman_ranks.round(4).to_latex(index=False, float_format="%.4f"),
    }
    (out_dir / "paper_tables.tex").write_text(
        "\n\n".join(f"% {name}\n{table}" for name, table in latex.items()),
        encoding="utf-8",
    )


def save_figures(out_dir: Path, benchmark_df: pd.DataFrame, ablation_df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 140,
        }
    )
    dataset_pivot = benchmark_df.groupby(["dataset", "algorithm"])["ari"].mean().reset_index().pivot(
        index="dataset", columns="algorithm", values="ari"
    )
    dataset_pivot = dataset_pivot.reindex(
        [
            "two_moons", "circles", "noisy_moons", "noisy_circles", "anisotropic_blobs",
            "varied_density", "iris", "wine", "breast_cancer", "digits_0_4",
            "ecoli", "glass", "vehicle", "cmc", "wdbc",
        ]
    )
    ax = dataset_pivot.plot(kind="bar", figsize=(12, 5.8), width=0.82)
    ax.set_title("Adjusted Rand Index Across Benchmark Datasets")
    ax.set_ylabel("ARI")
    ax.set_xlabel("")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(out_dir / "paper_fig_ari_by_dataset.png", dpi=180)
    plt.close()

    family_pivot = family_table(benchmark_df)
    family_pivot = family_pivot.reindex(
        [
            "nonconvex",
            "synthetic_gaussian",
            "synthetic_density",
            "real_low_dim",
            "real_tabular",
            "real_image_features",
        ]
    )
    ax = family_pivot.plot(kind="barh", figsize=(10.5, 5.6), width=0.8)
    ax.set_title("Mean ARI by Dataset Family")
    ax.set_xlabel("ARI")
    ax.set_ylabel("")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "paper_fig_ari_by_family.png", dpi=180)
    plt.close()

    ablation_pivot = ablation_df.groupby(["dataset", "algorithm"])["ari"].mean().reset_index().pivot(
        index="dataset", columns="algorithm", values="ari"
    )
    ablation_pivot = ablation_pivot.reindex(dataset_pivot.index)
    ax = ablation_pivot.plot(kind="bar", figsize=(11, 5.6), width=0.82)
    ax.set_title("Ablation Study of MBC Components")
    ax.set_ylabel("ARI")
    ax.set_xlabel("")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(out_dir / "paper_fig_ablation.png", dpi=180)
    plt.close()


def save_markdown_report(
    out_dir: Path,
    mean_ari: pd.DataFrame,
    families: pd.DataFrame,
    ranks: pd.DataFrame,
    tests: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    friedman_scalars: dict,
    friedman_ranks: pd.DataFrame,
) -> None:
    mbc_family = families["MBC"].sort_values(ascending=False)
    ablation_mean = mean_ari_table(ablation_df)
    present_datasets = sorted(benchmark_df["dataset"].unique())
    friedman_lines = [
        f"- Datasets (N={friedman_scalars['n_datasets']}), Algorithms (k={friedman_scalars['n_algorithms']})",
        f"- Friedman chi-squared = {friedman_scalars['friedman_chi2']:.4f}",
        f"- Iman-Davenport F = {friedman_scalars['iman_davenport_F']:.4f}, p-value = {friedman_scalars['p_value']:.4f}",
        f"- Nemenyi critical difference (CD) = {friedman_scalars['nemenyi_cd']:.4f} (average-rank units)",
        "- Algorithms whose average rank differs from the best by more than CD are flagged significant_vs_best above.",
    ]
    report = [
        "# Paper Experiment Report for Morphogenetic Buds Clustering",
        "",
        "## Experimental Design",
        "",
        f"- Seeds: {SEEDS}",
        f"- Datasets ({len(present_datasets)}): {', '.join(present_datasets)}.",
        "- Baselines: KMeans, GMM, Agglomerative, Spectral, DBSCAN.",
        "- Metrics: ARI, NMI, silhouette, Davies-Bouldin, Calinski-Harabasz, runtime.",
        "- Ground-truth labels are used only for external evaluation, never during clustering.",
        "",
        "## Main Mean ARI Ranking",
        "",
        mean_ari.round(4).to_markdown(),
        "",
        "## Average Rank by ARI",
        "",
        ranks.round(4).to_markdown(),
        "",
        "## Mean ARI by Dataset Family",
        "",
        families.round(4).to_markdown(),
        "",
        "## MBC Strength Profile",
        "",
        mbc_family.round(4).to_markdown(),
        "",
        "## Wilcoxon Tests Against MBC",
        "",
        tests.round(4).to_markdown(index=False),
        "",
        "## Friedman Test and Nemenyi Critical Difference",
        "",
        "\n".join(friedman_lines),
        "",
        friedman_ranks.round(4).to_markdown(index=False),
        "",
        "## Ablation Mean ARI",
        "",
        ablation_mean.round(4).to_markdown(),
        "",
        "## Interpretation",
        "",
        "MBC is strongest on nonconvex continuous structures, where morphogenetic continuity is useful.",
        "It is weaker on some high-dimensional tabular datasets because the current organ-fusion rule can over-segment compact classes.",
        "This supports positioning MBC as a morphology-aware clustering method rather than a universal replacement for all clustering algorithms.",
    ]
    (out_dir / "paper_experiment_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results" / "paper_results"
    benchmark_df, ablation_df = run_full_experiment()
    save_outputs(benchmark_df, ablation_df, out_dir)
    summary = {
        "output_dir": str(out_dir),
        "benchmark_rows": int(len(benchmark_df)),
        "ablation_rows": int(len(ablation_df)),
        "mean_ari": mean_ari_table(benchmark_df)["mean"].round(4).to_dict(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
