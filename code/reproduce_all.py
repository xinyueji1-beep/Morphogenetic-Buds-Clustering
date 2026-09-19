from __future__ import annotations

"""Reproduce all MBC manuscript and reviewer analyses from one script.

Run from the package root:
    python code/reproduce_all.py

The MBC implementation itself is in code/mbc.py. This file contains the
benchmark, robustness, external-case, figure, and reviewer-specific routines
as separately labeled sections so that the complete study remains inspectable
without navigating a collection of standalone scripts.
"""

import argparse
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PACKAGE_ROOT / "results" / "paper_results"



# ============================================================================
# Main benchmark and statistical analysis
# ============================================================================

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
import json as main_benchmark_json
import time as main_benchmark_time
import warnings as main_benchmark_warnings
from dataclasses import dataclass as main_benchmark_dataclass
from pathlib import Path as main_benchmark_Path
import matplotlib as main_benchmark_matplotlib
main_benchmark_matplotlib.use('Agg')
import matplotlib.pyplot as main_benchmark_plt
import numpy as main_benchmark_np
import pandas as main_benchmark_pd
from mbc import MorphogeneticBudsClustering as main_benchmark_MorphogeneticBudsClustering, kmeanspp_centers as main_benchmark_kmeanspp_centers, refine_centers as main_benchmark_refine_centers
from scipy.stats import f as main_benchmark_f, rankdata as main_benchmark_rankdata, wilcoxon as main_benchmark_wilcoxon, friedmanchisquare as main_benchmark_friedmanchisquare
from sklearn.cluster import AgglomerativeClustering as main_benchmark_AgglomerativeClustering, DBSCAN as main_benchmark_DBSCAN, KMeans as main_benchmark_KMeans, SpectralClustering as main_benchmark_SpectralClustering
from sklearn.datasets import fetch_openml as main_benchmark_fetch_openml, load_breast_cancer as main_benchmark_load_breast_cancer, load_digits as main_benchmark_load_digits, load_iris as main_benchmark_load_iris, load_wine as main_benchmark_load_wine, make_blobs as main_benchmark_make_blobs, make_circles as main_benchmark_make_circles, make_moons as main_benchmark_make_moons
from sklearn.decomposition import PCA as main_benchmark_PCA
from sklearn.metrics import adjusted_rand_score as main_benchmark_adjusted_rand_score, calinski_harabasz_score as main_benchmark_calinski_harabasz_score, davies_bouldin_score as main_benchmark_davies_bouldin_score, normalized_mutual_info_score as main_benchmark_normalized_mutual_info_score, silhouette_score as main_benchmark_silhouette_score
from sklearn.mixture import GaussianMixture as main_benchmark_GaussianMixture
from sklearn.neighbors import NearestNeighbors as main_benchmark_NearestNeighbors
from sklearn.preprocessing import LabelEncoder as main_benchmark_LabelEncoder, StandardScaler as main_benchmark_StandardScaler
main_benchmark_PACKAGE_ROOT = main_benchmark_Path(__file__).resolve().parents[1]
main_benchmark_LOCAL_MAIN_DATA = main_benchmark_PACKAGE_ROOT / 'data' / 'raw' / 'main_benchmark'
main_benchmark_SEEDS = [3, 7, 11, 19, 23]
main_benchmark_BASELINES = ['KMeans', 'GMM', 'Agglomerative', 'Spectral', 'DBSCAN', 'CompactRefineOnly']
main_benchmark_MBC_VARIANTS = ['MBC', 'MBC_no_compact', 'MBC_no_polarity', 'MBC_no_capillary', 'MBC_micro_only']
main_benchmark_ALGORITHMS = ['MBC', *main_benchmark_BASELINES]

@main_benchmark_dataclass(frozen=True)
class main_benchmark_DatasetSpec:
    name: str
    x: main_benchmark_np.ndarray
    y: main_benchmark_np.ndarray
    family: str

def main_benchmark_make_datasets(seed: int) -> list[main_benchmark_DatasetSpec]:
    iris = main_benchmark_load_iris()
    wine = main_benchmark_load_wine()
    cancer = main_benchmark_load_breast_cancer()
    digits = main_benchmark_load_digits()
    moons_x, moons_y = main_benchmark_make_moons(n_samples=500, noise=0.07, random_state=seed)
    circles_x, circles_y = main_benchmark_make_circles(n_samples=500, factor=0.42, noise=0.045, random_state=seed)
    varied_x, varied_y = main_benchmark_make_varied_density(seed)
    blobs_x, blobs_y = main_benchmark_make_blobs(n_samples=500, centers=4, cluster_std=[0.55, 0.8, 1.15, 0.45], random_state=seed)
    noisy_moons_x, noisy_moons_y = main_benchmark_make_moons(n_samples=500, noise=0.13, random_state=seed + 1)
    noisy_circles_x, noisy_circles_y = main_benchmark_make_circles(n_samples=500, factor=0.5, noise=0.06, random_state=seed + 1)
    mask = digits.target < 5
    specs = [main_benchmark_DatasetSpec('iris', iris.data, iris.target, 'real_low_dim'), main_benchmark_DatasetSpec('wine', wine.data, wine.target, 'real_tabular'), main_benchmark_DatasetSpec('breast_cancer', cancer.data, cancer.target, 'real_tabular'), main_benchmark_DatasetSpec('digits_0_4', digits.data[mask], digits.target[mask], 'real_image_features'), main_benchmark_DatasetSpec('two_moons', moons_x, moons_y, 'nonconvex'), main_benchmark_DatasetSpec('circles', circles_x, circles_y, 'nonconvex'), main_benchmark_DatasetSpec('noisy_moons', noisy_moons_x, noisy_moons_y, 'nonconvex'), main_benchmark_DatasetSpec('noisy_circles', noisy_circles_x, noisy_circles_y, 'nonconvex'), main_benchmark_DatasetSpec('varied_density', varied_x, varied_y, 'synthetic_density'), main_benchmark_DatasetSpec('anisotropic_blobs', blobs_x @ main_benchmark_np.array([[0.6, -0.7], [1.4, 0.35]]), blobs_y, 'synthetic_gaussian')]
    openml_candidates = [('ecoli', 'real_tabular'), ('glass', 'real_tabular'), ('vehicle', 'real_tabular'), ('cmc', 'real_tabular'), ('wdbc', 'real_tabular')]
    for name, family in openml_candidates:
        loaded = main_benchmark_load_openml_clean(name, seed=seed)
        if loaded is None:
            continue
        ox, oy = loaded
        specs.append(main_benchmark_DatasetSpec(name, ox, oy, family))
    return specs

def main_benchmark_make_varied_density(seed: int) -> tuple[main_benchmark_np.ndarray, main_benchmark_np.ndarray]:
    rng = main_benchmark_np.random.default_rng(seed)
    left = rng.normal(loc=[-1.8, 0.0], scale=[0.18, 0.55], size=(140, 2))
    middle = rng.normal(loc=[0.0, 0.2], scale=[0.62, 0.2], size=(220, 2))
    right = rng.normal(loc=[1.9, -0.1], scale=[0.28, 0.32], size=(140, 2))
    x = main_benchmark_np.vstack([left, middle, right])
    y = main_benchmark_np.r_[main_benchmark_np.zeros(len(left), dtype=int), main_benchmark_np.ones(len(middle), dtype=int), main_benchmark_np.full(len(right), 2)]
    return (x, y)

def main_benchmark_load_openml_clean(name: str, version: int=1, seed: int | None=None) -> tuple[main_benchmark_np.ndarray, main_benchmark_np.ndarray] | None:
    """Load a small OpenML classification dataset as (x, y) with integer labels.

    Returns None when the dataset is unavailable, non-numeric, or too small so
    that the benchmark stays runnable even without network access.
    """
    if seed is not None:
        local_path = main_benchmark_LOCAL_MAIN_DATA / f'{name}_seed{seed}.csv'
        if local_path.exists():
            frame = main_benchmark_pd.read_csv(local_path)
            return (frame.iloc[:, 1:].to_numpy(dtype=float), frame.iloc[:, 0].to_numpy(dtype=int))
    try:
        bunch = main_benchmark_fetch_openml(name=name, version=version, as_frame=False, parser='auto')
        x = main_benchmark_np.asarray(bunch.data, dtype=float)
        y_raw = main_benchmark_np.asarray(bunch.target)
        if x.ndim != 2 or x.shape[0] != y_raw.shape[0]:
            return None
        if main_benchmark_np.isnan(x).any():
            return None
        if y_raw.dtype.kind in ('U', 'S', 'O'):
            y = main_benchmark_LabelEncoder().fit_transform(y_raw.astype(str)).astype(int)
        else:
            y = main_benchmark_np.asarray(y_raw, dtype=float)
            if main_benchmark_np.isnan(y).any():
                return None
            y = y.astype(int)
        if x.shape[0] < 60 or len(main_benchmark_np.unique(y)) < 2:
            return None
        return (x, y)
    except Exception as exc:
        print(f'[warn] skipped OpenML dataset {name}: {exc}')
        return None

def main_benchmark_preprocess(x: main_benchmark_np.ndarray) -> main_benchmark_np.ndarray:
    x = main_benchmark_StandardScaler().fit_transform(x)
    if x.shape[1] > 10:
        n_components = min(10, x.shape[0] - 1, x.shape[1])
        x = main_benchmark_PCA(n_components=n_components, random_state=0).fit_transform(x)
    return x

def main_benchmark_run_algorithm(name: str, x: main_benchmark_np.ndarray, n_clusters: int, seed: int) -> main_benchmark_np.ndarray:
    if name == 'MBC':
        return main_benchmark_run_mbc(x, n_clusters, seed).fit_predict(x)
    if name == 'MBC_no_polarity':
        model = main_benchmark_run_mbc(x, n_clusters, seed)
        model.tangent_alignment = 0.0
        return model.fit_predict(x)
    if name == 'MBC_no_capillary':
        model = main_benchmark_run_mbc(x, n_clusters, seed)
        model.max_bridge_void = 1.0
        return model.fit_predict(x)
    if name == 'MBC_no_compact':
        model = main_benchmark_run_mbc(x, n_clusters, seed)
        model.compact_refinement_steps = 0
        return model.fit_predict(x)
    if name == 'MBC_micro_only':
        model = main_benchmark_run_mbc(x, n_clusters, seed)
        model.fit(x)
        return model.bud_labels_
    if name == 'KMeans':
        return main_benchmark_KMeans(n_clusters=n_clusters, n_init=30, random_state=seed).fit_predict(x)
    if name == 'GMM':
        return main_benchmark_GaussianMixture(n_components=n_clusters, random_state=seed).fit_predict(x)
    if name == 'Agglomerative':
        return main_benchmark_AgglomerativeClustering(n_clusters=n_clusters).fit_predict(x)
    if name == 'Spectral':
        return main_benchmark_SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=min(12, len(x) - 1), random_state=seed).fit_predict(x)
    if name == 'DBSCAN':
        k = min(6, len(x))
        distances = main_benchmark_NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)[0][:, -1]
        eps = float(main_benchmark_np.quantile(distances, 0.72))
        return main_benchmark_DBSCAN(eps=eps, min_samples=max(4, x.shape[1] + 1)).fit_predict(x)
    if name == 'CompactRefineOnly':
        rng = main_benchmark_np.random.default_rng(seed)
        best_labels = None
        best_inertia = main_benchmark_np.inf
        for _ in range(10):
            centers = main_benchmark_kmeanspp_centers(x, n_clusters, rng)
            labels, _, inertia = main_benchmark_refine_centers(x, centers, 18)
            if inertia < best_inertia:
                best_labels = labels
                best_inertia = inertia
        return best_labels
    raise ValueError(name)

def main_benchmark_run_mbc(x: main_benchmark_np.ndarray, n_clusters: int, seed: int) -> main_benchmark_MorphogeneticBudsClustering:
    return main_benchmark_MorphogeneticBudsClustering(initial_buds=max(12, n_clusters * 6), steps=140, growth_rate=0.15, inhibition=0.045, apoptosis_mass=max(2.5, len(x) * 0.0055), birth_quantile=0.88, organ_merge_radius=3.6, valley_ratio=0.18, tangent_alignment=0.35, max_bridge_void=0.46, target_clusters=n_clusters, compact_refinement_steps=18, compact_refinement_restarts=10, compact_refinement_dim_threshold=3, random_state=seed)

def main_benchmark_evaluate(x: main_benchmark_np.ndarray, y: main_benchmark_np.ndarray, labels: main_benchmark_np.ndarray) -> dict[str, float]:
    result = {'ari': main_benchmark_adjusted_rand_score(y, labels), 'nmi': main_benchmark_normalized_mutual_info_score(y, labels), 'clusters_found': float(len(set(labels)) - (1 if -1 in labels else 0))}
    valid_x, valid_labels = main_benchmark_remove_noise_for_internal_metrics(x, labels)
    if len(set(valid_labels)) >= 2 and len(set(valid_labels)) < len(valid_labels):
        result['silhouette'] = main_benchmark_silhouette_score(valid_x, valid_labels)
        result['davies_bouldin'] = main_benchmark_davies_bouldin_score(valid_x, valid_labels)
        result['calinski_harabasz'] = main_benchmark_calinski_harabasz_score(valid_x, valid_labels)
    else:
        result['silhouette'] = main_benchmark_np.nan
        result['davies_bouldin'] = main_benchmark_np.nan
        result['calinski_harabasz'] = main_benchmark_np.nan
    return result

def main_benchmark_remove_noise_for_internal_metrics(x: main_benchmark_np.ndarray, labels: main_benchmark_np.ndarray) -> tuple[main_benchmark_np.ndarray, main_benchmark_np.ndarray]:
    if -1 not in labels:
        return (x, labels)
    mask = labels != -1
    return (x[mask], labels[mask])

def main_benchmark_run_full_experiment() -> tuple[main_benchmark_pd.DataFrame, main_benchmark_pd.DataFrame]:
    benchmark_rows = []
    ablation_rows = []
    for seed in main_benchmark_SEEDS:
        for spec in main_benchmark_make_datasets(seed):
            x = main_benchmark_preprocess(spec.x)
            y = spec.y
            n_clusters = len(main_benchmark_np.unique(y))
            for algorithm in main_benchmark_ALGORITHMS:
                row = main_benchmark_run_one(spec, x, y, n_clusters, algorithm, seed)
                benchmark_rows.append(row)
            for variant in main_benchmark_MBC_VARIANTS:
                row = main_benchmark_run_one(spec, x, y, n_clusters, variant, seed)
                ablation_rows.append(row)
    return (main_benchmark_pd.DataFrame(benchmark_rows), main_benchmark_pd.DataFrame(ablation_rows))

def main_benchmark_run_one(spec: main_benchmark_DatasetSpec, x: main_benchmark_np.ndarray, y: main_benchmark_np.ndarray, n_clusters: int, algorithm: str, seed: int) -> dict:
    start = main_benchmark_time.perf_counter()
    with main_benchmark_warnings.catch_warnings():
        main_benchmark_warnings.simplefilter('ignore')
        labels = main_benchmark_run_algorithm(algorithm, x, n_clusters, seed)
    seconds = main_benchmark_time.perf_counter() - start
    row = {'dataset': spec.name, 'family': spec.family, 'algorithm': algorithm, 'seed': seed, 'n_samples': len(x), 'n_features': x.shape[1], 'true_clusters': n_clusters, 'seconds': seconds}
    row.update(main_benchmark_evaluate(x, y, labels))
    return row

def main_benchmark_aggregate(df: main_benchmark_pd.DataFrame) -> main_benchmark_pd.DataFrame:
    metrics = ['ari', 'nmi', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'seconds', 'clusters_found']
    return df.groupby(['dataset', 'family', 'algorithm'], as_index=False)[metrics].agg(['mean', 'std']).reset_index()

def main_benchmark_flat_columns(df: main_benchmark_pd.DataFrame) -> main_benchmark_pd.DataFrame:
    df = df.copy()
    df.columns = ['_'.join((str(part) for part in col if part)) for col in df.columns.to_flat_index()]
    return df

def main_benchmark_mean_ari_table(df: main_benchmark_pd.DataFrame) -> main_benchmark_pd.DataFrame:
    return df.groupby('algorithm')['ari'].agg(['mean', 'std']).sort_values('mean', ascending=False)

def main_benchmark_family_table(df: main_benchmark_pd.DataFrame) -> main_benchmark_pd.DataFrame:
    return df.groupby(['family', 'algorithm'])['ari'].mean().reset_index().pivot(index='family', columns='algorithm', values='ari')

def main_benchmark_rank_table(df: main_benchmark_pd.DataFrame) -> main_benchmark_pd.DataFrame:
    per_dataset = df.groupby(['dataset', 'algorithm'])['ari'].mean().reset_index()
    rows = []
    for dataset, group in per_dataset.groupby('dataset'):
        ranks = main_benchmark_rankdata(-group['ari'].to_numpy(), method='average')
        for algorithm, rank in zip(group['algorithm'], ranks):
            rows.append({'dataset': dataset, 'algorithm': algorithm, 'rank': rank})
    return main_benchmark_pd.DataFrame(rows).groupby('algorithm')['rank'].agg(['mean', 'std']).sort_values('mean')

def main_benchmark_wilcoxon_against_mbc(df: main_benchmark_pd.DataFrame) -> main_benchmark_pd.DataFrame:
    rows = []
    mean_scores = df.groupby(['dataset', 'algorithm'])['ari'].mean().reset_index()
    mbc = mean_scores[mean_scores['algorithm'] == 'MBC'].set_index('dataset')['ari']
    for algorithm in main_benchmark_BASELINES:
        other = mean_scores[mean_scores['algorithm'] == algorithm].set_index('dataset')['ari']
        common = mbc.index.intersection(other.index)
        if len(common) < 2:
            continue
        statistic, p_value = main_benchmark_wilcoxon(mbc.loc[common], other.loc[common], zero_method='wilcox')
        rows.append({'comparison': f'MBC vs {algorithm}', 'mbc_mean': float(mbc.loc[common].mean()), 'other_mean': float(other.loc[common].mean()), 'mean_difference': float((mbc.loc[common] - other.loc[common]).mean()), 'wilcoxon_statistic': float(statistic), 'p_value': float(p_value)})
    return main_benchmark_pd.DataFrame(rows)
main_benchmark_NEMENYI_Q = {2: 1.96, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.85, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.312, 14: 3.352, 15: 3.389, 16: 3.422, 17: 3.452, 18: 3.48, 19: 3.506, 20: 3.53}

def main_benchmark_friedman_nemenyi(df: main_benchmark_pd.DataFrame) -> tuple[dict, main_benchmark_pd.DataFrame]:
    """Friedman omnibus test (Iman-Davenport F correction) + Nemenyi CD.

    Returns a scalar dict and a per-algorithm rank table with significance flags
    relative to the best-ranked algorithm.
    """
    per_dataset = df.groupby(['dataset', 'algorithm'])['ari'].mean().reset_index()
    algorithms = sorted(per_dataset['algorithm'].unique())
    k = len(algorithms)
    datasets = sorted(per_dataset['dataset'].unique())
    n = len(datasets)
    rank_sum = {a: 0.0 for a in algorithms}
    for ds in datasets:
        sub = per_dataset[per_dataset['dataset'] == ds]
        r = main_benchmark_rankdata(-sub['ari'].to_numpy(), method='average')
        for a, rank in zip(sub['algorithm'], r):
            rank_sum[a] += rank
    avg_rank = {a: rank_sum[a] / n for a in algorithms}
    wide = per_dataset.pivot(index='dataset', columns='algorithm', values='ari')
    wide = wide[algorithms]
    cols = [wide[a].to_numpy(dtype=float) for a in algorithms]
    chi2, p_value = main_benchmark_friedmanchisquare(*cols)
    df1 = k - 1
    df2 = (k - 1) * (n - 1)
    if n * df1 - chi2 <= 0:
        F = float('inf')
    else:
        F = (n - 1) * chi2 / (n * df1 - chi2)
    q = main_benchmark_NEMENYI_Q.get(k, 3.53)
    cd = q * main_benchmark_np.sqrt(k * (k + 1) / (6.0 * n))
    best_rank = min(avg_rank.values())
    rows = []
    for a in sorted(algorithms, key=lambda x: avg_rank[x]):
        diff = avg_rank[a] - best_rank
        rows.append({'algorithm': a, 'average_rank': float(avg_rank[a]), 'diff_from_best': float(diff), 'significant_vs_best': bool(diff > cd)})
    ranks_df = main_benchmark_pd.DataFrame(rows)
    scalars = {'n_datasets': n, 'n_algorithms': k, 'friedman_chi2': float(chi2), 'iman_davenport_F': float(F), 'p_value': float(p_value), 'nemenyi_cd': float(cd)}
    return (scalars, ranks_df)

def main_benchmark_save_outputs(benchmark_df: main_benchmark_pd.DataFrame, ablation_df: main_benchmark_pd.DataFrame, out_dir: main_benchmark_Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(out_dir / 'paper_benchmark_raw.csv', index=False, encoding='utf-8-sig')
    ablation_df.to_csv(out_dir / 'paper_ablation_raw.csv', index=False, encoding='utf-8-sig')
    benchmark_summary = main_benchmark_flat_columns(main_benchmark_aggregate(benchmark_df))
    ablation_summary = main_benchmark_flat_columns(main_benchmark_aggregate(ablation_df))
    benchmark_summary.to_csv(out_dir / 'paper_benchmark_summary.csv', index=False, encoding='utf-8-sig')
    ablation_summary.to_csv(out_dir / 'paper_ablation_summary.csv', index=False, encoding='utf-8-sig')
    mean_ari = main_benchmark_mean_ari_table(benchmark_df)
    families = main_benchmark_family_table(benchmark_df)
    ranks = main_benchmark_rank_table(benchmark_df)
    tests = main_benchmark_wilcoxon_against_mbc(benchmark_df)
    friedman_scalars, friedman_ranks = main_benchmark_friedman_nemenyi(benchmark_df)
    mean_ari.to_csv(out_dir / 'paper_mean_ari_ranking.csv', encoding='utf-8-sig')
    families.to_csv(out_dir / 'paper_family_ari.csv', encoding='utf-8-sig')
    ranks.to_csv(out_dir / 'paper_average_ranks.csv', encoding='utf-8-sig')
    tests.to_csv(out_dir / 'paper_wilcoxon_vs_mbc.csv', index=False, encoding='utf-8-sig')
    friedman_ranks.to_csv(out_dir / 'paper_friedman_nemenyi.csv', index=False, encoding='utf-8-sig')
    (out_dir / 'paper_friedman_scalars.json').write_text(main_benchmark_json.dumps(friedman_scalars, indent=2), encoding='utf-8')
    main_benchmark_save_latex_tables(out_dir, mean_ari, families, ranks, tests, friedman_scalars, friedman_ranks)
    main_benchmark_save_figures(out_dir, benchmark_df, ablation_df)
    main_benchmark_save_markdown_report(out_dir, mean_ari, families, ranks, tests, benchmark_df, ablation_df, friedman_scalars, friedman_ranks)

def main_benchmark_save_latex_tables(out_dir: main_benchmark_Path, mean_ari: main_benchmark_pd.DataFrame, families: main_benchmark_pd.DataFrame, ranks: main_benchmark_pd.DataFrame, tests: main_benchmark_pd.DataFrame, friedman_scalars: dict, friedman_ranks: main_benchmark_pd.DataFrame) -> None:
    latex = {'mean_ari': mean_ari.round(4).to_latex(float_format='%.4f'), 'family_ari': families.round(4).to_latex(float_format='%.4f'), 'average_ranks': ranks.round(4).to_latex(float_format='%.4f'), 'wilcoxon': tests.round(4).to_latex(index=False, float_format='%.4f'), 'friedman_ranks': friedman_ranks.round(4).to_latex(index=False, float_format='%.4f')}
    (out_dir / 'paper_tables.tex').write_text('\n\n'.join((f'% {name}\n{table}' for name, table in latex.items())), encoding='utf-8')

def main_benchmark_save_figures(out_dir: main_benchmark_Path, benchmark_df: main_benchmark_pd.DataFrame, ablation_df: main_benchmark_pd.DataFrame) -> None:
    main_benchmark_plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11, 'legend.fontsize': 9, 'figure.dpi': 140})
    dataset_pivot = benchmark_df.groupby(['dataset', 'algorithm'])['ari'].mean().reset_index().pivot(index='dataset', columns='algorithm', values='ari')
    dataset_pivot = dataset_pivot.reindex(['two_moons', 'circles', 'noisy_moons', 'noisy_circles', 'anisotropic_blobs', 'varied_density', 'iris', 'wine', 'breast_cancer', 'digits_0_4', 'ecoli', 'glass', 'vehicle', 'cmc', 'wdbc'])
    ax = dataset_pivot.plot(kind='bar', figsize=(12, 5.8), width=0.82)
    ax.set_title('Adjusted Rand Index Across Benchmark Datasets')
    ax.set_ylabel('ARI')
    ax.set_xlabel('')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.tick_params(axis='x', rotation=35)
    main_benchmark_plt.tight_layout()
    main_benchmark_plt.savefig(out_dir / 'paper_fig_ari_by_dataset.png', dpi=180)
    main_benchmark_plt.close()
    family_pivot = main_benchmark_family_table(benchmark_df)
    family_pivot = family_pivot.reindex(['nonconvex', 'synthetic_gaussian', 'synthetic_density', 'real_low_dim', 'real_tabular', 'real_image_features'])
    ax = family_pivot.plot(kind='barh', figsize=(10.5, 5.6), width=0.8)
    ax.set_title('Mean ARI by Dataset Family')
    ax.set_xlabel('ARI')
    ax.set_ylabel('')
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis='x', alpha=0.25)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    main_benchmark_plt.tight_layout()
    main_benchmark_plt.savefig(out_dir / 'paper_fig_ari_by_family.png', dpi=180)
    main_benchmark_plt.close()
    ablation_pivot = ablation_df.groupby(['dataset', 'algorithm'])['ari'].mean().reset_index().pivot(index='dataset', columns='algorithm', values='ari')
    ablation_pivot = ablation_pivot.reindex(dataset_pivot.index)
    ax = ablation_pivot.plot(kind='bar', figsize=(11, 5.6), width=0.82)
    ax.set_title('Ablation Study of MBC Components')
    ax.set_ylabel('ARI')
    ax.set_xlabel('')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.tick_params(axis='x', rotation=35)
    main_benchmark_plt.tight_layout()
    main_benchmark_plt.savefig(out_dir / 'paper_fig_ablation.png', dpi=180)
    main_benchmark_plt.close()

def main_benchmark_save_markdown_report(out_dir: main_benchmark_Path, mean_ari: main_benchmark_pd.DataFrame, families: main_benchmark_pd.DataFrame, ranks: main_benchmark_pd.DataFrame, tests: main_benchmark_pd.DataFrame, benchmark_df: main_benchmark_pd.DataFrame, ablation_df: main_benchmark_pd.DataFrame, friedman_scalars: dict, friedman_ranks: main_benchmark_pd.DataFrame) -> None:
    mbc_family = families['MBC'].sort_values(ascending=False)
    ablation_mean = main_benchmark_mean_ari_table(ablation_df)
    present_datasets = sorted(benchmark_df['dataset'].unique())
    friedman_lines = [f"- Datasets (N={friedman_scalars['n_datasets']}), Algorithms (k={friedman_scalars['n_algorithms']})", f"- Friedman chi-squared = {friedman_scalars['friedman_chi2']:.4f}", f"- Iman-Davenport F = {friedman_scalars['iman_davenport_F']:.4f}, p-value = {friedman_scalars['p_value']:.4f}", f"- Nemenyi critical difference (CD) = {friedman_scalars['nemenyi_cd']:.4f} (average-rank units)", '- Algorithms whose average rank differs from the best by more than CD are flagged significant_vs_best above.']
    report = ['# Paper Experiment Report for Morphogenetic Buds Clustering', '', '## Experimental Design', '', f'- Seeds: {main_benchmark_SEEDS}', f"- Datasets ({len(present_datasets)}): {', '.join(present_datasets)}.", '- Baselines: KMeans, GMM, Agglomerative, Spectral, DBSCAN.', '- Metrics: ARI, NMI, silhouette, Davies-Bouldin, Calinski-Harabasz, runtime.', '- Ground-truth labels are used only for external evaluation, never during clustering.', '', '## Main Mean ARI Ranking', '', mean_ari.round(4).to_markdown(), '', '## Average Rank by ARI', '', ranks.round(4).to_markdown(), '', '## Mean ARI by Dataset Family', '', families.round(4).to_markdown(), '', '## MBC Strength Profile', '', mbc_family.round(4).to_markdown(), '', '## Wilcoxon Tests Against MBC', '', tests.round(4).to_markdown(index=False), '', '## Friedman Test and Nemenyi Critical Difference', '', '\n'.join(friedman_lines), '', friedman_ranks.round(4).to_markdown(index=False), '', '## Ablation Mean ARI', '', ablation_mean.round(4).to_markdown(), '', '## Interpretation', '', 'MBC is strongest on nonconvex continuous structures, where morphogenetic continuity is useful.', 'It is weaker on some high-dimensional tabular datasets because the current organ-fusion rule can over-segment compact classes.', 'This supports positioning MBC as a morphology-aware clustering method rather than a universal replacement for all clustering algorithms.']
    (out_dir / 'paper_experiment_report.md').write_text('\n'.join(report), encoding='utf-8')

def main_benchmark_main() -> None:
    out_dir = main_benchmark_Path(__file__).resolve().parents[1] / 'results' / 'paper_results'
    benchmark_df, ablation_df = main_benchmark_run_full_experiment()
    main_benchmark_save_outputs(benchmark_df, ablation_df, out_dir)
    summary = {'output_dir': str(out_dir), 'benchmark_rows': int(len(benchmark_df)), 'ablation_rows': int(len(ablation_df)), 'mean_ari': main_benchmark_mean_ari_table(benchmark_df)['mean'].round(4).to_dict()}
    print(main_benchmark_json.dumps(summary, indent=2))

main_benchmark = SimpleNamespace(make_datasets=main_benchmark_make_datasets, preprocess=main_benchmark_preprocess)


# ============================================================================
# External real-data benchmark
# ============================================================================

"""
External real-data benchmark for Morphogenetic Buds Clustering.

This script adds OpenML datasets, including the biologically motivated
MiceProtein dataset, to strengthen the paper evidence.
"""
import json as external_json
import time as external_time
import warnings as external_warnings
from pathlib import Path as external_Path
import matplotlib as external_matplotlib
external_matplotlib.use('Agg')
import matplotlib.pyplot as external_plt
import numpy as external_np
import pandas as external_pd
from mbc import MorphogeneticBudsClustering as external_MorphogeneticBudsClustering, kmeanspp_centers as external_kmeanspp_centers, refine_centers as external_refine_centers
from sklearn.cluster import AgglomerativeClustering as external_AgglomerativeClustering, DBSCAN as external_DBSCAN, KMeans as external_KMeans, SpectralClustering as external_SpectralClustering
from sklearn.datasets import fetch_openml as external_fetch_openml
from sklearn.decomposition import PCA as external_PCA
from sklearn.metrics import adjusted_rand_score as external_adjusted_rand_score, normalized_mutual_info_score as external_normalized_mutual_info_score, silhouette_score as external_silhouette_score
from sklearn.mixture import GaussianMixture as external_GaussianMixture
from sklearn.neighbors import NearestNeighbors as external_NearestNeighbors
from sklearn.preprocessing import LabelEncoder as external_LabelEncoder, StandardScaler as external_StandardScaler
external_PACKAGE_ROOT = external_Path(__file__).resolve().parents[1]
external_LOCAL_EXTERNAL_DATA = external_PACKAGE_ROOT / 'data' / 'processed' / 'external_real_benchmark'
external_SEEDS = [3, 7, 11]
external_ALGORITHMS = ['MBC', 'KMeans', 'GMM', 'Agglomerative', 'Spectral', 'DBSCAN', 'CompactRefineOnly']
external_OPENML_DATASETS = [{'name': 'MiceProtein_8class', 'id': 40966, 'family': 'biological_protein_expression', 'target_mode': 'class'}, {'name': 'MiceProtein_genotype', 'id': 40966, 'family': 'biological_factor', 'target_mode': 'prefix'}, {'name': 'MiceProtein_behavior', 'id': 40966, 'family': 'biological_factor', 'target_mode': 'middle'}, {'name': 'MiceProtein_treatment', 'id': 40966, 'family': 'biological_factor', 'target_mode': 'suffix'}, {'name': 'Leukemia_2class', 'id': 45090, 'family': 'biological_gene_expression'}, {'name': 'Leukemia_3class', 'id': 45091, 'family': 'biological_gene_expression'}, {'name': 'Lymphoma_3class', 'id': 45094, 'family': 'biological_gene_expression'}, {'name': 'credit-g', 'id': 31, 'family': 'real_mixed_tabular'}, {'name': 'diabetes', 'id': 37, 'family': 'real_medical_tabular'}, {'name': 'vehicle', 'id': 54, 'family': 'real_shape_features'}, {'name': 'ionosphere', 'id': 59, 'family': 'real_signal_features'}, {'name': 'blood-transfusion', 'id': 1464, 'family': 'real_medical_tabular'}]

def external_load_openml_dataset(spec: dict) -> tuple[str, str, external_np.ndarray, external_np.ndarray]:
    local_path = external_LOCAL_EXTERNAL_DATA / f"{spec['name']}_processed.csv"
    if local_path.exists():
        frame = external_pd.read_csv(local_path)
        return (spec['name'], spec['family'], frame.iloc[:, 1:].to_numpy(dtype=float), frame.iloc[:, 0].to_numpy(dtype=int))
    raw_candidates = sorted((external_PACKAGE_ROOT / 'data' / 'raw' / 'openml').glob(f"openml_{spec['id']}_*.csv"))
    if raw_candidates:
        raw_frame = external_pd.read_csv(raw_candidates[0])
        target_column = next((column for column in ('type', 'class', 'target') if column in raw_frame.columns), raw_frame.columns[-1])
        target = raw_frame.pop(target_column).astype(str)
        if spec.get('target_mode') == 'prefix':
            target = target.str.split('-').str[0]
        elif spec.get('target_mode') == 'middle':
            target = target.str.split('-').str[1]
        elif spec.get('target_mode') == 'suffix':
            target = target.str.split('-').str[2]
        x = external_prepare_features(raw_frame)
        y = external_LabelEncoder().fit_transform(target)
        return (spec['name'], spec['family'], x, y)
    bunch = external_fetch_openml(data_id=spec['id'], as_frame=True, parser='auto')
    x = external_prepare_features(bunch.data)
    target = bunch.target.astype(str)
    if spec.get('target_mode') == 'prefix':
        target = target.str.split('-').str[0]
    elif spec.get('target_mode') == 'middle':
        target = target.str.split('-').str[1]
    elif spec.get('target_mode') == 'suffix':
        target = target.str.split('-').str[2]
    y = external_LabelEncoder().fit_transform(target)
    return (spec['name'], spec['family'], x, y)

def external_prepare_features(frame: external_pd.DataFrame) -> external_np.ndarray:
    frame = frame.copy()
    numeric_cols = frame.select_dtypes(include=[external_np.number]).columns.tolist()
    categorical_cols = [col for col in frame.columns if col not in numeric_cols]
    parts = []
    if numeric_cols:
        numeric = frame[numeric_cols].apply(external_pd.to_numeric, errors='coerce')
        numeric = numeric.fillna(numeric.median(numeric_only=True))
        parts.append(numeric)
    if categorical_cols:
        categorical = frame[categorical_cols].astype('object')
        categorical = categorical.fillna(categorical.mode(dropna=True).iloc[0])
        parts.append(external_pd.get_dummies(categorical, drop_first=False, dtype=float))
    x = external_pd.concat(parts, axis=1).to_numpy(dtype=float)
    x = external_StandardScaler().fit_transform(x)
    if x.shape[1] > 12:
        n_components = min(12, x.shape[1], x.shape[0] - 1)
        x = external_PCA(n_components=n_components, random_state=0).fit_transform(x)
    return x

def external_run_algorithm(name: str, x: external_np.ndarray, n_clusters: int, seed: int) -> external_np.ndarray:
    if name == 'MBC':
        model = external_MorphogeneticBudsClustering(initial_buds=max(12, n_clusters * 6), steps=140, growth_rate=0.15, inhibition=0.045, apoptosis_mass=max(2.5, len(x) * 0.0055), birth_quantile=0.88, organ_merge_radius=3.6, valley_ratio=0.18, tangent_alignment=0.35, max_bridge_void=0.46, target_clusters=n_clusters, compact_refinement_steps=18, compact_refinement_restarts=10, compact_refinement_dim_threshold=3, random_state=seed)
        return model.fit_predict(x)
    if name == 'KMeans':
        return external_KMeans(n_clusters=n_clusters, n_init=30, random_state=seed).fit_predict(x)
    if name == 'GMM':
        return external_GaussianMixture(n_components=n_clusters, random_state=seed).fit_predict(x)
    if name == 'Agglomerative':
        return external_AgglomerativeClustering(n_clusters=n_clusters).fit_predict(x)
    if name == 'Spectral':
        return external_SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=min(12, len(x) - 1), random_state=seed).fit_predict(x)
    if name == 'DBSCAN':
        k = min(6, len(x))
        distances = external_NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)[0][:, -1]
        eps = float(external_np.quantile(distances, 0.72))
        return external_DBSCAN(eps=eps, min_samples=max(4, x.shape[1] + 1)).fit_predict(x)
    if name == 'CompactRefineOnly':
        rng = external_np.random.default_rng(seed)
        best_labels = None
        best_inertia = external_np.inf
        for _ in range(10):
            centers = external_kmeanspp_centers(x, n_clusters, rng)
            labels, _, inertia = external_refine_centers(x, centers, 18)
            if inertia < best_inertia:
                best_labels = labels
                best_inertia = inertia
        return best_labels
    raise ValueError(name)

def external_valid_silhouette(x: external_np.ndarray, labels: external_np.ndarray) -> float:
    if -1 in labels:
        mask = labels != -1
        x = x[mask]
        labels = labels[mask]
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return float('nan')
    return float(external_silhouette_score(x, labels))

def external_benchmark() -> external_pd.DataFrame:
    rows = []
    for spec in external_OPENML_DATASETS:
        try:
            dataset_name, family, x, y = external_load_openml_dataset(spec)
        except Exception as exc:
            print(f"[warn] skipped OpenML dataset {spec['name']}: {exc}")
            continue
        n_clusters = len(external_np.unique(y))
        for seed in external_SEEDS:
            for algorithm in external_ALGORITHMS:
                start = external_time.perf_counter()
                with external_warnings.catch_warnings():
                    external_warnings.simplefilter('ignore')
                    labels = external_run_algorithm(algorithm, x, n_clusters, seed)
                rows.append({'dataset': dataset_name, 'family': family, 'algorithm': algorithm, 'seed': seed, 'n_samples': len(x), 'n_features_after_preprocess': x.shape[1], 'true_clusters': n_clusters, 'clusters_found': len(set(labels)) - (1 if -1 in labels else 0), 'ari': external_adjusted_rand_score(y, labels), 'nmi': external_normalized_mutual_info_score(y, labels), 'silhouette': external_valid_silhouette(x, labels), 'seconds': external_time.perf_counter() - start})
    return external_pd.DataFrame(rows)

def external_save_outputs(df: external_pd.DataFrame, out_dir: external_Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / 'external_real_benchmark_raw.csv'
    summary_path = out_dir / 'external_real_benchmark_summary.csv'
    report_path = out_dir / 'external_real_benchmark_report.md'
    fig_path = out_dir / 'external_real_benchmark_ari.png'
    df.to_csv(raw_path, index=False, encoding='utf-8-sig')
    summary = df.groupby(['dataset', 'family', 'algorithm'])['ari'].agg(['mean', 'std']).reset_index().sort_values(['dataset', 'mean'], ascending=[True, False])
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    ranking = df.groupby('algorithm')['ari'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    biological = df[df['family'] == 'biological_protein_expression'].groupby('algorithm')['ari'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    family = df.groupby(['family', 'algorithm'])['ari'].mean().reset_index().pivot(index='family', columns='algorithm', values='ari')
    gene_expression = df[df['family'] == 'biological_gene_expression'].groupby('algorithm')['ari'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    pivot = df.groupby(['dataset', 'algorithm'])['ari'].mean().reset_index().pivot(index='dataset', columns='algorithm', values='ari')
    ax = pivot.plot(kind='bar', figsize=(12, 5.8), width=0.82)
    ax.set_title('External Real-Data Benchmark')
    ax.set_ylabel('ARI')
    ax.set_xlabel('')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.tick_params(axis='x', rotation=35)
    external_plt.tight_layout()
    external_plt.savefig(fig_path, dpi=180)
    external_plt.close()
    lines = ['# External Real-Data Benchmark', '', '## Overall Mean ARI', '', ranking.round(4).to_markdown(), '', '## Biological Dataset: MiceProtein', '', biological.round(4).to_markdown(), '', '## Biological Gene Expression Datasets', '', gene_expression.round(4).to_markdown(), '', '## Mean ARI by Family', '', family.round(4).to_markdown(), '', '## ARI by Dataset', '', pivot.round(4).to_markdown(), '', '## Files', '', f'- Raw results: `{raw_path.name}`', f'- Summary: `{summary_path.name}`', f'- Figure: `{fig_path.name}`']
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(external_json.dumps({'raw': str(raw_path), 'summary': str(summary_path), 'report': str(report_path), 'figure': str(fig_path)}, indent=2))
    print(ranking.round(4).to_string())

def external_main() -> None:
    out_dir = external_Path(__file__).resolve().parents[1] / 'results' / 'paper_results'
    df = external_benchmark()
    external_save_outputs(df, out_dir)

external_api = SimpleNamespace(prepare_features=external_prepare_features, run_algorithm=external_run_algorithm, OPENML_DATASETS=external_OPENML_DATASETS, load_openml_dataset=external_load_openml_dataset)


# ============================================================================
# Parameter-sensitivity analysis
# ============================================================================

"""
Parameter sensitivity analysis for Morphogenetic Buds Clustering.

The analysis perturbs one parameter at a time around the paper default and
evaluates representative datasets.
"""
from pathlib import Path as sensitivity_Path
import matplotlib as sensitivity_matplotlib
sensitivity_matplotlib.use('Agg')
import matplotlib.pyplot as sensitivity_plt
import numpy as sensitivity_np
import pandas as sensitivity_pd
from mbc import MorphogeneticBudsClustering as sensitivity_MorphogeneticBudsClustering
from sklearn.datasets import load_digits as sensitivity_load_digits, load_wine as sensitivity_load_wine, make_circles as sensitivity_make_circles, make_moons as sensitivity_make_moons
from sklearn.decomposition import PCA as sensitivity_PCA
from sklearn.metrics import adjusted_rand_score as sensitivity_adjusted_rand_score
from sklearn.preprocessing import StandardScaler as sensitivity_StandardScaler
sensitivity_SEEDS = [3, 7, 11]

def sensitivity_datasets(seed: int) -> list[tuple[str, sensitivity_np.ndarray, sensitivity_np.ndarray]]:
    wine = sensitivity_load_wine()
    digits = sensitivity_load_digits()
    mask = digits.target < 5
    moons_x, moons_y = sensitivity_make_moons(n_samples=500, noise=0.07, random_state=seed)
    circles_x, circles_y = sensitivity_make_circles(n_samples=500, factor=0.42, noise=0.045, random_state=seed)
    return [('two_moons', moons_x, moons_y), ('circles', circles_x, circles_y), ('wine', wine.data, wine.target), ('digits_0_4', digits.data[mask], digits.target[mask])]

def sensitivity_preprocess(x: sensitivity_np.ndarray) -> sensitivity_np.ndarray:
    x = sensitivity_StandardScaler().fit_transform(x)
    if x.shape[1] > 10:
        x = sensitivity_PCA(n_components=min(10, x.shape[1], x.shape[0] - 1), random_state=0).fit_transform(x)
    return x

def sensitivity_default_params(n_clusters: int) -> dict:
    return {'initial_buds': max(12, n_clusters * 6), 'steps': 140, 'growth_rate': 0.15, 'inhibition': 0.045, 'birth_quantile': 0.88, 'organ_merge_radius': 3.6, 'valley_ratio': 0.18, 'tangent_alignment': 0.35, 'max_bridge_void': 0.46, 'target_clusters': n_clusters, 'compact_refinement_steps': 18, 'compact_refinement_restarts': 10, 'compact_refinement_dim_threshold': 3}

def sensitivity_run_sensitivity() -> sensitivity_pd.DataFrame:
    grid = {'organ_merge_radius': [2.8, 3.2, 3.6, 4.0, 4.4], 'tangent_alignment': [0.0, 0.2, 0.35, 0.5, 0.65], 'compact_refinement_steps': [0, 5, 10, 18, 30], 'compact_refinement_restarts': [1, 3, 5, 10, 15]}
    rows = []
    for seed in sensitivity_SEEDS:
        for dataset_name, raw_x, y in sensitivity_datasets(seed):
            x = sensitivity_preprocess(raw_x)
            n_clusters = len(sensitivity_np.unique(y))
            for param, values in grid.items():
                for value in values:
                    params = sensitivity_default_params(n_clusters)
                    params[param] = value
                    params['apoptosis_mass'] = max(2.5, len(x) * 0.0055)
                    params['random_state'] = seed
                    labels = sensitivity_MorphogeneticBudsClustering(**params).fit_predict(x)
                    rows.append({'dataset': dataset_name, 'seed': seed, 'parameter': param, 'value': value, 'ari': sensitivity_adjusted_rand_score(y, labels)})
    return sensitivity_pd.DataFrame(rows)

def sensitivity_save_outputs(df: sensitivity_pd.DataFrame, out_dir: sensitivity_Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'mbc_sensitivity_raw.csv', index=False, encoding='utf-8-sig')
    summary = df.groupby(['parameter', 'value'])['ari'].agg(['mean', 'std']).reset_index().sort_values(['parameter', 'value'])
    summary.to_csv(out_dir / 'mbc_sensitivity_summary.csv', index=False, encoding='utf-8-sig')
    fig, axes = sensitivity_plt.subplots(2, 2, figsize=(11, 7), dpi=160)
    axes = axes.ravel()
    for ax, (param, group) in zip(axes, summary.groupby('parameter')):
        ax.errorbar(group['value'], group['mean'], yerr=group['std'], marker='o', capsize=3)
        ax.set_title(param)
        ax.set_xlabel('value')
        ax.set_ylabel('ARI')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'mbc_sensitivity.png')
    sensitivity_plt.close(fig)
    lines = ['# MBC Parameter Sensitivity', '', 'Mean ARI over two nonconvex datasets and two real feature datasets with three seeds.', '', summary.round(4).to_markdown(index=False)]
    (out_dir / 'mbc_sensitivity_report.md').write_text('\n'.join(lines), encoding='utf-8')

def sensitivity_main() -> None:
    out_dir = sensitivity_Path(__file__).resolve().parents[1] / 'results' / 'paper_results'
    df = sensitivity_run_sensitivity()
    sensitivity_save_outputs(df, out_dir)
    print(df.groupby('parameter')['ari'].mean().round(4).to_string())


# ============================================================================
# Runtime and biological visual analyses
# ============================================================================

"""
Runtime analysis and biological-data visualization for MBC.
"""
import time as runtime_bio_time
import warnings as runtime_bio_warnings
from pathlib import Path as runtime_bio_Path
import matplotlib as runtime_bio_matplotlib
runtime_bio_matplotlib.use('Agg')
import matplotlib.pyplot as runtime_bio_plt
import numpy as runtime_bio_np
import pandas as runtime_bio_pd
runtime_bio_prepare_features = external_api.prepare_features
runtime_bio_run_algorithm = external_api.run_algorithm
from mbc import MorphogeneticBudsClustering as runtime_bio_MorphogeneticBudsClustering
from sklearn.cluster import AgglomerativeClustering as runtime_bio_AgglomerativeClustering, DBSCAN as runtime_bio_DBSCAN, KMeans as runtime_bio_KMeans, SpectralClustering as runtime_bio_SpectralClustering
from sklearn.datasets import fetch_openml as runtime_bio_fetch_openml, make_blobs as runtime_bio_make_blobs, make_circles as runtime_bio_make_circles, make_moons as runtime_bio_make_moons
from sklearn.decomposition import PCA as runtime_bio_PCA
from sklearn.metrics import adjusted_rand_score as runtime_bio_adjusted_rand_score
from sklearn.mixture import GaussianMixture as runtime_bio_GaussianMixture
from sklearn.neighbors import NearestNeighbors as runtime_bio_NearestNeighbors
from sklearn.preprocessing import LabelEncoder as runtime_bio_LabelEncoder, StandardScaler as runtime_bio_StandardScaler
runtime_bio_SEEDS = [3, 7, 11]
runtime_bio_ALGORITHMS = ['MBC', 'KMeans', 'GMM', 'Agglomerative', 'Spectral', 'DBSCAN']

def runtime_bio_load_local_external(name: str) -> tuple[runtime_bio_np.ndarray, runtime_bio_np.ndarray] | None:
    aliases = {'Leukemia_3class': 'Leukemia_3class', 'Lymphoma_3class': 'Lymphoma_3class'}
    path = runtime_bio_Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'external_real_benchmark' / f'{aliases[name]}_processed.csv'
    if not path.exists():
        return None
    frame = runtime_bio_pd.read_csv(path)
    return (frame.iloc[:, 1:].to_numpy(float), frame.iloc[:, 0].to_numpy(int))

def runtime_bio_run_runtime_analysis(out_dir: runtime_bio_Path) -> runtime_bio_pd.DataFrame:
    rows = []
    sizes = [200, 500, 1000, 2000]
    for n_samples in sizes:
        for seed in runtime_bio_SEEDS:
            x, y = runtime_bio_make_moons(n_samples=n_samples, noise=0.07, random_state=seed)
            x = runtime_bio_StandardScaler().fit_transform(x)
            n_clusters = len(runtime_bio_np.unique(y))
            for algorithm in runtime_bio_ALGORITHMS:
                start = runtime_bio_time.perf_counter()
                with runtime_bio_warnings.catch_warnings():
                    runtime_bio_warnings.simplefilter('ignore')
                    labels = runtime_bio_run_runtime_algorithm(algorithm, x, n_clusters, seed)
                rows.append({'dataset': 'two_moons', 'n_samples': n_samples, 'algorithm': algorithm, 'seed': seed, 'seconds': runtime_bio_time.perf_counter() - start, 'ari': runtime_bio_adjusted_rand_score(y, labels)})
    df = runtime_bio_pd.DataFrame(rows)
    df.to_csv(out_dir / 'runtime_scaling_raw.csv', index=False, encoding='utf-8-sig')
    summary = df.groupby(['n_samples', 'algorithm'])[['seconds', 'ari']].agg(['mean', 'std']).reset_index()
    summary.to_csv(out_dir / 'runtime_scaling_summary.csv', index=False, encoding='utf-8-sig')
    pivot = df.groupby(['n_samples', 'algorithm'])['seconds'].mean().reset_index().pivot(index='n_samples', columns='algorithm', values='seconds')
    ax = pivot.plot(marker='o', figsize=(9, 5.4))
    ax.set_title('Runtime Scaling on Two Moons')
    ax.set_xlabel('number of samples')
    ax.set_ylabel('seconds')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    runtime_bio_plt.tight_layout()
    runtime_bio_plt.savefig(out_dir / 'runtime_scaling.png', dpi=180)
    runtime_bio_plt.close()
    return df

def runtime_bio_run_runtime_algorithm(name: str, x: runtime_bio_np.ndarray, n_clusters: int, seed: int) -> runtime_bio_np.ndarray:
    if name == 'MBC':
        return runtime_bio_MorphogeneticBudsClustering(initial_buds=max(12, n_clusters * 6), steps=140, growth_rate=0.15, inhibition=0.045, apoptosis_mass=max(2.5, len(x) * 0.0055), birth_quantile=0.88, organ_merge_radius=3.6, valley_ratio=0.18, tangent_alignment=0.35, max_bridge_void=0.46, target_clusters=n_clusters, compact_refinement_steps=18, compact_refinement_restarts=10, compact_refinement_dim_threshold=3, random_state=seed).fit_predict(x)
    if name == 'KMeans':
        return runtime_bio_KMeans(n_clusters=n_clusters, n_init=30, random_state=seed).fit_predict(x)
    if name == 'GMM':
        return runtime_bio_GaussianMixture(n_components=n_clusters, random_state=seed).fit_predict(x)
    if name == 'Agglomerative':
        return runtime_bio_AgglomerativeClustering(n_clusters=n_clusters).fit_predict(x)
    if name == 'Spectral':
        return runtime_bio_SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=min(12, len(x) - 1), random_state=seed).fit_predict(x)
    if name == 'DBSCAN':
        k = min(6, len(x))
        distances = runtime_bio_NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)[0][:, -1]
        eps = float(runtime_bio_np.quantile(distances, 0.72))
        return runtime_bio_DBSCAN(eps=eps, min_samples=max(4, x.shape[1] + 1)).fit_predict(x)
    raise ValueError(name)

def runtime_bio_make_bio_visuals(out_dir: runtime_bio_Path) -> None:
    datasets = [('Leukemia_3class', 45091), ('Lymphoma_3class', 45094)]
    rows = []
    for name, data_id in datasets:
        local = runtime_bio_load_local_external(name)
        if local is not None:
            x, y = local
        else:
            try:
                bunch = runtime_bio_fetch_openml(data_id=data_id, as_frame=True, parser='auto')
            except Exception as exc:
                print(f'[warn] skipped biological visualization {name}: {exc}')
                continue
            x = runtime_bio_prepare_features(bunch.data)
            y = runtime_bio_LabelEncoder().fit_transform(bunch.target.astype(str))
        labels = runtime_bio_run_algorithm('MBC', x, len(runtime_bio_np.unique(y)), seed=7)
        coords = runtime_bio_PCA(n_components=2, random_state=0).fit_transform(x)
        ari = runtime_bio_adjusted_rand_score(y, labels)
        rows.append({'dataset': name, 'ari': ari, 'n_samples': len(x), 'n_features_after_preprocess': x.shape[1]})
        fig, axes = runtime_bio_plt.subplots(1, 2, figsize=(10, 4.4), dpi=180)
        axes[0].scatter(coords[:, 0], coords[:, 1], c=y, cmap='tab10', s=34, alpha=0.86, linewidths=0)
        axes[0].set_title(f'{name}: biological labels')
        axes[1].scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=34, alpha=0.86, linewidths=0)
        axes[1].set_title(f'MBC clusters, ARI={ari:.3f}')
        for ax in axes:
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / f'bio_visual_{name}.png')
        runtime_bio_plt.close(fig)
    runtime_bio_pd.DataFrame(rows).to_csv(out_dir / 'bio_visual_summary.csv', index=False, encoding='utf-8-sig')

def runtime_bio_save_report(out_dir: runtime_bio_Path, runtime_df: runtime_bio_pd.DataFrame) -> None:
    runtime_summary = runtime_df.groupby(['n_samples', 'algorithm'])['seconds'].mean().reset_index().pivot(index='n_samples', columns='algorithm', values='seconds')
    ari_summary = runtime_df.groupby(['n_samples', 'algorithm'])['ari'].mean().reset_index().pivot(index='n_samples', columns='algorithm', values='ari')
    bio_summary = runtime_bio_pd.read_csv(out_dir / 'bio_visual_summary.csv')
    lines = ['# Runtime and Biological Visualization Report', '', '## Runtime Scaling', '', runtime_summary.round(4).to_markdown(), '', '## ARI During Runtime Scaling', '', ari_summary.round(4).to_markdown(), '', '## Biological PCA Visualizations', '', bio_summary.round(4).to_markdown(index=False), '', '## Figures', '', '- `runtime_scaling.png`', '- `bio_visual_Leukemia_3class.png`', '- `bio_visual_Lymphoma_3class.png`']
    (out_dir / 'runtime_and_bio_visuals_report.md').write_text('\n'.join(lines), encoding='utf-8')

def runtime_bio_main() -> None:
    out_dir = runtime_bio_Path(__file__).resolve().parents[1] / 'results' / 'paper_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_df = runtime_bio_run_runtime_analysis(out_dir)
    runtime_bio_make_bio_visuals(out_dir)
    runtime_bio_save_report(out_dir, runtime_df)
    print((out_dir / 'runtime_and_bio_visuals_report.md').read_text(encoding='utf-8'))


# ============================================================================
# YYC200 hyperspectral case study
# ============================================================================

"""
Hyperspectral yyc200 case study for MBC.

The script reads the ENVI BSQ cube stored next to the submission package,
builds robust PCA spectral features, optionally appends weak spatial
coordinates, and compares KMeans with MorphogeneticBudsClustering.
"""
import re as hyperspectral_re
import sys as hyperspectral_sys
import time as hyperspectral_time
from pathlib import Path as hyperspectral_Path
import matplotlib.pyplot as hyperspectral_plt
import numpy as hyperspectral_np
import pandas as hyperspectral_pd
from sklearn.cluster import KMeans as hyperspectral_KMeans
from sklearn.decomposition import PCA as hyperspectral_PCA
from sklearn.metrics import calinski_harabasz_score as hyperspectral_calinski_harabasz_score, davies_bouldin_score as hyperspectral_davies_bouldin_score, silhouette_score as hyperspectral_silhouette_score
from sklearn.preprocessing import StandardScaler as hyperspectral_StandardScaler
from mbc import MorphogeneticBudsClustering as hyperspectral_MorphogeneticBudsClustering
hyperspectral_PACKAGE_ROOT = hyperspectral_Path(__file__).resolve().parents[1]
hyperspectral_WORKSPACE_ROOT = hyperspectral_PACKAGE_ROOT.parent
hyperspectral_FIG_DIR = hyperspectral_PACKAGE_ROOT / 'figures' / 'paper_figures'
hyperspectral_RESULTS_DIR = hyperspectral_PACKAGE_ROOT / 'results' / 'paper_results'

def hyperspectral_find_yyc200_paths() -> tuple[hyperspectral_Path, hyperspectral_Path]:
    candidates = [hyperspectral_PACKAGE_ROOT / 'data' / 'yyc200_hyperspectral', hyperspectral_WORKSPACE_ROOT, hyperspectral_WORKSPACE_ROOT / 'data' / 'yyc200_hyperspectral']
    for base in candidates:
        data_path = base / 'yyc200'
        hdr_path = base / 'yyc200.hdr'
        if data_path.exists() and hdr_path.exists():
            return (data_path, hdr_path)
    raise FileNotFoundError('Could not locate yyc200 and yyc200.hdr')
hyperspectral_DATA_PATH, hyperspectral_HDR_PATH = hyperspectral_find_yyc200_paths()

def hyperspectral_parse_header(path: hyperspectral_Path) -> dict:
    text = path.read_text(encoding='utf-8', errors='ignore')

    def get_int(name: str) -> int:
        match = hyperspectral_re.search(f'{name}\\s*=\\s*(\\d+)', text, hyperspectral_re.IGNORECASE)
        if not match:
            raise ValueError(f'Missing {name} in {path}')
        return int(match.group(1))
    wav_match = hyperspectral_re.search('wavelength\\s*=\\s*\\{([^}]*)\\}', text, hyperspectral_re.IGNORECASE | hyperspectral_re.DOTALL)
    if not wav_match:
        raise ValueError('Missing wavelength list')
    wavelengths = hyperspectral_np.array([float(x) for x in hyperspectral_re.findall('[-+]?\\d*\\.\\d+|\\d+', wav_match.group(1))], dtype=float)
    return {'samples': get_int('samples'), 'lines': get_int('lines'), 'bands': get_int('bands'), 'wavelengths': wavelengths}

def hyperspectral_load_cube(meta: dict) -> hyperspectral_np.ndarray:
    expected = meta['bands'] * meta['lines'] * meta['samples']
    raw = hyperspectral_np.fromfile(hyperspectral_DATA_PATH, dtype='<i2')
    if raw.size != expected:
        raise ValueError(f'Expected {expected} values, found {raw.size}')
    return raw.reshape(meta['bands'], meta['lines'], meta['samples']).astype(hyperspectral_np.float32)

def hyperspectral_stretch(x: hyperspectral_np.ndarray, low: float=2, high: float=98) -> hyperspectral_np.ndarray:
    lo, hi = hyperspectral_np.percentile(x, [low, high])
    if hi <= lo:
        return hyperspectral_np.zeros_like(x, dtype=hyperspectral_np.float32)
    return hyperspectral_np.clip((x - lo) / (hi - lo), 0, 1)

def hyperspectral_nearest_band(wavelengths: hyperspectral_np.ndarray, target: float) -> int:
    return int(hyperspectral_np.argmin(hyperspectral_np.abs(wavelengths - target)))

def hyperspectral_band_statistics(cube: hyperspectral_np.ndarray, wavelengths: hyperspectral_np.ndarray) -> hyperspectral_pd.DataFrame:
    rows = []
    for i, wavelength in enumerate(wavelengths):
        band = cube[i]
        rows.append({'band': i + 1, 'wavelength_nm': wavelength, 'p01': float(hyperspectral_np.percentile(band, 1)), 'p99': float(hyperspectral_np.percentile(band, 99)), 'mean': float(band.mean()), 'std': float(band.std()), 'zero_pct': float((band == 0).mean() * 100), 'sat4095_pct': float((band == 4095).mean() * 100)})
    return hyperspectral_pd.DataFrame(rows)

def hyperspectral_build_features(cube: hyperspectral_np.ndarray, band_stats: hyperspectral_pd.DataFrame) -> tuple[hyperspectral_np.ndarray, hyperspectral_np.ndarray, list[int], hyperspectral_PCA]:
    used_bands = band_stats[(band_stats['zero_pct'] < 20) & (band_stats['std'] > 1) & (band_stats['p99'] > band_stats['p01'])]['band'].astype(int).to_list()
    used_idx = [b - 1 for b in used_bands]
    x = hyperspectral_np.moveaxis(cube[used_idx], 0, -1).reshape(-1, len(used_idx))
    lo = hyperspectral_np.percentile(x, 1, axis=0)
    hi = hyperspectral_np.percentile(x, 99, axis=0)
    x = hyperspectral_np.clip(x, lo, hi)
    x = hyperspectral_StandardScaler().fit_transform(x)
    pca = hyperspectral_PCA(n_components=8, random_state=0)
    scores = pca.fit_transform(x)
    spectral = hyperspectral_StandardScaler().fit_transform(scores)
    lines, samples = (cube.shape[1], cube.shape[2])
    yy, xx = hyperspectral_np.mgrid[0:lines, 0:samples]
    xy = hyperspectral_np.c_[yy.reshape(-1) / (lines - 1), xx.reshape(-1) / (samples - 1)]
    xy = (xy - 0.5) * 2
    spatial_spectral = hyperspectral_np.c_[spectral, 0.35 * xy]
    return (spectral, spatial_spectral, used_bands, pca)

def hyperspectral_remap_by_mean_signal(labels: hyperspectral_np.ndarray, signal: hyperspectral_np.ndarray) -> hyperspectral_np.ndarray:
    labels = hyperspectral_np.asarray(labels)
    order = sorted(hyperspectral_np.unique(labels), key=lambda g: float(signal[labels == g].mean()))
    mapping = {old: new for new, old in enumerate(order)}
    return hyperspectral_np.array([mapping[x] for x in labels], dtype=hyperspectral_np.int16)

def hyperspectral_boundary_ratio(label_image: hyperspectral_np.ndarray) -> float:
    horizontal = label_image[:, 1:] != label_image[:, :-1]
    vertical = label_image[1:, :] != label_image[:-1, :]
    return float((horizontal.sum() + vertical.sum()) / (horizontal.size + vertical.size))

def hyperspectral_sampled_silhouette(x: hyperspectral_np.ndarray, labels: hyperspectral_np.ndarray, seed: int=7) -> float:
    n = len(labels)
    sample_size = min(6000, n)
    rng = hyperspectral_np.random.default_rng(seed)
    ids = rng.choice(n, sample_size, replace=False)
    return float(hyperspectral_silhouette_score(x[ids], labels[ids]))

def hyperspectral_evaluate(name: str, feature_name: str, x: hyperspectral_np.ndarray, labels: hyperspectral_np.ndarray, elapsed: float, lines: int, samples: int) -> dict:
    label_image = labels.reshape(lines, samples)
    return {'algorithm': name, 'feature': feature_name, 'clusters': int(len(hyperspectral_np.unique(labels))), 'silhouette_sample': hyperspectral_sampled_silhouette(x, labels), 'davies_bouldin': float(hyperspectral_davies_bouldin_score(x, labels)), 'calinski_harabasz': float(hyperspectral_calinski_harabasz_score(x, labels)), 'boundary_ratio': hyperspectral_boundary_ratio(label_image), 'runtime_seconds': elapsed}

def hyperspectral_run_kmeans(x: hyperspectral_np.ndarray, k: int, seed: int) -> tuple[hyperspectral_np.ndarray, float]:
    start = hyperspectral_time.perf_counter()
    labels = hyperspectral_KMeans(n_clusters=k, n_init=40, random_state=seed).fit_predict(x)
    return (labels, hyperspectral_time.perf_counter() - start)

def hyperspectral_run_mbc(x: hyperspectral_np.ndarray, k: int, seed: int) -> tuple[hyperspectral_np.ndarray, float]:
    model = hyperspectral_MorphogeneticBudsClustering(initial_buds=max(12, k * 6), steps=140, growth_rate=0.15, inhibition=0.045, apoptosis_mass=max(2.5, len(x) * 0.0055), birth_quantile=0.88, organ_merge_radius=3.6, valley_ratio=0.18, tangent_alignment=0.35, max_bridge_void=0.46, target_clusters=k, compact_refinement_steps=18, compact_refinement_restarts=10, compact_refinement_dim_threshold=3, random_state=seed)
    start = hyperspectral_time.perf_counter()
    labels = model.fit_predict(x)
    return (labels, hyperspectral_time.perf_counter() - start)

def hyperspectral_cluster_summary(cube: hyperspectral_np.ndarray, labels: hyperspectral_np.ndarray, wavelengths: hyperspectral_np.ndarray) -> hyperspectral_pd.DataFrame:
    flat_cube = hyperspectral_np.moveaxis(cube, 0, -1).reshape(-1, cube.shape[0])
    rows = []
    for g in hyperspectral_np.unique(labels):
        mask = labels == g
        mean_spectrum = flat_cube[mask].mean(axis=0)
        peak = int(hyperspectral_np.argmax(mean_spectrum[:60]))
        rows.append({'cluster': f'C{int(g) + 1}', 'pixel_count': int(mask.sum()), 'area_pct': float(mask.mean() * 100), 'mean_intensity': float(mean_spectrum.mean()), 'peak_band': peak + 1, 'peak_wavelength_nm': float(wavelengths[peak]), 'peak_value': float(mean_spectrum[peak])})
    return hyperspectral_pd.DataFrame(rows)

def hyperspectral_plot_case_figure(cube: hyperspectral_np.ndarray, wavelengths: hyperspectral_np.ndarray, pca: hyperspectral_PCA, spectral: hyperspectral_np.ndarray, kmeans_labels: hyperspectral_np.ndarray, mbc_labels: hyperspectral_np.ndarray, out_path: hyperspectral_Path) -> None:
    lines, samples = (cube.shape[1], cube.shape[2])
    r = hyperspectral_nearest_band(wavelengths, 680)
    g = hyperspectral_nearest_band(wavelengths, 560)
    b = hyperspectral_nearest_band(wavelengths, 490)
    rgb = hyperspectral_np.dstack([hyperspectral_stretch(cube[r]), hyperspectral_stretch(cube[g]), hyperspectral_stretch(cube[b])])
    pca_img = spectral[:, :3].reshape(lines, samples, 3)
    pca_rgb = hyperspectral_np.dstack([hyperspectral_stretch(pca_img[:, :, i]) for i in range(3)])
    kmeans_img = kmeans_labels.reshape(lines, samples)
    mbc_img = mbc_labels.reshape(lines, samples)
    cmap = hyperspectral_plt.get_cmap('tab10', 6)
    fig, axes = hyperspectral_plt.subplots(2, 2, figsize=(8.2, 8.0), dpi=180)
    panels = [(rgb, 'Approximate RGB', None), (pca_rgb, f'PCA feature view ({pca.explained_variance_ratio_[:3].sum() * 100:.1f}%)', None), (kmeans_img, 'KMeans, spectral-spatial features', cmap), (mbc_img, 'MBC, spectral-spatial features', cmap)]
    for ax, (image, title, panel_cmap) in zip(axes.ravel(), panels):
        if panel_cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=panel_cmap, interpolation='nearest', vmin=-0.5, vmax=5.5)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    if out_path.suffix.lower() != '.pdf':
        fig.savefig(out_path.with_suffix('.pdf'))
    hyperspectral_plt.close(fig)

def hyperspectral_plot_mbc_spectra(cube: hyperspectral_np.ndarray, labels: hyperspectral_np.ndarray, wavelengths: hyperspectral_np.ndarray, out_path: hyperspectral_Path) -> None:
    flat_cube = hyperspectral_np.moveaxis(cube, 0, -1).reshape(-1, cube.shape[0])
    fig, ax = hyperspectral_plt.subplots(figsize=(8.2, 4.6), dpi=180)
    for g in hyperspectral_np.unique(labels):
        mask = labels == g
        mean_spectrum = flat_cube[mask].mean(axis=0)
        ax.plot(wavelengths[:60], mean_spectrum[:60], label=f'C{int(g) + 1} ({mask.mean() * 100:.1f}%)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Mean digital number')
    ax.set_title('MBC mean spectra by cluster')
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    if out_path.suffix.lower() != '.pdf':
        fig.savefig(out_path.with_suffix('.pdf'))
    hyperspectral_plt.close(fig)

def hyperspectral_main() -> None:
    if not hyperspectral_DATA_PATH.exists() or not hyperspectral_HDR_PATH.exists():
        raise FileNotFoundError('yyc200 and yyc200.hdr must be stored next to the submission package')
    hyperspectral_FIG_DIR.mkdir(parents=True, exist_ok=True)
    hyperspectral_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = hyperspectral_parse_header(hyperspectral_HDR_PATH)
    cube = hyperspectral_load_cube(meta)
    wavelengths = meta['wavelengths']
    band_stats = hyperspectral_band_statistics(cube, wavelengths)
    spectral, spatial_spectral, used_bands, pca = hyperspectral_build_features(cube, band_stats)
    lines, samples = (meta['lines'], meta['samples'])
    mean_signal = cube.mean(axis=0).reshape(-1)
    k = 6
    seed = 7
    rows = []
    labels_for_fig = {}
    for feature_name, x in [('spectral_pca', spectral), ('spectral_pca_xy', spatial_spectral)]:
        labels, elapsed = hyperspectral_run_kmeans(x, k, seed)
        labels = hyperspectral_remap_by_mean_signal(labels, mean_signal)
        rows.append(hyperspectral_evaluate('KMeans', feature_name, x, labels, elapsed, lines, samples))
        if feature_name == 'spectral_pca_xy':
            labels_for_fig['KMeans'] = labels
        if feature_name == 'spectral_pca_xy':
            labels, elapsed = hyperspectral_run_mbc(x, k, seed)
            labels = hyperspectral_remap_by_mean_signal(labels, mean_signal)
            rows.append(hyperspectral_evaluate('MBC', feature_name, x, labels, elapsed, lines, samples))
            labels_for_fig['MBC'] = labels
    metrics = hyperspectral_pd.DataFrame(rows)
    metrics.to_csv(hyperspectral_RESULTS_DIR / 'hyperspectral_yyc_mbc_metrics.csv', index=False, encoding='utf-8-sig')
    mbc_labels = labels_for_fig['MBC']
    hyperspectral_cluster_summary(cube, mbc_labels, wavelengths).to_csv(hyperspectral_RESULTS_DIR / 'hyperspectral_yyc_mbc_cluster_summary.csv', index=False, encoding='utf-8-sig')
    hyperspectral_np.save(hyperspectral_RESULTS_DIR / 'hyperspectral_yyc_mbc_labels.npy', mbc_labels.reshape(lines, samples).astype(hyperspectral_np.int16))
    hyperspectral_plot_case_figure(cube, wavelengths, pca, spectral, labels_for_fig['KMeans'], mbc_labels, hyperspectral_FIG_DIR / 'hyperspectral_yyc_mbc_case.png')
    hyperspectral_plot_mbc_spectra(cube, mbc_labels, wavelengths, hyperspectral_FIG_DIR / 'hyperspectral_yyc_mbc_spectra.png')
    summary = {'lines': lines, 'samples': samples, 'bands': meta['bands'], 'used_bands': used_bands, 'pca_first3_variance_pct': float(pca.explained_variance_ratio_[:3].sum() * 100), 'target_clusters': k, 'seed': seed}
    hyperspectral_pd.Series(summary).to_json(hyperspectral_RESULTS_DIR / 'hyperspectral_yyc_mbc_summary.json', force_ascii=False, indent=2)
    print(metrics.round(4).to_string(index=False))
    print(f"Wrote {hyperspectral_FIG_DIR / 'hyperspectral_yyc_mbc_case.png'}")
    print(f"Wrote {hyperspectral_FIG_DIR / 'hyperspectral_yyc_mbc_spectra.png'}")


# ============================================================================
# Cluster-structure publication figures
# ============================================================================

"""
Create publication-style clustering figures for the MBC manuscript.
"""
from pathlib import Path as cluster_figures_Path
import matplotlib as cluster_figures_matplotlib
cluster_figures_matplotlib.use('Agg')
import matplotlib.pyplot as cluster_figures_plt
import numpy as cluster_figures_np
import pandas as cluster_figures_pd
cluster_figures_prepare_features = external_api.prepare_features
from mbc import MorphogeneticBudsClustering as cluster_figures_MorphogeneticBudsClustering
from sklearn.cluster import KMeans as cluster_figures_KMeans, SpectralClustering as cluster_figures_SpectralClustering
from sklearn.datasets import fetch_openml as cluster_figures_fetch_openml, make_circles as cluster_figures_make_circles, make_moons as cluster_figures_make_moons
from sklearn.decomposition import PCA as cluster_figures_PCA
from sklearn.metrics import adjusted_rand_score as cluster_figures_adjusted_rand_score
from sklearn.preprocessing import LabelEncoder as cluster_figures_LabelEncoder, StandardScaler as cluster_figures_StandardScaler
cluster_figures_PACKAGE_ROOT = cluster_figures_Path(__file__).resolve().parents[1]
cluster_figures_OUT_DIR = cluster_figures_PACKAGE_ROOT / 'results' / 'paper_results'
cluster_figures_FIG_DIR = cluster_figures_PACKAGE_ROOT / 'figures' / 'paper_figures'

def cluster_figures_load_local_external(name: str) -> tuple[cluster_figures_np.ndarray, cluster_figures_np.ndarray] | None:
    aliases = {'Leukemia-3': 'Leukemia_3class', 'Lymphoma-3': 'Lymphoma_3class'}
    path = cluster_figures_PACKAGE_ROOT / 'data' / 'processed' / 'external_real_benchmark' / f'{aliases[name]}_processed.csv'
    if not path.exists():
        return None
    frame = cluster_figures_pd.read_csv(path)
    return (frame.iloc[:, 1:].to_numpy(float), frame.iloc[:, 0].to_numpy(int))

def cluster_figures_set_style() -> None:
    cluster_figures_plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 160})

def cluster_figures_mbc_model(k: int, seed: int) -> cluster_figures_MorphogeneticBudsClustering:
    return cluster_figures_MorphogeneticBudsClustering(initial_buds=max(12, k * 6), steps=150, growth_rate=0.15, inhibition=0.045, apoptosis_mass=3.0, birth_quantile=0.88, organ_merge_radius=3.6, valley_ratio=0.18, tangent_alignment=0.35, max_bridge_void=0.46, target_clusters=k, compact_refinement_steps=0, random_state=seed)

def cluster_figures_scatter(ax, x: cluster_figures_np.ndarray, labels: cluster_figures_np.ndarray, title: str) -> None:
    ax.scatter(x[:, 0], x[:, 1], c=labels, cmap='tab10', s=16, alpha=0.88, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.grid(alpha=0.18)

def cluster_figures_make_nonconvex_showcase() -> None:
    datasets = [('Two moons', *cluster_figures_make_moons(n_samples=700, noise=0.065, random_state=9)), ('Concentric circles', *cluster_figures_make_circles(n_samples=700, factor=0.42, noise=0.04, random_state=9))]
    fig, axes = cluster_figures_plt.subplots(2, 4, figsize=(13.8, 7.2), dpi=180)
    summary = []
    for row, (name, raw_x, y) in enumerate(datasets):
        x = cluster_figures_StandardScaler().fit_transform(raw_x)
        k = len(cluster_figures_np.unique(y))
        labels_mbc = cluster_figures_mbc_model(k, 9).fit_predict(x)
        labels_spec = cluster_figures_SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=12, random_state=9).fit_predict(x)
        labels_km = cluster_figures_KMeans(n_clusters=k, n_init=30, random_state=9).fit_predict(x)
        panels = [(y, f'{name}: truth'), (labels_mbc, f'MBC, ARI={cluster_figures_adjusted_rand_score(y, labels_mbc):.3f}'), (labels_spec, f'Spectral, ARI={cluster_figures_adjusted_rand_score(y, labels_spec):.3f}'), (labels_km, f'KMeans, ARI={cluster_figures_adjusted_rand_score(y, labels_km):.3f}')]
        for col, (labels, title) in enumerate(panels):
            cluster_figures_scatter(axes[row, col], x, labels, title)
        summary.append({'dataset': name, 'MBC': cluster_figures_adjusted_rand_score(y, labels_mbc), 'Spectral': cluster_figures_adjusted_rand_score(y, labels_spec), 'KMeans': cluster_figures_adjusted_rand_score(y, labels_km)})
    fig.suptitle('Nonconvex morphology: why the original MBC core is retained', y=1.01, fontsize=14)
    fig.tight_layout()
    cluster_figures_save_figure(fig, 'publication_nonconvex_showcase.png')
    cluster_figures_pd.DataFrame(summary).to_csv(cluster_figures_OUT_DIR / 'publication_nonconvex_showcase_scores.csv', index=False)

def cluster_figures_make_morphogenesis_diagram() -> None:
    x, y = cluster_figures_make_moons(n_samples=620, noise=0.065, random_state=13)
    x = cluster_figures_StandardScaler().fit_transform(x)
    model = cluster_figures_mbc_model(2, 13)
    labels = model.fit_predict(x)
    centers = model.centers_
    organs = model.organ_centers_
    fig, axes = cluster_figures_plt.subplots(1, 3, figsize=(13.2, 4.1), dpi=180)
    axes[0].scatter(x[:, 0], x[:, 1], c='#6b7280', s=16, alpha=0.72, linewidths=0)
    axes[0].set_title('Tissue cells emit morphogens')
    axes[1].scatter(x[:, 0], x[:, 1], c='#d1d5db', s=14, alpha=0.65, linewidths=0)
    axes[1].scatter(centers[:, 0], centers[:, 1], c='#111827', s=145, marker='*', edgecolors='white', linewidths=0.8, label='micro-buds')
    if organs is not None:
        axes[1].scatter(organs[:, 0], organs[:, 1], c='#ef4444', s=70, marker='o', edgecolors='white', linewidths=0.9, label='organs')
    axes[1].set_title('Buds grow, inhibit, and fuse')
    axes[1].legend(frameon=False)
    cluster_figures_scatter(axes[2], x, labels, f'Organ-level clusters, ARI={cluster_figures_adjusted_rand_score(y, labels):.3f}')
    for ax in axes:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.grid(alpha=0.18)
    fig.tight_layout()
    cluster_figures_save_figure(fig, 'publication_morphogenesis_diagram.png')

def cluster_figures_make_bio_pca_polished() -> None:
    datasets = [('Leukemia-3', 45091), ('Lymphoma-3', 45094)]
    fig, axes = cluster_figures_plt.subplots(2, 2, figsize=(10.8, 8.0), dpi=180)
    rows = []
    for row, (name, data_id) in enumerate(datasets):
        local = cluster_figures_load_local_external(name)
        if local is not None:
            x, y = local
        else:
            try:
                bunch = cluster_figures_fetch_openml(data_id=data_id, as_frame=True, parser='auto')
            except Exception as exc:
                print(f'[warn] skipped publication biological figure {name}: {exc}')
                for ax in axes[row]:
                    ax.text(0.5, 0.5, f'{name}\\nunavailable', ha='center', va='center')
                    ax.set_axis_off()
                continue
            x = cluster_figures_prepare_features(bunch.data)
            y = cluster_figures_LabelEncoder().fit_transform(bunch.target.astype(str))
        labels = cluster_figures_mbc_model(len(cluster_figures_np.unique(y)), 7).fit_predict(x)
        coords = cluster_figures_PCA(n_components=2, random_state=0).fit_transform(x)
        cluster_figures_scatter(axes[row, 0], coords, y, f'{name}: biological labels')
        cluster_figures_scatter(axes[row, 1], coords, labels, f'MBC clusters, ARI={cluster_figures_adjusted_rand_score(y, labels):.3f}')
        rows.append({'dataset': name, 'ari': cluster_figures_adjusted_rand_score(y, labels)})
    fig.suptitle('Biological expression data: PCA view', y=1.01, fontsize=14)
    fig.tight_layout()
    cluster_figures_save_figure(fig, 'publication_bio_expression_pca.png')
    cluster_figures_pd.DataFrame(rows).to_csv(cluster_figures_OUT_DIR / 'publication_bio_expression_pca_scores.csv', index=False)

def cluster_figures_save_figure(fig: cluster_figures_plt.Figure, name: str) -> None:
    cluster_figures_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster_figures_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cluster_figures_OUT_DIR / name
    fig_path = cluster_figures_FIG_DIR / name
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(fig_path, bbox_inches='tight')
    if fig_path.suffix.lower() != '.pdf':
        fig.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
    cluster_figures_plt.close(fig)

def cluster_figures_main() -> None:
    cluster_figures_set_style()
    cluster_figures_make_nonconvex_showcase()
    cluster_figures_make_morphogenesis_diagram()
    cluster_figures_make_bio_pca_polished()
    print('Saved publication figures to', cluster_figures_OUT_DIR)


# ============================================================================
# Additional publication figures
# ============================================================================

from pathlib import Path as advanced_figures_Path
import matplotlib.pyplot as advanced_figures_plt
import numpy as advanced_figures_np
import pandas as advanced_figures_pd
advanced_figures_ROOT = advanced_figures_Path(__file__).resolve().parents[1]
advanced_figures_RESULTS = advanced_figures_ROOT / 'results' / 'paper_results'
advanced_figures_FIGURES = advanced_figures_ROOT / 'figures' / 'paper_figures'
advanced_figures_ALGO_ORDER = ['MBC', 'Spectral', 'KMeans', 'GMM', 'Agglomerative', 'CompactRefineOnly', 'DBSCAN']

def advanced_figures__style():
    advanced_figures_plt.rcParams.update({'figure.dpi': 160, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans', 'axes.edgecolor': '#25313d', 'axes.linewidth': 0.8, 'axes.labelcolor': '#26323f', 'xtick.color': '#26323f', 'ytick.color': '#26323f', 'text.color': '#1f2a36'})

def advanced_figures__save_figure(fig, filename):
    advanced_figures_FIGURES.mkdir(parents=True, exist_ok=True)
    path = advanced_figures_FIGURES / filename
    fig.savefig(path, bbox_inches='tight')
    if path.suffix.lower() != '.pdf':
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')

def advanced_figures__ordered_columns(df):
    cols = [c for c in advanced_figures_ALGO_ORDER if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]

def advanced_figures__heatmap(data, title, outfile, cmap='viridis', vmin=0, vmax=1, highlight=('MBC',), cbar_label='ARI', figsize=None):
    data = advanced_figures__ordered_columns(data)
    rows, cols = data.shape
    if figsize is None:
        figsize = (max(7.5, cols * 1.08), max(3.9, rows * 0.58 + 1.7))
    fig, ax = advanced_figures_plt.subplots(figsize=figsize)
    values = data.to_numpy(dtype=float)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(advanced_figures_np.arange(cols))
    ax.set_yticks(advanced_figures_np.arange(rows))
    ax.set_xticklabels(data.columns, rotation=35, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(data.index)
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=14)
    for i in range(rows):
        row = values[i]
        finite = advanced_figures_np.isfinite(row)
        best = advanced_figures_np.nanmax(row[finite]) if finite.any() else advanced_figures_np.nan
        for j in range(cols):
            value = values[i, j]
            if not advanced_figures_np.isfinite(value):
                continue
            is_best = advanced_figures_np.isclose(value, best)
            text_color = 'white' if value > vmin + 0.62 * (vmax - vmin) else '#15202b'
            label = f'{value:.2f}' + ('*' if is_best else '')
            ax.text(j, i, label, ha='center', va='center', fontsize=8.2, color=text_color)
    for name in highlight:
        if name in data.columns:
            j = list(data.columns).index(name)
            ax.add_patch(advanced_figures_plt.Rectangle((j - 0.5, -0.5), 1, rows, fill=False, edgecolor='#ffb000', linewidth=2.2, clip_on=False))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.tick_params(length=0)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Dataset / family')
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.015)
    cbar.set_label(cbar_label)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(advanced_figures_np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(advanced_figures_np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.1)
    ax.tick_params(which='minor', bottom=False, left=False)
    fig.tight_layout()
    advanced_figures__save_figure(fig, outfile)
    advanced_figures_plt.close(fig)

def advanced_figures_family_heatmap():
    df = advanced_figures_pd.read_csv(advanced_figures_RESULTS / 'paper_family_ari.csv').set_index('family')
    order = ['nonconvex', 'synthetic_gaussian', 'real_tabular', 'real_image_features', 'real_low_dim', 'synthetic_density']
    df = df.loc[[x for x in order if x in df.index]]
    advanced_figures__heatmap(df, 'Performance landscape across data morphologies', 'paper_fig_family_heatmap.png', cmap='YlGnBu', highlight=('MBC',), figsize=(9.2, 5.3))

def advanced_figures_dataset_heatmap():
    df = advanced_figures_pd.read_csv(advanced_figures_RESULTS / 'paper_benchmark_summary.csv')
    pivot = df.pivot_table(index='dataset', columns='algorithm', values='ari_mean', aggfunc='mean')
    order = ['two_moons', 'circles', 'noisy_moons', 'noisy_circles', 'anisotropic_blobs', 'varied_density', 'iris', 'wine', 'breast_cancer', 'digits_0_4', 'ecoli', 'glass', 'vehicle', 'cmc', 'wdbc']
    pivot = pivot.loc[[x for x in order if x in pivot.index]]
    advanced_figures__heatmap(pivot, 'Dataset-level ARI matrix', 'paper_fig_dataset_heatmap.png', cmap='mako' if 'mako' in advanced_figures_plt.colormaps() else 'PuBuGn', highlight=('MBC',), figsize=(9.6, 5.5))

def advanced_figures_dataset_dot_profile():
    df = advanced_figures_pd.read_csv(advanced_figures_RESULTS / 'paper_benchmark_summary.csv')
    pivot = df.pivot_table(index='dataset', columns='algorithm', values='ari_mean', aggfunc='mean')
    order = ['two_moons', 'circles', 'noisy_moons', 'noisy_circles', 'anisotropic_blobs', 'varied_density', 'iris', 'wine', 'breast_cancer', 'digits_0_4', 'ecoli', 'glass', 'vehicle', 'cmc', 'wdbc']
    pivot = advanced_figures__ordered_columns(pivot.loc[[x for x in order if x in pivot.index]])
    fig, ax = advanced_figures_plt.subplots(figsize=(9.6, 6.0))
    y = advanced_figures_np.arange(len(pivot))
    baseline_cols = [c for c in pivot.columns if c != 'MBC']
    palette = {'MBC': '#d95f02', 'Spectral': '#4e79a7', 'KMeans': '#59a14f', 'GMM': '#b07aa1', 'Agglomerative': '#9c755f', 'CompactRefineOnly': '#7f7f7f', 'DBSCAN': '#bab0ab'}
    for i, dataset in enumerate(pivot.index):
        row = pivot.loc[dataset]
        ax.hlines(i, row.min(), row.max(), color='#c7d0d9', linewidth=1.8, zorder=1)
        ax.scatter(row[baseline_cols], advanced_figures_np.full(len(baseline_cols), i), s=42, color=[palette.get(c, '#999999') for c in baseline_cols], alpha=0.72, edgecolor='white', linewidth=0.5, zorder=2)
        ax.scatter(row['MBC'], i, marker='D', s=92, color=palette['MBC'], edgecolor='white', linewidth=0.9, zorder=4)
        best_alg = row.idxmax()
        ax.scatter(row[best_alg], i, marker='*', s=150, color='#f2c94c', edgecolor='#5c4a00', linewidth=0.5, zorder=5)
        ax.text(min(1.08, row['MBC'] + 0.025), i, f"{row['MBC']:.2f}", va='center', fontsize=8.2, color='#6b2d00')
    handles = [advanced_figures_plt.Line2D([0], [0], marker='D', color='none', markerfacecolor=palette['MBC'], markeredgecolor='white', markersize=8, label='MBC'), advanced_figures_plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='#f2c94c', markeredgecolor='#5c4a00', markersize=10, label='Best on dataset'), advanced_figures_plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#4e79a7', markeredgecolor='white', markersize=7, label='Baselines')]
    ax.legend(handles=handles, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlim(-0.06, 1.12)
    ax.set_xlabel('ARI')
    ax.set_title('Dataset-level performance profile', loc='left', fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', color='#d9e0e6', linewidth=0.8)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    advanced_figures__save_figure(fig, 'paper_fig_dataset_dotprofile.png')
    advanced_figures_plt.close(fig)

def advanced_figures_ablation_lollipop():
    df = advanced_figures_pd.read_csv(advanced_figures_RESULTS / 'paper_ablation_summary.csv')
    agg = df.groupby('algorithm', as_index=False).agg(mean=('ari_mean', 'mean'), std=('ari_mean', 'std')).sort_values('mean', ascending=True)
    labels = {'MBC': 'Full MBC', 'MBC_no_capillary': 'No capillary', 'MBC_no_polarity': 'No polarity', 'MBC_no_compact': 'No steady refinement', 'MBC_micro_only': 'Micro-buds only'}
    agg['label'] = agg['algorithm'].map(labels).fillna(agg['algorithm'])
    colors = ['#557c9e' if a != 'MBC' else '#d95f02' for a in agg['algorithm']]
    fig, ax = advanced_figures_plt.subplots(figsize=(8.6, 4.8))
    y = advanced_figures_np.arange(len(agg))
    ax.hlines(y, 0, agg['mean'], color='#b8c2cc', linewidth=4, alpha=0.7)
    ax.scatter(agg['mean'], y, s=145, c=colors, edgecolor='white', linewidth=1.4, zorder=3)
    for pos, (_, row) in enumerate(agg.iterrows()):
        ax.text(row['mean'] + 0.018, y[pos], f"{row['mean']:.3f}", va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(agg['label'])
    ax.set_xlim(0, max(0.82, agg['mean'].max() + 0.1))
    ax.set_xlabel('Mean ARI across benchmark datasets')
    ax.set_title('Ablation trajectory of morphogenetic mechanisms', loc='left', fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', color='#d9e0e6', linewidth=0.8)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    advanced_figures__save_figure(fig, 'paper_fig_ablation_lollipop.png')
    advanced_figures_plt.close(fig)

def advanced_figures_external_heatmap():
    df = advanced_figures_pd.read_csv(advanced_figures_RESULTS / 'external_real_benchmark_summary.csv')
    pivot = df.pivot_table(index='dataset', columns='algorithm', values='mean', aggfunc='mean')
    order = ['Leukemia_2class', 'Leukemia_3class', 'Lymphoma_3class', 'Lung_5class', 'MiceProtein', 'credit-g', 'diabetes', 'vehicle', 'ionosphere', 'blood']
    pivot = pivot.loc[[x for x in order if x in pivot.index]]
    advanced_figures__heatmap(pivot, 'External real-data ARI matrix', 'external_real_benchmark_heatmap.png', cmap='rocket_r' if 'rocket_r' in advanced_figures_plt.colormaps() else 'YlOrRd', vmin=-0.05, vmax=max(0.55, float(advanced_figures_np.nanmax(pivot.to_numpy()))), highlight=('MBC',), figsize=(9.8, 6.3))

def advanced_figures_external_rank_flow():
    df = advanced_figures_pd.read_csv(advanced_figures_RESULTS / 'external_real_benchmark_summary.csv')
    pivot = df.pivot_table(index='dataset', columns='algorithm', values='mean', aggfunc='mean')
    order = ['Leukemia_2class', 'Leukemia_3class', 'Lymphoma_3class', 'Lung_5class', 'MiceProtein_8class', 'credit-g', 'diabetes', 'vehicle', 'ionosphere', 'blood-transfusion']
    order = [x for x in order if x in pivot.index]
    pivot = advanced_figures__ordered_columns(pivot.loc[order])
    x = advanced_figures_np.arange(len(pivot))
    fig, ax = advanced_figures_plt.subplots(figsize=(10.6, 5.5))
    for col in pivot.columns:
        values = pivot[col].to_numpy(dtype=float)
        if col == 'MBC':
            ax.plot(x, values, color='#d95f02', linewidth=2.8, marker='D', markersize=6.5, label='MBC', zorder=4)
        elif col in ('GMM', 'Agglomerative', 'Spectral'):
            ax.plot(x, values, color='#6f7f8f', linewidth=1.4, marker='o', markersize=4.2, alpha=0.72, label=col, zorder=2)
        else:
            ax.plot(x, values, color='#b8c2cc', linewidth=1.0, marker='o', markersize=3.4, alpha=0.58, label=col, zorder=1)
    for i, dataset in enumerate(pivot.index):
        best_alg = pivot.loc[dataset].idxmax()
        best_val = pivot.loc[dataset, best_alg]
        ax.scatter(i, best_val, marker='*', s=145, color='#f2c94c', edgecolor='#5c4a00', linewidth=0.5, zorder=5)
    ax.axhline(0, color='#8b98a5', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=28, ha='right')
    ax.set_ylabel('ARI')
    ax.set_ylim(min(-0.08, advanced_figures_np.nanmin(pivot.to_numpy()) - 0.03), max(0.62, advanced_figures_np.nanmax(pivot.to_numpy()) + 0.05))
    ax.set_title('External real-data performance trajectories', loc='left', fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='y', color='#d9e0e6', linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    keep = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen and l in ['MBC', 'GMM', 'Agglomerative', 'Spectral', 'KMeans', 'DBSCAN', 'CompactRefineOnly']:
            keep.append((h, l))
            seen.add(l)
    ax.legend([h for h, _ in keep], [l for _, l in keep], frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.25))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    advanced_figures__save_figure(fig, 'external_real_benchmark_rankflow.png')
    advanced_figures_plt.close(fig)

def advanced_figures_sensitivity_small_multiples():
    df = advanced_figures_pd.read_csv(advanced_figures_RESULTS / 'mbc_sensitivity_summary.csv')
    params = list(df['parameter'].drop_duplicates())
    fig, axes = advanced_figures_plt.subplots(2, 2, figsize=(9.4, 6.0))
    axes = axes.ravel()
    for ax, param in zip(axes, params):
        sub = df[df['parameter'] == param].sort_values('value')
        ax.plot(sub['value'], sub['mean'], color='#2a6f97', linewidth=2.2)
        ax.fill_between(sub['value'].astype(float), (sub['mean'] - sub['std']).clip(lower=0), (sub['mean'] + sub['std']).clip(upper=1), color='#2a6f97', alpha=0.16, linewidth=0)
        ax.scatter(sub['value'], sub['mean'], color='#f28e2b', s=32, zorder=3)
        ax.set_title(param.replace('_', ' '), fontsize=10, loc='left')
        ax.set_ylim(0, 1.03)
        ax.grid(color='#d9e0e6', linewidth=0.7)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[len(params):]:
        ax.axis('off')
    fig.suptitle('Parameter response curves of MBC', x=0.03, ha='left', fontsize=13, fontweight='bold')
    fig.supxlabel('Parameter value')
    fig.supylabel('Mean ARI')
    fig.tight_layout()
    advanced_figures__save_figure(fig, 'mbc_sensitivity_curves.png')
    advanced_figures_plt.close(fig)

def advanced_figures_main():
    advanced_figures__style()
    advanced_figures_family_heatmap()
    advanced_figures_dataset_dot_profile()
    advanced_figures_ablation_lollipop()
    advanced_figures_external_rank_flow()
    advanced_figures_sensitivity_small_multiples()
    print('Advanced publication figures written to', advanced_figures_FIGURES)


# ============================================================================
# Hyperspectral analysis
# ============================================================================

import json as yyc_analysis_json
import re as yyc_analysis_re
from pathlib import Path as yyc_analysis_Path
import matplotlib.pyplot as yyc_analysis_plt
import numpy as yyc_analysis_np
import pandas as yyc_analysis_pd
from sklearn.cluster import KMeans as yyc_analysis_KMeans
from sklearn.decomposition import PCA as yyc_analysis_PCA
from sklearn.metrics import calinski_harabasz_score as yyc_analysis_calinski_harabasz_score, silhouette_score as yyc_analysis_silhouette_score
from sklearn.preprocessing import StandardScaler as yyc_analysis_StandardScaler
yyc_analysis_PACKAGE_ROOT = yyc_analysis_Path(__file__).resolve().parents[1]

def yyc_analysis_find_yyc200_paths(root: yyc_analysis_Path) -> tuple[yyc_analysis_Path, yyc_analysis_Path]:
    candidates = [root, root.parent / 'data' / 'yyc200_hyperspectral', root.parent]
    for base in candidates:
        data_path = base / 'yyc200'
        hdr_path = base / 'yyc200.hdr'
        if data_path.exists() and hdr_path.exists():
            return (data_path, hdr_path)
    raise FileNotFoundError('Could not find yyc200 and yyc200.hdr')
yyc_analysis_DATA_PATH, yyc_analysis_HDR_PATH = yyc_analysis_find_yyc200_paths(yyc_analysis_PACKAGE_ROOT / 'code')
yyc_analysis_OUT_DIR = yyc_analysis_PACKAGE_ROOT / 'results' / 'hyperspectral_exploration'
yyc_analysis_FIG_DIR = yyc_analysis_PACKAGE_ROOT / 'figures' / 'hyperspectral_exploration'
yyc_analysis_REPORT_PATH = yyc_analysis_OUT_DIR / 'yyc200_analysis_report.md'

def yyc_analysis_parse_envi_header(path: yyc_analysis_Path) -> dict:
    text = path.read_text(encoding='utf-8', errors='ignore')

    def get_int(name: str) -> int:
        match = yyc_analysis_re.search(f'{name}\\s*=\\s*(\\d+)', text, yyc_analysis_re.IGNORECASE)
        if not match:
            raise ValueError(f'Missing {name!r} in {path.name}')
        return int(match.group(1))
    wav_match = yyc_analysis_re.search('wavelength\\s*=\\s*\\{([^}]*)\\}', text, yyc_analysis_re.IGNORECASE | yyc_analysis_re.DOTALL)
    if not wav_match:
        raise ValueError('Missing wavelength list in header')
    wavelengths = yyc_analysis_np.array([float(x) for x in yyc_analysis_re.findall('[-+]?\\d*\\.\\d+|\\d+', wav_match.group(1))], dtype=float)
    interleave = yyc_analysis_re.search('interleave\\s*=\\s*(\\w+)', text, yyc_analysis_re.IGNORECASE)
    byte_order = yyc_analysis_re.search('byte order\\s*=\\s*(\\d+)', text, yyc_analysis_re.IGNORECASE)
    data_type = yyc_analysis_re.search('data type\\s*=\\s*(\\d+)', text, yyc_analysis_re.IGNORECASE)
    return {'samples': get_int('samples'), 'lines': get_int('lines'), 'bands': get_int('bands'), 'interleave': interleave.group(1).lower() if interleave else None, 'byte_order': int(byte_order.group(1)) if byte_order else None, 'data_type': int(data_type.group(1)) if data_type else None, 'wavelengths': wavelengths}

def yyc_analysis_envi_dtype(data_type: int, byte_order: int) -> yyc_analysis_np.dtype:
    if data_type != 2:
        raise ValueError(f'This script currently expects ENVI data type 2, got {data_type}')
    endian = '<' if byte_order == 0 else '>'
    return yyc_analysis_np.dtype(endian + 'i2')

def yyc_analysis_load_cube(meta: dict) -> yyc_analysis_np.ndarray:
    if meta['interleave'] != 'bsq':
        raise ValueError(f"This script currently expects BSQ interleave, got {meta['interleave']}")
    expected = meta['bands'] * meta['lines'] * meta['samples']
    raw = yyc_analysis_np.fromfile(yyc_analysis_DATA_PATH, dtype=yyc_analysis_envi_dtype(meta['data_type'], meta['byte_order']))
    if raw.size != expected:
        raise ValueError(f'Expected {expected} values, found {raw.size}')
    return raw.reshape((meta['bands'], meta['lines'], meta['samples'])).astype(yyc_analysis_np.float32)

def yyc_analysis_percentile_stretch(image: yyc_analysis_np.ndarray, low: float=2, high: float=98) -> yyc_analysis_np.ndarray:
    image = image.astype(yyc_analysis_np.float32)
    lo = yyc_analysis_np.nanpercentile(image, low)
    hi = yyc_analysis_np.nanpercentile(image, high)
    if not yyc_analysis_np.isfinite(lo) or not yyc_analysis_np.isfinite(hi) or hi <= lo:
        return yyc_analysis_np.zeros_like(image, dtype=yyc_analysis_np.float32)
    return yyc_analysis_np.clip((image - lo) / (hi - lo), 0, 1)

def yyc_analysis_nearest_band(wavelengths: yyc_analysis_np.ndarray, target: float) -> int:
    return int(yyc_analysis_np.argmin(yyc_analysis_np.abs(wavelengths - target)))

def yyc_analysis_save_rgb(cube: yyc_analysis_np.ndarray, wavelengths: yyc_analysis_np.ndarray, out_path: yyc_analysis_Path) -> tuple[int, int, int]:
    r = yyc_analysis_nearest_band(wavelengths, 680)
    g = yyc_analysis_nearest_band(wavelengths, 560)
    b = yyc_analysis_nearest_band(wavelengths, 490)
    rgb = yyc_analysis_np.dstack([yyc_analysis_percentile_stretch(cube[r]), yyc_analysis_percentile_stretch(cube[g]), yyc_analysis_percentile_stretch(cube[b])])
    yyc_analysis_plt.imsave(out_path, rgb)
    return (r, g, b)

def yyc_analysis_save_ndvi_like(cube: yyc_analysis_np.ndarray, wavelengths: yyc_analysis_np.ndarray, out_path: yyc_analysis_Path) -> dict:
    red = yyc_analysis_nearest_band(wavelengths, 680)
    nir = yyc_analysis_nearest_band(wavelengths, 805)
    red_band = cube[red].astype(yyc_analysis_np.float32)
    nir_band = cube[nir].astype(yyc_analysis_np.float32)
    denom = nir_band + red_band
    ndvi = yyc_analysis_np.divide(nir_band - red_band, denom, out=yyc_analysis_np.zeros_like(nir_band), where=denom != 0)
    ndvi = yyc_analysis_np.clip(ndvi, -1, 1)
    fig, ax = yyc_analysis_plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-0.6, vmax=0.6)
    ax.set_title('NDVI-like index')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('(NIR - red) / (NIR + red)')
    fig.tight_layout()
    fig.savefig(out_path)
    yyc_analysis_plt.close(fig)
    return {'red_band': red + 1, 'red_wavelength': float(wavelengths[red]), 'nir_band': nir + 1, 'nir_wavelength': float(wavelengths[nir]), 'min': float(yyc_analysis_np.min(ndvi)), 'mean': float(yyc_analysis_np.mean(ndvi)), 'max': float(yyc_analysis_np.max(ndvi))}

def yyc_analysis_make_band_stats(cube: yyc_analysis_np.ndarray, wavelengths: yyc_analysis_np.ndarray) -> yyc_analysis_pd.DataFrame:
    rows = []
    for i, wavelength in enumerate(wavelengths):
        band = cube[i]
        rows.append({'band': i + 1, 'wavelength_nm': wavelength, 'min': float(yyc_analysis_np.min(band)), 'p01': float(yyc_analysis_np.percentile(band, 1)), 'mean': float(yyc_analysis_np.mean(band)), 'median': float(yyc_analysis_np.median(band)), 'p99': float(yyc_analysis_np.percentile(band, 99)), 'max': float(yyc_analysis_np.max(band)), 'zero_pct': float(yyc_analysis_np.mean(band == 0) * 100), 'sat4095_pct': float(yyc_analysis_np.mean(band == 4095) * 100), 'std': float(yyc_analysis_np.std(band))})
    return yyc_analysis_pd.DataFrame(rows)

def yyc_analysis_prepare_features(cube: yyc_analysis_np.ndarray, band_stats: yyc_analysis_pd.DataFrame) -> tuple[yyc_analysis_np.ndarray, list[int], dict]:
    good = band_stats[(band_stats['zero_pct'] < 20) & (band_stats['std'] > 1) & (band_stats['p99'] > band_stats['p01'])]['band'].to_numpy()
    band_idx = [int(x) - 1 for x in good]
    x = yyc_analysis_np.moveaxis(cube[band_idx], 0, -1).reshape(-1, len(band_idx))
    lo = yyc_analysis_np.percentile(x, 1, axis=0)
    hi = yyc_analysis_np.percentile(x, 99, axis=0)
    x_clip = yyc_analysis_np.clip(x, lo, hi)
    x_scaled = yyc_analysis_StandardScaler().fit_transform(x_clip)
    prep = {'used_band_count': len(band_idx), 'used_bands': [i + 1 for i in band_idx], 'excluded_bands': [int(b) for b in band_stats.loc[~band_stats['band'].isin(good), 'band']], 'clip_percentiles': [1, 99]}
    return (x_scaled, band_idx, prep)

def yyc_analysis_run_pca(x_scaled: yyc_analysis_np.ndarray) -> tuple[yyc_analysis_PCA, yyc_analysis_np.ndarray]:
    pca = yyc_analysis_PCA(n_components=min(12, x_scaled.shape[1]), random_state=0)
    scores = pca.fit_transform(x_scaled)
    return (pca, scores)

def yyc_analysis_save_pca_outputs(scores: yyc_analysis_np.ndarray, pca: yyc_analysis_PCA, lines: int, samples: int, out_dir: yyc_analysis_Path) -> None:
    pca_img = scores[:, :3].reshape(lines, samples, 3)
    rgb = yyc_analysis_np.dstack([yyc_analysis_percentile_stretch(pca_img[:, :, i]) for i in range(3)])
    yyc_analysis_plt.imsave(out_dir / 'pca_rgb.png', rgb)
    fig, ax = yyc_analysis_plt.subplots(figsize=(7, 4), dpi=160)
    xs = yyc_analysis_np.arange(1, len(pca.explained_variance_ratio_) + 1)
    ax.bar(xs, pca.explained_variance_ratio_ * 100)
    ax.plot(xs, yyc_analysis_np.cumsum(pca.explained_variance_ratio_) * 100, marker='o', color='tab:red')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Explained variance (%)')
    ax.set_title('PCA explained variance')
    ax.set_xticks(xs)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'pca_variance.png')
    yyc_analysis_plt.close(fig)

def yyc_analysis_choose_k(scores: yyc_analysis_np.ndarray) -> tuple[int, yyc_analysis_pd.DataFrame]:
    rng = yyc_analysis_np.random.default_rng(7)
    cluster_x = scores[:, :8]
    sample_size = min(6000, cluster_x.shape[0])
    sample_idx = rng.choice(cluster_x.shape[0], sample_size, replace=False)
    rows = []
    best_k = None
    best_score = -yyc_analysis_np.inf
    for k in range(3, 11):
        km = yyc_analysis_KMeans(n_clusters=k, n_init=20, random_state=7)
        labels = km.fit_predict(cluster_x)
        sample_labels = labels[sample_idx]
        sil = yyc_analysis_silhouette_score(cluster_x[sample_idx], sample_labels)
        ch = yyc_analysis_calinski_harabasz_score(cluster_x, labels)
        rows.append({'k': k, 'silhouette_sample': sil, 'calinski_harabasz': ch, 'inertia': km.inertia_})
        adjusted = sil - (0.015 if k < 5 else 0)
        if adjusted > best_score:
            best_score = adjusted
            best_k = k
    return (int(best_k), yyc_analysis_pd.DataFrame(rows))

def yyc_analysis_run_clustering(scores: yyc_analysis_np.ndarray, k: int, lines: int, samples: int) -> tuple[yyc_analysis_np.ndarray, yyc_analysis_KMeans]:
    cluster_x = scores[:, :8]
    km = yyc_analysis_KMeans(n_clusters=k, n_init=40, random_state=11)
    labels = km.fit_predict(cluster_x)
    return (labels.reshape(lines, samples), km)

def yyc_analysis_save_cluster_outputs(labels_img: yyc_analysis_np.ndarray, cube: yyc_analysis_np.ndarray, wavelengths: yyc_analysis_np.ndarray, out_dir: yyc_analysis_Path) -> yyc_analysis_pd.DataFrame:
    k = int(labels_img.max()) + 1
    cmap = yyc_analysis_plt.get_cmap('tab10', k)
    fig, ax = yyc_analysis_plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(labels_img, cmap=cmap, interpolation='nearest', vmin=-0.5, vmax=k - 0.5)
    ax.set_title(f'KMeans spectral clusters (k={k})')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, ticks=yyc_analysis_np.arange(k), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([f'C{i + 1}' for i in range(k)])
    fig.tight_layout()
    fig.savefig(out_dir / 'kmeans_cluster_map.png')
    yyc_analysis_plt.close(fig)
    rows = []
    fig, ax = yyc_analysis_plt.subplots(figsize=(8, 4.6), dpi=160)
    flat_labels = labels_img.reshape(-1)
    flat_cube = yyc_analysis_np.moveaxis(cube, 0, -1).reshape(-1, cube.shape[0])
    for cluster in range(k):
        mask = flat_labels == cluster
        mean_spectrum = flat_cube[mask].mean(axis=0)
        area_pct = float(mask.mean() * 100)
        peak_band = int(yyc_analysis_np.argmax(mean_spectrum))
        rows.append({'cluster': f'C{cluster + 1}', 'pixel_count': int(mask.sum()), 'area_pct': area_pct, 'mean_intensity': float(mean_spectrum.mean()), 'peak_band': peak_band + 1, 'peak_wavelength_nm': float(wavelengths[peak_band]), 'peak_value': float(mean_spectrum[peak_band])})
        ax.plot(wavelengths[:60], mean_spectrum[:60], label=f'C{cluster + 1} ({area_pct:.1f}%)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Mean digital number')
    ax.set_title('Mean spectra by cluster')
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'cluster_spectra.png')
    yyc_analysis_plt.close(fig)
    return yyc_analysis_pd.DataFrame(rows)

def yyc_analysis_write_report(meta: dict, band_stats: yyc_analysis_pd.DataFrame, prep: dict, pca: yyc_analysis_PCA, k_metrics: yyc_analysis_pd.DataFrame, cluster_summary: yyc_analysis_pd.DataFrame, ndvi_summary: dict, rgb_bands: tuple[int, int, int]) -> None:
    bad_bands = band_stats[band_stats['zero_pct'] >= 20]['band'].astype(int).tolist()
    saturated = band_stats[band_stats['sat4095_pct'] >= 0.5][['band', 'sat4095_pct']]
    pca_top = pca.explained_variance_ratio_[:5] * 100
    best_row = k_metrics.sort_values('silhouette_sample', ascending=False).iloc[0]
    best_k = int(best_row['k'])
    chosen_k = int(cluster_summary.shape[0])
    chosen_row = k_metrics[k_metrics['k'] == chosen_k].iloc[0]

    def fmt_list(values: list[int]) -> str:
        return ', '.join((str(v) for v in values))
    report = f"# yyc200 高光谱数据建模分析\n\n## 数据概况\n\n- 文件格式：ENVI Standard，BSQ 排列，16-bit signed integer，little-endian。\n- 尺寸：{meta['lines']} 行 × {meta['samples']} 列 × {meta['bands']} 波段，共 {meta['lines'] * meta['samples']:,} 个像元。\n- 波长范围：{meta['wavelengths'][0]:.1f} nm 到 {meta['wavelengths'][-1]:.1f} nm。\n- 数值范围：{band_stats['min'].min():.0f} 到 {band_stats['max'].max():.0f}，整体为 12-bit 风格数字量级，最大值 4095 可视作饱和值。\n\n## 质量检查\n\n- 零值比例超过 20% 的波段：{fmt_list(bad_bands)}。\n- 本次 PCA/聚类使用 {prep['used_band_count']} 个信息较完整的波段：{fmt_list(prep['used_bands'])}。\n- 建模前对每个波段做了 1%-99% 分位数截断，再进行标准化，以降低饱和值和极端值影响。\n- 可见光近似 RGB 使用波段 R={rgb_bands[0] + 1} ({meta['wavelengths'][rgb_bands[0]]:.1f} nm), G={rgb_bands[1] + 1} ({meta['wavelengths'][rgb_bands[1]]:.1f} nm), B={rgb_bands[2] + 1} ({meta['wavelengths'][rgb_bands[2]]:.1f} nm)。\n"
    if not saturated.empty:
        saturated_text = ', '.join((f'B{int(row.band)}={row.sat4095_pct:.2f}%' for row in saturated.itertuples(index=False)))
        report += f'- 饱和像元比例超过 0.5% 的波段：{saturated_text}。\n'
    else:
        report += '- 各波段饱和值 4095 的比例都低于 0.5%。\n'
    report += f"\n## PCA 结果\n\n- 前 5 个主成分解释方差：{', '.join((f'PC{i + 1}={v:.2f}%' for i, v in enumerate(pca_top)))}。\n- 前 3 个主成分累计解释：{yyc_analysis_np.sum(pca.explained_variance_ratio_[:3]) * 100:.2f}%。\n- 前 5 个主成分累计解释：{yyc_analysis_np.sum(pca.explained_variance_ratio_[:5]) * 100:.2f}%。\n\n## 无监督聚类\n\n- 评估了 k=3 到 k=10 的 KMeans；样本轮廓系数最高的 k 为 {best_k} ({best_row['silhouette_sample']:.3f})。\n- 最终聚类图采用 k={chosen_k} ({chosen_row['silhouette_sample']:.3f})，原因是它与最高轮廓系数差距较小，同时能表达更细的城市地物光谱分区。\n- 这些类别是无监督光谱分区，不等同于已有地物标签。\n\n| 类别 | 像元数 | 面积占比 | 平均强度 | 峰值波段 | 峰值波长 |\n|---|---:|---:|---:|---:|---:|\n"
    for row in cluster_summary.itertuples(index=False):
        report += f'| {row.cluster} | {row.pixel_count:,} | {row.area_pct:.2f}% | {row.mean_intensity:.1f} | {row.peak_band} | {row.peak_wavelength_nm:.1f} nm |\n'
    report += f"\n## NDVI-like 指数\n\n- 使用 Red=B{ndvi_summary['red_band']} ({ndvi_summary['red_wavelength']:.1f} nm)，NIR=B{ndvi_summary['nir_band']} ({ndvi_summary['nir_wavelength']:.1f} nm)。\n- 指数范围：{ndvi_summary['min']:.3f} 到 {ndvi_summary['max']:.3f}，均值 {ndvi_summary['mean']:.3f}。\n- 由于原始数据是数字量级而非明确反射率校正结果，这里只能作为植被/高近红外响应的探索性指标。\n\n## 输出文件\n\n- `results/hyperspectral_exploration/band_stats.csv`：逐波段统计。\n- `results/hyperspectral_exploration/k_selection_metrics.csv`：聚类 k 值评估。\n- `results/hyperspectral_exploration/cluster_summary.csv`：聚类面积与典型强度。\n- `figures/hyperspectral_exploration/quicklook_rgb.png`：近似真彩色快视图。\n- `figures/hyperspectral_exploration/pca_rgb.png`：PC1-PC3 合成图。\n- `figures/hyperspectral_exploration/pca_variance.png`：PCA 解释方差。\n- `figures/hyperspectral_exploration/kmeans_cluster_map.png`：KMeans 聚类分区图。\n- `figures/hyperspectral_exploration/cluster_spectra.png`：各聚类平均光谱。\n- `figures/hyperspectral_exploration/ndvi_like.png`：NDVI-like 指数图。\n\n## 下一步建议\n\n如果有地面样本、分类标签或 ROI，可在当前脚本基础上改成监督分类，例如 Random Forest、SVM 或 XGBoost，并用混淆矩阵、OA、Kappa、F1 等指标评价。\n"
    yyc_analysis_REPORT_PATH.write_text(report, encoding='utf-8')

def yyc_analysis_main() -> None:
    yyc_analysis_OUT_DIR.mkdir(parents=True, exist_ok=True)
    yyc_analysis_FIG_DIR.mkdir(parents=True, exist_ok=True)
    meta = yyc_analysis_parse_envi_header(yyc_analysis_HDR_PATH)
    cube = yyc_analysis_load_cube(meta)
    wavelengths = meta['wavelengths']
    band_stats = yyc_analysis_make_band_stats(cube, wavelengths)
    band_stats.to_csv(yyc_analysis_OUT_DIR / 'band_stats.csv', index=False, encoding='utf-8-sig')
    rgb_bands = yyc_analysis_save_rgb(cube, wavelengths, yyc_analysis_FIG_DIR / 'quicklook_rgb.png')
    ndvi_summary = yyc_analysis_save_ndvi_like(cube, wavelengths, yyc_analysis_FIG_DIR / 'ndvi_like.png')
    x_scaled, band_idx, prep = yyc_analysis_prepare_features(cube, band_stats)
    pca, scores = yyc_analysis_run_pca(x_scaled)
    yyc_analysis_save_pca_outputs(scores, pca, meta['lines'], meta['samples'], yyc_analysis_FIG_DIR)
    chosen_k, k_metrics = yyc_analysis_choose_k(scores)
    k_metrics.to_csv(yyc_analysis_OUT_DIR / 'k_selection_metrics.csv', index=False, encoding='utf-8-sig')
    labels_img, _ = yyc_analysis_run_clustering(scores, chosen_k, meta['lines'], meta['samples'])
    yyc_analysis_np.save(yyc_analysis_OUT_DIR / 'kmeans_labels.npy', labels_img.astype(yyc_analysis_np.int16))
    cluster_summary = yyc_analysis_save_cluster_outputs(labels_img, cube, wavelengths, yyc_analysis_FIG_DIR)
    cluster_summary.to_csv(yyc_analysis_OUT_DIR / 'cluster_summary.csv', index=False, encoding='utf-8-sig')
    model_summary = {'metadata': {'samples': meta['samples'], 'lines': meta['lines'], 'bands': meta['bands'], 'interleave': meta['interleave'], 'data_type': meta['data_type'], 'byte_order': meta['byte_order']}, 'preprocessing': prep, 'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(), 'chosen_k': int(chosen_k), 'ndvi_like': ndvi_summary}
    (yyc_analysis_OUT_DIR / 'model_summary.json').write_text(yyc_analysis_json.dumps(model_summary, indent=2, ensure_ascii=False), encoding='utf-8')
    yyc_analysis_write_report(meta, band_stats, prep, pca, k_metrics, cluster_summary, ndvi_summary, rgb_bands)
    print(f'Wrote report: {yyc_analysis_REPORT_PATH}')
    print(f'Wrote outputs: {yyc_analysis_OUT_DIR}')
    print(f'Wrote figures: {yyc_analysis_FIG_DIR}')
    print(f'Chosen k: {chosen_k}')
    print('PCA first five explained variance (%):', yyc_analysis_np.round(pca.explained_variance_ratio_[:5] * 100, 2))


# ============================================================================
# Reviewer 1 analyses
# ============================================================================

import json as reviewer1_json
import sys as reviewer1_sys
from pathlib import Path as reviewer1_Path
import numpy as reviewer1_np
import pandas as reviewer1_pd
from sklearn.datasets import make_circles as reviewer1_make_circles, make_moons as reviewer1_make_moons
from sklearn.metrics import adjusted_rand_score as reviewer1_adjusted_rand_score, silhouette_score as reviewer1_silhouette_score
from sklearn.preprocessing import StandardScaler as reviewer1_StandardScaler
reviewer1_REPO = reviewer1_Path(__file__).resolve().parents[1]
reviewer1_sys.path.insert(0, str(reviewer1_REPO / 'code'))
from mbc import MorphogeneticBudsClustering as reviewer1_MorphogeneticBudsClustering
reviewer1_SEEDS = [3, 7, 11, 19, 23]
reviewer1_DATASETS = ['two_moons', 'circles', 'noisy_moons', 'noisy_circles', 'varied_density', 'anisotropic_blobs', 'iris', 'wine', 'breast_cancer', 'digits_0_4', 'vehicle']

def reviewer1_load_processed(name: str, seed: int) -> tuple[reviewer1_np.ndarray, reviewer1_np.ndarray]:
    path = reviewer1_REPO / 'data' / 'processed' / 'main_benchmark' / f'{name}_seed{seed}_processed.csv'
    frame = reviewer1_pd.read_csv(path)
    return (frame.iloc[:, 1:].to_numpy(float), frame.iloc[:, 0].to_numpy(int))

def reviewer1_mbc(x: reviewer1_np.ndarray, k: int, seed: int, **overrides) -> reviewer1_MorphogeneticBudsClustering:
    params = dict(initial_buds=max(12, k * 6), steps=140, growth_rate=0.15, inhibition=0.045, apoptosis_mass=max(2.5, len(x) * 0.0055), birth_quantile=0.88, organ_merge_radius=3.6, valley_ratio=0.18, tangent_alignment=0.35, max_bridge_void=0.46, target_clusters=k, compact_refinement_steps=18, compact_refinement_restarts=10, compact_refinement_dim_threshold=3, random_state=seed)
    params.update(overrides)
    return reviewer1_MorphogeneticBudsClustering(**params)

def reviewer1_safe_silhouette(x: reviewer1_np.ndarray, labels: reviewer1_np.ndarray) -> float:
    if -1 in labels:
        keep = labels != -1
        x, labels = (x[keep], labels[keep])
    if len(reviewer1_np.unique(labels)) < 2 or len(reviewer1_np.unique(labels)) >= len(labels):
        return float('nan')
    return float(reviewer1_silhouette_score(x, labels))

def reviewer1_run_k_analysis(out: reviewer1_Path) -> None:
    rows = []
    for name in ['two_moons', 'circles', 'noisy_moons', 'noisy_circles', 'varied_density', 'wine', 'breast_cancer']:
        for seed in [3, 7, 11]:
            x, y = reviewer1_load_processed(name, seed)
            true_k = len(reviewer1_np.unique(y))
            candidates = list(range(2, 7))
            for k in candidates:
                model = reviewer1_mbc(x, k, seed)
                labels = model.fit_predict(x)
                rows.append({'dataset': name, 'seed': seed, 'candidate_k': k, 'true_k': true_k, 'ari': reviewer1_adjusted_rand_score(y, labels), 'silhouette': reviewer1_safe_silhouette(x, labels)})
    raw = reviewer1_pd.DataFrame(rows)
    raw.to_csv(out / 'reviewer1_k_sensitivity_raw.csv', index=False)
    estimates = []
    for (name, seed), group in raw.groupby(['dataset', 'seed']):
        valid = group.dropna(subset=['silhouette'])
        chosen = valid.loc[valid['silhouette'].idxmax()]
        estimates.append({'dataset': name, 'seed': seed, 'true_k': int(group['true_k'].iloc[0]), 'estimated_k_by_silhouette': int(chosen['candidate_k']), 'estimated_k_ari': float(chosen['ari']), 'true_k_ari': float(group.loc[group['candidate_k'] == group['true_k'], 'ari'].iloc[0])})
    reviewer1_pd.DataFrame(estimates).to_csv(out / 'reviewer1_k_estimation_summary.csv', index=False)

def reviewer1_run_broad_sensitivity(out: reviewer1_Path) -> None:
    grid = {'organ_merge_radius': [2.8, 3.6, 4.4], 'tangent_alignment': [0.0, 0.35, 0.65], 'compact_refinement_steps': [0, 18, 30], 'compact_refinement_restarts': [1, 10, 15]}
    rows = []
    for name in reviewer1_DATASETS:
        for seed in reviewer1_SEEDS:
            x, y = reviewer1_load_processed(name, seed)
            k = len(reviewer1_np.unique(y))
            for parameter, values in grid.items():
                for value in values:
                    labels = reviewer1_mbc(x, k, seed, **{parameter: value}).fit_predict(x)
                    rows.append({'dataset': name, 'seed': seed, 'parameter': parameter, 'value': value, 'ari': reviewer1_adjusted_rand_score(y, labels)})
    raw = reviewer1_pd.DataFrame(rows)
    raw.to_csv(out / 'reviewer1_broad_sensitivity_raw.csv', index=False)
    summary = raw.groupby(['parameter', 'value'])['ari'].agg(['mean', 'std', 'min', 'max']).reset_index()
    summary.to_csv(out / 'reviewer1_broad_sensitivity_summary.csv', index=False)
    dataset_range = raw.groupby(['dataset', 'parameter'])['ari'].agg(lambda s: float(s.max() - s.min())).reset_index(name='range')
    dataset_range.to_csv(out / 'reviewer1_broad_sensitivity_ranges.csv', index=False)

def reviewer1_targeted_data(seed: int) -> list[tuple[str, reviewer1_np.ndarray, reviewer1_np.ndarray]]:
    rng = reviewer1_np.random.default_rng(seed)
    moons, y = reviewer1_make_moons(n_samples=420, noise=0.045, random_state=seed)
    bridge = reviewer1_np.c_[reviewer1_np.linspace(-0.45, 0.45, 26), reviewer1_np.linspace(0.48, 0.48, 26)]
    bridge_y = reviewer1_np.r_[reviewer1_np.zeros(13, dtype=int), reviewer1_np.ones(13, dtype=int)]
    bridge_x = reviewer1_np.vstack([moons, bridge])
    bridge_labels = reviewer1_np.r_[y, bridge_y]
    theta = reviewer1_np.linspace(0, 2 * reviewer1_np.pi, 280, endpoint=False)
    outer = reviewer1_np.c_[1.6 * reviewer1_np.cos(theta), 1.6 * reviewer1_np.sin(theta)] + rng.normal(0, 0.035, (280, 2))
    inner = reviewer1_np.c_[0.72 * reviewer1_np.cos(theta), 0.72 * reviewer1_np.sin(theta)] + rng.normal(0, 0.035, (280, 2))
    keep = ~((theta > 0.15) & (theta < 0.65))
    gap_x = reviewer1_np.vstack([outer[keep], inner])
    gap_y = reviewer1_np.r_[reviewer1_np.zeros(keep.sum(), dtype=int), reviewer1_np.ones(len(inner), dtype=int)]
    left_a = rng.normal([-1.5, 0.8], [0.12, 0.25], (110, 2))
    right_a = rng.normal([1.5, -0.8], [0.12, 0.25], (110, 2))
    left_b = rng.normal([-1.5, -0.8], [0.12, 0.25], (110, 2))
    right_b = rng.normal([1.5, 0.8], [0.12, 0.25], (110, 2))
    disconnected_x = reviewer1_np.vstack([left_a, right_a, left_b, right_b])
    disconnected_y = reviewer1_np.r_[reviewer1_np.zeros(220, dtype=int), reviewer1_np.ones(220, dtype=int)]
    return [('bridge', bridge_x, bridge_labels), ('gap', gap_x, gap_y), ('disconnected', disconnected_x, disconnected_y)]

def reviewer1_run_targeted_ablation(out: reviewer1_Path) -> None:
    rows = []
    for seed in [3, 7, 11, 19, 23]:
        for name, x, y in reviewer1_targeted_data(seed):
            x = reviewer1_StandardScaler().fit_transform(x)
            k = len(reviewer1_np.unique(y))
            variants = {'MBC_full': {}, 'no_polarity': {'tangent_alignment': 0.0}, 'no_channel_continuity': {'max_bridge_void': 1.0}, 'no_polarity_no_channel': {'tangent_alignment': 0.0, 'max_bridge_void': 1.0}, 'no_fusion_diagnostic': {'compact_refinement_steps': 0}}
            for variant, overrides in variants.items():
                model = reviewer1_mbc(x, k, seed, **overrides)
                labels = model.fit_predict(x)
                if variant == 'no_fusion_diagnostic':
                    labels = model.bud_labels_
                rows.append({'dataset': name, 'seed': seed, 'variant': variant, 'ari': reviewer1_adjusted_rand_score(y, labels), 'clusters': len(reviewer1_np.unique(labels))})
    raw = reviewer1_pd.DataFrame(rows)
    raw.to_csv(out / 'reviewer1_targeted_ablation_raw.csv', index=False)
    raw.groupby(['dataset', 'variant'])[['ari', 'clusters']].agg(['mean', 'std']).to_csv(out / 'reviewer1_targeted_ablation_summary.csv')

def reviewer1_main() -> None:
    out = reviewer1_REPO / 'results' / 'paper_results'
    out.mkdir(parents=True, exist_ok=True)
    reviewer1_run_k_analysis(out)
    reviewer1_run_broad_sensitivity(out)
    reviewer1_run_targeted_ablation(out)
    print(reviewer1_json.dumps({'output_dir': str(out), 'status': 'complete'}, indent=2))


# ============================================================================
# Reviewer 2 analyses
# ============================================================================

"""Additional analyses for the Reviewer 2 revision.

The script is deliberately self-contained and writes raw tables so that every
number quoted in the revision can be regenerated from the submission package.
All model-selection rules in this file are label-free; labels are read only for
the reported ARI.
"""
import itertools as reviewer2_itertools
import sys as reviewer2_sys
from pathlib import Path as reviewer2_Path
import numpy as reviewer2_np
import pandas as reviewer2_pd
from sklearn.cluster import AffinityPropagation as reviewer2_AffinityPropagation, AgglomerativeClustering as reviewer2_AgglomerativeClustering, HDBSCAN as reviewer2_HDBSCAN, KMeans as reviewer2_KMeans, MeanShift as reviewer2_MeanShift, SpectralClustering as reviewer2_SpectralClustering
from sklearn.datasets import make_blobs as reviewer2_make_blobs, make_circles as reviewer2_make_circles, make_moons as reviewer2_make_moons
from sklearn.metrics import adjusted_rand_score as reviewer2_adjusted_rand_score
from sklearn.mixture import GaussianMixture as reviewer2_GaussianMixture
from sklearn.preprocessing import StandardScaler as reviewer2_StandardScaler
reviewer2_REPO = reviewer2_Path(__file__).resolve().parents[1]
reviewer2_sys.path.insert(0, str(reviewer2_REPO / 'code'))
from mbc import MorphogeneticBudsClustering as reviewer2_MorphogeneticBudsClustering
reviewer2_SEEDS = [3, 7, 11]
reviewer2_PRIMARY_SEEDS = [3, 7, 11, 19, 23]

def reviewer2_mbc(x: reviewer2_np.ndarray, k: int, seed: int, **overrides) -> reviewer2_MorphogeneticBudsClustering:
    params = dict(initial_buds=max(12, k * 6), steps=140, growth_rate=0.15, inhibition=0.045, apoptosis_mass=max(2.5, len(x) * 0.0055), birth_quantile=0.88, organ_merge_radius=3.6, valley_ratio=0.18, tangent_alignment=0.35, max_bridge_void=0.46, target_clusters=k, compact_refinement_steps=18, compact_refinement_restarts=10, compact_refinement_dim_threshold=3, random_state=seed)
    params.update(overrides)
    return reviewer2_MorphogeneticBudsClustering(**params)

def reviewer2_spiral_data(seed: int, arms: int=3, n_per_arm: int=150) -> tuple[reviewer2_np.ndarray, reviewer2_np.ndarray]:
    rng = reviewer2_np.random.default_rng(seed)
    pieces = []
    labels = []
    for arm in range(arms):
        t = reviewer2_np.linspace(0.25, 2.25 * reviewer2_np.pi, n_per_arm)
        theta = t + 2.0 * reviewer2_np.pi * arm / arms
        radius = 0.2 + 0.2 * t
        pieces.append(reviewer2_np.c_[radius * reviewer2_np.cos(theta), radius * reviewer2_np.sin(theta)])
        labels.extend([arm] * n_per_arm)
    x = reviewer2_np.vstack(pieces) + rng.normal(scale=0.035, size=(arms * n_per_arm, 2))
    return (x, reviewer2_np.asarray(labels))

def reviewer2_s_curve_data(seed: int, n: int=480) -> tuple[reviewer2_np.ndarray, reviewer2_np.ndarray]:
    rng = reviewer2_np.random.default_rng(seed)
    t = rng.uniform(-1.0, 1.0, n)
    x = 1.4 * reviewer2_np.sin(reviewer2_np.pi * t) + rng.normal(0, 0.045, n)
    y = 1.2 * t + rng.normal(0, 0.045, n)
    y_label = reviewer2_np.digitize(t, [-0.34, 0.34])
    return (reviewer2_np.c_[x, y], y_label)

def reviewer2_bridge_data(seed: int, n: int=420) -> tuple[reviewer2_np.ndarray, reviewer2_np.ndarray]:
    rng = reviewer2_np.random.default_rng(seed)
    left = rng.normal([-1.8, 0.0], [0.18, 0.55], (n // 3, 2))
    right = rng.normal([1.8, 0.0], [0.18, 0.55], (n // 3, 2))
    bridge = reviewer2_np.c_[reviewer2_np.linspace(-1.5, 1.5, n - 2 * (n // 3)), rng.normal(0, 0.08, n - 2 * (n // 3))]
    return (reviewer2_np.vstack([left, right, bridge]), reviewer2_np.r_[reviewer2_np.zeros(len(left), int), reviewer2_np.ones(len(right) + len(bridge), int)])

def reviewer2_disconnected_data(seed: int, n: int=480) -> tuple[reviewer2_np.ndarray, reviewer2_np.ndarray]:
    rng = reviewer2_np.random.default_rng(seed)
    a = reviewer2_np.vstack([rng.normal([-1.5, -0.8], [0.16, 0.24], (n // 4, 2)), rng.normal([1.5, 0.8], [0.16, 0.24], (n // 4, 2))])
    b = reviewer2_np.vstack([rng.normal([-1.5, 0.8], [0.16, 0.24], (n // 4, 2)), rng.normal([1.5, -0.8], [0.16, 0.24], (n // 4, 2))])
    x = reviewer2_np.vstack([a, b])
    y = reviewer2_np.r_[reviewer2_np.zeros(n // 2, int), reviewer2_np.ones(n // 2, int)]
    return (x, y)

def reviewer2_make_nonconvex(seed: int) -> list[tuple[str, reviewer2_np.ndarray, reviewer2_np.ndarray]]:
    moons_x, moons_y = reviewer2_make_moons(480, noise=0.08, random_state=seed)
    circles_x, circles_y = reviewer2_make_circles(480, factor=0.42, noise=0.05, random_state=seed)
    return [('three_spirals', *reviewer2_spiral_data(seed)), ('s_curve_sections', *reviewer2_s_curve_data(seed)), ('bridged_blobs', *reviewer2_bridge_data(seed)), ('disconnected_manifolds', *reviewer2_disconnected_data(seed)), ('additional_moons', moons_x, moons_y), ('additional_circles', circles_x, circles_y)]

def reviewer2_run_related(name: str, x: reviewer2_np.ndarray, k: int, seed: int) -> reviewer2_np.ndarray:
    if name == 'MBC':
        return reviewer2_mbc(x, k, seed).fit_predict(x)
    if name == 'KMeans':
        return reviewer2_KMeans(k, n_init=30, random_state=seed).fit_predict(x)
    if name == 'GMM':
        return reviewer2_GaussianMixture(k, random_state=seed).fit_predict(x)
    if name == 'Agglomerative':
        return reviewer2_AgglomerativeClustering(n_clusters=k).fit_predict(x)
    if name == 'Spectral':
        return reviewer2_SpectralClustering(k, affinity='nearest_neighbors', n_neighbors=min(12, len(x) - 1), random_state=seed).fit_predict(x)
    if name == 'MeanShift':
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(8, len(x))).fit(x)
        bandwidth = float(reviewer2_np.median(nn.kneighbors(x)[0][:, -1])) * 2.0
        return reviewer2_MeanShift(bandwidth=max(bandwidth, 0.001), bin_seeding=True).fit_predict(x)
    if name == 'AffinityPropagation':
        return reviewer2_AffinityPropagation(random_state=seed, damping=0.75, max_iter=500).fit_predict(x)
    if name == 'HDBSCAN':
        return reviewer2_HDBSCAN(min_cluster_size=max(10, len(x) // 40), min_samples=5).fit_predict(x)
    raise ValueError(name)

def reviewer2_run_nonconvex_and_related(out: reviewer2_Path) -> None:
    methods = ['MBC', 'KMeans', 'GMM', 'Agglomerative', 'Spectral', 'MeanShift', 'AffinityPropagation', 'HDBSCAN']
    rows = []
    for seed in reviewer2_SEEDS:
        for dataset, raw_x, y in reviewer2_make_nonconvex(seed):
            x = reviewer2_StandardScaler().fit_transform(raw_x)
            k = len(reviewer2_np.unique(y))
            for method in methods:
                labels = reviewer2_run_related(method, x, k, seed)
                rows.append({'dataset': dataset, 'seed': seed, 'method': method, 'true_k': k, 'found_k': len(reviewer2_np.unique(labels[labels >= 0])) if reviewer2_np.any(labels >= 0) else 0, 'ari': reviewer2_adjusted_rand_score(y, labels)})
    reviewer2_pd.DataFrame(rows).to_csv(out / 'reviewer2_related_methods_raw.csv', index=False)
    summary = reviewer2_pd.DataFrame(rows).groupby('method')['ari'].agg(['mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)
    summary.to_csv(out / 'reviewer2_related_methods_summary.csv')

def reviewer2_run_channel_diagnostic(out: reviewer2_Path) -> None:
    from mbc import BudState, capillary_bridge_ok
    rows = []
    for seed in reviewer2_PRIMARY_SEEDS:
        rng = reviewer2_np.random.default_rng(seed)
        centers = reviewer2_np.array([[-1.0, 0.8], [1.0, 0.8], [-1.0, -0.8], [1.0, -0.8]])
        endpoints = reviewer2_np.vstack([rng.normal(centers[0], [0.08, 0.08], (100, 2)), rng.normal(centers[1], [0.08, 0.08], (100, 2)), rng.normal(centers[2], [0.08, 0.08], (100, 2)), rng.normal(centers[3], [0.08, 0.08], (100, 2))])
        populated = reviewer2_np.c_[reviewer2_np.linspace(-0.9, 0.9, 40), rng.normal(0.8, 0.04, 40)]
        for tissue, x in [('populated_channel', reviewer2_np.vstack([endpoints, populated])), ('empty_channel', endpoints)]:
            state = BudState(centers.copy(), reviewer2_np.ones(4), reviewer2_np.zeros((4, 2)))
            model = reviewer2_MorphogeneticBudsClustering(organ_merge_radius=3.6, valley_ratio=0.0, tangent_alignment=0.0, target_clusters=None, compact_refinement_steps=0, max_bridge_void=0.46, random_state=seed)
            accepted_with_channel = capillary_bridge_ok(x, centers[0], centers[1], 0.6, 0.46)
            accepted_without_channel = capillary_bridge_ok(x, centers[0], centers[1], 0.6, 1.0)
            organ_labels = model._form_organs(x, state, 0.6)
            rows.append({'dataset': tissue, 'seed': seed, 'variant': 'channel_predicate', 'accepted_edge': int(accepted_with_channel), 'accepted_without_channel': int(accepted_without_channel), 'organs': len(reviewer2_np.unique(organ_labels)), 'note': 'same candidate buds; only channel threshold changes'})
    reviewer2_pd.DataFrame(rows).to_csv(out / 'reviewer2_channel_continuity_raw.csv', index=False)
    reviewer2_pd.DataFrame(rows).groupby('dataset')[['accepted_edge', 'accepted_without_channel', 'organs']].agg(['mean', 'std']).to_csv(out / 'reviewer2_channel_continuity_summary.csv')

def reviewer2_run_initialization_winners(out: reviewer2_Path) -> None:
    data_dir = reviewer2_REPO / 'data' / 'processed' / 'main_benchmark'
    names = sorted({p.name.removesuffix('_seed3_processed.csv') for p in data_dir.glob('*_seed3_processed.csv')})
    rows = []
    for name in names:
        for seed in reviewer2_PRIMARY_SEEDS:
            frame = reviewer2_pd.read_csv(data_dir / f'{name}_seed{seed}_processed.csv')
            x, y = (frame.iloc[:, 1:].to_numpy(float), frame.iloc[:, 0].to_numpy(int))
            model = reviewer2_mbc(x, len(reviewer2_np.unique(y)), seed)
            model.fit(x)
            if model.refinement_winner_ is None:
                continue
            rows.append({'dataset': name, 'seed': seed, 'winner': model.refinement_winner_, 'winning_inertia': min(model.refinement_inertias_), 'mbc_inertia': model.refinement_inertias_[0], 'n_kmeanspp_candidates': len(model.refinement_inertias_) - 1})
    raw = reviewer2_pd.DataFrame(rows)
    raw.to_csv(out / 'reviewer2_initialization_winners.csv', index=False)
    raw.groupby('winner').size().rename('count').to_csv(out / 'reviewer2_initialization_winner_summary.csv')

def reviewer2_run_joint_sensitivity(out: reviewer2_Path) -> None:
    data_dir = reviewer2_REPO / 'data' / 'processed' / 'main_benchmark'
    datasets = ['two_moons', 'circles', 'varied_density', 'wine', 'digits_0_4']
    fields = ['growth_rate', 'inhibition', 'organ_merge_radius', 'valley_ratio', 'tangent_alignment', 'max_bridge_void', 'birth_quantile', 'steps']
    defaults = [0.18, 0.035, 4.2, 0.28, 0.42, 0.52, 0.92, 120]
    primary = [0.15, 0.045, 3.6, 0.18, 0.35, 0.46, 0.88, 140]
    rng = reviewer2_np.random.default_rng(2026)
    combinations = [defaults, primary]
    for _ in range(6):
        combinations.append([float(rng.choice([0.15, 0.18])), float(rng.choice([0.035, 0.045])), float(rng.choice([3.6, 4.2])), float(rng.choice([0.18, 0.28])), float(rng.choice([0.35, 0.42])), float(rng.choice([0.46, 0.52])), float(rng.choice([0.88, 0.92])), int(rng.choice([120, 140]))])
    rows = []
    for combo_id, values in enumerate(combinations):
        overrides = dict(zip(fields, values))
        for name in datasets:
            for seed in reviewer2_SEEDS:
                frame = reviewer2_pd.read_csv(data_dir / f'{name}_seed{seed}_processed.csv')
                x, y = (frame.iloc[:, 1:].to_numpy(float), frame.iloc[:, 0].to_numpy(int))
                model = reviewer2_mbc(x, len(reviewer2_np.unique(y)), seed, **overrides)
                labels = model.fit_predict(x)
                rows.append({'combination': combo_id, 'dataset': name, 'seed': seed, **overrides, 'ari': reviewer2_adjusted_rand_score(y, labels)})
    raw = reviewer2_pd.DataFrame(rows)
    raw.to_csv(out / 'reviewer2_joint_sensitivity_raw.csv', index=False)
    raw.groupby('combination')['ari'].agg(['mean', 'std', 'min', 'max']).to_csv(out / 'reviewer2_joint_sensitivity_summary.csv')

def reviewer2_run_undersegmentation(out: reviewer2_Path) -> None:
    rows = []
    for seed in reviewer2_PRIMARY_SEEDS:
        x, _ = reviewer2_make_blobs(n_samples=360, centers=[[-2.5, 0], [2.5, 0]], cluster_std=0.35, random_state=seed)
        x = reviewer2_StandardScaler().fit_transform(reviewer2_np.c_[x, reviewer2_np.zeros(len(x))])
        for k in [2, 4, 6]:
            model = reviewer2_mbc(x, k, seed, compact_refinement_steps=0)
            labels = model.fit_predict(x)
            rows.append({'seed': seed, 'supplied_k': k, 'clusters_without_refinement': len(reviewer2_np.unique(labels)), 'clusters_after_compact_refinement': len(reviewer2_np.unique(reviewer2_mbc(x, k, seed).fit_predict(x)))})
    reviewer2_pd.DataFrame(rows).to_csv(out / 'reviewer2_k_undersegmentation.csv', index=False)

def reviewer2_main() -> None:
    out = reviewer2_REPO / 'results' / 'paper_results'
    out.mkdir(parents=True, exist_ok=True)
    reviewer2_run_nonconvex_and_related(out)
    reviewer2_run_channel_diagnostic(out)
    reviewer2_run_initialization_winners(out)
    reviewer2_run_joint_sensitivity(out)
    reviewer2_run_undersegmentation(out)
    print(f'Reviewer 2 analyses written to {out}')


# ============================================================================
# Reviewer 3 analyses
# ============================================================================

"""Reviewer 3 diagnostics for the MBC revision.

The outputs are deliberately separated from the primary benchmark. They test
specific reviewer concerns (geometry, trigger sensitivity, runtime dimensions,
and stress configurations) and do not redefine the original 15-dataset score.
"""
import time as reviewer3_time
from pathlib import Path as reviewer3_Path
import matplotlib as reviewer3_matplotlib
reviewer3_matplotlib.use('Agg')
import matplotlib.pyplot as reviewer3_plt
import numpy as reviewer3_np
import pandas as reviewer3_pd
from sklearn.cluster import KMeans as reviewer3_KMeans
from sklearn.datasets import load_wine as reviewer3_load_wine, make_blobs as reviewer3_make_blobs
from sklearn.decomposition import PCA as reviewer3_PCA
from sklearn.metrics import adjusted_rand_score as reviewer3_adjusted_rand_score
from sklearn.preprocessing import LabelEncoder as reviewer3_LabelEncoder, StandardScaler as reviewer3_StandardScaler
reviewer3_ROOT = reviewer3_Path(__file__).resolve().parents[1]
reviewer3_CODE = reviewer3_ROOT / 'code'
reviewer3_RESULTS = reviewer3_ROOT / 'results' / 'paper_results'
reviewer3_FIGURES = reviewer3_ROOT / 'figures' / 'paper_figures'
import sys as reviewer3_sys
reviewer3_sys.path.insert(0, str(reviewer3_CODE))
from mbc import MorphogeneticBudsClustering as reviewer3_MorphogeneticBudsClustering, squared_distances as reviewer3_squared_distances
reviewer3_SEEDS = [3, 7, 11, 19, 23]

def reviewer3_mbc(x: reviewer3_np.ndarray, k: int, seed: int, **overrides) -> reviewer3_MorphogeneticBudsClustering:
    params = dict(initial_buds=max(12, 6 * k), steps=140, growth_rate=0.15, inhibition=0.045, apoptosis_mass=max(2.5, len(x) * 0.0055), birth_quantile=0.88, organ_merge_radius=3.6, valley_ratio=0.18, tangent_alignment=0.35, max_bridge_void=0.46, target_clusters=k, compact_refinement_steps=18, compact_refinement_restarts=10, compact_refinement_dim_threshold=3, random_state=seed)
    params.update(overrides)
    return reviewer3_MorphogeneticBudsClustering(**params)

def reviewer3_load_main(name: str, seed: int, processed: bool=True) -> tuple[reviewer3_np.ndarray, reviewer3_np.ndarray]:
    suffix = '_processed' if processed else ''
    path = reviewer3_ROOT / 'data' / ('processed' if processed else 'raw') / 'main_benchmark' / f'{name}_seed{seed}{suffix}.csv'
    frame = reviewer3_pd.read_csv(path)
    return (frame.iloc[:, 1:].to_numpy(float), frame.iloc[:, 0].to_numpy(int))

def reviewer3_label_profile(y: reviewer3_np.ndarray) -> str:
    return '/'.join((str(int(v)) for v in reviewer3_np.bincount(y)))

def reviewer3_run_geometry_panel() -> None:
    names = [('two_moons', 'Two Moons'), ('circles', 'Circles (non-convex annuli)'), ('noisy_moons', 'Noisy Two Moons'), ('noisy_circles', 'Noisy Circles'), ('varied_density', 'Varied Density'), ('anisotropic_blobs', 'Anisotropic Blobs')]
    fig, axes = reviewer3_plt.subplots(2, 3, figsize=(10.5, 6.6), dpi=180)
    rows = []
    for ax, (name, title) in zip(axes.flat, names):
        x, y = reviewer3_load_main(name, 3, processed=False)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap='tab10', s=7, linewidths=0, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        rows.append({'dataset': name, 'seed': 3, 'n_samples': len(x), 'n_features': x.shape[1], 'classes': len(reviewer3_np.unique(y)), 'label_counts': reviewer3_label_profile(y)})
    fig.suptitle('Primary Synthetic Generator Geometry', y=0.98)
    fig.tight_layout()
    reviewer3_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(reviewer3_FIGURES / 'reviewer3_synthetic_geometry.pdf')
    fig.savefig(reviewer3_RESULTS / 'reviewer3_synthetic_geometry.png')
    reviewer3_plt.close(fig)
    reviewer3_pd.DataFrame(rows).to_csv(reviewer3_RESULTS / 'reviewer3_synthetic_geometry_inventory.csv', index=False)

def reviewer3_coverage_inertia(x: reviewer3_np.ndarray, centers: reviewer3_np.ndarray) -> float:
    distances = reviewer3_squared_distances(x, centers).min(axis=1)
    return float(distances.sum())

def reviewer3_run_bud_dynamics() -> None:
    rows = []
    x_moons, y_moons = reviewer3_load_main('two_moons', 3, processed=True)
    moon_model = reviewer3_mbc(x_moons, len(reviewer3_np.unique(y_moons)), 3, compact_refinement_steps=0).fit(x_moons)
    snapshots = [0, 34, 69, 139]
    fig, axes = reviewer3_plt.subplots(1, 4, figsize=(12.0, 3.1), dpi=180)
    for ax, idx in zip(axes, snapshots):
        centers = moon_model.history_[idx]
        ax.scatter(x_moons[:, 0], x_moons[:, 1], c=y_moons, cmap='tab10', s=6, alpha=0.42, linewidths=0)
        ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=34, linewidths=1.0)
        ax.set_title(f'iteration {idx + 1}; buds={len(centers)}')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
    fig.suptitle('Micro-bud Evolution on Two Moons')
    fig.tight_layout()
    fig.savefig(reviewer3_FIGURES / 'reviewer3_bud_evolution.pdf')
    fig.savefig(reviewer3_RESULTS / 'reviewer3_bud_evolution.png')
    reviewer3_plt.close(fig)
    for iteration, centers in enumerate(moon_model.history_, 1):
        rows.append({'representation': 'two_moons_2d', 'iteration': iteration, 'buds': len(centers), 'coverage_inertia': reviewer3_coverage_inertia(x_moons, centers)})
    wine = reviewer3_load_wine()
    wine_raw = reviewer3_StandardScaler().fit_transform(wine.data)
    wine_pca = reviewer3_PCA(n_components=10, random_state=0).fit_transform(wine_raw)
    for representation, x in [('wine_original_13d', wine_raw), ('wine_pca_10d', wine_pca)]:
        model = reviewer3_mbc(x, len(reviewer3_np.unique(wine.target)), 3).fit(x)
        for iteration, centers in enumerate(model.history_, 1):
            rows.append({'representation': representation, 'iteration': iteration, 'buds': len(centers), 'coverage_inertia': reviewer3_coverage_inertia(x, centers)})
        rows.append({'representation': representation, 'iteration': 141, 'buds': len(model.centers_), 'coverage_inertia': float(reviewer3_squared_distances(x, model.organ_centers_).min(axis=1).sum()), 'stage': 'post_refinement'})
    trajectory = reviewer3_pd.DataFrame(rows)
    trajectory.to_csv(reviewer3_RESULTS / 'reviewer3_dynamics_trajectory.csv', index=False)
    fig, ax = reviewer3_plt.subplots(figsize=(7.6, 4.2), dpi=180)
    for representation, group in trajectory.groupby('representation'):
        group = group[group['iteration'] <= 140]
        ax.plot(group['iteration'], group['coverage_inertia'], label=representation)
    ax.set_xlabel('morphogenetic iteration')
    ax.set_ylabel('nearest-bud coverage inertia')
    ax.set_title('Finite-Step Bud Dynamics Before Compact Refinement')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(reviewer3_FIGURES / 'reviewer3_dynamics_trajectory.pdf')
    fig.savefig(reviewer3_RESULTS / 'reviewer3_dynamics_trajectory.png')
    reviewer3_plt.close(fig)

def reviewer3_run_trigger_sensitivity() -> None:
    rows = []
    datasets = ['two_moons', 'circles', 'iris', 'wine', 'digits_0_4']
    for dataset in datasets:
        for seed in reviewer3_SEEDS:
            x, y = reviewer3_load_main(dataset, seed, processed=True)
            for trigger in [2, 3, 4]:
                model = reviewer3_mbc(x, len(reviewer3_np.unique(y)), seed, compact_refinement_dim_threshold=trigger)
                labels = model.fit_predict(x)
                rows.append({'dataset': dataset, 'seed': seed, 'dimension': x.shape[1], 'trigger': trigger, 'refinement_active': int(x.shape[1] >= trigger), 'ari': reviewer3_adjusted_rand_score(y, labels)})
    raw = reviewer3_pd.DataFrame(rows)
    raw.to_csv(reviewer3_RESULTS / 'reviewer3_trigger_sensitivity_raw.csv', index=False)
    raw.groupby('trigger')['ari'].agg(['mean', 'std', 'min', 'max']).to_csv(reviewer3_RESULTS / 'reviewer3_trigger_sensitivity_summary.csv')

def reviewer3_timed_fit(x: reviewer3_np.ndarray, k: int, seed: int, **overrides) -> float:
    start = reviewer3_time.perf_counter()
    reviewer3_mbc(x, k, seed, **overrides).fit_predict(x)
    return reviewer3_time.perf_counter() - start

def reviewer3_make_runtime_data(n: int, k: int, d: int, seed: int) -> reviewer3_np.ndarray:
    centers = reviewer3_np.zeros((k, d))
    centers[:, :min(k, d)] = reviewer3_np.eye(k, d)[:, :min(k, d)] * 5.0
    x, _ = reviewer3_make_blobs(n_samples=n, centers=centers, cluster_std=1.0, random_state=seed)
    return reviewer3_StandardScaler().fit_transform(x)

def reviewer3_run_independent_runtime() -> None:
    rows = []
    settings = []
    settings.extend((('n', value, value, 4, 10) for value in [500, 1000, 2000]))
    settings.extend((('k', value, 900, value, 10) for value in [2, 4, 8, 16]))
    settings.extend((('d', value, 900, 4, value) for value in [2, 10, 30, 60]))
    for axis, value, n, k, d in settings:
        x = reviewer3_make_runtime_data(n, k, d, 3)
        seconds = reviewer3_timed_fit(x, k, 3, steps=60, compact_refinement_steps=12)
        rows.append({'axis': axis, 'value': value, 'n_samples': n, 'target_k': k, 'dimension': d, 'steps': 60, 'seconds': seconds})
    raw = reviewer3_pd.DataFrame(rows)
    raw.to_csv(reviewer3_RESULTS / 'reviewer3_runtime_axes_raw.csv', index=False)
    fig, axes = reviewer3_plt.subplots(1, 3, figsize=(10.6, 3.3), dpi=180)
    for ax, axis, xlabel in zip(axes, ['n', 'k', 'd'], ['samples n', 'target count K', 'dimension d']):
        group = raw[raw['axis'] == axis]
        ax.plot(group['value'], group['seconds'], marker='o')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('seconds')
        ax.grid(alpha=0.25)
    fig.suptitle('Independent Runtime Diagnostics for MBC')
    fig.tight_layout()
    fig.savefig(reviewer3_FIGURES / 'reviewer3_runtime_axes.pdf')
    fig.savefig(reviewer3_RESULTS / 'reviewer3_runtime_axes.png')
    reviewer3_plt.close(fig)

def reviewer3_run_stress_cases() -> None:
    rows = []
    rng = reviewer3_np.random.default_rng(3)
    large_x, large_y = reviewer3_make_blobs(n_samples=[9000, 1100], centers=[[-2.0, 0.0], [2.0, 0.0]], cluster_std=[0.85, 0.85], random_state=3)
    large_x = reviewer3_StandardScaler().fit_transform(large_x)
    start = reviewer3_time.perf_counter()
    large_labels = reviewer3_mbc(large_x, 2, 3, steps=80).fit_predict(large_x)
    rows.append({'case': 'large_n_10100_imbalanced', 'n_original': len(large_x), 'd_original': 2, 'd_used': 2, 'K': 2, 'largest_to_smallest': '9000:1100', 'algorithm': 'MBC', 'ari': reviewer3_adjusted_rand_score(large_y, large_labels), 'seconds': reviewer3_time.perf_counter() - start})
    counts = reviewer3_np.linspace(90, 12, 20).astype(int)
    high_parts = []
    high_labels = []
    centers = rng.normal(0, 1.3, size=(20, 6000))
    centers[:, :20] += reviewer3_np.eye(20) * 8.0
    for label, count in enumerate(counts):
        high_parts.append(rng.normal(centers[label], 0.8, size=(count, 6000)))
        high_labels.extend([label] * count)
    high_x = reviewer3_np.vstack(high_parts)
    high_y = reviewer3_np.asarray(high_labels)
    high_scaled = reviewer3_StandardScaler().fit_transform(high_x)
    high_pca = reviewer3_PCA(n_components=10, random_state=0).fit_transform(high_scaled)
    start = reviewer3_time.perf_counter()
    high_labels_pred = reviewer3_mbc(high_pca, 20, 3, steps=100).fit_predict(high_pca)
    rows.append({'case': 'high_d_6000_K20_imbalanced', 'n_original': len(high_x), 'd_original': 6000, 'd_used': 10, 'K': 20, 'largest_to_smallest': f'{counts.max()}:{counts.min()}', 'algorithm': 'MBC', 'ari': reviewer3_adjusted_rand_score(high_y, high_labels_pred), 'seconds': reviewer3_time.perf_counter() - start})
    reviewer3_pd.DataFrame(rows).to_csv(reviewer3_RESULTS / 'reviewer3_stress_cases.csv', index=False)

def reviewer3_run_dataset_inventory() -> None:
    rows = []
    sources = {'iris': 'scikit-learn load_iris', 'wine': 'scikit-learn load_wine', 'breast_cancer': 'scikit-learn load_breast_cancer', 'digits_0_4': 'scikit-learn load_digits, labels 0--4', 'two_moons': 'scikit-learn make_moons', 'circles': 'scikit-learn make_circles', 'noisy_moons': 'scikit-learn make_moons', 'noisy_circles': 'scikit-learn make_circles', 'varied_density': 'project generator make_varied_density', 'anisotropic_blobs': 'scikit-learn make_blobs plus linear transform', 'ecoli': 'OpenML ecoli', 'glass': 'OpenML glass', 'vehicle': 'OpenML vehicle', 'cmc': 'OpenML cmc', 'wdbc': 'OpenML wdbc'}
    families = {'two_moons': 'synthetic non-convex', 'circles': 'synthetic non-convex', 'noisy_moons': 'synthetic non-convex', 'noisy_circles': 'synthetic non-convex', 'varied_density': 'density-varying', 'anisotropic_blobs': 'compact Gaussian', 'iris': 'real low-dimensional', 'wine': 'real tabular', 'breast_cancer': 'real tabular', 'digits_0_4': 'image-feature', 'ecoli': 'real tabular', 'glass': 'real tabular', 'vehicle': 'real tabular', 'cmc': 'real tabular', 'wdbc': 'real tabular'}
    names = sorted((p.name.removesuffix('_seed3_processed.csv') for p in (reviewer3_ROOT / 'data' / 'processed' / 'main_benchmark').glob('*_seed3_processed.csv')))
    for name in names:
        raw_x, raw_y = reviewer3_load_main(name, 3, processed=False)
        proc_x, proc_y = reviewer3_load_main(name, 3, processed=True)
        rows.append({'scope': 'primary', 'dataset': name, 'source': sources[name], 'type': families[name], 'n_original': len(raw_x), 'n_used': len(proc_x), 'd_original': raw_x.shape[1], 'd_used': proc_x.shape[1], 'K': len(reviewer3_np.unique(proc_y)), 'class_counts': reviewer3_label_profile(proc_y)})
    external = [('MiceProtein_8class', 'OpenML 40966', 'gene-expression/protein'), ('Leukemia_2class', 'OpenML 45090', 'gene-expression'), ('Leukemia_3class', 'OpenML 45091', 'gene-expression'), ('Lymphoma_3class', 'OpenML 45094', 'gene-expression'), ('Lung_5class', 'OpenML 45093', 'gene-expression')]
    for name, source, kind in external:
        processed_path = reviewer3_ROOT / 'data' / 'processed' / 'external_real_benchmark' / f'{name}_processed.csv'
        if processed_path.exists():
            frame = reviewer3_pd.read_csv(processed_path)
            y = frame.iloc[:, 0].to_numpy(int)
            rows.append({'scope': 'external', 'dataset': name, 'source': source, 'type': kind, 'n_original': len(frame), 'n_used': len(frame), 'd_original': 'see source', 'd_used': frame.shape[1] - 1, 'K': len(reviewer3_np.unique(y)), 'class_counts': reviewer3_label_profile(y)})
            continue
        raw_candidates = sorted((reviewer3_ROOT / 'data' / 'raw' / 'openml').glob(f'openml_{source.split()[-1]}_*.csv'))
        if not raw_candidates:
            print(f'[warn] skipped inventory dataset {name}: processed and cached raw files are absent')
            continue
        raw = reviewer3_pd.read_csv(raw_candidates[0])
        target_column = next((column for column in ('type', 'class', 'target') if column in raw.columns), raw.columns[-1])
        y = reviewer3_LabelEncoder().fit_transform(raw.pop(target_column).astype(str))
        rows.append({'scope': 'external', 'dataset': name, 'source': f'{source} raw-cache fallback', 'type': kind, 'n_original': len(raw), 'n_used': len(raw), 'd_original': raw.shape[1], 'd_used': 'raw fallback', 'K': len(reviewer3_np.unique(y)), 'class_counts': reviewer3_label_profile(y)})
    rows.append({'scope': 'external', 'dataset': 'Asian Games Village YYC200', 'source': 'bundled ENVI BSQ scene', 'type': 'hyperspectral', 'n_original': 40000, 'n_used': 40000, 'd_original': 64, 'd_used': 10, 'K': 'unlabeled; illustrative K=6', 'class_counts': 'not available'})
    reviewer3_pd.DataFrame(rows).to_csv(reviewer3_RESULTS / 'reviewer3_dataset_inventory.csv', index=False)

def reviewer3_main() -> None:
    reviewer3_RESULTS.mkdir(parents=True, exist_ok=True)
    reviewer3_run_geometry_panel()
    reviewer3_run_bud_dynamics()
    reviewer3_run_trigger_sensitivity()
    reviewer3_run_independent_runtime()
    reviewer3_run_stress_cases()
    reviewer3_run_dataset_inventory()
    benchmark = reviewer3_pd.read_csv(reviewer3_RESULTS / 'paper_benchmark_raw.csv')
    benchmark.groupby('algorithm')['seconds'].sum().sort_values().rename('total_seconds').to_csv(reviewer3_RESULTS / 'reviewer3_full_suite_wallclock.csv')
    print(f'Reviewer 3 analyses written to {reviewer3_RESULTS}')


# ============================================================================
# Dataset export and manifest generation
# ============================================================================

import json as export_data_json
from pathlib import Path as export_data_Path
import numpy as export_data_np
import pandas as export_data_pd
from sklearn.datasets import fetch_openml as export_data_fetch_openml
from sklearn.preprocessing import LabelEncoder as export_data_LabelEncoder
export_data_ext = external_api
export_data_main_exp = main_benchmark
export_data_ROOT = export_data_Path(__file__).resolve().parents[1]
export_data_DATA_DIR = export_data_ROOT / 'data'
export_data_RAW_DIR = export_data_DATA_DIR / 'raw'
export_data_PROCESSED_DIR = export_data_DATA_DIR / 'processed'
export_data_SEEDS = [3, 7, 11, 19, 23]

def export_data_safe_name(name: str) -> str:
    keep = []
    for ch in str(name):
        keep.append(ch if ch.isalnum() or ch in ('-', '_') else '_')
    return ''.join(keep).strip('_')

def export_data_save_matrix(path: export_data_Path, x: export_data_np.ndarray, y: export_data_np.ndarray, **meta) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = export_data_pd.DataFrame(x, columns=[f'x{j:03d}' for j in range(x.shape[1])])
    df.insert(0, 'target', y)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    return {'file': str(path.relative_to(export_data_ROOT)).replace('\\', '/'), 'n_samples': int(x.shape[0]), 'n_features': int(x.shape[1]), **meta}

def export_data_export_main_benchmark() -> list[dict]:
    entries = []
    for seed in export_data_SEEDS:
        for spec in export_data_main_exp.make_datasets(seed):
            raw_file = export_data_RAW_DIR / 'main_benchmark' / f'{export_data_safe_name(spec.name)}_seed{seed}.csv'
            entries.append(export_data_save_matrix(raw_file, export_data_np.asarray(spec.x), export_data_np.asarray(spec.y), dataset=spec.name, seed=seed, type='main_raw'))
            processed = export_data_main_exp.preprocess(export_data_np.asarray(spec.x))
            proc_file = export_data_PROCESSED_DIR / 'main_benchmark' / f'{export_data_safe_name(spec.name)}_seed{seed}_processed.csv'
            entries.append(export_data_save_matrix(proc_file, processed, export_data_np.asarray(spec.y), dataset=spec.name, seed=seed, type='main_processed'))
    return entries

def export_data_unique_openml_specs() -> list[dict]:
    by_id: dict[int, dict] = {}
    for spec in export_data_ext.OPENML_DATASETS:
        by_id.setdefault(spec['id'], {'id': spec['id'], 'names': []})
        by_id[spec['id']]['names'].append(spec['name'])
    return list(by_id.values())

def export_data_export_openml_raw() -> list[dict]:
    entries = []
    for spec in export_data_unique_openml_specs():
        data_id = spec['id']
        local_candidates = sorted((export_data_RAW_DIR / 'openml').glob(f'openml_{data_id}_*.csv'))
        if local_candidates:
            path = local_candidates[0]
            frame = export_data_pd.read_csv(path)
            entries.append({'file': str(path.relative_to(export_data_ROOT)).replace('\\', '/'), 'openml_id': data_id, 'openml_name': path.stem, 'used_as': sorted(set(spec['names'])), 'n_samples': int(frame.shape[0]), 'n_columns_including_target': int(frame.shape[1]), 'type': 'openml_raw'})
            continue
        try:
            bunch = export_data_fetch_openml(data_id=data_id, as_frame=True, parser='auto')
        except Exception as exc:
            print(f'[warn] skipped raw OpenML dataset {data_id}: {exc}')
            continue
        frame = bunch.frame.copy()
        filename = f"openml_{data_id}_{export_data_safe_name(bunch.details.get('name', spec['names'][0]))}.csv"
        path = export_data_RAW_DIR / 'openml' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding='utf-8-sig')
        entries.append({'file': str(path.relative_to(export_data_ROOT)).replace('\\', '/'), 'openml_id': data_id, 'openml_name': bunch.details.get('name', ''), 'used_as': sorted(set(spec['names'])), 'n_samples': int(frame.shape[0]), 'n_columns_including_target': int(frame.shape[1]), 'type': 'openml_raw'})
    return entries

def export_data_export_external_processed() -> list[dict]:
    entries = []
    for spec in export_data_ext.OPENML_DATASETS:
        try:
            name, family, x, y = export_data_ext.load_openml_dataset(spec)
        except Exception as exc:
            print(f"[warn] skipped processed OpenML dataset {spec['name']}: {exc}")
            continue
        path = export_data_PROCESSED_DIR / 'external_real_benchmark' / f'{export_data_safe_name(name)}_processed.csv'
        entries.append(export_data_save_matrix(path, x, y, dataset=name, family=family, openml_id=spec['id'], type='external_processed'))
    return entries

def export_data_write_manifest(entries: list[dict]) -> None:
    export_data_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {'description': 'Datasets bundled for the MBC EI submission package. Raw data preserve source tables when available; processed data are the matrices used by the paper scripts after encoding, scaling, PCA and/or sampling.', 'entries': entries}
    (export_data_DATA_DIR / 'dataset_manifest.json').write_text(export_data_json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    rows = []
    for e in entries:
        rows.append({'file': e.get('file'), 'type': e.get('type'), 'dataset': e.get('dataset') or e.get('openml_name') or ','.join(e.get('used_as', [])), 'source_id': e.get('openml_id') or e.get('source') or '', 'n_samples': e.get('n_samples'), 'n_features_or_columns': e.get('n_features') or e.get('n_columns') or e.get('n_columns_including_target')})
    export_data_pd.DataFrame(rows).to_csv(export_data_DATA_DIR / 'dataset_manifest.csv', index=False, encoding='utf-8-sig')

def export_data_main() -> None:
    entries: list[dict] = []
    entries.extend(export_data_export_main_benchmark())
    entries.extend(export_data_export_openml_raw())
    entries.extend(export_data_export_external_processed())
    export_data_write_manifest(entries)
    print(export_data_json.dumps({'data_dir': str(export_data_DATA_DIR), 'entries': len(entries)}, indent=2))


# ============================================================================
# Unified execution entry point
# ============================================================================

RUNNERS = [
    ("Export bundled datasets", export_data_main),
    ("Main benchmark", main_benchmark_main),
    ("Sensitivity analysis", sensitivity_main),
    ("External real-data benchmark", external_main),
    ("Runtime and biological visuals", runtime_bio_main),
    ("YYC200 case study", hyperspectral_main),
    ("Cluster figures", cluster_figures_main),
    ("Advanced figures", advanced_figures_main),
    ("YYC200 analysis", yyc_analysis_main),
    ("Reviewer 1 analyses", reviewer1_main),
    ("Reviewer 2 analyses", reviewer2_main),
    ("Reviewer 3 analyses", reviewer3_main)
]


def normalize_generated_files() -> None:
    table_source = RESULT_DIR / "paper_tables.tex"
    table_target = RESULT_DIR / "paper_tables_latex_snippet.txt"
    if table_source.exists():
        if table_target.exists():
            table_target.unlink()
        table_source.replace(table_target)

    for name in [
        "publication_bio_expression_pca.pdf",
        "publication_bio_expression_pca.png",
        "publication_nonconvex_showcase.pdf",
        "publication_nonconvex_showcase.png",
    ]:
        source = PACKAGE_ROOT / "figures" / "paper_figures" / name
        if source.exists():
            target = RESULT_DIR / name
            if target.exists():
                target.unlink()
            shutil.move(str(source), str(target))

    preview_dir = PACKAGE_ROOT / "figures" / "paper_png_previews"
    figure_dir = PACKAGE_ROOT / "figures" / "paper_figures"
    if figure_dir.exists():
        preview_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(figure_dir.glob("*.png")):
            target = preview_dir / source.name
            if target.exists():
                target.unlink()
            shutil.move(str(source), str(target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all manuscript and reviewer analyses for the MBC paper."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the ordered analyses without running them",
    )
    parser.add_argument(
        "--start-at",
        metavar="LABEL",
        help="resume at an analysis label shown by --list",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for index, (label, _) in enumerate(RUNNERS, 1):
            print(f"{index:02d}. {label}")
        return

    runners = RUNNERS
    if args.start_at:
        labels = [label for label, _ in RUNNERS]
        if args.start_at not in labels:
            available = ", ".join(labels)
            raise ValueError(f"Unknown --start-at label: {args.start_at}. Choose one of: {available}")
        runners = RUNNERS[labels.index(args.start_at) :]

    for label, routine in runners:
        start = time.perf_counter()
        print(f"\n==> {label}", flush=True)
        routine()
        print(f"<== completed in {time.perf_counter() - start:.1f} s", flush=True)

    normalize_generated_files()
    print("\nAll manuscript and reviewer analyses completed.", flush=True)


if __name__ == "__main__":
    main()
