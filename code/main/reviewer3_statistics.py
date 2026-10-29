"""Recompute Reviewer 3 primary-suite block statistics from bundled ARI rows.

This is a lightweight statistical stage: it does not run clustering methods.
The raw input is results/exp/primary_ari.csv, which includes five seeds for
fourteen primary datasets and all eleven methods used in the revised tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "results" / "exp"
PRIMARY = EXP / "primary_ari.csv"

BLOCKS = {
    "Moons": ["two_moons", "noisy_moons"],
    "Circles": ["circles", "noisy_circles"],
    "Varied density": ["varied_density"],
    "Anisotropic blobs": ["anisotropic_blobs"],
    "Iris": ["iris"],
    "Wine": ["wine"],
    "Breast Cancer": ["breast_cancer"],
    "Digits 0-4": ["digits_0_4"],
    "Ecoli": ["ecoli"],
    "Glass": ["glass"],
    "Vehicle": ["vehicle"],
    "CMC": ["cmc"],
}


def main() -> None:
    raw = pd.read_csv(PRIMARY)
    methods = sorted(raw["method"].unique().tolist())
    expected_methods = {
        "AffinityPropagation", "Agglomerative", "CompactRefineOnly", "DBSCAN",
        "GMM", "GNG", "HDBSCAN", "KMeans", "MBC", "MeanShift", "Spectral",
    }
    if set(methods) != expected_methods:
        raise ValueError(f"Unexpected primary method set: {methods}")

    means = raw.pivot_table(index="dataset", columns="method", values="ari", aggfunc="mean")
    expected_datasets = {dataset for group in BLOCKS.values() for dataset in group}
    if set(means.index) != expected_datasets:
        raise ValueError(f"Unexpected primary dataset set: {sorted(means.index)}")
    if raw.groupby(["dataset", "method"]).size().nunique() != 1:
        raise ValueError("Primary results do not have a balanced seed count")

    raw_matrix = means[methods].to_numpy().T
    raw_chi2, raw_p = friedmanchisquare(*raw_matrix)
    collapsed = pd.DataFrame(
        {name: means.loc[datasets, methods].mean(axis=0) for name, datasets in BLOCKS.items()}
    ).T
    collapsed.index.name = "block"
    collapsed.to_csv(EXP / "primary_block_means.csv", float_format="%.12g")

    block_matrix = collapsed[methods].to_numpy().T
    block_chi2, block_p = friedmanchisquare(*block_matrix)
    n_blocks, n_methods = len(BLOCKS), len(methods)
    iman_davenport_f = ((n_blocks - 1) * block_chi2) / (
        n_blocks * (n_methods - 1) - block_chi2
    )
    iman_davenport_p = float(
        f_dist.sf(iman_davenport_f, n_methods - 1, (n_methods - 1) * (n_blocks - 1))
    )

    mbc = collapsed["MBC"]
    wilcoxon_rows = []
    for method in methods:
        if method == "MBC":
            continue
        other = collapsed[method]
        differences = mbc - other
        test = wilcoxon(
            mbc,
            other,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        wilcoxon_rows.append({
            "comparison": f"MBC vs {method}",
            "n_blocks": n_blocks,
            "n_nonzero_differences": int(np.count_nonzero(~np.isclose(differences, 0.0))),
            "mbc_mean": float(mbc.mean()),
            "other_mean": float(other.mean()),
            "mean_difference": float(differences.mean()),
            "wilcoxon_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
        })
    pd.DataFrame(wilcoxon_rows).to_csv(
        EXP / "primary_block_wilcoxon.csv", index=False, float_format="%.12g"
    )

    result = {
        "input": "results/exp/primary_ari.csv",
        "methods": methods,
        "n_raw_datasets": int(len(means)),
        "n_blocks": n_blocks,
        "block_definitions": BLOCKS,
        "raw_friedman_chi2": float(raw_chi2),
        "raw_friedman_p": float(raw_p),
        "block_friedman_chi2": float(block_chi2),
        "block_friedman_p": float(block_p),
        "iman_davenport_f": float(iman_davenport_f),
        "iman_davenport_p": iman_davenport_p,
        "wilcoxon_method": "paired, two-sided; zero differences excluded",
        "wilcoxon_output": "results/exp/primary_block_wilcoxon.csv",
    }
    (EXP / "primary_block_statistics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (EXP / "friedman.txt").write_text(
        f"RAW 14 datasets x {n_methods} methods: chi2={raw_chi2:.3f} p={raw_p:.4f}\n"
        f"BLOCK 12 datasets x {n_methods} methods: chi2={block_chi2:.3f} p={block_p:.4f}\n"
        f"Iman-Davenport F={iman_davenport_f:.3f} p={iman_davenport_p:.4f}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
