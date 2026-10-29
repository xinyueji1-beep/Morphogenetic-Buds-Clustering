"""Run and audit the canonical Growing Neural Gas comparator.

This script adds the true GNG rows to the cached primary and scale ledgers.
All other rows remain the already-run results, while the GNG rows carry their
own wall-clock measurements and realized node counts.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import friedmanchisquare
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from exp_driver import DATASETS, SEEDS, load, run_baseline  # noqa: E402
from reviewer3_scale_all_methods import high_d, large_n  # noqa: E402

EXP = ROOT / "results" / "exp"
PAPER = ROOT / "results" / "paper_results"
PRIMARY = EXP / "primary_ari.csv"


from scale_geometry import gen_scaled_moons


def run_primary() -> pd.DataFrame:
    old = pd.read_csv(PRIMARY)
    backup = EXP / "primary_ari_before_true_gng.csv"
    if not backup.exists():
        shutil.copy2(PRIMARY, backup)
    old = old[old.method != "GNG"].copy()
    rows = []
    for dataset, (K, family) in DATASETS.items():
        for seed in SEEDS:
            X, y = load(dataset, seed)
            started = time.perf_counter()
            labels, realized_nodes = run_baseline("GNG", X, K, seed)
            seconds = time.perf_counter() - started
            rows.append({
                "dataset": dataset, "family": family, "K": K, "seed": seed,
                "method": "GNG", "ari": adjusted_rand_score(y, labels),
                "k_found": realized_nodes, "time_s": seconds,
            })
            print("primary", dataset, seed, rows[-1]["ari"], realized_nodes, seconds, flush=True)
    new = pd.DataFrame(rows)
    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.sort_values(["dataset", "seed", "method"]).reset_index(drop=True)
    merged.to_csv(PRIMARY, index=False)
    new.to_csv(EXP / "primary_true_gng.csv", index=False)
    return merged


def recompute_primary_stats(prim: pd.DataFrame) -> dict:
    methods = list(prim.method.drop_duplicates())
    ds_order = list(DATASETS)
    piv = prim.pivot_table(index="dataset", columns="method", values="ari", aggfunc="mean")
    family_order = ["Non-convex", "Gaussian", "Density", "Image", "Low-d", "Tabular"]
    family = {}
    for method in methods:
        family[method] = {
            fam: float(piv.loc[[d for d in ds_order if DATASETS[d][1] == fam], method].mean())
            for fam in family_order
        }
    pd.DataFrame(family).T.to_csv(EXP / "family_means.csv")
    raw = np.asarray([piv.loc[ds_order, method] for method in methods])
    raw_stat, raw_p = friedmanchisquare(*raw)
    blocks = [["two_moons", "noisy_moons"], ["circles", "noisy_circles"],
              ["varied_density"], ["anisotropic_blobs"], ["iris"], ["wine"],
              ["breast_cancer"], ["digits_0_4"], ["ecoli"], ["glass"],
              ["vehicle"], ["cmc"]]
    collapsed = np.asarray([[piv.loc[b, method].mean() for b in blocks] for method in methods])
    block_stat, block_p = friedmanchisquare(*collapsed)
    k, n_blocks = len(methods), len(blocks)
    # Iman-Davenport correction: F_F = (N - 1) chi2 / (N(k - 1) - chi2).
    F = ((n_blocks - 1) * block_stat) / (n_blocks * (k - 1) - block_stat)
    p_id = float(f_dist.sf(F, k - 1, (k - 1) * (n_blocks - 1)))
    summary = {"methods": methods, "raw_chi2": float(raw_stat), "raw_p": float(raw_p),
               "block_chi2": float(block_stat), "block_p": float(block_p),
               "iman_davenport_F": float(F), "iman_davenport_p": p_id,
               "gng_mean": float(piv["GNG"].mean()),
               "gng_nonconvex": float(piv.loc[[d for d in ds_order if DATASETS[d][1] == "Non-convex"], "GNG"].mean())}
    (EXP / "true_gng_primary_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_scale() -> pd.DataFrame:
    raw_path = PAPER / "reviewer3_scale_all_methods_raw.csv"
    old = pd.read_csv(raw_path)
    backup = PAPER / "reviewer3_scale_all_methods_before_true_gng.csv"
    if not backup.exists():
        shutil.copy2(raw_path, backup)
    old = old[old.method != "GNG"].copy()
    rows = []
    for seed in (3, 7, 11):
        for case, generator in (("large_n_imbalanced_largeK_d30", large_n),
                                ("high_d_6000_K20_imbalanced_pca10", high_d)):
            X, y, K, d_used = generator(seed)
            started = time.perf_counter()
            # Three epochs keep the bounded scale stress within a reproducible
            # budget while retaining the full GNG growth/pruning mechanism.
            from gng import GrowingNeuralGas
            model = GrowingNeuralGas(
                n_epochs=3, max_nodes=max(32, min(120, 12 * K)),
                insertion_interval=max(50, len(X) // 4), max_age=50,
                random_state=seed)
            labels = model.fit_predict(X, K)
            seconds = time.perf_counter() - started
            rows.append({"case": case, "n": len(X),
                         "d_original": 6000 if case.startswith("high_d") else 30,
                         "d_used": d_used, "K": K, "seed": seed, "method": "GNG",
                         "ari": adjusted_rand_score(y, labels), "seconds": seconds,
                         "warnings": "", "realized_nodes": model.n_nodes_,
                         "actual_clusters": len(np.unique(labels)),
                         "insertions": model.result_.insertions,
                         "edge_deletions": model.result_.edge_deletions,
                         "node_deletions": model.result_.node_deletions})
            print("scale", case, seed, rows[-1]["ari"], model.n_nodes_, seconds, flush=True)
    new = pd.DataFrame(rows)
    new.to_csv(PAPER / "reviewer3_true_gng_scale_raw.csv", index=False)
    old = old.drop(columns=[c for c in ["realized_nodes", "insertions", "edge_deletions", "node_deletions"] if c in old], errors="ignore")
    merged = pd.concat([old, new[old.columns]], ignore_index=True)
    merged.to_csv(raw_path, index=False)
    return new


def run_nonconvex_scale() -> pd.DataFrame:
    rows = []
    from gng import GrowingNeuralGas
    for seed in (3, 7, 11):
        for n in (1000, 2000, 4000, 8000):
            X, y, K = gen_scaled_moons(n, seed)
            started = time.perf_counter()
            model = GrowingNeuralGas(n_epochs=3, max_nodes=48,
                                     insertion_interval=max(50, n // 4),
                                     max_age=50, random_state=seed)
            labels = model.fit_predict(X, K)
            seconds = time.perf_counter() - started
            rows.append({"n": n, "seed": seed, "method": "GNG",
                         "ari": adjusted_rand_score(y, labels), "time_s": seconds,
                         "realized_nodes": model.n_nodes_,
                         "insertions": model.result_.insertions,
                         "edge_deletions": model.result_.edge_deletions,
                         "node_deletions": model.result_.node_deletions})
            print("nonconvex", n, seed, rows[-1]["ari"], model.n_nodes_, seconds, flush=True)
    new = pd.DataFrame(rows)
    new.to_csv(EXP / "true_gng_nonconvex_scale.csv", index=False)
    return new


if __name__ == "__main__":
    primary = run_primary()
    summary = recompute_primary_stats(primary)
    scale = run_scale()
    nonconvex = run_nonconvex_scale()
    manifest = {"host": platform.platform(), "python": sys.executable,
                "primary_rows": len(primary), "scale_gng_rows": len(scale),
                "nonconvex_gng_rows": len(nonconvex), "stats": summary}
    (EXP / "true_gng_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

