"""Re-run every manuscript comparator on both compact scale generators."""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from exp_driver import run_baseline


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_results" / "reviewer3_scale_all_methods_raw.csv"
IN_PROGRESS = OUT.with_name(f"{OUT.stem}_in_progress{OUT.suffix}")
SEEDS = (3, 7, 11)
METHODS = (
    "MBC", "KMeans", "GMM", "Agglomerative", "Spectral", "DBSCAN",
    "HDBSCAN", "MeanShift", "AffinityPropagation", "CompactRefineOnly", "GNG",
)


def large_n(seed):
    rng = np.random.default_rng(seed + 4000)
    k = 15
    directions = rng.normal(0.0, 1.0, size=(k, 30))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    centers = 8.0 * directions
    sizes = np.round(np.geomspace(2000, 120, k)).astype(int)
    x = np.vstack([rng.normal(center, 1.0, size=(int(count), 30))
                   for center, count in zip(centers, sizes)])
    y = np.concatenate([np.full(int(count), label, dtype=int)
                        for label, count in enumerate(sizes)])
    return StandardScaler().fit_transform(x), y, k, 30


def high_d(seed):
    rng = np.random.default_rng(seed + 5000)
    counts = np.linspace(90, 12, 20).astype(int)
    centers = rng.normal(0.0, 1.3, size=(20, 6000))
    centers[:, :20] += np.eye(20) * 8.0
    x = np.vstack([rng.normal(centers[label], 0.8, size=(int(count), 6000))
                   for label, count in enumerate(counts)])
    y = np.concatenate([np.full(int(count), label, dtype=int)
                        for label, count in enumerate(counts)])
    x = StandardScaler().fit_transform(x)
    x = PCA(n_components=10, random_state=0).fit_transform(x)
    return x, y, 20, 10


def main():
    rows = []
    for seed in SEEDS:
        for case, generator in (("large_n_imbalanced_largeK_d30", large_n),
                                ("high_d_6000_K20_imbalanced_pca10", high_d)):
            x, y, k, d_used = generator(seed)
            for method in METHODS:
                caught = []
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    labels, _ = run_baseline(method, x, k, seed)
                    caught = [f"{type(w.message).__name__}: {w.message}" for w in seen]
                row = {
                    "case": case,
                    "n": len(x),
                    "d_original": 6000 if case.startswith("high_d") else 30,
                    "d_used": d_used,
                    "K": k,
                    "seed": seed,
                    "method": method,
                    "ari": adjusted_rand_score(y, labels),
                    "seconds": time.perf_counter() - started,
                    "warnings": " | ".join(caught),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(IN_PROGRESS, index=False)
                print(row, flush=True)

    # Keep the last complete ledger intact if a later run stops partway through.
    IN_PROGRESS.replace(OUT)


if __name__ == "__main__":
    main()
