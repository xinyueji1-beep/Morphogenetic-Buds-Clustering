"""Run the three missing R3 comparators on the two compact scale cases."""

from pathlib import Path
import platform
import time

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation, MeanShift
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_results"
SEEDS = (3, 7, 11)


def large_n(seed):
    rng = np.random.default_rng(seed + 4000)
    centers = rng.normal(0.0, 1.0, size=(15, 30))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    centers *= 8.0
    sizes = np.round(np.geomspace(2000, 120, 15)).astype(int)
    x = np.vstack([rng.normal(center, 1.0, size=(int(count), 30))
                   for center, count in zip(centers, sizes)])
    y = np.concatenate([np.full(int(count), label, dtype=int)
                        for label, count in enumerate(sizes)])
    return StandardScaler().fit_transform(x), y, 30


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
    return x, y, 10


def fit(method, x, seed):
    n = len(x)
    if method == "HDBSCAN":
        return hdbscan.HDBSCAN(
            min_cluster_size=max(10, n // 40), min_samples=5
        ).fit_predict(x)
    if method == "MeanShift":
        neighbors = NearestNeighbors(n_neighbors=min(8, n)).fit(x)
        bandwidth = 2.0 * float(np.median(neighbors.kneighbors(x)[0][:, -1]))
        return MeanShift(bandwidth=max(bandwidth, 0.001), bin_seeding=True).fit_predict(x)
    if method == "AffinityPropagation":
        return AffinityPropagation(
            random_state=seed, damping=0.75, max_iter=500
        ).fit_predict(x)
    raise ValueError(method)


def main():
    raw_path = OUT / "reviewer3_scale_added_methods_raw.csv"
    rows = pd.read_csv(raw_path).to_dict("records") if raw_path.exists() else []
    affinity_path = OUT / "reviewer3_scale_affinity_raw.csv"
    if affinity_path.exists():
        rows.extend(pd.read_csv(affinity_path).to_dict("records"))
    completed = {(row["case"], row["algorithm"], int(row["seed"])) for row in rows}
    for seed in SEEDS:
        for case, generator in (("large_n_imbalanced_largeK_d30", large_n),
                                ("high_d_6000_K20_imbalanced_pca10", high_d)):
            x, y, d_used = generator(seed)
            for method in ("HDBSCAN", "MeanShift", "AffinityPropagation"):
                if (case, method, seed) in completed:
                    continue
                started = time.perf_counter()
                labels = fit(method, x, seed)
                row = {
                    "case": case,
                    "n_original": len(x),
                    "d_original": 6000 if case.startswith("high_d") else 30,
                    "d_used": d_used,
                    "K": len(np.unique(y)),
                    "largest_to_smallest": "90:12" if case.startswith("high_d") else "2000:120",
                    "algorithm": method,
                    "seed": seed,
                    "ari": adjusted_rand_score(y, labels),
                    "seconds": time.perf_counter() - started,
                }
                rows.append(row)
                completed.add((case, method, seed))
                pd.DataFrame(rows).drop_duplicates(
                    ["case", "algorithm", "seed"], keep="last"
                ).to_csv(raw_path, index=False)
                print(case, method, seed, row["ari"], row["seconds"], flush=True)

    raw = pd.read_csv(raw_path)
    raw.groupby(["case", "algorithm"])[["ari", "seconds"]].mean().reset_index().to_csv(
        OUT / "reviewer3_scale_added_methods_summary.csv", index=False
    )
    print(f"Host: {platform.processor()} | {platform.system()} {platform.release()}", flush=True)


if __name__ == "__main__":
    main()
