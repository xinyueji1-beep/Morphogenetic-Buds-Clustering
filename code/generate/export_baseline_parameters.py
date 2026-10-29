"""Export the comparator call rules and data-derived density scales.

This is a metadata export only: it does not fit clustering models or update
result ledgers. The estimator call sites are in ``code/main/exp_driver.py``;
the special GNG budgets are in the reviewer-specific drivers listed below.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    MeanShift,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import hdbscan

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code" / "main"
sys.path.insert(0, str(CODE))
from exp_driver import DATASETS, SEEDS, load  # noqa: E402
from reviewer3_scale_all_methods import high_d, large_n  # noqa: E402
from scale_geometry import gen_scaled_moons  # noqa: E402


def estimator_defaults() -> dict:
    """Return defaults from the installed, pinned estimator implementations."""
    objects = {
        "KMeans": KMeans(),
        "GMM": GaussianMixture(),
        "Agglomerative": AgglomerativeClustering(),
        "Spectral": SpectralClustering(),
        "DBSCAN": DBSCAN(),
        "HDBSCAN": hdbscan.HDBSCAN(),
        "MeanShift": MeanShift(),
        "AffinityPropagation": AffinityPropagation(),
    }
    return {name: obj.get_params(deep=False) for name, obj in objects.items()}


def json_safe(value):
    """Preserve estimator infinity defaults as readable strings in strict JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)) and not math.isfinite(float(value)):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def data_derived_values(suite: str, case: str, seed: int, x: np.ndarray, k: int) -> dict:
    """Compute the same DBSCAN epsilon and MeanShift bandwidth as exp_driver."""
    n, d = x.shape
    nn6 = NearestNeighbors(n_neighbors=min(6, n)).fit(x)
    sixth_neighbor_distances = nn6.kneighbors(x)[0][:, -1]
    epsilon = float(np.quantile(sixth_neighbor_distances, 0.72))
    nn8 = NearestNeighbors(n_neighbors=min(8, n)).fit(x)
    eighth_neighbor_distances = nn8.kneighbors(x)[0][:, -1]
    bandwidth = max(float(np.median(eighth_neighbor_distances)) * 2.0, 0.001)
    return {
        "suite": suite,
        "case": case,
        "seed": int(seed),
        "n": int(n),
        "d": int(d),
        "K": int(k),
        "dbscan": {
            "eps": epsilon,
            "min_samples": max(4, int(d) + 1),
            "rule": "quantile 0.72 of the sixth-neighbor distances; kneighbors(X) includes the query point",
        },
        "mean_shift": {
            "bandwidth": bandwidth,
            "rule": "max(2 * median eighth-neighbor distance from kneighbors(X), 0.001); query point included",
        },
    }


def collect_density_values() -> list[dict]:
    rows: list[dict] = []

    # Primary empirical and generated benchmark matrices; these are the exact
    # preprocessed files consumed by the comparator runner.
    for dataset, (k, _family) in DATASETS.items():
        for seed in SEEDS:
            x, _y = load(dataset, seed)
            rows.append(data_derived_values("primary", dataset, seed, x, k))

    # Fixed SIPU observations: scale transformation is identical to exp_driver.
    for name in ("Aggregation", "Compound", "Spiral"):
        raw, _meta = arff.loadarff(ROOT / "data" / "raw" / "sipu" / f"{name.lower()}.arff")
        frame = pd.DataFrame(raw)
        x = StandardScaler().fit_transform(frame.iloc[:, :-1].to_numpy(float))
        k = int(frame.iloc[:, -1].nunique())
        for seed in (3, 7, 11):
            rows.append(data_derived_values("public_sipu_shapes", name, seed, x, k))

    # The two compact eleven-method stress cases use the generators called by
    # reviewer3_scale_all_methods.py; no clustering fit is performed here.
    for seed in (3, 7, 11):
        for case, generator in (("large_n_imbalanced_largeK_d30", large_n),
                                ("high_d_6000_K20_imbalanced_pca10", high_d)):
            x, _y, k, _d_used = generator(seed)
            rows.append(data_derived_values("compact_scale_all_methods", case, seed, x, k))

    # Shared true two-moon scale panel (the data are generated and standardized
    # by the same helper as the run driver).
    for seed in (3, 7, 11):
        for n in (1000, 2000, 4000, 8000):
            x, _y, k = gen_scaled_moons(n, seed)
            rows.append(data_derived_values("two_moon_scale", f"n={n}", seed, x, k))
    return rows


def main() -> None:
    primary_mbc = {
        "initial_buds": "max(12, 6*K)", "steps": 140, "growth_rate": 0.15,
        "inhibition": 0.045, "apoptosis_mass": "max(2.5, 0.0055*n)",
        "birth_quantile": 0.88, "organ_merge_radius": 3.6,
        "valley_ratio": 0.18, "tangent_alignment": 0.35,
        "max_bridge_void": 0.46, "target_clusters": "supplied K",
        "compact_refinement_steps": 18, "compact_refinement_restarts": 10,
        "compact_refinement_dim_threshold": 3, "random_state": "seed",
    }
    primary_gng = {
        "n_epochs": 12, "max_nodes": "max(32, min(120, 12*K))",
        "insertion_interval": "max(50, min(100, floor(n/4)))",
        "max_age": 50, "winner_learning_rate": 0.20,
        "neighbor_learning_rate": 0.006, "insertion_error_decay": 0.5,
        "global_error_decay": 0.995, "random_state": "seed",
        "post_grouping": "KMeans(n_clusters=min(K, node_count), n_init=20, random_state=seed)",
    }
    method_calls = {
        "MBC": {"source": "code/main/exp_driver.py:make_mbc", "arguments": primary_mbc},
        "KMeans": {"source": "code/main/exp_driver.py:run_baseline", "arguments": {
            "n_clusters": "K", "n_init": 30, "random_state": "seed",
            "other_arguments": "installed scikit-learn defaults",
        }},
        "GMM": {"source": "code/main/exp_driver.py:run_baseline", "arguments": {
            "n_components": "K", "covariance_type": "full", "random_state": "seed",
            "other_arguments": "installed scikit-learn defaults",
        }},
        "Agglomerative": {"source": "code/main/exp_driver.py:run_baseline", "arguments": {
            "n_clusters": "K", "linkage": "ward", "other_arguments": "installed scikit-learn defaults",
        }},
        "Spectral": {"source": "code/main/exp_driver.py:run_baseline", "arguments": {
            "n_clusters": "K", "affinity": "nearest_neighbors", "n_neighbors": "min(12, n-1)",
            "random_state": "seed", "other_arguments": "installed scikit-learn defaults",
        }},
        "DBSCAN": {"source": "code/main/exp_driver.py:run_baseline", "arguments": {
            "eps": "per-input value in data_derived_density_values", "min_samples": "max(4, d+1)",
            "other_arguments": "installed scikit-learn defaults",
        }},
        "HDBSCAN": {"source": "code/main/exp_driver.py:run_baseline", "arguments": {
            "min_cluster_size": "max(10, floor(n/40))", "min_samples": 5,
            "other_arguments": "installed hdbscan defaults",
        }},
        "MeanShift": {"source": "code/main/exp_driver.py:run_baseline", "arguments": {
            "bandwidth": "per-input value in data_derived_density_values", "bin_seeding": True,
            "other_arguments": "installed scikit-learn defaults",
        }},
        "AffinityPropagation": {"source": "code/main/exp_driver.py:run_baseline", "arguments": {
            "affinity": "euclidean", "preference": "None (library resolves median similarity)",
            "damping": 0.75, "max_iter": 500, "random_state": "seed",
            "other_arguments": "installed scikit-learn defaults, including convergence_iter=15",
        }},
        "CompactRefineOnly": {"source": "code/main/exp_driver.py:run_baseline", "arguments": {
            "candidates": 10, "initialization": "local k-means++ helper, numpy default_rng(seed)",
            "finite_lloyd_updates": 18, "selection": "lowest inertia",
        }},
        "GNG": {"source": "code/main/exp_driver.py:growing_neural_gas", "arguments": primary_gng},
    }

    gng_protocols = {
        "public_sipu_shapes": {"source": "code/main/reviewer3_true_gng_shapes.py", "arguments": primary_gng},
        "analogous_generated_shapes": {
            "source": "code/main/reviewer3_true_gng.py and code/main/reviewer2_related_methods.py",
            "arguments": "12 epochs; max_nodes=max(32,min(120,12*K)); insertion_interval=max(50,floor(n/4)); max_age=50; remaining GrowingNeuralGas defaults; supplied-K node grouping",
        },
        "compact_scale_all_methods": {
            "source": "code/main/reviewer3_true_gng.py", "arguments": {
                "n_epochs": 3, "max_nodes": "max(32, min(120, 12*K))",
                "insertion_interval": "max(50, floor(n/4))", "max_age": 50,
                "other_arguments": "GrowingNeuralGas defaults; supplied-K node grouping",
            },
        },
        "two_moon_scale": {
            "source": "code/main/reviewer3_true_gng.py", "arguments": {
                "n_epochs": 3, "max_nodes": 48, "insertion_interval": "max(50, floor(n/4))",
                "max_age": 50, "other_arguments": "GrowingNeuralGas defaults; supplied-K node grouping",
            },
        },
    }

    versions = {}
    for distribution in ("numpy", "pandas", "scipy", "scikit-learn", "hdbscan"):
        versions[distribution] = importlib.metadata.version(distribution)
    payload = {
        "schema_version": 1,
        "purpose": "Comparator settings and derived density scales; this export does not run clustering fits or modify result ledgers.",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": versions},
        "source_of_callsite_rules": "code/main/exp_driver.py and the suite-specific drivers listed for each protocol",
        "library_resolved_defaults": estimator_defaults(),
        "method_call_rules": method_calls,
        "gng_suite_protocols": gng_protocols,
        "generator_source_files": [
            "code/main/exp_driver.py", "code/main/reviewer3_scale_all_methods.py",
            "code/main/scale_geometry.py", "code/main/reviewer3_true_gng.py",
            "code/main/reviewer3_true_gng_shapes.py", "code/main/reviewer3_complexity_analysis.py",
        ],
        "data_derived_density_values": collect_density_values(),
    }
    target = ROOT / "results" / "exp" / "baseline_parameters.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(payload)
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {target.relative_to(ROOT)} with {len(payload['data_derived_density_values'])} input/seed records")


if __name__ == "__main__":
    main()
