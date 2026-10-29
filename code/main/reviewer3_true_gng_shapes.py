"""Add the canonical GNG comparator rows to the REAL SIPU shape benchmark.

The shape observations are the fixed public SIPU records (data/raw/sipu/*.arff).
This script never generates analogues: it runs canonical GNG on the fixed
observations and appends the GNG column to the real shape.csv that exp_driver
writes from the same records.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from gng import GrowingNeuralGas  # noqa: E402

OUT = ROOT / "results" / "exp"


def main():
    rows = []
    for name in ("Aggregation", "Compound", "Spiral"):
        data, _meta = arff.loadarff(ROOT / "data" / "raw" / "sipu" / f"{name.lower()}.arff")
        frame = pd.DataFrame(data)
        X = StandardScaler().fit_transform(frame.iloc[:, :-1].to_numpy(float))
        y = pd.factorize(frame.iloc[:, -1])[0]
        K = int(len(np.unique(y)))
        for seed in (3, 7, 11):
            started = time.perf_counter()
            model = GrowingNeuralGas(
                n_epochs=12, max_nodes=max(32, min(120, 12 * K)),
                insertion_interval=max(50, len(X) // 4), max_age=50,
                random_state=seed)
            labels = model.fit_predict(X, K)
            rows.append({"shape": name, "seed": seed, "method": "GNG",
                         "ari": adjusted_rand_score(y, labels),
                         "seconds": time.perf_counter() - started,
                         "realized_nodes": model.n_nodes_,
                         "insertions": model.result_.insertions,
                         "edge_deletions": model.result_.edge_deletions,
                         "node_deletions": model.result_.node_deletions})
            print(rows[-1], flush=True)

    pd.DataFrame(rows).to_csv(OUT / "true_gng_shape_raw.csv", index=False)

    # Append the GNG column to the real shape.csv produced by exp_driver.
    gng_mean = pd.DataFrame(rows).groupby("shape").ari.mean()
    shape = pd.read_csv(OUT / "shape.csv")
    if "shape" not in shape.columns:
        shape = shape.reset_index()
    shape = shape.set_index("shape")
    shape["GNG"] = [float(gng_mean.get(s, np.nan)) for s in shape.index]
    shape.reset_index().to_csv(OUT / "shape.csv", index=False)
    print("true_gng_shape_raw.csv written from real SIPU; GNG column appended to shape.csv")


if __name__ == "__main__":
    main()
