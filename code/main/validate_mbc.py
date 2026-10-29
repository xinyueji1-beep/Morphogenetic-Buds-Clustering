import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mbc import MorphogeneticBudsClustering, adjusted_rand_index

def load(path):
    df = pd.read_csv(path)
    y = df["target"].to_numpy()
    X = df.drop(columns=["target"]).to_numpy(dtype=float)
    return X, y

def run_mbc(X, K, seed):
    m = MorphogeneticBudsClustering(
        initial_buds=max(12, K*6), steps=140, growth_rate=0.15, inhibition=0.045,
        apoptosis_mass=max(2.5, len(X)*0.0055), birth_quantile=0.88,
        organ_merge_radius=3.6, valley_ratio=0.18, tangent_alignment=0.35,
        max_bridge_void=0.46, target_clusters=K, compact_refinement_steps=18,
        compact_refinement_restarts=10, compact_refinement_dim_threshold=3,
        random_state=seed)
    lab = m.fit_predict(X)
    return lab, m

base = PACKAGE_ROOT / "data" / "processed" / "main_benchmark"
for name, K in [("two_moons_seed3", 2), ("wine_seed3", 3)]:
    X, y = load(base / f"{name}_processed.csv")
    t0 = time.time()
    lab, model = run_mbc(X, K, 3)
    dt = time.time() - t0
    ari = adjusted_rand_index(y, lab)
    nk = len(model.centers_) if model.centers_ is not None else -1
    print(f"{name}: ARI={ari:.4f}  buds(final)={nk}  time={dt:.2f}s  n={len(X)} d={X.shape[1]}")
