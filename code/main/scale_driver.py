import sys, time, io, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mbc import adjusted_rand_index
from exp_driver import run_baseline, METHODS, load

OUT = PACKAGE_ROOT / "results" / "exp"
os.makedirs(OUT, exist_ok=True)

# scale methods = the seven already in Table 9 (high-d / large-n)
SCALE_METHODS = ["MBC", "KMeans", "GMM", "Agglomerative", "Spectral", "DBSCAN", "CompactRefineOnly"]
# non-convex scale competitors (2-D moons; MeanShift is safe here)
NC_METHODS = ["MBC", "Spectral", "HDBSCAN", "DBSCAN", "MeanShift", "KMeans", "Agglomerative", "GMM", "CompactRefineOnly"]

# pre-build generators
def gen_large_n(seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 8, size=(15, 30))
    sizes = [int(2000/ (1.18**i)) for i in range(15)][::-1]
    sizes = [max(120, s) for s in sizes]
    parts = [c + rng.normal(0, 1, (s, 30)) for c, s in zip(centers, sizes)]
    X = np.vstack(parts); y = np.repeat(np.arange(15), sizes)
    return X, y, 15

def gen_high_d(seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.3, (20, 6000)); centers[:, :20] += 8*np.eye(20)
    sizes = list(range(90, 90-20*7, -7)); sizes=[max(12,s) for s in sizes]
    parts=[c+rng.normal(0,0.8,(s,6000)) for c,s in zip(centers,sizes)]
    X=np.vstack(parts); y=np.repeat(np.arange(20),sizes)
    Xp=PCA(n_components=10, random_state=seed).fit_transform(X)
    return Xp, y, 20

from scale_geometry import gen_scaled_moons

rows=[]
y_cache = None
for seed in [3,7,11]:
    # large-n
    X,y_cache,K=gen_large_n(seed)
    for m in SCALE_METHODS:
        t0=time.time()
        try:
            lab,_=run_baseline(m,X,K,seed); ari=adjusted_rand_index(y_cache,lab)
        except Exception:
            raise
        dt=time.time()-t0
        rows.append(dict(case="large_n", seed=seed, method=m, ari=ari, time_s=dt))
    # high-d (after PCA to 10)
    X,y_cache,K=gen_high_d(seed)
    for m in SCALE_METHODS:
        t0=time.time()
        try:
            lab,_=run_baseline(m,X,K,seed); ari=adjusted_rand_index(y_cache,lab)
        except Exception:
            raise
        dt=time.time()-t0
        rows.append(dict(case="high_d", seed=seed, method=m, ari=ari, time_s=dt))
    # non-convex scale
    for n in [1000,2000,4000,8000]:
        X,y_cache,K=gen_scaled_moons(n,seed)
        for m in NC_METHODS:
            t0=time.time()
            try:
                lab,_=run_baseline(m,X,K,seed); ari=adjusted_rand_index(y_cache,lab)
            except Exception:
                raise
            dt=time.time()-t0
            rows.append(dict(case=f"moons_n{n}", seed=seed, method=m, ari=ari, time_s=dt))
            pd.DataFrame(rows).to_csv(OUT / 'scale_in_progress.csv', index=False)
            print(f'moons n={n} seed={seed} method={m} ari={ari:.6f} seconds={dt:.3f}', flush=True)

pd.DataFrame(rows).to_csv(f"{OUT}/scale.csv", index=False)
print("SCALE EXPERIMENT COMPLETE")
