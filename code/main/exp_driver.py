import sys, time, json, io
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import (KMeans, AgglomerativeClustering,
                             SpectralClustering, DBSCAN, MeanShift, AffinityPropagation)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import hdbscan
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mbc import MorphogeneticBudsClustering, adjusted_rand_index, kmeanspp_centers, refine_centers
from gng import GrowingNeuralGas

CSV = PACKAGE_ROOT / "data" / "processed" / "main_benchmark"
OUT = PACKAGE_ROOT / "results" / "exp"
import os; os.makedirs(OUT, exist_ok=True)

SEEDS = [3, 7, 11, 19, 23]
# dataset -> (K, family)
DATASETS = {
    "two_moons": (2, "Non-convex"), "noisy_moons": (2, "Non-convex"),
    "circles": (2, "Non-convex"), "noisy_circles": (2, "Non-convex"),
    "varied_density": (3, "Density"), "anisotropic_blobs": (4, "Gaussian"),
    "iris": (3, "Low-d"), "wine": (3, "Low-d"),
    "breast_cancer": (2, "Tabular"), "digits_0_4": (5, "Image"),
    "ecoli": (8, "Tabular"), "glass": (6, "Tabular"),
    "vehicle": (4, "Tabular"), "cmc": (3, "Tabular"),
}
METHODS = ["MBC", "KMeans", "GMM", "Agglomerative", "Spectral", "DBSCAN",
           "HDBSCAN", "MeanShift", "AffinityPropagation", "CompactRefineOnly", "GNG"]

def load(name, seed):
    df = pd.read_csv(CSV / f"{name}_seed{seed}_processed.csv")
    y = df["target"].to_numpy()
    X = df.drop(columns=["target"]).to_numpy(dtype=float)
    return X, y

def make_mbc(X, K, seed):
    return MorphogeneticBudsClustering(
        initial_buds=max(12, K*6), steps=140, growth_rate=0.15, inhibition=0.045,
        apoptosis_mass=max(2.5, len(X)*0.0055), birth_quantile=0.88,
        organ_merge_radius=3.6, valley_ratio=0.18, tangent_alignment=0.35,
        max_bridge_void=0.46, target_clusters=K, compact_refinement_steps=18,
        compact_refinement_restarts=10, compact_refinement_dim_threshold=3,
        random_state=seed)

def run_mbc(X, K, seed):
    m = make_mbc(X, K, seed)
    lab = m.fit_predict(X)
    k = len(m.centers_) if m.centers_ is not None else -1
    return lab, k

def growing_neural_gas(X, K, seed):
    """Canonical GNG with edge-based growth and supplied-K node grouping."""
    model = GrowingNeuralGas(
        n_epochs=12,
        max_nodes=max(32, min(120, 12 * K)),
        insertion_interval=max(50, min(100, len(X) // 4)),
        max_age=50,
        winner_learning_rate=0.20,
        neighbor_learning_rate=0.006,
        insertion_error_decay=0.5,
        global_error_decay=0.995,
        random_state=seed,
    )
    labels = model.fit_predict(X, K)
    return labels, model.n_nodes_

def run_baseline(name, X, K, seed):
    n, d = X.shape
    if name == "MBC":
        lab, k = run_mbc(X, K, seed); return lab, k
    if name == "KMeans":
        return KMeans(K, n_init=30, random_state=seed).fit_predict(X), K
    if name == "GMM":
        return GaussianMixture(K, covariance_type="full", random_state=seed).fit_predict(X), K
    if name == "Agglomerative":
        return AgglomerativeClustering(K, linkage="ward").fit_predict(X), K
    if name == "Spectral":
        return SpectralClustering(K, affinity="nearest_neighbors",
                                  n_neighbors=min(12, n-1), random_state=seed).fit_predict(X), K
    if name == "DBSCAN":
        from sklearn.neighbors import NearestNeighbors
        k = min(6, n)
        distances = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)[0][:, -1]
        eps = float(np.quantile(distances, 0.72))
        return DBSCAN(eps=eps, min_samples=max(4, d+1)).fit_predict(X), -1
    if name == "HDBSCAN":
        return hdbscan.HDBSCAN(min_cluster_size=max(10, n//40), min_samples=5).fit_predict(X), -1
    if name == "MeanShift":
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(8, n)).fit(X)
        bandwidth = float(np.median(nn.kneighbors(X)[0][:, -1])) * 2.0
        return MeanShift(bandwidth=max(bandwidth, 0.001), bin_seeding=True).fit_predict(X), -1
    if name == "AffinityPropagation":
        return AffinityPropagation(random_state=seed, damping=0.75, max_iter=500).fit_predict(X), -1
    if name == "CompactRefineOnly":
        rng = np.random.default_rng(seed); best=None; bi=np.inf
        for _ in range(10):
            c = kmeanspp_centers(X, K, rng)
            lab, _, j = refine_centers(X, c, 18)
            if j < bi: bi=j; best=lab
        return best, K
    if name == "GNG":
        return growing_neural_gas(X, K, seed)
    raise ValueError(name)

def _main():
    # ---------- 1. Primary suite ----------
    rows = []
    prediction_ledger = {}
    for ds, (K, fam) in DATASETS.items():
        for seed in SEEDS:
            X, y = load(ds, seed)
            key = f'{ds}/seed{seed}'
            prediction_ledger[key] = dict(dataset=ds, seed=seed, target=y.tolist(), predictions={})
            for meth in METHODS:
                t0 = time.time()
                lab, kf = run_baseline(meth, X, K, seed)
                dt = time.time()-t0
                ari = adjusted_rand_index(y, lab) if lab is not None else float("nan")
                prediction_ledger[key]['predictions'][meth] = np.asarray(lab).tolist()
                rows.append(dict(dataset=ds, family=fam, K=K, seed=seed, method=meth,
                                 ari=ari, k_found=kf, time_s=dt))
            pd.DataFrame(rows).to_csv(OUT / 'primary_in_progress.csv', index=False)
            print(f'primary {ds} seed={seed} completed {len(rows)} rows', flush=True)
    prim = pd.DataFrame(rows)
    prim.to_csv(f"{OUT}/primary_ari.csv", index=False)
    (OUT/'primary_predictions.json').write_text(json.dumps(prediction_ledger), encoding='utf-8')
    print("primary rows:", len(prim))

    # family means (Wine -> Low-d)
    piv = prim.pivot_table(index="dataset", columns="method", values="ari", aggfunc="mean")
    fam_mean = {}
    for meth in METHODS:
        per = {}
        for fam in ["Non-convex", "Gaussian", "Density", "Image", "Low-d", "Tabular"]:
            dss = [d for d in DATASETS if DATASETS[d][1]==fam]
            per[fam] = float(piv.loc[dss, meth].mean())
        fam_mean[meth] = per
    pd.DataFrame(fam_mean).T.to_csv(f"{OUT}/family_means.csv")
    print("family means written")

    # ---------- 2. Friedman (raw 14) + block-collapsed (12) ----------
    from scipy.stats import friedmanchisquare
    def friedman_p(mat):  # mat: methods x datasets
        stat, p = friedmanchisquare(*[mat[i] for i in range(mat.shape[0])])
        return stat, p
    # raw
    ds_order = list(DATASETS.keys())
    mat_raw = np.array([piv.loc[ds_order, m].to_numpy() for m in METHODS])
    s_raw, p_raw = friedman_p(mat_raw)
    # block-collapsed: Two/Noisy Moons -> 1; Circles/Noisy Circles -> 1 (12 blocks)
    blocks = [["two_moons","noisy_moons"],["circles","noisy_circles"],["varied_density"],
              ["anisotropic_blobs"],["iris"],["wine"],["breast_cancer"],["digits_0_4"],
              ["ecoli"],["glass"],["vehicle"],["cmc"]]
    mat_blk = np.array([[piv.loc[b, m].mean() for b in blocks] for m in METHODS])
    s_blk, p_blk = friedman_p(mat_blk)
    # Iman-Davenport
    k_m, n_b = len(METHODS), len(blocks)
    F_id = ((n_b-1)*s_blk)/( n_b*(k_m-1)-s_blk )
    from scipy.stats import f as fdist
    p_id = fdist.sf(F_id, k_m-1, (k_m-1)*(n_b-1))
    with io.open(f"{OUT}/friedman.txt","w",encoding="utf-8") as f:
        f.write(f"RAW 14 datasets x {len(METHODS)} methods: chi2={s_raw:.3f} p={p_raw:.4f}\n")
        f.write(f"BLOCK 12 datasets x {len(METHODS)} methods: chi2={s_blk:.3f} p={p_blk:.4f}\n")
        f.write(f"Iman-Davenport F={F_id:.3f} p={p_id:.4f}\n")
    print("friedman done")

    # ---------- 3. Controlled complexity (realized k + timing, full Eq 20) ----------
    def gen_mixture(n_target, K, d, seed, base_n=400):
        rng = np.random.default_rng(seed)
        # fixed geometry: K centers on circle radius 3*sqrt(K)
        ang = np.linspace(0, 2*np.pi, K, endpoint=False)
        R = 3.0*np.sqrt(K)
        centers = np.column_stack([R*np.cos(ang), R*np.sin(ang)])
        if d > 2:
            centers = np.hstack([centers, np.zeros((K, d-2))])
        pts = []
        per = max(20, n_target//K)
        for c in centers:
            pts.append(c + rng.normal(0, 0.3, size=(per, d)))
        Xb = np.vstack(pts)
        # replicate with jitter to reach n_target keeping geometry
        while len(Xb) < n_target:
            add = Xb[rng.choice(len(Xb), min(n_target-len(Xb), len(Xb)))]
            Xb = np.vstack([Xb, add + rng.normal(0, 0.05, add.shape)])
        return Xb[:n_target]
    comp_rows = []
    T, S, Rref = 140, 10, 18
    for axis, vals in [("n",[500,1000,2000,4000,8000,16000]),
                       ("K",[2,4,8,16,32,64]),
                       ("d",[2,10,30,60])]:
        for v in vals:
            for seed in [3,7,11]:
                if axis=="n": X=gen_mixture(v,8,30,seed)
                elif axis=="K": X=gen_mixture(2000,v,2,seed)
                else: X=gen_mixture(2000,8,v,seed)
                Kused = v if axis=="K" else 8
                t0=time.time()
                model = make_mbc(X, Kused, seed)
                lab = model.fit_predict(X)
                kf = len(model.centers_)
                dt=time.time()-t0
                n,d=X.shape
                entering_counts = [min(model.initial_buds,n)] + [max(1,len(c)) for c in model.history_[:-1]]
                comp_rows.append(dict(axis=axis, value=v, seed=seed, n=n, d=d, K=Kused,
                                      realized_k=kf, time_s=dt,
                                      mean_k=np.mean(entering_counts),
                                      mean_k_squared=np.mean(np.square(entering_counts)),
                                      max_k=max(entering_counts),
                                      refinement_active=int(d>=3)))
                print(f'complexity {axis}={v} seed={seed} k={kf} seconds={dt:.3f}',flush=True)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(f"{OUT}/complexity.csv", index=False)
    # fit full Eq 20: W = c1 T n k d + c2 T k^2 d + c3 k^2 n d + c4 S Rref n K d
    A = np.column_stack([
        T*comp.n*comp.mean_k*comp.d,
        T*comp.mean_k_squared*comp.d,
        comp.realized_k**2*comp.n*comp.d,
        comp.refinement_active*S*Rref*comp.n*comp.K*comp.d,
    ]).astype(float)
    b, *_ = np.linalg.lstsq(A, comp.time_s.values, rcond=None)
    pred = A@b
    ss_res = ((comp.time_s-pred)**2).sum(); ss_tot=((comp.time_s-comp.time_s.mean())**2).sum()
    r2 = 1-ss_res/ss_tot
    with io.open(f"{OUT}/complexity_fit.txt","w",encoding="utf-8") as f:
        f.write(f"Full Eq 20 fit (4 terms, wall-clock): coeffs c1..c4 = {b}\n")
        f.write(f"R^2 = {r2:.4f}\n")
        f.write("Note: realized_k varies across points; timing is not a clean wall-clock law.\n")
    print("complexity done")

    # ---------- 4. Labelled shape benchmark: REAL SIPU public records ----------
    # Fixed observations from the SIPU repository (data/raw/sipu/*.arff). Seeds
    # affect only the algorithms; the observations are never generated. This
    # replaces the earlier generator-defined analogues so the paper reports the
    # genuine public benchmark, not a synthetic look-alike.
    from scipy.io import arff as _sipu_arff
    from sklearn.preprocessing import StandardScaler as _SipuScaler
    shape_rows = []
    for nm in ("Aggregation", "Compound", "Spiral"):
        data, _meta = _sipu_arff.loadarff(PACKAGE_ROOT / "data" / "raw" / "sipu" / f"{nm.lower()}.arff")
        frame = pd.DataFrame(data)
        X = _SipuScaler().fit_transform(frame.iloc[:, :-1].to_numpy(float))
        y = pd.factorize(frame.iloc[:, -1])[0]
        K = int(len(np.unique(y)))
        for seed in (3, 7, 11):
            for meth in METHODS:
                lab, _kf = run_baseline(meth, X, K, seed)
                ari = adjusted_rand_index(y, lab)
                shape_rows.append(dict(shape=nm, seed=seed, method=meth, ari=ari))
    shape = pd.DataFrame(shape_rows)
    shape.to_csv(OUT / "shape_raw.csv", index=False)
    shape.pivot_table(index="shape", columns="method", values="ari", aggfunc="mean").to_csv(f"{OUT}/shape.csv")
    print("shape done (real SIPU public records)")
    print("ALL EXPERIMENTS COMPLETE")

if __name__ == "__main__":
    _main()
