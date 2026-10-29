from pathlib import Path
import os
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUT = PACKAGE_ROOT / "results" / "exp"
prim = pd.read_csv(OUT / "primary_ari.csv")
fam = pd.read_csv(OUT / "family_means.csv", index_col=0)
fried = (OUT / "friedman.txt").read_text(encoding="utf-8")
cfit = (OUT / "complexity_fit.txt").read_text(encoding="utf-8")

METHODS = ["MBC","KMeans","GMM","Agglomerative","Spectral","DBSCAN",
           "HDBSCAN","MeanShift","AffinityPropagation","CompactRefineOnly","GNG"]

# dataset -> family (must match exp_driver)
DATASETS = {
    "two_moons":"Non-convex","noisy_moons":"Non-convex","circles":"Non-convex","noisy_circles":"Non-convex",
    "varied_density":"Density","anisotropic_blobs":"Gaussian",
    "iris":"Low-d","wine":"Low-d","breast_cancer":"Tabular","digits_0_4":"Image",
    "ecoli":"Tabular","glass":"Tabular","vehicle":"Tabular","cmc":"Tabular",
}
FAMS = ["Non-convex","Gaussian","Density","Image","Low-d","Tabular"]

piv = prim.pivot_table(index="dataset", columns="method", values="ari", aggfunc="mean")
# overall per method = mean over 14 datasets
overall = {m: float(piv[m].mean()) for m in METHODS}

def rank_marks(series):
    # series: method->value ; return dict method-> mark (best/second/third/None)
    order = series.sort_values(ascending=False)
    marks = {}
    for i, m in enumerate(order.index):
        marks[m] = {0:"best",1:"second",2:"third"}.get(i, None)
    return marks

print("="*70)
print("TABLE 7 EXTENDED (11 methods): column = Overall, then families")
print("="*70)
cols = ["Overall"] + FAMS
header = "Method " + " ".join(f"{c:>11}" for c in cols)
print(header)
table7 = {}
for m in METHODS:
    rowvals = {}
    rowvals["Overall"] = overall[m]
    for f in FAMS:
        dss = [d for d in DATASETS if DATASETS[d]==f]
        rowvals[f] = float(piv.loc[dss, m].mean())
    table7[m] = rowvals
    print(f"{m:18}" + " ".join(f"{rowvals[c]:11.4f}" for c in cols))

# marks per column
marks_per_col = {}
for c in cols:
    s = pd.Series({m: table7[m][c] for m in METHODS})
    marks_per_col[c] = rank_marks(s)

print("\nMarks (best/second/third) per column:")
for c in cols:
    print(f"  {c}: " + ", ".join(f"{m}={marks_per_col[c][m]}" for m in METHODS if marks_per_col[c][m]))

print("\n" + "="*70)
print("FRIEDMAN")
print("="*70)
print(fried)

print("\n" + "="*70)
print("COMPLEXITY FIT (full Eq 20, realized k, controlled generator)")
print("="*70)
print(cfit)

# per-dataset best method among 11 (excluding MBC's own)
print("\n" + "="*70)
print("PER-DATASET best method (over 11 methods) and MBC ARI/s")
print("="*70)
for ds in DATASETS:
    row = piv.loc[ds]
    best_m = row.idxmax()
    best_v = row.max()
    mbc_v = row["MBC"]
    mbc_s = float(prim[(prim.dataset==ds)&(prim.method=="MBC")].time_s.mean())
    print(f"{ds:16} MBC={mbc_v:7.4f} s={mbc_s:6.2f} | best={best_m} ({best_v:.4f})")

# six-structure extension (Aggregation/Compound/Spiral) from shape.csv
shape_path = OUT / "shape.csv"
if os.path.exists(shape_path):
    sh = pd.read_csv(shape_path, index_col=0)
    print("\n" + "="*70)
    print("SHAPE BENCHMARK (Aggregation/Compound/Spiral) mean ARI over 3 seeds")
    print("="*70)
    print(sh.round(4).to_string())
    # marks per shape
    print("\nMarks per shape (best/second/third):")
    for s in sh.index:
        ser = sh.loc[s].sort_values(ascending=False)
        marks = {0:"best",1:"second",2:"third"}
        print(f"  {s}: " + ", ".join(f"{m}={marks.get(i,'')}" for i,m in enumerate(ser.index)))
else:
    print("\nshape.csv not found")

# scale benchmark
scale_path = PACKAGE_ROOT / "results" / "paper_results" / "reviewer3_scale_all_methods_summary.csv"
if os.path.exists(scale_path):
    sc = pd.read_csv(scale_path)
    print("\n" + "="*70)
    print("SCALE BENCHMARK (large_n, high_d, moons_n*)")
    print("="*70)
    for case in sc.case.unique():
        sub = sc[sc.case==case]
        print(f"\n--- {case} ---")
        runtime_col = "time_s" if "time_s" in sub.columns else "seconds"
        print(sub.pivot_table(index="method", values=["ari", runtime_col], aggfunc="mean").round(4).to_string())
else:
    print("\nscale.csv not found")
