"""Offline consistency audit for the bundled paper result ledgers.

This check reads only files shipped in this reproduction package. It does not
contact OpenML or any other network service and it does not regenerate the
large experiments.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "paper_results"
DATA = ROOT / "data"
parser = argparse.ArgumentParser(description="Audit the bundled offline ledgers and inputs.")
parser.add_argument("--r3", action="store_true", help="accepted for compatibility with the top-level runner")
parser.parse_args()
checks: list[dict] = []


def check(name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def approx(got: float, expected: float, tol: float = 5e-3) -> bool:
    return abs(float(got) - float(expected)) <= tol

# Main 14-dataset MBC means.
base_targets = {
    "two_moons": 1.0000, "noisy_moons": 0.6799, "circles": 1.0000,
    "noisy_circles": 1.0000, "varied_density": 0.0151,
    "anisotropic_blobs": 0.8895, "iris": 0.6161, "wine": 0.8975,
    "breast_cancer": 0.6673, "digits_0_4": 0.7335, "ecoli": 0.5279,
    "glass": 0.1604, "vehicle": 0.0758, "cmc": 0.0230,
}
base_path = RES / "paper_benchmark_raw.csv"
if base_path.exists():
    base = pd.read_csv(base_path)
    mbc = base[base["algorithm"].eq("MBC")]
    check("main MBC rows", len(mbc) == 70, f"rows={len(mbc)}")
    check("main MBC overall", approx(mbc["ari"].mean(), 0.5919), f"got={mbc['ari'].mean():.4f}, expected=0.5919")
    for dataset, expected in base_targets.items():
        vals = mbc.loc[mbc["dataset"].eq(dataset), "ari"]
        ok = len(vals) > 0 and approx(vals.mean(), expected)
        got = "missing" if len(vals) == 0 else f"{vals.mean():.4f}"
        check(f"main MBC {dataset}", ok, f"got={got}, expected={expected:.4f}")
else:
    check("main benchmark file", False, f"missing {base_path}")

# Compact scale ledger used by the revised scale table.
scale_summary = RES / "reviewer3_scale_benchmark_summary.csv"
if scale_summary.exists():
    scale = pd.read_csv(scale_summary)
    for case, expected in {
        "high_d_6000_K20_imbalanced_pca10": 0.9448,
        "large_n_imbalanced_largeK_d30": 0.8934,
    }.items():
        vals = scale.loc[(scale["case"].eq(case)) & (scale["algorithm"].eq("MBC")), "ari"]
        ok = len(vals) > 0 and approx(vals.iloc[0], expected)
        got = "missing" if len(vals) == 0 else f"{vals.iloc[0]:.4f}"
        check(f"scale summary MBC {case}", ok, f"got={got}, expected={expected:.4f}")
else:
    check("scale summary file", False, f"missing {scale_summary}")

scale_raw_path = RES / "reviewer3_scale_all_methods_raw.csv"
if scale_raw_path.exists():
    scale_raw = pd.read_csv(scale_raw_path)
    expected_methods = {"MBC", "KMeans", "GMM", "Agglomerative", "Spectral", "DBSCAN", "HDBSCAN", "MeanShift", "AffinityPropagation", "CompactRefineOnly", "GNG"}
    check("scale raw shape", len(scale_raw) == 66, f"rows={len(scale_raw)}")
    check("scale raw methods", set(scale_raw["method"]) == expected_methods, f"methods={len(set(scale_raw['method']))}")
    check("scale raw seeds", set(scale_raw["seed"]) == {3, 7, 11}, f"seeds={sorted(set(scale_raw['seed']))}")
    for case, expected in {
        "high_d_6000_K20_imbalanced_pca10": 0.9448,
        "large_n_imbalanced_largeK_d30": 0.8934,
    }.items():
        vals = scale_raw.loc[(scale_raw["case"].eq(case)) & (scale_raw["method"].eq("MBC")), "ari"]
        check(f"scale raw MBC {case}", len(vals) == 3 and approx(vals.mean(), expected), f"mean={vals.mean():.4f}, n={len(vals)}")
else:
    check("scale raw file", False, f"missing {scale_raw_path}")

# Corrected OpenML Lung-5 data and ledger.
raw_lung = DATA / "raw" / "openml" / "openml_45093_Lung.csv"
proc_lung = DATA / "processed" / "external_real_benchmark" / "Lung_5class_processed.csv"
if raw_lung.exists() and proc_lung.exists():
    raw_df = pd.read_csv(raw_lung)
    proc_df = pd.read_csv(proc_lung)
    raw_counts = sorted(raw_df.iloc[:, -1].value_counts().tolist())
    proc_counts = sorted(proc_df["target"].value_counts().tolist())
    check("Lung raw dimensions", raw_df.shape == (203, 12601), f"shape={raw_df.shape}")
    check("Lung raw labels", raw_counts == [6, 17, 20, 21, 139], f"counts={raw_counts}")
    check("Lung processed dimensions", proc_df.shape == (203, 13), f"shape={proc_df.shape}")
    check("Lung processed labels", proc_counts == [6, 17, 20, 21, 139], f"counts={proc_counts}")
else:
    check("Lung input files", False, "one or more Lung files are missing")

external = RES / "external_real_benchmark_raw.csv"
if external.exists():
    ext = pd.read_csv(external)
    lung = ext[ext["dataset"].eq("Lung_5class")].groupby("algorithm")["ari"].mean()
    expected = {"DBSCAN": 0.4092, "Spectral": 0.2908, "GMM": 0.2906, "Agglomerative": 0.2232, "MBC": 0.1492}
    for algorithm, target in expected.items():
        ok = algorithm in lung and approx(lung[algorithm], target, 1e-3)
        got = "missing" if algorithm not in lung else f"{lung[algorithm]:.4f}"
        check(f"Lung ledger {algorithm}", ok, f"got={got}, expected={target:.4f}")
else:
    check("external result file", False, f"missing {external}")

# CSV manifest must agree with the corrected files.
manifest = DATA / "dataset_manifest.csv"
if manifest.exists():
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
    lung_rows = [r for r in rows if "45093_Lung" in r["file"] or r["dataset"] == "Lung_5class"]
    ok = len(lung_rows) == 2 and all(r["n_samples"] == "203" for r in lung_rows)
    check("dataset manifest Lung rows", ok, f"n_samples={[r['n_samples'] for r in lung_rows]}")
else:
    check("dataset manifest", False, f"missing {manifest}")

# Comparator parameter export, generated from the bundled source call sites.
parameter_path = ROOT / "results" / "exp" / "baseline_parameters.json"
if parameter_path.exists():
    params = json.loads(parameter_path.read_text(encoding="utf-8"))
    records = params.get("data_derived_density_values", [])
    suite_counts = {}
    for record in records:
        suite_counts[record.get("suite")] = suite_counts.get(record.get("suite"), 0) + 1
    check("baseline parameter export schema", params.get("schema_version") == 1, f"schema={params.get('schema_version')}")
    check("baseline parameter methods", len(params.get("method_call_rules", {})) == 11, f"methods={len(params.get('method_call_rules', {}))}")
    check("baseline density record count", len(records) == 97, f"records={len(records)}")
    check("baseline density suite counts", suite_counts == {
        "primary": 70, "public_sipu_shapes": 9,
        "compact_scale_all_methods": 6, "two_moon_scale": 12,
    }, f"counts={suite_counts}")
    valid_density = all(
        row.get("dbscan", {}).get("eps", 0) > 0
        and row.get("mean_shift", {}).get("bandwidth", 0) > 0
        and row.get("dbscan", {}).get("min_samples", 0) >= 4
        for row in records
    )
    check("baseline density values finite and positive", valid_density, "all exported epsilon and bandwidth values are positive")
else:
    check("baseline parameter export", False, f"missing {parameter_path}")

# Hash manifest identifies the exact bundled data bytes, without asserting that
# the locally computed digests are upstream-provider checksums.
hash_path = DATA / "input_hashes.json"
if hash_path.exists():
    hash_manifest = json.loads(hash_path.read_text(encoding="utf-8"))
    hash_rows = hash_manifest.get("files", [])
    seen_paths = [row.get("file") for row in hash_rows]
    hash_values_ok = all(
        isinstance(row.get("bytes"), int) and row["bytes"] >= 0
        and len(row.get("sha256", "")) == 64
        and len(row.get("md5", "")) == 32
        for row in hash_rows
    )
    required_inputs = [
        "data/raw/openml/openml_45093_Lung.csv",
        "data/processed/external_real_benchmark/Lung_5class_processed.csv",
        "data/raw/sipu/aggregation.arff",
        "data/raw/sipu/compound.arff",
        "data/raw/sipu/spiral.arff",
    ]
    check("input hash manifest schema", hash_manifest.get("schema_version") == 1, f"schema={hash_manifest.get('schema_version')}")
    check("input hash manifest entries", len(seen_paths) > 0 and len(set(seen_paths)) == len(seen_paths), f"files={len(seen_paths)}")
    check("input hash manifest digests", hash_values_ok, "all bundled file digests have expected lengths")
    check("input hash required records", all(path in set(seen_paths) for path in required_inputs), "Lung raw/processed and public SIPU files present")
    hashes_match = True
    for row in hash_rows:
        path = ROOT / Path(row["file"])
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            hashes_match = False
            break
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if sha256 != row["sha256"] or md5 != row["md5"]:
            hashes_match = False
            break
    check("input hash manifest matches files", hashes_match, f"verified={len(hash_rows) if hashes_match else 'mismatch'}")
else:
    check("input hash manifest", False, f"missing {hash_path}")

summary = {"pass": sum(c["ok"] for c in checks), "fail": sum(not c["ok"] for c in checks), "checks": checks}
out = ROOT / "results" / "audit" / "audit_result.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"SUMMARY: PASS={summary['pass']} FAIL={summary['fail']}")
raise SystemExit(0 if summary["fail"] == 0 else 1)

