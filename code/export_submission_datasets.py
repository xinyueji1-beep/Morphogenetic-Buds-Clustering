from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

import external_real_benchmark_mbc as ext
import paper_experiments_mbc as main_exp


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SEEDS = [3, 7, 11, 19, 23]


def safe_name(name: str) -> str:
    keep = []
    for ch in str(name):
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep).strip("_")


def save_matrix(path: Path, x: np.ndarray, y: np.ndarray, **meta) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(x, columns=[f"x{j:03d}" for j in range(x.shape[1])])
    df.insert(0, "target", y)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        **meta,
    }


def export_main_benchmark() -> list[dict]:
    entries = []
    for seed in SEEDS:
        for spec in main_exp.make_datasets(seed):
            raw_file = RAW_DIR / "main_benchmark" / f"{safe_name(spec.name)}_seed{seed}.csv"
            entries.append(save_matrix(raw_file, np.asarray(spec.x), np.asarray(spec.y), dataset=spec.name, seed=seed, type="main_raw"))

            processed = main_exp.preprocess(np.asarray(spec.x))
            proc_file = PROCESSED_DIR / "main_benchmark" / f"{safe_name(spec.name)}_seed{seed}_processed.csv"
            entries.append(save_matrix(proc_file, processed, np.asarray(spec.y), dataset=spec.name, seed=seed, type="main_processed"))
    return entries


def unique_openml_specs() -> list[dict]:
    by_id: dict[int, dict] = {}
    for spec in ext.OPENML_DATASETS:
        by_id.setdefault(spec["id"], {"id": spec["id"], "names": []})
        by_id[spec["id"]]["names"].append(spec["name"])
    return list(by_id.values())


def export_openml_raw() -> list[dict]:
    entries = []
    for spec in unique_openml_specs():
        data_id = spec["id"]
        try:
            bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        except Exception as exc:
            print(f"[warn] skipped raw OpenML dataset {data_id}: {exc}")
            continue
        frame = bunch.frame.copy()
        filename = f"openml_{data_id}_{safe_name(bunch.details.get('name', spec['names'][0]))}.csv"
        path = RAW_DIR / "openml" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        entries.append(
            {
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "openml_id": data_id,
                "openml_name": bunch.details.get("name", ""),
                "used_as": sorted(set(spec["names"])),
                "n_samples": int(frame.shape[0]),
                "n_columns_including_target": int(frame.shape[1]),
                "type": "openml_raw",
            }
        )
    return entries


def export_external_processed() -> list[dict]:
    entries = []
    for spec in ext.OPENML_DATASETS:
        try:
            name, family, x, y = ext.load_openml_dataset(spec)
        except Exception as exc:
            print(f"[warn] skipped processed OpenML dataset {spec['name']}: {exc}")
            continue
        path = PROCESSED_DIR / "external_real_benchmark" / f"{safe_name(name)}_processed.csv"
        entries.append(save_matrix(path, x, y, dataset=name, family=family, openml_id=spec["id"], type="external_processed"))
    return entries


def write_manifest(entries: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "Datasets bundled for the MBC EI submission package. Raw data preserve source tables when available; processed data are the matrices used by the paper scripts after encoding, scaling, PCA and/or sampling.",
        "entries": entries,
    }
    (DATA_DIR / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for e in entries:
        rows.append(
            {
                "file": e.get("file"),
                "type": e.get("type"),
                "dataset": e.get("dataset") or e.get("openml_name") or ",".join(e.get("used_as", [])),
                "source_id": e.get("openml_id") or e.get("source") or "",
                "n_samples": e.get("n_samples"),
                "n_features_or_columns": e.get("n_features") or e.get("n_columns") or e.get("n_columns_including_target"),
            }
        )
    pd.DataFrame(rows).to_csv(DATA_DIR / "dataset_manifest.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    entries: list[dict] = []
    entries.extend(export_main_benchmark())
    entries.extend(export_openml_raw())
    entries.extend(export_external_processed())
    write_manifest(entries)
    print(json.dumps({"data_dir": str(DATA_DIR), "entries": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
