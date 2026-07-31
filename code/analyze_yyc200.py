from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def find_yyc200_paths(root: Path) -> tuple[Path, Path]:
    candidates = [
        root,
        root.parent / "data" / "yyc200_hyperspectral",
        root.parent,
    ]
    for base in candidates:
        data_path = base / "yyc200"
        hdr_path = base / "yyc200.hdr"
        if data_path.exists() and hdr_path.exists():
            return data_path, hdr_path
    raise FileNotFoundError("Could not find yyc200 and yyc200.hdr")


DATA_PATH, HDR_PATH = find_yyc200_paths(PACKAGE_ROOT / "code")
OUT_DIR = PACKAGE_ROOT / "results" / "hyperspectral_exploration"
FIG_DIR = PACKAGE_ROOT / "figures" / "hyperspectral_exploration"
REPORT_PATH = OUT_DIR / "yyc200_analysis_report.md"


def parse_envi_header(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def get_int(name: str) -> int:
        match = re.search(rf"{name}\s*=\s*(\d+)", text, re.IGNORECASE)
        if not match:
            raise ValueError(f"Missing {name!r} in {path.name}")
        return int(match.group(1))

    wav_match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, re.IGNORECASE | re.DOTALL)
    if not wav_match:
        raise ValueError("Missing wavelength list in header")

    wavelengths = np.array(
        [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", wav_match.group(1))],
        dtype=float,
    )
    interleave = re.search(r"interleave\s*=\s*(\w+)", text, re.IGNORECASE)
    byte_order = re.search(r"byte order\s*=\s*(\d+)", text, re.IGNORECASE)
    data_type = re.search(r"data type\s*=\s*(\d+)", text, re.IGNORECASE)

    return {
        "samples": get_int("samples"),
        "lines": get_int("lines"),
        "bands": get_int("bands"),
        "interleave": interleave.group(1).lower() if interleave else None,
        "byte_order": int(byte_order.group(1)) if byte_order else None,
        "data_type": int(data_type.group(1)) if data_type else None,
        "wavelengths": wavelengths,
    }


def envi_dtype(data_type: int, byte_order: int) -> np.dtype:
    if data_type != 2:
        raise ValueError(f"This script currently expects ENVI data type 2, got {data_type}")
    endian = "<" if byte_order == 0 else ">"
    return np.dtype(endian + "i2")


def load_cube(meta: dict) -> np.ndarray:
    if meta["interleave"] != "bsq":
        raise ValueError(f"This script currently expects BSQ interleave, got {meta['interleave']}")

    expected = meta["bands"] * meta["lines"] * meta["samples"]
    raw = np.fromfile(DATA_PATH, dtype=envi_dtype(meta["data_type"], meta["byte_order"]))
    if raw.size != expected:
        raise ValueError(f"Expected {expected} values, found {raw.size}")
    return raw.reshape((meta["bands"], meta["lines"], meta["samples"])).astype(np.float32)


def percentile_stretch(image: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    image = image.astype(np.float32)
    lo = np.nanpercentile(image, low)
    hi = np.nanpercentile(image, high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def nearest_band(wavelengths: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(wavelengths - target)))


def save_rgb(cube: np.ndarray, wavelengths: np.ndarray, out_path: Path) -> tuple[int, int, int]:
    # Approximate visible RGB with the nearest available bands.
    r = nearest_band(wavelengths, 680)
    g = nearest_band(wavelengths, 560)
    b = nearest_band(wavelengths, 490)
    rgb = np.dstack(
        [
            percentile_stretch(cube[r]),
            percentile_stretch(cube[g]),
            percentile_stretch(cube[b]),
        ]
    )
    plt.imsave(out_path, rgb)
    return r, g, b


def save_ndvi_like(cube: np.ndarray, wavelengths: np.ndarray, out_path: Path) -> dict:
    red = nearest_band(wavelengths, 680)
    nir = nearest_band(wavelengths, 805)
    red_band = cube[red].astype(np.float32)
    nir_band = cube[nir].astype(np.float32)
    denom = nir_band + red_band
    ndvi = np.divide(nir_band - red_band, denom, out=np.zeros_like(nir_band), where=denom != 0)
    ndvi = np.clip(ndvi, -1, 1)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.6, vmax=0.6)
    ax.set_title("NDVI-like index")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("(NIR - red) / (NIR + red)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "red_band": red + 1,
        "red_wavelength": float(wavelengths[red]),
        "nir_band": nir + 1,
        "nir_wavelength": float(wavelengths[nir]),
        "min": float(np.min(ndvi)),
        "mean": float(np.mean(ndvi)),
        "max": float(np.max(ndvi)),
    }


def make_band_stats(cube: np.ndarray, wavelengths: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, wavelength in enumerate(wavelengths):
        band = cube[i]
        rows.append(
            {
                "band": i + 1,
                "wavelength_nm": wavelength,
                "min": float(np.min(band)),
                "p01": float(np.percentile(band, 1)),
                "mean": float(np.mean(band)),
                "median": float(np.median(band)),
                "p99": float(np.percentile(band, 99)),
                "max": float(np.max(band)),
                "zero_pct": float(np.mean(band == 0) * 100),
                "sat4095_pct": float(np.mean(band == 4095) * 100),
                "std": float(np.std(band)),
            }
        )
    return pd.DataFrame(rows)


def prepare_features(cube: np.ndarray, band_stats: pd.DataFrame) -> tuple[np.ndarray, list[int], dict]:
    good = band_stats[
        (band_stats["zero_pct"] < 20)
        & (band_stats["std"] > 1)
        & (band_stats["p99"] > band_stats["p01"])
    ]["band"].to_numpy()
    band_idx = [int(x) - 1 for x in good]

    x = np.moveaxis(cube[band_idx], 0, -1).reshape(-1, len(band_idx))
    lo = np.percentile(x, 1, axis=0)
    hi = np.percentile(x, 99, axis=0)
    x_clip = np.clip(x, lo, hi)
    x_scaled = StandardScaler().fit_transform(x_clip)
    prep = {
        "used_band_count": len(band_idx),
        "used_bands": [i + 1 for i in band_idx],
        "excluded_bands": [int(b) for b in band_stats.loc[~band_stats["band"].isin(good), "band"]],
        "clip_percentiles": [1, 99],
    }
    return x_scaled, band_idx, prep


def run_pca(x_scaled: np.ndarray) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=min(12, x_scaled.shape[1]), random_state=0)
    scores = pca.fit_transform(x_scaled)
    return pca, scores


def save_pca_outputs(scores: np.ndarray, pca: PCA, lines: int, samples: int, out_dir: Path) -> None:
    pca_img = scores[:, :3].reshape(lines, samples, 3)
    rgb = np.dstack([percentile_stretch(pca_img[:, :, i]) for i in range(3)])
    plt.imsave(out_dir / "pca_rgb.png", rgb)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    xs = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    ax.bar(xs, pca.explained_variance_ratio_ * 100)
    ax.plot(xs, np.cumsum(pca.explained_variance_ratio_) * 100, marker="o", color="tab:red")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA explained variance")
    ax.set_xticks(xs)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_variance.png")
    plt.close(fig)


def choose_k(scores: np.ndarray) -> tuple[int, pd.DataFrame]:
    rng = np.random.default_rng(7)
    cluster_x = scores[:, :8]
    sample_size = min(6000, cluster_x.shape[0])
    sample_idx = rng.choice(cluster_x.shape[0], sample_size, replace=False)
    rows = []
    best_k = None
    best_score = -np.inf
    for k in range(3, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=7)
        labels = km.fit_predict(cluster_x)
        sample_labels = labels[sample_idx]
        sil = silhouette_score(cluster_x[sample_idx], sample_labels)
        ch = calinski_harabasz_score(cluster_x, labels)
        rows.append({"k": k, "silhouette_sample": sil, "calinski_harabasz": ch, "inertia": km.inertia_})
        # Avoid an overly coarse segmentation unless the silhouette margin is decisive.
        adjusted = sil - (0.015 if k < 5 else 0)
        if adjusted > best_score:
            best_score = adjusted
            best_k = k
    return int(best_k), pd.DataFrame(rows)


def run_clustering(scores: np.ndarray, k: int, lines: int, samples: int) -> tuple[np.ndarray, KMeans]:
    cluster_x = scores[:, :8]
    km = KMeans(n_clusters=k, n_init=40, random_state=11)
    labels = km.fit_predict(cluster_x)
    return labels.reshape(lines, samples), km


def save_cluster_outputs(
    labels_img: np.ndarray,
    cube: np.ndarray,
    wavelengths: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    k = int(labels_img.max()) + 1
    cmap = plt.get_cmap("tab10", k)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(labels_img, cmap=cmap, interpolation="nearest", vmin=-0.5, vmax=k - 0.5)
    ax.set_title(f"KMeans spectral clusters (k={k})")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(k), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([f"C{i + 1}" for i in range(k)])
    fig.tight_layout()
    fig.savefig(out_dir / "kmeans_cluster_map.png")
    plt.close(fig)

    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    flat_labels = labels_img.reshape(-1)
    flat_cube = np.moveaxis(cube, 0, -1).reshape(-1, cube.shape[0])
    for cluster in range(k):
        mask = flat_labels == cluster
        mean_spectrum = flat_cube[mask].mean(axis=0)
        area_pct = float(mask.mean() * 100)
        peak_band = int(np.argmax(mean_spectrum))
        rows.append(
            {
                "cluster": f"C{cluster + 1}",
                "pixel_count": int(mask.sum()),
                "area_pct": area_pct,
                "mean_intensity": float(mean_spectrum.mean()),
                "peak_band": peak_band + 1,
                "peak_wavelength_nm": float(wavelengths[peak_band]),
                "peak_value": float(mean_spectrum[peak_band]),
            }
        )
        ax.plot(wavelengths[:60], mean_spectrum[:60], label=f"C{cluster + 1} ({area_pct:.1f}%)")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean digital number")
    ax.set_title("Mean spectra by cluster")
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_spectra.png")
    plt.close(fig)
    return pd.DataFrame(rows)


def write_report(
    meta: dict,
    band_stats: pd.DataFrame,
    prep: dict,
    pca: PCA,
    k_metrics: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    ndvi_summary: dict,
    rgb_bands: tuple[int, int, int],
) -> None:
    bad_bands = band_stats[band_stats["zero_pct"] >= 20]["band"].astype(int).tolist()
    saturated = band_stats[band_stats["sat4095_pct"] >= 0.5][["band", "sat4095_pct"]]
    pca_top = pca.explained_variance_ratio_[:5] * 100
    best_row = k_metrics.sort_values("silhouette_sample", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    chosen_k = int(cluster_summary.shape[0])
    chosen_row = k_metrics[k_metrics["k"] == chosen_k].iloc[0]

    def fmt_list(values: list[int]) -> str:
        return ", ".join(str(v) for v in values)

    report = f"""# yyc200 高光谱数据建模分析

## 数据概况

- 文件格式：ENVI Standard，BSQ 排列，16-bit signed integer，little-endian。
- 尺寸：{meta["lines"]} 行 × {meta["samples"]} 列 × {meta["bands"]} 波段，共 {meta["lines"] * meta["samples"]:,} 个像元。
- 波长范围：{meta["wavelengths"][0]:.1f} nm 到 {meta["wavelengths"][-1]:.1f} nm。
- 数值范围：{band_stats["min"].min():.0f} 到 {band_stats["max"].max():.0f}，整体为 12-bit 风格数字量级，最大值 4095 可视作饱和值。

## 质量检查

- 零值比例超过 20% 的波段：{fmt_list(bad_bands)}。
- 本次 PCA/聚类使用 {prep["used_band_count"]} 个信息较完整的波段：{fmt_list(prep["used_bands"])}。
- 建模前对每个波段做了 1%-99% 分位数截断，再进行标准化，以降低饱和值和极端值影响。
- 可见光近似 RGB 使用波段 R={rgb_bands[0] + 1} ({meta["wavelengths"][rgb_bands[0]]:.1f} nm), G={rgb_bands[1] + 1} ({meta["wavelengths"][rgb_bands[1]]:.1f} nm), B={rgb_bands[2] + 1} ({meta["wavelengths"][rgb_bands[2]]:.1f} nm)。
"""
    if not saturated.empty:
        saturated_text = ", ".join(
            f"B{int(row.band)}={row.sat4095_pct:.2f}%"
            for row in saturated.itertuples(index=False)
        )
        report += f"- 饱和像元比例超过 0.5% 的波段：{saturated_text}。\n"
    else:
        report += "- 各波段饱和值 4095 的比例都低于 0.5%。\n"

    report += f"""
## PCA 结果

- 前 5 个主成分解释方差：{", ".join(f"PC{i + 1}={v:.2f}%" for i, v in enumerate(pca_top))}。
- 前 3 个主成分累计解释：{np.sum(pca.explained_variance_ratio_[:3]) * 100:.2f}%。
- 前 5 个主成分累计解释：{np.sum(pca.explained_variance_ratio_[:5]) * 100:.2f}%。

## 无监督聚类

- 评估了 k=3 到 k=10 的 KMeans；样本轮廓系数最高的 k 为 {best_k} ({best_row["silhouette_sample"]:.3f})。
- 最终聚类图采用 k={chosen_k} ({chosen_row["silhouette_sample"]:.3f})，原因是它与最高轮廓系数差距较小，同时能表达更细的城市地物光谱分区。
- 这些类别是无监督光谱分区，不等同于已有地物标签。

| 类别 | 像元数 | 面积占比 | 平均强度 | 峰值波段 | 峰值波长 |
|---|---:|---:|---:|---:|---:|
"""
    for row in cluster_summary.itertuples(index=False):
        report += (
            f"| {row.cluster} | {row.pixel_count:,} | {row.area_pct:.2f}% | "
            f"{row.mean_intensity:.1f} | {row.peak_band} | {row.peak_wavelength_nm:.1f} nm |\n"
        )

    report += f"""
## NDVI-like 指数

- 使用 Red=B{ndvi_summary["red_band"]} ({ndvi_summary["red_wavelength"]:.1f} nm)，NIR=B{ndvi_summary["nir_band"]} ({ndvi_summary["nir_wavelength"]:.1f} nm)。
- 指数范围：{ndvi_summary["min"]:.3f} 到 {ndvi_summary["max"]:.3f}，均值 {ndvi_summary["mean"]:.3f}。
- 由于原始数据是数字量级而非明确反射率校正结果，这里只能作为植被/高近红外响应的探索性指标。

## 输出文件

- `results/hyperspectral_exploration/band_stats.csv`：逐波段统计。
- `results/hyperspectral_exploration/k_selection_metrics.csv`：聚类 k 值评估。
- `results/hyperspectral_exploration/cluster_summary.csv`：聚类面积与典型强度。
- `figures/hyperspectral_exploration/quicklook_rgb.png`：近似真彩色快视图。
- `figures/hyperspectral_exploration/pca_rgb.png`：PC1-PC3 合成图。
- `figures/hyperspectral_exploration/pca_variance.png`：PCA 解释方差。
- `figures/hyperspectral_exploration/kmeans_cluster_map.png`：KMeans 聚类分区图。
- `figures/hyperspectral_exploration/cluster_spectra.png`：各聚类平均光谱。
- `figures/hyperspectral_exploration/ndvi_like.png`：NDVI-like 指数图。

## 下一步建议

如果有地面样本、分类标签或 ROI，可在当前脚本基础上改成监督分类，例如 Random Forest、SVM 或 XGBoost，并用混淆矩阵、OA、Kappa、F1 等指标评价。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    meta = parse_envi_header(HDR_PATH)
    cube = load_cube(meta)
    wavelengths = meta["wavelengths"]

    band_stats = make_band_stats(cube, wavelengths)
    band_stats.to_csv(OUT_DIR / "band_stats.csv", index=False, encoding="utf-8-sig")

    rgb_bands = save_rgb(cube, wavelengths, FIG_DIR / "quicklook_rgb.png")
    ndvi_summary = save_ndvi_like(cube, wavelengths, FIG_DIR / "ndvi_like.png")

    x_scaled, band_idx, prep = prepare_features(cube, band_stats)
    pca, scores = run_pca(x_scaled)
    save_pca_outputs(scores, pca, meta["lines"], meta["samples"], FIG_DIR)

    chosen_k, k_metrics = choose_k(scores)
    k_metrics.to_csv(OUT_DIR / "k_selection_metrics.csv", index=False, encoding="utf-8-sig")

    labels_img, _ = run_clustering(scores, chosen_k, meta["lines"], meta["samples"])
    np.save(OUT_DIR / "kmeans_labels.npy", labels_img.astype(np.int16))

    cluster_summary = save_cluster_outputs(labels_img, cube, wavelengths, FIG_DIR)
    cluster_summary.to_csv(OUT_DIR / "cluster_summary.csv", index=False, encoding="utf-8-sig")

    model_summary = {
        "metadata": {
            "samples": meta["samples"],
            "lines": meta["lines"],
            "bands": meta["bands"],
            "interleave": meta["interleave"],
            "data_type": meta["data_type"],
            "byte_order": meta["byte_order"],
        },
        "preprocessing": prep,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "chosen_k": int(chosen_k),
        "ndvi_like": ndvi_summary,
    }
    (OUT_DIR / "model_summary.json").write_text(
        json.dumps(model_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(meta, band_stats, prep, pca, k_metrics, cluster_summary, ndvi_summary, rgb_bands)
    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote outputs: {OUT_DIR}")
    print(f"Wrote figures: {FIG_DIR}")
    print(f"Chosen k: {chosen_k}")
    print("PCA first five explained variance (%):", np.round(pca.explained_variance_ratio_[:5] * 100, 2))


if __name__ == "__main__":
    main()
