"""
Hyperspectral yyc200 case study for MBC.

The script reads the ENVI BSQ cube stored next to the submission package,
builds robust PCA spectral features, optionally appends weak spatial
coordinates, and compares KMeans with MorphogeneticBudsClustering.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from morphogenetic_buds_clustering import MorphogeneticBudsClustering


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
FIG_DIR = PACKAGE_ROOT / "figures" / "paper_figures"
RESULTS_DIR = PACKAGE_ROOT / "results" / "paper_results"


def find_yyc200_paths() -> tuple[Path, Path]:
    candidates = [
        PACKAGE_ROOT / "data" / "yyc200_hyperspectral",
        WORKSPACE_ROOT,
        WORKSPACE_ROOT / "data" / "yyc200_hyperspectral",
    ]
    for base in candidates:
        data_path = base / "yyc200"
        hdr_path = base / "yyc200.hdr"
        if data_path.exists() and hdr_path.exists():
            return data_path, hdr_path
    raise FileNotFoundError("Could not locate yyc200 and yyc200.hdr")


DATA_PATH, HDR_PATH = find_yyc200_paths()


def parse_header(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def get_int(name: str) -> int:
        match = re.search(rf"{name}\s*=\s*(\d+)", text, re.IGNORECASE)
        if not match:
            raise ValueError(f"Missing {name} in {path}")
        return int(match.group(1))

    wav_match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, re.IGNORECASE | re.DOTALL)
    if not wav_match:
        raise ValueError("Missing wavelength list")
    wavelengths = np.array(
        [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", wav_match.group(1))],
        dtype=float,
    )
    return {
        "samples": get_int("samples"),
        "lines": get_int("lines"),
        "bands": get_int("bands"),
        "wavelengths": wavelengths,
    }


def load_cube(meta: dict) -> np.ndarray:
    expected = meta["bands"] * meta["lines"] * meta["samples"]
    raw = np.fromfile(DATA_PATH, dtype="<i2")
    if raw.size != expected:
        raise ValueError(f"Expected {expected} values, found {raw.size}")
    return raw.reshape(meta["bands"], meta["lines"], meta["samples"]).astype(np.float32)


def stretch(x: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def nearest_band(wavelengths: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(wavelengths - target)))


def band_statistics(cube: np.ndarray, wavelengths: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, wavelength in enumerate(wavelengths):
        band = cube[i]
        rows.append(
            {
                "band": i + 1,
                "wavelength_nm": wavelength,
                "p01": float(np.percentile(band, 1)),
                "p99": float(np.percentile(band, 99)),
                "mean": float(band.mean()),
                "std": float(band.std()),
                "zero_pct": float((band == 0).mean() * 100),
                "sat4095_pct": float((band == 4095).mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def build_features(cube: np.ndarray, band_stats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int], PCA]:
    used_bands = band_stats[
        (band_stats["zero_pct"] < 20)
        & (band_stats["std"] > 1)
        & (band_stats["p99"] > band_stats["p01"])
    ]["band"].astype(int).to_list()
    used_idx = [b - 1 for b in used_bands]

    x = np.moveaxis(cube[used_idx], 0, -1).reshape(-1, len(used_idx))
    lo = np.percentile(x, 1, axis=0)
    hi = np.percentile(x, 99, axis=0)
    x = np.clip(x, lo, hi)
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=8, random_state=0)
    scores = pca.fit_transform(x)
    spectral = StandardScaler().fit_transform(scores)

    lines, samples = cube.shape[1], cube.shape[2]
    yy, xx = np.mgrid[0:lines, 0:samples]
    xy = np.c_[yy.reshape(-1) / (lines - 1), xx.reshape(-1) / (samples - 1)]
    xy = (xy - 0.5) * 2

    spatial_spectral = np.c_[spectral, 0.35 * xy]
    return spectral, spatial_spectral, used_bands, pca


def remap_by_mean_signal(labels: np.ndarray, signal: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    order = sorted(np.unique(labels), key=lambda g: float(signal[labels == g].mean()))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels], dtype=np.int16)


def boundary_ratio(label_image: np.ndarray) -> float:
    horizontal = label_image[:, 1:] != label_image[:, :-1]
    vertical = label_image[1:, :] != label_image[:-1, :]
    return float((horizontal.sum() + vertical.sum()) / (horizontal.size + vertical.size))


def sampled_silhouette(x: np.ndarray, labels: np.ndarray, seed: int = 7) -> float:
    n = len(labels)
    sample_size = min(6000, n)
    rng = np.random.default_rng(seed)
    ids = rng.choice(n, sample_size, replace=False)
    return float(silhouette_score(x[ids], labels[ids]))


def evaluate(name: str, feature_name: str, x: np.ndarray, labels: np.ndarray, elapsed: float, lines: int, samples: int) -> dict:
    label_image = labels.reshape(lines, samples)
    return {
        "algorithm": name,
        "feature": feature_name,
        "clusters": int(len(np.unique(labels))),
        "silhouette_sample": sampled_silhouette(x, labels),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
        "boundary_ratio": boundary_ratio(label_image),
        "runtime_seconds": elapsed,
    }


def run_kmeans(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    labels = KMeans(n_clusters=k, n_init=40, random_state=seed).fit_predict(x)
    return labels, time.perf_counter() - start


def run_mbc(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, float]:
    model = MorphogeneticBudsClustering(
        initial_buds=max(12, k * 6),
        steps=140,
        growth_rate=0.15,
        inhibition=0.045,
        apoptosis_mass=max(2.5, len(x) * 0.0055),
        birth_quantile=0.88,
        organ_merge_radius=3.6,
        valley_ratio=0.18,
        tangent_alignment=0.35,
        max_bridge_void=0.46,
        target_clusters=k,
        compact_refinement_steps=18,
        compact_refinement_restarts=10,
        compact_refinement_dim_threshold=3,
        random_state=seed,
    )
    start = time.perf_counter()
    labels = model.fit_predict(x)
    return labels, time.perf_counter() - start


def cluster_summary(cube: np.ndarray, labels: np.ndarray, wavelengths: np.ndarray) -> pd.DataFrame:
    flat_cube = np.moveaxis(cube, 0, -1).reshape(-1, cube.shape[0])
    rows = []
    for g in np.unique(labels):
        mask = labels == g
        mean_spectrum = flat_cube[mask].mean(axis=0)
        peak = int(np.argmax(mean_spectrum[:60]))
        rows.append(
            {
                "cluster": f"C{int(g) + 1}",
                "pixel_count": int(mask.sum()),
                "area_pct": float(mask.mean() * 100),
                "mean_intensity": float(mean_spectrum.mean()),
                "peak_band": peak + 1,
                "peak_wavelength_nm": float(wavelengths[peak]),
                "peak_value": float(mean_spectrum[peak]),
            }
        )
    return pd.DataFrame(rows)


def plot_case_figure(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    pca: PCA,
    spectral: np.ndarray,
    kmeans_labels: np.ndarray,
    mbc_labels: np.ndarray,
    out_path: Path,
) -> None:
    lines, samples = cube.shape[1], cube.shape[2]
    r = nearest_band(wavelengths, 680)
    g = nearest_band(wavelengths, 560)
    b = nearest_band(wavelengths, 490)
    rgb = np.dstack([stretch(cube[r]), stretch(cube[g]), stretch(cube[b])])

    pca_img = spectral[:, :3].reshape(lines, samples, 3)
    pca_rgb = np.dstack([stretch(pca_img[:, :, i]) for i in range(3)])

    kmeans_img = kmeans_labels.reshape(lines, samples)
    mbc_img = mbc_labels.reshape(lines, samples)
    cmap = plt.get_cmap("tab10", 6)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.0), dpi=180)
    panels = [
        (rgb, "Approximate RGB", None),
        (pca_rgb, f"PCA feature view ({pca.explained_variance_ratio_[:3].sum() * 100:.1f}%)", None),
        (kmeans_img, "KMeans, spectral-spatial features", cmap),
        (mbc_img, "MBC, spectral-spatial features", cmap),
    ]
    for ax, (image, title, panel_cmap) in zip(axes.ravel(), panels):
        if panel_cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=panel_cmap, interpolation="nearest", vmin=-0.5, vmax=5.5)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_mbc_spectra(cube: np.ndarray, labels: np.ndarray, wavelengths: np.ndarray, out_path: Path) -> None:
    flat_cube = np.moveaxis(cube, 0, -1).reshape(-1, cube.shape[0])
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    for g in np.unique(labels):
        mask = labels == g
        mean_spectrum = flat_cube[mask].mean(axis=0)
        ax.plot(wavelengths[:60], mean_spectrum[:60], label=f"C{int(g) + 1} ({mask.mean() * 100:.1f}%)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean digital number")
    ax.set_title("MBC mean spectra by cluster")
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists() or not HDR_PATH.exists():
        raise FileNotFoundError("yyc200 and yyc200.hdr must be stored next to the submission package")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    meta = parse_header(HDR_PATH)
    cube = load_cube(meta)
    wavelengths = meta["wavelengths"]
    band_stats = band_statistics(cube, wavelengths)
    spectral, spatial_spectral, used_bands, pca = build_features(cube, band_stats)

    lines, samples = meta["lines"], meta["samples"]
    mean_signal = cube.mean(axis=0).reshape(-1)
    k = 6
    seed = 7

    rows = []
    labels_for_fig = {}
    for feature_name, x in [("spectral_pca", spectral), ("spectral_pca_xy", spatial_spectral)]:
        labels, elapsed = run_kmeans(x, k, seed)
        labels = remap_by_mean_signal(labels, mean_signal)
        rows.append(evaluate("KMeans", feature_name, x, labels, elapsed, lines, samples))
        if feature_name == "spectral_pca_xy":
            labels_for_fig["KMeans"] = labels

        if feature_name == "spectral_pca_xy":
            labels, elapsed = run_mbc(x, k, seed)
            labels = remap_by_mean_signal(labels, mean_signal)
            rows.append(evaluate("MBC", feature_name, x, labels, elapsed, lines, samples))
            labels_for_fig["MBC"] = labels

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "hyperspectral_yyc_mbc_metrics.csv", index=False, encoding="utf-8-sig")

    mbc_labels = labels_for_fig["MBC"]
    cluster_summary(cube, mbc_labels, wavelengths).to_csv(
        RESULTS_DIR / "hyperspectral_yyc_mbc_cluster_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.save(RESULTS_DIR / "hyperspectral_yyc_mbc_labels.npy", mbc_labels.reshape(lines, samples).astype(np.int16))

    plot_case_figure(
        cube,
        wavelengths,
        pca,
        spectral,
        labels_for_fig["KMeans"],
        mbc_labels,
        FIG_DIR / "hyperspectral_yyc_mbc_case.png",
    )
    plot_mbc_spectra(cube, mbc_labels, wavelengths, FIG_DIR / "hyperspectral_yyc_mbc_spectra.png")

    summary = {
        "lines": lines,
        "samples": samples,
        "bands": meta["bands"],
        "used_bands": used_bands,
        "pca_first3_variance_pct": float(pca.explained_variance_ratio_[:3].sum() * 100),
        "target_clusters": k,
        "seed": seed,
    }
    pd.Series(summary).to_json(RESULTS_DIR / "hyperspectral_yyc_mbc_summary.json", force_ascii=False, indent=2)

    print(metrics.round(4).to_string(index=False))
    print(f"Wrote {FIG_DIR / 'hyperspectral_yyc_mbc_case.png'}")
    print(f"Wrote {FIG_DIR / 'hyperspectral_yyc_mbc_spectra.png'}")


if __name__ == "__main__":
    sys.exit(main())
