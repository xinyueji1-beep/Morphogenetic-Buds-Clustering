from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper_results"
FIGURES = ROOT / "figures" / "paper_figures"

ALGO_ORDER = [
    "MBC",
    "Spectral",
    "KMeans",
    "GMM",
    "Agglomerative",
    "CompactRefineOnly",
    "DBSCAN",
]


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#25313d",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#26323f",
            "xtick.color": "#26323f",
            "ytick.color": "#26323f",
            "text.color": "#1f2a36",
        }
    )


def _save_figure(fig, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _ordered_columns(df):
    cols = [c for c in ALGO_ORDER if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]


def _heatmap(
    data,
    title,
    outfile,
    cmap="viridis",
    vmin=0,
    vmax=1,
    highlight=("MBC",),
    cbar_label="ARI",
    figsize=None,
):
    data = _ordered_columns(data)
    rows, cols = data.shape
    if figsize is None:
        figsize = (max(7.5, cols * 1.08), max(3.9, rows * 0.58 + 1.7))

    fig, ax = plt.subplots(figsize=figsize)
    values = data.to_numpy(dtype=float)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(data.columns, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(data.index)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=14)

    for i in range(rows):
        row = values[i]
        finite = np.isfinite(row)
        best = np.nanmax(row[finite]) if finite.any() else np.nan
        for j in range(cols):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            is_best = np.isclose(value, best)
            text_color = "white" if value > (vmin + 0.62 * (vmax - vmin)) else "#15202b"
            label = f"{value:.2f}" + ("*" if is_best else "")
            ax.text(j, i, label, ha="center", va="center", fontsize=8.2, color=text_color)

    for name in highlight:
        if name in data.columns:
            j = list(data.columns).index(name)
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, -0.5),
                    1,
                    rows,
                    fill=False,
                    edgecolor="#ffb000",
                    linewidth=2.2,
                    clip_on=False,
                )
            )

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.tick_params(length=0)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Dataset / family")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.015)
    cbar.set_label(cbar_label)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    _save_figure(fig, outfile)
    plt.close(fig)


def family_heatmap():
    df = pd.read_csv(RESULTS / "paper_family_ari.csv").set_index("family")
    order = [
        "nonconvex",
        "synthetic_gaussian",
        "real_tabular",
        "real_image_features",
        "real_low_dim",
        "synthetic_density",
    ]
    df = df.loc[[x for x in order if x in df.index]]
    _heatmap(
        df,
        "Performance landscape across data morphologies",
        "paper_fig_family_heatmap.png",
        cmap="YlGnBu",
        highlight=("MBC",),
        figsize=(9.2, 5.3),
    )


def dataset_heatmap():
    df = pd.read_csv(RESULTS / "paper_benchmark_summary.csv")
    pivot = df.pivot_table(index="dataset", columns="algorithm", values="ari_mean", aggfunc="mean")
    order = [
        "two_moons",
        "circles",
        "noisy_moons",
        "noisy_circles",
        "anisotropic_blobs",
        "varied_density",
        "iris",
        "wine",
        "breast_cancer",
        "digits_0_4",
        "ecoli",
        "glass",
        "vehicle",
        "cmc",
        "wdbc",
    ]
    pivot = pivot.loc[[x for x in order if x in pivot.index]]
    _heatmap(
        pivot,
        "Dataset-level ARI matrix",
        "paper_fig_dataset_heatmap.png",
        cmap="mako" if "mako" in plt.colormaps() else "PuBuGn",
        highlight=("MBC",),
        figsize=(9.6, 5.5),
    )


def dataset_dot_profile():
    df = pd.read_csv(RESULTS / "paper_benchmark_summary.csv")
    pivot = df.pivot_table(index="dataset", columns="algorithm", values="ari_mean", aggfunc="mean")
    order = [
        "two_moons",
        "circles",
        "noisy_moons",
        "noisy_circles",
        "anisotropic_blobs",
        "varied_density",
        "iris",
        "wine",
        "breast_cancer",
        "digits_0_4",
        "ecoli",
        "glass",
        "vehicle",
        "cmc",
        "wdbc",
    ]
    pivot = _ordered_columns(pivot.loc[[x for x in order if x in pivot.index]])

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    y = np.arange(len(pivot))
    baseline_cols = [c for c in pivot.columns if c != "MBC"]
    palette = {
        "MBC": "#d95f02",
        "Spectral": "#4e79a7",
        "KMeans": "#59a14f",
        "GMM": "#b07aa1",
        "Agglomerative": "#9c755f",
        "CompactRefineOnly": "#7f7f7f",
        "DBSCAN": "#bab0ab",
    }

    for i, dataset in enumerate(pivot.index):
        row = pivot.loc[dataset]
        ax.hlines(i, row.min(), row.max(), color="#c7d0d9", linewidth=1.8, zorder=1)
        ax.scatter(
            row[baseline_cols],
            np.full(len(baseline_cols), i),
            s=42,
            color=[palette.get(c, "#999999") for c in baseline_cols],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        ax.scatter(row["MBC"], i, marker="D", s=92, color=palette["MBC"], edgecolor="white", linewidth=0.9, zorder=4)
        best_alg = row.idxmax()
        ax.scatter(row[best_alg], i, marker="*", s=150, color="#f2c94c", edgecolor="#5c4a00", linewidth=0.5, zorder=5)
        ax.text(min(1.08, row["MBC"] + 0.025), i, f"{row['MBC']:.2f}", va="center", fontsize=8.2, color="#6b2d00")

    handles = [
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=palette["MBC"], markeredgecolor="white", markersize=8, label="MBC"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#f2c94c", markeredgecolor="#5c4a00", markersize=10, label="Best on dataset"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#4e79a7", markeredgecolor="white", markersize=7, label="Baselines"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlim(-0.06, 1.12)
    ax.set_xlabel("ARI")
    ax.set_title("Dataset-level performance profile", loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", color="#d9e0e6", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save_figure(fig, "paper_fig_dataset_dotprofile.png")
    plt.close(fig)


def ablation_lollipop():
    df = pd.read_csv(RESULTS / "paper_ablation_summary.csv")
    agg = (
        df.groupby("algorithm", as_index=False)
        .agg(mean=("ari_mean", "mean"), std=("ari_mean", "std"))
        .sort_values("mean", ascending=True)
    )
    labels = {
        "MBC": "Full MBC",
        "MBC_no_capillary": "No capillary",
        "MBC_no_polarity": "No polarity",
        "MBC_no_compact": "No steady refinement",
        "MBC_micro_only": "Micro-buds only",
    }
    agg["label"] = agg["algorithm"].map(labels).fillna(agg["algorithm"])
    colors = ["#557c9e" if a != "MBC" else "#d95f02" for a in agg["algorithm"]]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    y = np.arange(len(agg))
    ax.hlines(y, 0, agg["mean"], color="#b8c2cc", linewidth=4, alpha=0.7)
    ax.scatter(agg["mean"], y, s=145, c=colors, edgecolor="white", linewidth=1.4, zorder=3)
    for pos, (_, row) in enumerate(agg.iterrows()):
        ax.text(row["mean"] + 0.018, y[pos], f"{row['mean']:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(agg["label"])
    ax.set_xlim(0, max(0.82, agg["mean"].max() + 0.10))
    ax.set_xlabel("Mean ARI across benchmark datasets")
    ax.set_title("Ablation trajectory of morphogenetic mechanisms", loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", color="#d9e0e6", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    _save_figure(fig, "paper_fig_ablation_lollipop.png")
    plt.close(fig)


def external_heatmap():
    df = pd.read_csv(RESULTS / "external_real_benchmark_summary.csv")
    pivot = df.pivot_table(index="dataset", columns="algorithm", values="mean", aggfunc="mean")
    order = [
        "Leukemia_2class",
        "Leukemia_3class",
        "Lymphoma_3class",
        "Lung_5class",
        "MiceProtein",
        "credit-g",
        "diabetes",
        "vehicle",
        "ionosphere",
        "blood",
    ]
    pivot = pivot.loc[[x for x in order if x in pivot.index]]
    _heatmap(
        pivot,
        "External real-data ARI matrix",
        "external_real_benchmark_heatmap.png",
        cmap="rocket_r" if "rocket_r" in plt.colormaps() else "YlOrRd",
        vmin=-0.05,
        vmax=max(0.55, float(np.nanmax(pivot.to_numpy()))),
        highlight=("MBC",),
        figsize=(9.8, 6.3),
    )


def external_rank_flow():
    df = pd.read_csv(RESULTS / "external_real_benchmark_summary.csv")
    pivot = df.pivot_table(index="dataset", columns="algorithm", values="mean", aggfunc="mean")
    order = [
        "Leukemia_2class",
        "Leukemia_3class",
        "Lymphoma_3class",
        "Lung_5class",
        "MiceProtein_8class",
        "credit-g",
        "diabetes",
        "vehicle",
        "ionosphere",
        "blood-transfusion",
    ]
    order = [x for x in order if x in pivot.index]
    pivot = _ordered_columns(pivot.loc[order])
    x = np.arange(len(pivot))

    fig, ax = plt.subplots(figsize=(10.6, 5.5))
    for col in pivot.columns:
        values = pivot[col].to_numpy(dtype=float)
        if col == "MBC":
            ax.plot(x, values, color="#d95f02", linewidth=2.8, marker="D", markersize=6.5, label="MBC", zorder=4)
        elif col in ("GMM", "Agglomerative", "Spectral"):
            ax.plot(x, values, color="#6f7f8f", linewidth=1.4, marker="o", markersize=4.2, alpha=0.72, label=col, zorder=2)
        else:
            ax.plot(x, values, color="#b8c2cc", linewidth=1.0, marker="o", markersize=3.4, alpha=0.58, label=col, zorder=1)

    for i, dataset in enumerate(pivot.index):
        best_alg = pivot.loc[dataset].idxmax()
        best_val = pivot.loc[dataset, best_alg]
        ax.scatter(i, best_val, marker="*", s=145, color="#f2c94c", edgecolor="#5c4a00", linewidth=0.5, zorder=5)

    ax.axhline(0, color="#8b98a5", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=28, ha="right")
    ax.set_ylabel("ARI")
    ax.set_ylim(min(-0.08, np.nanmin(pivot.to_numpy()) - 0.03), max(0.62, np.nanmax(pivot.to_numpy()) + 0.05))
    ax.set_title("External real-data performance trajectories", loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", color="#d9e0e6", linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    keep = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen and l in ["MBC", "GMM", "Agglomerative", "Spectral", "KMeans", "DBSCAN", "CompactRefineOnly"]:
            keep.append((h, l))
            seen.add(l)
    ax.legend([h for h, _ in keep], [l for _, l in keep], frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, "external_real_benchmark_rankflow.png")
    plt.close(fig)


def sensitivity_small_multiples():
    df = pd.read_csv(RESULTS / "mbc_sensitivity_summary.csv")
    params = list(df["parameter"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.0))
    axes = axes.ravel()
    for ax, param in zip(axes, params):
        sub = df[df["parameter"] == param].sort_values("value")
        ax.plot(sub["value"], sub["mean"], color="#2a6f97", linewidth=2.2)
        ax.fill_between(
            sub["value"].astype(float),
            (sub["mean"] - sub["std"]).clip(lower=0),
            (sub["mean"] + sub["std"]).clip(upper=1),
            color="#2a6f97",
            alpha=0.16,
            linewidth=0,
        )
        ax.scatter(sub["value"], sub["mean"], color="#f28e2b", s=32, zorder=3)
        ax.set_title(param.replace("_", " "), fontsize=10, loc="left")
        ax.set_ylim(0, 1.03)
        ax.grid(color="#d9e0e6", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(params) :]:
        ax.axis("off")
    fig.suptitle("Parameter response curves of MBC", x=0.03, ha="left", fontsize=13, fontweight="bold")
    fig.supxlabel("Parameter value")
    fig.supylabel("Mean ARI")
    fig.tight_layout()
    _save_figure(fig, "mbc_sensitivity_curves.png")
    plt.close(fig)


def main():
    _style()
    family_heatmap()
    dataset_dot_profile()
    ablation_lollipop()
    external_rank_flow()
    sensitivity_small_multiples()
    print("Advanced publication figures written to", FIGURES)


if __name__ == "__main__":
    main()
