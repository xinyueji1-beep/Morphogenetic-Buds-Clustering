from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_DIR.parent
ARCHIVE_PATH = CODE_DIR / "pipeline_sources.zip"
RESULT_DIR = PACKAGE_ROOT / "results" / "paper_results"
SKLEARN_CACHE_DIR = PACKAGE_ROOT / "data" / "sklearn_cache_local"

PAPER_SCRIPTS = [
    "export_submission_datasets.py",
    "paper_experiments_mbc.py",
    "sensitivity_mbc.py",
    "external_real_benchmark_mbc.py",
    "runtime_and_bio_visuals_mbc.py",
    "hyperspectral_yyc_mbc_case.py",
    "publication_cluster_figures.py",
    "publication_advanced_figures.py",
    "analyze_yyc200.py",
]

DEMO_SCRIPTS = [
    "morphogenetic_buds_clustering.py",
    "benchmark_mbc.py",
]

SUPPORT_SOURCES = [
    "morphogenetic_buds_clustering.py",
]

DEMO_OUTPUTS = [
    "morphogenetic_buds_result.png",
    "morphogenetic_buds_trials.json",
    "mbc_benchmark_results.csv",
    "mbc_benchmark_results.json",
    "mbc_benchmark_summary.md",
    "mbc_benchmark_ari.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete MBC manuscript reproduction pipeline."
    )
    parser.add_argument(
        "--paper-only",
        action="store_true",
        help="run only manuscript experiments and figures, excluding legacy demo scripts",
    )
    parser.add_argument(
        "--keep-sources",
        action="store_true",
        help="keep restored legacy source files in code/ after the run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the script order without running it",
    )
    return parser.parse_args()


def scripts_to_run(paper_only: bool) -> list[str]:
    scripts = list(PAPER_SCRIPTS)
    if not paper_only:
        scripts += DEMO_SCRIPTS
    return scripts


def restore_archived_sources(required: list[str]) -> list[Path]:
    if not ARCHIVE_PATH.exists():
        missing = ", ".join(required)
        raise FileNotFoundError(
            f"Missing {ARCHIVE_PATH}. Cannot restore pipeline scripts: {missing}"
        )

    needed = sorted(set(required + SUPPORT_SOURCES))
    created: list[Path] = []
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:
        available = set(archive.namelist())
        for name in needed:
            target = CODE_DIR / name
            if target.exists():
                continue
            if name not in available:
                raise FileNotFoundError(f"{name} is not present in {ARCHIVE_PATH}")
            archive.extract(name, CODE_DIR)
            created.append(target)
    return created


def run_script(name: str) -> None:
    script = CODE_DIR / name
    start = time.perf_counter()
    print(f"\n==> running {name}", flush=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    # Keep OpenML downloads inside the submission package. This avoids
    # permission problems and incompatible caches created by another Python
    # or scikit-learn installation (for example a PyCharm virtualenv).
    SKLEARN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    env["SCIKIT_LEARN_DATA"] = str(SKLEARN_CACHE_DIR)
    subprocess.run([sys.executable, str(script)], cwd=CODE_DIR, env=env, check=True)
    print(f"<== finished {name} in {time.perf_counter() - start:.1f}s", flush=True)


def normalize_generated_files() -> None:
    tex = RESULT_DIR / "paper_tables.tex"
    snippet = RESULT_DIR / "paper_tables_latex_snippet.txt"
    if tex.exists():
        if snippet.exists():
            snippet.unlink()
        tex.replace(snippet)

    unused_paper_figures = [
        "publication_bio_expression_pca.pdf",
        "publication_bio_expression_pca.png",
        "publication_nonconvex_showcase.pdf",
        "publication_nonconvex_showcase.png",
    ]
    for name in unused_paper_figures:
        src = PACKAGE_ROOT / "figures" / "paper_figures" / name
        if src.exists():
            dst = RESULT_DIR / name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))

    preview_dir = PACKAGE_ROOT / "figures" / "paper_png_previews"
    for src in sorted((PACKAGE_ROOT / "figures" / "paper_figures").glob("*.png")):
        preview_dir.mkdir(parents=True, exist_ok=True)
        dst = preview_dir / src.name
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))

    demo_dir = PACKAGE_ROOT / "results" / "demo_outputs"
    moved = False
    for name in DEMO_OUTPUTS:
        src = CODE_DIR / name
        if src.exists():
            demo_dir.mkdir(parents=True, exist_ok=True)
            dst = demo_dir / name
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            moved = True
    if moved:
        print(f"Moved legacy demo outputs to {demo_dir}", flush=True)


def cleanup_restored_sources(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    args = parse_args()
    scripts = scripts_to_run(args.paper_only)

    if args.list:
        for idx, script in enumerate(scripts, 1):
            print(f"{idx:02d}. {script}")
        return

    created_sources = restore_archived_sources(scripts)
    try:
        for script in scripts:
            run_script(script)
        normalize_generated_files()
    finally:
        if created_sources and not args.keep_sources:
            cleanup_restored_sources(created_sources)

    print("\nReproduction pipeline completed.", flush=True)


if __name__ == "__main__":
    main()
