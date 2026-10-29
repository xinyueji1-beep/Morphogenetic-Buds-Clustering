"""Single offline entry point for the complete paper support package.

Order: generate JSON/checkpointed result ledgers, draw figures from those
ledgers, then run the manuscript-to-results audit and numerical report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATE = ROOT / "code" / "generate" / "generate_results.py"
PLOT = ROOT / "code" / "plot" / "plot_results.py"
AUDIT = ROOT / "audit" / "audit_repo_vs_manuscript.py"
NUMBERS = ROOT / "code" / "main" / "build_numbers.py"
PARAMETERS = ROOT / "code" / "generate" / "export_baseline_parameters.py"
INPUT_HASHES = ROOT / "code" / "generate" / "export_input_hashes.py"
LOG_DIR = ROOT / "results" / "generated_json"


def _run(command: list[str], log_name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / log_name
    env = os.environ.copy()
    env["MBC_OFFLINE"] = "1"
    env["MPLBACKEND"] = "Agg"
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen(
            command, cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            stream.write(line)
            stream.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = proc.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete MBC paper package offline.")
    parser.add_argument("--list", action="store_true", help="list the ordered top-level stages")
    parser.add_argument("--stage", help="resume generation at a named generation stage")
    parser.add_argument("--force", action="store_true", help="rerun completed generation stages")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print("1. code/generate/generate_results.py (JSON ledgers and checkpoints)")
        print("2. export comparator parameter defaults and per-input density scales")
        print("3. hash the bundled input bytes for the local integrity manifest")
        print("4. code/plot/plot_results.py (figures from ledgers)")
        print("5. audit/audit_repo_vs_manuscript.py --r3")
        print("6. code/main/build_numbers.py")
        return
    started = time.perf_counter()
    generate = [sys.executable, str(GENERATE)]
    if args.stage:
        generate += ["--stage", args.stage]
    if args.force:
        generate.append("--force")
    _run(generate, "run_generate.log")
    _run([sys.executable, str(PARAMETERS)], "export_baseline_parameters.log")
    _run([sys.executable, str(INPUT_HASHES)], "export_input_hashes.log")
    _run([sys.executable, str(PLOT)], "run_plot.log")
    if AUDIT.exists():
        _run([sys.executable, str(AUDIT), "--r3"], "run_audit.log")
    else:
        print("audit/audit_repo_vs_manuscript.py not found; skipping audit step (pure code+data reproduction).")
    _run([sys.executable, str(NUMBERS)], "run_numbers.log")
    manifest = {
        "status": "complete",
        "offline": True,
        "elapsed_seconds": time.perf_counter() - started,
        "commands": [
            "python code/generate/generate_results.py",
            "python code/plot/plot_results.py",
            "python audit/audit_repo_vs_manuscript.py --r3",
            "python code/main/build_numbers.py",
        ],
        "logs": ["results/generated_json/run_generate.log", "results/generated_json/export_baseline_parameters.log", "results/generated_json/export_input_hashes.log", "results/generated_json/run_plot.log", "results/generated_json/run_audit.log", "results/generated_json/run_numbers.log"],
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "run_all_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
