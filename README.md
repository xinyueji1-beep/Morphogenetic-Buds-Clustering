# MBC R3 offline reproduction package (checked 2026-10-27)

This package contains the Python source, bundled input data, result ledgers, generated figures, and the notebook used for the revised manuscript. It is separate from the submission-only TeX package. The public repository is not required.

## Contents

- `code/main/`: experiment implementations and individual drivers.
- `code/generate/`: checkpointed result generation.
- `code/generate/export_baseline_parameters.py`: offline export of resolved estimator defaults and data-derived DBSCAN/MeanShift values; it does not fit clustering models.
- `code/plot/`: figures generated from the result ledgers.
- `code/run_all.py`: offline orchestration entry point.
- `code/notebooks/run_all.ipynb`: notebook wrapper for the entry point.
- `data/`: bundled raw and processed inputs. The corrected OpenML 45093 Lung file has 203 rows and five labels.
- `data/input_hashes.json`: SHA-256 and MD5 digests for the exact bundled raw and processed data files. These are local integrity hashes, not upstream-provider checksum claims.
- `results/`: bundled CSV/JSON result ledgers and figures.
- `results/exp/baseline_parameters.json`: comparator call rules, installed estimator defaults and 97 per-input density-scale records for the primary, public SIPU and scale suites.
- `audit/`: offline consistency check used for this checked package.
- `requirements.txt`: versions tested in the local `hologpu` Python 3.10.18 environment.

The package does not bundle Python or dependency wheels. Running it without network access requires an environment where the listed dependencies are already installed. The recorded experiment blocked Python network access, but the Windows host was not physically disconnected.

## Offline checks

From this directory, run:

```text
python audit/audit_repo_vs_manuscript.py
```

This reads only local files and checks the main MBC means, the two compact scale cases, the 11-method scale ledger, the corrected Lung-5 data and values, the CSV manifest, the comparator-parameter export, and every digest in `data/input_hashes.json`. A successful run writes `results/audit/audit_result.json`.

The top-level `code/run_all.py` regenerates both metadata exports between experiment generation and plotting. To regenerate either export by itself without rerunning model fits:

```text
python code/generate/export_baseline_parameters.py
python code/generate/export_input_hashes.py
```

To see the ordered pipeline stages:

```text
python code/run_all.py --list
```

## Re-running experiments

The package includes the result ledgers and verified stage snapshots from the archived offline run. Running `python code/run_all.py` without `--force` resumes only snapshots whose source and output hashes still match; changed stages are recomputed. In the eleven-method scale driver, incomplete output is written to an `_in_progress.csv` file, and the last complete ledger is replaced only after all rows finish.

A targeted Affinity Propagation recheck was completed in the local Windows `hologpu` environment on the 10,446-observation large-sample case for all three seeds. The runs completed in 124.190, 137.470 and 108.665 seconds, with ARI values 0.091228796, 0.095228576 and 0.090477391. The host had 31.73 GiB visible RAM; a process-memory sample during the first fit showed 4.36 GiB private memory. Full run details and the tested code hashes are recorded in `results/generated_json/affinity_propagation_large_n_recheck_20261027.json`. This targeted check does not constitute a forced rerun of every pipeline stage. No network access is needed or used.

Affinity Propagation uses an O(n^2) pairwise matrix. To rerun the full eleven-method scale stage:

```text
python code/generate/generate_results.py --stage r3_scale_all_methods --force
```

Incomplete stage output is kept in an `_in_progress.csv` file; the complete ledger is replaced only when all rows finish.

## Data and result facts checked here

- Lung-5 raw and processed inputs: 203 observations, five classes with counts 139/21/20/17/6.
- Lung-5 three-seed mean ARI: DBSCAN 0.4092, Spectral 0.2908, GMM 0.2906, Agglomerative 0.2232, MBC 0.1492.
- Main 14-dataset MBC mean: 0.5919.
- Compact scale MBC means: 0.9448 (high-dimensional case) and 0.8934 (large-n case).

Timing values remain hardware dependent. Clustering scores and dataset contents are the values stored in the supplied ledgers.
