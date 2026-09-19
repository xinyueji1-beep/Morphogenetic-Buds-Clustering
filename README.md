# MBC Final Independent Reproduction Folder

This folder is the complete handoff package for the revised MBC manuscript.
It contains the tracked manuscript, separate point-by-point responses to
Reviewers 1, 2, and 3, the source code, cached data, result tables, figures, and
the journal LaTeX class/package files.
No separate Supplementary Materials file is required: the supporting inventories,
diagnostics, limitations, and cited revised text are included in the main
manuscript, while the reproducible raw outputs and data remain in this folder.

## Main Documents

- `MBC_CMC_revision_tracked.tex/pdf`: latest tracked manuscript. Reviewer 1 changes are standard red, Reviewer 2 changes are standard blue, and Reviewer 3 changes are marked in deep green.
- `Response_to_Reviewer_1.tex/pdf`: individual response to Reviewer 1.
- `Response_to_Reviewer_2.tex/pdf`: individual response to Reviewer 2.
- `Response_to_Reviewer_3.tex/pdf`: individual response to Reviewer 3.
- `Definitions/tsp.cls`: the original journal class used by the manuscript, together with all supporting files in `Definitions/`.

## Reproduction Contents

- `code/mbc.py`: MBC implementation.
- `code/reproduce_all.py`: one readable entry script containing the main
  benchmark, robustness analyses, external cases, figure generation, and all
  Reviewer 1--3 analyses.
- `data/`: bundled benchmark, external, processed, cached, and hyperspectral data.
- `results/paper_results/`: original numerical outputs plus `reviewer1_*.csv`, `reviewer2_*.csv`, and `reviewer3_*.csv` raw/summary files.
- `figures/`: generated experiment figures and LaTeX figure assets.
- `docs/`: dataset provenance and supporting documentation.
- `requirements.txt`: Python dependencies.

## Environment

The experiments were run with Python 3.10.18 in the `hologpu` Conda environment
on Windows. PyCharm is optional; the commands below run from a normal terminal.

## Reproduce

Open PowerShell in this folder and run:

```powershell
conda activate hologpu
python -m pip install -r requirements.txt
python code/reproduce_all.py
```

The unified pipeline uses cached data under `data/` whenever available.
OpenML is only a fallback for a missing dataset. The bundled Lung-5 dataset is
used only for the external inventory; if its processed CSV is absent, the
script uses the matching raw OpenML cache when present and otherwise records a
warning and continues. Outputs are written to `results/paper_results/` and
generated figures to `figures/`. To inspect the execution order without
running analyses, use:

```powershell
python code/reproduce_all.py --list
```

## Compile The Tracked Files

MiKTeX or another LaTeX distribution is required:

```powershell
pdflatex MBC_CMC_revision_tracked.tex
pdflatex MBC_CMC_revision_tracked.tex
pdflatex Response_to_Reviewer_1.tex
pdflatex Response_to_Reviewer_1.tex
pdflatex Response_to_Reviewer_2.tex
pdflatex Response_to_Reviewer_2.tex
pdflatex Response_to_Reviewer_3.tex
pdflatex Response_to_Reviewer_3.tex
```

All code, data, package files, and result files needed by the supplied
manuscript/revision analyses are included in this folder, including
`Definitions/tsp.cls`. Results can vary slightly across operating systems,
Python versions, and numerical libraries; host-dependent wall-clock values
should be interpreted as relative diagnostics.
