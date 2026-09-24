# MBC — Final Submission Package

**Manuscript:** *Morphogenetic Buds Clustering: Local Prototype Dynamics and Continuity-Aware Fusion for
Continuous Non-Convex Structures*
**Authors:** Xuewen Shen¹, Xinyue Ji¹, Min Zhao¹, Liang Hong¹,²,*
**Package date:** 2026-09-21

This package contains everything required to (i) submit the revision and (ii) regenerate every number,
table, and figure reported in the manuscript and in the three response letters. Intermediate review
materials — verification scripts, consistency reports, and LaTeX build artefacts — have been removed.

---

## 1. What to send to the editorial office

| File | Pages | Purpose |
|---|---|---|
| `Cover_Letter.pdf` | 1 | Letter to the Editor-in-Chief |
| `MBC_CMC_revision_tracked.pdf` | 26 | Revised manuscript, changes tracked in colour |
| `Response_to_Reviewer_1.pdf` | 7 | Point-to-point reply to Reviewer 1 (6 comments) |
| `Response_to_Reviewer_2.pdf` | 8 | Point-to-point reply to Reviewer 2 (9 comments) |
| `Response_to_Reviewer_3.pdf` | 16 | Point-to-point reply to Reviewer 3 (21 comments) |

Sources are included alongside each PDF, so nothing has to be re-typeset before submission.

**Reading the tracked manuscript.** Reviewer 1 additions appear in red, Reviewer 2 in blue, Reviewer 3 in
green, and follow-up author corrections in purple. Unchanged text is black.

## 2. Reproducing every reported result

Python 3.10 or newer with the pinned dependencies in `requirements.txt`. From the package root, create an isolated environment and install the dependencies:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe code\reproduce_all.py --list
.venv\Scripts\python.exe code\reproduce_all.py
```

On macOS or Linux, use `python3 -m venv .venv`, `./.venv/bin/python -m pip install -r requirements.txt`, and `./.venv/bin/python code/reproduce_all.py`. The complete run regenerates all analyses, tables, and figures and overwrites files under `results/` and `figures/`; it may take substantially longer than the quick `--list` check. The bundled raw and processed caches are sufficient for the reported analyses, so network access is normally unnecessary.

The script resolves all paths from its own location, so it runs correctly from the package root without
configuration. Two source files carry the whole method and analysis:

* `code/mbc.py` — the MBC implementation.
* `code/reproduce_all.py` — main benchmark, external real-data benchmark, sensitivity study, runtime and
  biological analyses, the hyperspectral case study, all paper figures, and the Reviewer 1–3 analyses.

The script resolves all paths from its own location and can be run from any working directory. If an
optional cached external file is absent, the script attempts an OpenML download; if that is unavailable,
the affected diagnostic row is skipped with a warning rather than silently fabricated. `Lung_5class` is
bundled for the Reviewer 3 dataset inventory and is not part of the main external benchmark summary.

## 3. Package contents

```
Cover_Letter.tex / .pdf              cover letter to the editor
MBC_CMC_revision_tracked.tex / .pdf  revised manuscript + journal class usage
Response_to_Reviewer_{1,2,3}.tex/.pdf
Definitions/                         journal class (tsp.cls) and supporting style files
attrib.sty                           local stub, see note below
code/                                mbc.py, reproduce_all.py
data/                                manifests, cached datasets, hyperspectral cube
results/                             every result table referenced in the text
figures/paper_figures/               the eight PDF figures included in the manuscript
figures/hyperspectral_exploration/   supporting figures for the exploratory case study
docs/dataset_sources.md              provenance and licensing of every dataset
requirements.txt                     pinned Python dependencies
```

No separate supplementary-materials file is needed: the inventories, diagnostics, and limitations are
reported inside the revised manuscript, and this package regenerates them. Precomputed CSV, JSON, PNG,
and PDF result files are included so that the submission can be inspected without rerunning the full study.

## 4. Recompiling the LaTeX sources

```powershell
pdflatex MBC_CMC_revision_tracked.tex    # run twice
pdflatex Response_to_Reviewer_1.tex      # run twice
pdflatex Response_to_Reviewer_2.tex
pdflatex Response_to_Reviewer_3.tex
pdflatex Cover_Letter.tex
```

The manuscript uses `Definitions/tsp.cls`, which loads the `attrib` package. That package is absent from
some TeX Live distributions, and the class loads it without using any of its commands; the bundled
`attrib.sty` is therefore a minimal local stub that satisfies the loading step. Keep it next to the
`.tex` file. It was verified to compile cleanly under TeX Live 2026.

## 5. Verification performed before packaging

* All 34 reference entries are cited, none are orphaned, and they are numbered strictly in order of first
  appearance.
* Each quotation block labelled "Corresponding revised text" in the three response letters reproduces the
  tracked manuscript verbatim — 119 of 119 fragments matched.
* The five submission PDFs were checked for page counts, readable text, and missing placeholder strings.
* `code/reproduce_all.py --list` was executed successfully from the package root.
* The data manifest was checked against the bundled raw and processed files; optional inventory-only files
  are identified in `docs/dataset_sources.md`.

## 6. Declarations

Funding: Open Fund of Zhejiang Key Laboratory of Film and TV Media Technology (2024E10023); National
Natural Science Foundation of China (61671404).
Code and supporting materials: <https://github.com/xinyueji1-beep/Morphogenetic-Buds-Clustering>.
Third-party datasets retain their original licences; see `docs/dataset_sources.md`.
