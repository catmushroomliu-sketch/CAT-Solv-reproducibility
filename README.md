# CAT-Solv Reproducibility Package

This repository contains the curated reproducibility materials for the CAT-Solv solvent-out-of-distribution solubility benchmark and the MICA deployment analysis.

The local working project contains historical drafts, preview figures, graphical-layout experiments, large descriptor-generation intermediates, and manuscript-editing scratch files. Those materials are intentionally excluded. This release package is limited to the code, source tables, final manuscript/SI files, final figure files, and metadata needed to audit or rerun the reported quantitative results.

## Repository scope

- Fixed leave-solvent-out benchmark construction for five held-out solvents.
- Descriptor hierarchy and matched feature-count experiments.
- Twenty-seed robustness analyses across tree learners.
- Source tables for manuscript figures and reported metrics.
- Final manuscript, Supporting Information, TOC graphic, and manuscript/SI figure PDFs.
- MICA model and descriptor metadata snapshots connecting this analysis package to the separate MICA software repository.

MICA itself is maintained separately at:

<https://github.com/catmushroomliu-sketch/MICA>

## Directory layout

```text
docs/
  manuscript/                 Final manuscript PDF, TeX, BibTeX, and TOC graphic
  supporting_information/     Final Supporting Information PDF, TeX, and BibTeX
figures/
  manuscript/                 Final manuscript figure PDFs and TOC graphic
  supporting_information/     Final selected SI figure files
metadata/
  mica/                       MICA model, feature, solvent-domain, and range metadata
scripts/
  experiments/                Main experiment scripts used for model/benchmark reruns
source_data/
  experiments/                Aggregated and per-seed experiment outputs
  figures/                    CSV/JSON source tables used to audit plotted values
```

## What is included

- Final manuscript and Supporting Information files used for submission.
- Final manuscript figures: Figure 1--Figure 6, Figure 8, and the TOC graphic.
- Final selected Supporting Information figures.
- Lightweight source data for figure/table audit.
- Experiment scripts for IID evaluation, descriptor ablation, top-k feature selection, and learner robustness.
- MICA metadata snapshots needed to interpret the production `cat-solv-47:xgboost` model bundle.

## What is not included

The following materials are excluded from the GitHub-sized code release:

- Full raw BigSolDB-derived feature matrices.
- Full row-level prediction tables for all seeds and learners.
- Large pair-xTB intermediate structures, logs, and xTB/COSMO caches.
- Full trained model bundles when they are better distributed through MICA or a DOI-backed data archive.
- Figure layout/artwork scripts. The manuscript figures were assembled through a mixed Python/Origin/HTML/SVG workflow; source CSV/JSON tables and final figure PDFs are provided for audit of quantitative values.
- Local manuscript drafts, Word files, AI-editing scratch files, historical preview figures, and machine-local absolute paths.

Large data products should be deposited as a versioned Zenodo archive or another DOI-backed repository. See `DATA_AVAILABILITY.md` and `DATA_PACKAGE_MANIFEST.md`.

## Quick setup

```bash
conda env create -f environment.yml
conda activate catsolv-repro
python scripts/audit_release.py
```

The audit checks for large files, local absolute paths, and common release-blocking placeholders, and regenerates `metadata/release_file_manifest.tsv`.

## Reproducibility levels

1. **Figure/table audit:** use the files in `source_data/` to verify manuscript numbers and plotted values.
2. **Prepared-feature rerun:** use `scripts/experiments/` together with the prepared feature matrices described in `DATA_PACKAGE_MANIFEST.md`.
3. **End-to-end descriptor regeneration:** requires the original descriptor-generation workflow, xTB/COSMO software, and large intermediate files; this level is documented but not bundled in the GitHub repository.

## Release status

This local package is prepared as a submission release candidate. Before public posting, the repository should be pushed to GitHub, tagged, and archived through Zenodo or an equivalent DOI-backed service. After the DOI is assigned, update `DATA_AVAILABILITY.md` and `CITATION.cff` with the final DOI and release date.
