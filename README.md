# CAT-Solv Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22308071.svg)](https://doi.org/10.5281/zenodo.22308071)

This repository provides the reproducibility package for the CAT-Solv solvent-out-of-distribution (OOD) solubility benchmark and its associated MICA deployment workflow.

CAT-Solv evaluates solubility prediction when the solvent identity itself is absent from model development. The release contains the source data, experiment scripts, figure assets, and model metadata needed to inspect the reported results, reproduce the main benchmark analyses, and connect the manuscript results to the deployed MICA model bundle.

The archived release is available on Zenodo:

**DOI:** https://doi.org/10.5281/zenodo.22308071

---

## What this repository is for

This package is intended for readers who want to:

- audit the numerical values reported in the CAT-Solv manuscript and Supporting Information;
- reproduce the fixed five-solvent OOD benchmark and descriptor-ablation analyses;
- inspect the 20-seed robustness results across multiple tree learners;
- trace manuscript figures and tables back to their source CSV/JSON files;
- examine the final manuscript figures, Supporting Information figures, and TOC artwork;
- inspect the feature, solvent-domain, and applicability-domain metadata used by MICA;
- rerun the reported experiments when the corresponding prepared descriptor matrices are available.

The repository is organized as a publication-facing reproducibility package rather than as a complete snapshot of the original development workspace.

---

## Scientific scope

The main benchmark evaluates transfer to five solvent identities held out from all model-development steps.

The released analyses cover:

- fixed solvent-level OOD benchmark construction;
- random-split IID versus solvent-OOD evaluation;
- the descriptor hierarchy from 2D molecular descriptors to Gasteiger electrostatics and GFN2-xTB/COSMO surface descriptors;
- matched feature-count comparisons for CAT-Solv-47;
- seed-wise robustness across 20 train/validation resamplings;
- learner controls using XGBoost, Random Forest, Extra Trees, and HistGradientBoosting;
- solvent-resolved error analysis;
- SHAP attribution source data;
- pair-specific xTB negative-control experiments;
- MICA deployment metadata and prepared-descriptor inference benchmarks.

---

## Publication assets

The `figures/` directory contains the final publication graphics associated with the CAT-Solv study.

```text
figures/
  manuscript/
  supporting_information/
