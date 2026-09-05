# CAT-Solv Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22308071.svg)](https://doi.org/10.5281/zenodo.22308071)

This repository provides the reproducibility package for the CAT-Solv solvent-out-of-distribution (OOD) solubility benchmark and the associated MICA deployment workflow.

CAT-Solv evaluates solubility prediction when entire solvent identities are absent from model development. The benchmark is designed to distinguish performance under conventional random splitting from transfer to previously unseen solvents and to test whether quantum-chemistry-derived surface descriptors and solute-solvent compatibility features improve that transfer.

The archived release is available at:

**https://doi.org/10.5281/zenodo.22308071**

## Scope

The reproducibility package covers:

- construction of the fixed five-solvent OOD benchmark;
- random-split IID and solvent-OOD evaluation;
- descriptor-ablation experiments;
- Gasteiger electrostatic descriptor controls;
- GFN2-xTB/COSMO surface-descriptor experiments;
- matched feature-count comparisons for CAT-Solv-47;
- validation-based feature selection;
- 20-seed train/validation resampling;
- learner controls with XGBoost, Random Forest, Extra Trees, and HistGradientBoosting;
- solvent-resolved performance analysis;
- SHAP attribution data;
- pair-specific xTB negative-control experiments;
- metadata required to interpret the deployed CAT-Solv model ensemble;
- prepared-descriptor inference benchmarks used by MICA.

## Benchmark

The final aligned dataset contains 103,262 BigSolDB v2.1 records covering 1,394 solutes and 70 solvents.

Five solvent identities are fixed as the OOD test set before model development:

- DMSO
- cyclohexane
- ethylene glycol
- N-methyl-2-pyrrolidone (NMP)
- n-propyl acetate

These solvents are excluded from training, validation, early stopping, SHAP-based feature ranking, and feature selection.

The resulting split contains:

- **97,838 non-OOD rows** for model development;
- **5,424 fixed OOD rows** for final solvent-transfer evaluation.

Twenty seeds (42-61) vary only the train/validation partition within the non-OOD pool. The five-solvent OOD test set remains fixed.

## Descriptor hierarchy

The principal descriptor layers are:

- **2D-Min** — compact RDKit molecular descriptors and temperature;
- **2D-Elec** — 2D-Min plus Gasteiger partial-charge statistics;
- **Sigma-noElec** — 2D-Min plus monomer-level GFN2-xTB/COSMO surface descriptors;
- **Sigma-Solv** — 2D-Elec plus the COSMO surface descriptors;
- **CAT-Solv candidate pool** — Sigma-Solv plus deterministic solute-solvent compatibility features;
- **CAT-Solv-47** — a seed-specific 47-feature representation selected using validation-set SHAP ranking.

The 20 CAT-Solv-47 models use slightly different selected subsets. Their union contains 52 unique input features.

## Main reproducibility targets

The package can be used to reproduce or audit the following analyses:

### IID versus solvent-OOD evaluation

Random splitting provides an IID reference, while the fixed solvent-level holdout measures transfer to solvent identities absent from model development.

### Descriptor ablation

The descriptor ladder separates the effects of conventional 2D descriptors, Gasteiger electrostatic information, and GFN2-xTB/COSMO surface information.

### Matched feature-count comparison

CAT-Solv-47 and Sigma-Solv both contain 47 features per model, allowing descriptor content to be compared without attributing differences simply to feature dimensionality.

### Seed robustness

The primary analysis uses 20 train/validation seeds while preserving the same OOD test set.

### Learner controls

Descriptor effects are evaluated across:

- XGBoost
- Random Forest
- Extra Trees
- HistGradientBoosting

### Solvent-resolved analysis

Performance is reported separately for each held-out solvent to identify where descriptor enrichment improves transfer and where little or negative improvement is observed.

### SHAP attribution

SHAP data are provided for model interpretation and feature-attribution audits. SHAP values are treated as model attributions rather than mechanistic or causal quantities.

### Pair-xTB negative control

Additional solute-solvent dimer and small-cluster descriptors were evaluated under the fixed OOD protocol to test whether sparse optimized pair geometries improve over the monomer-level surface representation.

## MICA

MICA is the companion software package for deploying the frozen CAT-Solv ensemble.

MICA is maintained separately at:

**https://github.com/catmushroomliu-sketch/MICA**

The production model is identified as:

```text
cat-solv-47:xgboost
```

The default bundle contains 20 seed-specific XGBoost models.

For a prepared descriptor table, MICA provides:

- ensemble-mean solubility predictions;
- across-seed sample standard deviation;
- minimum and maximum predictions across seed models;
- feature and model metadata;
- applicability-domain checks;
- solvent-domain status;
- solvent-ranking utilities.

MICA currently operates on prepared descriptor tables. End-to-end generation of GFN2-xTB/COSMO descriptors from input SMILES is outside the deployed inference workflow.

### Repository structure
metadata/
  mica/                 Model, feature, range, solvent-domain,
                        and benchmark metadata

scripts/
  experiments/          Benchmark and model-analysis scripts

source_data/
  experiments/          Aggregated and per-seed experiment outputs
  figures/              Numerical source tables used for result auditing

Additional release metadata and data-scope information are provided in:

DATA_AVAILABILITY.md
DATA_PACKAGE_MANIFEST.md
## Data availability

BigSolDB v2.1 is the upstream experimental solubility source.

Because the complete prepared feature matrices and some intermediate quantum-chemistry outputs are substantially larger than the lightweight GitHub release, the repository focuses on scripts, metadata, and analysis-ready source tables.

See DATA_PACKAGE_MANIFEST.md for the relationship between the public repository, prepared feature data, model artifacts, and computational intermediates.

## Reproducibility levels

1. Numerical audit

Use the files under source_data/ to verify reported numerical values and analysis outputs without retraining the models.

2. Prepared-feature rerun

Use the experiment scripts together with the prepared descriptor matrices described in DATA_PACKAGE_MANIFEST.md.

3. Model inference

Use the MICA repository and the corresponding model metadata to reproduce inference with the frozen CAT-Solv ensemble.

4. Descriptor regeneration

Full descriptor regeneration additionally requires:

RDKit molecular processing;
conformer generation;
MMFF94 pre-optimization;
GFN2-xTB;
COSMO surface calculations;
extraction of molecular surface statistics.

Large intermediate calculation files are not distributed through the lightweight GitHub repository.

## Quick setup

conda env create -f environment.yml
conda activate catsolv-repro

Run the release audit with:

python scripts/audit_release.py

The audit checks the packaged release for common reproducibility and distribution issues and regenerates:

metadata/release_file_manifest.tsv
## Related resources

CAT-Solv reproducibility archive

https://doi.org/10.5281/zenodo.22308071

MICA

https://github.com/catmushroomliu-sketch/MICA

CAT-Solv reproducibility repository

https://github.com/catmushroomliu-sketch/CAT-Solv-reproducibility

## Release

Current archived submission release:

v0.1.1-submission

Zenodo record:

https://zenodo.org/records/22308071
