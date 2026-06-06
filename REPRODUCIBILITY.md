# Reproducibility Guide

## Analysis levels

The release supports three levels of reproducibility.

### Level 1: Figure and table audit

Use the CSV/JSON files in `source_data/` to verify manuscript numbers and inspect the data underlying each figure. Final figure files are staged in `figures/`, and manuscript/SI files are staged in `docs/`. This level is lightweight and is suitable for GitHub.

### Level 2: Model rerun from prepared feature matrices

Use the scripts in `scripts/experiments/` with the prepared BigSolDB/xTB/COSMO feature matrices. These matrices are too large for the code repository and should be distributed through the external data archive described in `DATA_PACKAGE_MANIFEST.md`.

### Level 3: End-to-end descriptor generation

End-to-end regeneration of xTB/COSMO descriptors and pair-xTB controls requires the original descriptor-generation pipeline, external quantum-chemistry software, and large intermediate files. This level should be documented in the data archive, but it is not expected to run from the GitHub repository alone.

## Expected raw data layout

If the full data archive is unpacked under this repository, use:

```text
data/raw/
  BigSolDB_features_v1_common.csv
  BigSolDB_features_v2_common.csv
  BigSolDB_features_v3_common.csv
  ood_predictions_all_seeds.csv
```

Alternatively set:

```bash
export CATSOLV_RAW_DATA_ROOT=/path/to/unpacked/data/raw
export CATSOLV_RELEASE_ROOT=/path/to/CAT-Solv-reproducibility
```

## Main experiment scripts

- `run_physchem_cross_20seed.py`: 2D/COSMO descriptor expansion and fixed-OOD evaluation.
- `run_topk_matched_feature_count.py`: validation-ranked feature selection and matched top-k tests.
- `run_learner_robustness_20seed.py`: robustness across XGBoost, Random Forest, Extra Trees, and HistGradientBoosting.
- `run_iid_evaluation.py`: random-split IID reference using the same non-OOD training pool.
- `validate_topk_experiment.py`: consistency checks for the top-k feature-selection experiment.

## Reproducibility cautions

- The 20 seeds share the same fixed OOD solvent set; seed-level consistency should be interpreted as robustness under resampling, not as 20 independent experimental replications.
- MICA throughput benchmarks refer to prepared descriptor CSV inference. They do not include upstream xTB/COSMO descriptor generation time.
- Some SI panels and complete experiment reruns require optional raw files that are intentionally not stored in the lightweight GitHub repository. Figure artwork/layout scripts are not part of this public release; source data and final figure PDFs are provided instead. Run `python scripts/audit_release.py` before tagging a release.
- The manuscript and Supporting Information TeX files in `docs/` have release-local figure paths. They are included for transparency and source review; the journal submission files remain the authoritative submission artifacts.
