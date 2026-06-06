# Data Package Manifest

This file describes the intended contents of the external data archive that accompanies the code repository.

## Required for figure/table audit

These files are already staged in `source_data/`:

- Figure source CSV/JSON files under `source_data/figures/`.
- Aggregated experiment summaries under `source_data/experiments/`.
- MICA model metadata snapshots under `metadata/mica/`.
- Final manuscript and Supporting Information files under `docs/`.
- Final manuscript and selected Supporting Information figures under `figures/`.

Figure layout scripts are not required for the public reproducibility package. The release provides final figure files plus the source CSV/JSON tables needed to audit plotted values.

## Required for full model rerun

Place these files in the external data archive under `data/raw/`:

```text
BigSolDB_features_v1_common.csv
BigSolDB_features_v2_common.csv
BigSolDB_features_v3_common.csv
ood_predictions_all_seeds.csv
selected_features_by_seed.csv
validation_shap_rankings_all_seeds.csv
```

Minimum metadata for each file:

- File name.
- SHA256 checksum.
- Number of rows and columns.
- Column dictionary or pointer to feature specification.
- Data-generation script or upstream source.
- License or redistribution note.

## Optional high-volume artifacts

These files are useful for deep review but should remain outside the lightweight GitHub repository:

- Full per-seed prediction tables by learner and feature layer.
- Pair-xTB descriptor tables and negative-control intermediate outputs.
- xTB/COSMO calculation logs and failed-job manifests.
- Trained XGBoost seed-model bundles used by MICA.
- MICA benchmark input/output CSVs.

## Do not archive

- Local absolute paths.
- macOS cache files.
- Notebook checkpoints.
- Draft manuscript PDFs unrelated to the submitted version.
- AI editing scratch files.
- Raw private local GUI exports.
- Historical figure variants not cited by the submitted manuscript or Supporting Information.
