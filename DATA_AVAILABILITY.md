# Data Availability

This release follows a lightweight repository plus external-large-data model:

1. **GitHub/Zenodo archived repository** for analysis code, documentation, lightweight source tables, final manuscript/SI files, final figure files, and release metadata.
2. **External large-data archive, when needed** for full prepared-feature reruns involving complete prepared descriptor matrices, row-level prediction tables, trained-model artifacts, and checksums that are too large or too specialized for the GitHub repository.

The CAT-Solv reproducibility repository is archived at:

<https://doi.org/10.5281/zenodo.20576521>

The MICA software package is maintained separately at:

<https://github.com/catmushroomliu-sketch/MICA>

## Draft manuscript Data Availability statement

The data underlying this study are available in the article, its Supporting Information, and the public repositories associated with this work. Source tables used to audit manuscript figures, analysis scripts, final figure files, and release metadata are available from the CAT-Solv reproducibility repository at <https://github.com/catmushroomliu-sketch/CAT-Solv-reproducibility> and archived on Zenodo at <https://doi.org/10.5281/zenodo.20576521>. The MICA software package is available at <https://github.com/catmushroomliu-sketch/MICA>. Data derived from public sources retain the terms and citation requirements of the corresponding upstream resources.

## Current release identifiers

- GitHub repository: <https://github.com/catmushroomliu-sketch/CAT-Solv-reproducibility>
- GitHub release tag: `v0.1.1-submission`
- Zenodo DOI: <https://doi.org/10.5281/zenodo.20576521>
- Release date: 2026-06-07

## External large-data archive requirements

If a separate large-data archive is prepared for full model reruns, it should include:

- Persistent identifier.
- File-level checksums.
- Machine-readable CSV, JSON, Parquet, or compressed table files where possible.
- Clear license and upstream-data notes.
- Version tags matching the manuscript/repository release.
- Metadata describing software versions and data-generation dates.

## Upstream data note

The public release distinguishes author-generated derived tables from upstream solubility records. If raw third-party data cannot be redistributed under author-controlled terms, the release should provide source instructions, processing scripts, and checksums rather than repackaging restricted files.
