# Data Availability

This release follows a two-record model:

1. **GitHub** for analysis code, documentation, lightweight source tables, final manuscript/SI files, final figure files, and release metadata.
2. **Zenodo or another DOI-backed repository** for large prepared descriptor matrices, row-level prediction tables, trained-model artifacts, and checksums that are too large or too specialized for the GitHub repository.

The MICA software package is maintained separately at:

<https://github.com/catmushroomliu-sketch/MICA>

## Draft manuscript Data Availability statement

The data underlying this study are available in the article, its Supporting Information, and the public repositories associated with this work. Source tables used to audit manuscript figures, analysis scripts, final figure files, and release metadata are available from the CAT-Solv reproducibility repository at <https://github.com/catmushroomliu-sketch/CAT-Solv-reproducibility>. Large prepared descriptor matrices, row-level prediction tables, trained-model artifacts, and checksums will be archived in a DOI-backed data record before public release. The MICA software package is available at <https://github.com/catmushroomliu-sketch/MICA>. Data derived from public sources retain the terms and citation requirements of the corresponding upstream resources.

## Fields to update after public release

- GitHub release tag.
- Zenodo concept DOI or version DOI for this repository.
- DOI for the external large-data archive, if released as a separate record.
- Final release date in `CITATION.cff`.

## Repository requirements

The DOI-backed data archive should include:

- Persistent identifier.
- File-level checksums.
- Machine-readable CSV, JSON, Parquet, or compressed table files where possible.
- Clear license and upstream-data notes.
- Version tags matching the manuscript/repository release.
- Metadata describing software versions and data-generation dates.

## Upstream data note

The public release distinguishes author-generated derived tables from upstream solubility records. If raw third-party data cannot be redistributed under author-controlled terms, the release should provide source instructions, processing scripts, and checksums rather than repackaging restricted files.
