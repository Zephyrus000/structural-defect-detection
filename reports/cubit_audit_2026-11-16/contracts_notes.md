# Contracts and lineage audit — CUBIT, 2026-11-16

Scope: read-only checks of the current CSV/YAML/JSON artifacts and pipeline code. This lane did not repeat the full image decode or raw file hash scan; those checks are owned by the main audit.

## Verified integrity

The root and prepared manifests are byte-identical and match the recorded SHA256. The validation report, local data.yaml, and mapping config also match their recorded hashes. All 3,139 manifest rows have unique sample IDs and correspond to keep decisions in the raw inventory. Splits are 2,523 train / 299 validation / 317 test. Source paths, mask paths, split assignments, dimensions, and source_image_id were checked against inventory by the earlier staged review and the current source-ID check.

The raw inventory contains 589 exact file-hash duplicate groups, including 222 groups crossing splits (444 records). All 444 cross-split records are excluded as spalling-only; none enters the prepared manifest. The prepared validation report records no same-split exact decoded-image duplicates. This is not evidence about near-duplicates.

## Findings and interpretation

### HIGH: structure/session independence is not established

Evidence: zero of 3,139 rows has structure_id or capture_session_id. Every source_image_id is exactly the decoded RGB image hash. Nevertheless validation.json reports samples_without_group_metadata=0. `scripts/prepare_training_data.py:200` generates the hash-valued source_image_id; `src/structural_defect_detection/data/manifest.py:145` accepts that field alongside physical structure and capture-session IDs, and lines 148–153 count it as group metadata.

Consequence: zero missing groups must not be read as a clean structure/session leakage audit. Related views, crops, re-encodings or separate frames of one facade can still cross train/validation/test. The report's own leakage_limit caveat partially mitigates presentation, but no real group separation is established.

Remedy: report coverage separately for structure_id, capture_session_id and content identifiers; obtain real grouping metadata or adjudicate related-image groups before claiming structure-generalization results. Keep exact-content checks.

### HIGH: full_crack coverage is an explicit assumption, not independently verified

Evidence: `configs/datasets/cubit_cif_crack.yaml:13` assumes pixels outside annotated defects are background; `scripts/prepare_training_data.py:199` writes annotation_coverage=full_crack. The mapping artifact itself lists annotation completeness/background assumption as an outstanding audit. All retained samples have positive crack pixels and total ignored pixels are zero. Class naming in data.yaml and a class-0 polygon do not establish exhaustive annotation coverage of the rest of the image.

Consequence: unlabeled cracks or other defects could be trained/evaluated as background. This is a data-quality risk rather than a demonstrated mislabeled pixel; integrity validation cannot resolve it.

Remedy: confirm the release's annotation protocol and conduct stratified expert overlay review. If coverage is incomplete, provide explicit supervised/ignore regions; do not turn unverified spalling-only images into negative crack labels.

### MEDIUM: evaluation is conditional on crack-annotated images

Evidence: every retained whole image is crack-positive (minimum 8,858 crack pixels/image). The pipeline excludes 3,839 spalling-only samples and 18 invalid annotation/rasterization samples. `configs/datasets/cubit_cif_crack.yaml:12` documents the crack-positive inclusion policy.

Consequence: the prepared set can measure segmentation on crack-annotated scenes and background pixels inside those scenes. It cannot establish image-level false-positive burden on independently confirmed crack-free or spalling-only scenes, and its scores are not scores on the full official 6,996-image benchmark.

Remedy: describe crack_v1 as a filtered internal protocol; add independently adjudicated crack-free/other-defect controls for deployment evaluation. Preserve the conservative exclusion unless annotation coverage is confirmed.

## Exclusions

Current keep set: 3,139. Exclusions: 13 invalid_source annotations, 5 invalid_mask instances rasterizing to zero pixels, 3,839 cubit_no_confirmed_crack_coverage samples. The raw labels remain available and reasons are recorded. Invalid-source rows leave class/pixel-count fields blank; downstream summaries must not convert those blanks into a claim of negative annotations.

## Readiness within this lane

The contract and provenance checks pass for the filtered crack_v1 dataset. Controlled pipeline/ablation experiments are supportable when the coverage assumption is declared. Structure-independent generalization and deployment false-positive claims need additional evidence. Main-audit near-duplicate and visual-review results may impose further restrictions.

Skipped in this lane: full image/mask re-decoding, new raw hashing, visual annotation adjudication and near-duplicate search.
Residual risk: coverage and real structure/session isolation remain unproven despite the passing exact-content validation.
