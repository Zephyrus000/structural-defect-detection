# Annotation and semantic audit â€” CUBIT, 2026-11-16

Scope: read-only empirical inspection of the 18 current samples excluded by `crack_v1` for annotation/rasterization errors. The full-corpus re-scan and near-duplicate search are separate audit lanes. Source images, labels, manifests and selection policy were not changed.

## Measured geometry

- The 13 parser flags are exactly 12 collinear polygons and one non-collinear self-intersecting polygon (`train/labels/CR1_2993.txt:1`). The latter has one proper edge crossing (edges 2 and 4), a pixel bbox about 3.0015 Ã— 1.002 px at the bottom-right of a 4500 Ã— 3000 image, and zero signed area.
- Under the installed `pycocotools==2.0.11` COCO polygon rasterizer at each image's original dimensions, all 13 flagged polygons rasterize to zero pixels. This is an observed implementation-specific result; a zero signed/shoelace area does not generally prove a polygon has no rasterized support.
- Five additional parser-valid polygons rasterize to zero: `train/CR1_1446:33`, `train/CR1_1447:31`, `train/CR1_1448:31`, `train/CR1_2777:27`, `train/CR1_2778:23`. Their nonzero continuous geometric areas are about 0.5005â€“1.0033 pixelÂ² and their bboxes are ~1.002 px high, near y=3.999 px. These are rasterization-boundary cases; the probe does not establish that the source annotations are semantically wrong.
- All 18 excluded images contain other valid annotations whose union has positive area. Fifteen are crack-annotated images and three are spalling-annotated images. The 15 excluded crack images contain 280 other instances with nonzero COCO raster support, covering 951,363 pixels in per-image unions (285 instances pass the parser, but five of those have zero raster support). For example, `train/CR1_2993` retains 23 other valid instances with union 123,390 pixels.

`annotation_semantics.json` records per-image dimensions, decisions, every polygon's line/class/geometry/raster support, and valid-instance union area. `annotation_issue_detail.csv` lists the 18 offending polygon instances.

## Contract and ML consequences

1. `scripts/prepare_training_data.py:102` rejects the entire sample when the shared parser returns any issue. `scripts/prepare_training_data.py:134` likewise rejects a sample when any instance rasterizes to zero. The current 18 exclusions therefore follow the explicit conservative preparation policy; this audit has not found an unrecorded skip. They do discard usable annotations elsewhere in those images.
2. `configs/datasets/cubit_cif_crack.yaml:12` keeps only crack-annotated CUBIT images. The next line explicitly assumes that pixels outside annotated defects in those images are background. This is an assumption, not an empirical completeness guarantee. Filtering spalling-only images avoids treating those images as confirmed crack negatives, but does not prove exhaustive crack annotation in retained images or that unmarked other defects are absent.
3. The prepared manifest's `annotation_coverage: full_crack` (`scripts/prepare_training_data.py:199`) represents that adopted assumption. It should not be presented as a source-authored annotation-completeness certification. Retain the explicit assumption in experiment documentation; use reviewed region-level ignore masks or an independently reviewed background protocol if completeness cannot be confirmed.

## Reviewed primary material

- [Official CUBIT-InSeg repository](https://github.com/BenyunZhao/CUBIT-InSeg): release links, paper and explicit dataset CC BY 4.0 license are provided, but the accessible README does not define exhaustive annotation/background coverage.
- [Publisher article record](https://www.sciencedirect.com/science/article/pii/S0926580526002219) and [CUHK institutional article record](https://research.cuhk.edu.hk/en/publications/from-instance-segmentation-to-physical-quantification-high-resolu/): accessible abstract describes 6,996 images, 62,187 instances and two classes. The publisher full article returned HTTP 403 during this audit, so its detailed annotation protocol was not reviewed. These accessible records do not establish exhaustive background coverage.
- [Linked benchmark extension website](https://benyunzhao.github.io/CUBIT-Dataset/): describes a larger 12,500+ image benchmark, additional taxonomy and a different split. Those extension claims must not be transplanted to this local 6,996-image two-class release.
- The article publication license reported by the publisher/institution concerns the paper; it is distinct from the dataset license explicitly granted in the source repository.

## Remediation without changing this audit's inputs

Keep the 18 samples quarantined under the current documented policy for the baseline. For a later dataset version, request corrected annotations from maintainers or manually review the exact line-level cases and record an approved polygon-level correction/ignore policy. Rebuild masks and fingerprints in a new version; do not silently remove individual polygons from the canonical raw release. A source-protocol clarification or independent completeness study is still required before claiming verified negative background pixels or benchmark-equivalent evaluation.

No training, source-label correction or exhaustive human annotation review was performed. Privacy/PII presence was not visually assessed in this targeted geometry lane.
