# CUBIT-InSeg exploratory data audit — 25 October 2026

This report describes the local YOLO-polygon release at `data/CUBIT-InSeg`. Source images, labels, and official split assignments were not changed. No model was trained or evaluated.

## Full-corpus findings

| Official partition | Images / matching labels | Crack-annotated images | Spalling-annotated images | Polygons flagged for zero signed/shoelace area |
|---|---:|---:|---:|---:|
| Training | 5,596 | 2,535 | 3,061 | 8 |
| Validation | 699 | 299 | 400 | 1 |
| Test | 701 | 320 | 381 | 4 |
| Total | 6,996 | 3,154 | 3,842 | 13 |

- All 6,996 image headers were readable. There were no missing/orphan label files or empty label files. This does not establish full-image decoding integrity or exhaustive annotation coverage.
- The local `data.yaml` defines only `0: crack` and `1: spalling`; corrosion is unsupported by these labels. Every image had annotations for exactly one of these two classes. This is an annotation observation, not proof that the other defect is absent from the photograph.
- There are 51,865 crack polygons and 10,322 spalling polygons, including the 13 flagged polygons. The 13 flags occur in 13 files and mean zero signed/shoelace area under the audit's numeric tolerance, not necessarily zero rasterized pixels. Twelve polygons are collinear; `train/labels/CR1_2993.txt`, line 1, is a non-collinear self-intersecting bow-tie whose signed lobes cancel. Review and record correction/exclusion decisions before full segmentation training. The audit retains the original exclusion policy for these flags and makes no claim that every flagged polygon rasterizes to an empty mask.
- Original resolutions are 4,500 × 3,000 (4,377 images), 4,800 × 3,200 (2,523), and 3,000 × 4,500 (96). These are 13.5–15.36 megapixel images; resizing/tiling requires checks that thin cracks survive and predictions map correctly to the original coordinates.

## Confirmed exact-image duplication across official partitions

SHA256 was computed over every image file (18,830,559,406 bytes total) and every paired label file. There are 589 exact-image duplicate groups, each containing two image records. This represents 1,178 image records and 6,407 unique image-file hashes. All 589 duplicate pairs also have byte-identical label files, and all are spalling-annotated. No complementary/different annotations were observed within these exact pairs.

| Split boundary | Exact duplicate pairs | Affected image records |
|---|---:|---:|
| Training–validation | 99 | 198 |
| Training–test | 110 | 220 |
| Validation–test | 13 | 26 |
| Total crossing partitions | 222 | 444 |

The 444 affected records comprise 209 training, 112 validation, and 123 test images. Official partitions therefore contain confirmed exact-image leakage. Preserve the official release for provenance, but establish a reviewed internal duplicate/group-aware evaluation protocol before model training. Keeping verified duplicate pairs in one partition or deduplicating identical image/label pairs are possible decisions to document. Do not silently rewrite official benchmark results. SHA256 does not exclude re-encoded copies, near-duplicates, related crops, or images from the same structure/session. The configured layout supplies no verified grouping CSV.

## Training-only raster-mask sample

Uniform sampling without replacement selected 64 readable paired training records with seed 42 (35 crack-annotated and 29 spalling-annotated). Valid polygons were scaled to original image resolution and rasterized using COCO per-class unions. No flagged polygons occurred in this sample.

- Across these 64 sampled images, pixel-weighted annotated crack coverage was **0.378%**, spalling coverage **4.683%**, and their union **5.061%**.
- These estimates describe this sample only. They are not full-corpus pixel proportions. Unannotated pixels have not independently been confirmed as defect-free background.
- The small crack coverage motivates imbalance-aware losses and checks on thin-defect preservation. The figures do not establish optimal loss weights, resolution, or model accuracy.

## Figures and evidence

- `eda_split_class_incidence.png` / `.svg`: full-corpus official split counts and class annotation incidence (10 × 3.3 inches, PNG at 250 dpi).
- `eda_resolution_mask_occupancy.png` / `.svg`: full-corpus resolution counts and per-image occupancy in the 64-image training-only mask sample. Occupancy boxes include zero coverage for an unannotated class.
- `training_overlay_01.png` through `_04.png`: original-image and annotation-overlay pairs for the first four sampled training records in sorted order. These are annotation previews, not predictions; the four previews are all crack-annotated.
- `image_inventory.csv`: one record per source image, paths, dimensions, class polygon counts, validation flags, and source image/label hashes.
- `annotation_issues.csv`: exact file and line of every geometric flag.
- `exact_image_duplicate_members.csv`: reproducibly generated group membership and annotation-hash evidence.
- `exact_image_duplicate_pairs.csv`: convenient equivalent pair-level comparison.
- `training_mask_sample.csv`: selected image identifiers and original-resolution class pixel measurements.
- `eda_summary.json`: full machine-readable summary, inventory fingerprint, software versions, exact duplicate groups, sampling method, and limitations.

Suggested caption: **Local CUBIT-InSeg audit.** Panel A inventories all 6,996 official image records; panel B shows the proportion with at least one annotation of each class. Class incidence does not establish annotation completeness. Separate SHA256 checks detected 222 exact-image duplicate pairs crossing official partitions, requiring a reviewed internal evaluation protocol before training.

## Reproduction

From the project root, choose a new output directory because the full audit refuses to overwrite an existing directory:

```powershell
.\.venv\Scripts\python.exe scripts\eda_cubit_inseg.py --root data\CUBIT-InSeg --output reports\cubit_eda_new --mask-samples 64 --overlay-samples 4 --seed 42
python scripts\eda_cubit_inseg.py --render-only --output reports\cubit_eda_new
.\.venv\Scripts\python.exe scripts\eda_cubit_inseg.py --derive-only --output reports\cubit_eda_new
```

The scan uses the project environment (numpy, Pillow, PyYAML, pycocotools). Figure-only mode requires matplotlib; the existing system Python supplied it for this run. No packages or dependencies were installed. Python compilation, Ruff checks, and integrity assertions passed for inventory totals, unique record identifiers, image/label pairing, annotation class exclusivity, fraction ranges, and deterministic training-only sampling. Figures and one overlay were visually inspected.

GitNexus was queried before adding the standalone script. Impact was UNKNOWN because its new path/symbols were absent from the current index; confirming text searches found no existing application references. Existing application code was not edited. The untracked script was not covered by the git-diff-based detect_changes result (which reported zero); this is not represented as a clean graph regression check. No commit was created.
