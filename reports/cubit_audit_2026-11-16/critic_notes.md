# Final draft critic — contracts and factual coverage

Verdict: PASS after two source-anchor corrections. No unresolved concrete factual, arithmetic, local-link or unsafe-recommendation defect found in the final draft.

Reviewed FINDINGS.md and audit_summary.json against contracts_evidence.json, near_duplicates/summary.json, near_duplicates/visual_review.json, ml_sampling.json, and annotation_semantics.json. This was a bounded read-only draft review, not a repeat of the image scans.

Verified:
- Raw/prepared counts 6,996 / 3,139 and splits 2,523 / 299 / 317.
- Candidate totals 2,238 = 589 exact-file pairs + 1,649 nonexact candidates; 20 prepared cross-split candidates = 12 train-test + 8 train-validation, involving 37 images.
- All-20 thumbnail visual-review scope is distinguished from four strong visual examples; physical group IDs and all-pair native-resolution adjudication remain unverified.
- Blank-crop arithmetic 1,879 / 2,560 = 73.3984375%; quartile results and the limited bootstrap interpretation match the ML evidence. No accuracy defect is inferred from these rates.
- Exclusion arithmetic 13 parser flags + 5 rasterization cases + 3,839 spalling-only = 3,857; 6,996 - 3,857 = 3,139. The 15 crack / 3 spalling quarantined-image breakdown and 280 other positive-raster annotations / 951,363 union pixels match annotation_semantics.json.
- Passing exact-content checks are not presented as proof of structure/session independence or exhaustive annotations. Prior full prepared validation is explicitly disclosed as reused.
- Proposed changes are separate versioned protocols and expert review, not silent editing of source labels or official benchmark assignments. Spalling-only images are not relabeled automatically as negatives.
- Every local artifact link in FINDINGS.md exists.

Corrected and verified source anchors: prepare_training_data.py:200 for hash-valued source_image_id; prepare_training_data.py:199 for full_crack assignment. Earlier draft offsets :202 and :201 were incorrect.

Scope note: “valid” annotations in the quarantine discussion should be read as parser/raster validity; the body describes positive COCO raster support and does not establish expert semantic correctness.

Skipped: new raw hashing, image decoding, model runs and independent visual adjudication; these belong to the main audit evidence.
Residual risk remains the draft's stated annotation-coverage, physical-grouping and heuristic near-duplicate limitations.
