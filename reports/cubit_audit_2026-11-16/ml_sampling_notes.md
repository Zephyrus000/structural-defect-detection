# CUBIT ML sampling audit — 2026-11-16

This lane reads the current prepared CUBIT manifest and training masks only. It does not train a model, change the dataset, or use test masks for a model probe.

## Measured supervision exposure

- [MEDIUM, sampling efficiency] `src/structural_defect_detection/data/dataset.py:55` uniformly chooses one 256×256 crop per image per epoch. There is no positive-pixel condition. The configured 20 epochs were reproduced for 128 train images sampled as 32 images without replacement from each rank quartile of full-image annotation occupancy (selection seed 42). Of 2,560 crops, **1,879 contain zero annotated crack pixels (73.3984%)**, while 681 contain at least one. Population weighting by quartile size gives **73.4038%**, with a 2,000-replicate stratified image-cluster bootstrap interval **71.2182–75.5920%**. This interval describes sampling variation within the current image population and does not cover annotation uncertainty or new-site generalization.
- Foreground exposure varies by annotation-density quartile: blank-crop rates are 89.0625%, 77.9688%, 66.7188%, and 59.8438%. Median crop foreground count is zero; the 75th percentile is 179.75 of 65,536 pixels. Prepared train masks contain 0.7695% annotated foreground pixels in aggregate; this denominator excludes spalling-only/invalid images and differs from raw-corpus occupancy estimates.
- Over all 20 configured epoch crops, 5/128 sampled images never expose an annotated crack pixel. The median fraction of each sampled image's annotated crack pixels ever presented is 9.4554%; the mean is 9.9579%. Median unique source-image area seen is 9.2804%. This is an exposure measurement, **not proof of a model accuracy defect**. Consider a positive-aware crop protocol with explicitly retained background crops, then compare it in a controlled pilot before an expensive long run. Do not silently change sampling while comparing experiments.

## Correctness checks and active transforms

The independently reproduced seed/crop calculation was checked against the real `SegmentationDataset.transform` on 8 real image/mask pairs at epochs 0 and 19 (16 cases). Crop coordinates, post-flip target arrays, foreground counts, and valid counts match exactly. All sampled raw-mask foreground counts equal the preparation inventory.

`dataset.py:35` seeds by training seed, epoch, source name, and sample ID, so worker scheduling does not alter these crops. Active augmentation is horizontal/vertical flips (`dataset.py:67`). Image normalization uses mean [0.485,0.456,0.406] and std [0.229,0.224,0.225]. The active mode is crop, and evaluation is overlapping native-resolution tiles; these modes do not downsize thin annotations. The resize branch and model-internal thin-feature retention were not tested.

## Evaluation and population limits

- All 3,139 retained images have positive foreground in the preparation inventory: train 2,523, validation 299, test 317. There are no empty-ground-truth images in this manifest. Background pixels still contribute to segmentation metrics, but this protocol does not measure image-level false alarms on independently verified clean/spalling-only scenes. Spalling-only images must not be blindly relabeled as crack-negative: crack annotation completeness is unproven. Add a separately audited negative/ignore protocol if that evaluation is needed.
- [Operational cost] Current 256 tiles with overlap 32 imply **83,720 tile windows per validation pass** over 299 validation images. Compare this with 2,523 train crops per epoch (`dataset.py:104`). This is a deterministic workload count, not a measured runtime. Time a controlled pilot before committing to a long run.
- No verified structure or capture-session grouping is supplied; foreground-density stratification cannot establish site-level independence. No test label was used to choose a crop policy or tune a parameter in this lane.

## Artifacts

- `ml_sampling.json`: protocol, exact denominators, per-epoch/stratum summaries, quantiles and limitations.
- `ml_sampling_images.csv`: selected 128 train IDs and 20-epoch exposure per image.
- `ml_sampling_crops.csv`: all 2,560 train crop coordinates/counts.

No dataset, manifest, config, or source file was changed. No model was trained; coverage completeness and model accuracy remain unverified.
