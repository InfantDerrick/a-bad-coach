# RTX 5080 baseline — 2026-10-31

## Environment and validation

- Repository starting point: `3a4a7ee`.
- Python 3.12.14 in `.venv` (created from the bundled runtime).
- PyTorch 2.14.1+cu130; CUDA available; NVIDIA GeForce RTX 5080.
- A CUDA matrix multiplication completed successfully.
- Installed repository requirements after CUDA PyTorch.
- All 10 tests passed using the command in AGENTS.md.

## Training

```powershell
.venv/Scripts/python.exe -u scripts/train_court_keypoints.py --dataset data/court_dataset --output data/models/court_keypoints_5080.pt --epochs 40 --batch-size 16 --device cuda
```

All 40 epochs completed. Best synthetic validation MAE: **58.5868 input pixels**, at epoch 40. The original CPU checkpoint remains intact. History is in `data/models/court_keypoints_5080.history.json`.

The original checkpoint's recorded MAE was 69.37. This run improved that synthetic score, but batch size/optimization differ and this is not evidence of usable court accuracy.

## Real-frame check: failed

Frame 0 is in training; this is a fit diagnostic, not held-out real-world validation.

- Seven annotated visible landmarks.
- Six predictions pass the default 0.55 confidence threshold.
- Six matched predictions: average error **475.77 source-image pixels**, or **10.80% of image diagonal**.
- `far_short_service_right`, `net_right`, and `near_short_service_right` collapse to the same point (1485, 1935).
- The existing homography fitter nevertheless reports five inliers and 2.51 px reprojection RMSE.
- Visual inspection confirms incorrect landmarks and an unusable court outline.

Do not promote this model or its calibration to trustworthy movement reporting. Low inlier residual alone can admit degenerate solutions.

Local diagnostic files (ignored/reproducible):

- `data/processed/court_prediction_5080/court_overlay.png`
- `data/processed/court_prediction_5080/comparison.png` (blue circles: annotations; red crosses: predictions)
- `data/processed/court_prediction_5080/real_frame_evaluation.json`
- `data/processed/court_prediction_5080/prediction.json`

## Next engineering work

1. Add regression coverage for collapsed/duplicate landmark predictions and reject degenerate homographies. Check distinct image locations, geometric spread, matrix conditioning, and projected court plausibility; retain per-point residuals and refit accepted inliers.
2. Add reusable real-frame evaluation with annotated visibility coverage and normalized keypoint error, alongside fit/abstention diagnostics. Never substitute reprojection residual for ground-truth prediction accuracy.
3. Diagnose training localization failure and evaluate alternative training settings with fixed grouped splits. Keep the baseline as a reference.
4. Collect diverse real source videos for genuine held-out evaluation as described in AGENTS.md.

## Windows Git note

The sandbox blocks the bundled MSYS shell. Native Windows SSH and the absolute `git-lfs.exe` path were needed. LFS objects were fetched and all three original asset SHA-256 hashes verified before materializing the working files. Git currently shows the materialized original LFS files as modified despite matching object hashes; do not commit those as content changes without checking pointer normalization.

## One-frame memorization diagnostic (2026-10-31)

Ran the existing model and weighted heatmap loss from random seed 7 on frame 0 only, with augmentation disabled, using CUDA. No production checkpoint was overwritten.

```powershell
.venv/Scripts/python.exe -u scripts/diagnose_court_memorization.py --steps 1000 --device cuda --figure ../outputs/one_frame_memorization.png
```

- Independently checked the resized label coordinates and Gaussian target peaks: passed for this frame.
- After 200 optimization steps, all seven annotated visible landmarks exceeded confidence 0.55.
- Best mean peak error: 1.2723 input pixels; maximum: 2.0198 input pixels.
- The nearest possible integer heatmap-grid locations have the same 1.2723-pixel mean error. This diagnostic reaches the grid quantization floor for this example.
- Ran all 1000 steps; no improvement below that floor.
- Existing inference adds a half-cell offset (`+0.5` before multiplying by stride) that the training targets/metric do not use. This adds [2,2] input pixels. With the unmodified inference path, mean source-frame error is 26.14 pixels; raw heatmap-grid error corresponds to about 9.54 source pixels. This coordinate-convention mismatch should be reconciled with a regression test before quoting deployment accuracy. It does not explain the much larger baseline localization failure by itself.
- Diagnostic checkpoint and history live in ignored `data/processed/one_frame_diagnostic/`. This checkpoint memorizes one image and must not replace the general court model.

Interpretation: the current architecture/loss can fit this particular example and its tensor labels align. This does not prove every image/label transform is correct or establish performance on other frames. The mixed-data baseline failure remains unexplained. Next isolate learning on the 12 real frames, then compare controlled synthetic/real mixtures and update counts. Also align training and inference coordinate decoding. All 16 existing tests still pass.
