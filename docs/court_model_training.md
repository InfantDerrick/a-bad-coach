# Court model training

The court detector uses a **canonical geometric court prior**, not four independent corner guesses. A model predicts named visible line intersections; RANSAC aligns those observations to the known 6.10 m × 13.40 m court. Once aligned, the complete court can be projected even when boundaries fall outside the image.

## Why partial views are trainable

The court has fixed topology and dimensions. The same baseline, service lines, center lines, sidelines, and net relationship appears under every valid camera homography. Training therefore combines:

1. synthetic courts with random perspective, camera roll, cropping, blur, lighting, and occlusion;
2. real fixed-camera frames labeled with the named intersections that are actually visible;
3. grouped evaluation on entirely unseen videos, venues, and angles.

Synthetic data teaches geometry but is not sufficient evidence of real-world accuracy. Real target-domain footage is required for fine-tuning and release validation.

## Prepare real frames

```bash
python scripts/prepare_court_dataset.py data/raw/bad_1.MOV \
  --dataset data/court_dataset \
  --count 24
```

Add videos from different courts and camera positions with the same command. Each source video remains a separate group for validation.

## Label visible intersections

```bash
python scripts/court_annotation_server.py --dataset data/court_dataset
```

Open `http://127.0.0.1:8765`. For each named intersection, click its image position or explicitly mark it invisible. Do not guess off-screen locations; the geometric fitter extrapolates those later.

For a static clip, label several frames to measure annotator consistency, but do not mistake near-duplicate frames for dataset diversity.

## Generate geometry pretraining data

```bash
python scripts/generate_synthetic_courts.py \
  --dataset data/court_dataset \
  --samples 5000 \
  --scenes 500
```

## Train

```bash
python scripts/train_court_keypoints.py \
  --dataset data/court_dataset \
  --output data/models/court_keypoints.pt \
  --epochs 50 \
  --batch-size 4
```

The default split is grouped by source video. Training refuses to make a normal validation split from only one real video. `--allow-frame-split` exists solely for plumbing checks and must not be used for accuracy claims.

## Predict and fit the full court

```bash
python scripts/infer_court_keypoints.py data/raw/bad_1.MOV \
  --checkpoint data/models/court_keypoints.pt \
  --frame-index 0 \
  --output-dir data/processed/court_prediction
```

The prediction contains visible keypoint confidence, RANSAC inliers, reprojection error, a calibration file, and an overlay whose outer boundary may extend beyond the image.

## Release requirements

- Fine-tune using real frames from multiple phones, venues, floor colors, lighting conditions, portrait/landscape views, and partial-court compositions.
- Keep complete venues/videos out of training.
- Measure keypoint error, homography reprojection error, player-position error in metres, and failure/abstention rate.
- Reject predictions with fewer than four non-collinear high-confidence landmarks, poor RANSAC consensus, or high reprojection error.
- Stabilize a valid calibration across a static clip; do not redetect the court independently on every frame.

## Geometry quality gates and annotation evaluation

The fitter conservatively rejects duplicate/near-duplicate image locations (within 0.1% of landmark span), nearly collinear points, less than 60% consensus (minimum four), numerically unstable normalized transforms, and full-court outlines crossing the projective horizon. It refits accepted inliers and reports residuals for every supplied landmark. These are engineering checks, not validated accuracy guarantees. Inference writes `status: abstained` with a reason and removes stale calibration/overlay files from its output directory when geometry fails.

Compare predictions against labeled images separately from the model's internal fit residuals:

```powershell
.venv/Scripts/python.exe scripts/evaluate_court_keypoints.py --dataset data/court_dataset --annotation-glob 'bad_1*.json' --checkpoint data/models/court_keypoints_5080.pt --device cuda --output data/processed/court_prediction_5080/real_evaluation.json
```

The report includes per-landmark source-pixel error, mean error normalized by image diagonal, visible-point coverage, missed visible points, predictions on invisible points, fit diagnostics, and abstention rate. Missing predictions reduce coverage and are not treated as zero error. A successful evaluator exit means the report was produced, not that predictions passed accuracy checks.

The RTX 5080 baseline abstains on all 12 real frames at confidence 0.55. Frame 0 still has 475.77 px average matched-landmark error. Eleven manual annotation fits pass; frame 144 abstains because extrapolating its fit to the full court crosses the projective horizon, despite a small visible-point residual. Revisit that fit before using it for complete court calibration. These real frames are in training, so this evaluation does not establish generalization.
