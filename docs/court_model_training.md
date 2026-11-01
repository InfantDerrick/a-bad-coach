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
  --epochs 40 \
  --batch-size 8 \
  --device mps
```

Defaults are 384×672 input with subpixel peak metrics, early stopping, optional `--resume` / `--amp` (CUDA) / `--workers`. Held-out prefixes from `data/held_out_sources.json` (`bad_2`) are refused. The default split is grouped by source video. `--allow-frame-split` exists solely for plumbing checks and must not be used for accuracy claims.

Harder synthetic (center crops + edge-neighbor ghosts):

```bash
python scripts/generate_synthetic_courts.py \
  --dataset data/court_dataset \
  --samples 240 --scenes 40 --hard-fraction 0.5 \
  --filename-prefix synthetic-hard --seed 23
```

## Predict and fit the full court

```bash
python scripts/infer_court_keypoints.py data/raw/bad_1.MOV \
  --checkpoint data/models/court_keypoints.pt \
  --frame-index 0 \
  --output-dir data/processed/court_prediction

python scripts/infer_court_video.py data/raw/bad_1.MOV \
  --checkpoint data/models/court_keypoints.pt \
  --stride 30 \
  --output-dir data/processed/court_video_bad1
```

Single-frame inference selects a primary court hypothesis when edge landmarks are ambiguous. Video inference adds temporal lock so the court does not flicker every frame. Calibrations remain `model_prediction_unverified` until manually verified; movement metrics stay gated.

## Release requirements

- Fine-tune using real frames from multiple phones, venues, floor colors, lighting conditions, portrait/landscape views, and partial-court compositions.
- Keep complete venues/videos out of training.
- Measure keypoint error, homography reprojection error, player-position error in metres, and failure/abstention rate.
- Reject predictions with fewer than four non-collinear high-confidence landmarks, poor RANSAC consensus, or high reprojection error.
- Stabilize a valid calibration across a static clip; do not redetect the court independently on every frame.

## Geometry quality gates and annotation evaluation

The fitter conservatively rejects duplicate/near-duplicate image locations (within 0.1% of landmark span), nearly collinear points, less than 60% consensus (minimum four), numerically unstable normalized transforms, and full-court outlines crossing the projective horizon. It refits accepted inliers and reports residuals for every supplied landmark. These are engineering checks, not validated accuracy guarantees. Inference writes `status: abstained` with a reason and removes stale calibration/overlay files from its output directory when geometry fails.

Compare predictions against labeled images separately from the model's internal fit residuals:

```bash
python scripts/evaluate_court_keypoints.py \
  --dataset data/court_eval_bad2 --annotation-glob '*.json' \
  --checkpoint data/models/court_keypoints.pt --device mps \
  --output data/processed/bad2_evaluation.json \
  --paired-held-out-output data/processed/bad2_release_gate.json
```

The report includes per-landmark source-pixel error, mean error normalized by image diagonal, visible-point coverage, missed visible points, predictions on invisible points, fit diagnostics, abstention rate, and a `release_gate` block. Missing predictions reduce coverage and are not treated as zero error. A successful evaluator exit means the report was produced, not that predictions passed accuracy checks.

The RTX 5080 baseline abstains on all 12 real frames at confidence 0.55. Frame 0 still has 475.77 px average matched-landmark error. Eleven manual annotation fits pass; frame 144 abstains because extrapolating its fit to the full court crosses the projective horizon, despite a small visible-point residual. Revisit that fit before using it for complete court calibration. These real frames are in training, so this evaluation does not establish generalization.
