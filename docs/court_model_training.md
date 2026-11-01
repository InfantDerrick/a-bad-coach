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
