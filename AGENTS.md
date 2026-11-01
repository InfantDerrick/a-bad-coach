# Agent handoff

## Product direction

Build an offline-first badminton video analysis MVP. A user uploads a clip, local GPU inference extracts court geometry, players, shuttle, shots, speed, split-step and movement timing, then produces a report/dashboard. Do not present derived speed or movement metrics as trustworthy until court calibration passes quality checks.

The immediate focus is reliable court detection from partial, portrait, oblique, and occluded views. The court has a fixed geometric prior; predict named visible intersections and fit the complete court instead of independently guessing four corners.

## Current repository state

- The completed court dataset is in `data/court_dataset` and contains 132 samples:
  - 12 manually annotated frames from `data/raw/bad_1.MOV`.
  - 120 generated partial-view samples across 20 synthetic scene groups.
- All 132 samples have complete, schema-valid annotations.
- Real-frame all-point homography fits are geometrically consistent (roughly 0.75-7.4 px RMSE on the original 2160x3840 frames).
- Wikimedia Commons samples were removed because search results were not reliable enough.
- Git LFS tracks `*.MOV` and `*.pt`. Run `git lfs pull` after cloning.
- Reproducible files under `data/processed/` are intentionally ignored.
- The local annotation UI is `scripts/court_annotation_server.py`; it includes keyboard shortcuts and partial-visibility labeling.
- The movement report remains diagnostic-only because its earlier calibration was provisional.

## Annotation convention

The 16 landmarks in `src/pipeline/court_keypoints.py` use `far` for the side across the net and `near` for the camera side. `left/right outer` means the outer doubles sidelines, not the inner singles sidelines.

Only click an exact visible line intersection. Use `null` / **Not visible** when it is outside the frame, occluded, or ambiguous. Never extrapolate a missing point by hand.

## Court model

Relevant files:

- `src/pipeline/court_keypoints.py`: canonical court dimensions, annotation validation, homography fitting.
- `scripts/train_court_keypoints.py`: small heatmap network and grouped source split.
- `scripts/infer_court_keypoints.py`: frame inference, confidence filtering, homography and overlay output.
- `scripts/generate_synthetic_courts.py`: synthetic geometry generation.
- `scripts/prepare_court_dataset.py`: real-frame extraction.
- `docs/court_model_training.md`: workflow and release requirements.

`data/models/court_keypoints.pt` is only an interrupted CPU baseline. It was saved around epoch 11 of a planned 40-epoch run at 288x512 input. Its held-out synthetic validation MAE is 69.37 input pixels, so it is not production-ready. The training script currently starts from scratch and does not resume from this checkpoint.

With seed 7 and a grouped split, all 12 real frames are in training; five complete synthetic scene groups are held out. This prevents synthetic scene leakage, but it does not measure real-world generalization because all real frames come from one clip.

## Continue on the RTX 5080 PC

Use Python 3.11 or 3.12. Install a CUDA-enabled PyTorch build from the official PyTorch installer, then install the remaining requirements.

PowerShell outline:

```powershell
git lfs pull
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the current CUDA-enabled torch build from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The verification command must report `True` and the RTX 5080 before training.

Run the first complete baseline:

```powershell
python scripts/train_court_keypoints.py `
  --dataset data/court_dataset `
  --output data/models/court_keypoints_5080.pt `
  --epochs 40 `
  --batch-size 16 `
  --device cuda
```

Keep the existing checkpoint until the new run has been evaluated. Do not overwrite it blindly.

Then infer an annotated real frame and inspect the overlay:

```powershell
python scripts/infer_court_keypoints.py data/raw/bad_1.MOV `
  --checkpoint data/models/court_keypoints_5080.pt `
  --frame-index 0 `
  --output-dir data/processed/court_prediction_5080 `
  --device cuda
```

Compare predictions against the frame-0 annotation, not just the synthetic validation number. The model should abstain when it has fewer than four non-collinear, high-confidence points or poor geometric consensus.

## Known technical follow-ups

1. Add epoch/resume support, unbuffered progress, early stopping, mixed precision, configurable data-loader workers, and explicit GPU/CPU warnings to the trainer.
2. Improve homography fitting. OpenCV RANSAC once selected a poor subset even though an all-point fit gave about 4.8 px RMSE. Refit with all accepted inliers and report per-landmark residuals before declaring labels or predictions wrong.
3. Add a real-frame evaluation command that reports normalized keypoint error, homography reprojection error, inlier count, and failure/abstention rate.
4. Add real videos from different phones, venues, floor colors, lighting, landscape/portrait framing, and partial-court angles. Split by entire source video/venue.
5. Fine-tune/evaluate on licensed professional-match footage only after provenance and usage rights are recorded. Do not claim “pro-trained” accuracy from the current single amateur clip.
6. Once calibration is reliable, stabilize one court solution across a static clip rather than redetecting independently on every frame, then reconnect player/shuttle/shot telemetry.

## Verification

Current test command:

```bash
python -m unittest tests/test_analyze_movement.py tests/test_court_training.py tests/test_commons_scraper.py
```

At handoff, all 10 tests pass.
