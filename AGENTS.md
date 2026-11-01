# Agent handoff

## Immediate task: annotate bad_2 locally

Read `docs/annotate_bad2.md` for complete setup, annotation, and push-back instructions.

The handoff includes **27 unannotated JPEG frames** in `data/court_eval_bad2`: 24 evenly spaced samples and the three frozen-model samples. The first three manifest entries are frame 1861 (~31.1s), frame 9306 (~155.4s), and frame 16750 (~279.7s). Start with those. No model predictions are prefilled.

```bash
git pull --ff-only origin master
git lfs pull
# Activate a Python 3.11/3.12 environment; no GPU required for annotation.
python -m pip install numpy opencv-python
python scripts/court_annotation_server.py --dataset data/court_eval_bad2
```

Open http://127.0.0.1:8765. Resolve all 16 landmarks per image, save, and wait for Saved before changing images. Labels go to `data/court_eval_bad2/annotations/*.json`. Commit and push those labels when done. Do not rerun the sampler on another machine; the prepared frames and manifest are already portable and source-grouped.

## Product direction and dataset separation

Build an offline-first badminton video analysis MVP. Local inference should extract court geometry, players, shuttle, shots, speed, split-step/movement timing, and an evidence-linked dashboard. Do not present derived movement/speed as trustworthy until calibration passes quality checks.

The immediate focus remains reliable court detection under partial, portrait, oblique, and occluded views. Predict named visible intersections and fit the fixed court geometry; do not independently guess four corners.

- `data/court_dataset` is the training dataset: 12 labeled real frames from bad_1 plus 120 labeled synthetic frames across 20 scene groups, 132 total.
- `data/court_eval_bad2` is a **separate held-out source for evaluation**, not training. All 27 images currently require human annotation.
- Do not silently merge bad_2 into training or tune thresholds against it while describing it as an untouched test source. Future model selection against this source makes it validation data; reserve another source for independent final testing.
- The existing seed-7 split holds out five synthetic groups (30 frames), with all 12 real bad_1 frames in training. Its validation score does not establish real-world generalization.
- Add diverse new real sources for training; split by entire source video/venue.
- Licensed professional footage requires recorded provenance/usage rights before use. Do not claim pro-trained accuracy from these clips.

## Annotation convention

The 16 landmarks in `src/pipeline/court_keypoints.py` use far = across the net and near = camera side. Left/right outer means doubles sidelines, not inner singles sidelines. For the net, label its court-plane intersection with the outer sideline, not the top of the net.

Only click exact visible intersections. Use null / Not visible for points outside the frame, occluded, or ambiguous. Never extrapolate labels by hand. The UI requires every landmark resolved before saving and does not autosave on image changes.

## Current model evidence

- Interrupted CPU baseline: `data/models/court_keypoints.pt`; not production-ready.
- Completed 40-epoch GPU baseline: `data/models/court_keypoints_5080.pt`, best held-out synthetic MAE 58.5868 input pixels. It fails the real-frame calibration checks. See `docs/5080_baseline.md`.
- A no-augmentation one-frame diagnostic reaches its nearest-grid error floor (1.2723 input pixels) after 200 updates. The basic model/loss can learn that image. Diagnostic artifacts remain ignored under data/processed.
- Three matched 1000-update, no-augmentation diagnostic runs have real bad_1 training-frame mean input error: real-only 1.58, natural mix 2.61, 50/50 mix 1.63 pixels. All reach 100% visible-label coverage at confidence 0.55, but full-court fits still reject 8/12, 11/12, and 9/12 frames. See `docs/training_comparison.md`.
- Diagnostic checkpoint copies and detailed training metrics are versioned under `data/models/diagnostics/training_comparison/`. Do not promote them to a general-purpose court model.
- All three frozen models fail on sampled unseen bad_2 frames (0-1 confident landmark per frame, all nine fits abstain). No bad_2 training has occurred. Frozen outputs: `data/court_eval_bad2/frozen_predictions.json`; visual evidence: `docs/figures/`. See `docs/bad2_unseen_check.md`.
- No numeric accuracy claim can be made on bad_2 until annotations exist.

## Implemented fixes and next engineering work

- Inference now decodes heatmap coordinates without the erroneous +0.5-cell offset, matching training labels/metrics. Regression test includes letterbox padding.
- Fitter rejects duplicate/collapsed points, near-collinearity, weak consensus, unstable normalized transforms, and full-court horizon crossings. It refits accepted inliers and reports every supplied point's residual.
- Inference writes explicit abstention and removes stale calibration/overlay output when a fit fails. Geometry checks must stay intact.
- `scripts/evaluate_court_keypoints.py` separates annotation-based error and visible-point coverage from fit residuals and abstention. Missing detections are not zero error.
- Remaining localization issue: one output cell at current resolution is 30 pixels in the original portrait video; nearby intersections can merge. Investigate subpixel decoding/higher input or heatmap resolution and inspect court extrapolation before reconnecting movement telemetry.
- Trainer still needs resume support, early stopping, mixed precision, configurable workers, and explicit device warnings. Current trainer starts from scratch; do not overwrite checkpoints blindly.
- Stabilize one court solution across static clips once detection/calibration is reliable. The earlier movement-report calibration remains provisional/diagnostic.

## Local GPU setup and verification

Use Python 3.11/3.12, CUDA-enabled PyTorch from its official installer, then requirements.txt. The verified RTX 5080 environment uses Python 3.12.14 and torch 2.14.1+cu130. Verify CUDA availability and GPU name before training. Annotation alone needs neither torch nor CUDA.

```bash
python -m unittest tests/test_analyze_movement.py tests/test_court_training.py tests/test_commons_scraper.py
```

All **17 tests** pass at handoff. Relevant files: `scripts/train_court_keypoints.py`, `scripts/infer_court_keypoints.py`, `scripts/evaluate_court_keypoints.py`, `scripts/diagnose_court_memorization.py`, `scripts/compare_court_training.py`, and `src/pipeline/court_keypoints.py`.

## Storage and transfer

Git LFS tracks both `*.MOV` and `*.mov`, plus `*.pt`. Run git lfs pull after pulling; bad_2.mov is about 2.08 GB. Extracted annotation JPEGs are committed normally, so the annotator can use them without the original MOV or model weights. Reproducible data/processed outputs and .venv are ignored.

This sandbox blocks the bundled MSYS shell. Native Windows OpenSSH and direct git-lfs.exe were needed for transfer. Verify LFS pointers and object hashes; do not commit materialized binaries as ordinary Git blobs. Normal external-machine Git/Git LFS installation should handle the standard filters.
