# Agent handoff

## Product goal

Drop in a phone/tripod badminton clip and recover the **complete** doubles court even when only part is visible. Named visible intersections + the fixed 6.10×13.40 m geometric prior extrapolate off-screen edges. Neighbor courts at frame edges auto-resolve to one primary court. Movement/speed metrics stay gated until calibration is verified.

Read [docs/court_data_collection.md](docs/court_data_collection.md) for data targets and [docs/court_model_training.md](docs/court_model_training.md) for train/eval workflow.

## Dataset separation (enforced)

- `data/court_dataset` — training only (`bad_1`, `bad_3`–`bad_18`, `yt_sonic_*`, synthetic). Currently 640 annotated frames / 26 real sources.
- `data/court_eval_bad2` — held-out eval for `bad_2` (annotated).
- `data/held_out_sources.json` — forbidden training prefixes (`bad_2`) and a slot for a future final-test video.
- `scripts/train_court_keypoints.py` refuses held-out source prefixes.
- YouTube Sonic Smash clips carry `usage_rights: fair_use_review_needed` in `data/raw/youtube_sources/`; do not claim pro-licensed training from them.

Do not merge `bad_2` into training or tune thresholds against it while describing it as untouched. After model selection on `bad_2`, reserve another independent video for final testing.

## Annotation convention

The 16 landmarks in `src/pipeline/court_keypoints.py` use far = across the net and near = camera side. Left/right outer = doubles sidelines. Net = court-plane intersection with the outer sideline, not the top of the net.

Only click exact visible intersections. Use null / Not visible when outside the frame, occluded, or ambiguous. Never extrapolate by hand.

## Current evidence

- Baseline diagnostics memorized `bad_1` landmarks but failed on unseen `bad_2`.
- After labeling `bad_3`–`bad_11` and training `court_keypoints_m1_v2.pt` / `m1_v3.pt`, held-out `bad_2` improved (coverage ~55%, some fits) but is not release-ready.
- `bad_12`–`bad_18` plus 9 Sonic Smash YouTube sources are fully annotated (178 new frames); retrain before re-scoring `bad_2`.
- Localization still limited by heatmap cell size; trainer defaults are now 384×672 with subpixel peak decode.
- Primary-court selection, temporal lock (`scripts/infer_court_video.py`), and release-gate summaries live in `src/pipeline/court_tracking.py`.

## Implementation map

- `src/pipeline/court_keypoints.py` — schema, validation, homography fit
- `src/pipeline/court_tracking.py` — subpixel decode, primary picker, temporal lock, release gate, held-out guard
- `scripts/train_court_keypoints.py` — resume, early stopping, AMP (CUDA), workers, device warnings, visibility-aware loss
- `scripts/infer_court_keypoints.py` / `scripts/infer_court_video.py` — single-frame and locked clip inference
- `scripts/evaluate_court_keypoints.py` — annotation metrics + `release_gate` / `metrics_trustworthy`
- `scripts/generate_synthetic_courts.py` — harder center-crop and edge-neighbor modes
- `scripts/report_court_dataset.py` — source diversity report
- `scripts/analyze_movement.py` — keeps model calibrations provisional; metrics stay diagnostic until verified

## Train / eval on Apple MPS or CUDA

```bash
python scripts/report_court_dataset.py
python scripts/train_court_keypoints.py \
  --dataset data/court_dataset \
  --output data/models/court_keypoints_m1_v3.pt \
  --epochs 40 --batch-size 8 --device mps
python scripts/evaluate_court_keypoints.py \
  --dataset data/court_eval_bad2 --annotation-glob '*.json' \
  --checkpoint data/models/court_keypoints_m1_v3.pt --device mps \
  --output data/processed/bad2_evaluation_m1_v3.json \
  --paired-held-out-output data/processed/bad2_release_gate_m1_v3.json
```

Release gate thresholds (calibratable frames with ≥4 labeled visibles): coverage ≥0.8, MAE ≤80 px, abstention ≤0.4, fit acceptance ≥0.55. Do not present movement metrics as trustworthy until calibration status is `verified`.

## Verification

```bash
python -m unittest tests/test_analyze_movement.py tests/test_court_training.py tests/test_commons_scraper.py
```
