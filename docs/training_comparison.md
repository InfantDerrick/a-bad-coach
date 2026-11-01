# Real/synthetic court training comparison

## Result

The model can fit all 12 real frames from the existing clip. Synthetic data does not prevent fitting those frames when training runs longer, and balanced sampling substantially improves real-frame learning per update over the original mix. These experiments do not measure performance on new real videos.

The inference decoder's extra half-heatmap-cell offset was removed. A regression test checks the entire inference decode path with nonzero letterbox padding: source (35,50) maps to heatmap (32,40) and back to (35,50). The diagnostic script now records zero decoding offset. Earlier reports retain historical results made with the old decoder.

## Controlled setup

Command (from the repository root):

```powershell
.venv/Scripts/python.exe -u scripts/compare_court_training.py --steps 1000 --device cuda --figure ../outputs/court_training_comparison.png
```

- CUDA / RTX 5080, seed 7; same random model initialization for each arm.
- Same architecture, loss, AdamW learning rate 0.0003, 288x512 input, stride 4.
- Exactly 1000 optimizer updates per arm, batch size 16, sampling with replacement.
- Augmentation disabled for all arms to isolate learning on the fixed examples.
- The original source-group split is retained: 12 real frames + 90 synthetic training frames; 30 synthetic frames from five entirely held-out scene groups.
- Real-only draws all 16 examples from the real pool; natural mix draws uniformly from the 102-example training pool; balanced mix draws 8 real and 8 synthetic examples per batch.
- All 12 real evaluation frames are training examples from ONE video. No held-out real-world accuracy claim is possible.

## Final-step measurements

Mean raw input error includes every annotated visible point, regardless of confidence. Coverage uses the fixed 0.55 threshold; missing points are not treated as zero error.

| Setup | Real-frame mean input error | Mean source-image error on confident visible predictions | Visible coverage | Rejected court fits | Held-out synthetic mean input error | Real examples sampled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real only | 1.58 px | 11.83 px | 100% | 8/12 | 259.37 px | 16000 |
| Original ratio (natural mix) | 2.61 px | 19.60 px | 100% | 11/12 | 34.99 px | 1864 |
| 50/50 mix | 1.63 px | 12.24 px | 100% | 9/12 | 40.84 px | 8000 |

The original 40-epoch checkpoint, evaluated with the corrected decoder, has 61.11 input-pixel error over all real visible labels, 55.8% visible coverage, and 12/12 rejected fits. Its training used brightness augmentation and about 280 updates, so it is contextual evidence, not another controlled arm. Do not attribute the entire difference to sampling alone.

During the controlled natural-mix run, real-frame error fell from 82.43 input pixels at step 200 to 21.24 at 400, 9.48 at 600, and 2.61 at 1000. The balanced run reached 2.22 by step 600. This supports a training-budget/exposure explanation; it does not isolate synthetic interference from reduced real-example exposure. Exposure-matched comparisons and multiple seeds would be needed for a stronger causal claim.

## Remaining calibration problem

Training-fit accuracy is much better, but the full-court fitter still rejects many predictions. The real-only run has five duplicate-location failures, two horizon failures, and one consensus failure. One output heatmap cell is 4 input pixels, or 30 pixels in the original 2160x3840 clip. Nearby labeled intersections can land on the same coarse grid position even when the heatmap peaks are close to their individual labels. Small landmark residuals also do not establish stable extrapolation of off-screen boundaries.

Keep geometry checks intact and all three checkpoints diagnostic-only. Next investigate subpixel peak decoding and/or higher input/output resolution, compare per-landmark errors against the annotations, and inspect full-court projections. Then evaluate different real source videos before reconnecting trustworthy movement metrics.

## Saved artifacts

- `data/processed/training_comparison/comparison.json`: histories, predictions, coverage, per-point errors, rejection reasons, source split, and exposure counts.
- `data/processed/training_comparison/original_baseline_corrected.json`: contextual baseline evaluation.
- `data/processed/training_comparison/{real_only,natural_mix,balanced_mix}.pt`: final diagnostic checkpoints. Original model files remain intact.
- `../outputs/court_training_comparison.png`: annotated frame-0 comparison.

All 17 tests pass; diff whitespace check passes with LFS filters disabled for this environment.
