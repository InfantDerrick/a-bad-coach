# Frozen-model check on bad_2.mov

The new source video was tested before any training or annotation changes. It remains outside the court training manifest. Results are preliminary qualitative checks of three sampled frames, not a complete-video accuracy estimate.

Video: `data/raw/bad_2.mov`, 2,077,342,712 bytes; decoded dimensions 2160x3840; nominal OpenCV FPS 59.88835; 18,613 frames; duration about 310.795 seconds. The scene has a wood floor and a different background/camera framing from bad_1.

Frames were selected at fixed 10%, 50%, and 90% positions before inference: 1861 (31.074s), 9306 (155.389s), and 16750 (279.687s).

The frozen final checkpoints in `data/processed/training_comparison` were run through the corrected inference decoder on CUDA at confidence 0.55. No thresholds were tuned against this video.

| Model | Confident landmarks at 31.1s | At 155.4s | At 279.7s | Court fits |
| --- | ---: | ---: | ---: | --- |
| Real only | 0 | 0 | 0 | All abstained |
| Natural mix | 1 | 0 | 1 | All abstained |
| Balanced mix | 1 | 1 | 1 | All abstained |

All nine predictions abstained because fewer than four landmarks passed the threshold. The balanced model's only confident channel is `net_right` (landmark 9); visual inspection shows its guesses misplaced on the floor rather than the intended net-floor/sideline intersection. Confidence is not verified accuracy.

No ground-truth annotations exist for these samples, so no numeric keypoint accuracy/coverage is claimed. These observations show failure on the sampled unseen-source frames; they do not measure the whole clip or isolate the cause of the domain change.

Saved pre-training evidence:

- `data/processed/bad_2_unseen/video.json`
- `data/processed/bad_2_unseen/frame_{1861,9306,16750}.jpg`
- `data/processed/bad_2_unseen/predictions.json` (all model outputs, confidence values, checkpoint paths, sample indices, and rejection reasons)
- `../outputs/bad2_unseen_predictions.png`
- `../outputs/bad2_model_comparison.png`

Keep bad_2 out of training for now. Next label visible named intersections in these fixed samples to obtain a quantitative source-held-out benchmark. Further model selection on bad_2 would make it validation data, so reserve another source video for an independent final test. Collect additional diverse sources for training instead of reporting memorized-frame performance as generalization.
