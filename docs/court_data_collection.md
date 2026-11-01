# Court data collection targets

Goal: drop in an arbitrary phone/tripod clip and recover the **complete** doubles court from partial views via named landmarks + geometric prior.

## Dataset roles

| Path | Role |
| --- | --- |
| `data/court_dataset` | Training only |
| `data/court_eval_bad2` | Held-out evaluation (`bad_2`) |
| Future final-test video | Untouched release check (see `data/held_out_sources.json`) |

Never train on held-out prefixes listed in `data/held_out_sources.json`. The trainer refuses `bad_2*` sources.

## Scale ladder

1. **Now:** real sources `bad_1`, `bad_3`–`bad_11` plus synthetic scenes.
2. **Tier 1:** ~8–12 real venues/phones/angles (in progress).
3. **Tier 2:** ~25–40 real source videos for “works on most amateur clips”.
4. Per source: label ~8–20 frames spanning the clip; skip near-duplicates.

## What to film

- Different gyms, floor colors, lighting, phones
- Portrait and landscape
- Partial courts, oblique angles, occlusion
- Occasional center-only zooms and neighbor courts at edges
- A few no-court close-ups (all landmarks Not visible)

Prefer your own gym footage for core training. Pro/broadcast clips need recorded provenance/usage rights.

### YouTube → random training frames

```bash
python -m pip install yt-dlp
python scripts/prepare_youtube_court_frames.py \
  'https://www.youtube.com/watch?v=VIDEO_ID' \
  --name my_gym_clip \
  --count 12 \
  --mode random \
  --usage-rights own_upload
python scripts/court_annotation_server.py --dataset data/court_dataset
```

This downloads into `data/raw/`, writes provenance to `data/raw/youtube_sources/`, and appends sampled JPEGs to `data/court_dataset` for labeling. Use `--mode uniform` for evenly spaced frames. Only use videos you have rights to train on.

## Labeling rule

Click only exact visible named intersections. Mark everything else Not visible. Never extrapolate by hand — the fitter completes off-screen edges from the 6.10×13.40 m template.

## After each data bump

```bash
python scripts/report_court_dataset.py --dataset data/court_dataset
python scripts/train_court_keypoints.py --dataset data/court_dataset --output data/models/court_keypoints_m1_vN.pt --device mps
python scripts/evaluate_court_keypoints.py \
  --dataset data/court_eval_bad2 --annotation-glob '*.json' \
  --checkpoint data/models/court_keypoints_m1_vN.pt --device mps \
  --output data/processed/bad2_evaluation_m1_vN.json \
  --paired-held-out-output data/processed/bad2_release_gate_m1_vN.json
```
