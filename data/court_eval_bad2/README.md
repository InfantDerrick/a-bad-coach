# bad_2: held-out court annotation set

27 extracted and annotated frames, intentionally separate from the training dataset. Listed in `data/held_out_sources.json`. Follow [the annotation instructions](../../docs/annotate_bad2.md) if labels need edits.

```bash
python scripts/court_annotation_server.py --dataset data/court_eval_bad2
python scripts/evaluate_court_keypoints.py \
  --dataset data/court_eval_bad2 --annotation-glob '*.json' \
  --checkpoint data/models/court_keypoints_m1_v3.pt --device mps \
  --output data/processed/bad2_evaluation.json \
  --paired-held-out-output data/processed/bad2_release_gate.json
```

Do not train on this dataset while it is being used for source-held-out evaluation.
