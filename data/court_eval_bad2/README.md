# bad_2: held-out court annotation set

27 extracted frames, intentionally separate from the training dataset. No labels are prefilled. Follow [the annotation instructions](../../docs/annotate_bad2.md). Start with the first three manifest entries (the frozen baseline samples).

Run from repository root:

```bash
python scripts/court_annotation_server.py --dataset data/court_eval_bad2
```

Do not train on this dataset while it is being used for source-held-out evaluation.
