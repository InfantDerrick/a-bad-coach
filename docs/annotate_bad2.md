# Annotate bad_2 on your local machine

## Pull and start

From your existing repository checkout:

```bash
git pull --ff-only origin master
git lfs pull
```

The 27 extracted JPEGs are committed directly, so annotation does not need the video decoded or a GPU. `git lfs pull` also retrieves the 2.08 GB source video and the model files. Install Git LFS if `git lfs` is unavailable.

Use an existing Python 3.11/3.12 environment, or create one:

```bash
python -m venv .venv
```

Activate on Windows PowerShell with `.venv\Scripts\Activate.ps1`; on Linux/macOS use `source .venv/bin/activate`. Use `python3` instead of `python` to create the environment if that is how Python is installed.

Only these packages are needed for annotation; CUDA/PyTorch are unnecessary:

```bash
python -m pip install numpy opencv-python
python scripts/court_annotation_server.py --dataset data/court_eval_bad2
```

Open **http://127.0.0.1:8765**. Keep the terminal running. If port 8765 is occupied, add `--port 8766` and open that port instead.

## What to label

There are **27 unannotated frames** from one video in `data/court_eval_bad2`: 24 evenly spaced frames plus the three previously tested frames. The first three entries are frame 1861 (~31.1s), 9306 (~155.4s), and 16750 (~279.7s); label those first. They intentionally have no predictions prefilled.

For each of the 16 named landmarks:

- Click the exact center of the visible intersection of the named lines.
- Mark **Not visible / off-screen** if it is off-screen, blocked, or ambiguous. Do not extrapolate.
- Far = across the net; near = camera side. Left/right outer = doubles sidelines, not singles sidelines.
- Net floor line = the net's court-plane location at the outer sideline, not the top of the net.
- Resolve all 16 entries, then click **Save annotation**. Wait for **Saved** before changing images. Changing images does not automatically save.

Shortcuts: J/K or down/up selects landmarks; V marks not visible; C clears; S saves; [ and ] changes images.

Labels are saved locally to `data/court_eval_bad2/annotations/*.json`. These are the files to commit after annotation:

```bash
git add data/court_eval_bad2/annotations
git commit -m "Annotate held-out bad_2 court frames"
git push origin master
```

The UI can be stopped with Ctrl+C and resumed using the same command. Existing saved annotations load automatically.

## Keep this source out of training

`data/court_dataset` remains the training dataset. `data/court_eval_bad2` is a separate source-held-out evaluation dataset. Do not merge them or train on bad_2 while comparing models against it.

The frozen nine pre-annotation predictions are saved in `data/court_eval_bad2/frozen_predictions.json`. Diagnostic checkpoint copies and the earlier training comparison are committed under `data/models/diagnostics/training_comparison/`. Images of the earlier results are in `docs/figures/`.

Once annotations are available, a GPU-equipped machine can evaluate the balanced checkpoint with:

```bash
python scripts/evaluate_court_keypoints.py --dataset data/court_eval_bad2 --annotation-glob '*.json' --checkpoint data/models/diagnostics/training_comparison/balanced_mix.pt --device cuda --output data/processed/bad2_evaluation.json
```

The evaluator uses completed annotation JSON files only, so the first three labels are enough for an initial quantitative check. Reports separate prediction accuracy/coverage from court-fit acceptance. Tuning models against this source makes it validation data; another independent source will be needed for final testing.
