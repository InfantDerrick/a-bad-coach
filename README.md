# Badminton AI Coach

An offline-first prototype for turning a fixed-camera badminton clip into an auditable local movement report. See `plan.md` for the product and validation roadmap.

## Python environment (0.1)

Use **Python 3.11 or 3.12** for this project if you can. System Python 3.14+ may not install **MediaPipe** or **PyTorch** wheels yet.

**Option A — `pyenv` (recommended)**

```bash
pyenv install 3.12.8
cd /path/to/a-bad-coach
pyenv local 3.12.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Option B — Fedora `python3.12` package (if available)**

```bash
sudo dnf install python3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Layout (0.2)

- `notebooks/` — experiments  
- `src/pipeline/` — ML pipeline (later)  
- `src/api/`, `src/frontend/` — reserved for later phases  
- `data/raw/` — put rally videos here  
- `data/processed/` — frames, exports  
- `data/models/` — weights (later)  
- `scripts/` — small utilities (e.g. Phase 0 video demo)

## First offline movement report

The current vertical slice uses an explicit court calibration and a frame-zero player bounding box. It produces:

- `report.json` — versioned metadata, quality gates, and aggregate metrics;
- `trajectory.csv` — per-frame pixels, court coordinates, and speed;
- `movement_overlay.mp4` — player box, foot point, court boundary, and trail;
- `court_path.png` — top-down path and six-zone occupancy;
- `report.html` — a completely local dashboard.

Run the checked-in sample:

```bash
python scripts/analyze_movement.py \
  data/raw/bad_1.MOV \
  --calibration data/config/sample_calibration.json \
  --player-seed data/config/sample_player_seed.json \
  --out-dir data/processed/movement_report
```

Then open `data/processed/movement_report/report.html` in a browser.

The sample calibration and player seed are deliberately marked `provisional`. The dashboard metrics are engineering diagnostics—not coaching conclusions—until the overlay is inspected and both inputs are verified. Shuttle speed is unavailable for this 60 FPS clip under the current quality gate.

Run the unit tests with:

```bash
python -m unittest tests/test_analyze_movement.py
```

## Training court detection

Court detection is trained around a canonical court template so it can align a partial view and extrapolate off-screen boundaries. The repository now includes:

- a grouped real-frame sampler;
- an offline browser annotation UI for 16 named court intersections;
- a synthetic perspective/crop/occlusion generator;
- a heatmap-model training script with source-video-separated validation;
- partial-landmark RANSAC fitting and inference.

See [docs/court_model_training.md](docs/court_model_training.md) for the workflow. The checked-in/generated smoke-test checkpoint is only a plumbing test and is not a usable real-video model.

## Phase 0 video checks

1. Put a short `.mp4` in `data/raw/` (e.g. rename it `sample.mp4` or pass the path explicitly).
2. Run:

```bash
source .venv/bin/activate
python scripts/phase0_video_basics.py data/raw/your_video.mp4 --no-show
```

Open `data/processed/phase0/` to confirm PNGs were written.

3. Optional: `jupyter lab` in `notebooks/` and repeat the same operations in a notebook for learning.

## Annotate the new held-out video

See [docs/annotate_bad2.md](docs/annotate_bad2.md) for the prepared 27-frame annotation set, local startup, and label commit instructions.
