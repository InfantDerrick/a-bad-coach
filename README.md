# Badminton AI Coach

See `plan.md` for the full roadmap. This repo follows **Phase 0** of that plan.

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

## Phase 0 checks

1. Put a short `.mp4` in `data/raw/` (e.g. rename it `sample.mp4` or pass the path explicitly).
2. Run:

```bash
source .venv/bin/activate
python scripts/phase0_video_basics.py data/raw/your_video.mp4 --no-show
```

Open `data/processed/phase0/` to confirm PNGs were written.

3. Optional: `jupyter lab` in `notebooks/` and repeat the same operations in a notebook for learning.
