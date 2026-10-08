# GLAM

**Glaucoma Longitudinal Analysis Model** — code for predicting visual field (VF) progression from longitudinal perimetry, with an explicit **time-split** formulation that separates the observation window from the progression label.

Public repository: [github.com/taiaburbd/glam](https://github.com/taiaburbd/glam)

## Overview

GLAM combines a per-visit encoder, bidirectional LSTM, clinical branch, and attention fusion to estimate subsequent mean deviation (MD) and visual field index (VFI) progression rates from serial Humphrey 24-2 total-deviation maps.

The forecast pipeline in this repo implements the **honest task** used in our reanalysis work:

| Formulation | Input | Label |
|-------------|--------|--------|
| Leaky (illustrative) | Full visit sequence | OLS slope over the **same** visits |
| Time-split (default here) | First **3** visits | OLS slope of **visit 4 onward** |

Under the time split, reported deep-learning accuracy collapses unless the label window is kept separate from the input window. Scripts here reproduce preprocessing, training, baselines (ridge, gradient boosting, cohort mean), and figures for that setting.

Manuscript sources live locally under `paper/` (not tracked in git). Cohort files live under `data/` (not tracked in git).

## Requirements

- Python **3.11+**
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

```bash
uv sync
# optional: copy API keys for W&B / Anthropic helpers
cp .env.example .env
```

## Data

Download public cohorts yourself and place them under `data/raw/`:

| Dataset | URL | Notes |
|---------|-----|--------|
| UWHVF | [uw-biomedical-ml/uwhvf](https://github.com/uw-biomedical-ml/uwhvf) | `data/raw/uwhvf/alldata.json` for the forecast pipeline |
| GRAPE | [Figshare](https://doi.org/10.6084/m9.figshare.c.6406319) | Used by `scripts/external_validation_grape.py` |

Processed tensors, splits, and checkpoints stay on disk only (`data/`, `checkpoints/` are gitignored).

## Quick start (time-split forecast)

From the repository root:

```bash
# 1. Build forecast splits (3 visits in, label from future visits)
uv run python scripts/preprocess_uwhvf_forecast.py

# 2. Train GLAM + run baselines; writes metrics and figures
uv run python scripts/run_forecast_experiment.py

# 3. Optional: leakage comparison figure and GRAPE external check
uv run python scripts/make_leakage_figure.py
uv run python scripts/external_validation_grape.py
```

Outputs:

- `checkpoints/glam_forecast.pt` — trained weights  
- `reports/forecast/` — JSON metrics and PNG figures  
- `data/forecast/splits/` — patient-level train/val/test IDs  

## Legacy training path

The original full-sequence Hydra training entry point is still available:

```bash
uv run python scripts/preprocess_uwhvf.py   # expects data under data/raw/
uv run python scripts/train.py              # configs/config.yaml
```

Prefer the forecast scripts above for analyses aligned with time-split validation.

## Repository layout

```
glam/
├── configs/           # Hydra configuration
├── scripts/           # Preprocessing, training, forecast experiment, figures
├── src/
│   ├── data/          # Loaders and preprocessing helpers
│   ├── models/        # Encoder, BiLSTM, fusion, GLAM head
│   ├── losses/        # Regression losses
│   └── utils/         # Metrics, reproducibility
├── reports/           # Experiment outputs (forecast metrics tracked in git)
├── notebooks/         # Exploratory analysis
├── presentation/      # Slide text for talks
├── tests/
├── data/              # Local only (gitignored)
└── paper/             # Local manuscripts (gitignored)
```

## Citation

If you use this code, cite the associated progression / leakage reanalysis work and the public datasets (UWHVF, GRAPE). Author contact: [taiabur@visioneyebd.org](mailto:taiabur@visioneyebd.org).

## License

See repository defaults; dataset licences follow UWHVF (CC BY 4.0) and GRAPE terms on Figshare.
