# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle competition project for the **Higgs Boson Machine Learning Challenge**. The goal is binary classification: predict whether a physics event is a Higgs boson signal (`s`/1) or background noise (`b`/0) based on particle collision features. This is a university course project (Taller de Aprendizaje Automatico 2026).

## Environment Setup

```bash
# Requires Kaggle credentials as env vars: KAGGLE_USERNAME, KAGGLE_API_TOKEN
# On Codespaces/Linux:
chmod +x setup.sh && ./setup.sh
# On Windows (Git Bash):
bash setup.sh
```

Originally designed for GitHub Codespaces (Kaggle secrets go in repo Settings > Secrets and Variables > Codespaces). Also works locally on Windows with Git Bash and `unzip` available.

Python venv is at `.venv/`. Dependencies: pandas, scikit-learn, numpy, matplotlib, seaborn, sweetviz, kaggle.

## Data

- `data/training.csv` — labeled training set (~250k events), indexed by `EventId`. Label column: `Label` ('s'/'b'). Weight column: `Weight`.
- `data/test.csv` — unlabeled test set for Kaggle submission.
- Missing values are encoded as **-999.0** (not NaN).
- `PRI_jet_num` (0-3) determines which other features are meaningful; many features are -999 when jet count is low.
- All features are numeric (30 columns of physics-derived quantities).
- Data directory is gitignored.

## Architecture

Notebook-based workflow in `notebooks/`:

1. **data_exploration.ipynb** — EDA: distributions, correlation matrix, mutual information, missing value analysis, eta-phi event visualizations. Uses `rand_state = 42` and 50k samples for plotting performance.
2. **data_preprocessing.ipynb** — Feature engineering and preprocessing (in progress).
3. **model_selection.ipynb** — Model comparison and selection (in progress).

## Key Conventions

- Random state fixed at `42` for reproducibility.
- Labels are converted to binary: `(Label == 's').astype(int)`.
- `Weight` column is dropped from features before modeling.
- Language: code comments and notebook text are in Spanish.
