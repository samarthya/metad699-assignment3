# AD699 Assignment 3: Classification Trees & Random Forests

## Course: AD699 (Predictive Analytics)

## Project Overview

Predict college enrollment yield using decision trees and random forests. The workflow covers data loading, feature engineering, model training, evaluation, and side-by-side comparison.

## Repository Contents

- `assignment.ipynb`: Primary notebook with commented code and outputs.
- `data/Colleges.csv`: Input dataset (original provided by course).
- `outputs/`: Saved plots (yield distribution, tree visualization, confusion matrices, model comparisons).

## How to Run (with `uv`)

1. Install `uv` if needed: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
2. Sync dependencies from `pyproject.toml`: `uv sync` (creates `.venv` and installs locked deps if present).
3. Activate the environment: `source .venv/bin/activate`.
4. Open `assignment.ipynb` in Jupyter/VS Code and run all cells.

## Outputs

- Plots are saved automatically to `outputs/` when notebooks are run.

## Notes

- Random seed is set to `SEED = 750` for reproducibility.
- Plots and metrics are generated on the provided dataset; rerun notebooks if data changes.
