# AD699 Assignment 3: Classification Trees & Random Forests

Understand how tree-based methods work, their strengths/weaknesses, and when to use ensemble methods.

## Scenario 

You're hired as a Data Analyst at the Department of Education. Your boss says: "We need to predict which colleges will have high enrollment rates (yield) to allocate marketing funds effectively."

Your Mission: Use historical data to build a prediction model.

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

## What is a decision tree?


| Concept | Analogy | Key Takeaway |
|:---|:---|:---|
| Single Decision Tree | The Single Expert: You ask one hiring manager to decide on a candidate. They create a flow chart of rules: Is their GPA > 3.5? If yes, is their experience > 3 years? If yes, Hire. | Fast and Simple. But if their rules are too rigid, they might miss a great candidate (or hire a bad one) if the candidate's profile is slightly unusual. |
| Random Forest| The Hiring Committee: You gather 100 hiring managers. Each one gets a slightly different resume and focuses on a different set of skills. They all vote. You go with the majority decision. | Slower, but Far More Reliable. The combined wisdom cancels out the individual mistakes or biases of any single manager. |


## Classification problem

Shape - 776x18 

INPUT: College characteristics (features)
PROCESS: Machine learning model
OUTPUT: HIGH or LOW (class label)