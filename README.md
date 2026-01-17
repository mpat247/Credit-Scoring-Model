# Credit Scoring Model

## Overview
This project builds a credit scoring pipeline and studies how model performance changes under data poisoning attacks and defenses. It focuses on high-level experimentation rather than publishing results.

## Repository Layout
- `CreditScoring.ipynb`: End-to-end notebook that ties the workflow together.
- `src/`: Reusable modules for preprocessing, training, attacks, and defenses.
- `data/`: Input datasets used for experiments.
- `models/`: Saved model artifacts.
- `results/`: Generated plots, tables, and reports from experiments.

## High-Level Workflow
1. **Data preparation**: load data, clean column names, handle missing values, encode categorical variables, scale features, and select top features.
2. **Model training**: train baseline classifiers (Random Forest and XGBoost).
3. **Evaluation**: compute standard metrics (accuracy, AUC, F1) and generate reports/plots.
4. **Data poisoning scenarios**: simulate label flipping, feature manipulation, backdoor injections, and synthetic data injections.
5. **Defenses**: apply anomaly detection (Isolation Forest, Local Outlier Factor, Autoencoder) and retrain models on filtered data.
6. **Artifacts**: store metrics, plots, and trained models under `results/` and `models/`.
