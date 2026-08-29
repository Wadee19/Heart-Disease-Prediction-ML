# Model Card

## Purpose

Educational binary-classification project demonstrating a reproducible, leakage-safe machine-learning workflow on the UCI Statlog (Heart) dataset.

## Data

270 observations, 13 input attributes, and a binary heart-disease target. The dataset is small and historical.

## Evaluation design

The workflow creates a stratified 80/20 holdout before preprocessing or model selection. Cross-validation and tuning run only on the 216-row training partition; the 54-row holdout is used once for final evaluation.

## Verified evaluation

The selected model is a class-weighted Random Forest chosen by 5-fold stratified cross-validation ROC-AUC.

- Random Forest CV ROC-AUC: **0.913 ± 0.043**
- Logistic Regression CV ROC-AUC: **0.913 ± 0.042**
- Holdout ROC-AUC: **0.882**
- Holdout recall: **0.833**
- Holdout precision: **0.741**
- Holdout F1: **0.784**
- Holdout accuracy: **0.796**

Results are reproduced by GitHub Actions from a clean environment and exported as `artifacts/metrics.json`.

## Limitations

No external or prospective validation. The dataset is too small and historical to support clinical generalization. There is no calibration study, fairness/subgroup analysis, clinical utility assessment, or evaluation on a contemporary external cohort.

## Intended use

Portfolio and education only. Not for diagnosis, treatment, risk communication to patients, or medical decision-making.
