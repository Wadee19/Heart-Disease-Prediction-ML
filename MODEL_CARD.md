# Model Card

## Purpose
Educational binary-classification project demonstrating a reproducible, leakage-safe machine-learning workflow on the UCI Statlog (Heart) dataset.

## Data
270 observations, 13 input attributes, and a binary heart-disease target. The dataset is small and historical.

## Evaluation
The workflow creates a stratified holdout before preprocessing or model selection. Cross-validation and tuning run only on the training partition; the holdout is used once for final evaluation.

## Limitations
No external or prospective validation. Performance may not generalize across populations or healthcare settings.

## Intended use
Portfolio and education only. Not for diagnosis, treatment, or medical decision-making.
