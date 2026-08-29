# Heart Disease Prediction — Leakage-Safe ML Pipeline

A compact, reproducible machine-learning project for binary heart-disease classification using the **UCI Statlog (Heart)** dataset.

The original notebook has been preserved for project history, while this V2 adds a production-style training pipeline, reproducible evaluation, automated tests, CI, and an optional Streamlit demo.

> **Educational project only.** This repository is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.

## Why this V2 exists

The first notebook version applied SMOTE before the train/test split. That allows information from synthetic samples derived from the full dataset to influence the held-out set and can overstate performance. V2 fixes that methodology:

```text
raw data
   ↓
stratified train/test split
   ↓
training set only
   ↓
preprocessing + cross-validation + model selection
   ↓
untouched final test set
```

Because the dataset is small and only moderately imbalanced, V2 does **not** force SMOTE. Instead, candidate models use class weighting and are compared using stratified cross-validation.

## Dataset

The included CSV matches the **UCI Statlog (Heart)** dataset structure:

- 270 observations
- 13 predictive attributes
- binary target: `Presence` / `Absence`

Source/reference: https://archive.ics.uci.edu/dataset/145/statlog+heart

## Models

V2 compares Logistic Regression and Random Forest. Model selection is based on **5-fold stratified CV ROC-AUC** on the training split only. The selected model is then evaluated once on the untouched test set.

Reported metrics: Accuracy, Precision, Recall, F1, ROC-AUC, and cross-validation ROC-AUC mean/std.

No V2 performance number is hard-coded into this README. Run the pipeline to reproduce the result on your environment.

## Project structure

```text
.
├── Heart-Disease-Prediction-ML.ipynb   # legacy exploratory notebook
├── Heart_Disease_Prediction.csv        # dataset
├── src/
│   └── train.py                        # reproducible training/evaluation pipeline
├── tests/
│   └── test_pipeline.py                # pipeline/data smoke tests
├── app.py                              # optional Streamlit demo
├── .github/workflows/tests.yml         # CI
├── requirements.txt
└── README.md
```

## Reproduce locally

```bash
python -m venv .venv
pip install -r requirements.txt
python src/train.py
```

The run writes `artifacts/model.joblib` and `artifacts/metrics.json`.

To run the demo after training:

```bash
streamlit run app.py
```

## Run tests

```bash
pytest -q
```

## Methodological notes

- The test set is split **before** preprocessing or model selection.
- Preprocessing lives inside scikit-learn pipelines, so transformations are fitted only on each training fold.
- Hyperparameters are selected using stratified cross-validation.
- The final test set is evaluated once after model selection.
- A fixed random seed is used for reproducibility.
- With only 270 rows, uncertainty matters; cross-validation dispersion should be considered alongside point estimates.

## Limitations

This is a small historical tabular dataset and is not representative of modern clinical populations. There is no external validation, prospective validation, calibration study, subgroup fairness study, or clinical utility assessment. Results must not be generalized to real patients.

## Tech stack

Python · pandas · NumPy · scikit-learn · joblib · Streamlit · pytest · GitHub Actions

## Author

Ahmed Wadee
