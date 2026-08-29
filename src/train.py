"""Train and evaluate heart-disease classifiers without test-set leakage.

The final test split is held out before preprocessing or model selection.
Cross-validation and hyperparameter tuning operate only on the training split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Heart Disease"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

NUMERIC_FEATURES = [
    "Age",
    "BP",
    "Cholesterol",
    "Max HR",
    "ST depression",
]


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Statlog Heart dataset and map labels to 0/1."""
    data = pd.read_csv(path)
    expected = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET])
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    X = data.drop(columns=[TARGET]).copy()
    y = data[TARGET].map({"Absence": 0, "Presence": 1})
    if y.isna().any():
        bad_labels = sorted(data.loc[y.isna(), TARGET].astype(str).unique())
        raise ValueError(f"Unexpected target labels: {bad_labels}")
    return X, y.astype(int)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def candidate_searches(cv: StratifiedKFold) -> dict[str, GridSearchCV]:
    """Return two interpretable model searches using the same safe preprocessing."""
    logistic = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    forest = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    common = dict(scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    return {
        "logistic_regression": GridSearchCV(
            logistic,
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["liblinear", "lbfgs"],
            },
            **common,
        ),
        "random_forest": GridSearchCV(
            forest,
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 2, 4],
            },
            **common,
        ),
    }


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }


def train(
    data_path: str | Path,
    output_dir: str | Path = "artifacts",
    test_size: float = 0.20,
) -> dict[str, object]:
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    searches = candidate_searches(cv)

    summary: dict[str, object] = {
        "dataset_rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": test_size,
        "random_state": RANDOM_STATE,
        "models": {},
    }

    best_name = ""
    best_search: GridSearchCV | None = None
    best_cv_auc = -np.inf

    for name, search in searches.items():
        search.fit(X_train, y_train)
        model_summary = {
            "cv_roc_auc_mean": float(search.best_score_),
            "cv_roc_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "best_params": search.best_params_,
        }
        summary["models"][name] = model_summary
        if search.best_score_ > best_cv_auc:
            best_cv_auc = float(search.best_score_)
            best_name = name
            best_search = search

    assert best_search is not None
    final_metrics = evaluate(best_search.best_estimator_, X_test, y_test)
    summary["selected_model"] = best_name
    summary["test_metrics"] = final_metrics

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_search.best_estimator_, output / "model.joblib")
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-safe heart-disease classifiers.")
    parser.add_argument("--data", default="Heart_Disease_Prediction.csv")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    result = train(args.data, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
