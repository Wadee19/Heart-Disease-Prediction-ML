from pathlib import Path
import pandas as pd
TARGET = "Heart Disease"
def load_data(path: str | Path = "Heart_Disease_Prediction.csv"):
    df = pd.read_csv(path)
    if TARGET not in df.columns: raise ValueError(f"Missing target column: {TARGET}")
    X = df.drop(columns=[TARGET]); y = df[TARGET].map({"Absence": 0, "Presence": 1})
    if y.isna().any(): raise ValueError("Unexpected target labels")
    return X, y
