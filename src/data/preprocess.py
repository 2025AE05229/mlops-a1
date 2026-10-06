"""Data preprocessing and cleaning module for UCI Heart Disease dataset."""

from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_RAW_PATH = Path("data/raw/processed.cleveland.data")
DEFAULT_CLEAN_PATH = Path("data/processed/heart_disease_cleaned.csv")

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def load_raw_data(filepath: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES, na_values="?")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Impute missing values: median for numerical, mode for categorical
    for col in df.columns:
        if df[col].isnull().any():
            if col in NUMERICAL_COLS:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Convert multiclass target (0-4) to binary (0 = no disease, 1 = disease present)
    df["target"] = (df["target"] > 0).astype(int)

    # Cast discrete columns to integer types
    int_cols = ["age", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]
    for col in int_cols:
        df[col] = df[col].astype(int)

    return df


def preprocess_pipeline(raw_path: Path = DEFAULT_RAW_PATH, clean_path: Path = DEFAULT_CLEAN_PATH) -> pd.DataFrame:
    raw_df = load_raw_data(raw_path)
    clean_df = clean_data(raw_df)

    out = Path(clean_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out, index=False)

    print(f"Processed: {len(clean_df)} rows, {len(clean_df.columns)} cols -> {out}")
    print(f"Target distribution:\n{clean_df['target'].value_counts(normalize=True).round(4).to_dict()}")
    return clean_df


if __name__ == "__main__":
    preprocess_pipeline()
