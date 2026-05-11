"""Load and preprocess the Telco Customer Churn dataset."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ID_COL,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges has blank strings for ~11 rows with tenure=0 — coerce + fill
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)
    df = df.drop(columns=[ID_COL])
    return df


def split(df: pd.DataFrame):
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_relative = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed(splits: dict[str, pd.DataFrame]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)


def prepare(persist: bool = True):
    """End-to-end: load → clean → split. Optionally persist to parquet."""
    df = clean(load_raw())
    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    if persist:
        save_processed(
            {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train.to_frame(),
                "y_val": y_val.to_frame(),
                "y_test": y_test.to_frame(),
            }
        )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = prepare()
    print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    print(f"churn rate (train): {y_train.mean():.4f}")
