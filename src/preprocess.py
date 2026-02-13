"""Data loading and feature scaling."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import FEATURES


def load_data(train_path: str, valid_path: str) -> tuple:
    """Load train/valid CSVs and split into features + labels."""
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)

    X_train = train[FEATURES].copy()
    y_train = train["label"].copy()
    X_valid = valid[FEATURES].copy()
    y_valid = valid["label"].copy()

    print(f"Train: {X_train.shape[0]} rows, Valid: {X_valid.shape[0]} rows")
    return X_train, y_train, X_valid, y_valid


def scale_features(X_train: pd.DataFrame, X_valid: pd.DataFrame) -> tuple:
    """Fit StandardScaler on train, transform both splits.

    Standardizing is critical for Logistic Regression — without it,
    features on different scales (e.g. recency_days 0-60 vs fatigue_score 0-1)
    produce incomparable coefficients.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    return X_train_scaled, X_valid_scaled, scaler

