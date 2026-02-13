"""Logistic Regression training, evaluation, and weight export."""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from .config import FEATURES, LR_PARAMS


def train_model(X_train_scaled: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    """Train Logistic Regression on standardized features."""
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train_scaled, y_train)
    print(f"Logistic Regression trained — converged in {model.n_iter_[0]} iterations")
    return model


def evaluate(model: LogisticRegression, X_valid_scaled: np.ndarray,
             y_valid: pd.Series) -> dict:
    """Compute key metrics on the validation set."""
    probs = model.predict_proba(X_valid_scaled)[:, 1]

    # Precision@K helper
    def p_at_k(k):
        top_k = np.argsort(probs)[-k:]
        return y_valid.iloc[top_k].mean()

    metrics = {
        "AUC-ROC": roc_auc_score(y_valid, probs),
        "PR-AUC": average_precision_score(y_valid, probs),
        "Brier": brier_score_loss(y_valid, probs),
        "P@100": p_at_k(100),
        "P@200": p_at_k(200),
    }

    print("\n--- Validation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def extract_weights(model: LogisticRegression) -> pd.DataFrame:
    """Normalize LR coefficients so |weights| sum to 1.0."""
    coefs = model.coef_[0]
    abs_sum = np.sum(np.abs(coefs))

    weights = pd.DataFrame({
        "feature": FEATURES,
        "weight": coefs / abs_sum,
    }).sort_values("feature")  # alphabetical for consistency

    print("\n--- Normalized Weights ---")
    for _, row in weights.iterrows():
        print(f"  {row['feature']:>23s}: {row['weight']:+.6f}")
    return weights


def export_weights(weights: pd.DataFrame, output_dir: str) -> str:
    """Save weights to CSV in the given directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "weights.csv")
    weights.to_csv(path, index=False)
    print(f"\nWeights exported to {path}")
    return path

