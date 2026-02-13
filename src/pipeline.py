"""
Lead Scoring Training Pipeline

Runs the full training flow: load data → scale → train → evaluate → export weights.
Works both locally and as an Azure ML pipeline step via CLI args.

Usage:
    python -m src.pipeline --train data/train.csv --valid data/valid.csv --output outputs/weights
"""

import argparse
import numpy as np

from .config import SEED
from .preprocess import load_data, scale_features
from .train import train_model, evaluate, extract_weights, export_weights


def run(train_path: str, valid_path: str, output_dir: str) -> None:
    """Execute the full training pipeline."""
    np.random.seed(SEED)

    # 1. Load & split
    X_train, y_train, X_valid, y_valid = load_data(train_path, valid_path)

    # 2. Standardize features
    X_train_scaled, X_valid_scaled, scaler = scale_features(X_train, X_valid)

    # 3. Train
    model = train_model(X_train_scaled, y_train)

    # 4. Evaluate
    metrics = evaluate(model, X_valid_scaled, y_valid)

    # 5. Extract & export weights
    weights = extract_weights(model)
    export_weights(weights, output_dir)

    print("\nPipeline complete.")


def main():
    parser = argparse.ArgumentParser(description="Lead scoring training pipeline")
    parser.add_argument("--train", default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--valid", default="data/valid.csv", help="Path to validation CSV")
    parser.add_argument("--output", default="outputs/weights", help="Directory for weight export")
    args = parser.parse_args()

    run(args.train, args.valid, args.output)


if __name__ == "__main__":
    main()

