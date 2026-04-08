"""
run_pipeline.py
---------------
End-to-end pipeline: load raw data → feature engineering → train all models
→ evaluate → save best model.

Usage:
    python run_pipeline.py
    python run_pipeline.py --data data/raw/spotify_tracks.csv --output models/
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import load_raw, basic_clean
from src.feature_engineering import engineer_features, ENGINEERED_FEATURES
from src.modeling import get_models, train_evaluate_all, evaluate, save_model

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "popularity"


def parse_args():
    parser = argparse.ArgumentParser(description="Spotify Popularity Prediction Pipeline")
    parser.add_argument(
        "--data", default="data/raw/spotify_tracks.csv",
        help="Path to raw Spotify CSV"
    )
    parser.add_argument(
        "--output", default="models/",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--processed", default="data/processed/spotify_features_engineered.csv",
        help="Path to save processed CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # ── 1. Load & clean ──────────────────────────────────────────────────────
    print("\n[1/5] Loading and cleaning data...")
    df = load_raw(args.data)
    df = basic_clean(df)
    print(f"  Shape after cleaning: {df.shape}")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print("\n[2/5] Engineering features...")
    df_feat, genre_means = engineer_features(df)

    # Save genre means for inference
    with open(os.path.join(args.output, "genre_means.json"), "w") as f:
        json.dump(genre_means, f, indent=2)

    # Save processed dataset
    df_feat.to_csv(args.processed, index=False)
    print(f"  Features engineered: {len(ENGINEERED_FEATURES)} total")
    print(f"  Processed data saved → {args.processed}")

    # ── 3. Prepare train/test split ──────────────────────────────────────────
    print("\n[3/5] Preparing train/test split (80/20)...")

    # Keep only engineered feature columns that exist
    available_features = [f for f in ENGINEERED_FEATURES if f in df_feat.columns]
    X = df_feat[available_features].fillna(df_feat[available_features].median())
    y = df_feat[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"  Train: {X_train_sc.shape}  |  Test: {X_test_sc.shape}")
    print(f"  Target mean: {y.mean():.2f}  |  std: {y.std():.2f}")

    # ── 4. Train & evaluate all models ───────────────────────────────────────
    print("\n[4/5] Training and evaluating all models...")
    results = train_evaluate_all(X_train_sc, y_train, X_test_sc, y_test)

    print("\n── Model Comparison ─────────────────────────────────────────────")
    print(results.to_string(index=False))

    results.to_csv("reports/model_comparison.csv", index=False)
    print("\n  Results saved → reports/model_comparison.csv")

    # ── 5. Save best model ───────────────────────────────────────────────────
    print("\n[5/5] Saving best model and artifacts...")
    best_name = results.iloc[0]["Model"]
    print(f"  Best model: {best_name}  (R²={results.iloc[0]['R2']:.4f})")

    best_model = get_models()[best_name]
    best_model.fit(X_train_sc, y_train)

    save_model(best_model, os.path.join(args.output, "best_model.pkl"))
    save_model(scaler, os.path.join(args.output, "scaler.pkl"))

    # Save feature list
    with open(os.path.join(args.output, "feature_list.json"), "w") as f:
        json.dump(available_features, f, indent=2)

    print("\n── Pipeline Complete ────────────────────────────────────────────")
    print(f"  Best Model : {best_name}")
    print(f"  RMSE       : {results.iloc[0]['RMSE']}")
    print(f"  MAE        : {results.iloc[0]['MAE']}")
    print(f"  R²         : {results.iloc[0]['R2']}")
    print(f"  CV R² (5k) : {results.iloc[0]['CV_R2_mean']} ± {results.iloc[0]['CV_R2_std']}")
    print("\nArtifacts saved to:", args.output)


if __name__ == "__main__":
    main()
