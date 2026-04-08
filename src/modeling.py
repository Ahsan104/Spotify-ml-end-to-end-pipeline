"""
modeling.py
-----------
Model definitions, training, evaluation, and comparison utilities.
"""

import time
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import lightgbm as lgb


# ── Model registry ────────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return a dict of {name: model_instance} for all candidate models.
    Parameters chosen to balance training speed and performance.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=42, n_jobs=-1, verbose=-1
        ),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, and R² for a set of predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}


def train_evaluate_all(
    X_train, y_train, X_test, y_test, cv_folds: int = 5
) -> pd.DataFrame:
    """
    Train all models, evaluate on test set, and run CV.

    Returns
    -------
    pd.DataFrame with columns: Model, RMSE, MAE, R2, CV_R2_mean, CV_R2_std, TrainTime_s
    """
    models = get_models()
    rows = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"  Training {name}...", end=" ", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)

        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)

        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)

        rows.append({
            "Model": name,
            **metrics,
            "CV_R2_mean": round(cv_scores.mean(), 4),
            "CV_R2_std": round(cv_scores.std(), 4),
            "TrainTime_s": elapsed,
        })
        print(f"R²={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}  [{elapsed}s]")

    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    joblib.dump(model, path)
    print(f"  Saved model → {path}")


def load_model(path: str):
    return joblib.load(path)


# ── Trade-off summary ─────────────────────────────────────────────────────────

MODEL_TRADEOFFS = {
    "Linear Regression": {
        "interpretability": "Very High",
        "performance": "Low",
        "training_speed": "Very Fast",
        "best_for": "Baseline, stakeholder communication",
    },
    "Ridge": {
        "interpretability": "Very High",
        "performance": "Low-Medium",
        "training_speed": "Very Fast",
        "best_for": "When multicollinearity is a concern",
    },
    "Lasso": {
        "interpretability": "High",
        "performance": "Low-Medium",
        "training_speed": "Fast",
        "best_for": "Automatic feature selection",
    },
    "ElasticNet": {
        "interpretability": "High",
        "performance": "Low-Medium",
        "training_speed": "Fast",
        "best_for": "Balance between Ridge and Lasso",
    },
    "Random Forest": {
        "interpretability": "Medium",
        "performance": "High",
        "training_speed": "Medium",
        "best_for": "Non-linear patterns, robust to outliers",
    },
    "XGBoost": {
        "interpretability": "Low-Medium",
        "performance": "Very High",
        "training_speed": "Medium",
        "best_for": "Kaggle-style competitions, production systems",
    },
    "LightGBM": {
        "interpretability": "Low-Medium",
        "performance": "Very High",
        "training_speed": "Fast",
        "best_for": "Large datasets, best accuracy/speed trade-off",
    },
}
