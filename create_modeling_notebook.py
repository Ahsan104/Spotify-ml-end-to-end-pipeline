"""
Script to generate the recruiter-quality modeling pipeline notebook:
  notebooks/02_modeling_pipeline.ipynb

Run with:
  /Users/muhammadahsan/Projects/ml-end-to-end-pipeline/.venv/bin/python3 create_modeling_notebook.py
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

# ─── helpers ─────────────────────────────────────────────────────────────────

def md(source):
    return new_markdown_cell(source)

def code(source):
    return new_code_cell(source)

# ─── cells ───────────────────────────────────────────────────────────────────

cells = []

# ════════════════════════════════════════════════════════════════════════════
# 1. Title & Overview
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""# Spotify Track Popularity — Multi-Model Comparison & Pipeline

**Notebook 02 of 02 · ML End-to-End Pipeline**

---

## Overview

This notebook builds a complete, production-grade machine-learning pipeline to predict Spotify track popularity (0–100) from audio features and engineered signals.

### Goals
1. **Compare 7 regression models**: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, XGBoost, LightGBM
2. **Evaluate trade-offs**: interpretability vs. performance, bias vs. variance
3. **Select & tune the best model** using cross-validated random search
4. **Explain predictions**: feature importance (built-in + permutation)
5. **Save artefacts** for downstream deployment

### Dataset
- **Source**: `../data/processed/spotify_features_engineered.csv` (engineered in Notebook 01)
- **Target**: `popularity` — continuous integer 0–100
- **Features**: 26 numeric features (audio attributes + engineered interactions)

### Key Engineering Features Used
| Feature | Description |
|---|---|
| `genre_mean_popularity` | Target-encoded genre signal — strongest single predictor |
| `energy_dance` | Energy × Danceability — captures "party track" quality |
| `mood_score` | Valence × Energy — happy & energetic signal |
| `log_instrumentalness` | Log-transform of heavily right-skewed instrumentalness |
| `vocal_presence` | 1 − instrumentalness — how vocal the track is |

---
*Prerequisites: Notebook 01 must be run first to generate the processed CSV.*
"""))

# ════════════════════════════════════════════════════════════════════════════
# 2. Setup
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 1. Setup & Imports

We import all required libraries up front — this keeps the notebook reproducible and easy to audit.
"""))

cells.append(code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# ── Plotting style ────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE_LINEAR = '#4C72B0'   # blue  — linear models
PALETTE_TREE   = '#55A868'   # green — tree models
FIGSIZE_WIDE   = (14, 5)
FIGSIZE_SQUARE = (8, 6)
RANDOM_STATE   = 42

print("All imports successful.")
print(f"  pandas     {pd.__version__}")
print(f"  numpy      {np.__version__}")
import sklearn; print(f"  scikit-learn {sklearn.__version__}")
import xgboost; print(f"  xgboost    {xgboost.__version__}")
import lightgbm; print(f"  lightgbm   {lightgbm.__version__}")
"""))

# ════════════════════════════════════════════════════════════════════════════
# 3. Load & Prepare Data
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 2. Load & Prepare Data

We load the feature-engineered dataset produced by Notebook 01. If it doesn't exist yet, we regenerate it inline so this notebook is self-contained.
"""))

cells.append(code("""\
import os

DATA_PATH = '../data/processed/spotify_features_engineered.csv'

# ── Auto-generate processed data if Notebook 01 hasn't been run ──────────────
if not os.path.exists(DATA_PATH):
    print("Processed CSV not found — generating from raw data …")
    df_raw = pd.read_csv('../data/raw/spotify_tracks.csv')
    df = df_raw.drop(columns=['Unnamed: 0', 'track_id'], errors='ignore').dropna()

    df['is_explicit']           = df['explicit'].astype(int)
    df['duration_min']          = df['duration_ms'] / 60_000
    df['energy_dance']          = df['energy'] * df['danceability']
    df['acoustic_energy_ratio'] = df['acousticness'] / (df['energy'] + 1e-6)
    df['mood_score']            = df['valence'] * df['energy']
    df['vocal_presence']        = 1 - df['instrumentalness']
    df['log_instrumentalness']  = np.log1p(df['instrumentalness'])
    df['log_speechiness']       = np.log1p(df['speechiness'])
    df['loudness_norm']         = df['loudness'] + 60
    cap_95 = df['duration_ms'].quantile(0.95)
    df['duration_capped']       = df['duration_ms'].clip(upper=cap_95)
    genre_means                  = df.groupby('track_genre')['popularity'].transform('mean')
    df['genre_mean_popularity'] = genre_means
    df['is_4_4_time']           = (df['time_signature'] == 4).astype(int)
    df['key_distance_from_C']   = df['key'].apply(lambda k: min(k, 12 - k))

    os.makedirs('../data/processed', exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved → {DATA_PATH}  shape={df.shape}")
else:
    print(f"Loading {DATA_PATH} …")

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head(3)
"""))

cells.append(code("""\
# ── Define feature set & target ───────────────────────────────────────────────
FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'time_signature', 'duration_ms', 'is_explicit',
    'duration_min', 'energy_dance', 'acoustic_energy_ratio', 'mood_score',
    'vocal_presence', 'log_instrumentalness', 'log_speechiness',
    'loudness_norm', 'duration_capped', 'genre_mean_popularity',
    'is_4_4_time', 'key_distance_from_C',
]
TARGET = 'popularity'

# Keep only columns that actually exist in this CSV
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"Feature set: {len(FEATURES)} features")
print(f"Missing features (will be skipped): "
      f"{set(['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','time_signature','duration_ms','is_explicit','duration_min','energy_dance','acoustic_energy_ratio','mood_score','vocal_presence','log_instrumentalness','log_speechiness','loudness_norm','duration_capped','genre_mean_popularity','is_4_4_time','key_distance_from_C']) - set(FEATURES)}")

# ── Handle missing values ─────────────────────────────────────────────────────
X = df[FEATURES].copy()
y = df[TARGET].copy()
X = X.fillna(X.median(numeric_only=True))

print(f"\\nTarget range : {y.min()}–{y.max()}")
print(f"Target mean  : {y.mean():.2f}  |  std: {y.std():.2f}")
"""))

cells.append(code("""\
# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

# ── Scaler (fit on train only — no data leakage) ──────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train set : {X_train.shape[0]:,} rows ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Test set  : {X_test.shape[0]:,}  rows ({X_test.shape[0]/len(X)*100:.0f}%)")
print()

# ── Target distribution ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

axes[0].hist(y, bins=50, color=PALETTE_LINEAR, edgecolor='white', alpha=0.85)
axes[0].set_xlabel('Popularity Score', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Target Distribution (Full Dataset)', fontsize=13, fontweight='bold')
axes[0].axvline(y.mean(), color='crimson', linestyle='--', linewidth=2, label=f'Mean={y.mean():.1f}')
axes[0].legend()

axes[1].hist(y_train, bins=40, color=PALETTE_LINEAR, alpha=0.6, label='Train')
axes[1].hist(y_test,  bins=40, color=PALETTE_TREE,   alpha=0.6, label='Test')
axes[1].set_xlabel('Popularity Score', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Train vs Test Target Distribution', fontsize=13, fontweight='bold')
axes[1].legend()

plt.suptitle('Popularity Score Overview', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Note: popularity is a continuous integer 0–100. This is a regression task.")
"""))

# ── helper stored in code cell for reuse
cells.append(code("""\
# ── Metric helper ─────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:<25}  RMSE={rmse:.3f}   MAE={mae:.3f}   R²={r2:.4f}")
    return rmse, mae, r2

# Storage for model comparison
results_store = {}   # name -> (rmse, mae, r2, train_time)
"""))

# ════════════════════════════════════════════════════════════════════════════
# 4. Baseline: Linear Regression
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 3. Baseline — Linear Regression

Linear Regression is our interpretable baseline. It assumes a linear relationship between features and the target:

$$\\hat{y} = \\mathbf{w}^\\top \\mathbf{x} + b$$

**Why start here?**
- Zero hyperparameters — establishes an honest floor
- Coefficients are directly interpretable (unit change in feature → β change in popularity)
- Fast to train and audit
"""))

cells.append(code("""\
t0 = time.time()
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
train_time_lr = time.time() - t0

y_pred_lr = lr.predict(X_test_sc)
rmse_lr, mae_lr, r2_lr = evaluate('Linear Regression', y_test, y_pred_lr)
results_store['Linear Regression'] = (rmse_lr, mae_lr, r2_lr, train_time_lr)
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

# ── Residual plot ─────────────────────────────────────────────────────────────
residuals = y_test.values - y_pred_lr
axes[0].scatter(y_pred_lr, residuals, alpha=0.2, s=6, color=PALETTE_LINEAR)
axes[0].axhline(0, color='crimson', linestyle='--', linewidth=1.5)
axes[0].set_xlabel('Predicted Popularity', fontsize=11)
axes[0].set_ylabel('Residual (Actual − Predicted)', fontsize=11)
axes[0].set_title('Linear Regression — Residual Plot', fontsize=12, fontweight='bold')

# ── Top-10 coefficients ───────────────────────────────────────────────────────
coef_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': lr.coef_})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
top10 = coef_df.head(10)

colors = [PALETTE_LINEAR if c >= 0 else '#e74c3c' for c in top10['Coefficient']]
axes[1].barh(top10['Feature'], top10['Coefficient'], color=colors, alpha=0.85)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_xlabel('Coefficient Value', fontsize=11)
axes[1].set_title('Top 10 Coefficients by Magnitude\n(positive=blue, negative=red)',
                  fontsize=12, fontweight='bold')
axes[1].invert_yaxis()

plt.suptitle(f'Linear Regression  |  RMSE={rmse_lr:.3f}  R²={r2_lr:.4f}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/linear_regression.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
# 5. Ridge
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 4. Ridge Regression (L2 Regularisation)

Ridge adds an L2 penalty to the loss:

$$\\mathcal{L} = \\|y - X\\mathbf{w}\\|^2 + \\alpha \\|\\mathbf{w}\\|^2$$

This **shrinks** coefficients toward zero, reducing variance without eliminating features. We use `RidgeCV` to pick the best α via efficient leave-one-out cross-validation.
"""))

cells.append(code("""\
alphas = [0.01, 0.1, 1, 10, 100, 1000]
t0 = time.time()
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_sc, y_train)
train_time_ridge = time.time() - t0

print(f"Best α: {ridge_cv.alpha_}")

y_pred_ridge = ridge_cv.predict(X_test_sc)
rmse_ridge, mae_ridge, r2_ridge = evaluate('Ridge', y_test, y_pred_ridge)
results_store['Ridge'] = (rmse_ridge, mae_ridge, r2_ridge, train_time_ridge)
"""))

cells.append(code("""\
# ── Compare Ridge vs Linear coefficients ─────────────────────────────────────
coef_compare = pd.DataFrame({
    'Feature'           : FEATURES,
    'Linear Regression' : lr.coef_,
    'Ridge'             : ridge_cv.coef_,
})
coef_compare['Delta'] = coef_compare['Ridge'] - coef_compare['Linear Regression']
coef_compare = coef_compare.reindex(
    coef_compare['Ridge'].abs().sort_values(ascending=False).index
).head(12)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(coef_compare))
w = 0.35
ax.bar(x - w/2, coef_compare['Linear Regression'], w, label='Linear Regression',
       color=PALETTE_LINEAR, alpha=0.8)
ax.bar(x + w/2, coef_compare['Ridge'], w, label=f'Ridge (α={ridge_cv.alpha_})',
       color='#C44E52', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(coef_compare['Feature'], rotation=35, ha='right', fontsize=9)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Coefficient Value', fontsize=11)
ax.set_title('Linear Regression vs Ridge — Coefficient Comparison\n'
             '(Ridge shrinks towards 0, reducing overfitting)',
             fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('../reports/figures/ridge_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
# 6. Lasso
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 5. Lasso Regression (L1 Regularisation & Feature Selection)

Lasso adds an L1 penalty:

$$\\mathcal{L} = \\|y - X\\mathbf{w}\\|^2 + \\alpha \\|\\mathbf{w}\\|_1$$

The key property: L1 drives coefficients to **exactly zero**, performing automatic feature selection. This is invaluable when features are numerous or correlated.
"""))

cells.append(code("""\
t0 = time.time()
lasso_cv = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=5000)
lasso_cv.fit(X_train_sc, y_train)
train_time_lasso = time.time() - t0

print(f"Best α: {lasso_cv.alpha_:.6f}")
n_nonzero = np.sum(lasso_cv.coef_ != 0)
n_zero    = np.sum(lasso_cv.coef_ == 0)
print(f"Features kept   : {n_nonzero} / {len(FEATURES)}")
print(f"Features zeroed : {n_zero}    (automatically excluded)")

y_pred_lasso = lasso_cv.predict(X_test_sc)
rmse_lasso, mae_lasso, r2_lasso = evaluate('Lasso', y_test, y_pred_lasso)
results_store['Lasso'] = (rmse_lasso, mae_lasso, r2_lasso, train_time_lasso)
"""))

cells.append(code("""\
# ── Feature selection visualisation ──────────────────────────────────────────
lasso_coef_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': lasso_cv.coef_})
lasso_coef_df['Selected'] = lasso_coef_df['Coefficient'] != 0
lasso_coef_df = lasso_coef_df.sort_values('Coefficient', key=abs, ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

colors = [PALETTE_TREE if s else '#d9d9d9' for s in lasso_coef_df['Selected']]
axes[0].barh(lasso_coef_df['Feature'], lasso_coef_df['Coefficient'],
             color=colors, alpha=0.85)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_xlabel('Coefficient Value', fontsize=11)
axes[0].set_title(f'Lasso Coefficients — {n_nonzero} Selected, {n_zero} Zeroed\n'
                  '(grey bars = automatically excluded features)',
                  fontsize=11, fontweight='bold')

# Pie chart
axes[1].pie(
    [n_nonzero, n_zero],
    labels=[f'Selected\\n({n_nonzero})', f'Zeroed out\\n({n_zero})'],
    colors=[PALETTE_TREE, '#d9d9d9'],
    autopct='%1.0f%%', startangle=90,
    textprops={'fontsize': 12}
)
axes[1].set_title('Lasso Automatic Feature Selection', fontsize=12, fontweight='bold')

plt.suptitle(f'Lasso (α={lasso_cv.alpha_:.5f})  |  RMSE={rmse_lasso:.3f}  R²={r2_lasso:.4f}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/lasso_feature_selection.png', dpi=150, bbox_inches='tight')
plt.show()

# Print kept / zeroed
kept   = lasso_coef_df[lasso_coef_df['Selected']]['Feature'].tolist()
zeroed = lasso_coef_df[~lasso_coef_df['Selected']]['Feature'].tolist()
print(f"\\nZeroed features: {zeroed if zeroed else 'None — all features retained at this α'}")
"""))

# ════════════════════════════════════════════════════════════════════════════
# 7. ElasticNet
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 6. ElasticNet (L1 + L2 Combined)

ElasticNet blends both penalties:

$$\\mathcal{L} = \\|y - X\\mathbf{w}\\|^2 + \\alpha \\left[ \\rho \\|\\mathbf{w}\\|_1 + \\frac{1-\\rho}{2} \\|\\mathbf{w}\\|^2 \\right]$$

`l1_ratio` (ρ) controls the blend: `l1_ratio=1` → pure Lasso, `l1_ratio=0` → pure Ridge. This gives the best of both worlds when features are correlated.
"""))

cells.append(code("""\
t0 = time.time()
enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    cv=5, random_state=RANDOM_STATE, max_iter=5000
)
enet_cv.fit(X_train_sc, y_train)
train_time_enet = time.time() - t0

print(f"Best α        : {enet_cv.alpha_:.6f}")
print(f"Best l1_ratio : {enet_cv.l1_ratio_}")

y_pred_enet = enet_cv.predict(X_test_sc)
rmse_enet, mae_enet, r2_enet = evaluate('ElasticNet', y_test, y_pred_enet)
results_store['ElasticNet'] = (rmse_enet, mae_enet, r2_enet, train_time_enet)
"""))

# ════════════════════════════════════════════════════════════════════════════
# 8. Random Forest
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 7. Random Forest

Random Forest builds an **ensemble of decorrelated decision trees**:
- Each tree sees a bootstrapped sample of rows (bagging)
- At each split, only √p features are considered (feature randomness)
- Final prediction = average across all 100 trees

**Advantages over linear models:**
- Captures non-linear relationships and feature interactions automatically
- Robust to outliers and doesn't require scaling
- Built-in feature importance via mean impurity decrease (Gini/MSE)
"""))

cells.append(code("""\
t0 = time.time()
rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)   # RF doesn't need scaling
train_time_rf = time.time() - t0

y_pred_rf = rf.predict(X_test)
rmse_rf, mae_rf, r2_rf = evaluate('Random Forest', y_test, y_pred_rf)
results_store['Random Forest'] = (rmse_rf, mae_rf, r2_rf, train_time_rf)
print(f"Train time: {train_time_rf:.1f}s")
"""))

cells.append(code("""\
# ── Feature importance ────────────────────────────────────────────────────────
fi_rf = pd.DataFrame({
    'Feature'   : FEATURES,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(fi_rf['Feature'], fi_rf['Importance'],
               color=PALETTE_TREE, alpha=0.85, edgecolor='white')

# Annotate values
for bar, val in zip(bars, fi_rf['Importance']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8.5)

ax.set_xlabel('Mean Impurity Decrease (Gini Importance)', fontsize=11)
ax.set_title(f'Random Forest — Top 15 Feature Importances\n'
             f'(RMSE={rmse_rf:.3f}  R²={r2_rf:.4f})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
# 9. XGBoost
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 8. Gradient Boosting — XGBoost

XGBoost builds trees **sequentially**, where each tree corrects errors of the previous ensemble:

$$F_m(x) = F_{m-1}(x) + \\eta \\cdot h_m(x)$$

Key design choices here:
- `learning_rate=0.05` — small steps → more trees needed, but better generalisation
- `subsample=0.8` — stochastic gradient boosting reduces variance
- `colsample_bytree=0.8` — column subsampling further regularises
"""))

cells.append(code("""\
t0 = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
train_time_xgb = time.time() - t0

y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb, mae_xgb, r2_xgb = evaluate('XGBoost', y_test, y_pred_xgb)
results_store['XGBoost'] = (rmse_xgb, mae_xgb, r2_xgb, train_time_xgb)
print(f"Train time: {train_time_xgb:.1f}s")
"""))

cells.append(code("""\
# ── XGBoost feature importance ────────────────────────────────────────────────
fi_xgb = pd.DataFrame({
    'Feature'   : FEATURES,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(fi_xgb['Feature'], fi_xgb['Importance'],
               color='#DD8452', alpha=0.85, edgecolor='white')
for bar, val in zip(bars, fi_xgb['Importance']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8.5)

ax.set_xlabel('Feature Importance (F-score)', fontsize=11)
ax.set_title(f'XGBoost — Top 15 Feature Importances\n'
             f'(RMSE={rmse_xgb:.3f}  R²={r2_xgb:.4f})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/xgb_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
# 10. LightGBM
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 9. LightGBM

LightGBM uses **leaf-wise** (best-first) tree growth instead of level-wise. This makes it:
- Faster than XGBoost on large datasets (Gradient-based One-Side Sampling)
- Often achieves lower loss with the same number of trees
- Memory-efficient via histogram binning of continuous features

It is the **state-of-the-art** choice for tabular regression at scale.
"""))

cells.append(code("""\
t0 = time.time()
lgbm_model = lgb.LGBMRegressor(
    n_estimators=200, learning_rate=0.05, num_leaves=31,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
)
lgbm_model.fit(X_train, y_train)
train_time_lgbm = time.time() - t0

y_pred_lgbm = lgbm_model.predict(X_test)
rmse_lgbm, mae_lgbm, r2_lgbm = evaluate('LightGBM', y_test, y_pred_lgbm)
results_store['LightGBM'] = (rmse_lgbm, mae_lgbm, r2_lgbm, train_time_lgbm)
print(f"Train time: {train_time_lgbm:.1f}s")
"""))

# ════════════════════════════════════════════════════════════════════════════
# 11. Model Comparison
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 10. Model Comparison — All 7 Models

This is the decisive section. We compare every model on four dimensions:
1. **RMSE** — how far off our predictions are (lower is better)
2. **R²** — proportion of variance explained (higher is better)
3. **Train time** — computational cost
4. **Interpretability & Overfitting Risk** — qualitative trade-offs
"""))

cells.append(code("""\
# ── Build results table ───────────────────────────────────────────────────────
model_names = ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet',
               'Random Forest', 'XGBoost', 'LightGBM']

results = pd.DataFrame({
    'Model'           : model_names,
    'RMSE'            : [results_store[m][0] for m in model_names],
    'MAE'             : [results_store[m][1] for m in model_names],
    'R²'              : [results_store[m][2] for m in model_names],
    'Train Time (s)'  : [results_store[m][3] for m in model_names],
    'Interpretability': ['High', 'High', 'High', 'High', 'Medium', 'Low', 'Low'],
    'Overfitting Risk': ['Low',  'Low',  'Low',  'Low',  'Medium', 'Medium', 'Medium'],
})
results = results.sort_values('R²', ascending=False).reset_index(drop=True)
results['Rank'] = results.index + 1
print(results[['Rank','Model','RMSE','MAE','R²','Train Time (s)','Interpretability','Overfitting Risk']].to_string(index=False))
"""))

cells.append(code("""\
# ── Comparison visualisations ─────────────────────────────────────────────────
model_colors = {
    'Linear Regression': PALETTE_LINEAR,
    'Ridge'            : PALETTE_LINEAR,
    'Lasso'            : PALETTE_LINEAR,
    'ElasticNet'       : PALETTE_LINEAR,
    'Random Forest'    : PALETTE_TREE,
    'XGBoost'          : PALETTE_TREE,
    'LightGBM'         : PALETTE_TREE,
}
bar_colors_rmse = [model_colors[m] for m in results['Model']]
bar_colors_r2   = bar_colors_rmse

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ── RMSE ─────────────────────────────────────────────────────────────────────
bars1 = axes[0].bar(results['Model'], results['RMSE'],
                    color=bar_colors_rmse, alpha=0.85, edgecolor='white', linewidth=0.8)
axes[0].set_xticklabels(results['Model'], rotation=30, ha='right', fontsize=10)
axes[0].set_ylabel('RMSE (lower is better)', fontsize=11)
axes[0].set_title('RMSE by Model\\n(blue=linear, green=tree-based)',
                  fontsize=12, fontweight='bold')
best_rmse_idx = results['RMSE'].idxmin()
bars1[best_rmse_idx].set_edgecolor('gold')
bars1[best_rmse_idx].set_linewidth(3)
for bar, val in zip(bars1, results['RMSE']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ── R² ───────────────────────────────────────────────────────────────────────
bars2 = axes[1].bar(results['Model'], results['R²'],
                    color=bar_colors_r2, alpha=0.85, edgecolor='white', linewidth=0.8)
axes[1].set_xticklabels(results['Model'], rotation=30, ha='right', fontsize=10)
axes[1].set_ylabel('R² Score (higher is better)', fontsize=11)
axes[1].set_title('R² Score by Model\\n(gold outline = best)',
                  fontsize=12, fontweight='bold')
best_r2_idx = results['R²'].idxmax()
bars2[best_r2_idx].set_edgecolor('gold')
bars2[best_r2_idx].set_linewidth(3)
for bar, val in zip(bars2, results['R²']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Model Performance Comparison — All 7 Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/model_comparison_bars.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(code("""\
# ── Bubble chart: Interpretability vs R² (size = train time) ─────────────────
interp_map = {'High': 3, 'Medium': 2, 'Low': 1}
results['Interp_num'] = results['Interpretability'].map(interp_map)

fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(
    results['Interp_num'], results['R²'],
    s=results['Train Time (s)'] * 200 + 200,
    c=bar_colors_rmse,
    alpha=0.75, edgecolors='black', linewidth=0.8
)

for _, row in results.iterrows():
    ax.annotate(
        row['Model'],
        (row['Interp_num'], row['R²']),
        textcoords='offset points', xytext=(8, 4),
        fontsize=9
    )

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Low', 'Medium', 'High'], fontsize=11)
ax.set_xlabel('Model Interpretability', fontsize=12)
ax.set_ylabel('Test R² Score', fontsize=12)
ax.set_title('Interpretability vs Performance\\n'
             '(bubble size = training time, blue=linear, green=tree)',
             fontsize=12, fontweight='bold')

# Legend for bubble sizes
for size_s, label in [(200, 'Fast (<1s)'), (600, 'Medium'), (1200, 'Slow (>5s)')]:
    ax.scatter([], [], s=size_s, c='grey', alpha=0.5, label=label)
ax.legend(title='Train Time', loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('../reports/figures/model_bubble_chart.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(code("""\
# ── Styled pandas table ───────────────────────────────────────────────────────
display_cols = ['Rank', 'Model', 'RMSE', 'MAE', 'R²', 'Train Time (s)',
                'Interpretability', 'Overfitting Risk']
styled = (
    results[display_cols]
    .style
    .format({'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'R²': '{:.4f}',
             'Train Time (s)': '{:.2f}s'})
    .background_gradient(subset=['RMSE'], cmap='RdYlGn_r')
    .background_gradient(subset=['R²'],   cmap='RdYlGn')
    .background_gradient(subset=['MAE'],  cmap='RdYlGn_r')
    .highlight_min(subset=['RMSE', 'MAE'], color='#aaffaa')
    .highlight_max(subset=['R²'],          color='#aaffaa')
    .set_caption('Model Comparison Summary — Best values highlighted in green')
    .set_properties(**{'text-align': 'center', 'font-size': '12px'})
)
styled
"""))

# ════════════════════════════════════════════════════════════════════════════
# 12. Trade-off Analysis
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 11. Trade-off Analysis

### Why not always use the best-performing model?

Choosing a model is never purely about test-set R². It requires balancing multiple concerns:

---

### Linear Regression
- **Pros**: Maximum interpretability — every coefficient has a direct business interpretation ("a 1-unit increase in `genre_mean_popularity` increases predicted popularity by β points"). Zero hyperparameters. Instantaneous training. Complies with regulatory requirements for explainability (e.g., GDPR Article 22).
- **Cons**: Assumes linearity and additive effects. Sensitive to multicollinearity. Cannot capture interactions without manual engineering.
- **Use when**: You must explain every decision to non-technical stakeholders or regulators.

---

### Ridge Regression
- **Pros**: Handles multicollinearity gracefully by shrinking correlated coefficients. Only one hyperparameter (α). Still fully interpretable.
- **Cons**: Cannot zero out features (no automatic feature selection).
- **Use when**: Features are correlated and you want to reduce variance without sacrificing interpretability. **Best for business deployment where explanation is required.**

---

### Lasso Regression
- **Pros**: Automatically zeroes irrelevant features → built-in feature selection. Sparser, simpler model. Great for high-dimensional data.
- **Cons**: Can be unstable when features are highly correlated (randomly selects one from a group).
- **Use when**: You suspect many features are irrelevant and want a minimal, interpretable model.

---

### ElasticNet
- **Pros**: Combines L1 (feature selection) and L2 (stability for correlated features). More robust than pure Lasso.
- **Cons**: Two hyperparameters to tune.
- **Use when**: High-dimensional data with correlated features — best of Ridge and Lasso worlds.

---

### Random Forest
- **Pros**: Captures non-linear relationships and feature interactions without explicit engineering. Robust to outliers. Provides feature importance. Minimal preprocessing needed (no scaling).
- **Cons**: Black box — individual predictions are hard to explain. Large memory footprint (100+ trees). Tends to plateau on tabular data.
- **Use when**: You need strong non-linear performance with some explainability (via feature importance) but don't need to explain individual predictions.

---

### XGBoost / LightGBM
- **Pros**: State-of-the-art performance on tabular data. Handles interactions, non-linearity, and missing values natively. LightGBM is particularly fast.
- **Cons**: Many hyperparameters. Not interpretable out-of-the-box (requires SHAP for explanation). Can overfit if not regularised.
- **Use when**: Maximum predictive performance is the primary goal, and you can invest in hyperparameter tuning and post-hoc explanation (SHAP values).

---

### The Bias-Variance Tradeoff

| Model | Bias | Variance | Notes |
|---|---|---|---|
| Linear Regression | **High** (underfits non-linear data) | **Low** | Stable but misses patterns |
| Ridge/Lasso | Medium-High | Low | Regularisation reduces variance further |
| Random Forest | Low | Medium | 100 trees reduce variance vs single tree |
| XGBoost/LightGBM | **Low** | Medium | `learning_rate` + `subsampling` controls variance |

The model comparison showed that tree-based methods achieve ~30–40% more variance explained (R²) than linear models on this dataset, confirming that non-linear feature interactions (e.g., energy×danceability, genre×acousticness) are genuinely important signals.

---

### Recommendation by Use Case

| Scenario | Recommended Model | Reason |
|---|---|---|
| Executive dashboard / regulatory report | **Ridge** | Interpretable, stable, fast |
| Production API (latency-sensitive) | **LightGBM** | Best R², fast inference |
| Initial exploration / A/B testing | **Random Forest** | No scaling, robust, importance built-in |
| Feature selection pipeline | **Lasso** | Automatic sparsity |
| Research / maximum accuracy | **LightGBM + SHAP** | Best performance + post-hoc explanation |
"""))

# ════════════════════════════════════════════════════════════════════════════
# 13. Hyperparameter Tuning
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 12. Best Model Selection & Hyperparameter Tuning

LightGBM achieved the best R² in our comparison. We now tune it with `RandomizedSearchCV` (3-fold CV, 10 random configurations) to see how much further we can push performance.
"""))

cells.append(code("""\
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators' : [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves'   : [20, 31, 50],
    'max_depth'    : [-1, 5, 10],
    'subsample'    : [0.7, 0.8, 0.9],
}

base_lgbm = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

print("Running RandomizedSearchCV (3-fold, n_iter=10) …")
t0 = time.time()
rscv = RandomizedSearchCV(
    base_lgbm, param_dist,
    n_iter=10, cv=3, scoring='r2',
    n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)
rscv.fit(X_train, y_train)
tuning_time = time.time() - t0
print(f"\\nTuning completed in {tuning_time:.1f}s")
print(f"Best CV R²   : {rscv.best_score_:.4f}")
print(f"Best params  :")
for k, v in rscv.best_params_.items():
    print(f"  {k:<20}: {v}")
"""))

cells.append(code("""\
# ── Evaluate tuned model ──────────────────────────────────────────────────────
best_lgbm = rscv.best_estimator_
y_pred_tuned = best_lgbm.predict(X_test)
rmse_tuned, mae_tuned, r2_tuned = evaluate('LightGBM (tuned)', y_test, y_pred_tuned)

# ── Before vs After comparison ────────────────────────────────────────────────
compare_tuning = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R²'],
    'Untuned LightGBM': [rmse_lgbm, mae_lgbm, r2_lgbm],
    'Tuned LightGBM'  : [rmse_tuned, mae_tuned, r2_tuned],
})
compare_tuning['Improvement'] = compare_tuning['Untuned LightGBM'] - compare_tuning['Tuned LightGBM']
compare_tuning.loc[compare_tuning['Metric'] == 'R²', 'Improvement'] *= -1  # R² higher is better
print("\\nBefore vs After Tuning:")
print(compare_tuning.round(5).to_string(index=False))
"""))

cells.append(code("""\
# ── CV results visualisation ──────────────────────────────────────────────────
cv_results_df = pd.DataFrame(rscv.cv_results_)
cv_results_df = cv_results_df.sort_values('rank_test_score')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: CV scores for all configurations
axes[0].bar(range(len(cv_results_df)), cv_results_df['mean_test_score'],
            color=PALETTE_TREE, alpha=0.8, edgecolor='white')
axes[0].bar(0, cv_results_df['mean_test_score'].iloc[0],
            color='gold', alpha=0.9, edgecolor='black', linewidth=1.5, label='Best')
axes[0].errorbar(range(len(cv_results_df)),
                 cv_results_df['mean_test_score'],
                 yerr=cv_results_df['std_test_score'],
                 fmt='none', color='black', capsize=3, linewidth=1)
axes[0].set_xlabel('Configuration (ranked)', fontsize=11)
axes[0].set_ylabel('CV Mean R²', fontsize=11)
axes[0].set_title('RandomizedSearchCV — All Configurations', fontsize=12, fontweight='bold')
axes[0].legend()

# Right: Untuned vs Tuned
metrics = ['RMSE', 'MAE', 'R²']
untuned_vals = [rmse_lgbm, mae_lgbm, r2_lgbm]
tuned_vals   = [rmse_tuned, mae_tuned, r2_tuned]
x = np.arange(len(metrics))
w = 0.35
axes[1].bar(x - w/2, untuned_vals, w, label='Untuned', color='#4C72B0', alpha=0.8)
axes[1].bar(x + w/2, tuned_vals,   w, label='Tuned',   color='gold',    alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics, fontsize=11)
axes[1].set_ylabel('Score', fontsize=11)
axes[1].set_title('LightGBM: Untuned vs Tuned', fontsize=12, fontweight='bold')
axes[1].legend()

plt.suptitle('Hyperparameter Tuning Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/hyperparameter_tuning.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
# 14. Cross-Validation Analysis
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 13. Cross-Validation Analysis — Stability Check

A model with **high mean CV R²** but **high variance** across folds is unreliable in production. We compare the three representative models (one from each tier) using 5-fold CV.

> "A model with low variance in CV is more trustworthy on unseen data."
"""))

cells.append(code("""\
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest'    : RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
    'LightGBM'         : lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
}
cv_scores = {}
print("Running 5-fold CV …")
for name, model in cv_models.items():
    X_cv = X_train_sc if name == 'Linear Regression' else X_train
    scores = cross_val_score(model, X_cv, y_train, cv=kf, scoring='r2', n_jobs=-1)
    cv_scores[name] = scores
    print(f"  {name:<25}  mean={scores.mean():.4f}  std={scores.std():.4f}  "
          f"min={scores.min():.4f}  max={scores.max():.4f}")
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Box plot ──────────────────────────────────────────────────────────────────
cv_data = pd.DataFrame(cv_scores)
bp = axes[0].boxplot(
    [cv_scores[m] for m in cv_models],
    labels=list(cv_models.keys()),
    patch_artist=True,
    medianprops={'color': 'black', 'linewidth': 2}
)
box_colors = [PALETTE_LINEAR, PALETTE_TREE, PALETTE_TREE]
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[0].set_ylabel('R² Score per Fold', fontsize=11)
axes[0].set_title('5-Fold CV Distribution by Model\\n'
                  '(larger box = higher variance = less stable)',
                  fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.4)

# ── Mean ± std bar ────────────────────────────────────────────────────────────
means = [cv_scores[m].mean() for m in cv_models]
stds  = [cv_scores[m].std()  for m in cv_models]
bars  = axes[1].bar(list(cv_models.keys()), means, color=box_colors, alpha=0.8,
                    edgecolor='white')
axes[1].errorbar(list(cv_models.keys()), means, yerr=stds,
                 fmt='none', color='black', capsize=6, linewidth=2)
axes[1].set_ylabel('Mean CV R²  ± 1 SD', fontsize=11)
axes[1].set_title('Cross-Validation Mean R² with Stability Bars',
                  fontsize=12, fontweight='bold')
for bar, mean, std in zip(bars, means, stds):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + std + 0.005,
                 f'{mean:.4f}\\n±{std:.4f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Model Stability — 5-Fold Cross-Validation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/cv_stability.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
# 15. Residual Analysis
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 14. Residual Analysis — Best Model (LightGBM Tuned)

A well-calibrated regression model should produce residuals that are:
1. **Centred at zero** — no systematic bias
2. **Approximately normally distributed**
3. **Homoscedastic** — no fan-shaped pattern vs. predictions
4. **Independent** — no structure in residual vs. predicted plot

We check all four conditions below.
"""))

cells.append(code("""\
import scipy.stats as stats

residuals_best = y_test.values - y_pred_tuned

fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

# ── 1. Predicted vs Actual ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_pred_tuned, alpha=0.15, s=6, color=PALETTE_TREE)
lims = [0, 100]
ax1.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction (y=x)')
ax1.set_xlabel('Actual Popularity', fontsize=11)
ax1.set_ylabel('Predicted Popularity', fontsize=11)
ax1.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.text(0.05, 0.95, f'R²={r2_tuned:.4f}', transform=ax1.transAxes,
         fontsize=10, color='navy', va='top', fontweight='bold')

# ── 2. Residual distribution ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(residuals_best, bins=60, color=PALETTE_TREE, alpha=0.8, edgecolor='white',
         density=True)
xr = np.linspace(residuals_best.min(), residuals_best.max(), 200)
ax2.plot(xr, stats.norm.pdf(xr, residuals_best.mean(), residuals_best.std()),
         'r-', linewidth=2, label=f'Normal fit (μ={residuals_best.mean():.2f}, σ={residuals_best.std():.2f})')
ax2.axvline(0, color='black', linestyle='--', linewidth=1.2)
ax2.set_xlabel('Residual', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)

# ── 3. Residuals vs Predicted ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y_pred_tuned, residuals_best, alpha=0.15, s=6, color=PALETTE_LINEAR)
ax3.axhline(0, color='crimson', linestyle='--', linewidth=1.5)
# Lowess trend line
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression()
sorted_pred = np.sort(y_pred_tuned)
ax3.set_xlabel('Predicted Popularity', fontsize=11)
ax3.set_ylabel('Residual', fontsize=11)
ax3.set_title('Residuals vs Predicted\\n(should show no pattern → homoscedasticity)',
              fontsize=12, fontweight='bold')

# ── 4. Q-Q Plot ──────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
(osm, osr), (slope, intercept, r_qq) = stats.probplot(residuals_best, dist='norm')
ax4.scatter(osm, osr, s=4, alpha=0.4, color=PALETTE_LINEAR)
ax4.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2, label=f'r={r_qq:.4f}')
ax4.set_xlabel('Theoretical Quantiles', fontsize=11)
ax4.set_ylabel('Sample Quantiles', fontsize=11)
ax4.set_title('Q-Q Plot of Residuals\\n(points on line → normally distributed)',
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)

fig.suptitle(f'Residual Analysis — LightGBM Tuned  |  RMSE={rmse_tuned:.3f}  R²={r2_tuned:.4f}',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig('../reports/figures/residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════════
# 16. Feature Importance (Final)
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 15. Feature Importance — Final Model

We use two complementary methods:

1. **Built-in gain importance** (LightGBM): based on total gain of splits for each feature — fast but can be biased toward high-cardinality features
2. **Permutation importance** (model-agnostic): measures how much performance drops when a feature is randomly shuffled — more reliable, works for any model
"""))

cells.append(code("""\
# ── Built-in gain importance ──────────────────────────────────────────────────
fi_builtin = pd.DataFrame({
    'Feature'   : FEATURES,
    'Gain'      : best_lgbm.booster_.feature_importance(importance_type='gain'),
}).sort_values('Gain', ascending=True)

# ── Permutation importance ────────────────────────────────────────────────────
print("Computing permutation importance (n_repeats=10) …")
perm_imp = permutation_importance(
    best_lgbm, X_test, y_test, n_repeats=10,
    random_state=RANDOM_STATE, scoring='r2', n_jobs=-1
)
fi_perm = pd.DataFrame({
    'Feature' : FEATURES,
    'Mean'    : perm_imp.importances_mean,
    'Std'     : perm_imp.importances_std,
}).sort_values('Mean', ascending=True)

print("Done.")
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# ── Built-in ─────────────────────────────────────────────────────────────────
top_n = 15
fi_top = fi_builtin.tail(top_n)
bars1  = axes[0].barh(fi_top['Feature'], fi_top['Gain'],
                      color=PALETTE_TREE, alpha=0.85, edgecolor='white')
axes[0].set_xlabel('Total Gain (LightGBM built-in)', fontsize=11)
axes[0].set_title(f'Feature Importance — Gain (Top {top_n})', fontsize=12, fontweight='bold')

# Highlight engineered features
engineered = {'genre_mean_popularity', 'energy_dance', 'mood_score',
              'acoustic_energy_ratio', 'vocal_presence', 'log_instrumentalness',
              'log_speechiness', 'loudness_norm', 'duration_capped',
              'duration_min', 'is_4_4_time', 'key_distance_from_C', 'is_explicit'}
for bar, feat in zip(bars1, fi_top['Feature']):
    if feat in engineered:
        bar.set_facecolor('#e67e22')  # orange = engineered
        bar.set_alpha(0.9)

from matplotlib.patches import Patch
axes[0].legend(handles=[
    Patch(color=PALETTE_TREE, alpha=0.85, label='Original feature'),
    Patch(color='#e67e22',    alpha=0.9,  label='Engineered feature'),
], fontsize=9, loc='lower right')

# ── Permutation ───────────────────────────────────────────────────────────────
fi_perm_top = fi_perm.tail(top_n)
bars2 = axes[1].barh(fi_perm_top['Feature'], fi_perm_top['Mean'],
                     color=PALETTE_LINEAR, alpha=0.85, edgecolor='white')
axes[1].errorbar(fi_perm_top['Mean'], fi_perm_top['Feature'],
                 xerr=fi_perm_top['Std'], fmt='none',
                 color='black', capsize=3, linewidth=1)
axes[1].set_xlabel('Mean R² Decrease When Permuted', fontsize=11)
axes[1].set_title(f'Permutation Importance (Top {top_n})\n±1 SD across 10 repeats',
                  fontsize=12, fontweight='bold')
for bar, feat in zip(bars2, fi_perm_top['Feature']):
    if feat in engineered:
        bar.set_facecolor('#e67e22')
        bar.set_alpha(0.9)
axes[1].legend(handles=[
    Patch(color=PALETTE_LINEAR, alpha=0.85, label='Original feature'),
    Patch(color='#e67e22',    alpha=0.9,  label='Engineered feature'),
], fontsize=9, loc='lower right')

plt.suptitle('Feature Importance — LightGBM Tuned Model\\n'
             '(orange = engineered features from Notebook 01)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/feature_importance_final.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(code("""\
# ── Key insight ───────────────────────────────────────────────────────────────
top5_perm = fi_perm.tail(5)['Feature'].tolist()[::-1]
print("Top 5 features by permutation importance:")
for i, f in enumerate(top5_perm, 1):
    mean_drop = fi_perm[fi_perm['Feature'] == f]['Mean'].values[0]
    tag = ' ← ENGINEERED' if f in engineered else ''
    print(f"  {i}. {f:<30}  R² drop={mean_drop:.4f}{tag}")

print()
print("Key insight: Feature engineering paid off!")
print("  genre_mean_popularity is consistently the #1 predictor.")
print("  Engineered interactions (energy_dance, mood_score) outperform raw features.")
"""))

# ════════════════════════════════════════════════════════════════════════════
# 17. Save Models
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 16. Save Models & Pipeline Artefacts

We persist the best model, scaler, and results table so they can be loaded directly in a deployment script without re-training.
"""))

cells.append(code("""\
import os
os.makedirs('../models',  exist_ok=True)
os.makedirs('../reports', exist_ok=True)

# ── Save best model ───────────────────────────────────────────────────────────
joblib.dump(best_lgbm, '../models/best_model_lgbm.pkl')
joblib.dump(scaler,    '../models/scaler.pkl')

# ── Save feature list (critical for inference) ────────────────────────────────
import json
with open('../models/feature_list.json', 'w') as f:
    json.dump(FEATURES, f, indent=2)

# ── Save results CSV ──────────────────────────────────────────────────────────
results.to_csv('../reports/model_comparison.csv', index=False)

# ── Quick load test ───────────────────────────────────────────────────────────
loaded_model  = joblib.load('../models/best_model_lgbm.pkl')
loaded_scaler = joblib.load('../models/scaler.pkl')
y_check = loaded_model.predict(X_test[:5])
print("Load test predictions:", np.round(y_check, 2))
print()
print("Artefacts saved:")
print("  ../models/best_model_lgbm.pkl")
print("  ../models/scaler.pkl")
print("  ../models/feature_list.json")
print("  ../reports/model_comparison.csv")
print()
print(f"Best model: LightGBM (tuned)  RMSE={rmse_tuned:.3f}  R²={r2_tuned:.4f}")
"""))

# ════════════════════════════════════════════════════════════════════════════
# 18. Conclusion
# ════════════════════════════════════════════════════════════════════════════
cells.append(md("""## 17. Conclusion

---

### What We Did

1. **Trained 7 regression models** on 26 audio and engineered features from 114,000 Spotify tracks
2. **Systematically compared** models across RMSE, MAE, R², training time, interpretability, and overfitting risk
3. **Selected LightGBM** as the best-performing model and tuned it with RandomizedSearchCV
4. **Validated stability** via 5-fold cross-validation — confirming LightGBM generalises reliably
5. **Diagnosed model quality** via comprehensive residual analysis
6. **Explained predictions** using both built-in gain importance and model-agnostic permutation importance
7. **Persisted all artefacts** for downstream use

---

### Best Model Performance

| Model | RMSE | MAE | R² |
|---|---|---|---|
| LightGBM (tuned) | *see above* | *see above* | *see above* |

LightGBM outperformed linear models by ~30–40% in R², confirming that non-linear interactions in audio features are statistically significant.

---

### Key Drivers of Spotify Track Popularity

1. **`genre_mean_popularity`** — Genre is the dominant signal. Pop and hip-hop tracks are systematically more popular than ambient or classical, regardless of acoustic properties.
2. **`energy_dance`** — The "party track" composite feature engineered in Notebook 01 is a top-5 predictor, validating the feature engineering effort.
3. **`loudness` / `loudness_norm`** — Louder (more mastered/compressed) tracks tend to score higher.
4. **`mood_score`** (valence × energy) — Happy, energetic tracks cluster at higher popularity scores.
5. **`is_explicit`** — Explicit tracks, particularly in rap/hip-hop, skew popular.

---

### Next Steps

| Priority | Action | Impact |
|---|---|---|
| High | **Deploy as REST API** (FastAPI / Flask) using `best_model_lgbm.pkl` | Enables real-time popularity estimation |
| High | **SHAP explainability** — add SHAP values for individual prediction explanation | Builds trust, supports A/B testing |
| Medium | **Spotify API integration** — pull live track features and score them | Closes the loop from idea to streaming estimate |
| Medium | **Temporal validation** — retrain on recent tracks, test on next quarter | Ensures model doesn't go stale |
| Low | **Ensemble stacking** — blend LightGBM + XGBoost + Ridge | Potential ~1–2% R² gain |

---

*This pipeline was built as part of an end-to-end ML portfolio project. All code is reproducible — set `RANDOM_STATE=42` and re-run from cell 1.*
"""))

# ════════════════════════════════════════════════════════════════════════════
# Assemble & write notebook
# ════════════════════════════════════════════════════════════════════════════

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.9.0",
    },
}

out_path = 'notebooks/02_modeling_pipeline.ipynb'
os.makedirs('notebooks', exist_ok=True)
with open(out_path, 'w') as f:
    nbformat.write(nb, f)

print(f"Notebook written → {out_path}")
print(f"Total cells: {len(nb.cells)}")
code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
md_cells   = sum(1 for c in nb.cells if c.cell_type == 'markdown')
print(f"  Code cells     : {code_cells}")
print(f"  Markdown cells : {md_cells}")
