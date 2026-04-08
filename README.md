# Spotify Track Popularity Prediction
### End-to-End Machine Learning Pipeline

> Predicting what makes a song popular — a full data science workflow from raw audio features to a deployed-ready model.

---

## Overview

This project builds a complete predictive modeling pipeline to predict the **popularity score** (0–100) of a Spotify track using its audio features. The dataset contains **114,000 tracks across 114 genres**, sourced from Spotify's audio analysis API.

The goal is not just to build a model — it is to demonstrate the full data science process: understanding the data, extracting signal through feature engineering, systematically comparing models, and making principled decisions about which model to deploy.

---

## Project Structure

```
ml-end-to-end-pipeline/
├── data/
│   ├── raw/                        # Original Spotify CSV (114K rows)
│   └── processed/                  # Feature-engineered dataset
├── notebooks/
│   ├── 01_eda_feature_engineering.ipynb   # Deep EDA + 14 engineered features
│   └── 02_modeling_pipeline.ipynb         # 7-model comparison + trade-off analysis
├── src/
│   ├── preprocessing.py            # Data loading and cleaning utilities
│   ├── feature_engineering.py      # All feature transforms (sklearn-compatible)
│   └── modeling.py                 # Model registry, training, evaluation
├── models/                         # Saved model artifacts (.pkl)
├── reports/
│   └── figures/                    # Auto-generated plots from notebooks
├── run_pipeline.py                 # CLI: runs the full pipeline end-to-end
└── requirements.txt
```

---

## The Data

| Property | Value |
|---|---|
| Source | Spotify Tracks Dataset (via HuggingFace) |
| Rows | 114,000 tracks |
| Features | 18 raw audio features |
| Target | `popularity` (integer, 0–100) |
| Genres | 114, each with exactly 1,000 tracks |

**Audio features include:** `danceability`, `energy`, `loudness`, `valence`, `tempo`, `acousticness`, `instrumentalness`, `speechiness`, `liveness`, and more.

---

## Exploratory Data Analysis

The EDA notebook (`01_eda_feature_engineering.ipynb`) covers:

- **Target distribution**: `popularity` is bimodal — ~35% of tracks have 0 popularity (obscure), with a second peak around 35–50 (mainstream). This is a challenging regression target.
- **Genre analysis**: Pop, K-Pop, and Latin genres lead in average popularity. Genre encodes significant signal.
- **Feature distributions**: `instrumentalness` and `speechiness` are heavily right-skewed; `loudness` has a natural floor around -60 dB.
- **Correlation analysis**: `loudness` and `energy` positively correlate with popularity; `acousticness` and `instrumentalness` negatively correlate.
- **Outlier detection**: `duration_ms` has long-tail outliers (very long tracks). Capped at the 95th percentile.

---

## Feature Engineering

14 new features were created on top of the 18 raw audio features:

| Feature | Description | Rationale |
|---|---|---|
| `energy_dance` | `energy × danceability` | Interaction: high-energy + danceable → more popular |
| `mood_score` | `valence × energy` | Captures "happy & energetic" feel |
| `acoustic_energy_ratio` | `acousticness / (energy + ε)` | Differentiates acoustic vs electric |
| `vocal_presence` | `1 - instrumentalness` | Direct signal: songs with vocals are more popular |
| `log_instrumentalness` | `log1p(instrumentalness)` | Corrects heavy right skew |
| `log_speechiness` | `log1p(speechiness)` | Corrects heavy right skew |
| `loudness_norm` | `loudness + 60` | Shifts to positive range |
| `duration_capped` | `duration_ms` clipped at 95th pct | Removes duration outliers |
| `genre_mean_popularity` | Target-encoded genre | Genre is the strongest single predictor |
| `is_4_4_time` | Binary: 4/4 time signature | Most popular music is in 4/4 |
| `key_distance_from_C` | Circle of fifths distance | Encodes harmonic "accessibility" |
| `is_explicit` | Bool → int | Explicit tracks skew more popular |
| `duration_min` | `duration_ms / 60000` | More interpretable unit |

---

## Models Compared

Seven models were trained and evaluated on an 80/20 train-test split with 5-fold cross-validation:

| Model | RMSE | MAE | R² | Interpretability |
|---|---|---|---|---|
| Linear Regression | ~18.8 | ~14.2 | ~0.29 | Very High |
| Ridge | ~18.7 | ~14.1 | ~0.30 | Very High |
| Lasso | ~18.9 | ~14.3 | ~0.28 | High (sparse) |
| ElasticNet | ~18.8 | ~14.2 | ~0.29 | High |
| Random Forest | ~16.2 | ~11.9 | ~0.47 | Medium |
| XGBoost | ~15.8 | ~11.5 | ~0.50 | Low–Medium |
| **LightGBM** | **~15.5** | **~11.2** | **~0.52** | Low–Medium |

*Note: exact values depend on your run; R² ~0.52 means we explain ~52% of variance in popularity — a strong result given how subjective popularity is.*

---

## Trade-off Analysis

**Why not always pick the best-performing model?**

| Scenario | Recommended Model | Why |
|---|---|---|
| Stakeholder presentation | Ridge | Coefficients are directly explainable |
| Feature selection needed | Lasso | Automatically zeros out irrelevant features |
| Production with interpretability | Random Forest | Good performance + feature importances |
| Maximum predictive accuracy | LightGBM | Best RMSE/R², fast on large data |
| Debugging feature quality | Linear Regression | Baseline to validate feature engineering |

**Bias-Variance tradeoff**: Linear models underfit (high bias, low variance). Tree ensembles fit the non-linear relationships but require regularization to avoid overfitting. LightGBM with tuned `num_leaves` and `subsample` gives the best generalization.

---

## Key Findings

1. **Genre is the strongest predictor** — genre mean popularity alone explains ~15% of variance
2. **Loudness and energy drive popularity** — louder, higher-energy tracks score higher
3. **Instrumentalness is a negative signal** — tracks without vocals are generally less popular
4. **Feature engineering worked** — `energy_dance`, `mood_score`, and `genre_mean_popularity` all rank in the top-10 features by importance
5. **The problem is hard** — ~35% of tracks with popularity=0 create a floor effect; a classification approach (popular/not) may complement this regression

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/Ahsan104/Spotify-ml-end-to-end-pipeline.git
cd Spotify-ml-end-to-end-pipeline

# 2. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Download the dataset
# Place spotify_tracks.csv in data/raw/
# (Download from: https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)

# 4. Run the full pipeline
python run_pipeline.py

# 5. Or explore the notebooks interactively
jupyter lab
```

---

## Notebooks

| Notebook | Description |
|---|---|
| [`01_eda_feature_engineering.ipynb`](notebooks/01_eda_feature_engineering.ipynb) | Data exploration, distribution analysis, correlation heatmaps, 14 engineered features |
| [`02_modeling_pipeline.ipynb`](notebooks/02_modeling_pipeline.ipynb) | 7-model comparison, trade-off analysis, hyperparameter tuning, residual analysis, SHAP-style feature importance |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458)
![Jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626)

---

## Author

**Muhammad Ahsan**
[GitHub](https://github.com/Ahsan104) · Machine Learning / Data Science

---

*Dataset: [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) by Maharshi Pandya*
