"""
feature_engineering.py
-----------------------
All feature creation transformations, applied consistently
to train and test sets.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def engineer_features(df: pd.DataFrame, genre_means: dict = None) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Spotify DataFrame (output of preprocessing.basic_clean).
    genre_means : dict, optional
        Mapping {genre: mean_popularity} computed on training set.
        If None (i.e., during training), it is computed from df and returned.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with engineered features appended.
    genre_means : dict
        Genre → mean popularity mapping (to be stored and reused on test data).
    """
    df = df.copy()

    # --- Interpretability transforms ---
    df['duration_min'] = df['duration_ms'] / 60_000

    # --- Interaction features ---
    df['energy_dance'] = df['energy'] * df['danceability']
    df['mood_score'] = df['valence'] * df['energy']
    df['acoustic_energy_ratio'] = df['acousticness'] / (df['energy'] + 1e-6)

    # --- Vocal presence (inverse of instrumentalness) ---
    df['vocal_presence'] = 1 - df['instrumentalness']

    # --- Log transforms for heavily skewed features ---
    df['log_instrumentalness'] = np.log1p(df['instrumentalness'])
    df['log_speechiness'] = np.log1p(df['speechiness'])

    # --- Loudness: shift to positive range ---
    df['loudness_norm'] = df['loudness'] + 60

    # --- Outlier capping: duration at 95th percentile of training data ---
    # During training we compute cap; during inference caller passes it
    cap_95 = df['duration_ms'].quantile(0.95)
    df['duration_capped'] = df['duration_ms'].clip(upper=cap_95)

    # --- Binary flags ---
    df['is_4_4_time'] = (df['time_signature'] == 4).astype(int)

    # --- Circle of fifths distance from C (key 0) ---
    df['key_distance_from_C'] = df['key'].apply(lambda k: min(k, 12 - k) if k >= 0 else 0)

    # --- Genre mean popularity encoding (target encoding) ---
    if genre_means is None:
        genre_means = df.groupby('track_genre')['popularity'].mean().to_dict()
    df['genre_mean_popularity'] = df['track_genre'].map(genre_means).fillna(
        np.mean(list(genre_means.values()))
    )

    return df, genre_means


# ── Scikit-learn compatible transformer ──────────────────────────────────────

class SpotifyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer wrapping engineer_features().
    Fit on training data computes genre means; transform applies all steps.
    """

    def __init__(self):
        self.genre_means_ = None

    def fit(self, X: pd.DataFrame, y=None):
        # Need popularity column to compute genre means during fit
        _, self.genre_means_ = engineer_features(X, genre_means=None)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_out, _ = engineer_features(X, genre_means=self.genre_means_)
        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params):
        return self.fit(X, y).transform(X)


ENGINEERED_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'time_signature', 'duration_ms', 'is_explicit',
    # engineered
    'duration_min', 'energy_dance', 'acoustic_energy_ratio', 'mood_score',
    'vocal_presence', 'log_instrumentalness', 'log_speechiness',
    'loudness_norm', 'duration_capped', 'genre_mean_popularity',
    'is_4_4_time', 'key_distance_from_C',
]
