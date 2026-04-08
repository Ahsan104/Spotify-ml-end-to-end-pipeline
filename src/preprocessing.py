"""
preprocessing.py
----------------
Raw data loading and cleaning utilities.
"""

import pandas as pd
import numpy as np


RAW_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature', 'duration_ms', 'explicit',
    'track_genre', 'popularity'
]

META_COLS = ['track_id', 'artists', 'album_name', 'track_name']


def load_raw(path: str) -> pd.DataFrame:
    """Load raw Spotify CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path)

    # Drop index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=['artists', 'album_name', 'track_name'])

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning:
    - Convert explicit bool → int
    - Remove exact duplicate tracks (same track_id)
    - Reset index
    """
    df = df.copy()

    if df.duplicated(subset=['track_id']).any():
        n_dupes = df.duplicated(subset=['track_id']).sum()
        print(f"  Dropping {n_dupes} duplicate track_ids")
        df = df.drop_duplicates(subset=['track_id'])

    df['is_explicit'] = df['explicit'].astype(int)
    df = df.reset_index(drop=True)

    return df


def get_numeric_features(df: pd.DataFrame) -> list:
    """Return list of numeric audio feature column names (excludes target and meta)."""
    exclude = META_COLS + ['track_genre', 'explicit', 'popularity']
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame with null counts, dtypes, and basic stats."""
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique': df.nunique()
    })
    return summary
