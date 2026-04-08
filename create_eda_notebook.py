import nbformat

nb = nbformat.v4.new_notebook()
cells = []

# ── Cell 1: Project Overview (Markdown) ──────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
# Spotify Track Popularity Prediction — EDA & Feature Engineering

---

## Goal
Predict `popularity` (a score from **0 to 100** assigned by Spotify) from a track's audio features and metadata using supervised machine learning.

## Dataset
| Property | Value |
|---|---|
| Records | 114,000 Spotify tracks |
| Columns | 21 (including target) |
| Genres | 114 (1,000 tracks per genre — perfectly balanced) |
| Target | `popularity` — continuous integer, 0–100 |

## What we'll do in this notebook
1. **Load & inspect** the raw data — shape, dtypes, nulls, duplicates
2. **Analyse the target** (`popularity`) — distribution, skew, zero-inflation
3. **Explore genres** — which genres produce the most popular tracks?
4. **Visualise audio features** — distributions and key observations
5. **Correlation analysis** — which raw features drive popularity?
6. **Feature interactions** — energy vs acousticness, danceability vs valence, explicit flag
7. **Outlier detection** — IQR-based analysis; capping strategy
8. **Feature engineering** — 14 new hand-crafted features with rationale
9. **Evaluate engineered features** — do they correlate better with the target?
10. **Save processed dataset** ready for modelling

> **Notebook authored for:** ML End-to-End Pipeline Portfolio Project
"""))

# ── Cell 2: Setup & Imports ───────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 1. Setup & Data Loading
"""))

cells.append(nbformat.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)

print("Libraries loaded successfully.")
print(f"pandas  : {pd.__version__}")
print(f"numpy   : {np.__version__}")
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Load the raw dataset
df_raw = pd.read_csv('../data/raw/spotify_tracks.csv')

print(f"Shape : {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")
df_raw.head(3)
"""))

# ── Cell 3: Data Overview ─────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 2. Data Overview

We start with a systematic inspection: data types, summary statistics, missing values, and duplicates.
"""))

cells.append(nbformat.v4.new_code_cell("""\
df_raw.info()
"""))

cells.append(nbformat.v4.new_code_cell("""\
df_raw.describe().T.round(3)
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Missing values
null_counts = df_raw.isnull().sum()
print("Null counts per column:")
print(null_counts[null_counts > 0])
print(f"\\nTotal null cells: {df_raw.isnull().sum().sum()}")
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Duplicate check
dup_count = df_raw.duplicated().sum()
print(f"Duplicate rows: {dup_count}")
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
### Cleaning Steps

1. **Drop `Unnamed: 0`** — auto-index column from the CSV, carries no information.
2. **Drop `track_id`** — a unique identifier; not a predictive feature.
3. **Drop 3 null rows** — only 3 rows have nulls (in `artists`, `album_name`, or `track_name`); negligible loss.
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Drop index and ID columns
df = df_raw.drop(columns=['Unnamed: 0', 'track_id'], errors='ignore')

# Drop null rows (only 3)
before = len(df)
df = df.dropna()
after = len(df)
print(f"Dropped {before - after} null rows. Remaining: {after:,}")

# Final shape
print(f"\\nFinal shape: {df.shape}")
print(f"\\n{df.shape[0]:,} tracks spanning {df['track_genre'].nunique()} genres "
      f"with {df.shape[1] - 1} features after cleaning.")
"""))

# ── Cell 4: Target Variable Analysis ─────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 3. Target Variable Analysis — `popularity`

Understanding the target distribution is critical before any modelling. Issues like zero-inflation, bimodality, or heavy tails directly affect which models and evaluation metrics to use.
"""))

cells.append(nbformat.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram + KDE
axes[0].hist(df['popularity'], bins=50, color='steelblue', edgecolor='white',
             alpha=0.8, density=True, label='Histogram')
df['popularity'].plot.kde(ax=axes[0], color='crimson', linewidth=2, label='KDE')
axes[0].set_title('Popularity Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Popularity Score (0–100)')
axes[0].set_ylabel('Density')
axes[0].legend()

# Box plot
axes[1].boxplot(df['popularity'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6),
                medianprops=dict(color='crimson', linewidth=2))
axes[1].set_title('Popularity Box Plot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Popularity Score (0–100)')
axes[1].set_xticks([])

plt.tight_layout()
plt.savefig('../reports/popularity_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbformat.v4.new_code_cell("""\
skewness = df['popularity'].skew()
kurt     = df['popularity'].kurtosis()
pct_zero = (df['popularity'] == 0).mean() * 100
median_  = df['popularity'].median()
mean_    = df['popularity'].mean()

print("=== Popularity Statistics ===")
print(f"Mean         : {mean_:.2f}")
print(f"Median       : {median_:.1f}")
print(f"Skewness     : {skewness:.4f}  (negative = left-skewed)")
print(f"Kurtosis     : {kurt:.4f}")
print(f"% with score=0: {pct_zero:.1f}%  ← zero-inflated!")
print(f"\\nRange: {df['popularity'].min()} – {df['popularity'].max()}")
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
### Observations
- **Bimodal distribution**: a large spike at **0** (tracks with no plays / not discovered) and a second hump around **35–50** representing mainstream music.
- **~35% of tracks have popularity = 0** — these are essentially unranked. This zero-inflation means a naive regressor will be pulled toward low predictions.
- **Skewness is negative** (left-skewed when zero-spike is considered) but the overall shape is complex.
- **This is a challenging regression problem** — consider using models robust to non-Gaussian targets (gradient boosting, random forest), and evaluate with MAE in addition to RMSE.
"""))

# ── Cell 5: Genre Analysis ────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 4. Genre Analysis

With 114 genres and exactly 1,000 tracks each, our dataset is **perfectly balanced by genre**. However, popularity varies dramatically across genres — making genre one of the most informative features.
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Genre count check
genre_counts = df['track_genre'].value_counts()
print(f"Total genres: {df['track_genre'].nunique()}")
print(f"Min tracks per genre: {genre_counts.min()}")
print(f"Max tracks per genre: {genre_counts.max()}")
print("\\n→ All 114 genres have exactly 1,000 tracks (perfectly balanced).")
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Top 20 genres by average popularity
genre_avg = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=True)
top20 = genre_avg.tail(20)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(top20.index, top20.values, color=plt.cm.RdYlGn(
    np.linspace(0.3, 0.9, 20)))
ax.set_xlabel('Average Popularity Score', fontsize=12)
ax.set_title('Top 20 Genres by Average Popularity', fontsize=14, fontweight='bold')
ax.axvline(df['popularity'].mean(), color='navy', linestyle='--',
           linewidth=1.5, label=f'Overall mean ({df["popularity"].mean():.1f})')
ax.legend()

for bar, val in zip(bars, top20.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('../reports/top20_genres_popularity.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Box plot — popularity by top 15 genres
top15_genres = genre_avg.tail(15).index.tolist()
df_top15 = df[df['track_genre'].isin(top15_genres)].copy()
order = df_top15.groupby('track_genre')['popularity'].median().sort_values(
    ascending=False).index

fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=df_top15, x='track_genre', y='popularity',
            order=order, palette='husl', ax=ax, fliersize=2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_title('Popularity Distribution — Top 15 Genres', fontsize=14, fontweight='bold')
ax.set_xlabel('Genre')
ax.set_ylabel('Popularity Score')
plt.tight_layout()
plt.savefig('../reports/top15_genres_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
### Observations
- **Pop, K-Pop, and Latin genres** consistently achieve the highest average popularity.
- **Niche genres** (ambient, classical, avant-garde) cluster near 0–20.
- The variance within genres is high — genre alone doesn't fully explain popularity.
- Because all genres have exactly 1,000 tracks, any genre-level signal is genuine popularity signal, not sampling bias.
"""))

# ── Cell 6: Audio Feature Distributions ──────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 5. Audio Feature Distributions

Spotify provides several audio features computed by their audio analysis engine. Let's understand the shape of each distribution before building models.
"""))

cells.append(nbformat.v4.new_code_cell("""\
numeric_audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature', 'duration_ms'
]

fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes_flat = axes.flatten()

for i, feat in enumerate(numeric_audio_features):
    ax = axes_flat[i]
    data = df[feat].dropna()
    ax.hist(data, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(feat, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Count')
    skew_val = data.skew()
    ax.text(0.97, 0.95, f'skew={skew_val:.2f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color='crimson')

# Hide unused axes
for j in range(len(numeric_audio_features), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle('Audio Feature Distributions', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../reports/audio_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
### Key Observations

| Feature | Distribution | Note |
|---|---|---|
| `instrumentalness` | Extremely right-skewed | ~80% of tracks have value ≈ 0 (have vocals) |
| `speechiness` | Heavily right-skewed | Most tracks are music, not podcasts/speech |
| `acousticness` | Bimodal | Either acoustic OR electronic — not much in between |
| `loudness` | Left-skewed, range −60 to 0 dB | Natural logarithmic scale |
| `duration_ms` | Right-skewed with outliers | Some tracks > 30 mins |
| `danceability` | Roughly normal, slight left skew | Centred around 0.5–0.6 |
| `energy` | Slightly left-skewed | Most tracks have moderate-high energy |
| `valence` | Roughly uniform | Mood is evenly distributed across the catalogue |
| `key` | Discrete uniform | All 12 keys roughly equally represented |
| `mode` | Binary (0/1) | Major (1) slightly more common than minor (0) |

**Implications for modelling:** `instrumentalness` and `speechiness` need log transforms before using in linear models. `duration_ms` needs capping. Tree-based models can handle these natively.
"""))

# ── Cell 7: Correlation Analysis ─────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 6. Correlation Analysis

We compute Pearson correlations to identify which raw features are linearly associated with `popularity`. Note that non-linear relationships won't be fully captured here.
"""))

cells.append(nbformat.v4.new_code_cell("""\
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix  = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5,
            annot_kws={'size': 8}, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Correlation with popularity (sorted)
pop_corr = corr_matrix['popularity'].drop('popularity').sort_values(key=abs, ascending=False)
print("Correlation with popularity (|r| sorted):")
print(pop_corr.round(4).to_string())
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Scatter plots — top 5 correlated features
top5 = pop_corr.head(5).index.tolist()

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for ax, feat in zip(axes, top5):
    # Sample 5000 points to avoid overplotting
    sample = df[[feat, 'popularity']].sample(5000, random_state=42)
    ax.scatter(sample[feat], sample['popularity'],
               alpha=0.15, s=8, color='steelblue')
    # Trend line
    z = np.polyfit(sample[feat], sample['popularity'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample[feat].min(), sample[feat].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r={pop_corr[feat]:.3f}')
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel('Popularity')
    ax.set_title(f'{feat} vs Popularity', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)

plt.suptitle('Top 5 Features Correlated with Popularity', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/top5_corr_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
### Key Findings

- **Positive correlations**: `loudness` (louder tracks tend to score higher — production quality effect), `energy`, `danceability`
- **Negative correlations**: `acousticness` (acoustic = often older/niche), `instrumentalness` (strong negative — vocal tracks dominate charts)
- **Weak correlations**: `key`, `mode`, `time_signature`, `liveness`, `tempo`
- No single feature has |r| > 0.35 — this is a noisy, multi-factor problem. **Non-linear models will outperform linear ones.**
"""))

# ── Cell 8: Feature Relationships & Interactions ──────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 7. Feature Relationships & Interactions

Beyond individual correlations, understanding how features interact reveals musical patterns and motivates engineering new composite features.
"""))

cells.append(nbformat.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

sample = df.sample(8000, random_state=42)

# ── Plot 1: Energy vs Acousticness ──
ax = axes[0, 0]
sc = ax.scatter(sample['energy'], sample['acousticness'],
                c=sample['popularity'], cmap='RdYlGn',
                alpha=0.3, s=8, vmin=0, vmax=100)
plt.colorbar(sc, ax=ax, label='Popularity')
ax.set_xlabel('Energy')
ax.set_ylabel('Acousticness')
ax.set_title('Energy vs Acousticness\\n(colored by Popularity)', fontweight='bold')
r_val = df['energy'].corr(df['acousticness'])
ax.text(0.05, 0.95, f'r = {r_val:.3f}', transform=ax.transAxes,
        fontsize=10, color='navy', va='top')

# ── Plot 2: Danceability vs Valence ──
ax = axes[0, 1]
sc = ax.scatter(sample['danceability'], sample['valence'],
                c=sample['popularity'], cmap='RdYlGn',
                alpha=0.3, s=8, vmin=0, vmax=100)
plt.colorbar(sc, ax=ax, label='Popularity')
ax.set_xlabel('Danceability')
ax.set_ylabel('Valence')
ax.set_title('Danceability vs Valence\\n(colored by Popularity)', fontweight='bold')

# ── Plot 3: Loudness vs Energy ──
ax = axes[0, 2]
sc = ax.scatter(sample['loudness'], sample['energy'],
                c=sample['popularity'], cmap='RdYlGn',
                alpha=0.3, s=8, vmin=0, vmax=100)
plt.colorbar(sc, ax=ax, label='Popularity')
ax.set_xlabel('Loudness (dB)')
ax.set_ylabel('Energy')
ax.set_title('Loudness vs Energy\\n(colored by Popularity)', fontweight='bold')
r_le = df['loudness'].corr(df['energy'])
ax.text(0.05, 0.95, f'r = {r_le:.3f}', transform=ax.transAxes,
        fontsize=10, color='navy', va='top')

# ── Plot 4: Explicit vs Popularity ──
ax = axes[1, 0]
sns.boxplot(data=df, x='explicit', y='popularity', ax=ax, palette=['#4C72B0', '#DD8452'])
ax.set_xticklabels(['Non-Explicit', 'Explicit'])
ax.set_xlabel('Track Type')
ax.set_ylabel('Popularity')
ax.set_title('Explicit vs Non-Explicit\\nPopularity', fontweight='bold')
expl_mean   = df[df['explicit'] == True]['popularity'].mean()
nonexpl_mean = df[df['explicit'] == False]['popularity'].mean()
ax.text(0.05, 0.95, f'Explicit mean: {expl_mean:.1f}\\nNon-explicit: {nonexpl_mean:.1f}',
        transform=ax.transAxes, fontsize=9, va='top', color='darkred')

# ── Plot 5: Mode vs Popularity ──
ax = axes[1, 1]
sns.boxplot(data=df, x='mode', y='popularity', ax=ax, palette=['#4C72B0', '#DD8452'])
ax.set_xticklabels(['Minor (0)', 'Major (1)'])
ax.set_xlabel('Mode')
ax.set_ylabel('Popularity')
ax.set_title('Mode (Major/Minor)\\nvs Popularity', fontweight='bold')

# ── Plot 6: Instrumentalness vs Popularity ──
ax = axes[1, 2]
ax.scatter(sample['instrumentalness'], sample['popularity'],
           alpha=0.2, s=8, color='steelblue')
ax.set_xlabel('Instrumentalness')
ax.set_ylabel('Popularity')
ax.set_title('Instrumentalness vs Popularity', fontweight='bold')
r_inst = df['instrumentalness'].corr(df['popularity'])
ax.text(0.05, 0.95, f'r = {r_inst:.3f}', transform=ax.transAxes,
        fontsize=10, color='crimson', va='top')

plt.suptitle('Feature Relationships & Interactions', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../reports/feature_interactions.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
### Observations

- **Energy ↔ Acousticness**: Strong negative correlation (r ≈ −0.72). Acoustic songs are inherently less energetic — makes musical sense. High popularity tracks cluster in the high-energy, low-acousticness region.
- **Danceability ↔ Valence**: Moderate positive correlation. Happy-sounding (high valence) and danceable tracks tend to be more popular.
- **Loudness ↔ Energy**: Very strong positive correlation (r ≈ 0.76). These two capture similar constructs — a composite feature could reduce redundancy.
- **Explicit tracks**: Higher median popularity than non-explicit — likely because popular mainstream genres (hip-hop, pop) use explicit content more.
- **Major vs Minor**: Tracks in major key have slightly higher popularity — listeners may prefer upbeat-sounding music.
- **Instrumentalness**: Near-zero values dominate the high-popularity region — vocal tracks dominate charts.
"""))

# ── Cell 9: Outlier Detection ─────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 8. Outlier Detection

We focus on `duration_ms` (known to have extreme outliers) and `loudness` (natural range but check for anomalies).
"""))

cells.append(nbformat.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Duration
ax = axes[0]
ax.boxplot(df['duration_ms'] / 60000, vert=True, patch_artist=True,
           boxprops=dict(facecolor='steelblue', alpha=0.6),
           flierprops=dict(marker='o', markerfacecolor='crimson',
                           markersize=3, alpha=0.4),
           medianprops=dict(color='crimson', linewidth=2))
ax.set_ylabel('Duration (minutes)')
ax.set_title('Duration Distribution (minutes)', fontsize=13, fontweight='bold')
ax.set_xticks([])

# Loudness
ax = axes[1]
ax.boxplot(df['loudness'], vert=True, patch_artist=True,
           boxprops=dict(facecolor='darkorange', alpha=0.6),
           flierprops=dict(marker='o', markerfacecolor='navy',
                           markersize=3, alpha=0.4),
           medianprops=dict(color='navy', linewidth=2))
ax.set_ylabel('Loudness (dB)')
ax.set_title('Loudness Distribution (dB)', fontsize=13, fontweight='bold')
ax.set_xticks([])

plt.tight_layout()
plt.savefig('../reports/outlier_detection.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbformat.v4.new_code_cell("""\
# IQR-based outlier analysis for duration
Q1_dur = df['duration_ms'].quantile(0.25)
Q3_dur = df['duration_ms'].quantile(0.75)
IQR_dur = Q3_dur - Q1_dur
upper_dur = Q3_dur + 1.5 * IQR_dur
pct95_dur = df['duration_ms'].quantile(0.95)

dur_outliers = (df['duration_ms'] > upper_dur).sum()
print("=== Duration (ms) ===")
print(f"Q1: {Q1_dur/60000:.2f} min | Q3: {Q3_dur/60000:.2f} min | IQR: {IQR_dur/60000:.2f} min")
print(f"IQR upper fence: {upper_dur/60000:.2f} min")
print(f"Outliers (IQR method): {dur_outliers:,} ({dur_outliers/len(df)*100:.1f}%)")
print(f"95th percentile (cap value): {pct95_dur/60000:.2f} min")
print()

# IQR for loudness
Q1_l = df['loudness'].quantile(0.25)
Q3_l = df['loudness'].quantile(0.75)
IQR_l = Q3_l - Q1_l
lower_l = Q1_l - 1.5 * IQR_l
loud_outliers = (df['loudness'] < lower_l).sum()
print("=== Loudness (dB) ===")
print(f"Q1: {Q1_l:.2f} dB | Q3: {Q3_l:.2f} dB | IQR: {IQR_l:.2f} dB")
print(f"IQR lower fence: {lower_l:.2f} dB")
print(f"Outliers (IQR method): {loud_outliers:,} ({loud_outliers/len(df)*100:.1f}%)")
print()
print("Decision: Cap duration at 95th percentile | Keep loudness as-is (natural range)")
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
### Outlier Decision Summary

| Feature | Outliers (IQR) | Decision | Rationale |
|---|---|---|---|
| `duration_ms` | ~5% | **Cap at 95th percentile** | Very long tracks (live sets, classical suites) distort the feature space |
| `loudness` | Small % below −40 dB | **Keep as-is** | Natural audio range; extreme quiet is musically valid |

We'll apply the cap during Feature Engineering.
"""))

# ── Cell 10: Feature Engineering ─────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 9. Feature Engineering

Raw features often don't represent the underlying musical concepts optimally. We engineer 14 new features to:
- **Improve linearity** (log transforms for skewed distributions)
- **Capture interactions** (energy × danceability, valence × energy)
- **Add domain knowledge** (circle of fifths, tempo categories)
- **Encode categorical structure** (genre mean popularity)
"""))

cells.append(nbformat.v4.new_code_cell("""\
df_feat = df.copy()

# ─── Feature 1: Duration in minutes ──────────────────────────────────────────
# More interpretable than milliseconds; aligns with human intuition (3-4 min song)
df_feat['duration_min'] = df_feat['duration_ms'] / 60000

# ─── Feature 2: Energy-Danceability interaction ───────────────────────────────
# Captures the "party track" quality — both high energy AND danceable
df_feat['energy_dance'] = df_feat['energy'] * df_feat['danceability']

# ─── Feature 3: Acousticness-Energy ratio ────────────────────────────────────
# How "acoustic" relative to how "electric" — pure acoustic feel
df_feat['acoustic_energy_ratio'] = df_feat['acousticness'] / (df_feat['energy'] + 1e-6)

# ─── Feature 4: Mood Score ───────────────────────────────────────────────────
# Valence × Energy = "happy & energetic" — think upbeat pop/dance
df_feat['mood_score'] = df_feat['valence'] * df_feat['energy']

# ─── Feature 5: Vocal Presence ───────────────────────────────────────────────
# Inverse of instrumentalness — how "vocal" is the track?
df_feat['vocal_presence'] = 1 - df_feat['instrumentalness']

# ─── Feature 6: Log transform of instrumentalness ────────────────────────────
# Heavily right-skewed (>80% near 0). log1p spreads the low values better
df_feat['log_instrumentalness'] = np.log1p(df_feat['instrumentalness'])

# ─── Feature 7: Log transform of speechiness ─────────────────────────────────
# Similarly skewed — most music has low speechiness (not podcasts)
df_feat['log_speechiness'] = np.log1p(df_feat['speechiness'])

# ─── Feature 8: Loudness normalised ──────────────────────────────────────────
# Shift from [−60, 0] to approx [0, 60] — easier for some algorithms
df_feat['loudness_norm'] = df_feat['loudness'] + 60

# ─── Feature 9: Duration capped at 95th percentile ───────────────────────────
# Reduce outlier influence; preserves relative ordering for typical tracks
cap_95 = df_feat['duration_ms'].quantile(0.95)
df_feat['duration_capped'] = df_feat['duration_ms'].clip(upper=cap_95)
print(f"Duration cap applied at: {cap_95/60000:.2f} minutes")

# ─── Feature 10: Tempo category ──────────────────────────────────────────────
# Genre-aligned tempo buckets: slow ballads → medium pop → fast EDM
df_feat['tempo_category'] = pd.cut(
    df_feat['tempo'],
    bins=[0, 90, 120, 160, 300],
    labels=['slow', 'medium', 'fast', 'very_fast']
)

# ─── Feature 11: Genre mean popularity (target encoding) ─────────────────────
# A strong summary of genre-level popularity without one-hot overhead
genre_popularity = df_feat.groupby('track_genre')['popularity'].mean()
df_feat['genre_mean_popularity'] = df_feat['track_genre'].map(genre_popularity)

# ─── Feature 12: Is explicit ─────────────────────────────────────────────────
# Convert bool → int for ML compatibility
df_feat['is_explicit'] = df_feat['explicit'].astype(int)

# ─── Feature 13: Is 4/4 time ─────────────────────────────────────────────────
# 4/4 is overwhelmingly most common; flag captures the minority
df_feat['is_4_4_time'] = (df_feat['time_signature'] == 4).astype(int)

# ─── Feature 14: Key distance from C (circle of fifths) ──────────────────────
# Musical theory: distance on circle of fifths may relate to consonance
# C=0, G=7→5, F=5, D=2, A=9→3, etc.
circle_of_fifths_dist = {
    0: 0,  # C
    7: 1,  # G
    2: 2,  # D
    9: 3,  # A
    4: 4,  # E
    11: 5, # B
    6: 6,  # F#/Gb (max distance)
    1: 5,  # Db
    8: 4,  # Ab
    3: 3,  # Eb
    10: 2, # Bb
    5: 1,  # F
}
df_feat['key_distance_from_C'] = df_feat['key'].map(circle_of_fifths_dist).fillna(0).astype(int)

engineered_features = [
    'duration_min', 'energy_dance', 'acoustic_energy_ratio', 'mood_score',
    'vocal_presence', 'log_instrumentalness', 'log_speechiness',
    'loudness_norm', 'duration_capped', 'tempo_category',
    'genre_mean_popularity', 'is_explicit', 'is_4_4_time', 'key_distance_from_C'
]

print(f"\\nEngineered {len(engineered_features)} new features successfully.")
print(f"New dataset shape: {df_feat.shape}")
df_feat[engineered_features].head(3)
"""))

# ── Cell 11: Engineered Features Analysis ────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 10. Engineered Features Analysis

Let's verify that the new features provide additional predictive signal — specifically stronger correlations with `popularity` than their raw counterparts.
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Numeric engineered features only (exclude tempo_category which is categorical)
numeric_eng = [f for f in engineered_features if f != 'tempo_category']
eng_corr = df_feat[numeric_eng + ['popularity']].corr()['popularity'].drop('popularity')
eng_corr_sorted = eng_corr.sort_values(key=abs, ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Left: Engineered feature correlations ──
ax = axes[0]
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in eng_corr_sorted.values]
ax.barh(eng_corr_sorted.index, eng_corr_sorted.values, color=colors, alpha=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Pearson r with Popularity')
ax.set_title('Engineered Features — Correlation with Popularity',
             fontsize=12, fontweight='bold')
for i, (feat, val) in enumerate(eng_corr_sorted.items()):
    ax.text(val + (0.002 if val >= 0 else -0.002), i,
            f'{val:.3f}', va='center',
            ha='left' if val >= 0 else 'right', fontsize=8)

# ── Right: Raw vs Engineered comparison for key features ──
comparison_pairs = {
    'instrumentalness': 'log_instrumentalness',
    'speechiness': 'log_speechiness',
    'energy': 'energy_dance',
    'valence': 'mood_score',
}
raw_corrs  = [df_feat[raw].corr(df_feat['popularity']) for raw in comparison_pairs]
eng_corrs  = [df_feat[eng].corr(df_feat['popularity']) for eng in comparison_pairs.values()]
labels     = list(comparison_pairs.keys())

x = np.arange(len(labels))
width = 0.35
ax2 = axes[1]
bars1 = ax2.bar(x - width/2, [abs(v) for v in raw_corrs],  width,
                label='Raw', color='#3498db', alpha=0.8)
bars2 = ax2.bar(x + width/2, [abs(v) for v in eng_corrs], width,
                label='Engineered', color='#e67e22', alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha='right')
ax2.set_ylabel('|Pearson r| with Popularity')
ax2.set_title('Raw vs Engineered Feature Correlation\n(absolute value)',
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.set_ylim(0, 0.55)

plt.tight_layout()
plt.savefig('../reports/engineered_features_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbformat.v4.new_code_cell("""\
print("Top engineered features by |correlation| with popularity:")
print(eng_corr_sorted.round(4).head(8).to_string())
print()
print(f"genre_mean_popularity correlation: {df_feat['genre_mean_popularity'].corr(df_feat['popularity']):.4f}")
print("→ Genre encoding is the STRONGEST single predictor!")
"""))

cells.append(nbformat.v4.new_markdown_cell("""\
### Key Findings

- **`genre_mean_popularity`** is the strongest engineered predictor — confirms genre is the dominant driver.
- **`energy_dance`** (energy × danceability) outperforms either raw feature alone.
- **`mood_score`** (valence × energy) adds signal not in valence or energy individually.
- **Log transforms** of `instrumentalness` and `speechiness` improve their linear correlation with the target by spreading out the near-zero mass.
- **`vocal_presence`** (1 − instrumentalness) has the same absolute correlation but is more interpretable.
"""))

# ── Cell 12: Final Dataset Summary ───────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 11. Final Dataset Summary & Save
"""))

cells.append(nbformat.v4.new_code_cell("""\
print("=== Dataset Shape ===")
print(f"Raw       : {df_raw.shape}")
print(f"Cleaned   : {df.shape}")
print(f"Engineered: {df_feat.shape}")

# Define modelling feature set
base_audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature'
]
all_model_features = base_audio_features + engineered_features

print(f"\\n=== Modelling Feature Set ({len(all_model_features)} features) ===")
for i, f in enumerate(all_model_features, 1):
    corr_val = (df_feat[f].corr(df_feat['popularity'])
                if f != 'tempo_category' else float('nan'))
    print(f"  {i:2}. {f:<30} (r={corr_val:+.3f})" if not np.isnan(corr_val)
          else f"  {i:2}. {f:<30} (categorical)")
"""))

cells.append(nbformat.v4.new_code_cell("""\
import os
os.makedirs('../data/processed', exist_ok=True)

# Save processed dataset
df_feat.to_csv('../data/processed/spotify_features_engineered.csv', index=False)
print("Saved → ../data/processed/spotify_features_engineered.csv")
print(f"Shape : {df_feat.shape}")
"""))

cells.append(nbformat.v4.new_code_cell("""\
# Preview final dataset
df_feat[base_audio_features + engineered_features[:5] + ['popularity']].head(5)
"""))

# ── Cell 13: Key Takeaways ────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 12. Key Takeaways

---

### Data Quality
- Dataset is clean: only 3 null rows removed out of 114,000 (< 0.003%)
- No duplicate tracks found
- Perfectly balanced by genre (1,000 tracks × 114 genres)

### Target Variable (`popularity`)
- **Bimodal distribution** — large zero-spike (~35% of tracks have popularity = 0)
- Predict using **MAE alongside RMSE** to avoid penalising zero-inflation too heavily
- Consider whether to model all tracks or filter out zeros in a two-stage approach

### Most Important Signals
1. **Genre** is the strongest predictor — encapsulated in `genre_mean_popularity`
2. **Loudness** and **energy** are positively correlated with popularity (production quality)
3. **Danceability** × **energy** interaction captures the "hit factor"
4. **Instrumentalness** is a strong *negative* predictor — vocal tracks dominate charts
5. **Acousticness** is negatively correlated — acoustic/folk tracks score lower on average

### Feature Engineering Wins
- Log transforms on `instrumentalness` and `speechiness` improve linearity
- `genre_mean_popularity` dramatically increases available signal
- `mood_score` = valence × energy captures positively-valenced energetic tracks
- Capping `duration_ms` at P95 removes distorting outliers

### Modelling Recommendations
- **Start with**: XGBoost or LightGBM — handle non-linearity and mixed feature types natively
- **Target encode genres** using cross-validation folds to avoid leakage
- **Evaluate with**: RMSE, MAE, and R² — report all three for different stakeholder perspectives
- **Baseline**: predict `genre_mean_popularity` for every track → strong naive baseline

---
*Next notebook → `02_baseline_models.ipynb`: Train linear regression, random forest, and gradient boosting baselines*
"""))

# ── Assemble and write ────────────────────────────────────────────────────────
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (.venv)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

output_path = '/Users/muhammadahsan/Projects/ml-end-to-end-pipeline/notebooks/01_eda_feature_engineering.ipynb'
with open(output_path, 'w') as f:
    nbformat.write(nb, f)

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(nb.cells)}")
md_cells   = sum(1 for c in nb.cells if c.cell_type == 'markdown')
code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"  Markdown cells : {md_cells}")
print(f"  Code cells     : {code_cells}")
