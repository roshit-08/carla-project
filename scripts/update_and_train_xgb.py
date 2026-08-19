"""
Script to update notebooks/xgboost_harsh_driving_v2.ipynb with:
1. Magnetometer removal
2. S-curve & net yaw engineered features
3. Split manifest matching CNN (artifacts/split_manifest.csv)
4. Train & evaluate XGBoost and save joblib artifact.
"""
import json
from pathlib import Path

NB_PATH = Path("notebooks/xgboost_harsh_driving_v2.ipynb")
with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# ── CELL 3: Imports & Config ──────────────────────────────────────────────────
cells[2]["source"] = """\
# ── Cell 3: Imports & Configuration ──────────────────────────────────────────
from __future__ import annotations

import os
import sys
import random
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import xgboost as xgb

from imblearn.over_sampling import SMOTE, RandomOverSampler
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATASET_DIR = ROOT / "dataset"
SRC_DIR     = ROOT / "src"
ARTIFACTS_DIR = ROOT / "notebooks" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from harsh_event_model import load_row_labeled_data

# IMU columns without magnetometer
IMU_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# ── Window Parameters ─────────────────────────────────────────────────────────
WINDOW_SIZE     = 25
STEP_SIZE       = 5
MIN_EVENT_RATIO = 0.30   # fraction of window that must be non-safe to label as event

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

print(f"ROOT          : {ROOT}")
print(f"DATASET_DIR   : {DATASET_DIR}  (exists={DATASET_DIR.exists()})")
print(f"ARTIFACTS_DIR : {ARTIFACTS_DIR}")
print(f"IMU columns (No Mag) : {IMU_COLUMNS}")
"""

# ── CELL 4: Data Loading & S-Curve Feature Engineering ────────────────────────
cells[3]["source"] = """\
# ── Cell 4: Data Loading & Derived Row-Level Features ─────────────────────────
raw_df = load_row_labeled_data(DATASET_DIR)

print(f"Total rows: {len(raw_df):,}")
print(f"Source files: {raw_df['source_file'].nunique()}")
print(f"\\nEvent label distribution:")
print(raw_df["event_label"].value_counts())

# ── Median-filter smoothing per file ──────────────────────────────────────────
clean_df = raw_df.copy()
for col in IMU_COLUMNS:
    clean_df[col] = (
        clean_df.groupby("source_file")[col]
        .transform(lambda s: s.rolling(window=3, min_periods=1, center=True).median())
    )

# ── S-Curve & Net Yaw Feature Engineering ─────────────────────────────────────
def add_engineered_features(df):
    df = df.copy()

    # Magnitudes (acc + gyro only)
    df['acc_mag']  = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

    # Jerk
    for axis in ['acc_x', 'acc_y', 'acc_z']:
        df[f'{axis}_jerk'] = df.groupby('source_file')[axis].transform(
            lambda s: s.diff().fillna(0.0)
        )

    # Lateral / longitudinal ratios and energy
    df['lat_long_ratio']        = df['acc_y'].abs() / (df['acc_x'].abs() + 1e-6)
    df['acc_x_abs']             = df['acc_x'].abs()
    df['acc_y_abs']             = df['acc_y'].abs()
    df['gyro_z_abs']            = df['gyro_z'].abs()
    df['acc_lat_energy']        = df['acc_y'] ** 2
    df['acc_long_energy']       = df['acc_x'] ** 2
    df['gyro_yaw_energy']       = df['gyro_z'] ** 2
    df['lat_long_energy_ratio'] = df['acc_y_abs'] / (df['acc_x_abs'] + 1e-6)
    df['turn_vs_lateral']       = df['gyro_z_abs'] / (df['acc_y_abs'] + 1e-6)
    df['yaw_acc_corr']          = df['gyro_z'] * df['acc_y']

    # Differentials
    df['gyro_z_diff'] = df.groupby('source_file')['gyro_z'].transform(
        lambda s: s.diff().fillna(0.0)
    )
    df['acc_y_diff'] = df.groupby('source_file')['acc_y'].transform(
        lambda s: s.diff().fillna(0.0)
    )

    # Rolling statistics
    for col in ['gyro_z', 'acc_y', 'gyro_x', 'gyro_y']:
        df[f'{col}_roll_std5'] = df.groupby('source_file')[col].transform(
            lambda s: s.rolling(5, min_periods=1).std().fillna(0.0)
        )
    df['acc_y_roll_mean5'] = df.groupby('source_file')['acc_y'].transform(
        lambda s: s.rolling(5, min_periods=1).mean().fillna(0.0)
    )

    # Sign change (oscillation indicator)
    df['gyro_z_sign_change'] = df.groupby('source_file')['gyro_z'].transform(
        lambda s: (s.shift(1) * s < 0).astype(int).fillna(0)
    )

    # Roll / pitch features
    df['gyro_x_abs']          = df['gyro_x'].abs()
    df['gyro_y_abs']          = df['gyro_y'].abs()
    df['gyro_x_energy']       = df['gyro_x'] ** 2
    df['gyro_y_energy']       = df['gyro_y'] ** 2
    df['gyro_roll_pitch_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2)
    df['gyro_total_mag']      = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
    df['yaw_vs_roll_pitch']   = df['gyro_z_abs'] / (df['gyro_roll_pitch_mag'] + 1e-6)

    # Sustained-yaw peak features
    df['gyro_z_roll_max5']    = df.groupby('source_file')['gyro_z_abs'].transform(
        lambda s: s.rolling(5,  min_periods=1).max().fillna(0.0)
    )
    df['gyro_z_roll_max10']   = df.groupby('source_file')['gyro_z_abs'].transform(
        lambda s: s.rolling(10, min_periods=1).max().fillna(0.0)
    )
    df['gyro_z_roll_energy5'] = df.groupby('source_file')['gyro_yaw_energy'].transform(
        lambda s: s.rolling(5, min_periods=1).mean().fillna(0.0)
    )
    df['yaw_dominance']       = df['gyro_z_abs'] / (df['gyro_total_mag'] + 1e-6)

    # ── S-CURVE & NET YAW FEATURES FOR LANE CHANGE DISCRIMINATION ─────────────
    df['net_yaw_cumsum20'] = df.groupby('source_file')['gyro_z'].transform(
        lambda s: s.rolling(20, min_periods=1).sum().fillna(0.0)
    )
    df['gyro_z_roll_min20'] = df.groupby('source_file')['gyro_z'].transform(
        lambda s: s.rolling(20, min_periods=1).min().fillna(0.0)
    )
    df['gyro_z_roll_max20'] = df.groupby('source_file')['gyro_z'].transform(
        lambda s: s.rolling(20, min_periods=1).max().fillna(0.0)
    )
    df['biphasic_yaw_signature'] = df['gyro_z_roll_min20'] * df['gyro_z_roll_max20']
    df['acc_y_jerk_std5'] = df.groupby('source_file')['acc_y_jerk'].transform(
        lambda s: s.rolling(5, min_periods=1).std().fillna(0.0)
    )

    return df

clean_df = add_engineered_features(clean_df)

FEATURE_CHANNELS = IMU_COLUMNS + [
    'acc_mag', 'gyro_mag',
    'acc_x_jerk', 'acc_y_jerk', 'acc_z_jerk',
    'lat_long_ratio',
    'acc_x_abs', 'acc_y_abs', 'gyro_z_abs',
    'acc_lat_energy', 'acc_long_energy', 'gyro_yaw_energy',
    'lat_long_energy_ratio', 'turn_vs_lateral', 'yaw_acc_corr',
    'gyro_z_diff', 'acc_y_diff',
    'gyro_z_roll_std5', 'acc_y_roll_std5', 'acc_y_roll_mean5',
    'gyro_z_sign_change',
    'gyro_x_abs', 'gyro_y_abs', 'gyro_x_energy', 'gyro_y_energy',
    'gyro_roll_pitch_mag', 'gyro_total_mag', 'yaw_vs_roll_pitch',
    'gyro_x_roll_std5', 'gyro_y_roll_std5',
    'gyro_z_roll_max5', 'gyro_z_roll_max10',
    'gyro_z_roll_energy5', 'yaw_dominance',
    'net_yaw_cumsum20', 'biphasic_yaw_signature', 'acc_y_jerk_std5',
]

print(f"\\nFeature channels count ({len(FEATURE_CHANNELS)}): {FEATURE_CHANNELS}")
"""

# ── CELL 7: Split Manifest Alignment (Matching CNN dataset split) ─────────────
cells[6]["source"] = """\
# ── Cell 7: Split Manifest Alignment ─────────────────────────────────────────
# Read split manifest to guarantee train/val/test splits match CNN exactly
manifest_path = ROOT / 'artifacts' / 'split_manifest.csv'
if manifest_path.exists():
    split_df = pd.read_csv(manifest_path)
    train_files = set(split_df[split_df['split'] == 'train']['group_file'])
    val_files   = set(split_df[split_df['split'] == 'val']['group_file'])
    test_files  = set(split_df[split_df['split'] == 'test']['group_file'])
    print("Loaded split_manifest.csv successfully.")
    print(f"Train files: {len(train_files)} | Val files: {len(val_files)} | Test files: {len(test_files)}")
else:
    print("split_manifest.csv not found, using GroupShuffleSplit fallback.")

# Build Window Dataset
X_df, y_series, groups_series = build_window_dataset(
    clean_df,
    feature_channels=FEATURE_CHANNELS,
    window_size=WINDOW_SIZE,
    step_size=STEP_SIZE,
    min_event_ratio=MIN_EVENT_RATIO,
)

le = LabelEncoder()
label_order = ['harsh_left_lane_change', 'harsh_left_turn', 'harsh_right_lane_change',
               'harsh_right_turn', 'safe', 'sudden_acceleration', 'sudden_braking']
le.fit(label_order)
y_encoded = le.transform(y_series)

if manifest_path.exists():
    train_mask = groups_series.isin(train_files)
    val_mask   = groups_series.isin(val_files)
    test_mask  = groups_series.isin(test_files)

    X_train, y_train = X_df[train_mask], y_encoded[train_mask]
    X_val,   y_val   = X_df[val_mask],   y_encoded[val_mask]
    X_test,  y_test  = X_df[test_mask],  y_encoded[test_mask]
else:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val_idx, test_idx = next(gss.split(X_df, y_encoded, groups_series))
    X_test, y_test = X_df.iloc[test_idx], y_encoded[test_idx]
    train_val_groups = groups_series.iloc[train_val_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=SEED)
    train_idx_rel, val_idx_rel = next(gss_val.split(X_df.iloc[train_val_idx], y_encoded[train_val_idx], train_val_groups))

    train_idx = train_val_idx[train_idx_rel]
    val_idx   = train_val_idx[val_idx_rel]

    X_train, y_train = X_df.iloc[train_idx], y_encoded[train_idx]
    X_val,   y_val   = X_df.iloc[val_idx],   y_encoded[val_idx]

print(f"Split sizes | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
"""

# Clear outputs across all cells
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("✓ XGBoost notebook patched with Magnetometer removal, S-curve features, and matching split manifest.")
