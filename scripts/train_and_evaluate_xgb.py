"""
Train and Evaluate XGBoost Model with S-Curve Features & Magnetometer-Free Setup
"""
import sys
import random
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATASET_DIR = ROOT / "dataset"
SRC_DIR     = ROOT / "src"
ARTIFACTS_DIR = ROOT / "notebooks" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from harsh_event_model import load_row_labeled_data

IMU_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
WINDOW_SIZE = 25
STEP_SIZE = 5
MIN_EVENT_RATIO = 0.30

# Load raw dataset
print("Loading dataset...")
raw_df = load_row_labeled_data(DATASET_DIR)

# Median smoothing
clean_df = raw_df.copy()
for col in IMU_COLUMNS:
    clean_df[col] = (
        clean_df.groupby("source_file")[col]
        .transform(lambda s: s.rolling(window=3, min_periods=1, center=True).median())
    )

# Feature engineering
def add_engineered_features(df):
    df = df.copy()
    df['acc_mag']  = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

    for axis in ['acc_x', 'acc_y', 'acc_z']:
        df[f'{axis}_jerk'] = df.groupby('source_file')[axis].transform(lambda s: s.diff().fillna(0.0))

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

    df['gyro_z_diff'] = df.groupby('source_file')['gyro_z'].transform(lambda s: s.diff().fillna(0.0))
    df['acc_y_diff']  = df.groupby('source_file')['acc_y'].transform(lambda s: s.diff().fillna(0.0))

    for col in ['gyro_z', 'acc_y', 'gyro_x', 'gyro_y']:
        df[f'{col}_roll_std5'] = df.groupby('source_file')[col].transform(lambda s: s.rolling(5, min_periods=1).std().fillna(0.0))
    df['acc_y_roll_mean5'] = df.groupby('source_file')['acc_y'].transform(lambda s: s.rolling(5, min_periods=1).mean().fillna(0.0))

    df['gyro_z_sign_change'] = df.groupby('source_file')['gyro_z'].transform(lambda s: (s.shift(1) * s < 0).astype(int).fillna(0))

    df['gyro_x_abs']          = df['gyro_x'].abs()
    df['gyro_y_abs']          = df['gyro_y'].abs()
    df['gyro_x_energy']       = df['gyro_x'] ** 2
    df['gyro_y_energy']       = df['gyro_y'] ** 2
    df['gyro_roll_pitch_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2)
    df['gyro_total_mag']      = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
    df['yaw_vs_roll_pitch']   = df['gyro_z_abs'] / (df['gyro_roll_pitch_mag'] + 1e-6)

    df['gyro_z_roll_max5']    = df.groupby('source_file')['gyro_z_abs'].transform(lambda s: s.rolling(5, min_periods=1).max().fillna(0.0))
    df['gyro_z_roll_max10']   = df.groupby('source_file')['gyro_z_abs'].transform(lambda s: s.rolling(10, min_periods=1).max().fillna(0.0))
    df['gyro_z_roll_energy5'] = df.groupby('source_file')['gyro_yaw_energy'].transform(lambda s: s.rolling(5, min_periods=1).mean().fillna(0.0))
    df['yaw_dominance']       = df['gyro_z_abs'] / (df['gyro_total_mag'] + 1e-6)

    # S-Curve & Net Yaw
    df['net_yaw_cumsum20'] = df.groupby('source_file')['gyro_z'].transform(lambda s: s.rolling(20, min_periods=1).sum().fillna(0.0))
    df['gyro_z_roll_min20'] = df.groupby('source_file')['gyro_z'].transform(lambda s: s.rolling(20, min_periods=1).min().fillna(0.0))
    df['gyro_z_roll_max20'] = df.groupby('source_file')['gyro_z'].transform(lambda s: s.rolling(20, min_periods=1).max().fillna(0.0))
    df['biphasic_yaw_signature'] = df['gyro_z_roll_min20'] * df['gyro_z_roll_max20']
    df['acc_y_jerk_std5'] = df.groupby('source_file')['acc_y_jerk'].transform(lambda s: s.rolling(5, min_periods=1).std().fillna(0.0))

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

def extract_window_summary_features(window_2d: np.ndarray, col_names: list) -> dict:
    feats = {}
    for i, col in enumerate(col_names):
        x = window_2d[:, i].astype(np.float64)
        dx = np.diff(x)
        abs_x = np.abs(x)

        feats[f"{col}_mean"]     = np.mean(x)
        feats[f"{col}_std"]      = np.std(x)
        feats[f"{col}_min"]      = np.min(x)
        feats[f"{col}_max"]      = np.max(x)
        feats[f"{col}_range"]    = np.max(x) - np.min(x)
        feats[f"{col}_rms"]      = float(np.sqrt(np.mean(x**2)))
        feats[f"{col}_abs_mean"] = float(np.mean(abs_x))
        feats[f"{col}_abs_max"]  = float(np.max(abs_x))
        feats[f"{col}_delta"]    = float(x[-1] - x[0])
        feats[f"{col}_diff_std"] = float(np.std(dx)) if dx.size else 0.0

    # Cross-sensor interactions
    gz_idx = col_names.index("gyro_z")
    ay_idx = col_names.index("acc_y")
    gz = window_2d[:, gz_idx]
    ay = window_2d[:, ay_idx]
    feats["corr_ay_gz"] = float(np.corrcoef(ay, gz)[0, 1]) if np.std(ay) > 0 and np.std(gz) > 0 else 0.0

    return feats

def build_window_dataset(df, feature_channels, window_size=25, step_size=5, min_event_ratio=0.30):
    x_rows, y_rows, group_rows = [], [], []
    for file_name, g in df.groupby("source_file"):
        g = g.reset_index(drop=True)
        if len(g) < window_size:
            continue
        values = g[feature_channels].to_numpy(dtype=np.float32)
        labels = g["event_label"].to_numpy()

        for start in range(0, len(g) - window_size + 1, step_size):
            window_vals   = values[start : start + window_size]
            window_labels = labels[start : start + window_size]
            non_safe = window_labels[window_labels != "safe"]
            if len(non_safe) == 0 or (len(non_safe) / window_size) < min_event_ratio:
                target = "safe"
            else:
                target = Counter(non_safe).most_common(1)[0][0]

            feats = extract_window_summary_features(window_vals, feature_channels)
            x_rows.append(feats)
            y_rows.append(target)
            group_rows.append(file_name)

    return pd.DataFrame(x_rows), pd.Series(y_rows, name="event_label"), pd.Series(group_rows, name="source_file")

print("Building windowed dataset...")
X_df, y_series, groups_series = build_window_dataset(
    clean_df, feature_channels=FEATURE_CHANNELS, window_size=WINDOW_SIZE, step_size=STEP_SIZE, min_event_ratio=MIN_EVENT_RATIO
)

le = LabelEncoder()
label_order = ['harsh_left_lane_change', 'harsh_left_turn', 'harsh_right_lane_change',
               'harsh_right_turn', 'safe', 'sudden_acceleration', 'sudden_braking']
le.fit(label_order)
y_encoded = le.transform(y_series)

# Split using split_manifest.csv if present
manifest_path = ROOT / 'artifacts' / 'split_manifest.csv'
if manifest_path.exists():
    split_df = pd.read_csv(manifest_path)
    train_files = set(split_df[split_df['split'] == 'train']['group_file'])
    val_files   = set(split_df[split_df['split'] == 'val']['group_file'])
    test_files  = set(split_df[split_df['split'] == 'test']['group_file'])

    train_mask = groups_series.isin(train_files)
    val_mask   = groups_series.isin(val_files)
    test_mask  = groups_series.isin(test_files)

    X_train, y_train = X_df[train_mask], y_encoded[train_mask]
    X_val,   y_val   = X_df[val_mask],   y_encoded[val_mask]
    X_test,  y_test  = X_df[test_mask],  y_encoded[test_mask]
    print("Dataset split using split_manifest.csv successfully.")
else:
    print("split_manifest.csv not found!")
    sys.exit(1)

print(f"Dataset split counts | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Apply RandomOverSampler to train split
ros = RandomOverSampler(random_state=SEED)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# Compute class weights
classes = np.unique(y_train_res)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_res)
class_weight_dict = dict(zip(classes, weights))

# Train XGBoost Classifier
print("Training XGBoost Classifier...")
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
    eval_metric='mlogloss'
)

sample_weights = np.array([class_weight_dict[y] for y in y_train_res])
model.fit(
    X_train_res, y_train_res,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Evaluate on Test Set
y_pred = model.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
print("\n" + "="*60)
print(f"XGBoost Model Test Set Results (Macro F1 = {macro_f1:.4f})")
print("="*60)
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("\nConfusion Matrix:")
print(cm_df)

# Save joblib bundle
bundle = {
    'model': model,
    'label_encoder': le,
    'feature_names': list(X_train.columns),
    'feature_channels': FEATURE_CHANNELS,
    'window_size': WINDOW_SIZE,
    'step_size': STEP_SIZE,
    'min_event_ratio': MIN_EVENT_RATIO,
    'macro_f1': macro_f1,
    'classes': list(le.classes_),
}
joblib.dump(bundle, ARTIFACTS_DIR / "harsh_event_xgb_v2.joblib")
print(f"\nSaved XGBoost model bundle to {ARTIFACTS_DIR / 'harsh_event_xgb_v2.joblib'}")
