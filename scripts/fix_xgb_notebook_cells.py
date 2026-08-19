import json
from pathlib import Path

NB_PATH = Path("notebooks/xgboost_harsh_driving_v2.ipynb")
with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# Update Cell 6 definition of build_window_dataset if needed and Cell 7 mask checking
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
    train_mask = np.isin(groups_series, list(train_files))
    val_mask   = np.isin(groups_series, list(val_files))
    test_mask  = np.isin(groups_series, list(test_files))

    X_train, y_train = X_df[train_mask], y_encoded[train_mask]
    X_val,   y_val   = X_df[val_mask],   y_encoded[val_mask]
    X_test,  y_test  = X_df[test_mask],  y_encoded[test_mask]
else:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val_idx, test_idx = next(gss.split(X_df, y_encoded, groups_series))
    X_test, y_test = X_df.iloc[test_idx], y_encoded[test_idx]
    train_val_groups = pd.Series(groups_series).iloc[train_val_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=SEED)
    train_idx_rel, val_idx_rel = next(gss_val.split(X_df.iloc[train_val_idx], y_encoded[train_val_idx], train_val_groups))

    train_idx = train_val_idx[train_idx_rel]
    val_idx   = train_val_idx[val_idx_rel]

    X_train, y_train = X_df.iloc[train_idx], y_encoded[train_idx]
    X_val,   y_val   = X_df.iloc[val_idx],   y_encoded[val_idx]

print(f"Split sizes | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
"""

for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("✓ Fixed Cell 7 in XGBoost notebook.")
