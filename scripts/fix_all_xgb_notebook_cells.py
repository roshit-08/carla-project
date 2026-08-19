import json
from pathlib import Path

NB_PATH = Path("notebooks/xgboost_harsh_driving_v2.ipynb")
with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# Fix Cell 8 (Cell index 7)
cells[7]["source"] = """\
# ── Cell 8: RandomOverSampler / SMOTE on Training Split ──────────────────────────────
# Applied ONLY to the training data — val/test remain untouched.

print("Before Resampling:")
print(pd.Series(y_train).value_counts().rename(index=dict(enumerate(le.classes_))))

ros = RandomOverSampler(random_state=SEED)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

print("\\nAfter Resampling:")
print(pd.Series(y_train_res).value_counts().rename(index=dict(enumerate(le.classes_))))
print(f"\\nTraining samples: {len(X_train):,} → {len(X_train_res):,} after resampling")
"""

# Fix Cell 9 (Optuna tuning)
cells[8]["source"] = """\
# ── Cell 9: Optuna Hyperparameter Optimization ────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    model = xgb.XGBClassifier(
        **params,
        random_state=SEED,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    classes = np.unique(y_train_res)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_res)
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([class_weight_dict[y] for y in y_train_res])

    model.fit(
        X_train_res,
        y_train_res,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = model.predict(X_val)
    return float(f1_score(y_val, preds, average="macro"))

print("Running Optuna study (15 trials)...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=15, timeout=120)

print(f"\\nBest trial macro-F1 : {study.best_value:.4f}")
print("Best parameters:")
for k, v in study.best_params.items():
    print(f"  {k:<20}: {v}")
"""

# Fix Cell 10 (Final Training)
cells[9]["source"] = """\
# ── Cell 10: Final Model Training ─────────────────────────────────────────────
best_params = study.best_params

final_xgb = xgb.XGBClassifier(
    **best_params,
    random_state=SEED,
    n_jobs=-1,
    eval_metric="mlogloss",
)

classes = np.unique(y_train_res)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_res)
class_weight_dict = dict(zip(classes, weights))
sample_weights = np.array([class_weight_dict[y] for y in y_train_res])

final_xgb.fit(
    X_train_res,
    y_train_res,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

val_preds = final_xgb.predict(X_val)
val_f1 = f1_score(y_val, val_preds, average="macro")
print(f"Validation Macro F1: {val_f1:.4f}")
"""

# Fix Cell 11 (Test Set Evaluation)
cells[10]["source"] = """\
# ── Cell 11: Final Test Evaluation & Confusion Matrix ────────────────────────
test_preds = final_xgb.predict(X_test)

macro_f1 = f1_score(y_test, test_preds, average="macro")
print("=" * 65)
print(f"FINAL TEST SET EVALUATION (Macro F1 = {macro_f1:.4f})")
print("=" * 65)
print(classification_report(y_test, test_preds, target_names=le.classes_, digits=4))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
)
plt.title("XGBoost v2 — Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "xgb_v2_confusion_matrix.png", bbox_inches="tight")
plt.show()
"""

# Fix Cell 12 (Export Bundle)
cells[11]["source"] = """\
# ── Cell 12: Export Joblib Model Bundle ──────────────────────────────────────
bundle = {
    "model": final_xgb,
    "label_encoder": le,
    "feature_names": list(X_train.columns),
    "feature_channels": FEATURE_CHANNELS,
    "window_size": WINDOW_SIZE,
    "step_size": STEP_SIZE,
    "min_event_ratio": MIN_EVENT_RATIO,
    "macro_f1": macro_f1,
    "classes": list(le.classes_),
}

bundle_path = ARTIFACTS_DIR / "harsh_event_xgb_v2.joblib"
joblib.dump(bundle, bundle_path)
print(f"Saved XGBoost model bundle to: {bundle_path}")
"""

for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("✓ Updated cells 8-12 in XGBoost notebook successfully.")
