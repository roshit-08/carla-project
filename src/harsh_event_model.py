from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

IMU_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
]

EVENT_CLASSES = [
    "safe",
    "sudden_acceleration",
    "sudden_braking",
    "harsh_left_turn",
    "harsh_right_turn",
    "harsh_left_lane_change",
    "harsh_right_lane_change",
]


def normalize_event_name(raw_name: str) -> str:
    text = str(raw_name).strip().lower().replace("_", "-")

    if "safe" in text or "no-movement" in text:
        return "safe"
    if "sudden-acc" in text or "acceleration" in text:
        return "sudden_acceleration"
    if "sudden-brake" in text or "braking" in text:
        return "sudden_braking"
    if "left-turn" in text:
        return "harsh_left_turn"
    if "right-turn" in text:
        return "harsh_right_turn"
    if "left-line-chg" in text or ("left" in text and "lane" in text):
        return "harsh_left_lane_change"
    if "right-line-chg" in text or ("right" in text and "lane" in text):
        return "harsh_right_lane_change"

    return "safe"


def infer_event_from_filename(file_name: str) -> str:
    return normalize_event_name(file_name)


def _read_file_list(dataset_dir: Path, file_list_path: Optional[Path]) -> List[Path]:
    if file_list_path is None or not file_list_path.exists():
        return sorted(dataset_dir.glob("*.csv"))

    files: List[Path] = []
    with file_list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if not name:
                continue
            candidate = dataset_dir / name
            if candidate.exists() and candidate.suffix.lower() == ".csv":
                files.append(candidate)

    if files:
        return files
    return sorted(dataset_dir.glob("*.csv"))


def load_row_labeled_data(dataset_dir: str | Path, file_list_name: str = "raw_data_name.txt") -> pd.DataFrame:
    dataset_path = Path(dataset_dir)
    file_list_path = dataset_path / file_list_name
    csv_paths = _read_file_list(dataset_path, file_list_path)

    frames: List[pd.DataFrame] = []
    required_cols = set(IMU_COLUMNS + ["event_class", "harsh_event", "timestamp", "road_type"])

    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        if not required_cols.issubset(frame.columns):
            missing = sorted(required_cols - set(frame.columns))
            raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

        for col in IMU_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        frame["event_class"] = pd.to_numeric(frame["event_class"], errors="coerce").fillna(0).astype(int)
        frame = frame.dropna(subset=IMU_COLUMNS).copy()

        file_event = infer_event_from_filename(csv_path.name)
        harsh_event_mode = normalize_event_name(frame["harsh_event"].mode(dropna=True).iloc[0])
        event_name = file_event if file_event != "safe" else harsh_event_mode

        frame["event_label"] = np.where(frame["event_class"] == 1, event_name, "safe")
        frame["source_file"] = csv_path.name
        frames.append(frame)

    if not frames:
        raise ValueError(f"No CSV files found in {dataset_path}")

    data = pd.concat(frames, ignore_index=True)
    data["event_label"] = data["event_label"].apply(normalize_event_name)
    return data


def _window_features(window_df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for col in columns:
        values = window_df[col].to_numpy(dtype=float)
        if values.size == 0:
            continue

        abs_values = np.abs(values)
        diff = np.diff(values)
        pos_values = values[values > 0]
        neg_values = values[values < 0]

        features[f"{col}_mean"] = float(np.mean(values))
        features[f"{col}_std"] = float(np.std(values))
        features[f"{col}_median"] = float(np.median(values))
        features[f"{col}_min"] = float(np.min(values))
        features[f"{col}_max"] = float(np.max(values))
        features[f"{col}_range"] = float(np.max(values) - np.min(values))
        features[f"{col}_q25"] = float(np.percentile(values, 25))
        features[f"{col}_q75"] = float(np.percentile(values, 75))
        features[f"{col}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        features[f"{col}_rms"] = float(np.sqrt(np.mean(np.square(values))))
        features[f"{col}_energy"] = float(np.sum(np.square(values)))
        features[f"{col}_abs_sum"] = float(np.sum(abs_values))
        features[f"{col}_abs_mean"] = float(np.mean(abs_values))
        features[f"{col}_abs_std"] = float(np.std(abs_values))
        features[f"{col}_delta"] = float(values[-1] - values[0])
        features[f"{col}_first"] = float(values[0])
        features[f"{col}_last"] = float(values[-1])
        features[f"{col}_pos_ratio"] = float(np.count_nonzero(values > 0) / values.size)
        features[f"{col}_neg_ratio"] = float(np.count_nonzero(values < 0) / values.size)
        features[f"{col}_pos_mean"] = float(np.mean(pos_values)) if pos_values.size else 0.0
        features[f"{col}_neg_mean"] = float(np.mean(neg_values)) if neg_values.size else 0.0
        features[f"{col}_diff_mean"] = float(np.mean(diff)) if diff.size else 0.0
        features[f"{col}_diff_std"] = float(np.std(diff)) if diff.size else 0.0
        features[f"{col}_diff_skew"] = float(pd.Series(diff).skew()) if diff.size else 0.0
        features[f"{col}_diff_kurtosis"] = float(pd.Series(diff).kurtosis()) if diff.size else 0.0
        features[f"{col}_skew"] = float(pd.Series(values).skew())
        features[f"{col}_kurtosis"] = float(pd.Series(values).kurtosis())

    return features


def build_window_dataset(
    row_data: pd.DataFrame,
    window_size: int = 25,
    step_size: int = 5,
    min_event_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.Series]:
    feature_rows: List[Dict[str, float]] = []
    labels: List[str] = []

    for _, file_df in row_data.groupby("source_file", sort=True):
        file_df = file_df.reset_index(drop=True)
        if len(file_df) < window_size:
            continue

        for start in range(0, len(file_df) - window_size + 1, step_size):
            window = file_df.iloc[start : start + window_size]
            non_safe = window.loc[window["event_label"] != "safe", "event_label"]

            if non_safe.empty or (len(non_safe) / window_size) < min_event_ratio:
                label = "safe"
            else:
                label = non_safe.value_counts().idxmax()

            features = _window_features(window, IMU_COLUMNS)
            feature_rows.append(features)
            labels.append(label)

    if not feature_rows:
        raise ValueError("No windows created. Reduce window_size or verify input data.")

    x = pd.DataFrame(feature_rows).fillna(0.0)
    y = pd.Series(labels, name="label")
    return x, y


@dataclass
class TrainingArtifacts:
    model: Pipeline
    label_encoder: LabelEncoder
    feature_columns: List[str]
    class_report: str
    confusion_matrix: np.ndarray
    val_class_report: Optional[str] = None
    val_confusion_matrix: Optional[np.ndarray] = None


def _build_rf_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=600,
                    max_depth=24,
                    min_samples_split=6,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_event_model_from_splits(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> TrainingArtifacts:
    label_encoder = LabelEncoder()
    all_labels = pd.concat([y_train, y_val, y_test], axis=0).astype(str)
    label_encoder.fit(all_labels)

    y_train_encoded = label_encoder.transform(y_train.astype(str))
    y_val_encoded = label_encoder.transform(y_val.astype(str))
    y_test_encoded = label_encoder.transform(y_test.astype(str))

    model = _build_rf_pipeline(random_state=random_state)
    model.fit(x_train, y_train_encoded)

    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    all_label_ids = np.arange(len(label_encoder.classes_))
    target_names = list(label_encoder.classes_)

    val_report = classification_report(
        y_val_encoded,
        y_val_pred,
        labels=all_label_ids,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    val_cm = confusion_matrix(y_val_encoded, y_val_pred, labels=all_label_ids)

    test_report = classification_report(
        y_test_encoded,
        y_test_pred,
        labels=all_label_ids,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    test_cm = confusion_matrix(y_test_encoded, y_test_pred, labels=all_label_ids)

    return TrainingArtifacts(
        model=model,
        label_encoder=label_encoder,
        feature_columns=list(x_train.columns),
        class_report=test_report,
        confusion_matrix=test_cm,
        val_class_report=val_report,
        val_confusion_matrix=val_cm,
    )


def train_event_model(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> TrainingArtifacts:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    model = _build_rf_pipeline(random_state=random_state)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.inverse_transform(np.unique(y_test)),
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    return TrainingArtifacts(
        model=model,
        label_encoder=label_encoder,
        feature_columns=list(x.columns),
        class_report=report,
        confusion_matrix=cm,
    )


def save_model_bundle(
    output_path: str | Path,
    artifacts: TrainingArtifacts,
    window_size: int,
    step_size: int,
    min_event_ratio: float,
) -> None:
    bundle = {
        "model": artifacts.model,
        "label_encoder": artifacts.label_encoder,
        "feature_columns": artifacts.feature_columns,
        "window_size": window_size,
        "step_size": step_size,
        "min_event_ratio": min_event_ratio,
        "imu_columns": IMU_COLUMNS,
    }
    joblib.dump(bundle, output_path, protocol=4)


class RealtimeHarshEventDetector:
    def __init__(
        self,
        model_bundle_path: str | Path,
        min_confidence: float = 0.45,
        consecutive_hits: int = 1,
        heuristic_enabled: bool = True,
        accel_threshold: float = 2.0,
        brake_threshold: float = -2.0,
        turn_threshold: float = 40.0,
        lane_change_threshold: float = 2.0,
        invert_gyro_z: bool = False,
        invert_acc_y: bool = False,
    ) -> None:
        self.bundle = joblib.load(model_bundle_path)
        self.model: Pipeline = self.bundle["model"]
        self.label_encoder: LabelEncoder = self.bundle["label_encoder"]
        self.feature_columns: List[str] = self.bundle["feature_columns"]
        self.window_size: int = int(self.bundle["window_size"])
        self.step_size: int = int(self.bundle["step_size"])
        self.imu_columns: List[str] = list(self.bundle["imu_columns"])
        self.min_confidence = float(min_confidence)
        self.consecutive_hits = int(consecutive_hits)

        self.heuristic_enabled = bool(heuristic_enabled)
        self.accel_threshold = float(accel_threshold)
        self.brake_threshold = float(brake_threshold)
        self.turn_threshold = float(turn_threshold)
        self.lane_change_threshold = float(lane_change_threshold)
        self.invert_gyro_z = bool(invert_gyro_z)
        self.invert_acc_y = bool(invert_acc_y)

        self.buffer: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self._tick = 0
        self._active_label: Optional[str] = None
        self._candidate_label: Optional[str] = None
        self._candidate_hits = 0

    def _build_feature_row(self) -> pd.DataFrame:
        window_df = pd.DataFrame(list(self.buffer))
        features = _window_features(window_df, self.imu_columns)
        row = pd.DataFrame([features])

        for feature_name in self.feature_columns:
            if feature_name not in row.columns:
                row[feature_name] = 0.0

        row = row[self.feature_columns]
        return row

    def _smooth_label(self, raw_label: str) -> str:
        if self._active_label is None:
            if raw_label == "safe":
                return "safe"
            self._candidate_label = raw_label
            self._candidate_hits = 1
            if self._candidate_hits >= self.consecutive_hits:
                self._active_label = raw_label
            return self._active_label or "safe"

        if raw_label == self._active_label:
            self._candidate_label = None
            self._candidate_hits = 0
            return self._active_label

        if raw_label == self._candidate_label:
            self._candidate_hits += 1
        else:
            self._candidate_label = raw_label
            self._candidate_hits = 1

        if self._candidate_hits >= self.consecutive_hits:
            self._active_label = raw_label if raw_label != "safe" else None
            self._candidate_label = None
            self._candidate_hits = 0

        return self._active_label or "safe"

    def update(self, sample: Dict[str, float]) -> Optional[Dict[str, object]]:
        cleaned = {}
        for col in self.imu_columns:
            if col not in sample:
                raise KeyError(f"Missing IMU column in sample: {col}")
            cleaned[col] = float(sample[col])

        if self.invert_gyro_z:
            cleaned["gyro_z"] = -cleaned["gyro_z"]
        if self.invert_acc_y:
            cleaned["acc_y"] = -cleaned["acc_y"]

        self.buffer.append(cleaned)
        self._tick += 1

        if len(self.buffer) < self.window_size:
            return {
                "ready": False,
                "reason": "warming_up",
                "buffer_size": len(self.buffer),
                "required": self.window_size,
            }

        if (self._tick % self.step_size) != 0:
            return None

        row = self._build_feature_row()
        probabilities = self.model.predict_proba(row)[0]
        pred_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_idx])
        raw_label = str(self.label_encoder.inverse_transform([pred_idx])[0])

        if confidence < self.min_confidence:
            raw_label = "safe"

        # heuristic fallback for sharper events when ml model is uncertain.
        if self.heuristic_enabled and raw_label == "safe" and self.buffer:
            last = self.buffer[-1]
            acc_x = last.get("acc_x", 0.0)
            acc_y = last.get("acc_y", 0.0)
            gyro_z = last.get("gyro_z", 0.0)

            if acc_x >= self.accel_threshold:
                raw_label = "sudden_acceleration"
                confidence = max(confidence, 0.50)
            elif acc_x <= self.brake_threshold:
                raw_label = "sudden_braking"
                confidence = max(confidence, 0.50)
            elif gyro_z >= self.turn_threshold:
                raw_label = "harsh_right_turn"
                confidence = max(confidence, 0.50)
            elif gyro_z <= -self.turn_threshold:
                raw_label = "harsh_left_turn"
                confidence = max(confidence, 0.50)
            elif acc_y >= self.lane_change_threshold:
                raw_label = "harsh_right_lane_change"
                confidence = max(confidence, 0.50)
            elif acc_y <= -self.lane_change_threshold:
                raw_label = "harsh_left_lane_change"
                confidence = max(confidence, 0.50)

        smoothed_label = self._smooth_label(raw_label)

        class_names = self.label_encoder.inverse_transform(np.arange(len(probabilities)))
        proba_dict = {cls: float(prob) for cls, prob in zip(class_names, probabilities)}

        return {
            "ready": True,
            "raw_prediction": raw_label,
            "prediction": smoothed_label,
            "confidence": confidence,
            "probabilities": proba_dict,
        }