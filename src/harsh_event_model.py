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

    model = Pipeline(
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


# ==============================================================================
# PyTorch 1D-CNN Sequence Harsh Event Detector
# ==============================================================================

import torch
import torch.nn as nn

class TimeSeriesCNN1D(nn.Module):
    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 4
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, T, F] -> Transpose to [B, F, T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Global average pooling along temporal dimension (dim 2)
        pooled = x.mean(dim=2)
        return self.head(pooled)


class RealtimeCNNHarshEventDetector:
    def __init__(
        self,
        model_bundle_path: str | Path,
        min_confidence: float = 0.35,
        consecutive_hits: int = 2,
        heuristic_enabled: bool = True,
        accel_threshold: float = 2.0,
        brake_threshold: float = -2.0,
        turn_threshold: float = 40.0,
        lane_change_threshold: float = 2.0,
        invert_gyro_z: bool = False,
        invert_acc_y: bool = False,
    ) -> None:
        self.bundle = torch.load(model_bundle_path, map_location=torch.device("cpu"))
        
        self.in_features = int(self.bundle["in_features"])
        self.n_classes = int(self.bundle["n_classes"])
        
        self.model = TimeSeriesCNN1D(self.in_features, self.n_classes)
        self.model.load_state_dict(self.bundle["model_state_dict"])
        self.model.eval()
        
        self.scaler = self.bundle["scaler"]
        self.label_encoder_classes = self.bundle["label_encoder_classes"]
        self.feature_cols_stream = list(self.bundle["feature_cols_stream"])
        self.window_size = int(self.bundle["window_size"])
        self.step_size = int(self.bundle["step_size"])
        
        self.min_confidence = float(min_confidence)
        self.consecutive_hits = int(consecutive_hits)
        self.heuristic_enabled = bool(heuristic_enabled)
        
        self.accel_threshold = float(accel_threshold)
        self.brake_threshold = float(brake_threshold)
        self.turn_threshold = float(turn_threshold)
        # Convert turn threshold from degrees/sec to radians/sec internally for raw IMU matching
        self.turn_threshold_rad = np.deg2rad(self.turn_threshold)
        self.lane_change_threshold = float(lane_change_threshold)
        self.invert_gyro_z = bool(invert_gyro_z)
        self.invert_acc_y = bool(invert_acc_y)
        
        # Buffer needs window_size + 10 elements to compute rolling features without edge effects
        self.history_size = self.window_size + 10
        self.buffer: Deque[Dict[str, float]] = deque(maxlen=self.history_size)
        self._tick = 0
        self._active_label: Optional[str] = None
        self._candidate_label: Optional[str] = None
        self._candidate_hits = 0

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

    def _build_sequence_tensor(self) -> torch.Tensor:
        df = pd.DataFrame(list(self.buffer))
        
        # Compute engineered features
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
        df['mag_mag'] = np.sqrt(df['mag_x']**2 + df['mag_y']**2 + df['mag_z']**2)
        
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            df[f'{axis}_jerk'] = df[axis].diff().fillna(0.0)
            
        df['lat_long_ratio'] = df['acc_y'].abs() / (df['acc_x'].abs() + 1e-6)
        df['acc_x_abs'] = df['acc_x'].abs()
        df['acc_y_abs'] = df['acc_y'].abs()
        df['gyro_z_abs'] = df['gyro_z'].abs()
        
        df['acc_lat_energy'] = df['acc_y'] ** 2
        df['acc_long_energy'] = df['acc_x'] ** 2
        df['gyro_yaw_energy'] = df['gyro_z'] ** 2
        
        df['lat_long_energy_ratio'] = df['acc_y_abs'] / (df['acc_x_abs'] + 1e-6)
        df['turn_vs_lateral'] = df['gyro_z_abs'] / (df['acc_y_abs'] + 1e-6)
        df['yaw_acc_corr'] = df['gyro_z'] * df['acc_y']
        
        df['gyro_z_diff'] = df['gyro_z'].diff().fillna(0.0)
        df['acc_y_diff'] = df['acc_y'].diff().fillna(0.0)
        
        df['gyro_z_roll_std5'] = df['gyro_z'].rolling(5, min_periods=1).std().fillna(0.0)
        df['acc_y_roll_std5'] = df['acc_y'].rolling(5, min_periods=1).std().fillna(0.0)
        df['acc_y_roll_mean5'] = df['acc_y'].rolling(5, min_periods=1).mean().fillna(0.0)
        
        df['gyro_z_sign_change'] = (df['gyro_z'].shift(1) * df['gyro_z'] < 0).astype(int).fillna(0)
        
        df['gyro_x_abs'] = df['gyro_x'].abs()
        df['gyro_y_abs'] = df['gyro_y'].abs()
        df['gyro_x_energy'] = df['gyro_x'] ** 2
        df['gyro_y_energy'] = df['gyro_y'] ** 2
        df['gyro_roll_pitch_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2)
        df['gyro_total_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
        df['yaw_vs_roll_pitch'] = df['gyro_z_abs'] / (df['gyro_roll_pitch_mag'] + 1e-6)
        df['gyro_x_roll_std5'] = df['gyro_x'].rolling(5, min_periods=1).std().fillna(0.0)
        df['gyro_y_roll_std5'] = df['gyro_y'].rolling(5, min_periods=1).std().fillna(0.0)
        
        # Slices the last window_size (25) steps of computed features
        df_sliced = df.iloc[-self.window_size:]
        values = df_sliced[self.feature_cols_stream].to_numpy(dtype=np.float32)
        
        # Scale values using StandardScaler
        scaled_values = self.scaler.transform(values)
        
        # Convert to Tensor [1, window_size, 40]
        tensor = torch.tensor(scaled_values, dtype=torch.float32).unsqueeze(0)
        return tensor

    def update(self, sample: Dict[str, float]) -> Optional[Dict[str, object]]:
        cleaned = {}
        for col in IMU_COLUMNS:
            if col not in sample:
                raise KeyError(f"Missing IMU column in sample: {col}")
            cleaned[col] = float(sample[col])

        if self.invert_gyro_z:
            cleaned["gyro_z"] = -cleaned["gyro_z"]
        if self.invert_acc_y:
            cleaned["acc_y"] = -cleaned["acc_y"]

        self.buffer.append(cleaned)
        self._tick += 1

        if len(self.buffer) < self.history_size:
            return {
                "ready": False,
                "reason": "warming_up",
                "buffer_size": len(self.buffer),
                "required": self.history_size,
            }

        if (self._tick % self.step_size) != 0:
            return None

        # Build sequence tensor and run PyTorch CNN model
        tensor = self._build_sequence_tensor()
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].numpy()
            
        pred_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_idx])
        raw_label = str(self.label_encoder_classes[pred_idx])

        if confidence < self.min_confidence:
            raw_label = "safe"

        # Heuristic fallback for sharper events when ml model is uncertain
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
            elif gyro_z >= self.turn_threshold_rad:
                raw_label = "harsh_right_turn"
                confidence = max(confidence, 0.50)
            elif gyro_z <= -self.turn_threshold_rad:
                raw_label = "harsh_left_turn"
                confidence = max(confidence, 0.50)
            elif acc_y >= self.lane_change_threshold:
                raw_label = "harsh_right_lane_change"
                confidence = max(confidence, 0.50)
            elif acc_y <= -self.lane_change_threshold:
                raw_label = "harsh_left_lane_change"
                confidence = max(confidence, 0.50)

        smoothed_label = self._smooth_label(raw_label)

        proba_dict = {cls: float(prob) for cls, prob in zip(self.label_encoder_classes, probabilities)}

        return {
            "ready": True,
            "raw_prediction": raw_label,
            "prediction": smoothed_label,
            "confidence": confidence,
            "probabilities": proba_dict,
        }