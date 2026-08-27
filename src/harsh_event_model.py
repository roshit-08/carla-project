from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import time
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
        accel_threshold: float = 4.2,
        brake_threshold: float = -4.2,
        turn_threshold: float = 40.0,
        lane_change_threshold: float = 4.0,
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
        self.turn_threshold_rad = float(np.deg2rad(self.turn_threshold))
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
        feat_dict = {col: features.get(col, 0.0) for col in self.feature_columns}
        row = pd.DataFrame([feat_dict], columns=self.feature_columns)
        return row

    def _smooth_label(self, raw_label: str) -> str:
        if self._active_label is None:
            if raw_label == "safe":
                self._candidate_label = None
                self._candidate_hits = 0
                return "safe"
            
            if self._candidate_label is not None and self._candidate_label != "safe" and raw_label != "safe":
                # Accumulate hits across consecutive non-safe harsh event predictions
                self._candidate_hits += 1
                self._candidate_label = raw_label
            else:
                self._candidate_label = raw_label
                self._candidate_hits = 1

            if self._candidate_hits >= self.consecutive_hits:
                self._active_label = self._candidate_label
            return self._active_label or "safe"

        if raw_label == self._active_label:
            self._candidate_label = None
            self._candidate_hits = 0
            return self._active_label

        if raw_label == self._candidate_label or (raw_label != "safe" and self._candidate_label is not None and self._candidate_label != "safe"):
            self._candidate_hits += 1
            if raw_label != "safe":
                self._candidate_label = raw_label
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
            cleaned[col] = float(sample.get(col, 0.0))

        if self.invert_gyro_z:
            cleaned["gyro_z"] = -cleaned["gyro_z"]
        if self.invert_acc_y:
            cleaned["acc_y"] = -cleaned["acc_y"]

        # IMU Noise Gating for road curve and autopilot cruise stability
        if abs(cleaned.get("gyro_z", 0.0)) < 0.20:
            cleaned["gyro_z"] = 0.0
        if abs(cleaned.get("acc_x", 0.0)) < 0.80:
            cleaned["acc_x"] = 0.0
        if abs(cleaned.get("acc_y", 0.0)) < 0.80:
            cleaned["acc_y"] = 0.0

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

        # Heuristic fallback: require sustained multi-sample peak to avoid false triggers on normal driving
        if self.heuristic_enabled and raw_label == "safe" and len(self.buffer) >= 3:
            recent_acc_x = float(np.mean([s.get("acc_x", 0.0) for s in list(self.buffer)[-3:]]))
            recent_acc_y = float(np.mean([s.get("acc_y", 0.0) for s in list(self.buffer)[-3:]]))
            recent_gyro_z = float(np.mean([s.get("gyro_z", 0.0) for s in list(self.buffer)[-3:]]))
            last_acc_x = float(self.buffer[-1].get("acc_x", 0.0))

            if recent_acc_x >= self.accel_threshold or last_acc_x >= 4.2:
                raw_label = "sudden_acceleration"
                confidence = max(confidence, 0.55)
            elif recent_acc_x <= self.brake_threshold or last_acc_x <= -4.2:
                raw_label = "sudden_braking"
                confidence = max(confidence, 0.55)
            elif recent_gyro_z >= self.turn_threshold_rad:
                raw_label = "harsh_right_turn"
                confidence = max(confidence, 0.55)
            elif recent_gyro_z <= -self.turn_threshold_rad:
                raw_label = "harsh_left_turn"
                confidence = max(confidence, 0.55)
            elif recent_acc_y >= self.lane_change_threshold:
                raw_label = "harsh_right_lane_change"
                confidence = max(confidence, 0.55)
            elif recent_acc_y <= -self.lane_change_threshold:
                raw_label = "harsh_left_lane_change"
                confidence = max(confidence, 0.55)

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
    """Compact 1D-CNN with avg+max global pooling.
    The max branch explicitly captures peak-yaw events (turns),
    which are otherwise averaged away in avg-only pooling.

    Architecture matches the trained bundle in artifacts/harsh_event_cnn_bundle.pth:
      - 3 conv blocks: (kernel 5, 64ch) → (kernel 5, 128ch) → (kernel 3, 128ch)
      - Global avg + max pooling → 256-dim → classifier head
    """

    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1 — wider kernel to capture sustained patterns
            nn.Conv1d(in_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 3
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        # Global avg + max pooling concatenated → 256-dim classifier
        self.head = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)           # [B, T, F] → [B, F, T]
        x = self.conv(x)                # [B, 128, T]
        avg_pool = x.mean(dim=2)        # [B, 128]
        max_pool = x.max(dim=2).values  # [B, 128] — captures peak events
        pooled   = torch.cat([avg_pool, max_pool], dim=1)  # [B, 256]
        return self.head(pooled)


class RealtimeCNNHarshEventDetector:
    def __init__(
        self,
        model_bundle_path: str | Path,
        min_confidence: float = 0.35,
        consecutive_hits: int = 1,
        heuristic_enabled: bool = True,
        accel_threshold: float = 4.2,
        brake_threshold: float = -4.2,
        turn_threshold: float = 40.0,
        lane_change_threshold: float = 4.0,
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
        
        # Buffer exactly window_size deep — rolling features need min_periods=1 so smaller buffers work fine
        self.history_size = self.window_size
        self.buffer: Deque[Dict[str, float]] = deque(maxlen=self.history_size)
        self._tick = 0
        self._active_label: Optional[str] = None
        self._candidate_label: Optional[str] = None
        self._candidate_hits = 0

    def _smooth_label(self, raw_label: str) -> str:
        if self._active_label is None:
            if raw_label == "safe":
                self._candidate_label = None
                self._candidate_hits = 0
                return "safe"
            
            if self._candidate_label is not None and self._candidate_label != "safe" and raw_label != "safe":
                self._candidate_hits += 1
                self._candidate_label = raw_label
            else:
                self._candidate_label = raw_label
                self._candidate_hits = 1

            if self._candidate_hits >= self.consecutive_hits:
                self._active_label = self._candidate_label
            return self._active_label or "safe"

        if raw_label == self._active_label:
            self._candidate_label = None
            self._candidate_hits = 0
            return self._active_label

        if raw_label == self._candidate_label or (raw_label != "safe" and self._candidate_label is not None and self._candidate_label != "safe"):
            self._candidate_hits += 1
            if raw_label != "safe":
                self._candidate_label = raw_label
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

        # ── Magnitudes ────────────────────────────────────────────────────────
        df['acc_mag']  = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
        df['mag_mag']  = np.sqrt(df['mag_x']**2 + df['mag_y']**2 + df['mag_z']**2)

        # ── Jerk ──────────────────────────────────────────────────────────────
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            df[f'{axis}_jerk'] = df[axis].diff().fillna(0.0)

        # ── Lateral / longitudinal ratios and energy ──────────────────────────
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

        # ── Differentials ─────────────────────────────────────────────────────
        df['gyro_z_diff'] = df['gyro_z'].diff().fillna(0.0)
        df['acc_y_diff']  = df['acc_y'].diff().fillna(0.0)

        # ── Rolling statistics ────────────────────────────────────────────────
        df['gyro_z_roll_std5']  = df['gyro_z'].rolling(5, min_periods=1).std().fillna(0.0)
        df['acc_y_roll_std5']   = df['acc_y'].rolling(5, min_periods=1).std().fillna(0.0)
        df['acc_y_roll_mean5']  = df['acc_y'].rolling(5, min_periods=1).mean().fillna(0.0)

        # ── Sign change ───────────────────────────────────────────────────────
        df['gyro_z_sign_change'] = (df['gyro_z'].shift(1) * df['gyro_z'] < 0).astype(int).fillna(0)

        # ── Roll / pitch turn features ────────────────────────────────────────
        df['gyro_x_abs']          = df['gyro_x'].abs()
        df['gyro_y_abs']          = df['gyro_y'].abs()
        df['gyro_x_energy']       = df['gyro_x'] ** 2
        df['gyro_y_energy']       = df['gyro_y'] ** 2
        df['gyro_roll_pitch_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2)
        df['gyro_total_mag']      = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
        df['yaw_vs_roll_pitch']   = df['gyro_z_abs'] / (df['gyro_roll_pitch_mag'] + 1e-6)
        df['gyro_x_roll_std5']    = df['gyro_x'].rolling(5, min_periods=1).std().fillna(0.0)
        df['gyro_y_roll_std5']    = df['gyro_y'].rolling(5, min_periods=1).std().fillna(0.0)

        # ── Peak yaw features ─────────────────────────────────────────────────
        df['gyro_z_roll_max5']    = df['gyro_z_abs'].rolling(5,  min_periods=1).max().fillna(0.0)
        df['gyro_z_roll_max10']   = df['gyro_z_abs'].rolling(10, min_periods=1).max().fillna(0.0)
        df['gyro_z_roll_energy5'] = df['gyro_yaw_energy'].rolling(5, min_periods=1).mean().fillna(0.0)
        df['yaw_dominance']       = df['gyro_z_abs'] / (df['gyro_total_mag'] + 1e-6)

        # ── NEW S-CURVE & NET YAW FEATURES FOR LANE CHANGE DISCRIMINATION ─────────
        # Net Yaw Change (cumsum over window): turns accumulate large net angle, lane changes sum to ~0
        df['net_yaw_cumsum20']      = df['gyro_z'].rolling(20, min_periods=1).sum().fillna(0.0)
        # Biphasic Yaw Signature: rolling min * rolling max (negative when S-curve steering left then right occurs)
        gyro_z_min20                = df['gyro_z'].rolling(20, min_periods=1).min().fillna(0.0)
        gyro_z_max20                = df['gyro_z'].rolling(20, min_periods=1).max().fillna(0.0)
        df['biphasic_yaw_signature'] = gyro_z_min20 * gyro_z_max20
        # Lateral Jerk Standard Deviation
        df['acc_y_jerk_std5']       = df['acc_y_jerk'].rolling(5, min_periods=1).std().fillna(0.0)

        # Slices the last window_size steps and selects exactly the feature
        # columns stored in the bundle (handles old vs new bundles gracefully)
        df_sliced = df.iloc[-self.window_size:]
        values = df_sliced[self.feature_cols_stream].to_numpy(dtype=np.float32)

        # Scale values using StandardScaler fitted at training time
        scaled_values = self.scaler.transform(values)

        # Convert to Tensor [1, window_size, n_features]
        tensor = torch.tensor(scaled_values, dtype=torch.float32).unsqueeze(0)
        return tensor


    def update(self, sample: Dict[str, float]) -> Optional[Dict[str, object]]:
        cleaned = {}
        for col in IMU_COLUMNS:
            cleaned[col] = float(sample.get(col, 0.0))


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

        # Heuristic fallback: require sustained multi-sample peak to avoid false triggers on normal driving
        if self.heuristic_enabled and raw_label == "safe" and len(self.buffer) >= 3:
            recent_acc_x = float(np.mean([s.get("acc_x", 0.0) for s in list(self.buffer)[-3:]]))
            recent_acc_y = float(np.mean([s.get("acc_y", 0.0) for s in list(self.buffer)[-3:]]))
            recent_gyro_z = float(np.mean([s.get("gyro_z", 0.0) for s in list(self.buffer)[-3:]]))
            last_acc_x = float(self.buffer[-1].get("acc_x", 0.0))

            if recent_acc_x >= self.accel_threshold or last_acc_x >= 4.2:
                raw_label = "sudden_acceleration"
                confidence = max(confidence, 0.55)
            elif recent_acc_x <= self.brake_threshold or last_acc_x <= -4.2:
                raw_label = "sudden_braking"
                confidence = max(confidence, 0.55)
            elif recent_gyro_z >= self.turn_threshold_rad:
                raw_label = "harsh_right_turn"
                confidence = max(confidence, 0.55)
            elif recent_gyro_z <= -self.turn_threshold_rad:
                raw_label = "harsh_left_turn"
                confidence = max(confidence, 0.55)
            elif recent_acc_y >= self.lane_change_threshold:
                raw_label = "harsh_right_lane_change"
                confidence = max(confidence, 0.55)
            elif recent_acc_y <= -self.lane_change_threshold:
                raw_label = "harsh_left_lane_change"
                confidence = max(confidence, 0.55)

        smoothed_label = self._smooth_label(raw_label)

        proba_dict = {cls: float(prob) for cls, prob in zip(self.label_encoder_classes, probabilities)}

        return {
            "ready": True,
            "raw_prediction": raw_label,
            "prediction": smoothed_label,
            "confidence": confidence,
            "probabilities": proba_dict,
        }


class DriverSafetyScorer:
    """Real-time driver safety score calculator (0 - 100)."""

    PENALTIES = {
        "sudden_braking": 6.0,
        "harsh_left_turn": 5.0,
        "harsh_right_turn": 5.0,
        "harsh_left_lane_change": 4.0,
        "harsh_right_lane_change": 4.0,
        "sudden_acceleration": 3.0,
    }

    def __init__(self, initial_score: float = 100.0, recovery_rate_per_sec: float = 1.0 / 10.0):
        self.initial_score = initial_score
        self.current_score = initial_score
        self.recovery_rate_per_sec = recovery_rate_per_sec  # +1.0 pt per 10 sec of safe driving

        self.event_counts = {lbl: 0 for lbl in self.PENALTIES.keys()}
        self.total_samples = 0
        self.consecutive_safe_samples = 0
        self.start_time = time.time()
        self.last_event = "safe"
        self.last_event_time = 0.0

    def update(self, detected_label: str, confidence: float, dt_sec: float = 0.05) -> dict:
        self.total_samples += 1
        trip_duration = time.time() - self.start_time

        if detected_label in self.PENALTIES and detected_label != self.last_event:
            penalty = self.PENALTIES[detected_label]
            self.current_score = max(0.0, self.current_score - penalty)
            self.event_counts[detected_label] += 1
            self.consecutive_safe_samples = 0
            self.last_event = detected_label
            self.last_event_time = trip_duration
        elif detected_label == "safe":
            self.consecutive_safe_samples += 1
            # Add safe recovery bonus
            recovery = self.recovery_rate_per_sec * dt_sec
            self.current_score = min(100.0, self.current_score + recovery)
            if self.consecutive_safe_samples > 20:  # ~1 second of safe driving
                self.last_event = "safe"

        # Risk Rating Tier
        if self.current_score >= 90.0:
            risk_tier = "SAFE / SMOOTH"
            color_code = "GREEN"
        elif self.current_score >= 75.0:
            risk_tier = "MODERATE RISK"
            color_code = "YELLOW"
        else:
            risk_tier = "HIGH RISK / AGGRESSIVE"
            color_code = "RED"

        return {
            "score": round(self.current_score, 1),
            "risk_tier": risk_tier,
            "color_code": color_code,
            "total_events": sum(self.event_counts.values()),
            "event_counts": self.event_counts.copy(),
            "trip_duration_sec": round(trip_duration, 1),
        }


class EnsembleHarshEventDetector:
    """Multi-Model Soft-Voting Ensemble Detector (Random Forest v2 + XGBoost v2 + 1D-CNN)."""

    def __init__(
        self,
        rf_path: str = "notebooks/artifacts/harsh_event_rf_v2.joblib",
        xgb_path: str = "notebooks/artifacts/harsh_event_xgb_v2.joblib",
        cnn_path: str = "artifacts/harsh_event_cnn_bundle.pth",
        rf_weight: float = 0.40,
        xgb_weight: float = 0.30,
        cnn_weight: float = 0.30,
        min_confidence: float = 0.40,
        consecutive_hits: int = 2,
        use_heuristics: bool = True,
    ):
        self.min_confidence = float(min_confidence)
        self.consecutive_hits = int(consecutive_hits)
        self.rf_detector = RealtimeHarshEventDetector(model_bundle_path=rf_path, min_confidence=min_confidence, consecutive_hits=consecutive_hits, heuristic_enabled=use_heuristics) if Path(rf_path).exists() else None
        self.xgb_detector = RealtimeHarshEventDetector(model_bundle_path=xgb_path, min_confidence=min_confidence, consecutive_hits=consecutive_hits, heuristic_enabled=use_heuristics) if Path(xgb_path).exists() else None
        self.cnn_detector = RealtimeCNNHarshEventDetector(model_bundle_path=cnn_path, min_confidence=min_confidence, consecutive_hits=consecutive_hits, heuristic_enabled=use_heuristics) if Path(cnn_path).exists() else None

        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.cnn_weight = cnn_weight
        self.proba_history: Deque[np.ndarray] = deque(maxlen=3)

        # Extract classes
        for det in [self.rf_detector, self.xgb_detector, self.cnn_detector]:
            if det is not None:
                self.classes = list(det.label_encoder.classes_)
                break


    def update(self, packet: dict) -> dict:
        rf_res = self.rf_detector.update(packet) if self.rf_detector else None
        xgb_res = self.xgb_detector.update(packet) if self.xgb_detector else None
        cnn_res = self.cnn_detector.update(packet) if self.cnn_detector else None

        if not (rf_res and rf_res.get("ready")):
            return rf_res or {"ready": False, "reason": "warming_up"}

        # Soft voting weighted ensemble
        ensemble_probas = np.zeros(len(self.classes))
        total_w = 0.0

        if rf_res and rf_res.get("ready"):
            rf_p = np.array([rf_res["probabilities"].get(c, 0.0) for c in self.classes])
            ensemble_probas += self.rf_weight * rf_p
            total_w += self.rf_weight

        if xgb_res and xgb_res.get("ready"):
            xgb_p = np.array([xgb_res["probabilities"].get(c, 0.0) for c in self.classes])
            ensemble_probas += self.xgb_weight * xgb_p
            total_w += self.xgb_weight

        if cnn_res and cnn_res.get("ready"):
            cnn_p = np.array([cnn_res["probabilities"].get(c, 0.0) for c in self.classes])
            ensemble_probas += self.cnn_weight * cnn_p
            total_w += self.cnn_weight

        if total_w > 0:
            ensemble_probas /= total_w

        # 3-tick Rolling Probability Smoother
        self.proba_history.append(ensemble_probas)
        smooth_probas = np.mean(list(self.proba_history), axis=0)

        top_idx = int(np.argmax(smooth_probas))
        raw_prediction = self.classes[top_idx]
        raw_confidence = float(smooth_probas[top_idx])

        # Collect sub-detector prediction votes and confidences
        votes = []
        if rf_res and rf_res.get("ready"):
            votes.append((rf_res["prediction"], self.rf_weight, rf_res.get("confidence", 0.5)))
        if xgb_res and xgb_res.get("ready"):
            votes.append((xgb_res["prediction"], self.xgb_weight, xgb_res.get("confidence", 0.5)))
        if cnn_res and cnn_res.get("ready"):
            votes.append((cnn_res["prediction"], self.cnn_weight, cnn_res.get("confidence", 0.5)))

        # Unanimous sub-detector consensus: if all active weighted models agree on the same harsh event
        active_non_safe_votes = [
            pred for pred, weight, _ in votes if weight > 0.0 and pred != "safe"
        ]
        active_weights = [weight for pred, weight, _ in votes if weight > 0.0]
        if active_non_safe_votes and len(active_non_safe_votes) == len(active_weights) and len(set(active_non_safe_votes)) == 1:
            agreed_event = active_non_safe_votes[0]
            agreed_conf = max(raw_confidence, 0.55)
            raw_prediction = agreed_event
            raw_confidence = agreed_conf

        if raw_confidence < self.min_confidence:
            prediction = "safe"
            confidence = raw_confidence
        else:
            prediction = raw_prediction
            confidence = raw_confidence

        proba_dict = {cls: float(prob) for cls, prob in zip(self.classes, ensemble_probas)}

        return {
            "ready": True,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": proba_dict,
            "rf_prediction": rf_res.get("prediction") if rf_res else "N/A",
            "xgb_prediction": xgb_res.get("prediction") if xgb_res else "N/A",
            "cnn_prediction": cnn_res.get("prediction") if cnn_res else "N/A",
        }

