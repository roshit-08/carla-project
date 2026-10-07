from pathlib import Path

import pandas as pd

from harsh_event_model import IMU_COLUMNS, RealtimeHarshEventDetector


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    model_path = root / "artifacts" / "harsh_event_rf.joblib"
    dataset_dir = root / "dataset"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train from the notebook first.")

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    sample_csv = csv_files[0]
    df = pd.read_csv(sample_csv)
    for col in IMU_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=IMU_COLUMNS).reset_index(drop=True)

    detector = RealtimeHarshEventDetector(model_bundle_path=model_path, min_confidence=0.45, consecutive_hits=2)

    print(f"Streaming from: {sample_csv.name}")
    for idx, row in df.iterrows():
        packet = {c: float(row[c]) for c in IMU_COLUMNS}
        result = detector.update(packet)
        if isinstance(result, dict) and result.get("ready") and result.get("prediction") != "safe":
            ts = row.get("timestamp", idx)
            print(f"idx={idx} timestamp={ts} event={result['prediction']} confidence={result['confidence']:.3f}")


if __name__ == "__main__":
    main()