# Harsh Driving Event Detection (CARLA IMU)

This project trains and runs real-time harsh driving detection for:
- sudden_acceleration
- sudden_braking
- harsh_left_turn
- harsh_right_turn
- harsh_left_lane_change
- harsh_right_lane_change
- safe

Dataset schema used:
- sr_no, timestamp, road_type, harsh_event,
- acc_x, acc_y, acc_z,
- gyro_x, gyro_y, gyro_z,
- mag_x, mag_y, mag_z,
- event_class

## Project Structure

- notebooks/harsh_event_detection_project.ipynb: full training and inference workflow.
- src/harsh_event_model.py: reusable training and real-time detector module.
- artifacts/: saved model files and split manifests.
- dataset/: source CSV files.

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Open and run notebook:

   notebooks/harsh_event_detection_project.ipynb

4. Main outputs after training:
- artifacts/harsh_event_rf.joblib
- artifacts/harsh_event_rf_calibrated.joblib
- artifacts/harsh_event_cnn_bilstm.pt
- artifacts/split_manifest.csv
- artifacts/seq_training_history.csv

## Real-Time Integration

Use RealtimeHarshEventDetector from src/harsh_event_model.py.

Expected packet keys per timestep:
- acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z

Example:

from harsh_event_model import RealtimeHarshEventDetector

detector = RealtimeHarshEventDetector('artifacts/harsh_event_rf.joblib')
result = detector.update({
    'acc_x': 0.1, 'acc_y': -0.2, 'acc_z': 0.9,
    'gyro_x': 0.01, 'gyro_y': -0.02, 'gyro_z': 0.03,
    'mag_x': -95.0, 'mag_y': 80.0, 'mag_z': 90.0,
})

if result and result.get('ready'):
    print(result)

You can consume emitted events (timestamp, prediction, confidence) to compute your safety score in your own pipeline.
