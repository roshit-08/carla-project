#!/usr/bin/env python3
"""
CARLA Real-Time Harsh Event Detection Dashboard.
Uses Multi-Model Ensemble (Random Forest v2 + XGBoost v2 + PyTorch 1D-CNN)
with weighted soft-voting probability fusion to output current detected driving events in real time.
"""

import argparse
import sys
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add src to path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import carla
from harsh_event_model import EnsembleHarshEventDetector


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA Real-Time Harsh Event Detection Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA TCP port")
    parser.add_argument("--role-name", default="hero", help="Vehicle role_name filter")
    parser.add_argument("--sensor-tick", type=float, default=0.05, help="IMU sampling period (sec)")
    parser.add_argument("--rf-path", default="notebooks/artifacts/harsh_event_rf_v2.joblib", help="Random Forest bundle path")
    parser.add_argument("--xgb-path", default="notebooks/artifacts/harsh_event_xgb_v2.joblib", help="XGBoost bundle path")
    parser.add_argument("--cnn-path", default="artifacts/harsh_event_cnn_bundle.pth", help="1D-CNN bundle path")
    parser.add_argument("--rf-weight", type=float, default=0.40, help="Random Forest ensemble weight")
    parser.add_argument("--xgb-weight", type=float, default=0.30, help="XGBoost ensemble weight")
    parser.add_argument("--cnn-weight", type=float, default=0.30, help="CNN ensemble weight")
    parser.add_argument("--min-confidence", type=float, default=0.40, help="Minimum prediction confidence")
    parser.add_argument("--consecutive-hits", type=int, default=2, help="Consecutive hit thresholding")
    parser.add_argument("--print-safe", action="store_true", help="Print safe ticks")
    return parser.parse_args()


def find_vehicle(world: carla.World, role_name: str) -> carla.Actor:
    for actor in world.get_actors():
        if "vehicle." in actor.type_id:
            attributes = actor.attributes
            if attributes.get("role_name") == role_name:
                return actor
    actors = world.get_actors().filter("vehicle.*")
    if len(actors) > 0:
        return actors[0]
    raise RuntimeError("No vehicle found in CARLA world.")


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    vehicle = find_vehicle(world, args.role_name)
    print(f"✓ Attached to CARLA vehicle id={vehicle.id}, role_name={args.role_name}")

    # Initialize Ensemble Detector
    detector = EnsembleHarshEventDetector(
        rf_path=args.rf_path,
        xgb_path=args.xgb_path,
        cnn_path=args.cnn_path,
        rf_weight=args.rf_weight,
        xgb_weight=args.xgb_weight,
        cnn_weight=args.cnn_weight,
        min_confidence=args.min_confidence,
        consecutive_hits=args.consecutive_hits,
        use_heuristics=True,
    )

    lock = threading.Lock()
    event_counter = 0

    # Attach IMU Sensor
    bp_lib = world.get_blueprint_library()
    imu_bp = bp_lib.find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", str(args.sensor_tick))
    imu_transform = carla.Transform(carla.Location(x=0.0, z=0.0))
    imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)

    print("\n" + "=" * 70)
    print("        CARLA REAL-TIME HARSH EVENT DETECTION DASHBOARD")
    print("=" * 70)
    print("Ensemble Models Loaded : Random Forest v2, XGBoost v2, 1D-CNN")
    print("Listening to live IMU stream. Press Ctrl+C to stop.\n")

    def _on_imu(data: carla.IMUMeasurement):
        nonlocal event_counter
        packet = {
            "acc_x": float(data.accelerometer.x),
            "acc_y": float(data.accelerometer.y),
            "acc_z": float(data.accelerometer.z),
            "gyro_x": float(data.gyroscope.x),
            "gyro_y": float(data.gyroscope.y),
            "gyro_z": float(data.gyroscope.z),
        }

        with lock:
            res = detector.update(packet)
            if not res or not res.get("ready"):
                return

            label = res["prediction"]
            conf = res["confidence"]

            if label != "safe" or args.print_safe:
                if label != "safe":
                    event_counter += 1

                timestamp = time.strftime("%H:%M:%S")
                rf_p = res.get("rf_prediction", "N/A")
                xgb_p = res.get("xgb_prediction", "N/A")
                cnn_p = res.get("cnn_prediction", "N/A")

                print(
                    f"[{timestamp}] Event Detected: {label:<25} | "
                    f"Conf: {conf:4.2f} | RF: {rf_p:<22} | XGB: {xgb_p:<22} | CNN: {cnn_p:<22} | Event #{event_counter}"
                )

    imu_sensor.listen(_on_imu)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping Real-Time Event Detection Dashboard...")
    finally:
        imu_sensor.stop()
        imu_sensor.destroy()
        print("✓ Cleanup complete.")


if __name__ == "__main__":
    main()
