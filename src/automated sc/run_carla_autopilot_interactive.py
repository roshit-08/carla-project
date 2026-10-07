#!/usr/bin/env python3
"""
CARLA Interactive Autopilot & Real-Time Driver Safety Score Dashboard.
Vehicle drives continuously on CARLA Autopilot with strict lane keeping.
Prevents off-road diversions and collisions. Allows manual keyboard control
for triggering harsh events while automatically restoring smooth lane centering.
"""

import argparse
import sys
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add src directory to sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import carla
from harsh_event_model import DriverSafetyScorer, EnsembleHarshEventDetector


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA Interactive Autopilot Driver Safety Score System")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA TCP port")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager TCP port")
    parser.add_argument("--role-name", default="hero", help="Vehicle role_name filter")
    parser.add_argument("--sensor-tick", type=float, default=0.05, help="IMU sampling period (sec)")
    parser.add_argument("--rf-path", default="notebooks/artifacts/harsh_event_rf_v2.joblib", help="Random Forest bundle path")
    parser.add_argument("--xgb-path", default="notebooks/artifacts/harsh_event_xgb_v2.joblib", help="XGBoost bundle path")
    parser.add_argument("--cnn-path", default="artifacts/harsh_event_cnn_bundle.pth", help="1D-CNN bundle path")
    parser.add_argument("--rf-weight", type=float, default=0.40, help="Random Forest ensemble weight")
    parser.add_argument("--xgb-weight", type=float, default=0.30, help="XGBoost ensemble weight")
    parser.add_argument("--cnn-weight", type=float, default=0.30, help="CNN ensemble weight")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum prediction confidence")
    parser.add_argument("--consecutive-hits", type=int, default=2, help="Consecutive hit thresholding")
    parser.add_argument("--recovery-rate", type=float, default=0.30, help="Recovery points/sec (+3.0pt/10s, default: 0.30)")
    parser.add_argument("--print-safe", action="store_true", help="Print safe ticks")
    return parser.parse_args()


def find_or_spawn_vehicle(world: carla.World, role_name: str) -> carla.Actor:
    pts = world.get_map().get_spawn_points()
    spawn_point = pts[0] if pts else carla.Transform()

    for actor in world.get_actors():
        if "vehicle." in actor.type_id:
            if actor.attributes.get("role_name") == role_name:
                try:
                    actor.set_transform(spawn_point)
                    actor.set_target_velocity(carla.Vector3D(0, 0, 0))
                    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
                except Exception:
                    pass
                return actor

    actors = world.get_actors().filter("vehicle.*")
    if len(actors) > 0:
        v = actors[0]
        try:
            v.set_transform(spawn_point)
            v.set_target_velocity(carla.Vector3D(0, 0, 0))
            v.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
        except Exception:
            pass
        return v

    bp_lib = world.get_blueprint_library()
    bp_list = bp_lib.filter("vehicle.tesla.model3")
    bp = bp_list[0] if bp_list else bp_lib.filter("vehicle.*")[0]
    bp.set_attribute("role_name", role_name)
    vehicle = world.spawn_actor(bp, spawn_point)
    return vehicle


def start_spectator_follower(world: carla.World, vehicle: carla.Vehicle, stop_event: threading.Event):
    """Smoothly interpolated spectator camera follower to eliminate visual camera jittering."""
    spectator = world.get_spectator()
    curr_loc = None
    curr_yaw = None
    while not stop_event.is_set():
        try:
            t = vehicle.get_transform()
            fwd = t.get_forward_vector()
            target_loc = t.location - carla.Location(x=7.0 * fwd.x, y=7.0 * fwd.y, z=-3.0)
            target_yaw = t.rotation.yaw

            if curr_loc is None:
                curr_loc = target_loc
                curr_yaw = target_yaw
            else:
                # Exponential smoothing (alpha = 0.20) for silky smooth camera tracking
                curr_loc = carla.Location(
                    x=curr_loc.x + 0.20 * (target_loc.x - curr_loc.x),
                    y=curr_loc.y + 0.20 * (target_loc.y - curr_loc.y),
                    z=curr_loc.z + 0.20 * (target_loc.z - curr_loc.z)
                )
                yaw_diff = (target_yaw - curr_yaw + 180) % 360 - 180
                curr_yaw = curr_yaw + 0.20 * yaw_diff

            rot = carla.Rotation(pitch=-15.0, yaw=curr_yaw, roll=0.0)
            spectator.set_transform(carla.Transform(curr_loc, rot))
        except Exception:
            pass
        time.sleep(0.02)


def configure_traffic_manager(client: carla.Client, vehicle: carla.Vehicle, tm_port: int = 8000):
    """Configures Traffic Manager to ensure strict lane following without crashing into curbs/buildings."""
    try:
        tm = client.get_trafficmanager(tm_port)
    except RuntimeError as e:
        if "bind error" in str(e).lower():
            alternative_port = tm_port + 500
            print(f"⚠️ Traffic Manager port {tm_port} bind error. Retrying on port {alternative_port}...")
            tm = client.get_trafficmanager(alternative_port)
        else:
            raise e

    port = tm.get_port()
    vehicle.set_autopilot(True, port)

    try:
        tm.auto_lane_change(vehicle, False)
        tm.distance_to_leading_vehicle(vehicle, 2.0)
        tm.vehicle_percentage_speed_difference(vehicle, -20.0)
        tm.ignore_lights_percentage(vehicle, 100.0)
    except Exception as e:
        print(f"Traffic Manager Warning: {e}")
    return tm


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    vehicle = find_or_spawn_vehicle(world, args.role_name)
    tm = configure_traffic_manager(client, vehicle, args.tm_port)

    print(f"✓ Connected to CARLA vehicle id={vehicle.id}, Autopilot ENABLED with Strict Lane Keeping.")

    # Initialize Ensemble Detector & Safety Scorer
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
    scorer = DriverSafetyScorer(initial_score=100.0, recovery_rate_per_sec=args.recovery_rate)
    lock = threading.Lock()

    # Attach IMU Sensor
    bp_lib = world.get_blueprint_library()
    imu_bp = bp_lib.find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", str(args.sensor_tick))
    imu_sensor = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=0.0)), attach_to=vehicle)

    print("\n" + "=" * 75)
    print("   INTERACTIVE AUTOPILOT: REAL-TIME ENSEMBLE DRIVER SAFETY SCORE")
    print("=" * 75)
    print("Default Mode           : CARLA Autopilot (strict lane-keeping, safe score)")
    print("Manual Control         : Drive/override in manual_control.py for harsh events")
    print("Auto-Recovery          : Autopilot re-centers in lane smoothly & recovers score")
    print("Base Safety Score      : 100.0 / 100.0")
    print("Press Ctrl+C to stop.\n")

    def _on_imu(data: carla.IMUMeasurement):
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
            score_info = scorer.update(label, conf, dt_sec=args.sensor_tick)

            score = score_info["score"]
            tier = score_info["risk_tier"]
            total_ev = score_info["total_events"]
            timestamp = time.strftime("%H:%M:%S")

            rf_p = res.get("rf_prediction", "N/A")
            xgb_p = res.get("xgb_prediction", "N/A")
            cnn_p = res.get("cnn_prediction", "N/A")

            if label != "safe" or args.print_safe:
                print(
                    f"[{timestamp}] [Autopilot Active   ] Score: {score:5.1f}/100 | Tier: {tier:<20} | "
                    f"Event: {label:<22} (Conf: {conf:.2f} | RF:{rf_p} XGB:{xgb_p} CNN:{cnn_p}) | Events: {total_ev}"
                )

    imu_sensor.listen(_on_imu)

    # Spectator Camera Follower
    spectator_stop = threading.Event()
    spectator_thread = threading.Thread(
        target=start_spectator_follower, args=(world, vehicle, spectator_stop), daemon=True
    )
    spectator_thread.start()

    try:
        last_fix = time.time()
        while True:
            time.sleep(0.5)
            # Re-verify Traffic Manager alignment every 5 seconds without fighting manual control
            if time.time() - last_fix > 5.0:
                try:
                    tm.auto_lane_change(vehicle, False)
                except Exception:
                    pass
                last_fix = time.time()

    except KeyboardInterrupt:
        print("\nStopping Interactive Autopilot Dashboard...")
    finally:
        spectator_stop.set()
        vehicle.set_autopilot(False)
        if imu_sensor is not None and imu_sensor.is_alive:
            imu_sensor.stop()
            time.sleep(0.2)
            imu_sensor.destroy()
        print("✓ Cleanup complete.")


if __name__ == "__main__":
    main()
