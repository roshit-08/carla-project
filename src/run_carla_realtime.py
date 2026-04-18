from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import carla

from harsh_event_model import RealtimeHarshEventDetector


def _find_ego_vehicle(world: carla.World, role_name: str, fallback_index: int) -> Optional[carla.Vehicle]:
    vehicles = world.get_actors().filter("vehicle.*")
    if not vehicles:
        return None

    for actor in vehicles:
        if actor.attributes.get("role_name", "") == role_name:
            return actor

    idx = max(0, min(fallback_index, len(vehicles) - 1))
    return vehicles[idx]


def _rotation_matrix_body_to_world(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _world_to_body_vector(wx: float, wy: float, wz: float, rot: carla.Rotation) -> Tuple[float, float, float]:
    r = _rotation_matrix_body_to_world(rot.roll, rot.pitch, rot.yaw)
    # Inverse rotation for orthonormal matrix is transpose.
    bx = r[0][0] * wx + r[1][0] * wy + r[2][0] * wz
    by = r[0][1] * wx + r[1][1] * wy + r[2][1] * wz
    bz = r[0][2] * wx + r[1][2] * wy + r[2][2] * wz
    return bx, by, bz


def _virtual_magnetometer_from_vehicle(
    vehicle: carla.Vehicle,
    field_strength: float,
    declination_deg: float,
    inclination_deg: float,
) -> Tuple[float, float, float]:
    # Earth magnetic field in world frame: horizontal (declination) + vertical dip (inclination).
    dec = math.radians(declination_deg)
    inc = math.radians(inclination_deg)

    horizontal = field_strength * math.cos(inc)
    world_x = horizontal * math.cos(dec)
    world_y = horizontal * math.sin(dec)
    world_z = -field_strength * math.sin(inc)  # CARLA Z axis points up.

    rotation = vehicle.get_transform().rotation
    return _world_to_body_vector(world_x, world_y, world_z, rotation)


def _packet_from_imu(
    data: carla.IMUMeasurement,
    vehicle: carla.Vehicle,
    mag_mode: str,
    compass_scale: float,
    mag_field_strength: float,
    mag_declination_deg: float,
    mag_inclination_deg: float,
) -> dict:
    if mag_mode == "virtual3d":
        mag_x, mag_y, mag_z = _virtual_magnetometer_from_vehicle(
            vehicle=vehicle,
            field_strength=mag_field_strength,
            declination_deg=mag_declination_deg,
            inclination_deg=mag_inclination_deg,
        )
    else:
        # Fallback: use compass heading as pseudo X/Y and zero Z.
        mag_x = compass_scale * float(math.cos(data.compass))
        mag_y = compass_scale * float(math.sin(data.compass))
        mag_z = 0.0

    return {
        "acc_x": float(data.accelerometer.x),
        "acc_y": float(data.accelerometer.y),
        "acc_z": float(data.accelerometer.z),
        "gyro_x": float(math.degrees(data.gyroscope.x)),
        "gyro_y": float(math.degrees(data.gyroscope.y)),
        "gyro_z": float(math.degrees(data.gyroscope.z)),
        "mag_x": mag_x,
        "mag_y": mag_y,
        "mag_z": mag_z,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Realtime harsh-event detection from CARLA IMU stream")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--timeout", type=float, default=10.0, help="CARLA client timeout (s)")
    parser.add_argument("--role-name", default="hero", help="Vehicle role_name to attach IMU to")
    parser.add_argument("--fallback-vehicle-index", type=int, default=0, help="Fallback vehicle index if role_name is not found")
    parser.add_argument("--sensor-tick", type=float, default=0.05, help="IMU update interval in seconds")
    parser.add_argument("--model-path", default="artifacts/harsh_event_rf.joblib", help="Path to trained RF bundle")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum confidence for non-safe event")
    parser.add_argument("--consecutive-hits", type=int, default=2, help="Smoothing hits needed before event emit")
    parser.add_argument("--compass-scale", type=float, default=100.0, help="Scale factor for pseudo magnetometer from compass")
    parser.add_argument("--mag-mode", choices=["virtual3d", "compass2d"], default="virtual3d", help="Magnetometer mode")
    parser.add_argument("--mag-field-strength", type=float, default=100.0, help="Virtual Earth magnetic field strength")
    parser.add_argument("--mag-declination-deg", type=float, default=0.0, help="Virtual magnetic declination in degrees")
    parser.add_argument("--mag-inclination-deg", type=float, default=60.0, help="Virtual magnetic inclination in degrees")
    parser.add_argument("--print-safe", action="store_true", help="Print safe predictions too")
    parser.add_argument("--heuristic", action="store_true", default=False, help="Enable IMU heuristic fallback for harsh events")
    parser.add_argument("--accel-threshold", type=float, default=2.0, help="Acceleration threshold (m/s^2) for sudden acceleration heuristic")
    parser.add_argument("--brake-threshold", type=float, default=-2.0, help="Acceleration threshold (m/s^2) for sudden braking heuristic")
    parser.add_argument("--turn-threshold", type=float, default=40.0, help="Gyroscope Z threshold (deg/s) for harsh turn heuristic")
    parser.add_argument("--lane-change-threshold", type=float, default=2.0, help="Acceleration Y threshold (m/s^2) for lane-change heuristic")
    parser.add_argument("--invert-gyro-z", action="store_true", default=False, help="Invert gyro_z sign if CARLA yaw-rate direction differs from training data")
    parser.add_argument("--invert-acc-y", action="store_true", default=False, help="Invert acc_y sign if CARLA lateral acceleration direction differs from training data")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model bundle not found: {model_path}")
        print("Train/export first so artifacts/harsh_event_rf.joblib exists.")
        return 1

    detector = RealtimeHarshEventDetector(
        model_bundle_path=model_path,
        min_confidence=args.min_confidence,
        consecutive_hits=args.consecutive_hits,
        heuristic_enabled=args.heuristic,
        accel_threshold=args.accel_threshold,
        brake_threshold=args.brake_threshold,
        turn_threshold=args.turn_threshold,
        lane_change_threshold=args.lane_change_threshold,
        invert_gyro_z=args.invert_gyro_z,
        invert_acc_y=args.invert_acc_y,
    )

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    vehicle = _find_ego_vehicle(world, role_name=args.role_name, fallback_index=args.fallback_vehicle_index)
    if vehicle is None:
        print("No vehicle found in world. Spawn a vehicle/manual_control first.")
        return 1

    bp_lib = world.get_blueprint_library()
    imu_bp = bp_lib.find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", str(args.sensor_tick))

    imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
    print(f"Attached IMU to vehicle id={vehicle.id}, role_name={vehicle.attributes.get('role_name', '')}")
    print("Listening to realtime IMU stream. Press Ctrl+C to stop.")

    running = True
    last_announced_event = "safe"

    def _stop_handler(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    def _on_imu(data: carla.IMUMeasurement) -> None:
        nonlocal last_announced_event

        packet = _packet_from_imu(
            data=data,
            vehicle=vehicle,
            mag_mode=args.mag_mode,
            compass_scale=args.compass_scale,
            mag_field_strength=args.mag_field_strength,
            mag_declination_deg=args.mag_declination_deg,
            mag_inclination_deg=args.mag_inclination_deg,
        )
        result = detector.update(packet)

        if not (isinstance(result, dict) and result.get("ready")):
            return

        event = result.get("prediction", "safe")
        raw_event = result.get("raw_prediction", "safe")
        conf = float(result.get("confidence", 0.0))

        if event != "safe" and event == last_announced_event:
            return

        ts = f"{data.timestamp:.3f}"
        print(
            f"t={ts}s model={raw_event}->{event} conf={conf:.3f} "
            f"acc=({data.accelerometer.x:.2f},{data.accelerometer.y:.2f},{data.accelerometer.z:.2f}) "
            f"gyro=({data.gyroscope.x:.2f},{data.gyroscope.y:.2f},{data.gyroscope.z:.2f})"
        )

        if event == "safe":
            if args.print_safe and last_announced_event != "safe":
                last_announced_event = "safe"
                print(f">>> SAFE: {event} (confidence {conf:.3f})")
            else:
                last_announced_event = "safe"
            return

        last_announced_event = event
        print(f">>> HARSH EVENT: {event} (confidence {conf:.3f})")

    imu_sensor.listen(_on_imu)

    try:
        while running:
            time.sleep(0.2)
    finally:
        imu_sensor.stop()
        imu_sensor.destroy()
        print("IMU sensor cleaned up. Exiting.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
