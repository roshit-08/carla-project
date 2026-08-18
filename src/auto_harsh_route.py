import argparse
import csv
import math
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import carla

from harsh_event_model import RealtimeHarshEventDetector


def _rotation_matrix_body_to_world(
    roll_deg: float, pitch_deg: float, yaw_deg: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _world_to_body_vector(
    wx: float, wy: float, wz: float, rot: carla.Rotation
) -> Tuple[float, float, float]:
    r = _rotation_matrix_body_to_world(rot.roll, rot.pitch, rot.yaw)
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
    dec = math.radians(declination_deg)
    inc = math.radians(inclination_deg)

    horizontal = field_strength * math.cos(inc)
    world_x = horizontal * math.cos(dec)
    world_y = horizontal * math.sin(dec)
    world_z = -field_strength * math.sin(inc)

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
) -> Dict[str, float]:
    if mag_mode == "virtual3d":
        mag_x, mag_y, mag_z = _virtual_magnetometer_from_vehicle(
            vehicle=vehicle,
            field_strength=mag_field_strength,
            declination_deg=mag_declination_deg,
            inclination_deg=mag_inclination_deg,
        )
    else:
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


def build_route(start_waypoint: carla.Waypoint, route_length: float, spacing: float) -> List[carla.Waypoint]:
    waypoints: List[carla.Waypoint] = [start_waypoint]
    current = start_waypoint
    visited = set()

    while len(waypoints) * spacing < route_length:
        next_waypoints = current.next(spacing)
        if not next_waypoints:
            break

        current = next_waypoints[0]
        key = (
            current.road_id,
            current.lane_id,
            round(current.transform.location.x, 1),
            round(current.transform.location.y, 1),
        )
        if key in visited:
            break
        visited.add(key)
        waypoints.append(current)

    return waypoints


def get_current_speed_kmh(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)


def compute_steering_control(
    vehicle: carla.Vehicle, target_location: carla.Location, target_speed: float
) -> carla.VehicleControl:
    transform = vehicle.get_transform()
    forward = transform.get_forward_vector()
    location = transform.location
    dx = target_location.x - location.x
    dy = target_location.y - location.y
    distance = math.sqrt(dx * dx + dy * dy)

    dot = forward.x * dx + forward.y * dy
    det = forward.x * dy - forward.y * dx
    angle = math.atan2(det, dot) if distance > 0 else 0.0

    steer = max(-1.0, min(1.0, 1.5 * angle))
    speed = get_current_speed_kmh(vehicle)
    if speed < target_speed * 0.9:
        throttle = 0.7
        brake = 0.0
    elif speed > target_speed * 1.05:
        throttle = 0.0
        brake = 0.2
    else:
        throttle = 0.4
        brake = 0.0

    throttle *= max(0.25, 1.0 - abs(steer))
    steer *= 0.85

    return carla.VehicleControl(
        throttle=throttle,
        steer=steer,
        brake=brake,
        hand_brake=False,
        reverse=False,
    )


def event_control(
    event_name: str,
    vehicle: carla.Vehicle,
    current_waypoint: carla.Waypoint,
    target_waypoint: Optional[carla.Waypoint],
) -> carla.VehicleControl:
    if event_name == "sudden_acceleration":
        return carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0)
    if event_name == "sudden_braking":
        return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
    if event_name == "harsh_left_turn":
        return carla.VehicleControl(throttle=0.5, steer=-0.9, brake=0.0)
    if event_name == "harsh_right_turn":
        return carla.VehicleControl(throttle=0.5, steer=0.9, brake=0.0)
    if event_name in ("harsh_left_lane_change", "harsh_right_lane_change"):
        if target_waypoint is not None:
            return compute_steering_control(vehicle, target_waypoint.transform.location, 20.0)
        return carla.VehicleControl(
            throttle=0.4,
            steer=0.5 if event_name == "harsh_right_lane_change" else -0.5,
            brake=0.0,
        )

    return compute_steering_control(vehicle, current_waypoint.transform.location, 20.0)


def find_adjacent_lane_waypoint(current_waypoint: carla.Waypoint, event_name: str) -> Optional[carla.Waypoint]:
    if event_name == "harsh_left_lane_change":
        return current_waypoint.get_left_lane()
    if event_name == "harsh_right_lane_change":
        return current_waypoint.get_right_lane()
    return None


def select_vehicle_blueprint(bp_lib: carla.BlueprintLibrary) -> carla.ActorBlueprint:
    candidates = [bp for bp in bp_lib.filter("vehicle.*") if bp.has_attribute("number_of_wheels")]
    cars = [bp for bp in candidates if int(bp.get_attribute("number_of_wheels")) == 4]
    return random.choice(cars or candidates)


def find_or_spawn_hero_vehicle(
    world: carla.World,
    role_name: str,
    fallback_index: int,
    spawn_point: Optional[carla.Transform],
) -> carla.Vehicle:
    ego = None
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name", "") == role_name:
            ego = actor
            break

    if ego is not None:
        print(f"Using existing vehicle id={ego.id} role_name={role_name}")
        ego.set_autopilot(False)
        return ego

    bp_lib = world.get_blueprint_library()
    vehicle_bp = select_vehicle_blueprint(bp_lib)
    vehicle_bp.set_attribute("role_name", role_name)
    spawn_transform = spawn_point or random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
    vehicle.set_autopilot(False)
    print(f"Spawned vehicle id={vehicle.id} at {spawn_transform.location}")
    return vehicle


def main() -> int:
    parser = argparse.ArgumentParser(description="CARLA automated harsh driving route with realtime validation")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--timeout", type=float, default=10.0, help="CARLA client timeout")
    parser.add_argument("--model-path", default="artifacts/harsh_event_rf.joblib", help="Trained model bundle path")
    parser.add_argument("--route-length", type=float, default=200.0, help="Route length in meters")
    parser.add_argument("--route-spacing", type=float, default=5.0, help="Waypoint spacing in meters")
    parser.add_argument("--target-speed", type=float, default=25.0, help="Target speed in km/h")
    parser.add_argument("--sensor-tick", type=float, default=0.05, help="IMU update interval seconds")
    parser.add_argument("--print-safe", action="store_true", help="Print safe predictions too")
    parser.add_argument("--heuristic", action="store_true", help="Enable heuristic fallback for realtime detector")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum confidence for non-safe event")
    parser.add_argument("--mag-mode", choices=["virtual3d", "compass2d"], default="virtual3d", help="Magnetometer mode")
    parser.add_argument("--mag-field-strength", type=float, default=100.0, help="Virtual magnetic field strength")
    parser.add_argument("--mag-declination-deg", type=float, default=0.0, help="Virtual magnetic declination")
    parser.add_argument("--mag-inclination-deg", type=float, default=60.0, help="Virtual magnetic inclination")
    parser.add_argument("--invert-gyro-z", action="store_true", help="Invert gyro_z sign")
    parser.add_argument("--invert-acc-y", action="store_true", help="Invert acc_y sign")
    parser.add_argument("--role-name", default="hero", help="Role name for hero vehicle")
    parser.add_argument("--fallback-vehicle-index", type=int, default=0, help="Fallback vehicle index")
    parser.add_argument("--spawn-point-index", type=int, default=0, help="Spawn point index to use if new vehicle is created")
    parser.add_argument("--log-file", default="validation_log.csv", help="Path to save validation log CSV")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()

    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        print("No spawn points available in CARLA map.")
        return 1

    spawn_transform = spawn_points[min(args.spawn_point_index, len(spawn_points) - 1)]
    vehicle = find_or_spawn_hero_vehicle(world, args.role_name, args.fallback_vehicle_index, spawn_transform)

    detector = RealtimeHarshEventDetector(
        model_bundle_path=Path(args.model_path),
        min_confidence=args.min_confidence,
        consecutive_hits=2,
        heuristic_enabled=args.heuristic,
        accel_threshold=2.0,
        brake_threshold=-2.0,
        turn_threshold=40.0,
        lane_change_threshold=2.0,
        invert_gyro_z=args.invert_gyro_z,
        invert_acc_y=args.invert_acc_y,
    )

    imu_bp = world.get_blueprint_library().find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", str(args.sensor_tick))
    imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)

    last_announced = "safe"
    running = True
    log_file = open(args.log_file, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["timestamp", "event_type", "event_name", "confidence", "raw_prediction", "prediction", "route_index", "speed_kmh", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])

    def _stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    def _on_imu(data: carla.IMUMeasurement) -> None:
        nonlocal last_announced
        packet = _packet_from_imu(
            data=data,
            vehicle=vehicle,
            mag_mode=args.mag_mode,
            compass_scale=100.0,
            mag_field_strength=args.mag_field_strength,
            mag_declination_deg=args.mag_declination_deg,
            mag_inclination_deg=args.mag_inclination_deg,
        )
        result = detector.update(packet)
        if not (isinstance(result, dict) and result.get("ready")):
            return

        event = result.get("prediction", "safe")
        raw_event = result.get("raw_prediction", "safe")
        confidence = float(result.get("confidence", 0.0))
        ts = f"{data.timestamp:.3f}"
        speed = get_current_speed_kmh(vehicle)

        # Log every detection
        log_writer.writerow([
            ts, "detection", event, confidence, raw_event, event, current_route_index, speed,
            packet["acc_x"], packet["acc_y"], packet["acc_z"],
            packet["gyro_x"], packet["gyro_y"], packet["gyro_z"]
        ])

        if event == "safe":
            if args.print_safe and last_announced != "safe":
                last_announced = "safe"
                print(f"[{ts}] SAFE (conf={confidence:.3f})")
            else:
                last_announced = "safe"
            return

        if event != last_announced:
            print(f"[{ts}] HARSH EVENT DETECTED: {raw_event} -> {event} (conf={confidence:.3f})")
            last_announced = event

    imu_sensor.listen(_on_imu)

    start_wp = carla_map.get_waypoint(vehicle.get_transform().location)
    route = build_route(start_wp, args.route_length, args.route_spacing)
    if len(route) < 5:
        print("Could not build a waypoint route. Try a smaller route length or larger spacing.")
        imu_sensor.stop()
        imu_sensor.destroy()
        return 1

    print(f"Route size: {len(route)} waypoints, length approx. {len(route) * args.route_spacing:.1f} meters")
    print("Starting controlled drive. Press Ctrl+C to stop.")

    event_schedule = [
        (5, "sudden_acceleration", 1.5),
        (15, "sudden_braking", 1.5),
        (25, "harsh_right_turn", 2.0),
        (35, "harsh_left_lane_change", 2.0),
        (45, "harsh_right_lane_change", 2.0),
    ]
    event_done = {name: False for _, name, _ in event_schedule}
    current_route_index = 0
    active_event: Optional[str] = None
    event_end_time = 0.0
    lane_change_target: Optional[carla.Waypoint] = None
    control = carla.VehicleControl()

    try:
        while running:
            location = vehicle.get_transform().location
            if current_route_index < len(route) - 1:
                target_wp = route[current_route_index]
                distance = location.distance(target_wp.transform.location)
                if distance < 3.0 and current_route_index < len(route) - 1:
                    current_route_index += 1
                    target_wp = route[current_route_index]
            else:
                target_wp = route[-1]

            now = time.time()
            if active_event and now >= event_end_time:
                active_event = None
                lane_change_target = None

            if active_event is None:
                for idx, name, duration in event_schedule:
                    if current_route_index >= idx and not event_done[name]:
                        event_done[name] = True
                        active_event = name
                        # Log the trigger
                        log_writer.writerow([
                            time.time(), "trigger", name, 1.0, name, name, idx, get_current_speed_kmh(vehicle),
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Placeholder IMU values for trigger
                        ])
                        event_end_time = now + duration
                        if name in ("harsh_left_lane_change", "harsh_right_lane_change"):
                            lane_change_target = find_adjacent_lane_waypoint(target_wp, name)
                            if lane_change_target is None:
                                print(f"Skipping lane change at waypoint {idx}: adjacent lane unavailable")
                                active_event = None
                                continue
                        print(f"Triggering event: {name} at route index {idx}")
                        break

            if active_event:
                control = event_control(active_event, vehicle, target_wp, lane_change_target)
            else:
                control = compute_steering_control(vehicle, target_wp.transform.location, args.target_speed)

            vehicle.apply_control(control)
            time.sleep(0.05)

    finally:
        imu_sensor.stop()
        imu_sensor.destroy()
        log_file.close()
        print(f"Shutting down. Sensor cleaned up. Log saved to {args.log_file}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
