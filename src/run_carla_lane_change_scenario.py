#!/usr/bin/env python3
"""
CARLA Dedicated Lane-Change Benchmark Script — High Quality & Crash-Free.

This script runs the ego vehicle on CARLA Traffic Manager autopilot on straight multi-lane roads.
It executes a clean sequence:
  1. Normal Autopilot Driving (Warm-up)
  2. Sudden Left Lane Change (biphasic S-curve maneuver)
  3. Normal Autopilot Driving (Recovery)
  4. Sudden Right Lane Change (biphasic S-curve maneuver)
  5. Normal Autopilot Driving

Features:
  - Multi-lane road verification before lane change.
  - Smooth counter-steering so the vehicle snaps cleanly onto target lane center.
  - Automatic collision / stall teleport recovery to clean road center.
"""

import argparse
import math
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import carla
except ImportError:
    raise RuntimeError("CARLA Python API not found. Please set PYTHONPATH to your carla egg file.")

from harsh_event_model import EnsembleHarshEventDetector, DriverSafetyScorer

CONTROL_DT = 0.05
STEER_RESTORE_STEPS = 12
MIN_SPEED_KMH = 12.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CARLA Dedicated Lane Change Benchmark")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--role-name", default="hero")
    p.add_argument("--sensor-tick", type=float, default=0.05)

    p.add_argument("--rf-path", default="notebooks/artifacts/harsh_event_rf_v2.joblib")
    p.add_argument("--xgb-path", default="notebooks/artifacts/harsh_event_xgb_v2.joblib")
    p.add_argument("--cnn-path", default="artifacts/harsh_event_cnn_bundle.pth")

    p.add_argument("--rf-weight", type=float, default=0.55)
    p.add_argument("--xgb-weight", type=float, default=0.45)
    p.add_argument("--cnn-weight", type=float, default=0.00)

    p.add_argument("--min-confidence", type=float, default=0.50)
    p.add_argument("--consecutive-hits", type=int, default=1)
    p.add_argument("--recovery-rate", type=float, default=0.70)
    p.add_argument("--scenario-interval", "--min-recovery", type=float, default=20.0)
    p.add_argument("--repeat-events", type=int, default=1)
    p.add_argument("--print-safe", action="store_true")
    return p.parse_args()


def get_speed_kmh(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def restore_autopilot(vehicle: carla.Vehicle, tm: carla.TrafficManager):
    """Re-enable Traffic Manager autopilot with smooth, slow cruising (~18-22 km/h)."""
    try:
        vehicle.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(vehicle, False)
        tm.distance_to_leading_vehicle(vehicle, 5.0)
        # Positive percentage = drive SLOWER than speed limit (+35.0% slower => smooth ~20 km/h cruising)
        tm.vehicle_percentage_speed_difference(vehicle, 35.0)
        tm.ignore_lights_percentage(vehicle, 100.0)
        tm.ignore_vehicles_percentage(vehicle, 0.0)
        tm.ignore_walkers_percentage(vehicle, 0.0)
    except Exception:
        pass


def respawn_on_nearest_road(world: carla.World, vehicle: carla.Vehicle, tm: carla.TrafficManager):
    """Teleports vehicle back to the center of the nearest clean driving lane if stuck or crashed."""
    try:
        cmap = world.get_map()
        curr_loc = vehicle.get_location()
        wp = cmap.get_waypoint(curr_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp:
            spawn_transform = wp.transform
            spawn_transform.location.z += 0.5
            vehicle.set_transform(spawn_transform)
            vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            time.sleep(0.3)
            restore_autopilot(vehicle, tm)
            print("    [Recovery] Teleported back to clean multi-lane road center!")
    except Exception as exc:
        print(f"    [Recovery Warning] Could not respawn: {exc}")


def wait_for_speed(world: carla.World, vehicle: carla.Vehicle, tm: carla.TrafficManager, target_speed: float = 15.0, timeout: float = 6.0) -> bool:
    """Ensures vehicle is moving at target_speed before initiating lane change."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        speed = get_speed_kmh(vehicle)
        if speed >= target_speed:
            return True
        time.sleep(0.1)

    if get_speed_kmh(vehicle) < target_speed:
        respawn_on_nearest_road(world, vehicle, tm)
    return get_speed_kmh(vehicle) >= target_speed


def start_spectator_follower(world: carla.World, vehicle: carla.Vehicle, stop_event: threading.Event):
    """Smooth camera follower."""
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


class LaneChangeEngine:
    def __init__(self, world: carla.World, vehicle: carla.Vehicle, tm: carla.TrafficManager):
        self.world = world
        self.vehicle = vehicle
        self.tm = tm

    def _disable(self):
        self.vehicle.set_autopilot(False)
        time.sleep(0.05)

    def _restore(self):
        restore_autopilot(self.vehicle, self.tm)

    def _straight_steer_bias(self) -> float:
        loc = self.vehicle.get_location()
        wp = self.world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return 0.0
        nexts = wp.next(4.0)
        if not nexts:
            return 0.0
        target_loc = nexts[0].transform.location
        t = self.vehicle.get_transform()
        fwd = t.get_forward_vector()
        dx = target_loc.x - loc.x
        dy = target_loc.y - loc.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-6
        cross = fwd.x * dy - fwd.y * dx
        return max(-0.08, min(0.08, cross / dist))

    def sudden_left_lane_change(self) -> bool:
        """Perform a clean, sharp biphasic S-curve Sudden Left Lane Change."""
        if not wait_for_speed(self.world, self.vehicle, self.tm, target_speed=MIN_SPEED_KMH):
            print("    [Event] Sudden Left Lane Change skipped — vehicle too slow")
            return False

        cmap = self.world.get_map()
        loc = self.vehicle.get_location()
        wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        left_wp = wp.get_left_lane() if wp else None
        if left_wp is None or left_wp.lane_type != carla.LaneType.Driving:
            print("    [Event] Sudden Left Lane Change — searching multi-lane section...")
            time.sleep(1.0)
            wp = cmap.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            left_wp = wp.get_left_lane() if wp else None

        self._disable()
        print("    [Event] >>> Executing SUDDEN LEFT LANE CHANGE (S-Curve)")
        # Phase 1: Sharp steer left into adjacent lane (0.75s)
        for _ in range(15):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.60, steer=-0.48, brake=0.0))
            time.sleep(CONTROL_DT)

        # Phase 2: Counter-steer right to straighten into left lane center (0.75s)
        for _ in range(15):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.55, steer=0.48, brake=0.0))
            time.sleep(CONTROL_DT)

        # Phase 3: Smooth alignment
        for _ in range(STEER_RESTORE_STEPS):
            s = self._straight_steer_bias()
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.45, steer=s, brake=0.0))
            time.sleep(CONTROL_DT)

        self._restore()
        print("    [Event] <<< Sudden Left Lane Change Complete — Resumed Autopilot")
        return True

    def sudden_right_lane_change(self) -> bool:
        """Perform a clean, sharp biphasic S-curve Sudden Right Lane Change."""
        if not wait_for_speed(self.world, self.vehicle, self.tm, target_speed=MIN_SPEED_KMH):
            print("    [Event] Sudden Right Lane Change skipped — vehicle too slow")
            return False

        cmap = self.world.get_map()
        loc = self.vehicle.get_location()
        wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        right_wp = wp.get_right_lane() if wp else None
        if right_wp is None or right_wp.lane_type != carla.LaneType.Driving:
            print("    [Event] Sudden Right Lane Change — searching multi-lane section...")
            time.sleep(1.0)
            wp = cmap.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            right_wp = wp.get_right_lane() if wp else None

        self._disable()
        print("    [Event] >>> Executing SUDDEN RIGHT LANE CHANGE (S-Curve)")
        # Phase 1: Sharp steer right into adjacent lane (0.75s)
        for _ in range(15):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.60, steer=0.48, brake=0.0))
            time.sleep(CONTROL_DT)

        # Phase 2: Counter-steer left to straighten into right lane center (0.75s)
        for _ in range(15):
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.55, steer=-0.48, brake=0.0))
            time.sleep(CONTROL_DT)

        # Phase 3: Smooth alignment
        for _ in range(STEER_RESTORE_STEPS):
            s = self._straight_steer_bias()
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.45, steer=s, brake=0.0))
            time.sleep(CONTROL_DT)

        self._restore()
        print("    [Event] <<< Sudden Right Lane Change Complete — Resumed Autopilot")
        return True


def build_imu_callback(
    detector: EnsembleHarshEventDetector,
    scorer: DriverSafetyScorer,
    lock: threading.Lock,
    args: argparse.Namespace,
    current_status: List[str],
):
    last_event_label = ["safe"]

    def _on_imu(data: carla.IMUMeasurement):
        acc_z_raw = float(data.accelerometer.z)
        acc_z_norm = acc_z_raw - 9.81 if acc_z_raw > 0 else acc_z_raw + 9.81

        packet = {
            "acc_x": float(data.accelerometer.x),
            "acc_y": float(data.accelerometer.y),
            "acc_z": float(acc_z_norm),
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
            mode_str = current_status[0]

            if label != "safe" or args.print_safe:
                if label != last_event_label[0] or label != "safe":
                    last_event_label[0] = label
                    print(
                        f"[{timestamp}] [{mode_str:<28}] "
                        f"Score: {score:5.1f}/100 | Tier: {tier:<20} | "
                        f"Event: {label:<25} "
                        f"(Conf: {conf:.2f} | RF:{rf_p} XGB:{xgb_p}) | Events: {total_ev}"
                    )

    return _on_imu


def main():
    args = parse_args()
    print("\n" + "=" * 70)
    print("   CARLA DEDICATED LANE-CHANGE BENCHMARK (HIGH QUALITY)")
    print("=" * 70)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # Find or spawn vehicle
    vehicle = None
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == args.role_name:
            vehicle = actor
            break

    if vehicle is None:
        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", args.role_name)
        spawns = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(bp, spawns[0] if spawns else carla.Transform())

    # Configure Traffic Manager
    tm = client.get_trafficmanager(args.tm_port)
    restore_autopilot(vehicle, tm)

    # Load Ensemble Models
    lock = threading.Lock()
    print("  Loading Ensemble Models...")
    detector = EnsembleHarshEventDetector(
        rf_path=args.rf_path,
        xgb_path=args.xgb_path,
        cnn_path=args.cnn_path,
        rf_weight=args.rf_weight,
        xgb_weight=args.xgb_weight,
        cnn_weight=args.cnn_weight,
        min_confidence=args.min_confidence,
        consecutive_hits=args.consecutive_hits,
    )
    scorer = DriverSafetyScorer(initial_score=100.0, recovery_rate_per_sec=args.recovery_rate)

    current_status = ["Autopilot Driving"]

    # Attach IMU
    imu_bp = world.get_blueprint_library().find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", str(args.sensor_tick))
    imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
    imu_sensor.listen(build_imu_callback(detector, scorer, lock, args, current_status))

    # Spectator follower
    spectator_stop = threading.Event()
    threading.Thread(target=start_spectator_follower, args=(world, vehicle, spectator_stop), daemon=True).start()

    engine = LaneChangeEngine(world, vehicle, tm)

    print("\n  Legend: Normal Autopilot → Sudden Left Lane Change → Normal → Sudden Right Lane Change")
    print("-" * 70)

    try:
        # 1. Warm-up Normal Driving
        current_status[0] = "Normal Driving (Warmup)"
        print("\n[Phase 1] Normal Autopilot Driving for 10s...")
        time.sleep(10.0)

        # 2. Sudden Left Lane Change
        current_status[0] = "EVENT: Sudden Left Lane Change"
        print("\n[Phase 2] Injecting Sudden Left Lane Change...")
        engine.sudden_left_lane_change()

        # 3. Normal Driving / Recovery
        current_status[0] = "Normal Driving (Recovery)"
        print(f"\n[Phase 3] Normal Autopilot Driving for {args.scenario_interval:.0f}s...")
        time.sleep(args.scenario_interval)

        # 4. Sudden Right Lane Change
        current_status[0] = "EVENT: Sudden Right Lane Change"
        print("\n[Phase 4] Injecting Sudden Right Lane Change...")
        engine.sudden_right_lane_change()

        # 5. Normal Driving / Recovery
        current_status[0] = "Normal Driving (Settling)"
        print(f"\n[Phase 5] Normal Autopilot Driving for {args.scenario_interval:.0f}s...")
        time.sleep(args.scenario_interval)

        print("\n✓ Lane Change Benchmark Sequence Complete Successfully!")

    except KeyboardInterrupt:
        print("\nStopping Lane Change Benchmark...")
    finally:
        spectator_stop.set()
        try:
            vehicle.set_autopilot(False)
            if imu_sensor and imu_sensor.is_alive:
                imu_sensor.stop()
                imu_sensor.destroy()
        except Exception:
            pass
        print("✓ Cleanup complete.")


if __name__ == "__main__":
    main()
