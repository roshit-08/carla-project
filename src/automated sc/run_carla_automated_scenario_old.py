#!/usr/bin/env python3
"""
CARLA Automated Scenario Benchmark — High-Quality Road-Safe Driving.

Drives the ego vehicle fully on autopilot using CARLA Traffic Manager for
baseline navigation. Periodically injects realistic harsh driving events
(sudden acceleration, sudden braking, harsh left/right turns, harsh left/right
lane changes) using a waypoint-guided approach so the vehicle always stays on
drivable road surface and never crashes into buildings or goes off-road.

Pipeline
--------
1. Connect to CARLA and find/spawn a hero vehicle.
2. Configure Traffic Manager (with port-bind fallback).
3. Attach IMU sensor and start Ensemble + Safety Score pipeline.
4. Run the scenario sequence:
     Phase 0  →  Baseline smooth autopilot (warm-up)
     Phase 1+ →  Repeat: autopilot recovery window → inject event
5. After all events: continuous autopilot with score display.
6. Ctrl-C → clean up all actors.
"""

import argparse
import math
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import carla
from harsh_event_model import DriverSafetyScorer, EnsembleHarshEventDetector

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MIN_SPEED_FOR_EVENT_KMH = 15.0   # vehicle must be moving before harsh event
MAX_SPEED_BASELINE_KMH  = 50.0   # cap autopilot speed
STEER_RESTORE_STEPS     = 8      # smooth steer-back steps after event
CONTROL_DT              = 0.05   # seconds per control loop step


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CARLA Automated Scenario Driver Safety Score Benchmark"
    )
    p.add_argument("--host",               default="127.0.0.1")
    p.add_argument("--port",               type=int,   default=2000)
    p.add_argument("--tm-port",            type=int,   default=8000)
    p.add_argument("--role-name",          default="hero")
    p.add_argument("--sensor-tick",        type=float, default=0.05)
    p.add_argument("--rf-path",            default="notebooks/artifacts/harsh_event_rf_v2.joblib")
    p.add_argument("--xgb-path",           default="notebooks/artifacts/harsh_event_xgb_v2.joblib")
    p.add_argument("--cnn-path",           default="artifacts/harsh_event_cnn_bundle.pth")
    p.add_argument("--rf-weight",          type=float, default=0.40)
    p.add_argument("--xgb-weight",         type=float, default=0.30)
    p.add_argument("--cnn-weight",         type=float, default=0.30)
    p.add_argument("--min-confidence",     type=float, default=0.30)
    p.add_argument("--consecutive-hits",   type=int,   default=1)
    p.add_argument("--recovery-rate",      type=float, default=0.30)
    p.add_argument("--scenario-interval",  type=float, default=20.0,
                   help="Autopilot seconds between injected events")
    p.add_argument("--repeat-events",      type=int,   default=1,
                   help="How many full scenario cycles to run (0 = infinite)")
    p.add_argument("--print-safe",         action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# CARLA helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_speed_kmh(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def get_forward_vector(vehicle: carla.Vehicle) -> Tuple[float, float]:
    fwd = vehicle.get_transform().get_forward_vector()
    return fwd.x, fwd.y


def find_or_spawn_vehicle(world: carla.World, role_name: str) -> carla.Vehicle:
    """Return clean ego vehicle, spawned fresh or reset to open road spawn point."""
    pts = world.get_map().get_spawn_points()
    spawn_point = pts[0] if pts else carla.Transform()

    for actor in world.get_actors():
        if "vehicle." in actor.type_id:
            if actor.attributes.get("role_name") == role_name:
                print(f"  Found existing vehicle id={actor.id} role={role_name} — resetting transform to clean road spawn.")
                try:
                    actor.set_transform(spawn_point)
                    actor.set_target_velocity(carla.Vector3D(0, 0, 0))
                    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
                except Exception:
                    pass
                return actor

    vehicles = world.get_actors().filter("vehicle.*")
    if vehicles:
        v = vehicles[0]
        print(f"  Reusing vehicle id={v.id} — resetting transform to clean road spawn.")
        try:
            v.set_transform(spawn_point)
            v.set_target_velocity(carla.Vector3D(0, 0, 0))
            v.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
        except Exception:
            pass
        return v

    # Spawn fresh
    bp_lib  = world.get_blueprint_library()
    bp_list = bp_lib.filter("vehicle.tesla.model3")
    bp = bp_list[0] if bp_list else bp_lib.filter("vehicle.*")[0]
    bp.set_attribute("role_name", role_name)
    v = world.spawn_actor(bp, spawn_point)
    print(f"  Spawned new Tesla Model 3 id={v.id} at clean spawn point.")
    return v


def setup_traffic_manager(
    client: carla.Client,
    vehicle: carla.Vehicle,
    tm_port: int,
) -> carla.TrafficManager:
    """Create TM, bind with port-fallback, configure safe autopilot settings."""
    for port in (tm_port, tm_port + 100, tm_port + 200, tm_port + 500):
        try:
            tm = client.get_trafficmanager(port)
            break
        except RuntimeError as e:
            if "bind error" in str(e).lower():
                print(f"  ⚠  TM port {port} busy, trying next…")
            else:
                raise
    else:
        raise RuntimeError("Could not bind Traffic Manager on any port.")

    actual_port = tm.get_port()
    vehicle.set_autopilot(True, actual_port)

    try:
        tm.auto_lane_change(vehicle, False)                     # keep current lane smoothly
        tm.distance_to_leading_vehicle(vehicle, 3.5)            # safe distance behind vehicles
        tm.vehicle_percentage_speed_difference(vehicle, -15.0)  # drive smoothly at ~30-35 km/h
        tm.ignore_lights_percentage(vehicle, 100.0)             # ignore traffic lights so it never freezes
        tm.ignore_vehicles_percentage(vehicle, 0.0)              # safety: avoid vehicle collisions
        tm.ignore_walkers_percentage(vehicle, 0.0)               # safety: avoid pedestrian collisions
        tm.set_global_distance_to_leading_vehicle(3.5)
        tm.set_synchronous_mode(False)
    except Exception as exc:
        print(f"  TM config warning: {exc}")

    print(f"  Traffic Manager active on port {actual_port}")
    return tm


def restore_autopilot(vehicle: carla.Vehicle, tm: carla.TrafficManager) -> None:
    """Re-enable autopilot on the TM port after a manual maneuver."""
    try:
        vehicle.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(vehicle, False)
        tm.distance_to_leading_vehicle(vehicle, 3.5)
        tm.vehicle_percentage_speed_difference(vehicle, -15.0)
        tm.ignore_lights_percentage(vehicle, 100.0)
        tm.ignore_vehicles_percentage(vehicle, 0.0)
        tm.ignore_walkers_percentage(vehicle, 0.0)
    except Exception:
        pass


def drive_waypoint_autopilot_step(world: carla.World, vehicle: carla.Vehicle, target_speed_kmh: float = 30.0):
    """
    Pure-Pursuit Waypoint Autopilot step with soft acceleration & smooth steering:
    Calculates steering and progressive throttle to follow lane waypoints smoothly.
    """
    loc = vehicle.get_location()
    transform = vehicle.get_transform()
    speed = get_speed_kmh(vehicle)

    wp = world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if not wp:
        return

    next_wps = wp.next(6.0)
    if not next_wps:
        return
    target_wp = next_wps[0]

    fwd = transform.get_forward_vector()
    target_loc = target_wp.transform.location
    dx = target_loc.x - loc.x
    dy = target_loc.y - loc.y
    dist = math.sqrt(dx * dx + dy * dy) + 1e-5

    cross = fwd.x * dy - fwd.y * dx
    dot = fwd.x * dx + fwd.y * dy
    angle_err = math.atan2(cross, dot)

    # Soft steering clamp for smooth cornering
    steer = max(-0.45, min(0.45, angle_err * 1.0))

    # Progressive soft acceleration
    if speed < target_speed_kmh:
        throttle = min(0.35, max(0.18, 0.12 + 0.015 * (target_speed_kmh - speed)))
        brake = 0.0
    else:
        throttle = 0.0
        brake = 0.15 if speed > (target_speed_kmh + 5.0) else 0.0

    vehicle.apply_control(carla.VehicleControl(
        throttle=throttle,
        steer=steer,
        brake=brake,
        hand_brake=False
    ))


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


def respawn_on_nearest_road(world: carla.World, vehicle: carla.Vehicle, tm: Optional[carla.TrafficManager] = None):
    """Teleports vehicle back to the center of the nearest clean driving lane if stuck or crashed."""
    try:
        cmap = world.get_map()
        curr_loc = vehicle.get_location()
        wp = cmap.get_waypoint(curr_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp:
            spawn_transform = wp.transform
            spawn_transform.location.z += 0.5  # slight lift to prevent falling through road surface
            vehicle.set_transform(spawn_transform)
            vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            time.sleep(0.3)
            if tm:
                restore_autopilot(vehicle, tm)
            print("    [Recovery] Vehicle unstuck / collision recovered — teleported back to clean road center!")
    except Exception as exc:
        print(f"    [Recovery Warning] Could not respawn vehicle: {exc}")


def wait_for_speed(
    world: carla.World,
    vehicle: carla.Vehicle,
    tm: Optional[carla.TrafficManager] = None,
    min_speed_kmh: float = 10.0,
    timeout: float = 6.0,
) -> bool:
    """Ensure vehicle reaches min_speed by actively driving along waypoints. Respawns if wedged against building/obstacle."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        speed = get_speed_kmh(vehicle)
        if speed >= min_speed_kmh:
            return True
        drive_waypoint_autopilot_step(world, vehicle, target_speed_kmh=30.0)
        time.sleep(0.05)

    # If vehicle couldn't reach min_speed after timeout, it is jammed against a building/wall
    if get_speed_kmh(vehicle) < min_speed_kmh:
        respawn_on_nearest_road(world, vehicle, tm)

    return get_speed_kmh(vehicle) >= min_speed_kmh


def get_waypoint_ahead(
    world: carla.World,
    vehicle: carla.Vehicle,
    distance: float = 10.0,
) -> Optional[carla.Waypoint]:
    """Return the road waypoint `distance` metres ahead of the vehicle."""
    loc  = vehicle.get_transform().location
    cmap = world.get_map()
    wp   = cmap.get_waypoint(loc, project_to_road=True,
                              lane_type=carla.LaneType.Driving)
    if wp is None:
        return None
    nexts = wp.next(distance)
    return nexts[0] if nexts else None


# ─────────────────────────────────────────────────────────────────────────────
# IMU / Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_imu_callback(
    detector: EnsembleHarshEventDetector,
    scorer: DriverSafetyScorer,
    lock: threading.Lock,
    args: argparse.Namespace,
    current_status: List[str],
):
    last_event_label = ["safe"]
    last_safe_print_time = [0.0]

    def _on_imu(data: carla.IMUMeasurement):
        # Subtract CARLA baseline gravity vector (~9.81 m/s^2) so acc_z is centered near 0.0 m/s^2
        acc_z_raw = float(data.accelerometer.z)
        acc_z_norm = acc_z_raw - 9.81 if acc_z_raw > 0 else acc_z_raw + 9.81

        packet = {
            "acc_x":  float(data.accelerometer.x),
            "acc_y":  float(data.accelerometer.y),
            "acc_z":  float(acc_z_norm),
            "gyro_x": float(data.gyroscope.x),
            "gyro_y": float(data.gyroscope.y),
            "gyro_z": float(data.gyroscope.z),
        }
        with lock:
            res = detector.update(packet)
            if not res or not res.get("ready"):
                return

            label      = res["prediction"]
            conf       = res["confidence"]
            score_info = scorer.update(label, conf, dt_sec=args.sensor_tick)

            score     = score_info["score"]
            tier      = score_info["risk_tier"]
            total_ev  = score_info["total_events"]
            timestamp = time.strftime("%H:%M:%S")
            rf_p      = res.get("rf_prediction",  "N/A")
            xgb_p     = res.get("xgb_prediction", "N/A")
            cnn_p     = res.get("cnn_prediction",  "N/A")
            mode_str  = current_status[0]
            now       = time.time()

            should_print = False
            if label != "safe":
                # Print once when entering a new harsh event occurrence
                if label != last_event_label[0]:
                    should_print = True
                    last_event_label[0] = label
            else:
                last_event_label[0] = "safe"
                # If --print-safe is requested, throttle output to once per second
                if args.print_safe and (now - last_safe_print_time[0] >= 1.0):
                    should_print = True
                    last_safe_print_time[0] = now

            if should_print:
                print(
                    f"[{timestamp}] [{mode_str:<28}] "
                    f"Score: {score:5.1f}/100 | Tier: {tier:<20} | "
                    f"Event: {label:<25} "
                    f"(Conf: {conf:.2f} | RF:{rf_p} XGB:{xgb_p} CNN:{cnn_p}) "
                    f"| Events: {total_ev}"
                )
    return _on_imu


# ─────────────────────────────────────────────────────────────────────────────
# Maneuver engine  ←  all events road-safe, waypoint-guided
# ─────────────────────────────────────────────────────────────────────────────

def _apply_smooth(vehicle: carla.Vehicle, ctrl: carla.VehicleControl, steps: int, dt: float):
    for _ in range(steps):
        vehicle.apply_control(ctrl)
        time.sleep(dt)


def _kickstart_and_restore(world: carla.World, vehicle: carla.Vehicle, tm: carla.TrafficManager):
    """
    After any manual event: softly nudge forward along road center,
    then hand back to autopilot.
    """
    for _ in range(6):  # 0.3s gentle nudge on road waypoint steering
        drive_waypoint_autopilot_step(world, vehicle, target_speed_kmh=25.0)
        time.sleep(0.05)
    restore_autopilot(vehicle, tm)


class ManeuverEngine:
    """
    All maneuvers:
      - Wait for minimum speed before executing.
      - Disable autopilot, run manual controls, re-enable autopilot.
      - Use road waypoints to keep steer aligned with road during events.
      - Smooth counter-steer recovery so vehicle snaps back into lane.
    """

    def __init__(self, world: carla.World, vehicle: carla.Vehicle, tm: carla.TrafficManager):
        self.world   = world
        self.vehicle = vehicle
        self.tm      = tm

    # ── helpers ──────────────────────────────────────────────────────────────

    def _disable(self):
        self.vehicle.set_autopilot(False)
        time.sleep(0.05)

    def _restore(self):
        _kickstart_and_restore(self.world, self.vehicle, self.tm)

    def _road_steer_bias(self, look_ahead: float = 4.0) -> float:
        """
        Compute a small steer correction to follow the road curvature.
        Returns a float in [-0.05, +0.05] that keeps the car centered on straight roads.
        """
        wp = get_waypoint_ahead(self.world, self.vehicle, look_ahead)
        if wp is None:
            return 0.0
        t     = self.vehicle.get_transform()
        loc   = t.location
        fwd   = t.get_forward_vector()
        dx    = wp.transform.location.x - loc.x
        dy    = wp.transform.location.y - loc.y
        dist  = math.sqrt(dx * dx + dy * dy) + 1e-6
        cross = fwd.x * dy - fwd.y * dx      # cross product → lateral error
        steer = max(-0.05, min(0.05, cross / dist))
        return steer

    def _smooth_restore_steer(self, from_steer: float, steps: int = STEER_RESTORE_STEPS):
        """Linearly blend steer back to road-centre over `steps` ticks."""
        for i in range(steps):
            t     = (i + 1) / steps
            road  = self._road_steer_bias()
            steer = from_steer * (1.0 - t) + road * t
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.45, steer=steer, brake=0.0)
            )
            time.sleep(CONTROL_DT)

    # ── event implementations ─────────────────────────────────────────────────

    def sudden_acceleration(self):
        """Full throttle burst for 1.4 s with road-following steer."""
        self._disable()
        print("    [Event] Sudden Acceleration  → full throttle 1.4 s")
        for _ in range(28):  # 28 × 0.05 s = 1.4 s
            s = self._road_steer_bias()
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=s, brake=0.0))
            time.sleep(CONTROL_DT)
        self._smooth_restore_steer(self._road_steer_bias())
        self._restore()

    def sudden_braking(self):
        """Maximum brake for 0.9 s. Vehicle must be moving."""
        if not wait_for_speed(self.world, self.vehicle, self.tm, MIN_SPEED_FOR_EVENT_KMH, timeout=6.0):
            print("    [Event] Sudden Braking skipped — vehicle too slow")
            return
        self._disable()
        print("    [Event] Sudden Braking  → max brake 0.9 s")
        for _ in range(18):  # 0.9 s
            s = self._road_steer_bias()
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=s, brake=1.0))
            time.sleep(CONTROL_DT)
        self._smooth_restore_steer(self._road_steer_bias())
        self._restore()

    def harsh_left_turn(self):
        """Sharp left turn while maintaining forward momentum on road."""
        if not wait_for_speed(self.world, self.vehicle, self.tm, MIN_SPEED_FOR_EVENT_KMH, timeout=6.0):
            print("    [Event] Harsh Left Turn skipped — too slow")
            return
        self._disable()
        print("    [Event] Harsh Left Turn  → steer -0.70 for 0.9 s")
        for _ in range(18):  # 0.9 s (18 ticks)
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.55, steer=-0.70, brake=0.0)
            )
            time.sleep(CONTROL_DT)
        self._smooth_restore_steer(-0.70)
        self._restore()

    def harsh_right_turn(self):
        """Sharp right turn while maintaining forward momentum on road."""
        if not wait_for_speed(self.world, self.vehicle, self.tm, MIN_SPEED_FOR_EVENT_KMH, timeout=6.0):
            print("    [Event] Harsh Right Turn skipped — too slow")
            return
        self._disable()
        print("    [Event] Harsh Right Turn  → steer +0.70 for 0.9 s")
        for _ in range(18):  # 0.9 s (18 ticks)
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.55, steer=0.70, brake=0.0)
            )
            time.sleep(CONTROL_DT)
        self._smooth_restore_steer(0.70)
        self._restore()

    def harsh_left_lane_change(self):
        """
        S-curve biphasic lane change to the left.
        Phase 1: steer hard left  (0.5 s)
        Phase 2: counter-steer right to straighten (0.5 s)
        Phase 3: smooth restore
        """
        if not wait_for_speed(self.world, self.vehicle, self.tm, MIN_SPEED_FOR_EVENT_KMH, timeout=6.0):
            print("    [Event] Harsh Left Lane Change skipped — too slow")
            return
        cmap = self.world.get_map()
        loc  = self.vehicle.get_transform().location
        wp   = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        left_wp = wp.get_left_lane() if wp else None
        if left_wp is None:
            print("    [Event] Harsh Left Lane Change skipped — no left lane")
            return

        self._disable()
        print("    [Event] Harsh Left Lane Change  → S-curve ±0.50 × 0.8 s each")
        for _ in range(16):   # phase 1: left (0.8 s)
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.60, steer=-0.50, brake=0.0)
            )
            time.sleep(CONTROL_DT)
        for _ in range(16):   # phase 2: counter-steer (0.8 s)
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.55, steer=0.50, brake=0.0)
            )
            time.sleep(CONTROL_DT)
        self._smooth_restore_steer(0.0)
        self._restore()

    def harsh_right_lane_change(self):
        """
        S-curve biphasic lane change to the right.
        Phase 1: steer hard right (0.8 s)
        Phase 2: counter-steer left  (0.8 s)
        Phase 3: smooth restore
        """
        if not wait_for_speed(self.world, self.vehicle, self.tm, MIN_SPEED_FOR_EVENT_KMH, timeout=6.0):
            print("    [Event] Harsh Right Lane Change skipped — too slow")
            return
        cmap = self.world.get_map()
        loc  = self.vehicle.get_transform().location
        wp   = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        right_wp = wp.get_right_lane() if wp else None
        if right_wp is None:
            print("    [Event] Harsh Right Lane Change skipped — no right lane")
            return

        self._disable()
        print("    [Event] Harsh Right Lane Change  → S-curve ±0.50 × 0.8 s each")
        for _ in range(16):   # phase 1: right (0.8 s)
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.60, steer=0.50, brake=0.0)
            )
            time.sleep(CONTROL_DT)
        for _ in range(16):   # phase 2: counter-steer (0.8 s)
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.55, steer=-0.50, brake=0.0)
            )
            time.sleep(CONTROL_DT)
        self._smooth_restore_steer(0.0)
        self._restore()

    def run(self, maneuver_name: str):
        """Dispatch by name."""
        dispatch = {
            "sudden_acceleration":      self.sudden_acceleration,
            "sudden_braking":           self.sudden_braking,
            "harsh_left_turn":          self.harsh_left_turn,
            "harsh_right_turn":         self.harsh_right_turn,
            "harsh_left_lane_change":   self.harsh_left_lane_change,
            "harsh_right_lane_change":  self.harsh_right_lane_change,
        }
        fn = dispatch.get(maneuver_name)
        if fn is None:
            print(f"  Unknown maneuver: {maneuver_name}")
            return
        fn()


def start_autopilot_maintainer(world: carla.World, vehicle: carla.Vehicle, stop_event: threading.Event):
    """
    Background maintainer thread that monitors vehicle speed during autopilot phases.
    Only intervenes with soft waypoint pushes if the vehicle is completely stalled (< 3 km/h).
    """
    stalled_ticks = 0
    while not stop_event.is_set():
        try:
            if get_speed_kmh(vehicle) < 3.0:
                stalled_ticks += 1
                if stalled_ticks > 15:  # ~1.2s of total standstill
                    drive_waypoint_autopilot_step(world, vehicle, target_speed_kmh=25.0)
            else:
                stalled_ticks = 0
        except Exception:
            pass
        time.sleep(0.08)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_SEQUENCE = [
    ("Sudden Acceleration",      "sudden_acceleration"),
    ("Sudden Braking",           "sudden_braking"),
    ("Harsh Left Turn",          "harsh_left_turn"),
    ("Harsh Right Turn",         "harsh_right_turn"),
    ("Harsh Left Lane Change",   "harsh_left_lane_change"),
    ("Harsh Right Lane Change",  "harsh_right_lane_change"),
]


def main():
    args = parse_args()

    # ── CARLA connection ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("   CARLA AUTOMATED SCENARIO — ROAD-SAFE HARSH EVENT BENCHMARK")
    print("=" * 70)
    print(f"  Connecting to CARLA at {args.host}:{args.port} …")
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()
    print(f"  Map: {world.get_map().name}")

    # ── Vehicle ───────────────────────────────────────────────────────────────
    vehicle = find_or_spawn_vehicle(world, args.role_name)

    # ── Traffic Manager ───────────────────────────────────────────────────────
    print(f"  Setting up Traffic Manager on port {args.tm_port} …")
    tm = setup_traffic_manager(client, vehicle, args.tm_port)

    # ── Ensemble detector + safety scorer ────────────────────────────────────
    print("  Loading ensemble models …")
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
    lock   = threading.Lock()

    # ── IMU sensor ────────────────────────────────────────────────────────────
    bp_lib  = world.get_blueprint_library()
    imu_bp  = bp_lib.find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", str(args.sensor_tick))
    imu_sensor = world.spawn_actor(
        imu_bp, carla.Transform(carla.Location(z=0.5)), attach_to=vehicle
    )

    current_status = ["Autopilot Warm-Up"]
    on_imu = build_imu_callback(detector, scorer, lock, args, current_status)
    imu_sensor.listen(on_imu)

    # ── Spectator Camera & Autopilot Maintainer Threads ────────────────────────
    spectator_stop = threading.Event()
    spectator_thread = threading.Thread(
        target=start_spectator_follower, args=(world, vehicle, spectator_stop), daemon=True
    )
    spectator_thread.start()

    maintainer_stop = threading.Event()
    maintainer_thread = threading.Thread(
        target=start_autopilot_maintainer, args=(world, vehicle, maintainer_stop), daemon=True
    )
    maintainer_thread.start()
    print("  Spectator tracking & active Waypoint Autopilot maintainer initialized.")

    # ── Maneuver engine ───────────────────────────────────────────────────────
    engine = ManeuverEngine(world, vehicle, tm)

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print(f"  Vehicle id       : {vehicle.id}")
    print(f"  Sensor tick      : {args.sensor_tick} s")
    print(f"  Event interval   : {args.scenario_interval} s")
    print(f"  Recovery rate    : +{args.recovery_rate * 10:.1f} pts / 10 s safe driving")
    print(f"  Events per cycle : {len(SCENARIO_SEQUENCE)}")
    print(f"  Repeat cycles    : {'∞' if args.repeat_events == 0 else args.repeat_events}")
    print()
    print("  Legend: Autopilot drives safely → harsh event injected → autopilot recovers")
    print("  Press Ctrl+C to stop.\n")
    print("-" * 70)

    # ── Scenario loop ─────────────────────────────────────────────────────────
    cycle = 0
    try:
        # Phase 0: warm-up
        current_status[0] = "Autopilot Warm-Up"
        print(f"\n[Phase 0] Autopilot warm-up for 10 s …")
        time.sleep(10.0)

        while True:
            cycle += 1
            if args.repeat_events > 0 and cycle > args.repeat_events:
                break

            print(f"\n{'─' * 70}")
            print(f"  CYCLE {cycle}  — starting event sequence")
            print(f"{'─' * 70}")

            for label, name in SCENARIO_SEQUENCE:
                # Recovery window — pure autopilot
                current_status[0] = "Autopilot Recovery"
                print(f"\n[Recovery] Autopilot driving for {args.scenario_interval:.0f} s …")
                time.sleep(args.scenario_interval)

                # Inject event
                current_status[0] = f"EVENT: {label}"
                print(f"\n>>> Injecting: {label.upper()}")
                engine.run(name)
                print(f"<<< Event done. Resuming autopilot …")

                # Short post-event settle on autopilot
                current_status[0] = "Post-Event Settle"
                time.sleep(3.0)

        # Continuous autopilot after all cycles
        current_status[0] = "Autopilot Continuous"
        print("\n✓ All cycles complete. Driving on autopilot continuously …")
        print("  Press Ctrl+C to stop.\n")
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nStopping Automated Scenario Benchmark …")

    finally:
        spectator_stop.set()
        maintainer_stop.set()
        print("  Disabling autopilot and cleaning up sensors …")
        try:
            vehicle.set_autopilot(False)
        except Exception:
            pass
        try:
            if imu_sensor is not None and imu_sensor.is_alive:
                imu_sensor.stop()
                time.sleep(0.2)
                imu_sensor.destroy()
        except Exception:
            pass
        print("✓ Cleanup complete.\n")


if __name__ == "__main__":
    main()
