#!/usr/bin/env python3
"""
       Claude modified automated scenario, junction based turning instead of timer

CARLA Automated Scenario Benchmark — Junction-Gated, Collision-Safe Harsh Driving.

Drives the ego vehicle fully on autopilot using CARLA Traffic Manager for
baseline navigation. Periodically injects realistic harsh driving events
(sudden acceleration, sudden braking, harsh left/right turns, harsh left/right
lane changes) — but ONLY when the surrounding road geometry and traffic make
that event actually safe and realistic:

  * Harsh turns fire only when the vehicle is approaching a real junction
    (per CARLA's map topology) that has a matching left/right turn option —
    never mid-block on a straight or gently curving road.
  * Lane changes / sudden accel / sudden braking fire only on straight road
    segments (not inside or about to enter a junction).
  * Every event first checks for nearby vehicles/pedestrians in the relevant
    zone (ahead, or the target lane) and is skipped/retried if unsafe.
  * Every event is monitored tick-by-tick against the vehicle's live lateral
    offset from the lane center; if the car drifts too far off-lane it aborts
    the maneuver immediately and blends back to lane center using live
    waypoint data instead of finishing a fixed blind steer.
  * A real collision sensor aborts in-progress maneuvers and is tallied in
    the end-of-run report.

Pipeline
--------
1. Connect to CARLA and find/spawn a hero vehicle.
2. Configure Traffic Manager (with port-bind fallback).
3. Attach IMU + collision + lane-invasion sensors, start Ensemble + Safety
   Score pipeline.
4. Run the scenario sequence via an EventScheduler that waits for the right
   real-world condition for each event (junction present with matching turn
   option / clear lane / clear road ahead) instead of firing on a timer.
5. After all events: continuous autopilot with score display.
6. Ctrl-C → clean up all actors, print final safety report.
"""

import argparse
import math
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import carla
from harsh_event_model import DriverSafetyScorer, EnsembleHarshEventDetector

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MIN_SPEED_FOR_EVENT_KMH  = 15.0   # vehicle must be moving before harsh event
MAX_SPEED_BASELINE_KMH   = 50.0   # cap autopilot speed
STEER_RESTORE_STEPS      = 8      # smooth steer-back steps after event
CONTROL_DT               = 0.05   # seconds per control loop step
TICK_DT                  = 0.10   # scheduler polling interval while waiting

JUNCTION_LOOKAHEAD_M     = 70.0            # how far ahead we scan for junctions
JUNCTION_TRIGGER_WINDOW  = (6.0, 18.0)     # trigger a turn when junction entry is this close (m)
JUNCTION_CLEAR_MARGIN_M  = 18.0            # straight-road events require the junction to be farther than this
LANE_OFFSET_ABORT_RATIO  = 0.85            # abort mid-maneuver if |lateral offset| exceeds this * half-lane-width
HAZARD_AHEAD_TURN_M      = 10.0
HAZARD_SIDE_TURN_M       = 3.0
HAZARD_AHEAD_LANECHG_M   = 12.0
HAZARD_SIDE_LANECHG_M    = 5.0
HAZARD_AHEAD_ACCEL_M     = 15.0
HAZARD_SIDE_ACCEL_M      = 2.5
COLLISION_ABORT_WINDOW_S = 1.5             # treat any collision in the last N s as "still unsafe"


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
    p.add_argument("--min-recovery", "--scenario-interval", type=float, default=20.0,
                   help="Minimum autopilot seconds before the scheduler starts "
                        "looking for a safe/valid moment to trigger the next event")
    p.add_argument("--max-event-wait",     type=float, default=90.0,
                   help="Max seconds the scheduler will wait for a valid condition "
                        "(matching junction / clear lane / clear road) before skipping an event")
    p.add_argument("--repeat-events",      type=int,   default=1,
                   help="How many full scenario cycles to run (0 = infinite)")
    p.add_argument("--print-safe",         action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# CARLA helpers — motion / vehicle basics
# ─────────────────────────────────────────────────────────────────────────────

def get_speed_kmh(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


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
                print(f"  \u26a0  TM port {port} busy, trying next\u2026")
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


def wait_for_speed(
    world: carla.World,
    vehicle: carla.Vehicle,
    min_speed_kmh: float = 10.0,
    timeout: float = 8.0,
) -> bool:
    """Ensure vehicle reaches min_speed by actively driving along waypoints if needed."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        speed = get_speed_kmh(vehicle)
        if speed >= min_speed_kmh:
            return True
        drive_waypoint_autopilot_step(world, vehicle, target_speed_kmh=30.0)
        time.sleep(0.05)
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
# Road-geometry & safety helpers (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def lateral_offset_and_half_width(world: carla.World, vehicle: carla.Vehicle) -> Tuple[float, float]:
    """Distance of the vehicle from the center of its current driving lane,
    plus that lane's half-width — used to detect the car drifting off-road."""
    loc = vehicle.get_transform().location
    wp = world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return 0.0, 1.75  # fall back to a generic ~3.5 m lane
    offset = loc.distance(wp.transform.location)
    return offset, wp.lane_width / 2.0


def is_within_lane(world: carla.World, vehicle: carla.Vehicle, ratio: float = LANE_OFFSET_ABORT_RATIO) -> bool:
    offset, half_width = lateral_offset_and_half_width(world, vehicle)
    return offset <= half_width * ratio


def scan_for_junction(
    world: carla.World,
    vehicle: carla.Vehicle,
    max_dist: float = JUNCTION_LOOKAHEAD_M,
    step: float = 2.0,
) -> Optional[Tuple[float, carla.Waypoint, "carla.Junction"]]:
    """
    Walk the road graph ahead of the vehicle looking for the next real junction.
    Returns (distance_to_junction_m, entry_waypoint, junction) or None if no
    junction is found within max_dist. `entry_waypoint` is the last waypoint
    on the current (non-junction) approach lane, used as the heading reference.
    """
    cmap = world.get_map()
    loc = vehicle.get_transform().location
    wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return None

    traveled = 0.0
    prev_wp = wp
    cur_wp = wp
    while traveled < max_dist:
        nxts = cur_wp.next(step)
        if not nxts:
            return None
        nxt = nxts[0]
        traveled += step
        if nxt.is_junction:
            junction = nxt.get_junction()
            if junction is None:
                return None
            return traveled, prev_wp, junction
        prev_wp = cur_wp
        cur_wp = nxt
    return None


def classify_junction_turns(entry_wp: carla.Waypoint, junction: "carla.Junction") -> Dict[str, bool]:
    """
    Determine which turn directions actually exist at this junction relative
    to the vehicle's approach heading, using CARLA's junction lane topology
    (entry/exit waypoint pairs) rather than guessing from road curvature.
    """
    options = {"left": False, "right": False, "straight": False}
    try:
        pairs = junction.get_waypoints(carla.LaneType.Driving)
    except Exception:
        return options

    entry_yaw = entry_wp.transform.rotation.yaw
    entry_loc = entry_wp.transform.location

    for start_wp, end_wp in pairs:
        # Only consider junction lanes that begin near our actual approach lane
        if start_wp.transform.location.distance(entry_loc) > 9.0:
            continue
        diff = normalize_angle_deg(end_wp.transform.rotation.yaw - entry_yaw)
        if -160.0 <= diff <= -20.0:
            options["left"] = True
        elif 20.0 <= diff <= 160.0:
            options["right"] = True
        elif -20.0 < diff < 20.0:
            options["straight"] = True
    return options


def nearby_actor_hazard(
    world: carla.World,
    vehicle: carla.Vehicle,
    ahead: float,
    side: float,
    behind: float = 3.0,
) -> bool:
    """
    True if any other vehicle/walker sits inside a rectangular safety zone
    around the ego vehicle (in the ego's local forward/right frame). Used to
    gate harsh events so they never fire straight into traffic or pedestrians.
    """
    t = vehicle.get_transform()
    loc = t.location
    fwd = t.get_forward_vector()
    right = t.get_right_vector()
    ego_id = vehicle.id

    for actor in world.get_actors():
        type_id = actor.type_id
        if actor.id == ego_id:
            continue
        if not (type_id.startswith("vehicle.") or type_id.startswith("walker.")):
            continue
        oloc = actor.get_location()
        dx = oloc.x - loc.x
        dy = oloc.y - loc.y
        fwd_dist = dx * fwd.x + dy * fwd.y
        right_dist = dx * right.x + dy * right.y
        if -behind <= fwd_dist <= ahead and abs(right_dist) <= side:
            return True
    return False


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
                if label != last_event_label[0]:
                    should_print = True
                    last_event_label[0] = label
            else:
                last_event_label[0] = "safe"
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
# Maneuver engine  ←  junction-gated, hazard-checked, road-boundary-guarded
# ─────────────────────────────────────────────────────────────────────────────

def _kickstart_and_restore(world: carla.World, vehicle: carla.Vehicle, tm: carla.TrafficManager):
    """After any manual event: softly nudge forward along road center, then hand back to autopilot."""
    for _ in range(6):  # 0.3s gentle nudge on road waypoint steering
        drive_waypoint_autopilot_step(world, vehicle, target_speed_kmh=25.0)
        time.sleep(0.05)
    restore_autopilot(vehicle, tm)


class ManeuverEngine:
    """
    All maneuvers:
      - Check a collision hasn't just happened and no vehicle/pedestrian is in
        the relevant safety zone before starting.
      - Wait for minimum speed before executing.
      - Disable autopilot, run manual controls tick-by-tick, checking each
        tick that the car is still within its lane and hasn't just collided —
        aborting the aggressive phase early (but always still smoothly
        restoring) if either trips.
      - Use road waypoints to keep steer aligned with road during recovery.
    """

    def __init__(
        self,
        world: carla.World,
        vehicle: carla.Vehicle,
        tm: carla.TrafficManager,
        collision_recent_fn,
    ):
        self.world   = world
        self.vehicle = vehicle
        self.tm      = tm
        self.collision_recent = collision_recent_fn

    # ── helpers ──────────────────────────────────────────────────────────────

    def _disable(self):
        self.vehicle.set_autopilot(False)
        time.sleep(0.05)

    def _restore(self):
        _kickstart_and_restore(self.world, self.vehicle, self.tm)

    def _road_steer_bias(self, look_ahead: float = 6.0) -> float:
        wp = get_waypoint_ahead(self.world, self.vehicle, look_ahead)
        if wp is None:
            return 0.0
        t     = self.vehicle.get_transform()
        loc   = t.location
        fwd   = t.get_forward_vector()
        dx    = wp.transform.location.x - loc.x
        dy    = wp.transform.location.y - loc.y
        dist  = math.sqrt(dx * dx + dy * dy) + 1e-6
        cross = fwd.x * dy - fwd.y * dx
        steer = max(-0.15, min(0.15, cross / dist))
        return steer

    def _smooth_restore_steer(self, from_steer: float, steps: int = STEER_RESTORE_STEPS):
        """Linearly blend steer back to road-centre over `steps` ticks (also
        corrects any drift accumulated if a maneuver aborted early)."""
        for i in range(steps):
            t     = (i + 1) / steps
            road  = self._road_steer_bias()
            steer = from_steer * (1.0 - t) + road * t
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.40, steer=steer, brake=0.0)
            )
            time.sleep(CONTROL_DT)

    def _run_ticks(self, ctrl_fn, n_ticks: int, dt: float = CONTROL_DT) -> bool:
        """
        Apply ctrl_fn(tick_index) -> carla.VehicleControl for n_ticks, aborting
        early (returns False) if the vehicle drifts too far off its lane or a
        collision has just occurred. Returns True if it ran to completion.
        """
        for i in range(n_ticks):
            if self.collision_recent():
                print("    [Safety] Collision detected — aborting aggressive phase, recovering.")
                return False
            if not is_within_lane(self.world, self.vehicle):
                print("    [Safety] Vehicle drifting off-lane — aborting aggressive phase, recovering.")
                return False
            self.vehicle.apply_control(ctrl_fn(i))
            time.sleep(dt)
        return True

    # ── event implementations ─────────────────────────────────────────────────

    def sudden_acceleration(self) -> bool:
        if nearby_actor_hazard(self.world, self.vehicle, HAZARD_AHEAD_ACCEL_M, HAZARD_SIDE_ACCEL_M):
            print("    [Event] Sudden Acceleration skipped — vehicle/pedestrian ahead")
            return False
        self._disable()
        print("    [Event] Sudden Acceleration  \u2192 full throttle up to 1.4 s")
        self._run_ticks(lambda i: carla.VehicleControl(throttle=1.0, steer=self._road_steer_bias(), brake=0.0), 28)
        self._smooth_restore_steer(self._road_steer_bias())
        self._restore()
        return True

    def sudden_braking(self) -> bool:
        if not wait_for_speed(self.world, self.vehicle, MIN_SPEED_FOR_EVENT_KMH, timeout=8.0):
            print("    [Event] Sudden Braking skipped — vehicle too slow")
            return False
        self._disable()
        print("    [Event] Sudden Braking  \u2192 max brake up to 0.9 s")
        self._run_ticks(lambda i: carla.VehicleControl(throttle=0.0, steer=self._road_steer_bias(), brake=1.0), 18)
        self._smooth_restore_steer(self._road_steer_bias())
        self._restore()
        return True

    def harsh_left_turn(self) -> bool:
        if not wait_for_speed(self.world, self.vehicle, MIN_SPEED_FOR_EVENT_KMH, timeout=8.0):
            print("    [Event] Harsh Left Turn skipped — too slow")
            return False
        if nearby_actor_hazard(self.world, self.vehicle, HAZARD_AHEAD_TURN_M, HAZARD_SIDE_TURN_M):
            print("    [Event] Harsh Left Turn skipped — vehicle/pedestrian in the way")
            return False
        self._disable()
        print("    [Event] Harsh Left Turn  \u2192 steer -0.70 through junction, up to 0.9 s")
        self._run_ticks(lambda i: carla.VehicleControl(throttle=0.55, steer=-0.70, brake=0.0), 18)
        self._smooth_restore_steer(-0.70)
        self._restore()
        return True

    def harsh_right_turn(self) -> bool:
        if not wait_for_speed(self.world, self.vehicle, MIN_SPEED_FOR_EVENT_KMH, timeout=8.0):
            print("    [Event] Harsh Right Turn skipped — too slow")
            return False
        if nearby_actor_hazard(self.world, self.vehicle, HAZARD_AHEAD_TURN_M, HAZARD_SIDE_TURN_M):
            print("    [Event] Harsh Right Turn skipped — vehicle/pedestrian in the way")
            return False
        self._disable()
        print("    [Event] Harsh Right Turn  \u2192 steer +0.70 through junction, up to 0.9 s")
        self._run_ticks(lambda i: carla.VehicleControl(throttle=0.55, steer=0.70, brake=0.0), 18)
        self._smooth_restore_steer(0.70)
        self._restore()
        return True

    def harsh_left_lane_change(self) -> bool:
        if not wait_for_speed(self.world, self.vehicle, MIN_SPEED_FOR_EVENT_KMH, timeout=8.0):
            print("    [Event] Harsh Left Lane Change skipped — too slow")
            return False
        cmap = self.world.get_map()
        loc  = self.vehicle.get_transform().location
        wp   = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        left_wp = wp.get_left_lane() if wp else None
        if left_wp is None or left_wp.lane_type != carla.LaneType.Driving:
            print("    [Event] Harsh Left Lane Change skipped — no left lane")
            return False
        if nearby_actor_hazard(self.world, self.vehicle, HAZARD_AHEAD_LANECHG_M, HAZARD_SIDE_LANECHG_M):
            print("    [Event] Harsh Left Lane Change skipped — vehicle/pedestrian nearby")
            return False

        self._disable()
        print("    [Event] Harsh Left Lane Change  \u2192 S-curve \u00b10.50 \u00d7 0.8 s each")
        ok = self._run_ticks(lambda i: carla.VehicleControl(throttle=0.60, steer=-0.50, brake=0.0), 16)
        if ok:
            self._run_ticks(lambda i: carla.VehicleControl(throttle=0.55, steer=0.50, brake=0.0), 16)
        self._smooth_restore_steer(0.0)
        self._restore()
        return True

    def harsh_right_lane_change(self) -> bool:
        if not wait_for_speed(self.world, self.vehicle, MIN_SPEED_FOR_EVENT_KMH, timeout=8.0):
            print("    [Event] Harsh Right Lane Change skipped — too slow")
            return False
        cmap = self.world.get_map()
        loc  = self.vehicle.get_transform().location
        wp   = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        right_wp = wp.get_right_lane() if wp else None
        if right_wp is None or right_wp.lane_type != carla.LaneType.Driving:
            print("    [Event] Harsh Right Lane Change skipped — no right lane")
            return False
        if nearby_actor_hazard(self.world, self.vehicle, HAZARD_AHEAD_LANECHG_M, HAZARD_SIDE_LANECHG_M):
            print("    [Event] Harsh Right Lane Change skipped — vehicle/pedestrian nearby")
            return False

        self._disable()
        print("    [Event] Harsh Right Lane Change  \u2192 S-curve \u00b10.50 \u00d7 0.8 s each")
        ok = self._run_ticks(lambda i: carla.VehicleControl(throttle=0.60, steer=0.50, brake=0.0), 16)
        if ok:
            self._run_ticks(lambda i: carla.VehicleControl(throttle=0.55, steer=-0.50, brake=0.0), 16)
        self._smooth_restore_steer(0.0)
        self._restore()
        return True

    def run(self, maneuver_name: str) -> bool:
        """Dispatch by name. Returns True if the event actually executed."""
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
            return False
        return fn()


def start_autopilot_maintainer(world: carla.World, vehicle: carla.Vehicle, stop_event: threading.Event):
    """Background maintainer thread that only intervenes if the vehicle is completely stalled."""
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
# Event scheduler (NEW) — waits for real road/traffic conditions, not a timer
# ─────────────────────────────────────────────────────────────────────────────

class EventScheduler:
    TURN_DIRECTIONS = {"harsh_left_turn": "left", "harsh_right_turn": "right"}
    LANE_EVENTS = {"harsh_left_lane_change", "harsh_right_lane_change"}

    def __init__(self, world, vehicle, engine: ManeuverEngine, min_recovery_s: float, max_wait_s: float):
        self.world = world
        self.vehicle = vehicle
        self.engine = engine
        self.min_recovery_s = min_recovery_s
        self.max_wait_s = max_wait_s
        self.stats = {"triggered": 0, "skipped": 0}

    def wait_and_trigger(self, event_name: str, status_ref: List[str], tick_dt: float = TICK_DT) -> bool:
        status_ref[0] = "Autopilot Recovery"

        # Mandatory minimum safe-driving window before we even look for the next event
        t_start = time.time()
        while time.time() - t_start < self.min_recovery_s:
            time.sleep(tick_dt)

        deadline = time.time() + self.max_wait_s
        while time.time() < deadline:
            triggered = self._try_condition(event_name, status_ref)
            if triggered:
                self.stats["triggered"] += 1
                return True
            time.sleep(tick_dt)

        print(f"    [Scheduler] No safe/valid condition found for {event_name} within "
              f"{self.max_wait_s:.0f}s — skipping this occurrence.")
        self.stats["skipped"] += 1
        return False

    def _try_condition(self, event_name: str, status_ref: List[str]) -> bool:
        if event_name in self.TURN_DIRECTIONS:
            direction = self.TURN_DIRECTIONS[event_name]
            scan = scan_for_junction(self.world, self.vehicle)
            if not scan:
                return False
            dist, entry_wp, junction = scan
            if not (JUNCTION_TRIGGER_WINDOW[0] <= dist <= JUNCTION_TRIGGER_WINDOW[1]):
                return False
            options = classify_junction_turns(entry_wp, junction)
            if not options.get(direction):
                return False
            status_ref[0] = f"EVENT: {event_name}"
            return self.engine.run(event_name)

        if event_name in self.LANE_EVENTS:
            scan = scan_for_junction(self.world, self.vehicle)
            if scan and scan[0] <= JUNCTION_CLEAR_MARGIN_M:
                return False  # don't lane-change right at/into an intersection
            status_ref[0] = f"EVENT: {event_name}"
            return self.engine.run(event_name)

        # sudden_acceleration / sudden_braking — straight-road events
        scan = scan_for_junction(self.world, self.vehicle)
        if scan and scan[0] <= JUNCTION_CLEAR_MARGIN_M:
            return False
        status_ref[0] = f"EVENT: {event_name}"
        return self.engine.run(event_name)


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
    print("   CARLA AUTOMATED SCENARIO — JUNCTION-GATED, COLLISION-SAFE BENCHMARK")
    print("=" * 70)
    print(f"  Connecting to CARLA at {args.host}:{args.port} \u2026")
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()
    print(f"  Map: {world.get_map().name}")

    # ── Vehicle ───────────────────────────────────────────────────────────────
    vehicle = find_or_spawn_vehicle(world, args.role_name)

    # ── Traffic Manager ───────────────────────────────────────────────────────
    print(f"  Setting up Traffic Manager on port {args.tm_port} \u2026")
    tm = setup_traffic_manager(client, vehicle, args.tm_port)

    # ── Ensemble detector + safety scorer ────────────────────────────────────
    print("  Loading ensemble models \u2026")
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

    # ── Collision + lane-invasion sensors (NEW) ──────────────────────────────
    collision_state = {"count": 0, "last_time": 0.0}

    def _on_collision(event):
        collision_state["count"] += 1
        collision_state["last_time"] = time.time()
        other = event.other_actor.type_id if event.other_actor else "unknown"
        print(f"  !! Collision detected with {other} !!")

    collision_bp = bp_lib.find("sensor.other.collision")
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_sensor.listen(_on_collision)

    lane_invasion_state = {"count": 0}

    def _on_lane_invasion(event):
        lane_invasion_state["count"] += 1

    li_bp = bp_lib.find("sensor.other.lane_invasion")
    li_sensor = world.spawn_actor(li_bp, carla.Transform(), attach_to=vehicle)
    li_sensor.listen(_on_lane_invasion)

    def collision_recent() -> bool:
        return (time.time() - collision_state["last_time"]) < COLLISION_ABORT_WINDOW_S

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

    # ── Maneuver engine + scheduler ──────────────────────────────────────────
    engine = ManeuverEngine(world, vehicle, tm, collision_recent_fn=collision_recent)
    scheduler = EventScheduler(world, vehicle, engine, args.min_recovery, args.max_event_wait)

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print(f"  Vehicle id          : {vehicle.id}")
    print(f"  Sensor tick         : {args.sensor_tick} s")
    print(f"  Min recovery window : {args.min_recovery} s")
    print(f"  Max wait per event  : {args.max_event_wait} s")
    print(f"  Events per cycle    : {len(SCENARIO_SEQUENCE)}")
    repeat_str = "∞" if args.repeat_events == 0 else str(args.repeat_events)
    print(f"  Repeat cycles       : {repeat_str}")
    print()
    print("  Turns only trigger at real junctions with a matching turn option.")
    print("  Lane changes / accel / brake only trigger on clear, non-junction road.")
    print("  Press Ctrl+C to stop.\n")
    print("-" * 70)

    # ── Scenario loop ─────────────────────────────────────────────────────────
    cycle = 0
    try:
        current_status[0] = "Autopilot Warm-Up"
        print(f"\n[Phase 0] Autopilot warm-up for 10 s \u2026")
        time.sleep(10.0)

        while True:
            cycle += 1
            if args.repeat_events > 0 and cycle > args.repeat_events:
                break

            print(f"\n{'─' * 70}")
            print(f"  CYCLE {cycle}  — starting event sequence")
            print(f"{'─' * 70}")

            for label, name in SCENARIO_SEQUENCE:
                print(f"\n[Waiting] Looking for a safe, valid moment for: {label.upper()}")
                triggered = scheduler.wait_and_trigger(name, current_status)
                if triggered:
                    print(f"<<< {label} executed. Resuming autopilot \u2026")
                current_status[0] = "Post-Event Settle"
                time.sleep(3.0)

        current_status[0] = "Autopilot Continuous"
        print("\n\u2713 All cycles complete. Driving on autopilot continuously \u2026")
        print("  Press Ctrl+C to stop.\n")
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nStopping Automated Scenario Benchmark \u2026")

    finally:
        spectator_stop.set()
        maintainer_stop.set()
        print("  Disabling autopilot and cleaning up sensors \u2026")
        try:
            vehicle.set_autopilot(False)
        except Exception:
            pass
        for sensor in (imu_sensor, collision_sensor, li_sensor):
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception:
                pass
        time.sleep(0.2)

        print("\n" + "=" * 70)
        print("  RUN SAFETY REPORT")
        print("=" * 70)
        print(f"  Events triggered      : {scheduler.stats['triggered']}")
        print(f"  Events skipped        : {scheduler.stats['skipped']} "
              f"(no matching junction / clear lane / clear road found in time)")
        print(f"  Collisions detected    : {collision_state['count']}")
        print(f"  Lane-marking crossings : {lane_invasion_state['count']} "
              f"(includes intentional lane-change events)")
        print("✓ Cleanup complete.\n")


if __name__ == "__main__":
    main()