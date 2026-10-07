#!/usr/bin/env python3
"""
Spawn NPC Traffic for CARLA Simulation.
Spawns background vehicles controlled by CARLA's Traffic Manager (autopilot)
without interfering with the manual ego vehicle (role_name='hero').
"""

import argparse
import glob
import logging
import os
import random
import sys
import time

# Auto-detect and add CARLA egg if not already installed
try:
    import carla
except ImportError:
    carla_dist = "/home/nikhil/Downloads/CARLA_0.9.13/PythonAPI/carla/dist"
    if os.path.exists(carla_dist):
        eggs = glob.glob(os.path.join(carla_dist, f"carla-*{sys.version_info.major}.{sys.version_info.minor}-*.egg"))
        if eggs:
            sys.path.append(eggs[0])
    import carla


def parse_args():
    parser = argparse.ArgumentParser(description="Spawn NPC Traffic in CARLA")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host IP (default: 127.0.0.1)")
    parser.add_argument("-p", "--port", type=int, default=2000, help="CARLA TCP port (default: 2000)")
    parser.add_argument("-n", "--number-of-vehicles", type=int, default=30, help="Number of vehicles to spawn (default: 30)")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port (default: 8000)")
    parser.add_argument("--safe", action="store_true", default=True, help="Avoid 2-wheelers and fragile vehicles (default: True)")
    parser.add_argument("--speed-diff", type=float, default=20.0, help="Percentage speed difference under limit (default: 20%%)")
    parser.add_argument("--distance-to-leading", type=float, default=3.0, help="Distance to leading vehicle in meters (default: 3.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for Traffic Manager")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
    except Exception as e:
        print(f"❌ Failed to connect to CARLA server on {args.host}:{args.port}: {e}")
        return 1

    # Traffic Manager Setup
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(args.distance_to_leading)
    traffic_manager.global_percentage_speed_difference(args.speed_diff)

    if args.seed is not None:
        traffic_manager.set_random_device_seed(args.seed)

    # Note: Keep asynchronous mode so it does not conflict with manual_control.py
    traffic_manager.set_synchronous_mode(False)

    blueprints = world.get_blueprint_library().filter("vehicle.*")
    if args.safe:
        # Filter for 4-wheel vehicles only (avoid bikes/scooters that tip over)
        blueprints = [bp for bp in blueprints if int(bp.get_attribute("number_of_wheels")) == 4]

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    num_to_spawn = min(args.number_of_vehicles, len(spawn_points))
    if num_to_spawn == 0:
        print("⚠️ No spawn points available on this map.")
        return 1

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    for i in range(num_to_spawn):
        transform = spawn_points[i]
        blueprint = random.choice(blueprints)
        
        # Crucial: Ensure role_name is 'autopilot' so it never clashes with 'hero'
        blueprint.set_attribute("role_name", "autopilot")
        
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
            blueprint.set_attribute("driver_id", driver_id)

        batch.append(
            SpawnActor(blueprint, transform).then(
                SetAutopilot(FutureActor, True, traffic_manager.get_port())
            )
        )

    responses = client.apply_batch_sync(batch, False)
    vehicle_ids = [res.actor_id for res in responses if not res.error]

    print(f"✓ Successfully spawned {len(vehicle_ids)}/{num_to_spawn} NPC vehicles under Traffic Manager.")
    print("  All vehicles running on autopilot.")
    print("  Press Ctrl+C to stop traffic and clean up actors.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping traffic...")
    finally:
        print(f"Cleaning up {len(vehicle_ids)} vehicles...")
        client.apply_batch([carla.command.DestroyActor(v_id) for v_id in vehicle_ids])
        print("✓ Cleanup complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
