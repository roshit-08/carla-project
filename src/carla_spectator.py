from __future__ import annotations

import argparse
import sys
import time

import carla


def main() -> int:
    parser = argparse.ArgumentParser(description="CARLA spectator camera to follow the hero vehicle")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--timeout", type=float, default=10.0, help="CARLA client timeout")
    parser.add_argument("--role-name", default="hero", help="Vehicle role_name to follow")
    parser.add_argument("--camera-height", type=float, default=10.0, help="Camera height above vehicle")
    parser.add_argument("--camera-distance", type=float, default=20.0, help="Camera distance behind vehicle")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    spectator = world.get_spectator()
    print("Spectator camera active. Following hero vehicle. Press Ctrl+C to stop.")

    try:
        while True:
            vehicles = world.get_actors().filter("vehicle.*")
            hero = None
            for v in vehicles:
                if v.attributes.get("role_name", "") == args.role_name:
                    hero = v
                    break

            if hero is not None:
                transform = hero.get_transform()
                location = transform.location
                rotation = transform.rotation

                # Position camera behind and above the vehicle
                spectator_transform = carla.Transform(
                    location=carla.Location(
                        x=location.x - args.camera_distance * transform.get_forward_vector().x,
                        y=location.y - args.camera_distance * transform.get_forward_vector().y,
                        z=location.z + args.camera_height,
                    ),
                    rotation=carla.Rotation(
                        pitch=-15.0,  # Look down slightly
                        yaw=rotation.yaw,
                        roll=0.0,
                    ),
                )
                spectator.set_transform(spectator_transform)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Spectator stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())