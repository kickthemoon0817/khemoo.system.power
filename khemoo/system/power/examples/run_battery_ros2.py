"""Example: attach a battery to an OpenBot model and publish to ROS 2."""

from __future__ import annotations

import sys
from pathlib import Path

EXT_ROOT = Path(__file__).resolve().parents[4]
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))
EXTS_USER_ROOT = EXT_ROOT.parent
if str(EXTS_USER_ROOT) not in sys.path:
    sys.path.append(str(EXTS_USER_ROOT))
INPUT_CTRL_ROOT = EXTS_USER_ROOT / "khemoo.robot.input_control"
if str(INPUT_CTRL_ROOT) not in sys.path:
    sys.path.append(str(INPUT_CTRL_ROOT))

import argparse
import numpy as np
from isaacsim.simulation_app import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser(description="Battery-aware OpenBot ROS2 example")
    parser.add_argument("--headless", action="store_true", help="Run SimulationApp headless")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many simulation steps")
    return parser.parse_args()


def main():
    args = parse_args()
    simulation_app = SimulationApp({"headless": args.headless, "enableExts": ["isaacsim.ros2.bridge"]})
    # SimulationApp can override sys.path; re-append our extension roots defensively.
    for p in (EXT_ROOT, INPUT_CTRL_ROOT, EXTS_USER_ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from isaacsim.core.api import World
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.prims import define_prim
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.sensors.camera import Camera
    from isaacsim.sensors.physics import IMUSensor
    from pxr import Usd, UsdGeom, UsdPhysics

    from khemoo.system.power.battery import BatteryPrim
    from khemoo.system.power.sensors import BatteryCamera, BatteryIMU
    from khemoo.system.power.ros2_bridge import BatteryROS2Bridge
    from khemoo.robot.input_control.impl.keyboard_controller import KeyboardController
    from isaacsim.core.utils.types import ArticulationAction

    asset_root = EXT_ROOT
    openbot_usd = asset_root / "sources" / "openbot.usd"
    battery_cfg = Path(__file__).resolve().parents[1] / "configs" / "default_battery.json"

    def load_openbot(world: World, prim_path: str = "/World/OpenBot") -> str:
        """Reference the OpenBot USD into the stage and register it with the scene."""
        if not openbot_usd.exists():
            raise FileNotFoundError(f"OpenBot USD not found at {openbot_usd}")
        add_reference_to_stage(str(openbot_usd), prim_path)
        return prim_path

    def compute_wheel_params(robot_prim: str):
        stage = get_current_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        wheel_names = ["WheelFrontLeft", "WheelFrontRight", "WheelBehindLeft", "WheelBehindRight"]
        centers = {}
        xforms = {}
        radii = {}
        for name in wheel_names:
            prim = stage.GetPrimAtPath(f"{robot_prim}/{name}")
            if not prim or not prim.IsValid():
                continue

            bbox = bbox_cache.ComputeWorldBound(prim).GetBox()
            min_pt = bbox.GetMin()
            max_pt = bbox.GetMax()

            # Fallback if the computed bbox collapses (common when extents hints are missing)
            if (max_pt - min_pt).GetLength() < 1e-6:
                boundable = UsdGeom.Boundable(prim)
                extent = boundable.ComputeExtent(Usd.TimeCode.Default())
                if extent:
                    world_pts = [xform_cache.GetLocalToWorldTransform(prim).Transform(pt) for pt in extent]
                    min_pt = world_pts[0]
                    max_pt = world_pts[1]

            # If still degenerate, fall back to xform translation
            center = (min_pt + max_pt) * 0.5
            translation = xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            xforms[name] = translation
            if (max_pt - min_pt).GetLength() < 1e-6:
                center = translation

            size = max_pt - min_pt
            centers[name] = center
            radii[name] = 0.5 * max(size[0], size[1])
        if not all(k in centers for k in wheel_names):
            return 0.0, 0.0, 0.0
        track_width = float(abs(centers["WheelFrontLeft"][1] - centers["WheelFrontRight"][1]))
        wheel_base = float(abs(centers["WheelFrontLeft"][0] - centers["WheelBehindLeft"][0]))

        # If bbox-based measurements failed, fall back to xform translations
        if track_width < 1e-6 and all(k in xforms for k in ("WheelFrontLeft", "WheelFrontRight")):
            track_width = float(abs(xforms["WheelFrontLeft"][1] - xforms["WheelFrontRight"][1]))
        if wheel_base < 1e-6 and all(k in xforms for k in ("WheelFrontLeft", "WheelBehindLeft")):
            wheel_base = float(abs(xforms["WheelFrontLeft"][0] - xforms["WheelBehindLeft"][0]))

        # If radius is degenerate, use half of wheel thickness as a minimal value
        mean_radius = float(sum(radii.values()) / len(radii))
        if mean_radius < 1e-6 and radii:
            mean_radius = float(max(radii.values()))

        return mean_radius, track_width, wheel_base

    def resolve_wheel_joint_indices(articulation_view) -> dict:
        """
        Map wheel logical names (fl, fr, bl, br) to joint indices.

        The OpenBot USD uses generic joint names (RevoluteJoint, RevoluteJoint_0, ...)
        so we rely on the per-dof paths to disambiguate.
        """
        indices = {}
        # First try metadata if it contains readable names.
        meta_map = getattr(getattr(articulation_view, "_metadata", None), "joint_indices", None)
        if meta_map:
            for name, idx in meta_map.items():
                lower = name.lower()
                if "front" in lower and "left" in lower:
                    indices["fl"] = idx
                elif "front" in lower and "right" in lower:
                    indices["fr"] = idx
                elif ("behind" in lower or "rear" in lower or "back" in lower) and "left" in lower:
                    indices["bl"] = idx
                elif ("behind" in lower or "rear" in lower or "back" in lower) and "right" in lower:
                    indices["br"] = idx
        # Fallback: use dof paths to infer ordering.
        if len(indices) < 4:
            dof_paths = getattr(articulation_view, "_dof_paths", None)
            if dof_paths and len(dof_paths) > 0:
                for idx, path in enumerate(dof_paths[0]):
                    lower = path.lower()
                    if "front" in lower and "left" in lower:
                        indices["fl"] = idx
                    elif "front" in lower and "right" in lower:
                        indices["fr"] = idx
                    elif ("behind" in lower or "rear" in lower or "back" in lower) and "left" in lower:
                        indices["bl"] = idx
                    elif ("behind" in lower or "rear" in lower or "back" in lower) and "right" in lower:
                        indices["br"] = idx
        return indices

    def attach_battery_sensors(robot_prim: str, battery: BatteryPrim) -> tuple[BatteryCamera, BatteryIMU]:
        """Create camera/IMU prims on the robot and wrap them with battery-aware adapters."""
        sensors_root = f"{robot_prim}/sensors"
        define_prim(sensors_root, "Xform")
        stage = get_current_stage()
        sensors_prim = stage.GetPrimAtPath(sensors_root)
        UsdPhysics.RigidBodyAPI.Apply(sensors_prim)
        mass_api = UsdPhysics.MassAPI.Apply(sensors_prim)
        mass_api.CreateMassAttr(1.0)
        mass_api.CreateDiagonalInertiaAttr((0.01, 0.01, 0.01))

        cam_translation = (0.0, 0.09, 0.085)
        cam_orientation = (0.70710678, 0.70710678, 0.0, 0.0)  # rotate +90 deg about X to face +Y
        camera = BatteryCamera(
            Camera(
                f"{sensors_root}/front_camera",
                translation=cam_translation,
                orientation=cam_orientation,
                resolution=(640, 480),
                name="openbot_camera",
            ),
            battery,
            name="openbot_camera",
        )

        imu_translation = (0.0, 0.0, 0.04)
        imu = BatteryIMU(
            IMUSensor(f"{sensors_root}/imu", translation=imu_translation, name="openbot_imu"),
            battery,
            name="openbot_imu",
        )
        return camera, imu

    world = World(stage_units_in_meters=1.0)

    world.scene.add_default_ground_plane()
    robot_prim = load_openbot(world)
    robot = world.scene.add(Robot(robot_prim, name="openbot"))

    battery = BatteryPrim(world, name="openbot_battery", config_path=str(battery_cfg))
    cam, imu = attach_battery_sensors(robot_prim, battery)

    bridge = BatteryROS2Bridge(
        battery=battery,
        sensors={"camera": cam, "imu": imu},
        node_name="battery_sim_node",
        qos_profile="sensor_data",
    )

    # Keyboard controller to drive the robot
    kb = KeyboardController()
    kb.start()

    world.reset()
    wheel_radius, track_width, _ = compute_wheel_params(robot_prim)
    wheel_indices = {}
    steps = 0
    try:
        wheel_indices = resolve_wheel_joint_indices(robot._articulation_view)

        while simulation_app.is_running() and (args.max_steps is None or steps < args.max_steps):
            world.step(render=not args.headless)
            cmd = kb.get_scaled_command()

            if len(wheel_indices) == 4 and wheel_radius > 1e-4 and track_width > 1e-4:
                v = float(cmd[0])  # forward/backward (W/S)
                w = float(cmd[2])  # rotate (Q/E)

                # Remap strafe keys (A/D) to rotation when no explicit rotation is given.
                if abs(w) < 0.01 and abs(float(cmd[1])) > 0.01:
                    w = float(cmd[1]) * 2.0  # boost turning response for A/D

                left_vel = (v - 0.5 * track_width * w) / wheel_radius
                # Right joints are mirrored; flip the sign so forward commands don't spin in place.
                right_vel = -((v + 0.5 * track_width * w) / wheel_radius)
                joint_vels = np.array(
                    [left_vel, right_vel, left_vel, right_vel], dtype=np.float32
                )
                joint_ids = np.array(
                    [wheel_indices["fl"], wheel_indices["fr"], wheel_indices["bl"], wheel_indices["br"]],
                    dtype=np.int32,
                )
                robot.apply_action(ArticulationAction(joint_velocities=joint_vels, joint_indices=joint_ids))
            else:
                # Fallback: directly set base twist
                robot.set_linear_velocity(np.array([cmd[0], cmd[1], 0.0], dtype=np.float32))
                robot.set_angular_velocity(np.array([0.0, 0.0, cmd[2]], dtype=np.float32))

            cam.update()
            imu.update()
            bridge.publish()
            steps += 1
    finally:
        kb.shutdown()
        bridge.shutdown()
        simulation_app.close()
        World.clear_instance()


if __name__ == "__main__":
    main()
