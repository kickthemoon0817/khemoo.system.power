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
import carb
from isaacsim.simulation_app import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser(description="Battery-aware OpenBot ROS2 example")
    parser.add_argument("--headless", action="store_true", help="Run SimulationApp headless")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many simulation steps")
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=None,
        help="If set, save battery, camera, and IMU samples to this directory (headless recommended)",
    )
    parser.add_argument(
        "--capture-frames",
        type=int,
        default=100,
        help="Maximum number of frames to capture when --capture-dir is set (<=0 for unlimited)",
    )
    parser.add_argument(
        "--capture-interval",
        type=int,
        default=5,
        help="Capture every Nth simulation step (only when --capture-dir is set)",
    )
    parser.add_argument(
        "--map-usd",
        type=Path,
        default=Path("NVIDIA/Samples/Showcases/2023_2_1/IsaacWarehouse/IsaacWarehouse.usd"),
        help="USD environment to reference into the stage",
    )
    parser.add_argument(
        "--sensors-root",
        type=str,
        default=None,
        help="Existing sensors prim root that contains camera/imu prims (no new prims will be created)",
    )
    parser.add_argument(
        "--camera-prim",
        type=str,
        default=None,
        help="Existing camera prim path to wrap (default: <sensors_root>/front_camera)",
    )
    parser.add_argument(
        "--imu-prim",
        type=str,
        default=None,
        help="Existing IMU prim path to wrap (default: <sensors_root>/imu)",
    )
    parser.add_argument(
        "--battery-capacity-ah",
        type=float,
        default=None,
        help="Override battery capacity (Ah) for this run to speed up depletion testing",
    )
    parser.add_argument(
        "--battery-model",
        type=str,
        default=None,
        choices=["thevenin", "rint", "shepherd", "nernst"],
        help="Override battery model type (thevenin/rint/shepherd/nernst)",
    )
    parser.add_argument(
        "--extra-load-w",
        type=float,
        default=0.0,
        help="Optional additional constant load in watts to accelerate discharge",
    )
    parser.add_argument(
        "--stop-on-empty",
        action="store_true",
        help="Stop simulation when battery state of charge reaches zero",
    )
    parser.add_argument(
        "--auto-circle",
        action="store_true",
        help="Drive the robot in a constant circular motion (disables keyboard control)",
    )
    parser.add_argument(
        "--auto-linear",
        type=float,
        default=0.5,
        help="Linear velocity (m/s) when --auto-circle is enabled",
    )
    parser.add_argument(
        "--auto-angular",
        type=float,
        default=0.5,
        help="Angular velocity (rad/s) when --auto-circle is enabled",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    simulation_app = SimulationApp({"headless": args.headless})
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.ros2.bridge")
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
    openbot_usd = asset_root / "sources" / "openbot_with_smartphone.usd"
    battery_cfg = Path(__file__).resolve().parents[1] / "configs" / "default_battery.json"
    map_usd = Path(args.map_usd).expanduser().resolve()

    def load_openbot(world: World, prim_path: str = "/World/OpenBot") -> str:
        """Reference the OpenBot USD into the stage and register it with the scene."""
        if not openbot_usd.exists():
            raise FileNotFoundError(f"OpenBot USD not found at {openbot_usd}")
        add_reference_to_stage(str(openbot_usd), prim_path)
        return prim_path

    def load_environment(map_usd: Path, prim_path: str = "/World/Env") -> None:
        """Reference an environment USD. Prefers local paths; falls back to Nucleus via get_server_path."""
        map_str = str(map_usd)
        # Direct omniverse path is respected.
        if map_str.startswith("omniverse:"):
            add_reference_to_stage(map_str, prim_path)
            return
        # Local path check.
        if map_usd.exists():
            add_reference_to_stage(map_str, prim_path)
            return
        # Try resolving through nucleus get_server_path.
        try:
            from isaacsim.storage.native.nucleus import get_server_path

            server_root = get_server_path()
            if server_root:
                server_root = server_root.rstrip("/")
                rel = map_usd.as_posix().lstrip("/")
                server_url = f"{server_root}/{rel}"
                add_reference_to_stage(server_url, prim_path)
                return
        except Exception:
            pass
        carb.log_warn(f"[run_battery_ros2] Map USD not found locally or on Nucleus: {map_usd}")

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
        """
        Wrap existing camera/IMU prims on the robot with battery-aware adapters.

        Sensors are expected to already exist in the USD; no new prims are created or moved.
        """
        sensors_root = args.sensors_root or f"{robot_prim}/Chassis/galaxy_s20_ultra_cosmic_gray/Sensors"
        camera_path = args.camera_prim or f"{sensors_root}/Camera"
        imu_path = args.imu_prim or f"{sensors_root}/Imu_Sensor"

        stage = get_current_stage()
        if not stage.GetPrimAtPath(camera_path):
            carb.log_error(f"[run_battery_ros2] Camera prim not found at {camera_path}")
        if not stage.GetPrimAtPath(imu_path):
            carb.log_error(f"[run_battery_ros2] IMU prim not found at {imu_path}")

        cam_obj = Camera(camera_path, name="openbot_camera")
        try:
            cam_obj.initialize(attach_rgb_annotator=True)
        except Exception as exc:
            carb.log_warn(f"[run_battery_ros2] Failed to initialize/attach RGB annotator for camera {camera_path}: {exc}")

        camera = BatteryCamera(cam_obj, battery, name="openbot_camera")

        imu = BatteryIMU(
            IMUSensor(imu_path, name="openbot_imu"),
            battery,
            name="openbot_imu",
        )
        return camera, imu

    world = World(stage_units_in_meters=1.0)

    world.scene.add_default_ground_plane()
    robot_prim = load_openbot(world)
    robot = world.scene.add(Robot(robot_prim, name="openbot"))
    load_environment(map_usd)

    battery_config = None
    if args.battery_capacity_ah is not None:
        try:
            import json

            with battery_cfg.open("r", encoding="utf-8") as f:
                base_cfg = json.load(f)
            base_cfg["capacity_ah"] = float(args.battery_capacity_ah)
            if args.battery_model:
                base_cfg.setdefault("model", {})
                if isinstance(base_cfg["model"], dict):
                    base_cfg["model"]["type"] = args.battery_model
                else:
                    base_cfg["model"] = {"type": args.battery_model}
            battery_config = base_cfg
        except Exception:
            battery_config = {"capacity_ah": float(args.battery_capacity_ah)}
            if args.battery_model:
                battery_config["model"] = {"type": args.battery_model}
    elif args.battery_model:
        battery_config = {"model": {"type": args.battery_model}}

    battery = BatteryPrim(
        world,
        name="openbot_battery",
        config_path=None if battery_config else str(battery_cfg),
        config=battery_config,
    )
    cam, imu = attach_battery_sensors(robot_prim, battery)
    if args.extra_load_w > 0.0:
        battery.register_load("extra_cli_load", float(args.extra_load_w))

    capture_dir = None
    capture_records = []
    if args.capture_dir:
        capture_dir = Path(args.capture_dir).expanduser().resolve()
        capture_dir.mkdir(parents=True, exist_ok=True)
        (capture_dir / "camera_raw").mkdir(exist_ok=True)
        (capture_dir / "camera_battery").mkdir(exist_ok=True)
        (capture_dir / "imu_raw").mkdir(exist_ok=True)
        (capture_dir / "imu_battery").mkdir(exist_ok=True)

    bridge = BatteryROS2Bridge(
        battery=battery,
        sensors={"camera": cam, "imu": imu},
        node_name="battery_sim_node",
        qos_profile="sensor_data",
    )

    # Keyboard controller to drive the robot (disabled if auto-circle is on)
    kb = None if args.auto_circle else KeyboardController()
    if kb:
        kb.start()

    world.reset()
    wheel_radius, track_width, _ = compute_wheel_params(robot_prim)
    wheel_indices = {}
    steps = 0
    try:
        wheel_indices = resolve_wheel_joint_indices(robot._articulation_view)

        running = True
        render_this_run = (not args.headless) or bool(capture_dir)
        while simulation_app.is_running() and running and (args.max_steps is None or steps < args.max_steps):
            world.step(render=render_this_run)
            if kb:
                cmd = kb.get_scaled_command()
            else:
                # Auto-circle command
                cmd = np.array([args.auto_linear, 0.0, args.auto_angular], dtype=np.float32)

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

            cap_limit = args.capture_frames
            unlimited_cap = cap_limit is None or cap_limit <= 0
            if capture_dir and steps % max(1, args.capture_interval) == 0 and (unlimited_cap or len(capture_records) < cap_limit):
                state = battery.get_state()
                raw_cam_frame = getattr(cam, "_last_raw_frame", None)
                raw_imu_frame = getattr(imu, "_last_raw_frame", None)

                batt_cam_frame = cam._last_frame
                batt_imu_frame = imu._last_frame

                def save_arrays(frame, subdir: str, label_prefix: str):
                    saved = {}
                    if frame is None:
                        return saved
                    for k, v in frame.items():
                        if isinstance(v, np.ndarray):
                            out_path = capture_dir / subdir / f"{label_prefix}_{k}_{steps:06d}.npy"
                            np.save(out_path, v)
                            saved[k] = str(out_path)
                    return saved

                # Try to grab RGB(A) directly from the camera annotator for raw and battery-aware views.
                def save_camera_rgba(camera_obj, subdir: str, label_prefix: str):
                    try:
                        rgba = camera_obj.get_rgba(device="cpu")
                    except Exception:
                        rgba = None
                    if rgba is None or not isinstance(rgba, np.ndarray):
                        return {}
                    out_path = capture_dir / subdir / f"{label_prefix}_rgba_{steps:06d}.npy"
                    np.save(out_path, rgba)
                    return {"rgba": str(out_path)}

                raw_cam_files = save_arrays(raw_cam_frame, "camera_raw", "cam")
                raw_cam_files.update(save_camera_rgba(cam.camera, "camera_raw", "cam"))
                batt_cam_files = save_arrays(batt_cam_frame, "camera_battery", "cam_batt")
                batt_cam_files.update(save_camera_rgba(cam.camera, "camera_battery", "cam_batt"))
                raw_imu_files = save_arrays(raw_imu_frame, "imu_raw", "imu")
                batt_imu_files = save_arrays(batt_imu_frame, "imu_battery", "imu_batt")

                capture_records.append(
                    {
                        "step": steps,
                        "battery_state": {
                            "soc": state.get("soc"),
                            "terminal_v": state.get("terminal_v"),
                            "current_a": state.get("current_a"),
                        },
                        "camera_raw_keys": sorted(raw_cam_frame.keys()) if raw_cam_frame else None,
                        "camera_battery_keys": sorted(batt_cam_frame.keys()) if batt_cam_frame else None,
                        "imu_raw_keys": sorted(raw_imu_frame.keys()) if raw_imu_frame else None,
                        "imu_battery_keys": sorted(batt_imu_frame.keys()) if batt_imu_frame else None,
                        "camera_raw_arrays": raw_cam_files,
                        "camera_battery_arrays": batt_cam_files,
                        "imu_raw_arrays": raw_imu_files,
                        "imu_battery_arrays": batt_imu_files,
                    }
                )
                # If we've hit capture quota and no stopping condition, keep running unless asked otherwise.
            if args.stop_on_empty and battery.get_state().get("soc", 0.0) <= 0.0:
                running = False
            steps += 1
    finally:
        if capture_dir:
            import json

            manifest = {
                "records": capture_records,
                "total_frames": len(capture_records),
                "capture_limit": args.capture_frames,
                "capture_interval": args.capture_interval,
                "battery_capacity_ah": battery.capacity_ah,
                "max_steps": args.max_steps,
                "executed_steps": steps,
                "extra_load_w": args.extra_load_w,
                "stopped_on_empty": bool(args.stop_on_empty and (battery.get_state().get("soc", 0.0) <= 0.0)),
            }
            (capture_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2))
        if kb:
            kb.shutdown()
        bridge.shutdown()
        simulation_app.close()
        World.clear_instance()


if __name__ == "__main__":
    main()
