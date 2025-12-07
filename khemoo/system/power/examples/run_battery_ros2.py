"""Example: attach a battery to camera/IMU sensors and publish to ROS 2."""

from pathlib import Path

from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.physics import IMUSensor

from khemoo.system.power.battery import BatteryPrim
from khemoo.system.power.sensors import BatteryCamera, BatteryIMU
from khemoo.system.power.ros2_bridge import BatteryROS2Bridge


def main():
    world = World(stage_units_in_meters=1.0)

    # Load or assume a robot already exists at /World/Robot
    usd_path = Path("usd/mobile_robot.usd")
    if usd_path.exists():
        add_reference_to_stage(str(usd_path), "/World/Robot")
    robot = world.scene.add(Robot("/World/Robot", name="mobile_robot"))

    battery_cfg = Path(__file__).resolve().parents[1] / "configs" / "default_battery.json"
    battery = BatteryPrim(world, name="phone_battery", config_path=str(battery_cfg))
    cam = BatteryCamera(Camera("/World/Robot/camera"), battery, name="phone_cam")
    imu = BatteryIMU(IMUSensor("/World/Robot/imu"), battery, name="phone_imu")

    bridge = BatteryROS2Bridge(
        battery=battery,
        sensors={"camera": cam, "imu": imu},
        node_name="battery_sim_node",
        qos_profile="sensor_data",
    )

    world.reset()
    try:
        while simulation_app.is_running():
            world.step(render=True)
            cam.update()
            imu.update()
            bridge.publish()
    finally:
        bridge.shutdown()
        simulation_app.close()
        World.clear_instance()


if __name__ == "__main__":
    main()
