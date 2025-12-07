# Battery-Aware Sensors (khemoo.system.power)

This extension provides a lightweight battery model, wrappers for Isaac Sim cameras/IMUs, and an optional ROS 2 publisher for battery and sensor health.

## Features
- Thevenin 1RC battery model with JSON-configurable parameters and per-load power tracking.
- Battery-aware camera and IMU wrappers that can drop frames or add latency hints based on state-of-charge.
- ROS 2 publisher (`BatteryROS2Bridge`) that streams `sensor_msgs/BatteryState` and optional `sensor_health_interfaces/SensorHealth` messages when `rclpy` is available.

## Usage
1. Launch Isaac Sim with the extension enabled.
2. Create a `BatteryPrim`, register loads, and attach sensor wrappers:
   ```python
   from isaacsim.core.api import World
   from isaacsim.sensors.camera import Camera
   from isaacsim.sensors.physics import IMUSensor
   from khemoo.system.power.battery import BatteryPrim
   from khemoo.system.power.sensors import BatteryCamera, BatteryIMU
   from khemoo.system.power.ros2_bridge import BatteryROS2Bridge

   world = World(stage_units_in_meters=1.0)
   battery = BatteryPrim(world, name="phone_battery", config_path="khemoo/system/power/configs/default_battery.json")
   cam = BatteryCamera(Camera("/World/Robot/camera"), battery, name="phone_cam")
   imu = BatteryIMU(IMUSensor("/World/Robot/imu"), battery, name="phone_imu")
   bridge = BatteryROS2Bridge(battery=battery, sensors={"camera": cam, "imu": imu})
   ```
3. Ensure the battery update runs each physics tick by the registered callback; call `cam.update()` / `imu.update()` in your control loop before publishing with `bridge.publish()`.

## Notes
- `sensor_health_interfaces` is expected to be provided by a shared ROS 2 interfaces package; battery publishing works even if it is absent.
- Isaac Sim 5.1.0 APIs are used (`isaacsim.simulation_app`, `isaacsim.core.api`, `isaacsim.sensors.camera`, `isaacsim.sensors.physics`).
