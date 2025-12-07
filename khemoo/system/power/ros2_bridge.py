"""Lightweight ROS2 publisher for battery-aware sensors."""

from __future__ import annotations

import threading
from typing import Dict, Optional

import carb


class BatteryROS2Bridge:
    """Publishes BatteryState and optional SensorHealth messages via rclpy."""

    def __init__(
        self,
        battery,
        sensors: Optional[Dict[str, object]] = None,
        node_name: str = "battery_sim_node",
        qos_profile: str = "sensor_data",
        use_sim_time: bool = True,
    ) -> None:
        self.battery = battery
        self.sensors = sensors or {}
        self.node_name = node_name
        self._qos_profile_name = qos_profile
        self._use_sim_time = use_sim_time
        self._node = None
        self._battery_pub = None
        self._sensor_health_pub = None
        self._SensorHealth = None
        self._lock = threading.Lock()
        self._rclpy_ok = self._init_rclpy()

    # Initialization ---------------------------------------------------------

    def _init_rclpy(self) -> bool:
        try:
            import rclpy
            from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
            from sensor_msgs.msg import BatteryState
        except Exception as exc:
            carb.log_warn(f"[BatteryROS2Bridge] rclpy not available, skipping ROS2 publishing: {exc}")
            return False

        try:
            rclpy.init(args=None)
        except Exception:
            pass  # already initialized

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE if self._qos_profile_name != "sensor_data" else QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self._node = rclpy.create_node(self.node_name)
        if self._use_sim_time:
            self._node.set_parameters_by_dict({"use_sim_time": True})

        self._battery_pub = self._node.create_publisher(BatteryState, "/battery/state", qos)

        try:
            from sensor_health_interfaces.msg import SensorHealth  # type: ignore

            self._SensorHealth = SensorHealth
            self._sensor_health_pub = self._node.create_publisher(SensorHealth, "/battery/sensor_health", qos)
        except Exception as exc:
            carb.log_warn(f"[BatteryROS2Bridge] SensorHealth message unavailable, will publish battery only: {exc}")

        return True

    # Publishing ------------------------------------------------------------

    def publish(self) -> None:
        if not self._rclpy_ok:
            return

        import rclpy
        from sensor_msgs.msg import BatteryState

        with self._lock:
            state = self.battery.get_state()
            msg = BatteryState()
            msg.voltage = float(state["terminal_v"])
            msg.current = -abs(state["current_a"])  # discharge current is negative
            msg.charge = float(state["soc"] * self.battery.capacity_ah * 3600.0)
            msg.capacity = float(self.battery.capacity_ah * 3600.0)
            msg.percentage = float(state["soc"])
            msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            msg.present = state["soc"] > 0.0
            self._battery_pub.publish(msg)

            if self._sensor_health_pub and self._SensorHealth:
                for sensor in self.sensors.values():
                    if not hasattr(sensor, "get_health_report"):
                        continue
                    report = sensor.get_health_report()
                    smsg = self._SensorHealth()
                    # Defensive population; ignore missing fields in the message type
                    if hasattr(smsg, "name"):
                        smsg.name = str(report.get("name", ""))
                    if hasattr(smsg, "health"):
                        smsg.health = str(report.get("health", ""))
                    if hasattr(smsg, "soc"):
                        smsg.soc = float(report.get("soc", 0.0))
                    if hasattr(smsg, "voltage"):
                        smsg.voltage = float(report.get("voltage", 0.0))
                    if hasattr(smsg, "drop_prob"):
                        smsg.drop_prob = float(report.get("drop_prob", 0.0))
                    if hasattr(smsg, "latency_jitter"):
                        smsg.latency_jitter = float(report.get("latency_jitter", 0.0))
                    self._sensor_health_pub.publish(smsg)

        rclpy.spin_once(self._node, timeout_sec=0.0)

    # Cleanup ---------------------------------------------------------------

    def shutdown(self) -> None:
        if not self._rclpy_ok:
            return
        try:
            import rclpy
        except Exception:
            return
        if self._node:
            self._node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        self._rclpy_ok = False

    def __del__(self):
        self.shutdown()
