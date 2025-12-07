"""Battery-aware wrappers for Isaac Sim sensors."""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

import carb

from .battery import BatteryPrim

# Default degradation map shared across sensors
DEFAULT_DEGRADATION: List[Dict[str, float]] = [
    {"soc_max": 0.8, "label": "nominal", "drop_prob": 0.0, "latency_jitter": 0.0, "noise_std": 0.0},
    {"soc_max": 0.5, "label": "reduced", "drop_prob": 0.05, "latency_jitter": 0.01, "noise_std": 0.0},
    {"soc_max": 0.3, "label": "constrained", "drop_prob": 0.12, "latency_jitter": 0.02, "noise_std": 0.001},
    {"soc_max": 0.15, "label": "critical", "drop_prob": 0.25, "latency_jitter": 0.04, "noise_std": 0.002},
]


def _select_level(soc: float, table: List[Dict[str, float]]) -> Dict[str, float]:
    sorted_table = sorted(table, key=lambda b: b.get("soc_max", 1.0))
    for level in sorted_table:
        if soc <= level.get("soc_max", 1.0):
            return level
    return sorted_table[-1] if sorted_table else {"label": "nominal"}


class BatteryCamera:
    """Wraps isaacsim.sensors.camera.Camera to attach battery-aware behavior."""

    def __init__(
        self,
        camera,
        battery: BatteryPrim,
        name: str = "battery_camera",
        power_w: float = 2.0,
        degradation_table: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        self.camera = camera
        self.battery = battery
        self.name = name
        self._degradation_table = degradation_table or list(DEFAULT_DEGRADATION)
        self.battery.register_load(self.name, power_w)
        self._last_frame = None

    def update(self) -> Optional[dict]:
        """Fetch a frame; may drop it according to battery health."""
        state = self.battery.get_state()
        level = _select_level(state["soc"], self._degradation_table)
        drop_prob = float(level.get("drop_prob", 0.0))
        if drop_prob > 0.0 and random.random() < drop_prob:
            return None

        try:
            frame = self.camera.get_current_frame(clone=True)
        except Exception as exc:
            carb.log_error(f"[{self.name}] Failed to get camera frame: {exc}")
            return None

        frame["battery_state"] = state
        frame["battery_health"] = level.get("label", "nominal")
        frame["latency_jitter_s"] = level.get("latency_jitter", 0.0)
        frame["noise_std_hint"] = level.get("noise_std", 0.0)
        frame["timestamp_wall_s"] = time.time()
        self._last_frame = frame
        return frame

    def get_health_report(self) -> Dict[str, float]:
        state = self.battery.get_state()
        level = _select_level(state["soc"], self._degradation_table)
        return {
            "name": self.name,
            "health": level.get("label", "nominal"),
            "soc": state["soc"],
            "voltage": state["terminal_v"],
            "drop_prob": level.get("drop_prob", 0.0),
            "latency_jitter": level.get("latency_jitter", 0.0),
        }

    def destroy(self) -> None:
        self.battery.clear_load(self.name)


class BatteryIMU:
    """Wraps isaacsim.sensors.physics.IMUSensor to attach battery-aware behavior."""

    def __init__(
        self,
        imu_sensor,
        battery: BatteryPrim,
        name: str = "battery_imu",
        power_w: float = 0.5,
        degradation_table: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        self.imu = imu_sensor
        self.battery = battery
        self.name = name
        self._degradation_table = degradation_table or list(DEFAULT_DEGRADATION)
        self.battery.register_load(self.name, power_w)
        self._last_frame = None

    def update(self) -> Optional[dict]:
        state = self.battery.get_state()
        level = _select_level(state["soc"], self._degradation_table)
        drop_prob = float(level.get("drop_prob", 0.0))
        if drop_prob > 0.0 and random.random() < drop_prob:
            return None

        try:
            frame = self.imu.get_current_frame(read_gravity=True)
        except Exception as exc:
            carb.log_error(f"[{self.name}] Failed to get IMU frame: {exc}")
            return None

        frame["battery_state"] = state
        frame["battery_health"] = level.get("label", "nominal")
        frame["latency_jitter_s"] = level.get("latency_jitter", 0.0)
        frame["timestamp_wall_s"] = time.time()
        self._last_frame = frame
        return frame

    def get_health_report(self) -> Dict[str, float]:
        state = self.battery.get_state()
        level = _select_level(state["soc"], self._degradation_table)
        return {
            "name": self.name,
            "health": level.get("label", "nominal"),
            "soc": state["soc"],
            "voltage": state["terminal_v"],
            "drop_prob": level.get("drop_prob", 0.0),
            "latency_jitter": level.get("latency_jitter", 0.0),
        }

    def destroy(self) -> None:
        self.battery.clear_load(self.name)
