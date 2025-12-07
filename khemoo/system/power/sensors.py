"""Battery-aware wrappers for Isaac Sim sensors."""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

import carb
import numpy as np

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


def _voltage_factor(v: float, v_nom: float = 3.7, v_min: float = 3.4) -> float:
    """Return 0..1 factor of how close we are to cutoff (1 at/under v_min, 0 at/above v_nom)."""
    if v_nom <= v_min:
        return 1.0
    return float(max(0.0, min(1.0, (v_nom - v) / (v_nom - v_min))))


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
        self._last_raw_frame = None

    def update(self) -> Optional[dict]:
        """Fetch a frame; may drop it according to battery health."""
        state = self.battery.get_state()
        level = _select_level(state["soc"], self._degradation_table)
        drop_prob = float(level.get("drop_prob", 0.0))
        # Voltage/load driven boost to drop probability
        v = float(state.get("terminal_v", 3.7))
        drop_prob = min(1.0, drop_prob + 0.1 * _voltage_factor(v))

        # Latency jitter model (seconds)
        base_jitter = 0.003  # 3 ms nominal jitter
        tail = 0.02 * _voltage_factor(v)  # up to +20 ms near cutoff
        latency = max(0.0, random.gauss(base_jitter, base_jitter * 0.5) + tail * random.random())

        # Exposure/noise scaling with voltage
        exposure_scale = max(0.5, 1.0 - 0.3 * _voltage_factor(v, v_nom=3.86, v_min=3.4))
        noise_sigma = 2.0 + 4.0 * _voltage_factor(v, v_nom=3.86, v_min=3.4)  # uint8 domain

        # Decide drop
        if random.random() < drop_prob:
            return {
                "dropped": True,
                "battery_state": state,
                "battery_health": level.get("label", "nominal"),
                "drop_prob": drop_prob,
                "latency_jitter_s": latency,
                "exposure_scale": exposure_scale,
                "noise_sigma": noise_sigma,
                "timestamp_wall_s": time.time(),
            }

        try:
            frame = self.camera.get_current_frame(clone=True)
        except Exception as exc:
            carb.log_error(f"[{self.name}] Failed to get camera frame: {exc}")
            return None

        self._last_raw_frame = frame.copy()

        # Apply latency jitter by delaying and recording the delay.
        if latency > 0.0:
            time.sleep(latency)

        # Apply exposure scaling and noise on RGB/RGBA buffers if present.
        for key in ("rgba", "rgb"):
            if key in frame and isinstance(frame[key], np.ndarray) and frame[key].size > 0:
                arr = frame[key].astype(np.float32)
                arr *= exposure_scale
                if noise_sigma > 0.0:
                    arr += np.random.normal(0.0, noise_sigma, size=arr.shape).astype(np.float32)
                arr = np.clip(arr, 0.0, 255.0).astype(frame[key].dtype)
                frame[key] = arr

        frame["battery_state"] = state
        frame["battery_health"] = level.get("label", "nominal")
        frame["latency_jitter_s"] = latency
        frame["noise_std_hint"] = noise_sigma
        frame["exposure_scale"] = exposure_scale
        frame["drop_prob"] = drop_prob
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
        self._last_raw_frame = None

    def update(self) -> Optional[dict]:
        state = self.battery.get_state()
        level = _select_level(state["soc"], self._degradation_table)
        v = float(state.get("terminal_v", 3.7))
        drop_prob = min(1.0, float(level.get("drop_prob", 0.0)) + 0.05 * _voltage_factor(v))
        latency = max(0.0, random.gauss(0.002, 0.001) + 0.01 * random.random() * _voltage_factor(v))
        noise_sigma = 0.01 + 0.05 * _voltage_factor(v)  # rad/s or m/s^2 scale

        if random.random() < drop_prob:
            return {
                "dropped": True,
                "battery_state": state,
                "battery_health": level.get("label", "nominal"),
                "drop_prob": drop_prob,
                "latency_jitter_s": latency,
                "noise_sigma": noise_sigma,
                "timestamp_wall_s": time.time(),
            }

        try:
            frame = self.imu.get_current_frame(read_gravity=True)
        except Exception as exc:
            carb.log_error(f"[{self.name}] Failed to get IMU frame: {exc}")
            return None

        # Keep a raw copy before perturbation
        raw_copy = {}
        for k, v in frame.items():
            if isinstance(v, np.ndarray):
                raw_copy[k] = np.array(v)
        self._last_raw_frame = raw_copy if raw_copy else None

        if latency > 0.0:
            time.sleep(latency)

        # Inject additional noise/bias
        for key in ("angular_velocity", "ang_vel"):
            if key in frame and isinstance(frame[key], np.ndarray):
                frame[key] = frame[key] + np.random.normal(0.0, noise_sigma, size=frame[key].shape)
        for key in ("linear_acceleration", "lin_acc"):
            if key in frame and isinstance(frame[key], np.ndarray):
                frame[key] = frame[key] + np.random.normal(0.0, noise_sigma, size=frame[key].shape)
        for key in ("orientation",):
            if key in frame and isinstance(frame[key], np.ndarray):
                frame[key] = frame[key] + np.random.normal(0.0, noise_sigma, size=frame[key].shape)

        frame["battery_state"] = state
        frame["battery_health"] = level.get("label", "nominal")
        frame["latency_jitter_s"] = latency
        frame["noise_sigma"] = noise_sigma
        frame["drop_prob"] = drop_prob
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
