"""Battery model and power tracking utilities for Isaac Sim.

Model references:
- Thevenin 1RC / Rint: Chen & Rincón-Mora (2006) “Accurate Electrical Battery Model”
- Shepherd equation: Shepherd (1965), Tremblay & Dessaint (2009) “A Generic Battery Model”
- Nernst/logistic hybrid: He et al. (2011) “Battery Modeling and SOC Estimation”
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import carb

from .models import ModelType, create_battery_model

DEFAULT_CONFIG = {
    "capacity_ah": 3.0,
    "soc_init": 1.0,
    "r0": 0.05,
    "r1": 0.02,
    "c1": 2500.0,
    "model": {"type": "thevenin"},
    "temperature_C": 25.0,
    "temp_ref_C": 25.0,
    "temp_coeff_V_per_C": 0.0,
    "ocv_curve": [
        {"soc": 0.0, "v": 3.0},
        {"soc": 0.1, "v": 3.3},
        {"soc": 0.2, "v": 3.45},
        {"soc": 0.4, "v": 3.65},
        {"soc": 0.6, "v": 3.75},
        {"soc": 0.8, "v": 3.95},
        {"soc": 1.0, "v": 4.15},
    ],
    "degradation_breakpoints": [
        {"soc_max": 0.8, "label": "nominal"},
        {"soc_max": 0.5, "label": "reduced"},
        {"soc_max": 0.3, "label": "constrained"},
        {"soc_max": 0.15, "label": "critical"},
    ],
    # Shepherd / Tremblay parameters (see Tremblay & Dessaint 2009)
    "shepherd": {"e0": 3.7, "k": 0.01, "a_exp": 0.1, "b_exp": 3.0},
    # Nernst/logistic parameters (see He et al. 2011)
    "nernst": {"e0": 3.7, "k": 0.12},
}


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to the extension root if it is not absolute."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(__file__).resolve().parents[3]
    return root.joinpath(candidate)


class BatteryPrim:
    """Battery wrapper that tracks loads and delegates voltage computation to a model."""

    def __init__(
        self,
        world=None,
        name: str = "battery",
        config_path: Optional[str] = None,
        config: Optional[dict] = None,
        register_callback: bool = True,
    ) -> None:
        self.name = name
        self._world = world
        cfg = self._load_config(config_path, config)
        self.capacity_ah: float = float(cfg["capacity_ah"])
        self.soc: float = float(cfg.get("soc_init", 1.0))
        self.r0: float = float(cfg.get("r0", 0.0))
        self.r1: float = float(cfg.get("r1", 0.0))
        self.c1: float = float(cfg.get("c1", 0.0))
        self.temperature_C: float = float(cfg.get("temperature_C", 25.0))
        self.temp_ref_C: float = float(cfg.get("temp_ref_C", 25.0))
        self.temp_coeff_V_per_C: float = float(cfg.get("temp_coeff_V_per_C", 0.0))
        self._ocv_points: List[Dict[str, float]] = sorted(cfg["ocv_curve"], key=lambda p: p["soc"])
        self._breakpoints: List[Dict[str, float]] = sorted(cfg.get("degradation_breakpoints", []), key=lambda b: b["soc_max"])

        model_cfg = cfg.get("model", {}) or {}
        self._loads_w: Dict[str, float] = {}
        self._energy_used_Wh: float = 0.0
        self._lock = threading.Lock()
        self._last_dt: float = 0.0

        self._model, self.model_type = create_battery_model(
            model_cfg=model_cfg,
            ocv_lookup=self._interpolate_ocv,
            capacity_ah=self.capacity_ah,
            r0=self.r0,
            r1=self.r1,
            c1=self.c1,
            temp_ref_C=self.temp_ref_C,
            temp_coeff_V_per_C=self.temp_coeff_V_per_C,
        )
        self._model.set_temperature(self.temperature_C)

        if register_callback and self._world is not None:
            try:
                self._world.add_physics_callback(f"{self.name}_tick", self.update)
            except Exception as exc:
                carb.log_error(f"[{self.name}] Failed to register physics callback: {exc}")

    @staticmethod
    def _load_config(config_path: Optional[str], config: Optional[dict]) -> dict:
        if config is not None:
            return {**DEFAULT_CONFIG, **config}
        if config_path:
            resolved = _resolve_path(config_path)
            try:
                with resolved.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {**DEFAULT_CONFIG, **data}
            except FileNotFoundError:
                carb.log_error(f"[BatteryPrim] Config file not found: {resolved}")
            except Exception as exc:  # pragma: no cover - defensive logging
                carb.log_error(f"[BatteryPrim] Failed to load config {resolved}: {exc}")
        return dict(DEFAULT_CONFIG)

    # Public API -------------------------------------------------------------

    def register_load(self, name: str, power_w: float) -> None:
        """Register or update a consumer load in watts."""
        with self._lock:
            self._loads_w[name] = max(0.0, float(power_w))

    def clear_load(self, name: str) -> None:
        """Remove a consumer load."""
        with self._lock:
            self._loads_w.pop(name, None)

    def clear_all_loads(self) -> None:
        with self._lock:
            self._loads_w.clear()

    def update(self, step_size: float) -> None:
        """Advance the battery state by the provided step (seconds)."""
        if step_size is None or step_size <= 0.0:
            return

        with self._lock:
            total_power_w = sum(self._loads_w.values())
            ocv_est = self._model.estimate_ocv(self.soc)
            current_a = total_power_w / max(ocv_est, 1e-3)
            self._last_dt = step_size

            terminal_v, ocv_used = self._model.step(self.soc, current_a, step_size)
            delta_soc = (current_a * step_size) / (3600.0 * max(self.capacity_ah, 1e-6))
            self.soc = max(0.0, min(1.0, self.soc - delta_soc))
            self._energy_used_Wh += (total_power_w * step_size) / 3600.0
            self._state = {
                "soc": self.soc,
                "ocv_v": ocv_used,
                "terminal_v": terminal_v,
                "current_a": current_a,
                "power_w": total_power_w,
                "energy_used_Wh": self._energy_used_Wh,
                "health": self._health_label(self.soc),
                "dt": step_size,
                "model": self.model_type.value if isinstance(self.model_type, ModelType) else str(self.model_type),
            }

    def get_state(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._state) if hasattr(self, "_state") else {
                "soc": self.soc,
                "ocv_v": self._interpolate_ocv(self.soc),
                "terminal_v": self._interpolate_ocv(self.soc),
                "current_a": 0.0,
                "power_w": sum(self._loads_w.values()),
                "energy_used_Wh": self._energy_used_Wh,
                "health": self._health_label(self.soc),
                "dt": self._last_dt,
                "model": self.model_type.value if isinstance(self.model_type, ModelType) else str(self.model_type),
            }

    # Helpers ----------------------------------------------------------------

    def _interpolate_ocv(self, soc: float) -> float:
        pts = self._ocv_points
        if not pts:
            return 0.0
        soc = max(0.0, min(1.0, soc))
        for idx, pt in enumerate(pts):
            if soc <= pt["soc"]:
                if idx == 0:
                    return float(pt["v"])
                prev = pts[idx - 1]
                span = pt["soc"] - prev["soc"]
                t = 0.0 if span <= 0 else (soc - prev["soc"]) / span
                return float(prev["v"] + t * (pt["v"] - prev["v"]))
        return float(pts[-1]["v"])

    def _health_label(self, soc: float) -> str:
        for bp in self._breakpoints:
            if soc <= bp.get("soc_max", 1.0):
                return bp.get("label", "nominal")
        return "nominal"
