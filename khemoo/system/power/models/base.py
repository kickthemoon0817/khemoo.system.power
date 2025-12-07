"""Battery model abstraction used by BatteryPrim.

References:
- Chen & Rincón-Mora (2006) “Accurate Electrical Battery Model”
- Shepherd (1965), Tremblay & Dessaint (2009) “A Generic Battery Model”
- He et al. (2011) “Battery Modeling and SOC Estimation”
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Tuple


class BatteryModelBase(ABC):
    """Abstract base for battery voltage models."""

    def __init__(self, temp_ref_C: float = 25.0, temp_coeff_V_per_C: float = 0.0) -> None:
        self.temperature_C = temp_ref_C
        self.temp_ref_C = temp_ref_C
        self.temp_coeff_V_per_C = temp_coeff_V_per_C

    def set_temperature(self, temp_C: float) -> None:
        self.temperature_C = temp_C

    def _adjust_ocv(self, ocv: float) -> float:
        return ocv + self.temp_coeff_V_per_C * (self.temperature_C - self.temp_ref_C)

    @abstractmethod
    def estimate_ocv(self, soc: float) -> float:
        """Return open-circuit voltage estimate for a given SoC (no current load)."""

    @abstractmethod
    def step(self, soc: float, current_a: float, dt: float) -> Tuple[float, float]:
        """Return (terminal_v, ocv_used) for a given SoC, current, and dt."""


class OcvLookupMixin:
    """Shared helper to resolve OCV from a provided lookup callable."""

    def __init__(self, ocv_lookup: Callable[[float], float], *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        self._ocv_lookup = ocv_lookup

    def _ocv(self, soc: float) -> float:
        return self._ocv_lookup(soc)
