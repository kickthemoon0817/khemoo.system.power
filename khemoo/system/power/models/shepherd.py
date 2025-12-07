"""Shepherd/Tremblay model (Shepherd 1965; Tremblay & Dessaint 2009)."""

from __future__ import annotations

import math
from typing import Tuple

from .base import BatteryModelBase


class ShepherdModel(BatteryModelBase):
    def __init__(
        self,
        capacity_ah: float,
        r0: float,
        e0: float,
        k: float,
        a_exp: float,
        b_exp: float,
        temp_ref_C: float = 25.0,
        temp_coeff_V_per_C: float = 0.0,
    ) -> None:
        super().__init__(temp_ref_C=temp_ref_C, temp_coeff_V_per_C=temp_coeff_V_per_C)
        self.capacity_ah = max(capacity_ah, 1e-6)
        self.r0 = r0
        self.e0 = e0
        self.k = k
        self.a_exp = a_exp
        self.b_exp = b_exp

    def estimate_ocv(self, soc: float) -> float:
        soc_clamped = max(1e-4, min(0.9999, soc))
        ah_consumed = (1.0 - soc_clamped) * self.capacity_ah
        q = self.capacity_ah
        # E = E0 - K*Q/(Q - it)*i + A*exp(-B*it)
        polarization = self.k * (q / max(q - ah_consumed, 1e-6))
        ocv = self.e0 - polarization + self.a_exp * math.exp(-self.b_exp * ah_consumed)
        return self._adjust_ocv(ocv)

    def step(self, soc: float, current_a: float, dt: float) -> Tuple[float, float]:  # noqa: ARG002
        ocv = self.estimate_ocv(soc)
        terminal_v = max(0.0, ocv - current_a * self.r0)
        return terminal_v, ocv
