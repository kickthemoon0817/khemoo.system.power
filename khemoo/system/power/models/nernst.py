"""Nernst/logistic hybrid model (He et al., 2011)."""

from __future__ import annotations

import math
from typing import Tuple

from .base import BatteryModelBase


class NernstModel(BatteryModelBase):
    def __init__(
        self,
        r0: float,
        e0: float,
        k: float,
        temp_ref_C: float = 25.0,
        temp_coeff_V_per_C: float = 0.0,
    ) -> None:
        super().__init__(temp_ref_C=temp_ref_C, temp_coeff_V_per_C=temp_coeff_V_per_C)
        self.r0 = r0
        self.e0 = e0
        self.k = k

    def estimate_ocv(self, soc: float) -> float:
        soc_clamped = max(1e-4, min(0.9999, soc))
        ratio = soc_clamped / (1.0 - soc_clamped)
        ocv = self.e0 + self.k * math.log(ratio)
        return self._adjust_ocv(ocv)

    def step(self, soc: float, current_a: float, dt: float) -> Tuple[float, float]:  # noqa: ARG002
        ocv = self.estimate_ocv(soc)
        terminal_v = max(0.0, ocv - current_a * self.r0)
        return terminal_v, ocv
