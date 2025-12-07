"""Thevenin 1RC model (Chen & Rincón-Mora, 2006) with ohmic drop."""

from __future__ import annotations

import math
from typing import Callable, Tuple

from .base import BatteryModelBase, OcvLookupMixin


class TheveninModel(OcvLookupMixin, BatteryModelBase):
    def __init__(
        self,
        ocv_lookup: Callable[[float], float],
        r0: float,
        r1: float,
        c1: float,
        temp_ref_C: float = 25.0,
        temp_coeff_V_per_C: float = 0.0,
    ) -> None:
        BatteryModelBase.__init__(self, temp_ref_C=temp_ref_C, temp_coeff_V_per_C=temp_coeff_V_per_C)
        OcvLookupMixin.__init__(self, ocv_lookup)
        self.r0 = r0
        self.r1 = r1
        self.c1 = c1
        self._v_rc = 0.0

    def estimate_ocv(self, soc: float) -> float:
        return self._adjust_ocv(self._ocv(soc))

    def step(self, soc: float, current_a: float, dt: float) -> Tuple[float, float]:
        ocv = self.estimate_ocv(soc)
        if self.r1 > 0 and self.c1 > 0:
            alpha = math.exp(-dt / (self.r1 * self.c1))
            self._v_rc = self._v_rc * alpha + self.r1 * (1 - alpha) * current_a
        else:
            self._v_rc = 0.0
        terminal_v = max(0.0, ocv - current_a * self.r0 - self._v_rc)
        return terminal_v, ocv
