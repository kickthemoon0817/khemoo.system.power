"""Rint model: ohmic internal resistance applied to OCV lookup."""

from __future__ import annotations

from typing import Callable, Tuple

from .base import BatteryModelBase, OcvLookupMixin


class RintModel(OcvLookupMixin, BatteryModelBase):
    def __init__(
        self,
        ocv_lookup: Callable[[float], float],
        r0: float,
        temp_ref_C: float = 25.0,
        temp_coeff_V_per_C: float = 0.0,
    ) -> None:
        BatteryModelBase.__init__(self, temp_ref_C=temp_ref_C, temp_coeff_V_per_C=temp_coeff_V_per_C)
        OcvLookupMixin.__init__(self, ocv_lookup)
        self.r0 = r0

    def estimate_ocv(self, soc: float) -> float:
        return self._adjust_ocv(self._ocv(soc))

    def step(self, soc: float, current_a: float, dt: float) -> Tuple[float, float]:  # noqa: ARG002
        ocv = self.estimate_ocv(soc)
        terminal_v = max(0.0, ocv - current_a * self.r0)
        return terminal_v, ocv
