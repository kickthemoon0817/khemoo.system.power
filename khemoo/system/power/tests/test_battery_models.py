"""
Sanity checks for battery models using omni.kit.test AsyncTestCase.

References:
- Thevenin 1RC / Rint: Chen & Rincón-Mora (2006) “Accurate Electrical Battery Model”
- Shepherd: Shepherd (1965); Tremblay & Dessaint (2009) “A Generic Battery Model”
- Nernst/logistic: He et al. (2011) “Battery Modeling and SOC Estimation”
"""

from __future__ import annotations

import math

import omni.kit.test

from khemoo.system.power.battery import BatteryPrim


def approx_equal(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


class TestBatteryModels(omni.kit.test.AsyncTestCase):
    async def test_thevenin(self):
        cfg = {
            "capacity_ah": 3.0,
            "soc_init": 1.0,
            "r0": 0.05,
            "r1": 0.02,
            "c1": 2000.0,
            "model": {"type": "thevenin"},
            "ocv_curve": [{"soc": 0.0, "v": 4.0}, {"soc": 1.0, "v": 4.0}],
        }
        batt = BatteryPrim(world=None, config=cfg, register_callback=False)
        batt.register_load("cam", 4.0)  # 4 W at 4 V => ~1 A
        batt.update(1.0)
        state = batt.get_state()
        expected_current = 1.0
        # Chen & Rincón-Mora 2006: V = OCV - I*R0 - Vrc, with Vrc from 1RC branch
        alpha = math.exp(-1.0 / (cfg["r1"] * cfg["c1"]))
        v_rc = cfg["r1"] * (1 - alpha) * expected_current
        expected_v = 4.0 - expected_current * cfg["r0"] - v_rc
        self.assertTrue(approx_equal(state["current_a"], expected_current, tol=0.05), state)
        self.assertTrue(approx_equal(state["terminal_v"], expected_v, tol=0.02), state)

    async def test_rint(self):
        cfg = {
            "capacity_ah": 2.0,
            "soc_init": 1.0,
            "r0": 0.1,
            "r1": 0.0,
            "c1": 0.0,
            "model": {"type": "rint"},
            "ocv_curve": [{"soc": 0.0, "v": 4.0}, {"soc": 1.0, "v": 4.0}],
        }
        batt = BatteryPrim(world=None, config=cfg, register_callback=False)
        batt.register_load("cam", 2.0)  # 2 W / 4 V = 0.5 A
        batt.update(1.0)
        state = batt.get_state()
        expected_current = 0.5
        expected_v = 4.0 - expected_current * cfg["r0"]
        self.assertTrue(approx_equal(state["current_a"], expected_current, tol=1e-3), state)
        self.assertTrue(approx_equal(state["terminal_v"], expected_v, tol=1e-3), state)

    async def test_shepherd(self):
        cfg = {
            "capacity_ah": 3.0,
            "soc_init": 0.5,
            "r0": 0.05,
            "model": {"type": "shepherd", "shepherd": {"e0": 3.7, "k": 0.01, "a_exp": 0.1, "b_exp": 3.0}},
            "ocv_curve": [{"soc": 0.0, "v": 3.7}, {"soc": 1.0, "v": 3.7}],  # unused by shepherd but required
        }
        batt = BatteryPrim(world=None, config=cfg, register_callback=False)
        batt.register_load("cam", 3.7)  # ~1 A at mid-SoC
        batt.update(1.0)
        state = batt.get_state()
        q = cfg["capacity_ah"]
        ah_consumed = (1 - cfg["soc_init"]) * q
        polarization = 0.01 * (q / (q - ah_consumed))  # Shepherd polarization term
        ocv = 3.7 - polarization + 0.1 * math.exp(-3.0 * ah_consumed)
        expected_v = max(0.0, ocv - 1.0 * cfg["r0"])
        self.assertTrue(approx_equal(state["terminal_v"], expected_v, tol=0.05), state)

    async def test_nernst(self):
        cfg = {
            "capacity_ah": 2.5,
            "soc_init": 0.5,
            "r0": 0.05,
            "model": {"type": "nernst", "nernst": {"e0": 3.7, "k": 0.12}},
            "ocv_curve": [{"soc": 0.0, "v": 3.7}, {"soc": 1.0, "v": 3.7}],  # unused but required
        }
        batt = BatteryPrim(world=None, config=cfg, register_callback=False)
        batt.register_load("cam", 3.7)  # ~1 A
        batt.update(1.0)
        state = batt.get_state()
        ocv = 3.7  # ln(1) = 0 at soc=0.5
        expected_v = ocv - 1.0 * cfg["r0"]
        self.assertTrue(approx_equal(state["terminal_v"], expected_v, tol=0.05), state)
