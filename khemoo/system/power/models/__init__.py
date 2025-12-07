"""Factories for battery voltage models."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, Tuple

import carb

from .base import BatteryModelBase
from .nernst import NernstModel
from .rint import RintModel
from .shepherd import ShepherdModel
from .thevenin import TheveninModel


class ModelType(str, Enum):
    THEVENIN = "thevenin"
    RINT = "rint"
    SHEPHERD = "shepherd"
    NERNST = "nernst"


def create_battery_model(
    model_cfg: Dict,
    ocv_lookup: Callable[[float], float],
    capacity_ah: float,
    r0: float,
    r1: float,
    c1: float,
    temp_ref_C: float,
    temp_coeff_V_per_C: float,
) -> Tuple[BatteryModelBase, ModelType]:
    model_str = str(model_cfg.get("type", "thevenin")).lower()
    try:
        model_type = ModelType(model_str)
    except ValueError:
        carb.log_warn(f"[BatteryPrim] Unknown model '{model_str}', defaulting to thevenin")
        model_type = ModelType.THEVENIN

    if model_type == ModelType.RINT:
        model = RintModel(ocv_lookup=ocv_lookup, r0=r0, temp_ref_C=temp_ref_C, temp_coeff_V_per_C=temp_coeff_V_per_C)
    elif model_type == ModelType.SHEPHERD:
        mcfg = model_cfg.get("shepherd", model_cfg) or {}
        model = ShepherdModel(
            capacity_ah=capacity_ah,
            r0=r0,
            e0=float(mcfg.get("e0", 3.7)),
            k=float(mcfg.get("k", 0.01)),
            a_exp=float(mcfg.get("a_exp", 0.1)),
            b_exp=float(mcfg.get("b_exp", 3.0)),
            temp_ref_C=temp_ref_C,
            temp_coeff_V_per_C=temp_coeff_V_per_C,
        )
    elif model_type == ModelType.NERNST:
        mcfg = model_cfg.get("nernst", model_cfg) or {}
        model = NernstModel(
            r0=r0,
            e0=float(mcfg.get("e0", 3.7)),
            k=float(mcfg.get("k", 0.12)),
            temp_ref_C=temp_ref_C,
            temp_coeff_V_per_C=temp_coeff_V_per_C,
        )
    else:
        model = TheveninModel(
            ocv_lookup=ocv_lookup,
            r0=r0,
            r1=r1,
            c1=c1,
            temp_ref_C=temp_ref_C,
            temp_coeff_V_per_C=temp_coeff_V_per_C,
        )
        model_type = ModelType.THEVENIN

    return model, model_type
