"""Capillary pressure models: Brooks-Corey and Leverett J-function.

Both are written in terms of a normalized wetting saturation Sn in [0, 1]
(for oil-water drainage Sn = (Sw - Swl) / (1 - Swl)):

    Brooks-Corey:      Pc = pe * Sn**(-1/lam)
    Leverett J power:  J  = a * Sn**(-b)
    Leverett J exp:    J  = a * exp(-b * Sn)
    with               Pc = J * ift * cos(theta) / sqrt(k / phi)

Power laws grow without bound as Sn -> 0, so they are capped at ``pcmax``; without
a cap, Sn is floored at SN_FLOOR. At Sn = 1 the curve equals the entry (threshold)
pressure, which places the contact above the free water level.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Union

import numpy as np

SN_FLOOR = 0.01

# multiply Pa by this to get the unit
PRESSURE_UNITS = {"bar": 1e-5, "psi": 1.0 / 6894.757, "kPa": 1e-3, "MPa": 1e-6,
                  "atm": 1.0 / 101325.0}
MILLIDARCY_M2 = 9.869233e-16
G = 9.80665


def _check_units(units: str):
    if units not in PRESSURE_UNITS:
        raise ValueError(f"Unknown pressure units '{units}', expected one of {list(PRESSURE_UNITS)}")


def _power_law(sn, coef, exponent, pcmax):
    sn = np.clip(np.asarray(sn, dtype=float), 0.0, 1.0)
    if coef == 0:
        return np.zeros_like(sn)
    if pcmax is None:
        return coef * np.maximum(sn, SN_FLOOR) ** (-exponent)
    with np.errstate(divide="ignore"):
        pc = coef * sn ** (-exponent)
    return np.minimum(pc, pcmax)


@dataclass(frozen=True)
class BrooksCorey:
    """Pc = pe * Sn**(-1/lam); pe and pcmax are in the rock type's pc_units."""

    pe: float
    lam: float = 2.0
    pcmax: Optional[float] = None
    model = "brooks-corey"

    def __post_init__(self):
        if self.pe < 0 or self.lam <= 0:
            raise ValueError(f"Brooks-Corey needs pe >= 0 and lam > 0, got {self}")
        if self.pcmax is not None and self.pcmax < self.pe:
            raise ValueError(f"Brooks-Corey pcmax ({self.pcmax}) must be >= pe ({self.pe})")

    def pc(self, sn, units: str = "bar"):
        return _power_law(sn, self.pe, 1.0 / self.lam, self.pcmax)


@dataclass(frozen=True)
class LeverettJ:
    """J = a * Sn**(-b) (form "power") or a * exp(-b * Sn) (form "exp"), scaled with
    permeability (mD), porosity and IFT (mN/m)."""

    a: float
    b: float
    perm: float
    poro: float
    ift: float = 25.0
    theta: float = 0.0  # contact angle, degrees
    pcmax: Optional[float] = None  # in the rock type's pc_units
    form: str = "power"
    model = "leverett"

    def __post_init__(self):
        if self.a < 0 or self.b <= 0 or self.perm <= 0 or not 0 < self.poro < 1 or self.ift < 0:
            raise ValueError(f"Invalid Leverett J parameters: {self}")
        if self.form not in ("power", "exp"):
            raise ValueError(f"Leverett J form must be 'power' or 'exp', got {self.form!r}")

    def j(self, sn):
        """Dimensionless J at normalized saturation sn (uncapped for the power form at 0)."""
        sn = np.clip(np.asarray(sn, dtype=float), 0.0, 1.0)
        if self.form == "exp":
            return self.a * np.exp(-self.b * sn)
        with np.errstate(divide="ignore"):
            return self.a * sn ** (-self.b)

    def factor(self, units: str) -> float:
        """Pc / J in the requested units."""
        _check_units(units)
        sigma = self.ift * 1e-3 * math.cos(math.radians(self.theta))
        return sigma / math.sqrt(self.perm * MILLIDARCY_M2 / self.poro) * PRESSURE_UNITS[units]

    def pc(self, sn, units: str = "bar"):
        if self.form == "exp":
            pc = self.j(sn) * self.factor(units)
            return pc if self.pcmax is None else np.minimum(pc, self.pcmax)
        return _power_law(sn, self.a * self.factor(units), self.b, self.pcmax)


PcModel = Union[BrooksCorey, LeverettJ]
MODELS = {cls.model: cls for cls in (BrooksCorey, LeverettJ)}


def pc_from(value) -> Optional[PcModel]:
    """Build a Pc model from a config value: None, a model, or {"model": ..., **params}."""
    if value is None or isinstance(value, (BrooksCorey, LeverettJ)):
        return value
    params = dict(value)
    name = params.pop("model", "brooks-corey")
    if name not in MODELS:
        raise ValueError(f"Unknown Pc model '{name}', expected one of {list(MODELS)}")
    return MODELS[name](**params)


def pc_to_dict(model: Optional[PcModel]):
    return None if model is None else {"model": model.model, **asdict(model)}


def height_above_fwl(pc, drho: float, units: str = "bar"):
    """Height (m) above the free water level for capillary pressure pc and density
    difference drho (kg/m3) between the phases."""
    _check_units(units)
    return np.asarray(pc, dtype=float) / PRESSURE_UNITS[units] / (drho * G)
