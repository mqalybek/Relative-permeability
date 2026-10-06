"""Two-phase relative permeability models (Corey, LET) and Stone II three-phase oil.

Saturation conventions (Eclipse-style names):

    swl    connate water saturation (first Sw in SWOF)
    swcr   critical water saturation (krw = 0 for Sw <= swcr)
    sowcr  residual oil saturation to water (Sorw)
    sgl    connate gas saturation (first Sg in SGOF, usually 0)
    sgcr   critical gas saturation (krg = 0 for Sg <= sgcr)
    sogcr  residual oil saturation to gas (Sorg)

Capillary pressure (optional, zero when not set) uses the same normalization
as the oil curves: Pcow on (Sw - swl) / (1 - swl - sowcr), Pcog on the liquid
saturation (Sl - swl - sogcr) / (1 - sgl - swl - sogcr). See capillary.py.

Endpoints:

    krwr   krw at Sw = 1 - sowcr
    krwmax krw at Sw = 1 (defaults to krwr)
    krocw  krow at Sw = swl
    krgr   krg at Sg = 1 - swl - sogcr
    krgmax krg at Sg = 1 - swl (defaults to krgr)
    krogcg krog at Sg = sgl
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Union

import numpy as np

from .capillary import PcModel, _check_units, height_above_fwl, pc_from, pc_to_dict


@dataclass(frozen=True)
class LET:
    """LET correlation shape (Lomeland, Ebeltoft, Thomas, 2005)."""

    L: float
    E: float
    T: float

    def __post_init__(self):
        if self.L <= 0 or self.E <= 0 or self.T <= 0:
            raise ValueError(f"LET parameters must be positive, got {self}")

    def __call__(self, sn):
        a = sn**self.L
        b = (1.0 - sn) ** self.T
        return a / (a + self.E * b)


# A curve shape is either a Corey exponent or a LET triple.
Shape = Union[float, LET]


def shape_from(value) -> Shape:
    """Build a shape from a config value: number -> Corey exponent, dict -> LET."""
    if isinstance(value, LET):
        return value
    if isinstance(value, dict):
        return LET(**value)
    n = float(value)
    if n <= 0:
        raise ValueError(f"Corey exponent must be positive, got {n}")
    return n


def evaluate_shape(sn, shape: Shape):
    sn = np.clip(np.asarray(sn, dtype=float), 0.0, 1.0)
    if isinstance(shape, LET):
        return shape(sn)
    return sn**shape


def normalize(s, s_lo, s_hi):
    """Normalized saturation clipped to [0, 1]; avoids negative bases in s**n."""
    return np.clip((np.asarray(s, dtype=float) - s_lo) / (s_hi - s_lo), 0.0, 1.0)


def _tail(s, s_from, s_to, k_from, k_to):
    """Linear rise of kr from k_from at s_from to k_to at s_to (for s > s_from)."""
    if s_to <= s_from:
        return np.full_like(s, k_from)
    return k_from + (k_to - k_from) * np.clip((s - s_from) / (s_to - s_from), 0.0, 1.0)


def _grid(lo, hi, n, extra):
    """Uniform grid on [lo, hi] plus breakpoints (critical saturations)."""
    pts = np.concatenate([np.linspace(lo, hi, n), [e for e in extra if lo <= e <= hi]])
    return np.unique(np.round(pts, 10))


@dataclass
class RockType:
    """Saturation function set for one rock type (SATNUM region)."""

    name: str = "RT1"
    # water-oil
    swl: float = 0.18
    swcr: float = 0.18
    sowcr: float = 0.11
    krwr: float = 0.6
    krwmax: float | None = None
    krocw: float = 0.9
    nw: Shape = 3.0
    now: Shape = 3.0
    # gas-oil
    sgl: float = 0.0
    sgcr: float = 0.0
    sogcr: float = 0.11
    krgr: float = 0.1
    krgmax: float | None = None
    krogcg: float = 0.9
    ng: Shape = 2.0
    nog: Shape = 1.5
    # capillary pressure (None -> 0), values in pc_units
    pcow: Optional[PcModel] = None
    pcog: Optional[PcModel] = None
    pc_units: str = "bar"
    # optional comment written to exported tables
    description: str = field(default="", repr=False)

    def __post_init__(self):
        for attr in ("nw", "now", "ng", "nog"):
            setattr(self, attr, shape_from(getattr(self, attr)))
        self.pcow = pc_from(self.pcow)
        self.pcog = pc_from(self.pcog)
        if self.krwmax is None:
            self.krwmax = self.krwr
        if self.krgmax is None:
            self.krgmax = self.krgr
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self):
        errors = []

        def need(cond, msg):
            if not cond:
                errors.append(msg)

        for attr in ("swl", "swcr", "sowcr", "sgl", "sgcr", "sogcr"):
            need(0.0 <= getattr(self, attr) < 1.0, f"{attr} must be in [0, 1)")
        for attr in ("krwr", "krwmax", "krocw", "krgr", "krgmax", "krogcg"):
            need(0.0 < getattr(self, attr) <= 1.0, f"{attr} must be in (0, 1]")
        need(self.swcr >= self.swl, "swcr must be >= swl")
        need(self.swcr < 1.0 - self.sowcr, "swcr + sowcr must be < 1")
        need(self.krwmax >= self.krwr, "krwmax must be >= krwr")
        need(self.sgcr >= self.sgl, "sgcr must be >= sgl")
        need(self.sgcr < 1.0 - self.swl - self.sogcr, "sgcr + swl + sogcr must be < 1")
        need(self.sgl + self.swl + self.sogcr < 1.0, "sgl + swl + sogcr must be < 1")
        need(self.krgmax >= self.krgr, "krgmax must be >= krgr")
        try:
            _check_units(self.pc_units)
        except ValueError as e:
            errors.append(str(e))
        if errors:
            raise ValueError(f"Rock type '{self.name}': " + "; ".join(errors))

    def warnings(self) -> list[str]:
        """Physically legal but simulator-unfriendly combinations."""
        out = []
        if not np.isclose(self.krocw, self.krogcg):
            out.append(
                f"krocw ({self.krocw:g}) != krogcg ({self.krogcg:g}): Eclipse requires "
                "krow(Swl) == krog(Sg=0) for three-phase oil (Stone) models"
            )
        if self.sgl > 0:
            out.append("sgl > 0: Eclipse SGOF/SGFN normally start at Sg = 0")
        return out

    # ------------------------------------------------------------- water-oil
    def krw(self, sw):
        sw = np.asarray(sw, dtype=float)
        s_hi = 1.0 - self.sowcr
        k = self.krwr * evaluate_shape(normalize(sw, self.swcr, s_hi), self.nw)
        return np.where(sw > s_hi, _tail(sw, s_hi, 1.0, self.krwr, self.krwmax), k)

    def krow(self, sw):
        so = 1.0 - np.asarray(sw, dtype=float)
        return self.krocw * evaluate_shape(normalize(so, self.sowcr, 1.0 - self.swl), self.now)

    # --------------------------------------------------------------- gas-oil
    def krg(self, sg):
        sg = np.asarray(sg, dtype=float)
        s_hi = 1.0 - self.swl - self.sogcr
        k = self.krgr * evaluate_shape(normalize(sg, self.sgcr, s_hi), self.ng)
        return np.where(sg > s_hi, _tail(sg, s_hi, 1.0 - self.swl, self.krgr, self.krgmax), k)

    def krog(self, sg):
        sl = 1.0 - np.asarray(sg, dtype=float)
        lo = self.swl + self.sogcr
        return self.krogcg * evaluate_shape(normalize(sl, lo, 1.0 - self.sgl), self.nog)

    # ------------------------------------------------------ capillary pressure
    def pc_ow(self, sw):
        sw = np.asarray(sw, dtype=float)
        if self.pcow is None:
            return np.zeros_like(sw)
        return self.pcow.pc(normalize(sw, self.swl, 1.0 - self.sowcr), self.pc_units)

    def pc_og(self, sg):
        sl = 1.0 - np.asarray(sg, dtype=float)
        if self.pcog is None:
            return np.zeros_like(sl)
        sn = normalize(sl, self.swl + self.sogcr, 1.0 - self.sgl)
        return self.pcog.pc(sn, self.pc_units)

    def height_above_fwl(self, sw, drho: float = 250.0):
        """Height (m) above the free water level at which water saturation is sw,
        for oil-water density difference drho (kg/m3) — a saturation-height check."""
        return height_above_fwl(self.pc_ow(sw), drho, self.pc_units)

    # ---------------------------------------------------------------- tables
    def sw_grid(self, points: int = 20):
        return _grid(self.swl, 1.0, points, [self.swcr, 1.0 - self.sowcr])

    def sg_grid(self, points: int = 20):
        return _grid(self.sgl, 1.0 - self.swl, points, [self.sgcr, 1.0 - self.swl - self.sogcr])

    def water_oil_table(self, points: int = 20) -> dict[str, np.ndarray]:
        sw = self.sw_grid(points)
        return {"Sw": sw, "krw": self.krw(sw), "krow": self.krow(sw), "Pcow": self.pc_ow(sw)}

    def gas_oil_table(self, points: int = 20) -> dict[str, np.ndarray]:
        sg = self.sg_grid(points)
        return {"Sg": sg, "krg": self.krg(sg), "krog": self.krog(sg), "Pcog": self.pc_og(sg)}

    def oil_table(self, points: int = 20) -> dict[str, np.ndarray]:
        """SOF3-style table: So, krow(So), krog(So) with Sl = So + swl."""
        so_w = 1.0 - self.sw_grid(points)
        so_g = 1.0 - self.swl - self.sg_grid(points)
        so = np.unique(np.round(np.concatenate([so_w, so_g[so_g >= 0]]), 10))
        return {"So": so, "krow": self.krow(1.0 - so), "krog": self.krog(1.0 - self.swl - so)}

    # ----------------------------------------------------------- three-phase
    def kro_stone2(self, sw, sg):
        """Three-phase oil relative permeability, normalized Stone II (Aziz & Settari)."""
        krw, krg = self.krw(sw), self.krg(sg)
        krow, krog = self.krow(sw), self.krog(sg)
        kro = self.krocw * ((krow / self.krocw + krw) * (krog / self.krocw + krg) - (krw + krg))
        return np.clip(kro, 0.0, None)

    # --------------------------------------------------------------- helpers
    def fractional_flow(self, sw, mu_w: float = 0.5, mu_o: float = 2.0):
        """Water fractional flow fw (no gravity, no capillary pressure)."""
        lw = self.krw(sw) / mu_w
        lo = self.krow(sw) / mu_o
        total = lw + lo
        return np.divide(lw, total, out=np.zeros_like(total), where=total > 0)

    def to_dict(self) -> dict:
        d = asdict(self)  # LET shapes become {"L", "E", "T"} dicts
        d["pcow"], d["pcog"] = pc_to_dict(self.pcow), pc_to_dict(self.pcog)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RockType":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown rock type keys: {sorted(unknown)}")
        return cls(**d)
