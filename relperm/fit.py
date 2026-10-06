"""Fit Corey endpoint and exponent to laboratory (SCAL) points.

kr = kr_max * sn**n  ->  ln kr = ln kr_max + n ln sn, so the fit is a linear
least-squares problem in log space and needs no optimizer.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from .models import normalize


@dataclass
class CoreyFit:
    kr_max: float
    n: float
    rmse: float  # in kr units, over the points used
    used: int


def fit_corey(s, kr, s_lo: float, s_hi: float, kr_max: float | None = None) -> CoreyFit:
    """Fit kr = kr_max * ((s - s_lo) / (s_hi - s_lo))**n.

    For an oil curve pass s = So (i.e. 1 - Sw), s_lo = Sor, s_hi = 1 - Swl.
    If kr_max is given (e.g. measured endpoint), only the exponent is fitted.
    """
    s = np.asarray(s, dtype=float)
    kr = np.asarray(kr, dtype=float)
    sn = normalize(s, s_lo, s_hi)
    mask = (sn > 0) & (sn < 1) & (kr > 0)
    if mask.sum() < (1 if kr_max is not None else 2):
        raise ValueError("Not enough points with 0 < sn < 1 and kr > 0 to fit")
    x, y = np.log(sn[mask]), np.log(kr[mask])
    if kr_max is None:
        n, ln_a = np.polyfit(x, y, 1)
        kr_max = float(np.exp(ln_a))
    else:
        n = float(np.sum(x * (y - np.log(kr_max))) / np.sum(x * x))
    pred = kr_max * normalize(s, s_lo, s_hi) ** n
    rmse = float(np.sqrt(np.mean((pred[mask] - kr[mask]) ** 2)))
    return CoreyFit(kr_max=float(kr_max), n=float(n), rmse=rmse, used=int(mask.sum()))


def read_columns(path: str) -> dict[str, np.ndarray]:
    """Read a CSV with a header row (comma, semicolon or tab separated)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        rows = list(csv.reader(f, dialect))
    header = [h.strip() for h in rows[0]]
    data = np.array([[float(v.replace(",", ".")) if dialect.delimiter != "," else float(v)
                      for v in r] for r in rows[1:] if any(c.strip() for c in r)])
    return {h: data[:, i] for i, h in enumerate(header)}
