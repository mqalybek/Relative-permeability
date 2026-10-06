"""Write saturation tables in Eclipse / OPM / tNavigator and CMG formats."""

from __future__ import annotations

from typing import Iterable, Sequence

from .models import RockType

FORMATS = ("eclipse", "eclipse2", "cmg")


def _rows(columns: Sequence, fmt: str = "{:12.6f}") -> list[str]:
    return ["  " + " ".join(fmt.format(v) for v in row) for row in zip(*columns)]


def _eclipse_keyword(keyword: str, header: str, rock_types, table_fn, keys, points) -> str:
    lines = [keyword]
    for rt in rock_types:
        lines.append(f"-- {rt.name}{': ' + rt.description if rt.description else ''}")
        lines.append(f"-- {header}")
        t = table_fn(rt, points)
        lines += _rows([t[k] for k in keys])
        lines.append("/")
    return "\n".join(lines) + "\n"


def swof(rock_types: Iterable[RockType], points: int = 20) -> str:
    return _eclipse_keyword(
        "SWOF", "      Sw          krw          krow         Pcow",
        rock_types, RockType.water_oil_table, ("Sw", "krw", "krow", "Pcow"), points,
    )


def sgof(rock_types: Iterable[RockType], points: int = 20) -> str:
    return _eclipse_keyword(
        "SGOF", "      Sg          krg          krog         Pcog",
        rock_types, RockType.gas_oil_table, ("Sg", "krg", "krog", "Pcog"), points,
    )


def swfn(rock_types: Iterable[RockType], points: int = 20) -> str:
    return _eclipse_keyword(
        "SWFN", "      Sw          krw          Pcow",
        rock_types, RockType.water_oil_table, ("Sw", "krw", "Pcow"), points,
    )


def sgfn(rock_types: Iterable[RockType], points: int = 20) -> str:
    return _eclipse_keyword(
        "SGFN", "      Sg          krg          Pcog",
        rock_types, RockType.gas_oil_table, ("Sg", "krg", "Pcog"), points,
    )


def sof3(rock_types: Iterable[RockType], points: int = 20) -> str:
    return _eclipse_keyword(
        "SOF3", "      So          krow         krog",
        rock_types, RockType.oil_table, ("So", "krow", "krog"), points,
    )


def cmg(rock_types: Iterable[RockType], points: int = 20) -> str:
    """CMG IMEX/GEM/STARS *ROCKFLUID section with *SWT and *SLT tables."""
    lines = ["*ROCKFLUID"]
    for i, rt in enumerate(rock_types, start=1):
        lines.append(f"*RPT {i}")
        lines.append(f"** {rt.name}{': ' + rt.description if rt.description else ''}")
        wo = rt.water_oil_table(points)
        lines.append("*SWT")
        lines.append("**     Sw          krw          krow         Pcow")
        lines += _rows([wo["Sw"], wo["krw"], wo["krow"], wo["Pcow"]])
        # CMG tabulates gas-liquid data against liquid saturation, ascending.
        go = rt.gas_oil_table(points)
        sl = 1.0 - go["Sg"][::-1]
        lines.append("*SLT")
        lines.append("**     Sl          krg          krog         Pcog")
        lines += _rows([sl, go["krg"][::-1], go["krog"][::-1], go["Pcog"][::-1]])
        lines.append("")
    return "\n".join(lines)


def export(rock_types: Sequence[RockType], fmt: str = "eclipse", points: int = 20) -> str:
    """Render all rock types in one of FORMATS.

    eclipse  -> SWOF + SGOF (family I; also OPM Flow, tNavigator)
    eclipse2 -> SWFN + SGFN + SOF3 (family II)
    cmg      -> *ROCKFLUID with *SWT / *SLT
    """
    rock_types = list(rock_types)
    if fmt == "eclipse":
        return swof(rock_types, points) + "\n" + sgof(rock_types, points)
    if fmt == "eclipse2":
        return "\n".join(f(rock_types, points) for f in (swfn, sgfn, sof3))
    if fmt == "cmg":
        return cmg(rock_types, points)
    raise ValueError(f"Unknown format '{fmt}', expected one of {FORMATS}")
