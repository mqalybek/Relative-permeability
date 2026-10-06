"""Write saturation tables as Eclipse keywords (Eclipse, tNavigator, OPM Flow, Petrel import)."""

from __future__ import annotations

from typing import Iterable, Sequence

from .models import RockType

FORMATS = ("eclipse", "eclipse2")


def _rows(columns: Sequence, fmt: str = "{:12.6f}") -> list[str]:
    return ["  " + " ".join(fmt.format(v) for v in row) for row in zip(*columns)]


def _eclipse_keyword(keyword: str, header: str, rock_types, table_fn, keys, points) -> str:
    lines = [keyword]
    for rt in rock_types:
        lines.append(f"-- {rt.name}{': ' + rt.description if rt.description else ''}")
        lines.append(f"-- {header.format(u=rt.pc_units)}")
        t = table_fn(rt, points)
        lines += _rows([t[k] for k in keys])
        lines.append("/")
    return "\n".join(lines) + "\n"


def swof(rock_types: Iterable[RockType], points: int = 20) -> str:
    return _eclipse_keyword(
        "SWOF", "      Sw          krw          krow         Pcow({u})",
        rock_types, RockType.water_oil_table, ("Sw", "krw", "krow", "Pcow"), points,
    )


def sgof(rock_types: Iterable[RockType], points: int = 20) -> str:
    return _eclipse_keyword(
        "SGOF", "      Sg          krg          krog         Pcog({u})",
        rock_types, RockType.gas_oil_table, ("Sg", "krg", "krog", "Pcog"), points,
    )


def swfn(rock_types: Iterable[RockType], points: int = 20) -> str:
    return _eclipse_keyword(
        "SWFN", "      Sw          krw          Pcow({u})",
        rock_types, RockType.water_oil_table, ("Sw", "krw", "Pcow"), points,
    )


def sgfn(rock_types: Iterable[RockType], points: int = 20) -> str:
    return _eclipse_keyword(
        "SGFN", "      Sg          krg          Pcog({u})",
        rock_types, RockType.gas_oil_table, ("Sg", "krg", "Pcog"), points,
    )


def sof3(rock_types: Iterable[RockType], points: int = 20) -> str:
    return _eclipse_keyword(
        "SOF3", "      So          krow         krog",
        rock_types, RockType.oil_table, ("So", "krow", "krog"), points,
    )


def export(rock_types: Sequence[RockType], fmt: str = "eclipse", points: int = 20) -> str:
    """Render all rock types in one of FORMATS.

    eclipse  -> SWOF + SGOF (family I)
    eclipse2 -> SWFN + SGFN + SOF3 (family II)
    """
    rock_types = list(rock_types)
    if fmt == "eclipse":
        return swof(rock_types, points) + "\n" + sgof(rock_types, points)
    if fmt == "eclipse2":
        return "\n".join(f(rock_types, points) for f in (swfn, sgfn, sof3))
    raise ValueError(f"Unknown format '{fmt}', expected one of {FORMATS}")
