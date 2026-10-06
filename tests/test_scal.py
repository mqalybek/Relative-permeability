import json
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

from relperm.cli import load_config, main  # noqa: E402
from relperm.scal import (ALIAS_INDEX, _norm, _perm_label, build_rock_types,  # noqa: E402
                          fit_exponent, read_scal)

EXAMPLES = Path(__file__).parent.parent / "examples"


def _exact_water_oil(sample, k, swi, sorw, krwr, krocw, nw, now, rock_type=None):
    sw = np.linspace(swi, 1 - sorw, 9)
    sn = (sw - swi) / (1 - swi - sorw)
    df = pd.DataFrame({"sample": sample, "k, mD": k, "Sw": sw,
                       "krw": krwr * sn**nw, "kro": krocw * (1 - sn) ** now})
    if rock_type:
        df.insert(1, "rock_type", rock_type)
    return df


@pytest.mark.parametrize("header, canon", [
    ("Kв, %", "sw"),            # Latin K + Cyrillic в
    ("Кв, д.ед.", "sw"),
    ("Sw", "sw"),
    ("ОФП воды, д.ед.", "krw"),
    ("Krow", "kro"),
    ("Кпр, мД", "perm"),
    ("k (mD)", "perm"),
    ("№ образца", "sample"),
    ("Тип породы", "rock_type"),
    ("Кво", "swi"),
    ("Sg, д.ед.", "sg"),
])
def test_column_aliases(header, canon):
    assert ALIAS_INDEX[_norm(header)] == canon


def test_fit_exponent_exact():
    sn = np.linspace(0, 1, 11)
    n, rmse, used = fit_exponent(sn, sn**2.7)
    assert n == pytest.approx(2.7, rel=1e-4)
    assert rmse < 1e-6 and used == 9
    assert np.isnan(fit_exponent([0, 1], [0, 1])[0])


def test_recovers_exact_parameters(tmp_path):
    df = pd.concat([
        _exact_water_oil("A", 100, 0.2, 0.2, 0.4, 0.9, 2.5, 2.0),
        _exact_water_oil("B", 300, 0.24, 0.18, 0.5, 0.9, 2.5, 2.0),
    ])
    path = tmp_path / "wo.csv"
    df.to_csv(path, index=False)
    res = build_rock_types(read_scal(str(path)))
    (rt,) = res.rock_types
    assert rt.swl == pytest.approx(0.22) and rt.swcr == rt.swl
    assert rt.sowcr == pytest.approx(0.19)
    assert rt.krwr == pytest.approx(0.45) and rt.krocw == pytest.approx(0.9)
    assert rt.nw == pytest.approx(2.5, rel=1e-3) and rt.now == pytest.approx(2.0, rel=1e-3)
    assert rt.krogcg == pytest.approx(rt.krocw)  # no gas-oil data -> aligned default
    assert any("no gas-oil data" in w for w in res.warnings)


def test_decimal_comma_semicolon_csv(tmp_path):
    df = _exact_water_oil("A", 100, 0.2, 0.2, 0.4, 0.9, 3.0, 2.0)
    path = tmp_path / "ru.csv"
    path.write_text(df.to_csv(sep=";", index=False, decimal=","), encoding="utf-8")
    (rt,) = build_rock_types(read_scal(str(path))).rock_types
    assert rt.nw == pytest.approx(3.0, rel=1e-3)


def test_lab_report_xlsx():
    data = read_scal(str(EXAMPLES / "scal_lab_report.xlsx"))
    kinds = {sheet: kind for _, sheet, kind in data.sources}
    assert kinds["Образцы"] == "meta" and kinds["101"] == "wo" and kinds["Газ-нефть"] == "go"
    wo, go = data.water_oil, data.gas_oil
    assert wo["sw"].max() <= 1.0  # percent converted
    assert set(wo["sample"]) == {"101", "102", "103", "104", "201", "202", "203", "204"}
    assert wo["perm"].notna().all() and wo["rock_type"].notna().all()  # merged from metadata
    assert go["sample"].notna().all()  # sample ID written once per block is filled down

    res = build_rock_types(data)
    names = {rt.name for rt in res.rock_types}
    assert names == {"Песчаник", "Глинистый"}
    sand = next(rt for rt in res.rock_types if rt.name == "Песчаник")
    # synthetic data were generated with nw=2.5, now=2.2, ng=2.0, nog=1.8
    assert sand.nw == pytest.approx(2.5, abs=0.3)
    assert sand.now == pytest.approx(2.2, abs=0.3)
    assert sand.ng == pytest.approx(2.0, abs=0.3)
    assert sand.nog == pytest.approx(1.8, abs=0.3)
    assert sand.krogcg == sand.krocw
    swl = next(c for c in res.correlations if c["endpoint"] == "swl")
    assert swl["b"] == pytest.approx(-0.075, abs=0.015) and swl["r2"] > 0.9
    assert not any(c["endpoint"] == "sgcr" for c in res.correlations)  # constant -> skipped


def test_no_align_keeps_gas_oil_endpoint():
    data = read_scal(str(EXAMPLES / "scal_lab_report.xlsx"))
    sand = next(rt for rt in build_rock_types(data, align_kro=False).rock_types
                if rt.name == "Песчаник")
    assert sand.krogcg != sand.krocw


def test_perm_bins():
    assert _perm_label(5, [10, 100]) == "k<10"
    assert _perm_label(50, [10, 100]) == "k10-100"
    assert _perm_label(500, [10, 100]) == "k>=100"
    assert _perm_label(float("nan"), [10]) == "unknown_k"
    res = build_rock_types(read_scal(str(EXAMPLES / "scal_water_oil.csv")),
                           group_by="perm", perm_bins=[50])
    assert {rt.name for rt in res.rock_types} == {"k<50", "k>=50"}


def test_cli_scal(tmp_path):
    out = tmp_path / "out"
    main(["scal", str(EXAMPLES / "scal_lab_report.xlsx"), "-o", str(out), "-f", "cmg"])
    for name in ("rock_types.json", "samples.csv", "correlations.csv", "relperm_cmg.dat",
                 "qc_Песчаник.png", "lab_Песчаник_wo.csv", "endpoints_vs_perm.png"):
        assert (out / name).exists(), name
    rock_types, _ = load_config(str(out / "rock_types.json"))
    assert len(rock_types) == 2
    samples = pd.read_csv(out / "samples.csv")
    assert len(samples) == 12  # 8 water-oil + 4 gas-oil tests
    assert json.loads((out / "rock_types.json").read_text(encoding="utf-8"))["points"] == 20
