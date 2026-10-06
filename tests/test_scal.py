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
    assert len(samples) == 17  # 8 water-oil + 4 gas-oil + 5 capillary pressure tests
    assert set(samples["test"]) == {"water-oil", "gas-oil", "capillary"}
    assert json.loads((out / "rock_types.json").read_text(encoding="utf-8"))["points"] == 20


# ------------------------------------------------------------- capillary pressure

from relperm.capillary import LeverettJ  # noqa: E402
from relperm.scal import PcOptions, _pc_unit_factor, _strip_system_words, lab_system  # noqa: E402


@pytest.mark.parametrize("header, factor", [
    ("Pc, psi", 0.0689476), ("Рк, МПа", 10.0), ("Pc (kPa)", 0.01), ("Рк, атм", 1.01325),
    ("Pc, кгс/см2", 0.980665), ("Pc, bar", 1.0), ("Pc", None),
])
def test_pc_units_from_header(header, factor):
    assert _pc_unit_factor(header) == (pytest.approx(factor) if factor else None)


@pytest.mark.parametrize("text, system", [
    ("101 MICP", "mercury-air"), ("Ртуть-воздух", "mercury-air"), ("центрифуга", "air-brine"),
    ("газ-вода", "air-brine"), ("нефть-вода", "oil-brine"), ("Air-Brine", "air-brine"), ("Pc", None),
])
def test_lab_system_keywords(text, system):
    assert lab_system(text) == system


def test_sheet_names_map_to_samples():
    assert [_strip_system_words(x) for x in ("101 MICP", "MICP_101", "Рк 203 центрифуга")] == \
        ["101", "101", "203"]


def _pc_csv(tmp_path, system_value, units, sigma_lab):
    """Exact J = 0.15 Sn^-1.2 at reservoir sigma*cos = 30*cos30, written as lab Pc."""
    k, phi, swi, sorw = 100.0, 0.2, 0.2, 0.2
    sw = np.linspace(0.25, 0.75, 8)
    sn = (sw - swi) / (1 - swi - sorw)
    sigma_res = 30 * np.cos(np.radians(30))
    pc_res_pa = 0.15 * sn**-1.2 * sigma_res * 1e-3 / np.sqrt(k * 9.869233e-16 / phi)
    pc_lab = pc_res_pa * sigma_lab / sigma_res / {"psi": 6894.757, "kPa": 1e3}[units]
    df = pd.DataFrame({"sample": "S1", "k, mD": k, "poro": phi, "Swi": swi,
                       "system": system_value, "Sw": sw, f"Pc, {units}": pc_lab})
    wo = _exact_water_oil("S1", k, swi, sorw, 0.4, 0.9, 2.0, 2.0)
    p1, p2 = tmp_path / "pc.csv", tmp_path / "wo.csv"
    df.to_csv(p1, index=False)
    wo.to_csv(p2, index=False)
    return str(p1), str(p2)


@pytest.mark.parametrize("system, units, sigma_lab", [
    ("MICP", "psi", 480 * abs(np.cos(np.radians(140)))),
    ("air-brine", "kPa", 72.0),
])
def test_pc_recovers_leverett_j(tmp_path, system, units, sigma_lab):
    res = build_rock_types(read_scal(*_pc_csv(tmp_path, system, units, sigma_lab)))
    (rt,) = res.rock_types
    assert isinstance(rt.pcow, LeverettJ)
    assert rt.pcow.a == pytest.approx(0.15, rel=1e-3)
    assert rt.pcow.b == pytest.approx(1.2, rel=1e-3)
    assert rt.pcow.perm == pytest.approx(100) and rt.pcow.poro == pytest.approx(0.2)
    assert rt.pc_units == "bar"
    # rock type Pc at the measured saturations equals the reservoir Pc
    sw = 0.2 + 0.3 * 0.6
    expected = 0.15 * 0.3**-1.2 * LeverettJ(1, 1, 100, 0.2, 30, 30).factor("bar")
    assert rt.pc_ow(sw) == pytest.approx(expected, rel=1e-3)
    assert "Pcow(bar)" in __import__("relperm").export([rt], "eclipse")


def test_pc_without_perm_falls_back_to_brooks_corey(tmp_path):
    sw = np.linspace(0.3, 0.7, 6)
    pd.DataFrame({"sample": "S1", "Sw": sw, "Pc, bar": 0.1 * ((sw - 0.2) / 0.6) ** -0.5}) \
        .to_csv(tmp_path / "pc.csv", index=False)
    pd.DataFrame({"sample": "S1", "Sw": [0.2, 0.5, 0.8], "krw": [0, 0.1, 0.4],
                  "kro": [0.9, 0.2, 0]}).to_csv(tmp_path / "wo.csv", index=False)
    res = build_rock_types(read_scal(str(tmp_path / "pc.csv"), str(tmp_path / "wo.csv")),
                           pc=PcOptions(system="oil-brine", ift_res=48, theta_res=30))
    (rt,) = res.rock_types
    assert rt.pcow.model == "brooks-corey"
    assert rt.pcow.pe == pytest.approx(0.1, rel=1e-3) and rt.pcow.lam == pytest.approx(2.0, rel=1e-3)
    notes = [n for s in res.samples for n in s.notes]
    assert any("permeability" in n for n in notes)


def test_pc_unknown_units_and_system_are_reported(tmp_path):
    pd.DataFrame({"sample": "S1", "k": 50, "poro": 0.2, "Sw": [0.3, 0.5, 0.7],
                  "Pc": [1.0, 0.5, 0.3]}).to_csv(tmp_path / "pc.csv", index=False)
    res = build_rock_types(read_scal(str(tmp_path / "pc.csv")))
    notes = " ".join(n for s in res.samples for n in s.notes)
    assert "units not in header" in notes and "system unknown" in notes
    res = build_rock_types(read_scal(str(tmp_path / "pc.csv")),
                           pc=PcOptions(lab_units="psi", system="mercury-air"))
    notes = " ".join(n for s in res.samples for n in s.notes)
    assert "units" not in notes and "system" not in notes


def test_lab_report_capillary():
    data = read_scal(str(EXAMPLES / "scal_lab_report.xlsx"))
    cap = data.capillary
    assert set(cap["sample"]) == {"101", "102", "104", "202", "203"}
    systems = cap.groupby("sample")["pc_system"].first().to_dict()
    assert systems["101"] == "mercury-air" and systems["202"] == "air-brine"
    assert systems["104"] == "air-brine"  # from the "Система" column
    res = build_rock_types(data)
    sand = next(rt for rt in res.rock_types if rt.name == "Песчаник")
    shaly = next(rt for rt in res.rock_types if rt.name == "Глинистый")
    # generated with J = 0.12 Sn^-1.3 (sand) and 0.18 Sn^-1.1 (shaly) from MICP, centrifuge and plate
    assert sand.pcow.a == pytest.approx(0.12, rel=0.05) and sand.pcow.b == pytest.approx(1.3, rel=0.05)
    assert shaly.pcow.a == pytest.approx(0.18, rel=0.05) and shaly.pcow.b == pytest.approx(1.1, rel=0.05)
    wo = sand.water_oil_table(30)
    assert np.all(np.diff(wo["Pcow"]) <= 0) and wo["Pcow"][0] == pytest.approx(sand.pcow.pcmax)


def test_cli_scal_writes_pc_outputs(tmp_path):
    out = tmp_path / "out"
    main(["scal", str(EXAMPLES / "scal_lab_report.xlsx"), "-o", str(out), "--pc-units", "psi"])
    assert (out / "qc_pc_Песчаник.png").exists()
    lab = pd.read_csv(out / "lab_Песчаник_pc.csv")
    assert list(lab.columns) == ["Sw", "Pcow"]
    rock_types, _ = load_config(str(out / "rock_types.json"))
    assert all(rt.pc_units == "psi" and rt.pcow is not None for rt in rock_types)
