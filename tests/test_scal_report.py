"""A workbook laid out like a real Russian SCAL report (see scal_layout.py)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

sys.path.insert(0, str(Path(__file__).parent))
from scal_layout import KR_SAMPLES, PC_SAMPLES, TRUTH, TYPO_SAMPLE, build  # noqa: E402

from relperm.capillary import LeverettJ  # noqa: E402
from relperm.cli import load_config, main  # noqa: E402
from relperm.dependencies import fit_dependencies, find_anomalies, property_table  # noqa: E402
from relperm.scal import (PcOptions, _find_tables, _match_header, _sample_id,  # noqa: E402
                          build_rock_types, fit_j, read_scal, write_report)

PC = PcOptions(lab_units="MPa", system="oil-brine")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return str(build(tmp_path_factory.mktemp("report") / "report.xlsx"))


@pytest.fixture(scope="module")
def result(report, tmp_path_factory):
    res = build_rock_types(read_scal(report), pc=PC)
    import matplotlib

    matplotlib.use("Agg")
    write_report(res, str(tmp_path_factory.mktemp("out")), pc=PC)
    return res


def _fit(res, y, x=None):
    return next(f for f in res.dependencies["fits"] if f["y"] == y and (x is None or f["x"] == x))


@pytest.mark.parametrize("header, canon", [
    ("Кпр. мД", "perm"), ("Кп %", "poro"), ("Пористость. доли ед.", "poro"),
    ("Котн. воды. доли ед.", "krw"), ("Котн. нефти. доли ед.", "kro"),
    ("Проницаемость для газа *10-3 . мкм2  ", "perm"),
    ("Проницаемость по пластовой воде, мД", "perm_w"),
    ("Остаточная нефтенасыщенность, Sвост д. ед.", "sor"),
    ("Коэффициент вытеснения, β, д. ед.", "kdisp"),
    ("Показатель смачиваемости Амотта", "amott"), ("Рс", "pc"), ("Pc_Mpa", "pc"),
    ("№ скв.", "well"), ("Горизонт", "horizon"), ("Глубина (привязанная глубина), м ", "depth"),
    ("Тип смачиваемости", None), ("pc_res_atm", None), ("Длина модели L, см", None),
])
def test_report_headers(header, canon):
    assert _match_header(header)[0] == canon


def test_sample_ids_are_matched_across_sheets():
    # Cyrillic Р/Т and Latin P/T in the same ID; "обр." prefix with Кп/Кпр after a comma
    assert _sample_id("011004010РT01H") == _sample_id("011004010РТ01H") == "011004010PT01H"
    assert _sample_id("обр. 011004036РT03H, Кп=15,2%, Кпр=1,75мД") == "011004036PT03H"
    assert _sample_id("обр, 011004050РT03H") == "011004050PT03H"
    assert _sample_id(31.0) == "31" and _sample_id("Песчаник") == "Песчаник"


def test_tables_found(report):
    raw = pd.read_excel(report, sheet_name="ОФП", header=None, dtype=object)
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    raw.columns = range(raw.shape[1])
    summary, curves = _find_tables(raw)
    assert {c for c, _ in curves["cols"].values()} == {"sw", "krw", "kro", "sample", "well"}
    assert summary["end"] == curves["row"]  # the curve header ends the summary table

    kinds = {sheet: kind for _, sheet, kind in read_scal(report).sources}
    assert kinds == {"капл давл #1": "pc", "капл давл #2": "skipped (no sample column)",
                     "ОФП #1": "meta", "ОФП #2": "wo", "Коэф.вытес.": "meta", "смачив.": "meta"}


def test_report_contents(report):
    data = read_scal(report)
    wo, pc, meta = data.water_oil, data.capillary, data.samples
    assert wo["sample"].nunique() == len(KR_SAMPLES)
    assert pc["sample"].nunique() == len(PC_SAMPLES)
    assert pc["sw"].max() == pytest.approx(1.0)  # Кв in percent converted
    assert pc["poro"].max() < 0.3  # Кп % converted
    assert wo["well"].notna().all() and set(wo["well"]) == {"31", "32", "33"}
    # properties from three sheets merged under one ID
    s = meta.loc[_sample_id(KR_SAMPLES[0][2])]
    assert {"perm", "perm_w", "poro", "swi", "sor", "kdisp", "amott", "krw"} <= set(s.dropna().index)
    assert "Скважина №31" not in meta.index
    assert [d["property"] for d in data.discrepancies] == ["poro"]


def test_anomalies_are_flagged_and_excluded(result):
    props = result.dependencies["properties"]
    anomalies = find_anomalies(props, result.data.discrepancies)
    typo = _sample_id(TYPO_SAMPLE)
    assert {(a["sample"], a["property"]) for a in anomalies} == {(typo, "poro")}
    fit = _fit(result, "perm", "poro")
    assert fit["n"] == len(KR_SAMPLES) - 1 + len(PC_SAMPLES)  # the typo is not in the fit


def test_dependencies_recover_truth(result):
    lg = lambda key: TRUTH[key]  # noqa: E731
    a, b = lg("lg_k_poro")
    f = _fit(result, "perm", "poro")
    assert f["y_log"] and f["a"] == pytest.approx(a, abs=0.02) and f["b"] == pytest.approx(b, rel=0.01)
    a, b = lg("lg_kw")
    f = _fit(result, "perm_w", "perm")
    assert f["a"] == pytest.approx(a, abs=0.005) and f["b"] == pytest.approx(b, rel=0.005)
    # Swi and Sor are generated against lg Kw; with lg Kw = a + b lg Kg they are linear in lg Kg
    for key, y in (("swi", "swi"), ("sor", "sor")):
        f = _fit(result, y)
        a0, b0 = TRUTH[key]
        expected_b = b0 * (TRUTH["lg_kw"][1] if f["x"] == "perm" else 1)
        assert f["b"] == pytest.approx(expected_b, rel=0.02) and f["r2"] > 0.999
    a, b = lg("swirr_pc")
    f = _fit(result, "swirr_pc")
    assert f["a"] == pytest.approx(a, abs=1e-3) and f["b"] == pytest.approx(b, abs=1e-3)
    assert not any(f["y"] == "krw_end" for f in result.dependencies["fits"])  # constant: skipped


def test_exponential_j_recovered(result):
    (rt,) = result.rock_types
    assert isinstance(rt.pcow, LeverettJ) and rt.pcow.form == "exp"
    a, b = TRUTH["J_exp"]
    assert rt.pcow.a == pytest.approx(a, rel=1e-3) and rt.pcow.b == pytest.approx(b, rel=1e-3)
    fits = result.pc_fits["RT1"]
    assert fits["exp"]["rmse_swn"] < fits["power"]["rmse_swn"]


def test_fit_j_power_and_exp():
    sn = np.linspace(0.05, 0.95, 12)
    f = fit_j(sn, 0.3 * sn**-1.4, "power")
    assert (f["a"], f["b"]) == (pytest.approx(0.3), pytest.approx(1.4))
    f = fit_j(sn, 20 * np.exp(-4 * sn), "exp")
    assert (f["a"], f["b"]) == (pytest.approx(20), pytest.approx(4))
    assert f["rmse_swn"] < 1e-9
    with pytest.raises(ValueError):
        fit_j(sn, np.exp(sn), "exp")  # J growing with Sw: not a drainage curve


def test_leverett_exp_model():
    m = LeverettJ(a=20, b=4, perm=10, poro=0.2, ift=30, theta=30, form="exp")
    assert m.j(0) == pytest.approx(20) and m.j(1) == pytest.approx(20 * math.exp(-4))
    assert m.pc(0.5, "bar") == pytest.approx(20 * math.exp(-2) * m.factor("bar"))
    with pytest.raises(ValueError):
        LeverettJ(a=1, b=1, perm=1, poro=0.2, form="log")


def test_normalized_property_table(result):
    props = property_table(result)
    row = props.set_index("sample").loc[_sample_id(KR_SAMPLES[2][2])]
    assert row["kdisp"] == pytest.approx((1 - row["swi"] - row["sor"]) / (1 - row["swi"]), abs=1e-3)
    assert row["nw"] == pytest.approx(TRUTH["nw"], rel=1e-3)
    assert row["now"] == pytest.approx(TRUTH["now"], rel=1e-3)


def test_excel_report(result, tmp_path):
    out = tmp_path / "out"
    write_report(result, str(out), pc=PC)
    xl = pd.ExcelFile(out / "scal_report.xlsx")
    assert {"Образцы", "Зависимости", "Аномалии", "ОФП нормированные", "Кап. давление",
            "Типы пород", "J-функция"} <= set(xl.sheet_names)
    kr = xl.parse("ОФП нормированные")
    assert kr["Swn"].min() == pytest.approx(0) and kr["Swn"].max() == pytest.approx(1)
    assert kr["krw_n"].max() == pytest.approx(1) and kr["kro_n"].max() == pytest.approx(1)
    deps = xl.parse("Зависимости")
    assert deps["Зависимость"].str.contains("Кво").any()
    pc = xl.parse("Кап. давление")
    assert {"Swn", "J", "Pc lab, bar", "Pc res, bar"} <= set(pc.columns)
    for name in ("dependencies.png", "properties.csv", "anomalies.csv", "qc_pc_RT1.png"):
        assert (out / name).exists(), name


def test_cli_on_report(report, tmp_path, capsys):
    out = tmp_path / "cli"
    main(["scal", report, "-o", str(out), "--pc-lab-units", "MPa", "--pc-system", "oil-brine",
          "--group-by", "well", "-f", "eclipse"])
    text = capsys.readouterr().out
    assert "Dependencies" in text and "Кво" in text and "exp" in text
    rock_types, _ = load_config(str(out / "rock_types.json"))
    assert {rt.name for rt in rock_types} == {"well 31", "well 32", "well 33"}
    assert "SWOF" in (out / "relperm_eclipse.inc").read_text(encoding="utf-8")
