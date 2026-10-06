import json
from pathlib import Path

import numpy as np
import pytest

from relperm import BrooksCorey, LeverettJ, RockType, export, fit_brooks_corey, height_above_fwl
from relperm.capillary import SN_FLOOR
from relperm.cli import main


def test_no_pc_by_default():
    rt = RockType()
    assert np.all(rt.water_oil_table()["Pcow"] == 0)
    assert np.all(rt.gas_oil_table()["Pcog"] == 0)


def test_brooks_corey_pcow():
    rt = RockType(pcow={"model": "brooks-corey", "pe": 0.1, "lam": 2.0, "pcmax": 3.0})
    assert isinstance(rt.pcow, BrooksCorey)
    assert rt.pc_ow(rt.swl) == pytest.approx(3.0)            # capped
    assert rt.pc_ow(1 - rt.sowcr) == pytest.approx(0.1)      # entry pressure
    assert rt.pc_ow(1.0) == pytest.approx(0.1)
    sn = 0.25
    sw = rt.swl + sn * (1 - rt.swl - rt.sowcr)
    assert rt.pc_ow(sw) == pytest.approx(0.1 * sn ** -0.5)
    pc = rt.water_oil_table(30)["Pcow"]
    assert np.all(np.isfinite(pc)) and np.all(np.diff(pc) <= 0)  # SWOF: non-increasing


def test_uncapped_uses_floor():
    rt = RockType(pcow=BrooksCorey(pe=0.1, lam=2.0))
    assert rt.pc_ow(rt.swl) == pytest.approx(0.1 * SN_FLOOR ** -0.5)


def test_zero_entry_pressure_is_zero_everywhere():
    rt = RockType(pcow=BrooksCorey(pe=0.0, lam=2.0, pcmax=0.0))
    assert np.all(rt.pc_ow(np.linspace(0, 1, 11)) == 0)


def test_pcog_increases_with_gas():
    rt = RockType(pcog={"pe": 0.02, "lam": 2.5, "pcmax": 1.0})  # model defaults to Brooks-Corey
    go = rt.gas_oil_table(30)
    assert go["Pcog"][0] == pytest.approx(0.02)
    assert go["Pcog"][-1] == pytest.approx(1.0)
    assert np.all(np.diff(go["Pcog"]) >= 0)  # SGOF: non-decreasing


def test_leverett_scaling():
    lj = LeverettJ(a=0.3, b=1.5, perm=100, poro=0.2, ift=25, theta=0)
    # Pc = J * sigma / sqrt(k/phi): 0.3 * 0.025 N/m / sqrt(100 mD * 9.869e-16 / 0.2) = 10.68 kPa
    assert lj.pc(1.0, "kPa") == pytest.approx(10.676, rel=1e-3)
    assert lj.pc(1.0, "bar") == pytest.approx(0.10676, rel=1e-3)
    assert lj.pc(1.0, "psi") == pytest.approx(1.5484, rel=1e-3)
    # tighter rock -> higher Pc, by sqrt of the permeability ratio
    tight = LeverettJ(a=0.3, b=1.5, perm=1, poro=0.2)
    assert tight.pc(0.5) / lj.pc(0.5) == pytest.approx(10.0)


def test_height_above_fwl():
    # 1 bar with 250 kg/m3 -> 1e5 / (250 * 9.80665) = 40.8 m
    assert height_above_fwl(1.0, 250, "bar") == pytest.approx(40.79, rel=1e-3)
    rt = RockType(pcow=BrooksCorey(pe=0.1, lam=2, pcmax=3))
    assert rt.height_above_fwl(1 - rt.sowcr, 250) == pytest.approx(4.079, rel=1e-3)


def test_validation():
    with pytest.raises(ValueError):
        BrooksCorey(pe=0.5, lam=2, pcmax=0.1)
    with pytest.raises(ValueError):
        LeverettJ(a=0.3, b=1.5, perm=100, poro=1.5)
    with pytest.raises(ValueError, match="Unknown Pc model"):
        RockType(pcow={"model": "thomeer", "pe": 1})
    with pytest.raises(ValueError, match="pressure units"):
        RockType(pc_units="MPa")


def test_roundtrip_and_export_units():
    rt = RockType(pc_units="psi",
                  pcow={"model": "leverett", "a": 0.3, "b": 1.5, "perm": 50, "poro": 0.2},
                  pcog=BrooksCorey(pe=0.3, lam=2, pcmax=15))
    d = json.loads(json.dumps(rt.to_dict()))
    assert d["pcow"]["model"] == "leverett"
    assert RockType.from_dict(d) == rt
    text = export([rt], "eclipse", 10)
    assert "Pcow(psi)" in text and "Pcog(psi)" in text
    assert "Pcog(psi)" in export([rt], "cmg", 10)


def test_fit_brooks_corey():
    sw = np.linspace(0.25, 0.85, 10)
    sn = (sw - 0.2) / (0.9 - 0.2)
    pc = 0.08 * sn ** (-1 / 1.8)
    res = fit_brooks_corey(sw, pc, 0.2, 0.9)
    assert res.pe == pytest.approx(0.08, rel=1e-6)
    assert res.lam == pytest.approx(1.8, rel=1e-6)
    with pytest.raises(ValueError):
        fit_brooks_corey(sw, 1 / pc, 0.2, 0.9)


def test_cli_fit_pc(capsys):
    lab = Path(__file__).parent.parent / "examples" / "lab_water_oil.csv"
    main(["fit", str(lab), "--phase", "pcow", "--swl", "0.2", "--sorw", "0.15"])
    out = capsys.readouterr().out
    assert "pe" in out and "lam" in out
