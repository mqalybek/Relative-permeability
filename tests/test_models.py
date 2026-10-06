import numpy as np
import pytest

from relperm import LET, RockType, fit_corey


@pytest.fixture
def rt():
    return RockType()


def test_endpoints(rt):
    assert rt.krw(rt.swcr) == 0
    assert rt.krw(1 - rt.sowcr) == pytest.approx(rt.krwr)
    assert rt.krow(rt.swl) == pytest.approx(rt.krocw)
    assert rt.krow(1 - rt.sowcr) == 0
    assert rt.krg(rt.sgcr) == 0
    assert rt.krg(1 - rt.swl - rt.sogcr) == pytest.approx(rt.krgr)
    assert rt.krog(0.0) == pytest.approx(rt.krogcg)
    assert rt.krog(1 - rt.swl - rt.sogcr) == 0


def test_no_nan_or_negative_beyond_residual(rt):
    # The original script produced NaN/negative kro at So < Sor and patched it by hand.
    s = np.linspace(-0.1, 1.1, 241)
    for k in (rt.krw(s), rt.krow(s), rt.krg(s), rt.krog(s)):
        assert np.all(np.isfinite(k))
        assert np.all(k >= 0)


def test_tables_are_monotonic_and_start_correctly(rt):
    wo = rt.water_oil_table(20)
    assert wo["Sw"][0] == pytest.approx(rt.swl)
    assert wo["Sw"][-1] == pytest.approx(1.0)
    assert wo["krw"][0] == 0
    assert np.all(np.diff(wo["Sw"]) > 0)
    assert np.all(np.diff(wo["krw"]) >= 0)
    assert np.all(np.diff(wo["krow"]) <= 0)
    assert wo["krow"][-1] == 0

    go = rt.gas_oil_table(20)
    assert go["Sg"][0] == pytest.approx(rt.sgl)
    assert go["Sg"][-1] == pytest.approx(1 - rt.swl)
    assert np.all(np.diff(go["krg"]) >= 0)
    assert np.all(np.diff(go["krog"]) <= 0)
    assert go["krog"][-1] == 0


def test_critical_saturations_are_table_breakpoints():
    rt = RockType(swl=0.15, swcr=0.22, sgcr=0.05)
    assert np.any(np.isclose(rt.sw_grid(10), 0.22))
    assert np.any(np.isclose(rt.sw_grid(10), 1 - rt.sowcr))
    assert np.any(np.isclose(rt.sg_grid(10), 0.05))
    assert rt.krw(0.2) == 0 and rt.krg(0.04) == 0


def test_tail_to_krwmax():
    rt = RockType(krwr=0.4, krwmax=1.0)
    assert rt.krw(1.0) == pytest.approx(1.0)
    assert rt.krw(1 - rt.sowcr / 2) == pytest.approx(0.7)


def test_validation():
    with pytest.raises(ValueError, match="swcr must be >= swl"):
        RockType(swl=0.2, swcr=0.1)
    with pytest.raises(ValueError):
        RockType(krwr=1.5)
    with pytest.raises(ValueError):
        RockType(nw=-1)


def test_warning_for_inconsistent_kro_endpoint():
    assert RockType().warnings() == []
    assert any("krocw" in w for w in RockType(krocw=0.8, krogcg=0.9).warnings())


def test_let_shape():
    rt = RockType(nw={"L": 2, "E": 1, "T": 1})
    assert isinstance(rt.nw, LET)
    assert rt.krw(1 - rt.sowcr) == pytest.approx(rt.krwr)
    assert rt.krw(rt.swcr) == 0
    assert RockType.from_dict(rt.to_dict()) == rt


def test_stone2_reduces_to_two_phase(rt):
    sw = np.linspace(rt.swl, 1 - rt.sowcr, 11)
    np.testing.assert_allclose(rt.kro_stone2(sw, 0.0), rt.krow(sw), atol=1e-12)
    sg = np.linspace(0, 1 - rt.swl - rt.sogcr, 11)
    np.testing.assert_allclose(rt.kro_stone2(rt.swl, sg), rt.krog(sg), atol=1e-12)


def test_fractional_flow(rt):
    assert rt.fractional_flow(rt.swl) == 0
    assert rt.fractional_flow(1 - rt.sowcr) == pytest.approx(1.0)


def test_fit_recovers_parameters():
    s = np.linspace(0.2, 0.85, 12)
    sn = (s - 0.2) / (0.85 - 0.2)
    kr = 0.45 * sn**2.7
    res = fit_corey(s, kr, 0.2, 0.85)
    assert res.kr_max == pytest.approx(0.45, rel=1e-6)
    assert res.n == pytest.approx(2.7, rel=1e-6)
    fixed = fit_corey(s, kr, 0.2, 0.85, kr_max=0.45)
    assert fixed.n == pytest.approx(2.7, rel=1e-6)
