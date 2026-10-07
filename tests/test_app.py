"""Editor window: file loading and navigation (headless, Agg backend)."""

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from relperm.app import RelPermApp, scal_session  # noqa: E402

REPORT = str(Path(__file__).parent.parent / "examples" / "scal_report_ru.xlsx")
OPTS = dict(pc_lab_units="MPa", pc_system="oil-brine", group_by="well", drho=472)


def test_scal_session(tmp_path):
    items, out = scal_session(REPORT, out_dir=str(tmp_path / "out"), **OPTS)
    assert [rt.name for rt, _ in items] == ["well 31", "well 32", "well 33"]
    rt, labs = items[0]
    assert {"lab_wo", "lab_pc"} <= set(labs) and len(labs["lab_wo"]["Sw"]) > 0
    assert rt.pcow is not None
    for name in ("scal_report.xlsx", "relperm_eclipse.inc", "rock_types.json"):
        assert (tmp_path / "out" / name).exists()


def test_load_scal_and_navigate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "r.xlsx").write_bytes(Path(REPORT).read_bytes())
    app = RelPermApp()
    assert [b.label.get_text() for b in app.file_buttons] == ["Открыть отчёт керна…",
                                                              "Открыть .json…"]
    new = app.load_scal(str(tmp_path / "r.xlsx"), **OPTS)
    assert len(new.session) == 3 and new.rt.name == "well 31"
    assert new.s_drho.val == pytest.approx(472)
    assert "Готово" in new.status.get_text()
    assert any(b.label.get_text() == "Тип пород ▶" for b in new.file_buttons)
    nxt = new._show(new.index + 1)
    assert nxt.rt.name == "well 32" and len(plt.get_fignums()) == 1
    again = nxt.load_json(str(tmp_path / "r_relperm" / "rock_types.json"))
    assert again.rt.name == "well 31"
    plt.close("all")


def test_bad_file_is_reported(tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("nothing,here\n1,2\n")
    app = RelPermApp()
    assert app.load_scal(str(bad), **OPTS) is None
    assert "Не удалось" in app.status.get_text() or "не найдено" in app.status.get_text()
    plt.close("all")
