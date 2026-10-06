import json
import re

import numpy as np
import pytest

from relperm import RockType, export
from relperm.cli import main


def _table(text, keyword, comment="--"):
    """Parse numeric rows of the first table after `keyword`."""
    lines = text.splitlines()
    i = lines.index(keyword) + 1
    rows = []
    for line in lines[i:]:
        s = line.strip()
        if s.startswith(comment) or not s:
            continue
        if s == "/" or s.startswith("*"):
            break
        rows.append([float(v) for v in s.split()])
    return np.array(rows)


def test_eclipse_family_one():
    text = export([RockType(name="A"), RockType(name="B", nw=2)], "eclipse", 15)
    assert text.count("/\n") == 4  # 2 tables x (SWOF, SGOF)
    swof = _table(text, "SWOF")
    assert swof.shape[1] == 4
    assert swof[0, 0] == pytest.approx(0.18) and swof[-1, 0] == pytest.approx(1.0)
    sgof = _table(text, "SGOF")
    assert sgof[0, 0] == 0 and sgof[0, 1] == 0
    assert sgof[-1, 2] == 0


def test_eclipse_family_two():
    text = export([RockType()], "eclipse2", 10)
    for kw in ("SWFN", "SGFN", "SOF3"):
        assert kw in text
    sof3 = _table(text, "SOF3")
    assert sof3[0, 0] == 0
    assert sof3[-1, 0] == pytest.approx(1 - 0.18)
    assert sof3[-1, 1] == pytest.approx(0.9) and sof3[-1, 2] == pytest.approx(0.9)


def test_unknown_format():
    with pytest.raises(ValueError):
        export([RockType()], "cmg")


def test_cli_roundtrip(tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    main(["template", "-o", str(cfg)])
    assert json.loads(cfg.read_text())["rock_types"][0]["name"] == "RT1"
    out = tmp_path / "out.inc"
    main(["export", str(cfg), "-f", "eclipse", "-o", str(out)])
    assert "SWOF" in out.read_text()
    png = tmp_path / "plot.png"
    main(["plot", str(cfg), "-o", str(png)])
    assert png.stat().st_size > 0


def test_cli_fit(tmp_path, capsys):
    csv = tmp_path / "lab.csv"
    sw = np.linspace(0.2, 0.8, 8)
    krw = 0.5 * ((sw - 0.2) / 0.6) ** 3
    csv.write_text("Sw;krw\n" + "\n".join(f"{a:.4f};{b:.6f}".replace(".", ",")
                                         for a, b in zip(sw, krw)))
    main(["fit", str(csv), "--phase", "w", "--swcr", "0.2", "--sorw", "0.2"])
    out = capsys.readouterr().out
    n = float(re.search(r"n\s+=\s+([\d.]+)", out).group(1))
    assert n == pytest.approx(3.0, abs=0.01)
