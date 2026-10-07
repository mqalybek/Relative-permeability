"""Build rock types from special core analysis (SCAL) relative permeability data.

Workflow (normalize - average - denormalize):

1. Read lab tables from CSV / Excel. Header rows are found automatically, column
   names are matched against English and Russian aliases, percent values are
   converted to fractions. A table is water-oil (Sw + krw/kro), gas-oil
   (Sg + krg/krog) or sample metadata (sample + perm/poro/rock type/Swi).
2. Per sample: endpoints from the measured points (Swl, Sorw, krw(Sorw), kro(Swl);
   Sgcr, Sorg, krg(Sorg), krog(Sg=0)) and Corey exponents with those endpoints fixed.
3. Per group (rock type): averaged endpoints and one Corey exponent fitted to the
   pooled normalized curves of all samples in the group.
4. RockType objects for export, plus a QC report (tables and plots).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .capillary import MILLIDARCY_M2, PRESSURE_UNITS, BrooksCorey, LeverettJ, height_above_fwl
from .fit import fit_brooks_corey, fit_corey
from .models import RockType, normalize

# --------------------------------------------------------------------- reading

ALIASES = {
    "sample": ["sample", "sample id", "sampleid", "sample no", "plug", "plug id", "core", "core id",
               "образец", "№ образца", "номер образца", "№ обр", "обр", "проба", "шифр образца"],
    "well": ["well", "well no", "well name", "скв", "№ скв", "скважина", "№ скважины"],
    "rock_type": ["rock type", "rocktype", "rt", "satnum", "facies", "litho", "lithotype",
                  "тип породы", "тип", "литотип", "фация", "петротип"],
    "horizon": ["horizon", "zone", "formation", "горизонт", "пласт", "объект"],
    "depth": ["depth", "md", "глубина", "интервал", "глубина отбора"],
    "perm": ["perm", "permeability", "k", "kabs", "k abs", "ka", "kair", "kg", "kl", "kgas",
             "проницаемость", "кпр", "кпр абс", "кабс", "кпр газ", "кпр по газу",
             "проницаемость для газа", "проницаемость по газу", "абсолютная проницаемость"],
    "perm_w": ["kw", "k water", "kbrine", "perm water", "проницаемость по пластовой воде",
               "проницаемость по воде", "кпр по воде", "кпрв"],
    "poro": ["poro", "porosity", "phi", "пористость", "кп", "кпо", "коэффициент пористости"],
    "swi": ["swi", "swirr", "swc", "sw irr", "кво", "sво", "остаточная водонасыщенность"],
    "sor": ["sor", "sorw", "кно", "sно", "остаточная нефтенасыщенность"],
    "kdisp": ["kdisp", "ed", "displacement efficiency", "квыт", "коэффициент вытеснения"],
    "amott": ["amott", "amott index", "амотт", "индекс амотта", "показатель смачиваемости",
              "показатель смачиваемости амотта"],
    "sw": ["sw", "s w", "water saturation", "кв", "sв", "водонасыщенность"],
    "krw": ["krw", "kr w", "krwater", "офп воды", "офпв", "кфпв", "кпрв отн", "фп воды",
            "котн воды", "кот воды"],
    "kro": ["kro", "krow", "kr o", "kroil", "офп нефти", "офпн", "кфпн", "кпрн отн", "фп нефти",
            "котн нефти", "кот нефти"],
    "sg": ["sg", "s g", "gas saturation", "кг", "sг", "газонасыщенность"],
    "krg": ["krg", "kr g", "krgas", "офп газа", "офпг", "кфпг", "фп газа"],
    "krog": ["krog", "kro g", "офп нефти в системе газ-нефть", "офпнг"],
    "pc": ["pc", "pcow", "pcap", "pc lab", "capillary pressure", "капиллярное давление",
           "рк", "ркап", "pкап", "давление"],
    "shg": ["shg", "s hg", "snw", "hg saturation", "mercury saturation", "насыщенность ртутью",
            "ртутенасыщенность", "sрт", "кнр"],
    "system": ["system", "fluid system", "fluids", "method", "система", "флюиды", "метод"],
}
TEXT_COLUMNS = ("sample", "well", "rock_type", "horizon", "system")
# sample properties copied onto the curve tables of the same sample
META_COLUMNS = ("perm", "perm_w", "poro", "rock_type", "swi", "well", "horizon", "depth")
FRACTION_COLUMNS = ("sw", "sg", "swi", "sor", "kdisp", "poro", "krw", "kro", "krg", "krog", "shg")
# words that may follow a column name without changing it: units, "доли ед." and the like
_UNIT_WORDS = {"доли", "дол", "ед", "д", "мд", "md", "мкм2", "мпа", "mpa", "кпа", "kpa", "атм",
               "atm", "бар", "bar", "psi", "па", "pa", "см", "м", "мл", "кг", "fraction", "frac",
               "frac.", "%", "v/v", "дол.ед", "отн", "ед.", "д.ед", "д.ед."}

# lab pressure units -> bar, matched against the units part of the Pc header
PC_UNITS_TO_BAR = [
    (r"psi", 0.0689476), (r"[kк][pп][aа]", 0.01), (r"[mм][pп][aа]", 10.0),
    (r"кгс|kgf|at\b|ат\b", 0.980665), (r"atm|атм", 1.01325), (r"bar|бар", 1.0),
    (r"^\s*(pa|па)\s*$", 1e-5),
]

# (interfacial tension mN/m, contact angle deg) for lab fluid pairs and keywords to spot them
LAB_SYSTEMS = {
    "mercury-air": (480.0, 140.0),
    "oil-brine": (48.0, 30.0),
    "air-brine": (72.0, 0.0),
}
_SYSTEM_KEYWORDS = [  # checked in order
    ("mercury-air", ("mercury", "micp", "hg", "ртут")),
    ("oil-brine", ("oil", "нефть", "нефт", "керосин")),
    ("air-brine", ("air", "gas", "газ", "воздух", "centrifug", "центрифуг", "porous", "плупр",
                   "полупрон", "капилляриметр", "brine")),
]

# Cyrillic letters that look like Latin ones, so "Kв" (Latin K) matches "кв".
_LOOKALIKE = str.maketrans("кораснхе", "kopacnxe")
# Sample IDs mix them too ("011004036РT03H" with Cyrillic Р): compare in Latin capitals.
_ID_LOOKALIKE = str.maketrans("АВЕКМНОРСТХаеорсух", "ABEKMHOPCTXaeopcyx")


def _norm(name) -> str:
    s = str(name).strip().lower()
    s = re.split(r"[\(\[,]", s)[0]  # drop units: "Sw, д.ед.", "k (mD)"
    s = re.sub(r"[\s_\-.:;№#/]", "", s)
    return s.translate(_LOOKALIKE)


ALIAS_INDEX = {_norm(a): canon for canon, names in ALIASES.items() for a in names}
ALIAS_INDEX.pop("", None)


def _unit_like(word: str) -> bool:
    """Short unit-ish token ("мД", "%", "*10-3", "мкм2"), not a sample ID like "011004059РT01H"."""
    w = word.strip().lower()
    return w in _UNIT_WORDS or (len(w) <= 6 and bool(re.search(r"[\d%*/.]", w)))


def _match_header(text) -> tuple[str | None, str]:
    """Canonical column name and the leftover text (units) of a header cell.

    Trailing unit words are ignored: "Кпр. мД", "Пористость. доли ед.", "Pc_Mpa".
    """
    low = str(text).strip().lower()
    if not low or low == "nan":
        return None, ""
    m = re.search(r"[\(\[,]", low)
    base, units = (low[:m.start()], low[m.start():]) if m else (low, "")
    words = [w for w in re.split(r"[\s_]+", base) if w]
    for n in range(len(words), 0, -1):
        if n < len(words) and not all(_unit_like(w) for w in words[n:]):
            break
        canon = ALIAS_INDEX.get(_norm(" ".join(words[:n])))
        if canon:
            return canon, " ".join(words[n:]) + " " + units
    return None, ""


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("SCAL import needs pandas and openpyxl: pip install -e '.[scal]'") from e
    return pd


def _csv_separator(path: str) -> str:
    """';' or tab if every line has one (decimal-comma files), else ','."""
    with open(path, encoding="utf-8-sig") as f:
        lines = [ln for ln in (f.readline() for _ in range(50)) if ln.strip()]
    for sep in (";", "\t"):
        if lines and all(sep in ln for ln in lines):
            return sep
    return ","


def _pc_unit_factor(header) -> float | None:
    """bar per lab unit from a Pc header like "Pc, psi", "Рк (МПа)" or "Pc_Mpa"; None if absent."""
    _, units = _match_header(header)
    for pattern, factor in PC_UNITS_TO_BAR:
        if re.search(pattern, units.strip()):
            return factor
    return None


def lab_system(text) -> str | None:
    """Lab fluid pair from free text: column value, sheet or file name."""
    t = str(text).lower()
    for system, words in _SYSTEM_KEYWORDS:
        if any(w in t for w in words):
            return system
    return None


def _strip_system_words(name: str) -> str:
    """'101 MICP' -> '101', so Pc sheets join the kr sheets of the same sample."""
    out = re.sub(r"[_\-]+", " ", str(name))
    for _, words in _SYSTEM_KEYWORDS:
        for w in words:
            out = re.sub(rf"(?i)\b{re.escape(w)}\w*", "", out)
    out = re.sub(r"(?i)\b(pc|рк|кривая|curve)\b", "", out)
    return re.sub(r"[\s_\-]+", " ", out).strip() or str(name)


def _header_columns(row) -> dict:
    """{column index: (canonical name, header text)} for one row; first use of a name wins."""
    cols, seen = {}, set()
    for j, c in enumerate(row):
        canon, _ = _match_header(c)
        if canon and canon not in seen:
            cols[j] = (canon, str(c))
            seen.add(canon)
    return cols


def _find_tables(raw) -> list[dict]:
    """Tables on a sheet: header rows with at least two known columns.

    A later header row that reuses columns of an earlier table ends that table (two tables
    stacked on one sheet). Columns it does not redefine (sample, well, ...) are inherited, so
    curves below a summary table keep the sample IDs written in the summary's column.
    """
    tables = []
    for i in range(len(raw)):
        row = raw.iloc[i]
        cols = _header_columns(row)
        numbers = int(_to_number(row.dropna().astype(str)).notna().sum())
        if len(cols) < 2 or numbers > len(cols):  # data rows are mostly numbers
            continue
        inherited = {}
        for t in tables:
            if t["end"] is None and set(t["cols"]) & set(cols):
                t["end"] = i
                inherited.update({j: v for j, v in t["cols"].items()
                                  if j not in cols and v[0] in ("sample", "well", "horizon",
                                                                 "rock_type")})
        tables.append({"row": i, "cols": {**inherited, **cols}, "end": None})
    for t in tables:
        if t["end"] is None:
            t["end"] = len(raw)
    return tables


def _sample_id(v) -> str:
    """Clean sample ID: "обр. 011004036РT03H, Кп=15,2%" -> "011004036PT03H"."""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = str(v).strip()
    s = re.sub(r"(?i)^(обр(азец)?|sample|plug)[\s.,:;№#]*", "", s)
    s = next((part.strip() for part in s.split(",") if part.strip()), s)
    return s.translate(_ID_LOOKALIKE).upper() if re.search(r"\d", s) else s


def _to_number(series):
    pd = _pandas()
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False).str.replace(" ", "")
    return pd.to_numeric(s, errors="coerce")


_WELL_SEPARATOR = re.compile(r"(?i)^\s*(скважина|скв\.?|well)\s*№?\s*([\w\-/]+)\s*$")


def _standardize(raw, table: dict, source: str):
    """One table of a raw sheet -> (DataFrame with canonical columns, has a sample column)."""
    pd = _pandas()
    keep = sorted(table["cols"])
    df = raw.iloc[table["row"] + 1:table["end"], keep].copy()
    df.columns = [table["cols"][j][0] for j in keep]
    pc_factor = None
    for j in keep:
        canon, header = table["cols"][j]
        if canon == "pc" and pc_factor is None:
            pc_factor = _pc_unit_factor(header)
    has_sample = "sample" in df
    for c in df.columns:
        if c in TEXT_COLUMNS:
            v = df[c].where(df[c].notna())
            v = v.map(lambda x: str(x).strip() if x is not None and str(x) != "nan" else None)
            df[c] = v.where(~v.isin(["", "-", "—", "nan", None]))
        else:
            df[c] = _to_number(df[c])
    # "Скважина №31" rows inside a table: set the well for the rows below, then drop them
    if has_sample:
        sep = df["sample"].astype(str).str.extract(_WELL_SEPARATOR)[1]
        if sep.notna().any():
            df["well"] = (df["well"].fillna(sep) if "well" in df else sep).ffill()
            df = df[sep.isna()]
    if "well" in df:
        df["well"] = df["well"].ffill().map(_sample_id, na_action="ignore")
    curves = "sw" in df or "sg" in df or "shg" in df
    if curves:  # merged cells / IDs written once per block
        for c in ("sample", "rock_type", "horizon", "perm", "perm_w", "poro", "swi", "system"):
            if c in df:
                df[c] = df[c].ffill()
    if ("sg" in df or "krg" in df) and "krog" not in df and "kro" in df:
        df = df.rename(columns={"kro": "krog"})
    if not has_sample:
        df["sample"] = _strip_system_words(source) if "pc" in df else source
    df["sample"] = df["sample"].map(_sample_id, na_action="ignore")
    for c in FRACTION_COLUMNS:  # percent -> fraction
        if c in df and df[c].max(skipna=True) > 1.5:
            df[c] = df[c] / 100.0
    if "pc" in df:
        if "sw" not in df and "shg" in df:  # mercury (non-wetting) saturation
            df["sw"] = 1.0 - df["shg"]
        # Pc in bar when the unit is in the header; NaN factor -> default units later
        df["pc_bar_factor"] = pc_factor if pc_factor is not None else np.nan
        fallback = lab_system(source)  # e.g. "101 MICP"; None if the name says nothing
        if "system" in df:
            system = df["system"].map(lab_system)
            df["pc_system"] = system if fallback is None else system.fillna(fallback)
        else:
            df["pc_system"] = fallback
    df["source"] = source
    return pd.DataFrame(df), has_sample


def _kinds(df, has_sample: bool) -> list[str]:
    kinds = []
    if "sw" in df and ("krw" in df or "kro" in df):
        kinds.append("wo")
    if "sg" in df and ("krg" in df or "krog" in df):
        kinds.append("go")
    if "sw" in df and "pc" in df:
        kinds.append("pc")
    if not kinds and has_sample:
        kinds.append("meta")
    return kinds


@dataclass
class ScalData:
    water_oil: object  # pandas DataFrame: sample, Sw, krw, kro (+ meta)
    gas_oil: object  # pandas DataFrame: sample, Sg, krg, krog (+ meta)
    sources: list = field(default_factory=list)
    capillary: object = None  # pandas DataFrame: sample, Sw, pc, pc_bar_factor, pc_system (+ meta)
    samples: object = None  # pandas DataFrame indexed by sample: every property found
    discrepancies: list = field(default_factory=list)  # same property, different sheets


def _discrepancies(meta, tolerance=0.1) -> list[dict]:
    """Numeric properties of one sample that differ between sheets by more than tolerance."""
    out = []
    numeric = [c for c in meta.columns if c not in TEXT_COLUMNS + ("source",)
               and meta[c].dtype.kind in "fi"]
    for sample, d in meta.groupby("sample"):
        for c in numeric:
            v = d[[c, "source"]].dropna()
            if len(v) < 2:
                continue
            lo, hi = v[c].min(), v[c].max()
            if lo > 0 and hi / lo - 1 > tolerance:
                out.append({"sample": sample, "property": c,
                            "values": "; ".join(f"{r.source}: {r[c]:g}" for _, r in v.iterrows())})
    return out


def read_scal(*paths: str) -> ScalData:
    """Read SCAL tables from CSV/Excel files (all sheets and all tables on a sheet) and merge
    sample properties (k, porosity, Swi, Sor, ...) found anywhere by sample ID."""
    pd = _pandas()
    parts = {"wo": [], "go": [], "pc": [], "meta": []}
    sources = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
            sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        else:
            sheets = {stem: pd.read_csv(path, header=None, sep=_csv_separator(path),
                                        dtype=str, encoding="utf-8-sig")}
        for name, raw in sheets.items():
            raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
            raw.columns = range(raw.shape[1])
            tables = _find_tables(raw)
            if not tables:
                sources.append((path, str(name), "skipped"))
            for k, table in enumerate(tables):
                label = str(name) if len(tables) == 1 else f"{name} #{k + 1}"
                df, has_sample = _standardize(raw, table, str(name))
                if len(tables) > 1 and not has_sample:
                    # side calculations next to the data (no sample IDs): not lab data
                    sources.append((path, label, "skipped (no sample column)"))
                    continue
                kinds = _kinds(df, has_sample)
                sources.append((path, label, "+".join(kinds) or "skipped"))
                for kind in kinds:
                    parts[kind].append(df)

    def cat(frames, sat):
        if not frames:
            return pd.DataFrame(columns=["sample", sat])
        return pd.concat(frames, ignore_index=True).dropna(subset=[sat])

    wo, go = cat(parts["wo"], "sw"), cat(parts["go"], "sg")
    pc = cat(parts["pc"], "sw").dropna(subset=["pc"]) if parts["pc"] else cat([], "sw")
    samples, discrepancies = None, []
    if parts["meta"]:
        meta = pd.concat(parts["meta"], ignore_index=True).dropna(subset=["sample"])
        meta = meta[meta.drop(columns=["sample", "source"], errors="ignore").notna().any(axis=1)]
        discrepancies = _discrepancies(meta)
        samples = meta.drop(columns=["source"]).groupby("sample").first()
        for df in (wo, go, pc):
            for c in META_COLUMNS:
                if c in samples:
                    fill = df["sample"].map(samples[c])
                    df[c] = df[c].fillna(fill) if c in df else fill
    if wo.empty and go.empty and pc.empty:
        raise ValueError("No SCAL tables found (need Sw + krw/kro, Sg + krg/krog or Sw/SHg + Pc)")
    return ScalData(wo, go, sources, pc, samples, discrepancies)


# ------------------------------------------------------------------- analysis

def fit_exponent(sn, krn) -> tuple[float, float, int]:
    """Corey exponent of normalized kr = sn**n by least squares in linear space.

    Golden-section search on n in [0.3, 12]. Returns (n, rmse, points used).
    """
    sn, krn = np.asarray(sn, float), np.asarray(krn, float)
    m = np.isfinite(sn) & np.isfinite(krn) & (sn > 0) & (sn < 1)
    if not m.any():
        return math.nan, math.nan, 0
    x, y = sn[m], krn[m]
    sse = lambda n: float(np.sum((x**n - y) ** 2))  # noqa: E731
    lo, hi = math.log(0.3), math.log(12.0)  # search in log n
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = sse(math.exp(a)), sse(math.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = sse(math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = sse(math.exp(b))
    n = math.exp((lo + hi) / 2)
    return n, math.sqrt(sse(n) / m.sum()), int(m.sum())


@dataclass
class Sample:
    sample: str
    kind: str  # "wo", "go" or "pc"
    group: str = ""
    perm: float = math.nan
    poro: float = math.nan
    endpoints: dict = field(default_factory=dict)
    exponents: dict = field(default_factory=dict)
    rmse: dict = field(default_factory=dict)
    points: int = 0
    notes: list = field(default_factory=list)
    # normalized curves for pooling: name -> (sn, kr / kr_end)
    normalized: dict = field(default_factory=dict, repr=False)
    raw: object = field(default=None, repr=False)


def _first(d, col):
    if col in d and d[col].notna().any():
        return float(d[col].dropna().iloc[0])
    return math.nan


def analyze_water_oil(d) -> Sample:
    d = d.sort_values("sw")
    s = Sample(_sample_id(d["sample"].iloc[0]), "wo", perm=_first(d, "perm"),
               poro=_first(d, "poro"), points=len(d), raw=d)
    sw = d["sw"].to_numpy(float)
    swl = _first(d, "swi")
    if math.isnan(swl):
        swl = sw.min()
    elif sw.min() > swl + 0.01:
        s.notes.append(f"first Sw ({sw.min():.3f}) > Swi ({swl:.3f}); krocw taken at first point")
    sorw = 1.0 - sw.max()
    ep = {"swl": swl, "sorw": sorw}
    if "krw" in d and d["krw"].notna().any():
        krw = d["krw"].to_numpy(float)
        ep["krwr"] = float(krw[np.nanargmax(sw)])
        sn = normalize(sw, swl, 1.0 - sorw)
        s.normalized["krw"] = (sn, krw / ep["krwr"])
        s.exponents["nw"], s.rmse["nw"], _ = fit_exponent(sn, krw / ep["krwr"])
    if "kro" in d and d["kro"].notna().any():
        kro = d["kro"].to_numpy(float)
        ep["krocw"] = float(kro[np.nanargmin(sw)])
        sn = normalize(1.0 - sw, sorw, 1.0 - swl)
        s.normalized["kro"] = (sn, kro / ep["krocw"])
        s.exponents["now"], s.rmse["now"], _ = fit_exponent(sn, kro / ep["krocw"])
        if ep["krocw"] > 0.98:
            s.notes.append("kro(Swl) ~ 1: data likely normalized to ko(Swi), not absolute k")
    s.endpoints = ep
    return s


def analyze_gas_oil(d, swl: float) -> Sample:
    d = d.sort_values("sg")
    s = Sample(_sample_id(d["sample"].iloc[0]), "go", perm=_first(d, "perm"),
               poro=_first(d, "poro"), points=len(d), raw=d)
    sg = d["sg"].to_numpy(float)
    if not math.isnan(_first(d, "swi")):
        swl = _first(d, "swi")
    if math.isnan(swl):
        swl = 0.0
        s.notes.append("no Swi for gas-oil test; assumed 0")
    sorg = max(0.0, 1.0 - swl - sg.max())
    ep = {"swl": swl, "sorg": sorg, "sgcr": 0.0}
    if "krg" in d and d["krg"].notna().any():
        krg = d["krg"].to_numpy(float)
        zero = sg[krg <= 1e-6]
        ep["sgcr"] = float(zero.max()) if zero.size else 0.0
        ep["krgr"] = float(krg[np.nanargmax(sg)])
        sn = normalize(sg, ep["sgcr"], sg.max())
        s.normalized["krg"] = (sn, krg / ep["krgr"])
        s.exponents["ng"], s.rmse["ng"], _ = fit_exponent(sn, krg / ep["krgr"])
    if "krog" in d and d["krog"].notna().any():
        krog = d["krog"].to_numpy(float)
        ep["krogcg"] = float(krog[np.nanargmin(sg)])
        sn = normalize(1.0 - sg, swl + sorg, 1.0)
        s.normalized["krog"] = (sn, krog / ep["krogcg"])
        s.exponents["nog"], s.rmse["nog"], _ = fit_exponent(sn, krog / ep["krogcg"])
    s.endpoints = ep
    return s


@dataclass
class PcOptions:
    """How to turn lab capillary pressure into reservoir oil-water Pc."""

    lab_units: str | None = None  # used when the Pc header has no units: psi, kPa, MPa, bar, atm
    system: str | None = None  # lab fluid pair when not given in the data: see LAB_SYSTEMS
    ift_res: float = 30.0  # reservoir oil-water interfacial tension, mN/m
    theta_res: float = 30.0  # reservoir contact angle, degrees
    units: str = "bar"  # Pc units of the rock types
    j_form: str = "auto"  # Leverett J form: "power", "exp" or "auto" (better fit)
    drho: float = 250.0  # oil-water density difference for QC heights, kg/m3

    @property
    def sigma_res(self) -> float:
        """sigma * cos(theta) at reservoir conditions, mN/m."""
        return self.ift_res * math.cos(math.radians(self.theta_res))


def fit_j(sn, j, form: str) -> dict:
    """Fit J = a * Swn**(-b) ("power") or a * exp(-b * Swn) ("exp").

    In the lab Pc is set and Sw is measured, and the model has to reproduce the saturation
    distribution, so the regression minimizes the error in saturation: Swn (or ln Swn for
    the power form) against ln J. Uses points with 0 < Swn < 1 and J > 0. Returns a, b,
    rmse of Swn and the number of points used.
    """
    sn, j = np.asarray(sn, float), np.asarray(j, float)
    m = np.isfinite(sn) & np.isfinite(j) & (sn > 0) & (sn < 1) & (j > 0)
    if m.sum() < 2:
        raise ValueError("fewer than 2 Pc points with 0 < Swn < 1")
    x, ln_j = sn[m], np.log(j[m])
    target = np.log(x) if form == "power" else x
    slope, intercept = np.polyfit(ln_j, target, 1)
    if slope >= 0:
        raise ValueError("Swn does not decrease as Pc grows; not a drainage curve")
    b = -1.0 / slope
    a = float(np.exp(intercept * b))
    pred = np.exp(-(ln_j - math.log(a)) / b) if form == "power" else (math.log(a) - ln_j) / b
    rmse = float(np.sqrt(np.mean((np.clip(pred, 0, 1) - x) ** 2)))
    return {"form": form, "a": a, "b": float(b), "rmse_swn": rmse, "used": int(m.sum())}


def analyze_capillary(d, opts: PcOptions) -> Sample:
    """Lab Pc -> reservoir Pc -> Leverett J against normalized water saturation
    Swn = (Sw - Swirr) / (1 - Swirr), Swirr = lowest measured Sw (at the highest Pc)."""
    d = d.sort_values("sw").copy()
    s = Sample(_sample_id(d["sample"].iloc[0]), "pc", perm=_first(d, "perm"),
               poro=_first(d, "poro"), points=len(d), raw=d)
    factor = d["pc_bar_factor"].astype(float)
    if factor.isna().any():
        if opts.lab_units:
            factor = factor.fillna(1e-5 / PRESSURE_UNITS[opts.lab_units])
        else:
            factor = factor.fillna(1.0)
            s.notes.append("Pc units not in header; assumed bar (set --pc-lab-units)")
    system = d["pc_system"].dropna().iloc[0] if d["pc_system"].notna().any() else opts.system
    if system is None:
        system = "air-brine"
        s.notes.append("lab fluid system unknown; assumed air-brine (set --pc-system)")
    ift_lab, theta_lab = LAB_SYSTEMS[system]
    sigma_lab = ift_lab * abs(math.cos(math.radians(theta_lab)))
    d["pc_lab_bar"] = d["pc"].astype(float) * factor
    d["pc_res_bar"] = d["pc_lab_bar"] * opts.sigma_res / sigma_lab

    swirr = float(d["sw"].min())
    d["sn"] = normalize(d["sw"].to_numpy(float), swirr, 1.0)
    s.endpoints = {"swl": swirr, "swirr": swirr, "system": system,
                   "pc_lab_max_bar": float(d["pc_lab_bar"].max()),
                   "pc_res_max": float(d["pc_res_bar"].max())}
    if s.perm > 0 and 0 < s.poro < 1:
        sqrt_k_phi = math.sqrt(s.perm * MILLIDARCY_M2 / s.poro)
        d["J"] = d["pc_res_bar"] * 1e5 * sqrt_k_phi / (opts.sigma_res * 1e-3)
        s.normalized["J"] = (d["sn"].to_numpy(float), d["J"].to_numpy(float))
        for form in ("power", "exp"):
            try:
                f = fit_j(d["sn"], d["J"], form)
                s.exponents.update({f"J_a_{form}": f["a"], f"J_b_{form}": f["b"]})
                s.rmse[f"J_{form}"] = f["rmse_swn"]
            except ValueError:
                s.notes.append("too few Pc points inside (Swirr, 1) to fit J")
                break
    else:
        s.notes.append("no permeability/porosity: Pc not converted to J")
    s.normalized["pc"] = (d["sn"].to_numpy(float), d["pc_res_bar"].to_numpy(float))
    s.raw = d
    return s


def fit_group_capillary(members, opts: PcOptions, how: str = "mean"):
    """Pooled Leverett J (or Brooks-Corey when k/phi are missing) for one rock type.

    Returns (model, fits): fits holds both J forms for the report; opts.j_form picks one,
    "auto" takes the smaller saturation error.
    """
    pcs = [s for s in members if s.kind == "pc"]
    with_j = [s for s in pcs if "J" in s.normalized]
    sig = lambda x: float(f"{x:.5g}")  # noqa: E731
    if with_j:
        sn = np.concatenate([s.normalized["J"][0] for s in with_j])
        j = np.concatenate([s.normalized["J"][1] for s in with_j])
        fits = {form: fit_j(sn, j, form) for form in ("power", "exp")}
        form = opts.j_form if opts.j_form != "auto" else min(fits, key=lambda f: fits[f]["rmse_swn"])
        f = fits[form]
        perm = float(np.exp(np.mean(np.log([s.perm for s in with_j]))))  # geometric mean
        poro = _aggregate([s.poro for s in with_j], how)
        model = LeverettJ(a=sig(f["a"]), b=sig(f["b"]), perm=sig(perm), poro=sig(poro),
                          ift=opts.ift_res, theta=opts.theta_res, form=form)
        if form == "power":  # cap the singularity at the highest measured J
            j_max = float(np.nanmax(j[sn > 0])) if np.any(sn > 0) else f["a"]
            model = dataclasses.replace(model, pcmax=sig(max(j_max, f["a"])
                                                        * model.factor(opts.units)))
        return model, fits
    if pcs:
        sn = np.concatenate([s.normalized["pc"][0] for s in pcs])
        pc_bar = np.concatenate([s.normalized["pc"][1] for s in pcs])
        pc = pc_bar * 1e5 * PRESSURE_UNITS[opts.units]
        fit = fit_brooks_corey(sn, pc, 0.0, 1.0)
        return BrooksCorey(pe=sig(fit.pe), lam=sig(fit.lam),
                           pcmax=sig(max(float(np.nanmax(pc)), fit.pe))), {}
    return None, {}


def _perm_label(k, bins) -> str:
    if k is None or math.isnan(k):
        return "unknown_k"
    edges = [0.0, *bins, math.inf]
    for lo, hi in zip(edges, edges[1:]):
        if lo <= k < hi:
            if lo == 0:
                return f"k<{hi:g}"
            return f"k>={lo:g}" if hi == math.inf else f"k{lo:g}-{hi:g}"
    return "unknown_k"


@dataclass
class ScalResult:
    samples: list
    rock_types: list
    groups: dict  # name -> list of Sample
    correlations: list
    warnings: list
    pc_fits: dict = field(default_factory=dict)  # rock type -> {"power": {...}, "exp": {...}}
    data: object = None  # the ScalData the result was built from
    dependencies: dict = field(default_factory=dict)  # filled by write_report


def _aggregate(values, how):
    v = np.array([x for x in values if x is not None and not math.isnan(x)], float)
    if v.size == 0:
        return math.nan
    return float(np.median(v) if how == "median" else np.mean(v))


def _pooled(samples, curve):
    parts = [s.normalized[curve] for s in samples if curve in s.normalized]
    if not parts:
        return math.nan
    sn = np.concatenate([p[0] for p in parts])
    kr = np.concatenate([p[1] for p in parts])
    return fit_exponent(sn, kr)[0]


def build_rock_types(data: ScalData, group_by: str = "auto", perm_bins=(),
                     endpoints: str = "mean", align_kro: bool = True,
                     pc: PcOptions | None = None) -> ScalResult:
    """Analyze every sample and build one RockType per group.

    align_kro: set krog(Sg=0) equal to krow(Swl). Both are kro at Swi, but they are
    averaged over different sample sets; Eclipse needs them equal for Stone models.

    group_by: "rock_type" (column), "perm" (perm_bins), "none" (one group) or "auto"
    (rock_type column if present, else perm bins if given, else one group).

    pc: options for capillary pressure tables (lab -> reservoir conversion, units).
    """
    pc = pc or PcOptions()
    wo, go = data.water_oil, data.gas_oil
    cap = data.capillary if data.capillary is not None else wo.iloc[0:0]
    if group_by == "auto":
        has_rt = any("rock_type" in df and df["rock_type"].notna().any() for df in (wo, go, cap))
        group_by = "rock_type" if has_rt else ("perm" if perm_bins else "none")

    samples = [analyze_water_oil(d) for _, d in wo.groupby("sample", sort=False)] if len(wo) else []
    swl_by_sample = {s.sample: s.endpoints["swl"] for s in samples}

    def group_of(s: Sample) -> str:
        if group_by == "rock_type":
            col = s.raw.get("rock_type")
            if col is None or col.isna().all():
                return "unassigned"
            return _sample_id(col.dropna().iloc[0])
        if group_by == "perm":
            return _perm_label(s.perm, perm_bins)
        if group_by in ("horizon", "well"):
            col = s.raw.get(group_by)
            if col is None or col.isna().all():
                return "unassigned"
            value = str(col.dropna().iloc[0])
            return f"well {value}" if group_by == "well" else value
        return "RT1"

    for s in samples:
        s.group = group_of(s)
    wo_group_swl = {}
    for s in samples:
        wo_group_swl.setdefault(s.group, []).append(s.endpoints["swl"])

    go_samples = []
    for _, d in (go.groupby("sample", sort=False) if len(go) else []):
        sid = _sample_id(d["sample"].iloc[0])
        probe = Sample(sid, "go", perm=_first(d, "perm"), raw=d)
        grp = group_of(probe)
        swl = swl_by_sample.get(sid, _aggregate(wo_group_swl.get(grp, []), endpoints))
        s = analyze_gas_oil(d, swl)
        s.group = grp
        go_samples.append(s)
    samples += go_samples

    for _, d in (cap.groupby("sample", sort=False) if len(cap) else []):
        s = analyze_capillary(d, pc)
        s.group = group_of(s)
        samples.append(s)

    groups: dict = {}
    for s in samples:
        groups.setdefault(s.group, []).append(s)

    rock_types, warnings, pc_fits = [], [], {}
    for name, members in groups.items():
        w = [s for s in members if s.kind == "wo"]
        g = [s for s in members if s.kind == "go"]
        c = [s for s in members if s.kind == "pc"]
        agg = lambda ss, key: _aggregate([s.endpoints.get(key, math.nan) for s in ss], endpoints)  # noqa: E731
        params = {"name": name}
        if w:
            params.update(swl=agg(w, "swl"), sowcr=agg(w, "sorw"),
                          krwr=agg(w, "krwr"), krocw=agg(w, "krocw"),
                          nw=_pooled(w, "krw"), now=_pooled(w, "kro"))
            params["swcr"] = params["swl"]
        elif g or c:
            params["swl"] = params["swcr"] = agg(g or c, "swl")
            warnings.append(f"{name}: no water-oil data, water-oil curves are defaults")
        if g:
            params.update(sgcr=agg(g, "sgcr"), sogcr=agg(g, "sorg"), krgr=agg(g, "krgr"),
                          krogcg=agg(g, "krogcg"), ng=_pooled(g, "krg"), nog=_pooled(g, "krog"))
            krocw = params.get("krocw", math.nan)
            if align_kro and not math.isnan(krocw) and not math.isclose(krocw, params["krogcg"]):
                warnings.append(f"{name}: krogcg {params['krogcg']:.4f} from gas-oil data "
                                f"aligned to krocw {krocw:.4f}")
                params["krogcg"] = krocw
        else:
            if "krocw" in params:
                params["krogcg"] = params["krocw"]
            warnings.append(f"{name}: no gas-oil data, gas-oil curves are defaults")
        if c:
            try:
                params["pcow"], pc_fits[name] = fit_group_capillary(members, pc, endpoints)
                params["pc_units"] = pc.units
            except ValueError as e:
                warnings.append(f"{name}: capillary pressure not fitted ({e})")
        params = {k: v for k, v in params.items() if not (isinstance(v, float) and math.isnan(v))}
        params["description"] = (f"SCAL: {len(w)} water-oil, {len(g)} gas-oil, {len(c)} Pc samples, "
                                 f"{endpoints} endpoints")
        try:
            rt = RockType(**params)
        except ValueError as e:
            warnings.append(f"{name}: cannot build rock type ({e})")
            continue
        rock_types.append(rt)
        warnings += [f"{name}: {x}" for x in rt.warnings()]

    return ScalResult(samples, rock_types, groups, endpoint_correlations(samples), warnings,
                      pc_fits, data)


def endpoint_correlations(samples) -> list[dict]:
    """Linear fits endpoint = a + b * log10(k) over all samples with permeability."""
    out = []
    for kind, keys in (("wo", ("swl", "sorw", "krwr", "krocw")),
                       ("go", ("sgcr", "sorg", "krgr", "krogcg"))):
        for key in keys:
            pts = [(s.perm, s.endpoints[key]) for s in samples
                   if s.kind == kind and s.perm > 0 and key in s.endpoints]
            if len(pts) < 3:
                continue
            x = np.log10([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                continue
            b, a = np.polyfit(x, y, 1)
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - float(np.sum((y - (a + b * x)) ** 2)) / ss_tot if ss_tot > 0 else 1.0
            out.append({"endpoint": key, "a": float(a), "b": float(b), "r2": r2, "n": len(pts)})
    return out


# --------------------------------------------------------------------- report

def samples_table(result: ScalResult) -> list[dict]:
    rows = []
    for s in result.samples:
        test = {"wo": "water-oil", "go": "gas-oil", "pc": "capillary"}[s.kind]
        row = {"sample": s.sample, "test": test,
               "group": s.group, "perm": s.perm, "poro": s.poro, "points": s.points}
        row.update(s.endpoints)
        row.update(s.exponents)
        row.update({f"rmse_{k}": v for k, v in s.rmse.items()})
        row["notes"] = "; ".join(s.notes)
        rows.append(row)
    return rows


def write_report(result: ScalResult, out_dir: str, points: int = 20,
                 pc: PcOptions | None = None) -> list[str]:
    """Write rock_types.json, samples.csv, correlations.csv, lab overlays and QC plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pd = _pandas()
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def path(name):
        p = os.path.join(out_dir, name)
        written.append(p)
        return p

    with open(path("rock_types.json"), "w", encoding="utf-8") as f:
        json.dump({"points": points, "rock_types": [rt.to_dict() for rt in result.rock_types]},
                  f, indent=2, ensure_ascii=False)
    pd.DataFrame(samples_table(result)).to_csv(path("samples.csv"), index=False,
                                               float_format="%.5g")
    if result.correlations:
        pd.DataFrame(result.correlations).to_csv(path("correlations.csv"), index=False,
                                                 float_format="%.5g")

    by_name = {rt.name: rt for rt in result.rock_types}
    for name, members in result.groups.items():
        safe = re.sub(r"[^\w.-]+", "_", name)
        rt = by_name.get(name)
        # pooled lab points in the layout `relperm app --lab-wo/--lab-go` reads
        for kind, sat, cols in (("wo", "sw", ("krw", "kro")), ("go", "sg", ("krg", "krog"))):
            raws = [s.raw for s in members if s.kind == kind]
            if raws:
                df = pd.concat(raws)
                lab = pd.DataFrame({"Sw" if kind == "wo" else "Sg": df[sat]})
                for c in cols:
                    if c in df:
                        lab["krow" if c == "kro" else c] = df[c]
                lab.to_csv(path(f"lab_{safe}_{kind}.csv"), index=False, float_format="%.5g")
        _plot_group(plt, name, members, rt, path(f"qc_{safe}.png"))
        pcs = [s for s in members if s.kind == "pc"]
        if pcs and rt is not None and rt.pcow is not None:
            sw, pcow = denormalized_pc_points(pcs, rt)
            pd.DataFrame({"Sw": sw, "Pcow": pcow}).to_csv(path(f"lab_{safe}_pc.csv"), index=False,
                                                          float_format="%.5g")
            _plot_capillary(plt, name, pcs, rt, pc or PcOptions(), path(f"qc_pc_{safe}.png"),
                            result.pc_fits.get(name))

    if any(s.perm > 0 for s in result.samples):
        _plot_correlations(plt, result, path("endpoints_vs_perm.png"))
    from .dependencies import write_dependencies

    written += write_dependencies(result, out_dir, plt)
    return written


def _plot_group(plt, name, members, rt, filename):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"SCAL QC — {name}", fontsize=13)
    cmap = plt.get_cmap("tab10")
    specs = (("wo", "sw", (("krw", "krw", "nw"), ("kro", "krow", "now")), "Sw", axes[0]),
             ("go", "sg", (("krg", "krg", "ng"), ("krog", "krog", "nog")), "Sg", axes[1]))
    for kind, sat, curves, xlabel, (ax_raw, ax_norm) in specs:
        ss = [s for s in members if s.kind == kind]
        for i, s in enumerate(ss):
            c = cmap(i % 10)
            for (col, _, _), marker in zip(curves, ("o", "s")):
                if col in s.raw:
                    ax_raw.plot(s.raw[sat], s.raw[col], marker, mfc="none", color=c,
                                label=f"{s.sample} {col}" if marker == "o" else None)
                if col in s.normalized:
                    ax_norm.plot(*s.normalized[col], marker, mfc="none", color=c, ms=5)
        sn = np.linspace(0, 1, 200)
        for (col, label, exp_name), ls in zip(curves, ("-", "--")):
            if rt is not None and ss:
                n = getattr(rt, exp_name)
                ax_norm.plot(sn, sn**n, "k", ls=ls, lw=2, label=f"{label}*: n = {n:.2f}")
        if rt is not None and ss:
            s_range = (np.linspace(rt.swl, 1, 200) if kind == "wo"
                       else np.linspace(0, 1 - rt.swl, 200))
            f1, f2 = (rt.krw, rt.krow) if kind == "wo" else (rt.krg, rt.krog)
            ax_raw.plot(s_range, f1(s_range), "k-", lw=2.5, label="rock type")
            ax_raw.plot(s_range, f2(s_range), "k--", lw=2.5)
        title = "Water-oil" if kind == "wo" else "Gas-oil"
        ax_raw.set(title=f"{title}: lab points and rock type", xlabel=xlabel, ylabel="kr",
                   xlim=(0, 1), ylim=(0, 1.05))
        ax_norm.set(title=f"{title}: normalized, pooled fit", xlabel="Sn", ylabel="kr / kr_end",
                    xlim=(0, 1), ylim=(0, 1.05))
        for ax in (ax_raw, ax_norm):
            ax.grid(alpha=0.4)
            if ax.has_data():
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(filename, dpi=110)
    plt.close(fig)


def _plot_correlations(plt, result, filename):
    keys = [("wo", "swl"), ("wo", "sorw"), ("wo", "krwr"),
            ("go", "sgcr"), ("go", "sorg"), ("go", "krgr")]
    keys = [k for k in keys if any(s.kind == k[0] and k[1] in s.endpoints and s.perm > 0
                                   for s in result.samples)]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.6), squeeze=False)
    groups = sorted(result.groups)
    cmap = plt.get_cmap("tab10")
    corr = {c["endpoint"]: c for c in result.correlations}
    for ax, (kind, key) in zip(axes[0], keys):
        for i, g in enumerate(groups):
            pts = [(s.perm, s.endpoints[key]) for s in result.groups[g]
                   if s.kind == kind and key in s.endpoints and s.perm > 0]
            if pts:
                ax.semilogx(*zip(*pts), "o", color=cmap(i % 10), label=g)
        if key in corr:
            c = corr[key]
            k = np.logspace(*np.log10(ax.get_xlim()), 50)
            ax.semilogx(k, c["a"] + c["b"] * np.log10(k), "k--", lw=1,
                        label=f"{c['a']:.3f}{c['b']:+.3f}·lg k, R²={c['r2']:.2f}")
        ax.set(title=key, xlabel="k, mD")
        ax.grid(which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(filename, dpi=110)
    plt.close(fig)


def denormalized_pc_points(pcs, rt: RockType):
    """Lab Pc points moved onto the rock type: Sw from the sample's Swn and the rock type's
    Swl, Pc from the sample's J and the rock type's k/phi (or reservoir Pc as is)."""
    sw, pc = [], []
    for s in pcs:
        sn = s.raw["sn"].to_numpy(float)
        m = (sn > 0) & (sn < 1)
        sw.append(rt.swl + sn[m] * (1 - rt.swl))
        if isinstance(rt.pcow, LeverettJ) and "J" in s.raw:
            pc.append(s.raw["J"].to_numpy(float)[m] * rt.pcow.factor(rt.pc_units))
        else:
            pc.append(s.raw["pc_res_bar"].to_numpy(float)[m] * 1e5 * PRESSURE_UNITS[rt.pc_units])
    return np.concatenate(sw), np.concatenate(pc)


def _sample_colors(plt, samples):
    """Colors by lg k (with a colorbar mappable) when there are many samples, else tab10."""
    import matplotlib as mpl

    perms = [s.perm for s in samples]
    if len(samples) <= 10 or not all(k > 0 for k in perms):
        cmap = plt.get_cmap("tab10")
        return [cmap(i % 10) for i in range(len(samples))], None
    norm = mpl.colors.LogNorm(min(perms), max(perms))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    return [sm.to_rgba(k) for k in perms], sm


def _plot_capillary(plt, name, pcs, rt, opts: PcOptions, filename, fits=None):
    fig, (ax_lab, ax_j, ax_rt) = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(f"Capillary pressure QC — {name} ({len(pcs)} samples)", fontsize=13)
    colors, mappable = _sample_colors(plt, pcs)
    legend = mappable is None
    excluded_labeled = False
    for s, c in zip(pcs, colors):
        d = s.raw
        ax_lab.semilogy(d["sw"], d["pc_lab_bar"].where(d["pc_lab_bar"] > 0), "o-", mfc="none",
                        color=c, lw=0.8, ms=4,
                        label=f"{s.sample} ({s.endpoints['system']})" if legend else None)
        y = d["J"] if "J" in d else d["pc_res_bar"]
        used = (d["sn"] > 0) & (d["sn"] < 1) & (y > 0)
        ax_j.semilogy(d["sn"][used], y[used], "o", mfc="none", color=c, ms=4,
                      label=s.sample if legend else None)
        if (~used & (y > 0)).any():  # Swn = 0 or 1: not fitted
            ax_j.semilogy(d["sn"][~used & (y > 0)], y[~used & (y > 0)], "x", color="gray",
                          label="not used (Swn = 0 or 1)" if not excluded_labeled else None)
            excluded_labeled = True
    if mappable is not None:
        for ax in (ax_lab, ax_j):
            fig.colorbar(mappable, ax=ax, label="k, mD", pad=0.01)
    sn = np.linspace(0.005, 1, 200)
    if isinstance(rt.pcow, LeverettJ):
        m = rt.pcow
        for form, f in (fits or {m.form: {"a": m.a, "b": m.b, "rmse_swn": float("nan")}}).items():
            jj = f["a"] * (np.exp(-f["b"] * sn) if form == "exp" else sn ** -f["b"])
            expr = (f"{f['a']:.3g}·exp(−{f['b']:.3g}·Swn)" if form == "exp"
                    else f"{f['a']:.3g}·Swn^−{f['b']:.3g}")
            chosen = form == m.form
            ax_j.semilogy(sn, jj, "k-" if chosen else "k--", lw=2.2 if chosen else 1.2,
                          label=f"J = {expr}, rmse Swn {f['rmse_swn']:.3f}"
                          + (" (used)" if chosen else ""))
        ax_j.set(ylabel="Leverett J (reservoir)")
        info = f"k = {m.perm:.3g} mD, φ = {m.poro:.3g}"
    else:
        m = rt.pcow
        ax_j.semilogy(sn, m.pe * 1e-5 / PRESSURE_UNITS[rt.pc_units] * sn ** (-1 / m.lam), "k-",
                      lw=2, label=f"pe = {m.pe:.3g} {rt.pc_units}, λ = {m.lam:.3g}")
        ax_j.set(ylabel="reservoir Pc, bar")
        info = "Brooks-Corey (no k/φ)"
    sw_pts, pc_pts = denormalized_pc_points(pcs, rt)
    sw = np.linspace(rt.swl, 1, 300)
    ax_rt.plot(sw_pts, pc_pts, "o", mfc="none", color="gray", ms=4,
               label="lab, moved to rock type")
    ax_rt.plot(sw, rt.pc_ow(sw), "k-", lw=2, label="rock type Pcow")
    top = float(np.nanpercentile(np.r_[pc_pts, rt.pc_ow(1.0)], 98)) * 1.3
    top = max(top, 1.5 * float(rt.pc_ow(1.0)))
    ax_rt.set_ylim(0, top)
    ax_h = ax_rt.twinx()
    ax_h.set_ylim(0, float(height_above_fwl(top, opts.drho, rt.pc_units)))
    ax_h.set_ylabel(f"height above FWL, m (Δρ = {opts.drho:g} kg/m³)")
    ax_lab.set(title="Lab Pc (as measured, bar)", xlabel="Sw", ylabel="Pc lab, bar", xlim=(0, 1))
    ax_j.set(title=f"Pooled fit: {info}", xlabel="Swn = (Sw − Swirr) / (1 − Swirr)", xlim=(0, 1))
    ax_rt.set(title=f"Reservoir oil-water Pc (σ={opts.ift_res:g}, θ={opts.theta_res:g}°)",
              xlabel="Sw", ylabel=f"Pcow, {rt.pc_units}", xlim=(0, 1))
    for ax in (ax_lab, ax_j, ax_rt):
        ax.grid(which="both", alpha=0.3)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(filename, dpi=110)
    plt.close(fig)
