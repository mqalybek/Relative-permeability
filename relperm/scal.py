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

import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .models import RockType, normalize

# --------------------------------------------------------------------- reading

ALIASES = {
    "sample": ["sample", "sample id", "sampleid", "sample no", "plug", "plug id", "core", "core id",
               "образец", "№ образца", "номер образца", "№ обр", "обр", "проба", "шифр образца"],
    "rock_type": ["rock type", "rocktype", "rt", "satnum", "facies", "litho", "lithotype",
                  "тип породы", "тип", "литотип", "фация", "петротип"],
    "perm": ["perm", "permeability", "k", "kabs", "k abs", "ka", "kair", "kg", "kl",
             "проницаемость", "кпр", "кпр абс", "кабс", "кпр газ"],
    "poro": ["poro", "porosity", "phi", "пористость", "кп", "кпо"],
    "swi": ["swi", "swirr", "swc", "sw irr", "кво", "sво", "остаточная водонасыщенность"],
    "sw": ["sw", "s w", "water saturation", "кв", "sв", "водонасыщенность"],
    "krw": ["krw", "kr w", "krwater", "офп воды", "офпв", "кфпв", "кпрв отн", "фп воды"],
    "kro": ["kro", "krow", "kr o", "kroil", "офп нефти", "офпн", "кфпн", "кпрн отн", "фп нефти"],
    "sg": ["sg", "s g", "gas saturation", "кг", "sг", "газонасыщенность"],
    "krg": ["krg", "kr g", "krgas", "офп газа", "офпг", "кфпг", "фп газа"],
    "krog": ["krog", "kro g", "офп нефти в системе газ-нефть", "офпнг"],
}
TEXT_COLUMNS = ("sample", "rock_type")
META_COLUMNS = ("perm", "poro", "rock_type", "swi")
FRACTION_COLUMNS = ("sw", "sg", "swi", "poro", "krw", "kro", "krg", "krog")

# Cyrillic letters that look like Latin ones, so "Kв" (Latin K) matches "кв".
_LOOKALIKE = str.maketrans("кораснхе", "kopacnxe")


def _norm(name) -> str:
    s = str(name).strip().lower()
    s = re.split(r"[\(\[,]", s)[0]  # drop units: "Sw, д.ед.", "k (mD)"
    s = re.sub(r"[\s_\-.:;№#/]", "", s)
    return s.translate(_LOOKALIKE)


ALIAS_INDEX = {_norm(a): canon for canon, names in ALIASES.items() for a in names}


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


def _find_header(raw) -> int | None:
    for i in range(min(25, len(raw))):
        hits = {ALIAS_INDEX.get(_norm(c)) for c in raw.iloc[i] if str(c).strip()} - {None}
        if len(hits) >= 2:
            return i
    return None


def _sample_id(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def _to_number(series):
    pd = _pandas()
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False).str.replace(" ", "")
    return pd.to_numeric(s, errors="coerce")


def _standardize(raw, source: str):
    """Raw sheet (no header) -> table with canonical columns, or None."""
    pd = _pandas()
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    hdr = _find_header(raw)
    if hdr is None:
        return None
    columns, keep = [], []
    for j, c in enumerate(raw.iloc[hdr]):
        canon = ALIAS_INDEX.get(_norm(c))
        if canon and canon not in columns:
            columns.append(canon)
            keep.append(j)
    df = raw.iloc[hdr + 1:, keep].copy()
    df.columns = columns
    for c in columns:
        if c in TEXT_COLUMNS:
            df[c] = df[c].where(df[c].notna() & (df[c].astype(str).str.strip() != ""))
        else:
            df[c] = _to_number(df[c])
    # merged cells / IDs written once per block
    for c in ("sample", "rock_type", "perm", "poro", "swi"):
        if c in df and ("sw" in df or "sg" in df):
            df[c] = df[c].ffill()
    if ("sg" in df or "krg" in df) and "krog" not in df and "kro" in df:
        df = df.rename(columns={"kro": "krog"})
    if "sample" not in df:
        df["sample"] = source
    df["sample"] = df["sample"].map(_sample_id, na_action="ignore")
    for c in FRACTION_COLUMNS:  # percent -> fraction
        if c in df and df[c].max(skipna=True) > 1.5:
            df[c] = df[c] / 100.0
    return pd.DataFrame(df)


def _kind(df) -> str | None:
    if "sw" in df and ("krw" in df or "kro" in df):
        return "wo"
    if "sg" in df and ("krg" in df or "krog" in df):
        return "go"
    if any(c in df for c in META_COLUMNS):
        return "meta"
    return None


@dataclass
class ScalData:
    water_oil: object  # pandas DataFrame: sample, Sw, krw, kro (+ meta)
    gas_oil: object  # pandas DataFrame: sample, Sg, krg, krog (+ meta)
    sources: list = field(default_factory=list)


def read_scal(*paths: str) -> ScalData:
    """Read SCAL tables from CSV/Excel files (all sheets) and merge sample metadata."""
    pd = _pandas()
    parts = {"wo": [], "go": [], "meta": []}
    sources = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
            sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        else:
            sheets = {stem: pd.read_csv(path, header=None, sep=_csv_separator(path),
                                        dtype=str, encoding="utf-8-sig")}
        for name, raw in sheets.items():
            df = _standardize(raw, str(name))
            kind = _kind(df) if df is not None else None
            sources.append((path, str(name), kind or "skipped"))
            if kind:
                parts[kind].append(df)

    def cat(frames, sat):
        if not frames:
            return pd.DataFrame(columns=["sample", sat])
        return pd.concat(frames, ignore_index=True).dropna(subset=[sat])

    wo, go = cat(parts["wo"], "sw"), cat(parts["go"], "sg")
    if parts["meta"]:
        meta = (pd.concat(parts["meta"], ignore_index=True).dropna(subset=["sample"])
                .groupby("sample").first())
        for df in (wo, go):
            for c in META_COLUMNS:
                if c in meta:
                    fill = df["sample"].map(meta[c])
                    df[c] = df[c].fillna(fill) if c in df else fill
    if wo.empty and go.empty:
        raise ValueError("No relative permeability tables found (need Sw + krw/kro or Sg + krg/krog)")
    return ScalData(wo, go, sources)


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
    kind: str  # "wo" or "go"
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
                     endpoints: str = "mean", align_kro: bool = True) -> ScalResult:
    """Analyze every sample and build one RockType per group.

    align_kro: set krog(Sg=0) equal to krow(Swl). Both are kro at Swi, but they are
    averaged over different sample sets; Eclipse needs them equal for Stone models.

    group_by: "rock_type" (column), "perm" (perm_bins), "none" (one group) or "auto"
    (rock_type column if present, else perm bins if given, else one group).
    """
    wo, go = data.water_oil, data.gas_oil
    if group_by == "auto":
        has_rt = any("rock_type" in df and df["rock_type"].notna().any() for df in (wo, go))
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

    groups: dict = {}
    for s in samples:
        groups.setdefault(s.group, []).append(s)

    rock_types, warnings = [], []
    for name, members in groups.items():
        w = [s for s in members if s.kind == "wo"]
        g = [s for s in members if s.kind == "go"]
        agg = lambda ss, key: _aggregate([s.endpoints.get(key, math.nan) for s in ss], endpoints)  # noqa: E731
        params = {"name": name}
        if w:
            params.update(swl=agg(w, "swl"), sowcr=agg(w, "sorw"),
                          krwr=agg(w, "krwr"), krocw=agg(w, "krocw"),
                          nw=_pooled(w, "krw"), now=_pooled(w, "kro"))
            params["swcr"] = params["swl"]
        elif g:
            params["swl"] = params["swcr"] = agg(g, "swl")
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
        params = {k: v for k, v in params.items() if not (isinstance(v, float) and math.isnan(v))}
        params["description"] = (f"SCAL: {len(w)} water-oil, {len(g)} gas-oil samples, "
                                 f"{endpoints} endpoints")
        try:
            rt = RockType(**params)
        except ValueError as e:
            warnings.append(f"{name}: cannot build rock type ({e})")
            continue
        rock_types.append(rt)
        warnings += [f"{name}: {x}" for x in rt.warnings()]

    return ScalResult(samples, rock_types, groups, endpoint_correlations(samples), warnings)


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
        row = {"sample": s.sample, "test": "water-oil" if s.kind == "wo" else "gas-oil",
               "group": s.group, "perm": s.perm, "poro": s.poro, "points": s.points}
        row.update(s.endpoints)
        row.update(s.exponents)
        row.update({f"rmse_{k}": v for k, v in s.rmse.items()})
        row["notes"] = "; ".join(s.notes)
        rows.append(row)
    return rows


def write_report(result: ScalResult, out_dir: str, points: int = 20) -> list[str]:
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

    if any(s.perm > 0 for s in result.samples):
        _plot_correlations(plt, result, path("endpoints_vs_perm.png"))
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
