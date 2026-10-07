"""Petrophysical dependencies from SCAL: one property table per sample and regressions
(porosity-permeability, Swi / Sor / displacement efficiency / kr endpoints against lg k, ...),
with data checks. Labels follow Russian core-analysis reports (Кпр, Кп, Кво, Кно, Квыт).
"""

from __future__ import annotations

import math
import os

import numpy as np

LABELS = {
    "perm": "Кпр по газу, мД",
    "perm_w": "Кпр по воде, мД",
    "poro": "Кп, д.ед.",
    "swi": "Кво, д.ед.",
    "sor": "Кно, д.ед.",
    "kdisp": "Квыт, д.ед.",
    "krw_end": "krw(Кно)",
    "kro_end": "kro(Кво)",
    "nw": "nw (Кори)",
    "now": "no (Кори)",
    "swirr_pc": "Кво по Pc, д.ед.",
    "amott": "Индекс Амотта",
}
SHORT = {"perm": "Кпр", "perm_w": "Кпр.в", "poro": "Кп", "swi": "Кво", "sor": "Кно",
         "kdisp": "Квыт", "krw_end": "krw", "kro_end": "kro", "nw": "nw", "now": "no",
         "swirr_pc": "Кво(Pc)", "amott": "Амотт"}

# (y, x, x in log10, y in log10); x "k" = the permeability column with more points
DEPENDENCIES = [
    ("perm", "poro", False, True),
    ("swi", "k", True, False),
    ("sor", "k", True, False),
    ("kdisp", "k", True, False),
    ("sor", "swi", False, False),
    ("krw_end", "k", True, False),
    ("nw", "k", True, False),
    ("now", "k", True, False),
    ("swirr_pc", "perm", True, False),
    ("perm_w", "perm", True, True),
    ("amott", "k", True, False),
]


def _pd():
    from .scal import _pandas

    return _pandas()


def property_table(result):
    """One row per sample: properties from every sheet plus values derived from the curves."""
    pd = _pd()
    data = result.data
    rows = {}

    def row(sample):
        return rows.setdefault(sample, {"sample": sample})

    if data is not None and data.samples is not None:
        meta = data.samples
        for sample, r in meta.iterrows():
            d = row(sample)
            for c in ("well", "horizon", "depth", "perm", "perm_w", "poro", "swi", "sor",
                      "kdisp", "amott"):
                if c in meta and pd.notna(r[c]):
                    d[c] = r[c]
            if "krw" in meta and pd.notna(r["krw"]):
                d["krw_end"] = r["krw"]
            if "kro" in meta and pd.notna(r["kro"]):
                d["kro_end"] = r["kro"]
    for s in result.samples:
        d = row(s.sample)
        first = lambda col: (s.raw[col].dropna().iloc[0]  # noqa: E731
                             if col in s.raw and s.raw[col].notna().any() else None)
        for c in ("well", "horizon", "depth", "perm", "poro"):
            if c not in d and first(c) is not None:
                d[c] = first(c)
        if s.kind == "wo":
            d.setdefault("swi", s.endpoints.get("swl"))
            d.setdefault("sor", s.endpoints.get("sorw"))
            d.setdefault("krw_end", s.endpoints.get("krwr"))
            d.setdefault("kro_end", s.endpoints.get("krocw"))
            for k in ("nw", "now"):
                if k in s.exponents:
                    d[k] = s.exponents[k]
            d["has_kr"] = True
        elif s.kind == "pc":
            d["swirr_pc"] = s.endpoints.get("swirr")
            for k in ("J_a_exp", "J_b_exp", "J_a_power", "J_b_power"):
                if k in s.exponents:
                    d[k] = s.exponents[k]
            d["has_pc"] = True
    for d in rows.values():  # displacement efficiency from Swi and Sor when not reported
        if "kdisp" not in d and d.get("swi") is not None and d.get("sor") is not None:
            d["kdisp"] = (1 - d["swi"] - d["sor"]) / (1 - d["swi"])
    df = pd.DataFrame(list(rows.values()))
    order = ["sample", "well", "horizon", "depth", "perm", "perm_w", "poro", "swi", "sor",
             "kdisp", "krw_end", "kro_end", "nw", "now", "swirr_pc", "amott"]
    cols = [c for c in order if c in df] + [c for c in df if c not in order]
    return df[cols]


def find_anomalies(props, discrepancies=()) -> list[dict]:
    """Values that look like typos or unit errors. Flagged values are left out of the fits."""
    out = [{"sample": d["sample"], "property": d["property"], "value": d["values"],
            "reason": "разные значения на разных листах"} for d in discrepancies]

    def flag(mask, prop, reason):
        for _, r in props[mask].iterrows():
            out.append({"sample": r["sample"], "property": prop, "value": r[prop],
                        "reason": reason})

    if "poro" in props:
        flag((props["poro"] > 0.45) | (props["poro"] <= 0.01), "poro",
             "пористость вне 0.01–0.45 (опечатка или единицы?)")
    if "perm" in props and "perm_w" in props:
        flag(props["perm_w"] > props["perm"] * 1.05, "perm_w",
             "проницаемость по воде больше, чем по газу")
    if "swi" in props and "sor" in props:
        flag(props["swi"] + props["sor"] >= 0.95, "sor", "Кво + Кно ≥ 0.95")
    for prop in ("swi", "sor", "kdisp", "krw_end", "kro_end", "swirr_pc"):
        if prop in props:
            flag((props[prop] < 0) | (props[prop] > 1), prop, "вне диапазона 0–1")
    return out


def _x_column(props, y, x):
    """For x == "k": the permeability column (gas or water) with more points for y."""
    if x != "k":
        return x
    counts = {c: int((props[c] > 0).mul(props[y].notna()).sum()) if c in props and y in props
              else 0 for c in ("perm", "perm_w")}
    return "perm_w" if counts["perm_w"] > counts["perm"] else "perm"


def fit_dependencies(props, anomalies=()) -> list[dict]:
    excluded = {(a["sample"], a["property"]) for a in anomalies}
    fits = []
    for y, x, xlog, ylog in DEPENDENCIES:
        x = _x_column(props, y, x)
        if x not in props or y not in props:
            continue
        d = props[["sample", "well", x, y] if "well" in props else ["sample", x, y]].dropna()
        d = d[[(s, x) not in excluded and (s, y) not in excluded for s in d["sample"]]]
        if xlog:
            d = d[d[x] > 0]
        if ylog:
            d = d[d[y] > 0]
        if len(d) < 3:
            continue
        xv = np.log10(d[x].to_numpy(float)) if xlog else d[x].to_numpy(float)
        yv = np.log10(d[y].to_numpy(float)) if ylog else d[y].to_numpy(float)
        if np.ptp(xv) == 0 or np.ptp(yv) <= 1e-3 * max(np.abs(yv).max(), 1e-12):
            continue  # constant y (e.g. krw end point fixed in the report)
        b, a = np.polyfit(xv, yv, 1)
        res = yv - (a + b * xv)
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        fits.append({
            "y": y, "x": x, "x_log": xlog, "y_log": ylog, "a": float(a), "b": float(b),
            "r2": 1 - float(np.sum(res**2)) / ss_tot, "n": len(d),
            "rmse": float(np.sqrt(np.mean(res**2))),
            "formula": _formula(y, x, xlog, ylog, a, b),
        })
    return fits


def _formula(y, x, xlog, ylog, a, b) -> str:
    ys = f"lg({SHORT[y]})" if ylog else SHORT[y]
    xs = f"lg({SHORT[x]})" if xlog else SHORT[x]
    sign = "+" if b >= 0 else "−"
    return f"{ys} = {a:.4g} {sign} {abs(b):.4g}·{xs}"


def normalized_kr(result):
    """Normalized water-oil curves per sample: Swn = (Sw − Swi)/(1 − Swi − Sor), kr / kr_end."""
    pd = _pd()
    frames = []
    for s in result.samples:
        if s.kind != "wo":
            continue
        d = s.raw
        swi, sor = s.endpoints["swl"], s.endpoints["sorw"]
        f = pd.DataFrame({"sample": s.sample, "Sw": d["sw"].to_numpy(float)})
        f["Swn"] = (f["Sw"] - swi) / (1 - swi - sor)
        if "krw" in d:
            f["krw"] = d["krw"].to_numpy(float)
            f["krw_n"] = f["krw"] / s.endpoints["krwr"]
        if "kro" in d:
            f["kro"] = d["kro"].to_numpy(float)
            f["kro_n"] = f["kro"] / s.endpoints["krocw"]
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalized_pc(result):
    """Capillary curves per sample: Swn = (Sw − Swirr)/(1 − Swirr), lab and reservoir Pc, J."""
    pd = _pd()
    frames = []
    for s in result.samples:
        if s.kind != "pc":
            continue
        d = s.raw
        f = pd.DataFrame({"sample": s.sample, "k, mD": s.perm, "phi": s.poro,
                          "system": s.endpoints["system"], "Sw": d["sw"].to_numpy(float),
                          "Swn": d["sn"].to_numpy(float),
                          "Pc lab, bar": d["pc_lab_bar"].to_numpy(float),
                          "Pc res, bar": d["pc_res_bar"].to_numpy(float)})
        if "J" in d:
            f["J"] = d["J"].to_numpy(float)
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_dependencies(plt, props, fits, anomalies, filename):
    if not fits:
        return False
    excluded = {(a["sample"], a["property"]) for a in anomalies}
    ncol = 4
    nrow = math.ceil(len(fits) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.8 * nrow), squeeze=False)
    wells = sorted(props["well"].dropna().astype(str).unique()) if "well" in props else []
    cmap = plt.get_cmap("tab10")
    for ax, f in zip(axes.flat, fits):
        x, y = f["x"], f["y"]
        d = props[["sample", x, y] + (["well"] if wells else [])].dropna(subset=[x, y])
        bad = np.array([(s, x) in excluded or (s, y) in excluded for s in d["sample"]], bool)
        groups = [(w, d[(d["well"].astype(str) == w) & ~bad]) for w in wells] if wells \
            else [("", d[~bad])]
        for i, (w, g) in enumerate(groups):
            if len(g):
                ax.plot(g[x], g[y], "o", color=cmap(i % 10), ms=5, mfc="none",
                        label=f"скв. {w}" if w else None)
        if bad.any():
            ax.plot(d[x][bad], d[y][bad], "x", color="red", ms=7, mew=2,
                    label="исключено (аномалия)")
        xs = np.linspace(d[x][~bad].min(), d[x][~bad].max(), 100)
        xv = np.log10(xs) if f["x_log"] else xs
        yv = f["a"] + f["b"] * xv
        ax.plot(xs, 10**yv if f["y_log"] else yv, "k-", lw=1.5)
        if f["x_log"]:
            ax.set_xscale("log")
        if f["y_log"]:
            ax.set_yscale("log")
        ax.set(xlabel=LABELS[x], ylabel=LABELS[y])
        ax.set_title(f"{f['formula']}\nR² = {f['r2']:.2f}, n = {f['n']}", fontsize=9)
        ax.grid(which="both", alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(fits):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=110)
    plt.close(fig)
    return True


def write_dependencies(result, out_dir: str, plt) -> list[str]:
    """Write properties, dependencies, anomalies (CSV), a plot and an Excel workbook."""
    pd = _pd()
    props = property_table(result)
    discrepancies = result.data.discrepancies if result.data is not None else []
    anomalies = find_anomalies(props, discrepancies)
    fits = fit_dependencies(props, anomalies)
    result.dependencies = {"properties": props, "fits": fits, "anomalies": anomalies}
    written = []

    def path(name):
        p = os.path.join(out_dir, name)
        written.append(p)
        return p

    props.to_csv(path("properties.csv"), index=False, float_format="%.5g")
    fits_df = pd.DataFrame(fits)
    if len(fits_df):
        fits_df.to_csv(path("dependencies.csv"), index=False, float_format="%.5g")
    if anomalies:
        pd.DataFrame(anomalies).to_csv(path("anomalies.csv"), index=False)
    if plot_dependencies(plt, props, fits, anomalies, os.path.join(out_dir, "dependencies.png")):
        written.append(os.path.join(out_dir, "dependencies.png"))

    xlsx = path("scal_report.xlsx")
    with pd.ExcelWriter(xlsx) as xw:
        props.rename(columns=LABELS).to_excel(xw, sheet_name="Образцы", index=False)
        if len(fits_df):
            out = fits_df.assign(y=fits_df["y"].map(LABELS), x=fits_df["x"].map(LABELS))
            out = out.rename(columns={"formula": "Зависимость", "y": "Y", "x": "X",
                                      "x_log": "lg X", "y_log": "lg Y", "r2": "R²",
                                      "n": "Точек", "rmse": "СКО"})
            out[["Зависимость", "Y", "X", "lg X", "lg Y", "a", "b", "R²", "Точек", "СКО"]] \
                .to_excel(xw, sheet_name="Зависимости", index=False)
        pd.DataFrame(anomalies or [{"reason": "аномалий не найдено"}]).rename(columns={
            "sample": "Образец", "property": "Параметр", "value": "Значение",
            "reason": "Причина"}).to_excel(xw, sheet_name="Аномалии", index=False)
        kr = normalized_kr(result)
        if len(kr):
            kr.to_excel(xw, sheet_name="ОФП нормированные", index=False)
        pc = normalized_pc(result)
        if len(pc):
            pc.to_excel(xw, sheet_name="Кап. давление", index=False)
        rts = pd.DataFrame([{k: (str(v) if isinstance(v, dict) else v)
                             for k, v in rt.to_dict().items()} for rt in result.rock_types])
        if len(rts):
            rts.to_excel(xw, sheet_name="Типы пород", index=False)
        fits_pc = [{"Тип пород": name, "Форма": form, "a": f["a"], "b": f["b"],
                    "СКО Swn": f["rmse_swn"], "Точек": f["used"]}
                   for name, ff in result.pc_fits.items() for form, f in ff.items()]
        if fits_pc:
            pd.DataFrame(fits_pc).to_excel(xw, sheet_name="J-функция", index=False)
    return written

