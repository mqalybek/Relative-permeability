"""Interactive matplotlib editor for rock types, with file loading and export buttons.

Start it with `relperm` (no arguments), `relperm app`, `relperm-gui` or the .bat file in the
repository. "Открыть отчёт керна…" reads a SCAL workbook, builds rock types, writes the
report next to the file and shows the curves with the lab points.
"""

from __future__ import annotations

import dataclasses
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider

from .capillary import BrooksCorey, height_above_fwl
from .export import export
from .models import RockType

# (attribute, label, min, max) per slider column
WATER_OIL = [
    ("swl", "Swl", 0.0, 0.5),
    ("swcr", "Swcr", 0.0, 0.6),
    ("sowcr", "Sorw", 0.0, 0.5),
    ("krwr", "krw(Sorw)", 0.01, 1.0),
    ("krwmax", "krw(Sw=1)", 0.01, 1.0),
    ("krocw", "krow(Swl)", 0.01, 1.0),
    ("nw", "nw", 0.5, 8.0),
    ("now", "now", 0.5, 8.0),
]
GAS_OIL = [
    ("sgcr", "Sgcr", 0.0, 0.4),
    ("sogcr", "Sorg", 0.0, 0.5),
    ("krgr", "krg(Sorg)", 0.01, 1.0),
    ("krgmax", "krg(Sw=Swl)", 0.01, 1.0),
    ("krogcg", "krog(Sg=0)", 0.01, 1.0),
    ("ng", "ng", 0.5, 8.0),
    ("nog", "nog", 0.5, 8.0),
]

# editable Pc model fields: (field, label suffix, min, max); None bounds are set from units
PC_FIELDS = {
    "brooks-corey": [("pe", "pe", 0.0, None), ("lam", "λ", 0.3, 6.0), ("pcmax", "Pcmax", 0.0, None)],
    "leverett": [("a", "J a", 0.0, None), ("b", "J b", 0.1, 10.0),
                 ("perm", "k, mD", 1.0, None), ("pcmax", "Pcmax", 0.0, None)],
}
UNIT_SCALE = {"bar": 1.0, "atm": 1.0, "psi": 14.5, "kPa": 100.0, "MPa": 0.1}

COLORS = {"w": "#1f77b4", "o": "#2ca02c", "g": "#d62728", "fw": "#9467bd"}


SCAL_CHOICES = {  # option -> [(value, label shown in the dialog)]
    "pc_lab_units": [("MPa", "МПа"), ("bar", "бар"), ("atm", "атм"), ("kPa", "кПа"),
                     ("psi", "psi")],
    "pc_system": [("oil-brine", "нефть (керосин) – вода"), ("air-brine", "газ – вода"),
                  ("mercury-air", "ртуть – воздух (MICP)")],
    "group_by": [("none", "все образцы вместе"), ("well", "по скважинам"),
                 ("horizon", "по горизонтам"), ("rock_type", "по типу пород (колонка)"),
                 ("perm", "по классам проницаемости 10/100 мД")],
}


def scal_session(path: str, pc_lab_units: str = "MPa", pc_system: str = "oil-brine",
                 group_by: str = "none", drho: float = 250.0, out_dir: str | None = None):
    """Process a SCAL workbook like `relperm scal` and return what the editor shows.

    Returns (items, out_dir) where items is a list of (rock type, lab points) and out_dir
    holds the report, plots, rock_types.json and SWOF/SGOF tables for all rock types.
    """
    import pandas as pd

    from .scal import PcOptions, build_rock_types, denormalized_pc_points, read_scal, write_report

    pc = PcOptions(lab_units=pc_lab_units, system=pc_system, drho=drho)
    res = build_rock_types(read_scal(path), group_by=group_by,
                           perm_bins=[10.0, 100.0] if group_by == "perm" else (), pc=pc)
    out_dir = out_dir or os.path.splitext(path)[0] + "_relperm"
    write_report(res, out_dir, pc=pc)
    with open(os.path.join(out_dir, "relperm_eclipse.inc"), "w", encoding="utf-8") as f:
        f.write(export(res.rock_types, "eclipse"))
    items = []
    for rt in res.rock_types:
        members = res.groups[rt.name]
        labs = {}
        wo = [s.raw for s in members if s.kind == "wo"]
        if wo:
            d = pd.concat(wo)
            labs["lab_wo"] = {"Sw": d["sw"].to_numpy(float)}
            for src, dst in (("krw", "krw"), ("kro", "krow")):
                if src in d:
                    labs["lab_wo"][dst] = d[src].to_numpy(float)
        go = [s.raw for s in members if s.kind == "go"]
        if go:
            d = pd.concat(go)
            labs["lab_go"] = {"Sg": d["sg"].to_numpy(float)}
            for c in ("krg", "krog"):
                if c in d:
                    labs["lab_go"][c] = d[c].to_numpy(float)
        pcs = [s for s in members if s.kind == "pc"]
        if pcs and rt.pcow is not None:
            sw, pcow = denormalized_pc_points(pcs, rt)
            labs["lab_pc"] = {"Sw": sw, "Pcow": pcow}
        items.append((rt, labs))
    return items, out_dir


class RelPermApp:
    def __init__(self, rock_type: RockType | None = None, points: int = 20,
                 out_dir: str = ".", lab_wo: dict | None = None, lab_go: dict | None = None,
                 lab_pc: dict | None = None, session: list | None = None, index: int = 0,
                 drho: float = 250.0):
        # session: [(rock type, {"lab_wo": ..., "lab_go": ..., "lab_pc": ...}), ...]
        self.session, self.index = session, index
        if session:
            rock_type, labs = session[index]
            lab_wo, lab_go, lab_pc = labs.get("lab_wo"), labs.get("lab_go"), labs.get("lab_pc")
        self.initial = rock_type or RockType()
        self.rt = self.initial
        self.points = points
        self.out_dir = out_dir
        self.mu_w, self.mu_o = 0.5, 2.0
        self.drho = drho
        self.log_scale = False
        # Pc models edited by sliders; a missing model starts as zero Brooks-Corey
        self.pc_models = {
            "pcow": self.initial.pcow or BrooksCorey(pe=0.0, lam=2.0, pcmax=0.0),
            "pcog": self.initial.pcog or BrooksCorey(pe=0.0, lam=2.0, pcmax=0.0),
        }

        self.fig = plt.figure(figsize=(18, 9.5))
        count = f" ({index + 1}/{len(session)})" if session and len(session) > 1 else ""
        self.fig.canvas.manager.set_window_title(f"relperm — {self.rt.name}{count}")
        self.ax_wo = self.fig.add_axes([0.04, 0.50, 0.2, 0.38])
        self.ax_go = self.fig.add_axes([0.28, 0.50, 0.2, 0.38])
        self.ax_pc = self.fig.add_axes([0.52, 0.50, 0.2, 0.38])
        self.ax_h = self.ax_pc.twinx()
        self.ax_fw = self.fig.add_axes([0.79, 0.50, 0.2, 0.38])
        self.fig.text(0.40, 0.975, f"Тип пород: {self.rt.name}{count}", fontsize=11,
                      weight="bold", va="center")
        self.status = self.fig.text(0.40, 0.945, "", fontsize=9, va="center")
        self._add_file_buttons()

        s = np.linspace(0, 1, 401)
        self.s = s
        self.l_krw, = self.ax_wo.plot(s, s, lw=2, color=COLORS["w"], label="krw")
        self.l_krow, = self.ax_wo.plot(s, s, lw=2, color=COLORS["o"], label="krow")
        self.l_krg, = self.ax_go.plot(s, s, lw=2, color=COLORS["g"], label="krg")
        self.l_krog, = self.ax_go.plot(s, s, lw=2, color=COLORS["o"], label="krog")
        self.l_fw, = self.ax_fw.plot(s, s, lw=2, color=COLORS["fw"], label="fw")
        self.l_pcow, = self.ax_pc.plot(s, s, lw=2, color=COLORS["w"], label="Pcow(Sw)")
        self.l_pcog, = self.ax_pc.plot(s, s, lw=2, color=COLORS["g"], label="Pcog(Sg)")
        self.p_wo, = self.ax_wo.plot([], [], "o", ms=4, color="k", alpha=0.6, label="table")
        self.p_go, = self.ax_go.plot([], [], "o", ms=4, color="k", alpha=0.6, label="table")
        self._plot_lab(lab_wo, lab_go)
        if lab_pc:
            self.ax_pc.plot(lab_pc["Sw"], lab_pc["Pcow"], "s", mfc="none", color=COLORS["w"],
                            label="Pcow lab")

        for ax, title, xl in ((self.ax_wo, "Water-oil (SWOF)", "Sw"),
                              (self.ax_go, "Gas-oil (SGOF)", "Sg"),
                              (self.ax_pc, "Capillary pressure", "Sw (Pcow) / Sg (Pcog)"),
                              (self.ax_fw, "Water fractional flow", "Sw")):
            ax.set_title(title)
            ax.set_xlabel(xl)
            ax.set_xlim(0, 1)
            ax.grid(which="both", alpha=0.4)
        self.ax_wo.set_ylabel("kr")
        self.ax_fw.set_ylabel("fw")
        self.ax_fw.set_ylim(0, 1.02)
        self.ax_pc.set_ylabel(f"Pc, {self.initial.pc_units}")
        self.ax_h.set_ylabel("height above FWL for Pcow, m")
        for ax in (self.ax_wo, self.ax_go, self.ax_pc, self.ax_fw):
            ax.legend(loc="best", fontsize=8)

        self.sliders: dict[str, Slider] = {}
        self._add_column(WATER_OIL, x=0.06)
        self._add_column(GAS_OIL, x=0.30)
        self.pc_sliders: dict[tuple[str, str], Slider] = {}
        self._add_pc_column(x=0.56)
        self._add_controls(x=0.80)

        self.update()

    # ----------------------------------------------------------------- layout
    def _add_column(self, specs, x):
        for i, (attr, label, lo, hi) in enumerate(specs):
            value = getattr(self.initial, attr)
            ax = self.fig.add_axes([x, 0.38 - i * 0.045, 0.13, 0.025])
            if not isinstance(value, float):  # LET shape: not editable here
                ax.axis("off")
                ax.text(0, 0.5, f"{label}: {value} (fixed)", fontsize=8, va="center")
                continue
            slider = Slider(ax, label, lo, hi, valinit=float(np.clip(value, lo, hi)))
            slider.on_changed(lambda _: self.update())
            self.sliders[attr] = slider

    def _add_pc_column(self, x):
        scale = UNIT_SCALE[self.initial.pc_units]
        i = 0
        for key, model in self.pc_models.items():
            for name, label, lo, hi in PC_FIELDS[model.model]:
                value = getattr(model, name)
                if value is None:  # uncapped: start from the floored value, same curve
                    value = float(model.pc(0.0, self.initial.pc_units))
                if hi is None:
                    hi = {"pe": 2 * scale, "pcmax": 10 * scale, "a": 1.0, "perm": 1000.0}[name]
                    hi = max(hi, 1.5 * value)
                ax = self.fig.add_axes([x, 0.38 - i * 0.045, 0.13, 0.025])
                slider = Slider(ax, f"{key[:2].capitalize()}{key[2:]} {label}", lo, hi,
                                valinit=float(np.clip(value, lo, hi)))
                slider.on_changed(lambda _: self.update())
                self.pc_sliders[(key, name)] = slider
                i += 1

    def _add_controls(self, x):
        ax = self.fig.add_axes([x, 0.38, 0.13, 0.025])
        self.s_mu = Slider(ax, "μo/μw", 0.1, 100, valinit=self.mu_o / self.mu_w, valfmt="%.1f")
        self.s_mu.on_changed(lambda _: self.update())
        ax = self.fig.add_axes([x, 0.335, 0.13, 0.025])
        self.s_drho = Slider(ax, "Δρ w-o, kg/m³", 50, 1000, valinit=self.drho, valfmt="%.0f")
        self.s_drho.on_changed(lambda _: self.update())

        ax = self.fig.add_axes([x, 0.25, 0.12, 0.06], frameon=False)
        self.chk = CheckButtons(ax, ["log scale kr"], [False])
        self.chk.on_clicked(self._toggle_log)

        self.buttons = []
        for i, (label, cb) in enumerate((("Export SWOF/SGOF", lambda _: self._export("eclipse")),
                                         ("Export SWFN/SGFN/SOF3",
                                          lambda _: self._export("eclipse2")),
                                         ("Reset", lambda _: self._reset()))):
            b = Button(self.fig.add_axes([x, 0.17 - i * 0.06, 0.12, 0.045]), label)
            b.on_clicked(cb)
            self.buttons.append(b)

    def _add_file_buttons(self):
        specs = [("Открыть отчёт керна…", self._open_scal, 0.01, 0.15),
                 ("Открыть .json…", self._open_json, 0.17, 0.09)]
        if self.session and len(self.session) > 1:
            specs.append(("Тип пород ▶", lambda: self._show(self.index + 1), 0.27, 0.09))
        self.file_buttons = []
        for label, cb, x, w in specs:
            b = Button(self.fig.add_axes([x, 0.95, w, 0.04]), label, color="#e8f0fe",
                       hovercolor="#c9dcfb")
            b.on_clicked(lambda _, cb=cb: cb())
            self.file_buttons.append(b)

    # ------------------------------------------------------------ file loading
    def _show(self, index: int, session: list | None = None, out_dir: str | None = None,
              drho: float | None = None):
        """Replace this window with one showing session[index]."""
        session = session if session is not None else self.session
        new = RelPermApp(points=self.points, out_dir=out_dir or self.out_dir,
                         session=session, index=index % len(session),
                         drho=drho if drho is not None else float(self.s_drho.val))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                new.fig.show()
            except Exception:  # non-interactive backend (tests)
                pass
        plt.close(self.fig)
        return new

    def _tk_parent(self):
        import tkinter as tk

        window = getattr(self.fig.canvas.manager, "window", None)
        if isinstance(window, (tk.Tk, tk.Toplevel)):
            return window, False
        root = tk.Tk()
        root.withdraw()
        return root, True

    def _ask_file(self, title, filetypes):
        from tkinter import filedialog

        parent, temporary = self._tk_parent()
        try:
            return filedialog.askopenfilename(parent=parent, title=title, filetypes=filetypes)
        finally:
            if temporary:
                parent.destroy()

    def _ask_scal_options(self, filename):
        """Small dialog: lab Pc units, lab fluid system, grouping, density difference."""
        import tkinter as tk
        from tkinter import ttk

        parent, temporary = self._tk_parent()
        dlg = tk.Toplevel(parent)
        dlg.title("Параметры обработки")
        dlg.resizable(False, False)
        ttk.Label(dlg, text=os.path.basename(filename), font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, padx=12, pady=(12, 6), sticky="w")
        rows = [("Единицы Pc в отчёте (если нет в заголовке)", "pc_lab_units"),
                ("Флюиды в лаборатории (Pc)", "pc_system"),
                ("Типы пород", "group_by")]
        boxes = {}
        for i, (text, key) in enumerate(rows, start=1):
            ttk.Label(dlg, text=text).grid(row=i, column=0, padx=12, pady=4, sticky="w")
            labels = [lbl for _, lbl in SCAL_CHOICES[key]]
            box = ttk.Combobox(dlg, values=labels, state="readonly", width=32)
            box.current(0)
            box.grid(row=i, column=1, padx=12, pady=4)
            boxes[key] = box
        ttk.Label(dlg, text="Δρ вода – нефть, кг/м³").grid(row=4, column=0, padx=12, pady=4,
                                                         sticky="w")
        drho = ttk.Entry(dlg, width=10)
        drho.insert(0, "250")
        drho.grid(row=4, column=1, padx=12, pady=4, sticky="w")
        result = {}

        def ok():
            for key, box in boxes.items():
                result[key] = SCAL_CHOICES[key][box.current()][0]
            try:
                result["drho"] = float(drho.get().replace(",", "."))
            except ValueError:
                result["drho"] = 250.0
            dlg.destroy()

        buttons = ttk.Frame(dlg)
        buttons.grid(row=5, column=0, columnspan=2, pady=12)
        ttk.Button(buttons, text="Обработать", command=ok).pack(side="left", padx=6)
        ttk.Button(buttons, text="Отмена", command=dlg.destroy).pack(side="left", padx=6)
        dlg.transient(parent)
        dlg.grab_set()
        parent.wait_window(dlg)
        if temporary:
            parent.destroy()
        return result or None

    def _set_status(self, text, color):
        self.status.set_text(text)
        self.status.set_color(color)
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def _open_scal(self):
        path = self._ask_file("Отчёт керна (SCAL)", [("Excel / CSV", "*.xlsx *.xlsm *.xls *.csv"),
                                                     ("Все файлы", "*.*")])
        if not path:
            return
        options = self._ask_scal_options(path)
        if not options:
            return
        self.load_scal(path, **options)

    def load_scal(self, path, **options):
        self._set_status(f"Обработка {os.path.basename(path)}…", "black")
        try:
            items, out_dir = scal_session(path, **options)
        except Exception as e:  # show any reading problem in the window
            self._set_status(f"Не удалось обработать файл: {e}", "red")
            return None
        if not items:
            self._set_status("В файле не найдено ОФП / Pc", "red")
            return None
        new = self._show(0, items, out_dir, drho=options.get("drho"))
        new._set_status(f"Готово. Отчёт, графики и SWOF: {out_dir}", "green")
        return new

    def _open_json(self):
        path = self._ask_file("Типы пород (.json)", [("JSON", "*.json"), ("Все файлы", "*.*")])
        if path:
            self.load_json(path)

    def load_json(self, path):
        from .cli import load_config

        try:
            rock_types, points = load_config(path)
        except Exception as e:
            self._set_status(f"Не удалось открыть {os.path.basename(path)}: {e}", "red")
            return None
        self.points = points
        return self._show(0, [(rt, {}) for rt in rock_types], os.path.dirname(path) or ".")

    def _plot_lab(self, lab_wo, lab_go):
        if lab_wo:
            for key, color in (("krw", COLORS["w"]), ("krow", COLORS["o"])):
                if key in lab_wo:
                    self.ax_wo.plot(lab_wo["Sw"], lab_wo[key], "s", mfc="none",
                                    color=color, label=f"{key} lab")
            if "Pcow" in lab_wo:
                self.ax_pc.plot(lab_wo["Sw"], lab_wo["Pcow"], "s", mfc="none",
                                color=COLORS["w"], label="Pcow lab")
        if lab_go:
            for key, color in (("krg", COLORS["g"]), ("krog", COLORS["o"])):
                if key in lab_go:
                    self.ax_go.plot(lab_go["Sg"], lab_go[key], "s", mfc="none",
                                    color=color, label=f"{key} lab")
            if "Pcog" in lab_go:
                self.ax_pc.plot(lab_go["Sg"], lab_go["Pcog"], "s", mfc="none",
                                color=COLORS["g"], label="Pcog lab")

    # --------------------------------------------------------------- behavior
    def _build(self) -> RockType:
        values = {k: float(s.val) for k, s in self.sliders.items()}
        values["krwmax"] = max(values.get("krwmax", 0), values.get("krwr", self.initial.krwr))
        values["krgmax"] = max(values.get("krgmax", 0), values.get("krgr", self.initial.krgr))
        for key, model in self.pc_models.items():
            fields = {name: float(s.val) for (k, name), s in self.pc_sliders.items() if k == key}
            if "pe" in fields:
                fields["pcmax"] = max(fields["pcmax"], fields["pe"])
            values[key] = dataclasses.replace(model, **fields)
        return dataclasses.replace(self.initial, **values)

    def update(self):
        try:
            rt = self._build()
        except ValueError as e:
            self.status.set_text(str(e).split(": ", 1)[-1])
            self.status.set_color("red")
            self.fig.canvas.draw_idle()
            return
        self.rt = rt
        s = self.s
        sw_mask = s >= rt.swl
        sg_max = 1.0 - rt.swl
        self.l_krw.set_data(s[sw_mask], rt.krw(s[sw_mask]))
        self.l_krow.set_data(s[sw_mask], rt.krow(s[sw_mask]))
        sg = s[(s >= rt.sgl) & (s <= sg_max)]
        self.l_krg.set_data(sg, rt.krg(sg))
        self.l_krog.set_data(sg, rt.krog(sg))
        mu_ratio = self.s_mu.val
        self.l_fw.set_data(s[sw_mask], rt.fractional_flow(s[sw_mask], mu_w=1.0, mu_o=mu_ratio))

        pcow, pcog = rt.pc_ow(s[sw_mask]), rt.pc_og(sg)
        self.l_pcow.set_data(s[sw_mask], pcow)
        self.l_pcog.set_data(sg, pcog)
        pc_top = 1.05 * max(pcow.max(), pcog.max(), 1e-3 * UNIT_SCALE[rt.pc_units])
        self.ax_pc.set_ylim(0, pc_top)
        self.ax_h.set_ylim(0, float(height_above_fwl(pc_top, self.s_drho.val, rt.pc_units)))

        wo, go = rt.water_oil_table(self.points), rt.gas_oil_table(self.points)
        self.p_wo.set_data(np.r_[wo["Sw"], wo["Sw"]], np.r_[wo["krw"], wo["krow"]])
        self.p_go.set_data(np.r_[go["Sg"], go["Sg"]], np.r_[go["krg"], go["krog"]])

        for ax in (self.ax_wo, self.ax_go):
            if self.log_scale:
                ax.set_yscale("log")
                ax.set_ylim(1e-4, 1.1)
            else:
                ax.set_yscale("linear")
                ax.set_ylim(0, 1.02)

        notes = rt.warnings()
        if notes or self.status.get_color() != "green":
            self.status.set_text("⚠ " + notes[0] if notes else "")
            self.status.set_color("darkorange")
        self.fig.canvas.draw_idle()

    def _toggle_log(self, _):
        self.log_scale = not self.log_scale
        self.update()

    def _reset(self):
        for s in self.sliders.values():
            s.reset()
        for s in self.pc_sliders.values():
            s.reset()
        self.s_mu.reset()
        self.s_drho.reset()

    def _export(self, fmt):
        name = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.rt.name)
        path = os.path.join(self.out_dir, f"{name}_{fmt}.inc")
        with open(path, "w", encoding="utf-8") as f:
            f.write(export([self.rt], fmt, self.points))
        self._set_status(f"Сохранено: {os.path.abspath(path)}", "green")
        print(f"Saved {path}")

    def show(self):
        plt.show()


def run(rock_type: RockType | None = None, **kwargs) -> RelPermApp:
    app = RelPermApp(rock_type, **kwargs)
    app.show()
    return app


def main():
    """Entry point of the `relperm-gui` launcher (no console window on Windows)."""
    run()
