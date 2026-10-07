"""Command line interface: relperm {scal,app,export,plot,fit,template}."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .export import FORMATS, export
from .fit import fit_brooks_corey, fit_corey, read_columns
from .models import RockType

LAB_SYSTEMS_NAMES = ("mercury-air", "air-brine", "oil-brine")


def load_config(path: str | None) -> tuple[list[RockType], int]:
    if path is None:
        return [RockType()], 20
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    rock_types = [RockType.from_dict(d) for d in cfg.get("rock_types", [cfg])]
    return rock_types, int(cfg.get("points", 20))


def _warn(rock_types):
    for rt in rock_types:
        for w in rt.warnings():
            print(f"warning [{rt.name}]: {w}", file=sys.stderr)


def cmd_template(args):
    cfg = {"points": 20, "rock_types": [RockType().to_dict()]}
    text = json.dumps(cfg, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)


def cmd_export(args):
    rock_types, points = load_config(args.config)
    _warn(rock_types)
    text = export(rock_types, args.format, args.points or points)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved {args.output}", file=sys.stderr)
    else:
        print(text)


def cmd_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rock_types, _ = load_config(args.config)
    fig, (ax_wo, ax_go, ax_pc) = plt.subplots(1, 3, figsize=(17, 5))
    for rt in rock_types:
        sw = np.linspace(rt.swl, 1, 400)
        sg = np.linspace(rt.sgl, 1 - rt.swl, 400)
        ax_wo.plot(sw, rt.krw(sw), lw=2, label=f"{rt.name} krw")
        ax_wo.plot(sw, rt.krow(sw), lw=2, ls="--", label=f"{rt.name} krow")
        ax_go.plot(sg, rt.krg(sg), lw=2, label=f"{rt.name} krg")
        ax_go.plot(sg, rt.krog(sg), lw=2, ls="--", label=f"{rt.name} krog")
        ax_pc.plot(sw, rt.pc_ow(sw), lw=2, label=f"{rt.name} Pcow(Sw)")
        ax_pc.plot(sg, rt.pc_og(sg), lw=2, ls="--", label=f"{rt.name} Pcog(Sg)")
    for ax, title, xl in ((ax_wo, "Water-oil", "Sw"), (ax_go, "Gas-oil", "Sg")):
        ax.set(title=title, xlabel=xl, ylabel="kr", xlim=(0, 1))
        if args.log:
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1.1)
        else:
            ax.set_ylim(0, 1.02)
        ax.grid(which="both", alpha=0.4)
        ax.legend(fontsize=8)
    units = sorted({rt.pc_units for rt in rock_types})
    ax_pc.set(title="Capillary pressure", xlabel="Sw (Pcow) / Sg (Pcog)",
              ylabel=f"Pc, {'/'.join(units)}", xlim=(0, 1))
    ax_pc.grid(which="both", alpha=0.4)
    ax_pc.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}", file=sys.stderr)


def cmd_fit(args):
    data = read_columns(args.csv)
    if args.phase in ("pcow", "pcog"):
        if args.phase == "pcow":
            res = fit_brooks_corey(data[args.s_col or "Sw"], data[args.kr_col or "Pcow"],
                                   args.swl, 1 - args.sorw)
        else:
            res = fit_brooks_corey(1 - data[args.s_col or "Sg"], data[args.kr_col or "Pcog"],
                                   args.swl + args.sorg, 1.0)
        print(f"pe     = {res.pe:.4g}\nlam    = {res.lam:.3f}\n"
              f"RMSE   = {res.rmse:.4g} ({res.used} points)")
        return
    if args.phase == "w":
        res = fit_corey(data[args.s_col or "Sw"], data[args.kr_col or "krw"],
                        args.swcr, 1 - args.sorw, args.krmax)
    elif args.phase == "ow":
        res = fit_corey(1 - data[args.s_col or "Sw"], data[args.kr_col or "krow"],
                        args.sorw, 1 - args.swl, args.krmax)
    elif args.phase == "g":
        res = fit_corey(data[args.s_col or "Sg"], data[args.kr_col or "krg"],
                        args.sgcr, 1 - args.swl - args.sorg, args.krmax)
    else:  # og: liquid saturation Sl = 1 - Sg
        res = fit_corey(1 - data[args.s_col or "Sg"], data[args.kr_col or "krog"],
                        args.swl + args.sorg, 1.0, args.krmax)
    print(f"kr_max = {res.kr_max:.4f}\nn      = {res.n:.3f}\n"
          f"RMSE   = {res.rmse:.4g} ({res.used} points)")


def cmd_scal(args):
    from .scal import PcOptions, build_rock_types, read_scal, write_report

    data = read_scal(*args.files)
    for path, sheet, kind in data.sources:
        print(f"  {os.path.basename(path)} [{sheet}]: {kind}", file=sys.stderr)
    bins = [float(b) for b in args.perm_bins.split(",")] if args.perm_bins else []
    pc = PcOptions(lab_units=args.pc_lab_units, system=args.pc_system, ift_res=args.ift_res,
                   theta_res=args.theta_res, units=args.pc_units, drho=args.drho,
                   j_form=args.j_form)
    res = build_rock_types(data, group_by=args.group_by, perm_bins=bins,
                           endpoints=args.endpoints, align_kro=not args.no_align_kro, pc=pc)
    for w in res.warnings:
        print(f"warning: {w}", file=sys.stderr)
    for s in res.samples:
        for note in s.notes:
            print(f"note [{s.sample}]: {note}", file=sys.stderr)

    print(f"\n{'rock type':<14}{'n':>3} {'Swl':>6}{'Sorw':>7}{'krwr':>7}{'krocw':>7}"
          f"{'nw':>6}{'now':>6} {'Sgcr':>6}{'Sorg':>7}{'krgr':>7}{'ng':>6}{'nog':>6}")
    for rt in res.rock_types:
        n = len(res.groups[rt.name])
        print(f"{rt.name:<14}{n:>3} {rt.swl:6.3f}{rt.sowcr:7.3f}{rt.krwr:7.3f}{rt.krocw:7.3f}"
              f"{rt.nw:6.2f}{rt.now:6.2f} {rt.sgcr:6.3f}{rt.sogcr:7.3f}{rt.krgr:7.3f}"
              f"{rt.ng:6.2f}{rt.nog:6.2f}")
    for c in res.correlations:
        print(f"  {c['endpoint']:<7}= {c['a']:.4f} {c['b']:+.4f}*lg(k)   R2={c['r2']:.2f} (n={c['n']})")

    with_pc = [rt for rt in res.rock_types if rt.pcow is not None]
    if with_pc:
        print(f"\nCapillary pressure, reservoir oil-water (sigma={pc.ift_res:g} mN/m, "
              f"theta={pc.theta_res:g} deg), units {pc.units}:")
        for rt in with_pc:
            m = rt.pcow
            entry = float(rt.pc_ow(1.0))
            h = float(rt.height_above_fwl(1.0, pc.drho))
            n = sum(s.kind == "pc" for s in res.groups[rt.name])
            if m.model == "leverett":
                desc = f"Leverett J ({m.form}), k = {m.perm:.4g} mD, phi = {m.poro:.3g}"
            else:
                desc = f"Brooks-Corey pe = {m.pe:.4g}, lambda = {m.lam:.4g}"
            pcmax = "-" if m.pcmax is None else f"{m.pcmax:.4g}"
            print(f"  {rt.name:<14}{n:>2} samples  {desc}, Pcmax = {pcmax}; "
                  f"entry {entry:.4g} -> {h:.1f} m above FWL")
            for form, f in res.pc_fits.get(rt.name, {}).items():
                expr = (f"{f['a']:.4g}*exp(-{f['b']:.4g}*Swn)" if form == "exp"
                        else f"{f['a']:.4g}*Swn^-{f['b']:.4g}")
                mark = "  <- used" if form == getattr(m, "form", None) else ""
                print(f"      J = {expr:<26} rmse(Swn) = {f['rmse_swn']:.3f}, "
                      f"{f['used']} points{mark}")

    written = write_report(res, args.output, args.points, pc)
    dep = res.dependencies
    if dep.get("fits"):
        print(f"\nDependencies ({len(dep['properties'])} samples):")
        for f in dep["fits"]:
            print(f"  {f['formula']:<40} R2 = {f['r2']:.2f}  n = {f['n']}")
    for a in dep.get("anomalies", []):
        print(f"anomaly [{a['sample']}] {a['property']} = {a['value']}: {a['reason']}",
              file=sys.stderr)
    if args.export:
        path = os.path.join(args.output, f"relperm_{args.export}.inc")
        with open(path, "w", encoding="utf-8") as f:
            f.write(export(res.rock_types, args.export, args.points))
        written.append(path)
    print(f"\nSaved to {args.output}/: " + ", ".join(os.path.basename(p) for p in written),
          file=sys.stderr)


def cmd_app(args):
    rock_types, points = load_config(args.config)
    rt = next((r for r in rock_types if r.name == args.rock), None) if args.rock else rock_types[0]
    if rt is None:
        sys.exit(f"Rock type '{args.rock}' not found")
    from .app import run

    run(rt, points=points, out_dir=args.out_dir,
        lab_wo=read_columns(args.lab_wo) if args.lab_wo else None,
        lab_go=read_columns(args.lab_go) if args.lab_go else None,
        lab_pc=read_columns(args.lab_pc) if args.lab_pc else None)


def main(argv=None):
    p = argparse.ArgumentParser(prog="relperm", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scal", help="build rock types from SCAL lab data (CSV / Excel)")
    sc.add_argument("files", nargs="+", help="CSV / XLSX files; all sheets are read")
    sc.add_argument("-o", "--output", default="scal_out", help="output directory")
    sc.add_argument("--group-by", choices=["auto", "rock_type", "horizon", "well", "perm", "none"],
                    default="auto")
    sc.add_argument("--perm-bins", help="permeability class edges in mD, e.g. 10,100")
    sc.add_argument("--endpoints", choices=["mean", "median"], default="mean")
    sc.add_argument("--no-align-kro", action="store_true",
                    help="keep krog(Sg=0) from gas-oil data instead of krow(Swl)")
    sc.add_argument("-f", "--export", choices=FORMATS, help="also write simulator tables")
    sc.add_argument("--pc-lab-units", choices=["psi", "kPa", "MPa", "bar", "atm"],
                    help="lab Pc units when the column header has none")
    sc.add_argument("--pc-system", choices=list(LAB_SYSTEMS_NAMES),
                    help="lab fluid pair when the data do not say (default air-brine)")
    sc.add_argument("--ift-res", type=float, default=30.0,
                    help="reservoir oil-water IFT, mN/m (default 30)")
    sc.add_argument("--theta-res", type=float, default=30.0,
                    help="reservoir contact angle, degrees (default 30)")
    sc.add_argument("--pc-units", choices=["bar", "psi", "kPa", "atm"], default="bar",
                    help="Pc units of the output tables")
    sc.add_argument("--j-form", choices=["auto", "power", "exp"], default="auto",
                    help="Leverett J: a*Swn^-b, a*exp(-b*Swn) or the better fit (default)")
    sc.add_argument("--drho", type=float, default=250.0,
                    help="oil-water density difference for heights above FWL, kg/m3")
    sc.add_argument("-n", "--points", type=int, default=20)
    sc.set_defaults(func=cmd_scal)

    a = sub.add_parser("app", help="interactive editor with sliders")
    a.add_argument("config", nargs="?")
    a.add_argument("--rock", help="rock type name to edit (default: first)")
    a.add_argument("--out-dir", default=".", help="where export buttons save files")
    a.add_argument("--lab-wo", help="CSV with lab Sw, krw, krow to overlay")
    a.add_argument("--lab-go", help="CSV with lab Sg, krg, krog to overlay")
    a.add_argument("--lab-pc", help="CSV with lab Sw, Pcow to overlay")
    a.set_defaults(func=cmd_app)

    e = sub.add_parser("export", help="write simulator tables")
    e.add_argument("config", nargs="?")
    e.add_argument("-f", "--format", choices=FORMATS, default="eclipse")
    e.add_argument("-n", "--points", type=int, help="points per table (default from config)")
    e.add_argument("-o", "--output")
    e.set_defaults(func=cmd_export)

    pl = sub.add_parser("plot", help="save a PNG of the curves")
    pl.add_argument("config", nargs="?")
    pl.add_argument("-o", "--output", default="relperm.png")
    pl.add_argument("--log", action="store_true", help="logarithmic kr axis")
    pl.set_defaults(func=cmd_plot)

    f = sub.add_parser("fit", help="fit Corey kr or Brooks-Corey Pc to lab points")
    f.add_argument("csv")
    f.add_argument("--phase", choices=["w", "ow", "g", "og", "pcow", "pcog"], required=True,
                   help="w=krw, ow=krow, g=krg, og=krog (Corey); pcow, pcog (Brooks-Corey)")
    f.add_argument("--swl", type=float, default=0.0)
    f.add_argument("--swcr", type=float, default=0.0)
    f.add_argument("--sorw", type=float, default=0.0)
    f.add_argument("--sorg", type=float, default=0.0)
    f.add_argument("--sgcr", type=float, default=0.0)
    f.add_argument("--krmax", type=float, help="fix the endpoint, fit only n")
    f.add_argument("--s-col", help="saturation column name")
    f.add_argument("--kr-col", help="kr (or Pc) column name")
    f.set_defaults(func=cmd_fit)

    t = sub.add_parser("template", help="print a config template")
    t.add_argument("-o", "--output")
    t.set_defaults(func=cmd_template)

    argv = sys.argv[1:] if argv is None else list(argv)
    args = p.parse_args(argv or ["app"])  # no arguments: open the editor
    args.func(args)


if __name__ == "__main__":
    main()
