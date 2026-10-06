"""Command line interface: relperm {app,export,plot,fit,template}."""

from __future__ import annotations

import argparse
import json
import sys

from .export import FORMATS, export
from .fit import fit_corey, read_columns
from .models import RockType


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
    fig, (ax_wo, ax_go) = plt.subplots(1, 2, figsize=(12, 5))
    for rt in rock_types:
        sw = np.linspace(rt.swl, 1, 400)
        sg = np.linspace(rt.sgl, 1 - rt.swl, 400)
        ax_wo.plot(sw, rt.krw(sw), lw=2, label=f"{rt.name} krw")
        ax_wo.plot(sw, rt.krow(sw), lw=2, ls="--", label=f"{rt.name} krow")
        ax_go.plot(sg, rt.krg(sg), lw=2, label=f"{rt.name} krg")
        ax_go.plot(sg, rt.krog(sg), lw=2, ls="--", label=f"{rt.name} krog")
    for ax, title, xl in ((ax_wo, "Water-oil", "Sw"), (ax_go, "Gas-oil", "Sg")):
        ax.set(title=title, xlabel=xl, ylabel="kr", xlim=(0, 1))
        if args.log:
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1.1)
        else:
            ax.set_ylim(0, 1.02)
        ax.grid(which="both", alpha=0.4)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}", file=sys.stderr)


def cmd_fit(args):
    data = read_columns(args.csv)
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


def cmd_app(args):
    rock_types, points = load_config(args.config)
    rt = next((r for r in rock_types if r.name == args.rock), None) if args.rock else rock_types[0]
    if rt is None:
        sys.exit(f"Rock type '{args.rock}' not found")
    from .app import run

    run(rt, points=points, out_dir=args.out_dir,
        lab_wo=read_columns(args.lab_wo) if args.lab_wo else None,
        lab_go=read_columns(args.lab_go) if args.lab_go else None)


def main(argv=None):
    p = argparse.ArgumentParser(prog="relperm", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("app", help="interactive editor with sliders")
    a.add_argument("config", nargs="?")
    a.add_argument("--rock", help="rock type name to edit (default: first)")
    a.add_argument("--out-dir", default=".", help="where export buttons save files")
    a.add_argument("--lab-wo", help="CSV with lab Sw, krw, krow to overlay")
    a.add_argument("--lab-go", help="CSV with lab Sg, krg, krog to overlay")
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

    f = sub.add_parser("fit", help="fit Corey kr_max and exponent to lab points")
    f.add_argument("csv")
    f.add_argument("--phase", choices=["w", "ow", "g", "og"], required=True,
                   help="w=krw, ow=krow, g=krg, og=krog")
    f.add_argument("--swl", type=float, default=0.0)
    f.add_argument("--swcr", type=float, default=0.0)
    f.add_argument("--sorw", type=float, default=0.0)
    f.add_argument("--sorg", type=float, default=0.0)
    f.add_argument("--sgcr", type=float, default=0.0)
    f.add_argument("--krmax", type=float, help="fix the endpoint, fit only n")
    f.add_argument("--s-col", help="saturation column name")
    f.add_argument("--kr-col", help="kr column name")
    f.set_defaults(func=cmd_fit)

    t = sub.add_parser("template", help="print a config template")
    t.add_argument("-o", "--output")
    t.set_defaults(func=cmd_template)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
