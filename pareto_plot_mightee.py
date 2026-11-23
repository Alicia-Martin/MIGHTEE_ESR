"""
Pareto plot for a MIGHTEE ESR run with the standard profiles added, in the
style of the CLASH paper's pareto_CLASH.png.

One panel, complexity on x: Delta L(D) of the best ESR function at each
complexity on the left axis (red) and -Delta log L of the best-fitting one on
the right axis (blue), both relative to the minimum over everything plotted.
The standard profiles from fit_standard_profiles.py are drawn at their
complexity in both colours (red: their DL, blue: their -logL), one marker
shape per profile, with a black-marker legend above the plot.

Standard-profile complexities (number of nodes of the core_maths tree) for
physical-scale runs; for --rho-tag rhoFalse see COMP_RHOFALSE:
  power law          3   pow(x,a0), library comp 3 (exact)
  pseudo-isothermal  7   1/(x*(x + 1/x)), library comp 7 (exact)
  NFW                8   1/(x*pow((x + 1),2)), integers as nodes (hand-built tree)
  Burkert           10   1/((x + 1)*(pow(x,2) + 1)), same
  Einasto            -   not expressible in core_maths (no exp): no complexity
                         and no full DL. Omitted unless --einasto-comp is given,
                         in which case only its -logL (blue) is drawn there.

The same figure is made for the full sample and for each --min-points subset,
with the profiles summed over exactly the same galaxies as the ESR ranking.

Usage:
    python pareto_plot_mightee.py
    python pareto_plot_mightee.py --comps 3,4,5,6,7,8
    python pareto_plot_mightee.py --out-root output_new_data/output_mightee_rhoTrue
"""
import argparse
import bisect
import csv
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from combine_all_galaxies import (DEFAULT_MIN_POINTS, galaxy_subsets, global_dir_name,
                                  resolve_fn_dir)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GALAXY_FILE = os.path.join(BASE_DIR, "galaxy_names.txt")
DATA_FILE = os.path.join(BASE_DIR, "ALL_gbar_gobs_RAR_direct_phot_with_5_arc_radius_plus_Halpha_for_ESR.txt")

# Same colours as the CLASH paper figure (matplotlib Set1)
CM = plt.get_cmap("Set1")
DL_COLOR, LOGL_COLOR = CM(0), CM(1)

# (complexity, marker) -- see module docstring
PROFILE_STYLE = {
    "NFW": (8, "P"),
    "Burkert": (10, "s"),
    "pseudo-isothermal": (7, "D"),
    "power law": (3, "X"),
    "Einasto": (None, "o"),
}
LEGEND_NAME = {"pseudo-isothermal": "pISO"}
# --no-physical-scale (rhoFalse) forms: a0/(a1 + x**2) is comp 7 and a0*pow(x,a1)
# comp 5 in the library (exact); NFW is the 9-node a0/(x*pow((a1 + x),2)), Burkert a
# hand-built 13-node tree (upper bound)
COMP_RHOFALSE = {"NFW": 9, "Burkert": 13, "pseudo-isothermal": 7, "power law": 5, "Einasto": None}


def esr_per_comp(global_root, comps, dedup=True, dup_tol=0.05):
    """{comp: dict(DL, nll, fn_DL, fn_nll, removed)}: best DL and best -logL at each complexity.

    With dedup (default), a function whose summed -logL matches (within dup_tol)
    one already kept is a re-appearance of the same fit -- e.g. 1/(x*(x + 1)) at
    comp 8 is 1/(x**2 + x) from comp 6, and 1/(2*x**2 + x) is the same curve once
    rho0/rs absorb the factor 2 -- and is dropped, the same rule as
    combine_across_comp_global.py (which keys on the rounded summed -logL).
    Functions are visited from the lowest complexity up, and by DL within a
    complexity, so each fit is kept where it first (most cheaply) appears and
    each complexity shows its best NEW function, as a Pareto front should.
    "removed" lists the duplicates that would otherwise have been that comp's best.
    """
    rows = []
    for comp in comps:
        path = os.path.join(global_root, f"comp{comp}", f"global_final_{comp}.dat")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for r in csv.reader(f, delimiter=";"):
                if r and np.isfinite(float(r[2])) and np.isfinite(float(r[3])):
                    rows.append((comp, r[1], float(r[2]), float(r[3])))
    rows.sort(key=lambda t: (t[0], t[2]))

    kept_nll = []        # sorted summed -logL of every kept function (for bisect)
    first_by_nll = {}    # rounded -logL -> (function, comp) it was first kept as
    out, removed = {}, {}
    for comp, fn, DL, nll in rows:
        if dedup:
            k = bisect.bisect_left(kept_nll, nll - dup_tol)
            if k < len(kept_nll) and kept_nll[k] <= nll + dup_tol:
                if comp not in out:  # would have been this comp's best: report it
                    orig = first_by_nll.get(round(kept_nll[k], 6))
                    removed.setdefault(comp, []).append((fn, orig))
                continue
            bisect.insort(kept_nll, nll)
            first_by_nll[round(nll, 6)] = (fn, comp)
        e = out.setdefault(comp, dict(DL=DL, nll=nll, fn_DL=fn, fn_nll=fn))
        if nll < e["nll"]:
            e["nll"], e["fn_nll"] = nll, fn
    for comp, e in out.items():
        e["removed"] = removed.get(comp, [])
    return out


def profile_sums(df, gal_subset, prof):
    """{name: (DL or nan, -logL)} summed over gal_subset; skipped if any galaxy missing/failed."""
    out = {}
    for name, (_, aifeyn, _) in prof.items():
        s = df[(df["profile"] == name) & df["galaxy"].isin(gal_subset)]
        ok = len(s) == len(gal_subset) and np.all(np.isfinite(s["negloglike"]) & np.isfinite(s["codelen"]))
        if not ok:
            continue
        nll = float(s["negloglike"].sum())
        out[name] = (nll + float(s["codelen"].sum()) + aifeyn, nll)
    return out


def make_figure(esr, profs, save_base, einasto_comp=None, use_physical_scale=True):
    comps = np.array(sorted(esr))
    esr_DL = np.array([esr[c]["DL"] for c in comps])
    esr_nll = np.array([esr[c]["nll"] for c in comps])

    placed = {}  # name -> complexity, for the profiles that get drawn
    for name in profs:
        comp = (PROFILE_STYLE[name][0] if use_physical_scale else COMP_RHOFALSE[name])
        comp = einasto_comp if name == "Einasto" else comp
        if comp is not None:
            placed[name] = comp
    DL0 = min([np.nanmin(esr_DL)] + [profs[n][0] for n in placed if np.isfinite(profs[n][0])])
    nll0 = min([np.nanmin(esr_nll)] + [profs[n][1] for n in placed])

    fig, ax1 = plt.subplots(1, 1, figsize=(5.8, 4.4))
    ax2 = ax1.twinx()
    ax1.plot(comps, esr_DL - DL0, marker="o", color=DL_COLOR, ms=6, lw=2)
    ax2.plot(comps, esr_nll - nll0, marker="o", color=LOGL_COLOR, ms=6, lw=2)

    for name, comp in placed.items():
        DL, nll = profs[name]
        marker = PROFILE_STYLE[name][1]
        if np.isfinite(DL):
            ax1.scatter([comp], [DL - DL0], marker=marker, s=110, color=DL_COLOR, zorder=4)
        ax2.scatter([comp], [nll - nll0], marker=marker, s=110, color=LOGL_COLOR, zorder=4)

    ax1.set_xlabel("Complexity")
    ax1.set_ylabel(r"$\Delta L \left( D \right)$", color=DL_COLOR)
    ax2.set_ylabel(r"$-\Delta \log \mathcal{L}$", color=LOGL_COLOR)
    ax1.tick_params(axis="y", colors=DL_COLOR)
    ax2.tick_params(axis="y", colors=LOGL_COLOR)
    ax2.spines["left"].set_color(DL_COLOR)
    ax2.spines["right"].set_color(LOGL_COLOR)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    all_x = list(comps) + list(placed.values())
    ax1.set_xticks(range(int(min(all_x)), int(max(all_x)) + 1))

    handles = [plt.Line2D([], [], color="black", lw=0, marker=PROFILE_STYLE[n][1], ms=8,
                          label=LEGEND_NAME.get(n, n)) for n in placed]
    if handles:
        ax1.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
                   frameon=False, handletextpad=0.3, columnspacing=1.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{save_base}.pdf", placed


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comps", default=None, help="Comma-separated; default: every comp with a global_final file.")
    p.add_argument("--rho-tag", default="rhoTrue", choices=["rhoTrue", "rhoFalse"],
                   help="rhoFalse: no-physical-scale runs, compared with fit_standard_profiles.py "
                        "--no-physical-scale (standard_profiles_fit_rhoFalse.csv).")
    p.add_argument("--run-name", default="mightee")
    p.add_argument("--out-root", default=None, help="Folder holding GLOBAL/ and GLOBAL_minpts*/ "
                   "(default: fitting/output/output_<run-name>_<rho-tag>).")
    p.add_argument("--profiles-csv", default=None, help="Default: posterior_check_output/"
                   "standard_profiles_fit.csv, or ..._rhoFalse.csv with --rho-tag rhoFalse.")
    p.add_argument("--profiles", default="NFW,Burkert,pseudo-isothermal,power_law,Einasto",
                   help="Comma-separated, no spaces; write power law as power_law.")
    p.add_argument("--einasto-comp", type=int, default=None, help="Complexity to draw Einasto's "
                   "-logL at (it has no complexity in core_maths); omitted if not given.")
    p.add_argument("--data-file", default=DATA_FILE, help="Catalogue, for the min-points subsets.")
    p.add_argument("--min-points", default=",".join(map(str, DEFAULT_MIN_POINTS)))
    p.add_argument("--fn-library-dir", default=None)
    p.add_argument("--no-dedup", action="store_true", help="Keep functions that re-appear "
                   "(same summed -logL) at a higher complexity.")
    p.add_argument("--dup-tol", type=float, default=0.05, help="Two functions are the same fit "
                   "if their summed -logL differ by at most this (default 0.05 nats).")
    return p.parse_args()


def main():
    args = _parse_args()
    out_root = args.out_root or os.path.join(BASE_DIR, "fitting", "output",
                                             f"output_{args.run_name}_{args.rho_tag}")
    import fit_standard_profiles as fsp
    physical = args.rho_tag == "rhoTrue"
    prof_all = fsp.profiles(resolve_fn_dir(args.fn_library_dir), physical)
    if args.profiles_csv is None:
        args.profiles_csv = os.path.join(BASE_DIR, "posterior_check_output",
                                         f"standard_profiles_fit{'' if physical else '_rhoFalse'}{'_fml' if args.run_name.endswith('_fml') else ''}.csv")
    # Underscores mean spaces ("power_law"): addqueue drops quotes, so a
    # space inside --profiles would split it into separate arguments.
    names = [n.strip().replace("_", " ") for n in args.profiles.split(",") if n.strip()]
    unknown = [n for n in names if n not in prof_all]
    if unknown:
        print(f"Unknown profile(s) {unknown}; available: {list(prof_all)}")
    prof = {n: prof_all[n] for n in names if n in prof_all}

    df = pd.read_csv(args.profiles_csv) if os.path.exists(args.profiles_csv) else None
    if df is None:
        print(f"No standard-profile fits at {args.profiles_csv}; plotting ESR only.")

    with open(GALAXY_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]
    mp_list = [int(n) for n in args.min_points.split(",") if n.strip()]
    for mp, gal_subset in galaxy_subsets(galaxies, args.data_file, mp_list):
        global_root = os.path.join(out_root, global_dir_name(mp))
        comps = ([int(c) for c in args.comps.split(",")] if args.comps else
                 sorted(int(d[4:]) for d in os.listdir(global_root)
                        if d.startswith("comp") and d[4:].isdigit()) if os.path.isdir(global_root) else [])
        esr = esr_per_comp(global_root, comps, dedup=not args.no_dedup, dup_tol=args.dup_tol)
        if not esr:
            print(f"Skipping {global_root}: no global_final_*.dat (run combine_all_galaxies.py first).")
            continue
        profs = profile_sums(df, gal_subset, prof) if df is not None else {}
        label = ("all galaxies" if mp is None else f">= {mp} points") + f" ({len(gal_subset)} galaxies)"
        path, placed = make_figure(esr, profs, os.path.join(global_root, "pareto_with_standard_profiles"),
                                   einasto_comp=args.einasto_comp, use_physical_scale=physical)

        print(f"\n{label}: comps {sorted(esr)}")
        DL0 = min(e["DL"] for e in esr.values())
        for c in sorted(esr):
            e = esr[c]
            print(f"  comp {c:2d}: best DL   {e['DL']:9.2f} (dDL {e['DL'] - DL0:7.2f})  {e['fn_DL']}")
            print(f"           best -logL {e['nll']:9.2f}                   {e['fn_nll']}")
            for fn, orig in e["removed"]:
                was = f"{orig[0]} (comp {orig[1]})" if orig else "an earlier function"
                print(f"           removed as duplicate: {fn}  == {was}")
        for n, (DL, nll) in profs.items():
            where = f"at complexity {placed[n]}" if n in placed else "not drawn (no complexity)"
            print(f"  {n:18s}: DL {DL:9.2f}   -logL {nll:9.2f}   {where}")
        print(f"  Saved: {path} (+ .png)")


if __name__ == "__main__":
    main()
