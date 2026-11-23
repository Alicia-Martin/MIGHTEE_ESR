"""
Fitted f_ml vs Inc, one point per galaxy, for one function of an f_ml run
(default: the population winner 1/(x**2 + x) at comp 6). Reads each galaxy's
codelen_matches_comp<N>.dat (Inc, D, f written by match.py) -- no refitting.

    python plot_inc_vs_fml.py                                   # 1/(x**2 + x), comp 6
    python plot_inc_vs_fml.py --comp 7 --fcn "1/(x*(x + 1/x))"
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "ALL_gbar_gobs_RAR_direct_phot_with_5_arc_radius_plus_Halpha_for_ESR.txt")
ML_FILE = os.path.join(BASE_DIR, "RAR_MIGHTEE_total_ml_ratios_direct.txt")
MAX_PARAM_TOTAL = 6  # 4 shape + rho0, rs (rhoTrue)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comp", type=int, default=6)
    p.add_argument("--fcn", default="1/(x**2 + x)")
    p.add_argument("--run-dir", default=os.path.join(BASE_DIR, "fitting", "output", "output_mightee_fml_rhoTrue"))
    p.add_argument("--fn-library-dir", default=os.environ.get("ESR_FUNCTION_LIBRARY_DIR"))
    p.add_argument("--out", default=None, help="Default: <run-dir>/GLOBAL/inc_vs_fml_comp<N>_<function>.png")
    a = p.parse_args()
    if a.out is None:
        safe = "".join(c if c.isalnum() else "_" for c in a.fcn).strip("_")
        a.out = os.path.join(a.run_dir, "GLOBAL", f"dinc_vs_fml_comp{a.comp}_{safe}.png")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    with open(os.path.join(a.fn_library_dir, f"compl_{a.comp}", f"all_equations_{a.comp}.txt")) as f:
        fcns = [l.strip() for l in f]
    rows = [i for i, s in enumerate(fcns) if s == a.fcn]
    if not rows:
        raise SystemExit(f"{a.fcn!r} not in comp {a.comp}")
    galaxies = [l.strip() for l in open(os.path.join(BASE_DIR, "galaxy_names.txt")) if l.strip()]

    m = MAX_PARAM_TOTAL
    data = {}
    for g in galaxies:
        fp = os.path.join(a.run_dir, g, f"comp{a.comp}", f"codelen_matches_comp{a.comp}.dat")
        if os.path.exists(fp):
            data[g] = np.atleast_2d(np.genfromtxt(fp))
    if data[next(iter(data))].shape[1] < 3 + 2 * m + 3 + 3:
        raise SystemExit("no f column -- is this an _fml run?")
    # same string can appear as several sign variants: keep the one with the lowest population DL
    pop = [sum(d[r, 0] + d[r, 1] for d in data.values() if np.isfinite(d[r, 0] + d[r, 1])) for r in rows]
    row = rows[int(np.argmin(pop))]
    print(f"{a.fcn} comp {a.comp}: row {row}, {len(data)}/{len(galaxies)} galaxies")

    cat = pd.read_csv(DATA_FILE, sep=None, engine="python").groupby("Galaxy").first()
    inc_col = next(c for c in cat.columns if c.lower() in ("inc", "inclination"))
    ml = pd.read_csv(ML_FILE, sep=None, engine="python").set_index("galaxy")
    sig_f = 0.5 * (ml["ML_Ks_84"] - ml["ML_Ks_16"]) / ml["ML_Ks_50"]

    # sigma as in the codelen (match.py): 1/sqrt(H_ii) from the Hessian diagonal that
    # test_all_Fisher saved in derivs_comp<N>.dat -- the width along that parameter with
    # the others fixed. The row is the function's unique index (codelen_matches col 2);
    # the galaxy block starts after the m(m+1)/2 shape/rho0/rs entries: [0] = H_Inc,Inc,
    # [1] = H_D,D, [3+2m] = H_f,f.
    n_tri = m * (m + 1) // 2
    out = []
    for g, d in data.items():
        r = d[row]
        if not np.isfinite(r[0]) or r[0] == 0:
            continue
        dv = np.atleast_2d(np.genfromtxt(os.path.join(a.run_dir, g, f"comp{a.comp}", f"derivs_comp{a.comp}.dat")))
        gblock = dv[int(r[2]), n_tri:]
        h_inc, h_f = gblock[0], gblock[3 + 2 * m]
        sig = lambda h: 1.0 / np.sqrt(h) if np.isfinite(h) and h > 0 else np.nan
        out.append(dict(galaxy=g, inc_fit=r[3 + 2 * m], inc_sigma=sig(h_inc), f_fit=r[5 + 2 * m], f_sigma=sig(h_f),
                        inc_cat=float(cat.loc[g, inc_col]), sigma_f_prior=float(sig_f.get(g, np.nan))))
    df = pd.DataFrame(out)
    print(f"finite sigma (H_ii > 0): Inc {np.isfinite(df.inc_sigma).sum()}/{len(df)}, "
          f"f {np.isfinite(df.f_sigma).sum()}/{len(df)}")

    # Does "f < 1" go with "fitted Inc below catalogue Inc"? x = Inc_fit - Inc_cat, y = f - 1;
    # quadrant counts + Fisher exact test on the 2x2 table of signs.
    from scipy.stats import fisher_exact
    df["dinc"] = df.inc_fit - df.inc_cat
    df["df"] = df.f_fit - 1.0
    inc_low, f_low = df.dinc < 0, df["df"] < 0
    table = np.array([[np.sum(inc_low & f_low), np.sum(inc_low & ~f_low)],
                      [np.sum(~inc_low & f_low), np.sum(~inc_low & ~f_low)]])
    odds, pval = fisher_exact(table)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ok = np.isfinite(df.inc_sigma) & np.isfinite(df.f_sigma)
    # colour = number of radial data points in the galaxy
    n_pts = pd.read_csv(DATA_FILE, sep=None, engine="python").groupby("Galaxy").size()
    df["n_points"] = df.galaxy.map(n_pts).astype(int)
    groups = [("3 points", df.n_points <= 3, "#1f77b4"),
              ("4 points", df.n_points == 4, "#ff7f0e"),
              ("$\\geq$ 5 points", df.n_points >= 5, "#2ca02c")]
    for lab, sel, col in groups:
        m_ = ok & sel
        if m_.any():
            ax.errorbar(df.dinc[m_], df["df"][m_], xerr=df.inc_sigma[m_], yerr=df.f_sigma[m_],
                        fmt="o", ms=4, color=col, ecolor=col, elinewidth=0.8, alpha=0.75, capsize=0,
                        label=f"{lab} ({int(m_.sum())})")
    if (~ok).any():  # H_ii <= 0: the codelen uses the 1D integral fallback for these
        ax.plot(df.dinc[~ok], df["df"][~ok], "o", ms=5, mfc="none", color="#d62728",
                label=f"$H_{{ii}} \\leq 0$: no error bar ({(~ok).sum()})")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    for (xs, ys, n_q) in [(0.03, 0.03, table[0, 0]), (0.03, 0.97, table[0, 1]),
                          (0.97, 0.03, table[1, 0]), (0.97, 0.97, table[1, 1])]:
        ax.text(xs, ys, f"N = {n_q}", transform=ax.transAxes, fontsize=11, weight="bold",
                ha="left" if xs < 0.5 else "right", va="bottom" if ys < 0.5 else "top")
    ax.set_xlabel("fitted Inc $-$ catalogue Inc [deg]")
    ax.set_ylabel("$f_{ML} - 1$")
    ax.set_title(f"{a.fcn}  (comp {a.comp}),  N = {len(df)}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")

    print("\n2x2 table             f < 1    f >= 1")
    print(f"  Inc_fit <  Inc_cat   {table[0, 0]:5d}    {table[0, 1]:5d}")
    print(f"  Inc_fit >= Inc_cat   {table[1, 0]:5d}    {table[1, 1]:5d}")
    print(f"  Fisher exact: odds ratio {odds:.2f}, p = {pval:.3g}  "
          f"(odds > 1: f < 1 more common among galaxies whose Inc went down)")

    n = 10
    low_inc = set(df.nsmallest(n, "inc_fit").galaxy)
    low_f = set(df.nsmallest(n, "f_fit").galaxy)
    pd.set_option("display.width", 200)
    df = df.round(4)
    print(f"\n{n} lowest fitted Inc:\n{df.nsmallest(n, 'inc_fit').round(3).to_string(index=False)}")
    print(f"\n{n} lowest f:\n{df.nsmallest(n, 'f_fit').round(3).to_string(index=False)}")
    print(f"\nin both lists: {len(low_inc & low_f)}/{n}: {sorted(low_inc & low_f)}")
    csv = os.path.splitext(a.out)[0] + ".csv"
    df.to_csv(csv, index=False)
    print(f"\nSaved: {a.out}\n       {csv}")


if __name__ == "__main__":
    main()
