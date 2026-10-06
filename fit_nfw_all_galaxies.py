"""
Independent sanity check for today's Fisher/codelen fixes: fit the
well-known NFW profile (not an ESR-searched function) to every galaxy in
the MIGHTEE catalog and plot the result, using the exact same production
entry point validate_fit.py already validates against brute-force
baselines (test_all.optimise_fun_direct_nm, including its stage-2
galaxy_params_polish call).

NFW in ESR's physical-scale convention (rho(r) = rho0 * f(r/rs)):
    rho_NFW(r) = rho_s / [(r/rs) * (1 + r/rs)^2]
  =>  f(u) = 1/(u*(1+u)**2), u = r/rs -- no free shape parameter (nshape=0),
      only rho0 (=rho_s) and rs are fit, plus Inc/D as galaxy nuisance
      parameters. This exercises the nshape=0 code path this session's
      fixes specifically had to handle correctly.

Usage:
    python fit_nfw_all_galaxies.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from esr.fitting.dm_likelihood import MIGHTEELikelihood
from esr.fitting.test_all import optimise_fun_direct_nm, lambdify_equation

THIS_DIR = Path(__file__).resolve().parent
# ALL_gbar_gobs_RAR_direct_phot.txt is the fuller catalog (129 galaxies vs
# rar_with_ml_direct_phot.csv's 81 -- confirmed 2026-08-14 not a strict
# subset: 79 galaxies in common, 2 only in the 81-catalog, 50 only in the
# 129-catalog). Has every column dm_likelihood.py's _load_galaxy_data
# actually reads (z, inc, inc_err, Radius_arcsec, v_circ, v_rot_err_lo/hi,
# v_bar_total, v_bar_total_err) -- the columns it's missing relative to the
# 81-catalog (ml_K, Survey, sample, log_g*) are all unused by the
# likelihood. Same file validate_fit.py already defaults to.
DATA_FILE = THIS_DIR / "ALL_gbar_gobs_RAR_direct_phot.txt"
OUT_DIR = THIS_DIR / "nfw_fits_all129"
NFW_FCN = "1/(x*(1+x)**2)"
NSHAPE = 0  # NFW has no free shape parameter


def fit_one_galaxy(name: str) -> dict:
    likelihood = MIGHTEELikelihood(
        data_file=str(DATA_FILE), name=name, run_name="nfw_sanity_check",
        data_dir=str(THIS_DIR), use_physical_scale=True,
    )

    chi2, full_params, niter, count_lowest, success, inc_fit, d_fit, stage2_chi2 = optimise_fun_direct_nm(
        NFW_FCN, likelihood, tmax=60, pmin=-8, pmax=8,
        try_integration=False, log_opt=True, method="DIRECT",
    )

    rho0, rs = float(full_params[4]), float(full_params[5])

    return {
        "likelihood": likelihood,
        "name": name,
        "chi2": float(chi2),
        "rho0": rho0,
        "rs": rs,
        "inc_fit": float(inc_fit),
        "d_fit": float(d_fit),
        "stage2_chi2": float(stage2_chi2),
        "inc_true": float(likelihood.inc_true),
        "distance_true": float(likelihood.distance_true),
        "n_points": int(len(likelihood.xvar)),
    }


def plot_galaxy(pdf: PdfPages, result: dict, eq_numpy, integrated: bool):
    lik = result["likelihood"]
    r_data = np.asarray(lik.xvar)
    v_data = np.asarray(lik.yvar)
    v_bar = np.asarray(lik.v_bar)
    yerr_lo = np.asarray(lik.yerr_lo)
    yerr_hi = np.asarray(lik.yerr_hi)

    # get_pred's baryon term is self.v_bar itself (a fixed per-galaxy curve
    # index-aligned with self.xvar, from photometry -- not a smooth analytic
    # function), so get_pred can only be evaluated at the data radii, not an
    # arbitrary dense grid (confirmed live: broadcasting error, v_bar length
    # 16 vs a 200-point dense grid). Connect the fitted values at the actual
    # data points instead of fabricating an interpolated curve -- honest
    # about what the model actually predicts.
    a_fit = np.array([result["rho0"], result["rs"]])
    # get_pred already returns v_circ (sqrt applied internally), not v_circ^2.
    v_model_at_data = np.asarray(
        lik.get_pred(r_data, a_fit, eq_numpy, integrated=integrated,
                      D=result["d_fit"], Inc=result["inc_fit"])
    )
    residuals = v_data - v_model_at_data

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.5, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
    )

    ax1.errorbar(r_data, v_data, yerr=[yerr_lo, yerr_hi], fmt="o", ms=5,
                 color="black", ecolor="gray", capsize=2, label=r"$v_{\rm circ}$ (data)")
    ax1.plot(r_data, v_model_at_data, "-o", ms=4, color="crimson", lw=1.8, label="NFW fit")
    ax1.plot(r_data, v_bar, "--", color="steelblue", lw=1.3, alpha=0.8, label=r"$v_{\rm bar}$ (fixed)")
    ax1.set_ylabel(r"$v$ [km/s]")
    ax1.set_title(
        f"{result['name']}  (N={result['n_points']}, "
        f"$\\chi^2_{{\\rm stage2}}$={result['stage2_chi2']:.2f})"
    )
    ax1.legend(loc="lower right", fontsize=8)

    inc_line = (f"Inc: fit={result['inc_fit']:.1f}°, catalog={result['inc_true']:.1f}°   "
                f"D: fit={result['d_fit']:.0f}, catalog={result['distance_true']:.0f} kpc\n"
                f"$\\rho_0$={result['rho0']:.3e}, $r_s$={result['rs']:.3e} kpc")
    ax1.annotate(inc_line, xy=(0.02, 0.98), xycoords="axes fraction",
                 ha="left", va="top", fontsize=7.5,
                 bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.errorbar(r_data, residuals, yerr=[yerr_lo, yerr_hi], fmt="o", ms=4,
                 color="black", ecolor="gray", capsize=2)
    ax2.set_ylabel("data - fit")
    ax2.set_xlabel("R [kpc]")

    pdf.savefig(fig)
    plt.close(fig)


def plot_summary_page(pdf: PdfPages, results: list[dict]):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    chi2s = np.array([r["chi2"] for r in results])
    finite = np.isfinite(chi2s)
    lines = [
        "NFW sanity-check fit summary -- all MIGHTEE galaxies",
        "",
        f"Galaxies fit: {len(results)}",
        f"Finite chi2: {int(finite.sum())} / {len(results)}",
    ]
    if finite.any():
        lines += [
            f"chi2 median: {np.median(chi2s[finite]):.2f}",
            f"chi2 min:    {np.min(chi2s[finite]):.2f}",
            f"chi2 max:    {np.max(chi2s[finite]):.2f}",
        ]
    lines.append("")
    lines.append(f"{'Galaxy':<20}{'chi2':>10}{'stage2_chi2':>14}{'Inc_fit':>10}{'Inc_cat':>10}{'D_fit':>12}{'D_cat':>12}")
    lines.append("-" * 88)
    for r in sorted(results, key=lambda x: (not np.isfinite(x["chi2"]), x["chi2"])):
        lines.append(
            f"{r['name']:<20}{r['chi2']:>10.2f}{r['stage2_chi2']:>14.2f}"
            f"{r['inc_fit']:>10.1f}{r['inc_true']:>10.1f}{r['d_fit']:>12.0f}{r['distance_true']:>12.0f}"
        )

    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=7, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    # sep=None + engine="python" auto-sniffs the delimiter, matching
    # dm_likelihood.py's own _load_galaxy_data -- needed since
    # ALL_gbar_gobs_RAR_direct_phot.txt is whitespace-delimited, unlike
    # rar_with_ml_direct_phot.csv.
    df = pd.read_csv(DATA_FILE, sep=None, engine="python")
    galaxy_names = sorted(df["Galaxy"].unique())
    print(f"Fitting NFW to {len(galaxy_names)} galaxies...")

    results = []
    errors = []
    for i, name in enumerate(galaxy_names, 1):
        try:
            r = fit_one_galaxy(name)
            results.append(r)
            print(f"[{i}/{len(galaxy_names)}] {name}: chi2={r['chi2']:.2f} "
                  f"stage2_chi2={r['stage2_chi2']:.2f} Inc={r['inc_fit']:.1f} D={r['d_fit']:.0f}",
                  flush=True)
        except Exception as e:
            errors.append((name, f"{type(e).__name__}: {e}"))
            print(f"[{i}/{len(galaxy_names)}] {name}: ERROR {type(e).__name__}: {e}", flush=True)

    if errors:
        print(f"\n{len(errors)} galaxies errored:")
        for name, err in errors:
            print(f"  {name}: {err}")

    pdf_path = OUT_DIR / "nfw_fits_all_galaxies.pdf"
    with PdfPages(pdf_path) as pdf:
        plot_summary_page(pdf, results)
        for r in results:
            fcn_i, eq, integrated = r["likelihood"].run_sympify(NFW_FCN, tmax=5, try_integration=False)
            eq_numpy = lambdify_equation(eq, NSHAPE)
            plot_galaxy(pdf, r, eq_numpy, integrated)
    print(f"\nWrote {pdf_path}")

    csv_path = OUT_DIR / "nfw_fit_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Galaxy", "N_points", "chi2", "stage2_chi2", "rho0", "rs",
                          "Inc_fit", "Inc_catalog", "D_fit", "D_catalog"])
        for r in results:
            writer.writerow([r["name"], r["n_points"], r["chi2"], r["stage2_chi2"],
                              r["rho0"], r["rs"], r["inc_fit"], r["inc_true"],
                              r["d_fit"], r["distance_true"]])
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
