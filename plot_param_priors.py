"""
For a given MIGHTEE complexity: find the top-N population-ranked functions
(same selection as plot_top3_populations.py, via _top_functions.py), then
visualize, per function rank, how far each galaxy's fitted parameters sit
from the priors used during fitting.

The two parameter families used different priors during fitting, so they
get different plots rather than one metric pretending to fit both:

  - Inc, D: per-galaxy truncated-Gaussian priors (mean/sigma from the
    catalog's inclination and redshift-derived distance -- see
    MIGHTEELikelihood.inc_true/e_inc/distance_true/e_d). Shown as a pull
    histogram z = (fitted - prior_mean) / prior_sigma, x-axis zoomed
    symmetrically to the actual pull range so a tightly-constrained
    parameter's narrow spike is still readable, plus a
    fitted-vs-prior-mean scatter with prior-sigma
    errorbars and a y=x line, so individual outlier galaxies are visible,
    not just the aggregate spread. D's scatter uses log-log axes since D
    spans ~1.5 decades across the sample.

  - rho0, rs, and each active shape parameter: a flat box prior in
    log10-internal optimizer space, [-6, 10] (test_all.py's DIRECT search
    bounds -- the reported value itself comes from an unbounded polish
    stage afterward, so landing outside the box is possible and expected,
    not a bug). Shown as a histogram of sign(param)*log10(|param|) with the
    box edges marked, so how many fits left the global-search box is
    visible at a glance, plus a text annotation naming the galaxies that
    land furthest outside it.

Usage:
    python plot_param_priors.py --comp 6 --n-top 3 --rho-tag rhoTrue
    python plot_param_priors.py --comps 1,2,3,4,5,6 --n-top 3 --rho-tag rhoFalse
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from esr.fitting.dm_likelihood import MIGHTEELikelihood
from esr.generation import simplifier

from _top_functions import build_picks

FN_DIR = os.environ.get("ESR_FUNCTION_LIBRARY_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "function_library", "core_maths"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GALAXY_FILE = os.path.join(BASE_DIR, "galaxy_names.txt")
DATA_FILE = os.path.join(BASE_DIR, "ALL_gbar_gobs_RAR_direct_phot.txt")
MAX_FUN_PARAM = 4
# n_extra/max_param_total depend on use_physical_scale (rho0/rs only exist
# in physical-scale mode -- match.py's own column layout, n_extra=2 if
# use_physical_scale else 0) so they're computed per-run in main(), not
# fixed module constants.
DIRECT_LOG_BOUNDS = (-6.0, 10.0)  # test_all.py main(pmin=-6, pmax=10, ...)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    comp_group = p.add_mutually_exclusive_group(required=True)
    comp_group.add_argument("--comp", type=int, help="Single complexity.")
    comp_group.add_argument("--comps", help="Comma-separated complexities to rank AGAINST EACH OTHER "
                             "by population DL (e.g. 1,2,3,4,5,6) -- picks the true cross-complexity "
                             "winner(s), not just the best within one comp.")
    p.add_argument("--n-top", type=int, default=3)
    p.add_argument("--rho-tag", default="rhoTrue", choices=["rhoTrue", "rhoFalse"])
    p.add_argument("--run-name", default="mightee")
    p.add_argument("--out-root", default=None, help="Base dir holding <galaxy>/comp<N>/codelen_matches_comp<N>.dat "
                    "(default: <this dir>/output_glamdring/output_<run-name>_<rho-tag>)")
    p.add_argument("--pdf-out", default=None, help="Default: <this dir>/mightee_prior_separation_comp<comp>.pdf, "
                    "or <out-root>/GLOBAL/mightee_prior_separation_comp<lo>-<hi>.pdf for --comps")
    return p.parse_args()


def _pull_panel(ax_hist, ax_scatter, label, fit_vals, prior_means, prior_sigmas, log_scale=False):
    fit_vals = np.asarray(fit_vals, dtype=float)
    prior_means = np.asarray(prior_means, dtype=float)
    prior_sigmas = np.asarray(prior_sigmas, dtype=float)
    ok = np.isfinite(fit_vals) & np.isfinite(prior_means) & np.isfinite(prior_sigmas) & (prior_sigmas > 0)
    if not np.any(ok):
        ax_hist.set_title(f"{label}: no valid data")
        ax_scatter.set_title(f"{label}: no valid data")
        return

    z = (fit_vals[ok] - prior_means[ok]) / prior_sigmas[ok]
    # Zoom symmetrically to the actual pull range (not a fixed +/-5) so a
    # tightly-constrained parameter (e.g. D, which barely moves from its
    # prior during the polish stage) doesn't render as an unreadable spike
    # squeezed into a window sized for a loosely-constrained one (e.g. Inc).
    half_width = max(0.5, 1.2 * np.nanpercentile(np.abs(z), 99))
    xlim = (-half_width, half_width)
    ax_hist.hist(z, bins=30, density=True, color="#1f77b4", alpha=0.7, label=f"N={ok.sum()}")
    ax_hist.set_xlim(xlim)
    ax_hist.set_xlabel(f"({label}$_{{fit}}$ $-$ {label}$_{{prior}}$) / $\\sigma_{{prior}}$")
    ax_hist.set_ylabel("density")
    ax_hist.set_title(f"{label}: pull vs. prior")
    ax_hist.legend(fontsize=7)

    ax_scatter.errorbar(prior_means[ok], fit_vals[ok], yerr=prior_sigmas[ok], fmt="o",
                         ms=3, color="#1f77b4", ecolor="#1f77b4", alpha=0.5, capsize=0)
    lo = min(prior_means[ok].min(), fit_vals[ok].min())
    hi = max(prior_means[ok].max(), fit_vals[ok].max())
    ax_scatter.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="y = x")
    if log_scale:
        # D spans ~1.5 decades across the sample -- linear axes compress
        # the low end into the corner and make the spread hard to read.
        ax_scatter.set_xscale("log")
        ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel(f"{label} prior mean")
    ax_scatter.set_ylabel(f"{label} fitted")
    ax_scatter.set_title(f"{label}: fitted vs. prior mean")
    ax_scatter.legend(fontsize=7)


def _box_panel(ax, label, galaxy_val_pairs, n_annotate=8):
    galaxies_arr = np.asarray([g for g, _ in galaxy_val_pairs])
    vals = np.asarray([v for _, v in galaxy_val_pairs], dtype=float)
    ok = np.isfinite(vals) & (vals != 0)
    if not np.any(ok):
        ax.set_title(f"{label}: no valid data")
        return
    galaxies_ok = galaxies_arr[ok]
    log_vals = np.sign(vals[ok]) * np.log10(np.abs(vals[ok]))
    ax.hist(log_vals, bins=30, color="#2ca02c", alpha=0.7, label=f"N={ok.sum()}")
    for edge in DIRECT_LOG_BOUNDS:
        ax.axvline(edge, color="black", lw=1.2, ls="--")
    ax.set_xlabel(f"sign({label}) $\\times$ log10(|{label}|)  (internal DIRECT-search units)")
    ax.set_ylabel("count")
    ax.set_title(f"{label}: fitted value vs. search box {DIRECT_LOG_BOUNDS}")
    ax.legend(fontsize=7)

    lo_edge, hi_edge = min(DIRECT_LOG_BOUNDS), max(DIRECT_LOG_BOUNDS)
    outside_mask = (log_vals < lo_edge) | (log_vals > hi_edge)
    if np.any(outside_mask):
        dist = np.where(log_vals[outside_mask] < lo_edge,
                         lo_edge - log_vals[outside_mask],
                         log_vals[outside_mask] - hi_edge)
        order = np.argsort(dist)[::-1]
        names = galaxies_ok[outside_mask][order]
        values = log_vals[outside_mask][order]
        lines = [f"{n} ({v:.1f})" for n, v in zip(names[:n_annotate], values[:n_annotate])]
        if len(names) > n_annotate:
            lines.append(f"+{len(names) - n_annotate} more")
        ax.text(0.02, 0.98, "outside box:\n" + "\n".join(lines), transform=ax.transAxes,
                fontsize=6, va="top", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))


def main():
    args = _parse_args()
    use_physical_scale = args.rho_tag == "rhoTrue"
    out_root = args.out_root or os.path.join(
        BASE_DIR, "output_glamdring", f"output_{args.run_name}_{args.rho_tag}")
    n_extra = 2 if use_physical_scale else 0
    max_param_total = MAX_FUN_PARAM + n_extra

    with open(GALAXY_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]

    picks, comp_label = build_picks(args.comp, args.comps, args.n_top, out_root, FN_DIR, galaxies,
                                     run_name=args.run_name)
    n_top = len(picks)

    if args.pdf_out:
        pdf_out = args.pdf_out
    elif args.comps is not None:
        pdf_out = os.path.join(out_root, "GLOBAL", f"mightee_prior_separation_{comp_label}.pdf")
        os.makedirs(os.path.dirname(pdf_out), exist_ok=True)
    else:
        pdf_out = os.path.join(BASE_DIR, f"mightee_prior_separation_{comp_label}.pdf")

    top_fcn_strings = [p["fcn"] for p in picks]
    nshape_list = [simplifier.count_params([fs], MAX_FUN_PARAM)[0] for fs in top_fcn_strings]

    with PdfPages(pdf_out) as pdf:
        for k in range(n_top):
            per_galaxy_data = picks[k]["per_galaxy_data"]
            j = picks[k]["row"]
            nshape = nshape_list[k]

            inc_fit_vals, d_fit_vals = [], []
            inc_prior_means, inc_prior_sigmas = [], []
            d_prior_means, d_prior_sigmas = [], []
            rho0_vals, rs_vals = [], []  # (galaxy, value) pairs, for outlier annotation
            shape_vals = [[] for _ in range(nshape)]

            for galaxy in galaxies:
                if galaxy not in per_galaxy_data:
                    continue
                row = per_galaxy_data[galaxy][j]
                negloglike_k = row[0]
                if not np.isfinite(negloglike_k) or negloglike_k == 0:
                    continue
                params = row[3:3 + max_param_total]
                inc_fit, d_fit = row[3 + 2 * max_param_total], row[3 + 2 * max_param_total + 1]

                lk = MIGHTEELikelihood(data_file=DATA_FILE, name=galaxy, run_name="prior_plot_only",
                                        use_physical_scale=use_physical_scale)

                inc_fit_vals.append(inc_fit)
                d_fit_vals.append(d_fit)
                inc_prior_means.append(lk.inc_true)
                inc_prior_sigmas.append(lk.e_inc)
                d_prior_means.append(lk.distance_true)
                d_prior_sigmas.append(lk.e_d)
                if n_extra == 2:
                    rho0_vals.append((galaxy, params[MAX_FUN_PARAM]))
                    rs_vals.append((galaxy, params[MAX_FUN_PARAM + 1]))
                for pidx in range(nshape):
                    shape_vals[pidx].append((galaxy, params[pidx]))

            # rho0/rs only exist in physical-scale mode (n_extra=2) -- no
            # panel for them otherwise, rather than plotting an empty box.
            box_labels = (["rho0", "rs"] if n_extra == 2 else []) + \
                [f"shape_param_{pidx}" for pidx in range(nshape)]
            box_series = ([rho0_vals, rs_vals] if n_extra == 2 else []) + shape_vals
            n_box_panels = len(box_labels)
            n_cols = 2
            n_pull_rows = 2  # Inc, D -- each row is (hist, scatter)
            n_box_rows = -(-n_box_panels // n_cols)  # ceil
            n_rows = n_pull_rows + n_box_rows

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4 * n_rows))
            axes = np.atleast_2d(axes)

            _pull_panel(axes[0, 0], axes[0, 1], "Inc", inc_fit_vals, inc_prior_means, inc_prior_sigmas)
            _pull_panel(axes[1, 0], axes[1, 1], "D", d_fit_vals, d_prior_means, d_prior_sigmas, log_scale=True)

            for bi, (blabel, bvals) in enumerate(zip(box_labels, box_series)):
                r, c = divmod(bi, n_cols)
                _box_panel(axes[n_pull_rows + r, c], blabel, bvals)
            for bi in range(n_box_panels, n_box_rows * n_cols):
                r, c = divmod(bi, n_cols)
                axes[n_pull_rows + r, c].axis("off")

            fig.suptitle(f"comp={picks[k]['comp']}  rank #{k+1} (DL={picks[k]['DL']:.1f}): {top_fcn_strings[k]}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nSaved: {pdf_out}")


if __name__ == "__main__":
    main()
