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
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

import jax.numpy as jnp

from esr.fitting.dm_likelihood import MIGHTEELikelihood
from esr.fitting.test_all import lambdify_equation
from esr.generation import simplifier

from _top_functions import build_picks

# ESR function library: $ESR_FUNCTION_LIBRARY_DIR if set, else
# <repo>/function_library/core_maths.
FN_DIR = os.environ.get("ESR_FUNCTION_LIBRARY_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "function_library", "core_maths")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GALAXY_FILE = os.path.join(BASE_DIR, "galaxy_names.txt")
DATA_FILE = os.path.join(BASE_DIR, "ALL_gbar_gobs_RAR_direct_phot_with_5_arc_radius_plus_Halpha_for_ESR.txt")
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


def _pull_panel(ax_hist, ax_scatter, label, fit_vals, prior_means, prior_sigmas, log_scale=False,
                post_sigmas=None):
    fit_vals = np.asarray(fit_vals, dtype=float)
    post_sigmas = (np.full(fit_vals.shape, np.nan) if post_sigmas is None
                   else np.asarray(post_sigmas, dtype=float))
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
    ax_hist.legend(fontsize=7, loc="upper right")

    # x error bars = prior width (on the prior-mean axis); y error bars = posterior
    # width of the fitted value, sqrt(diag(H^-1)) of the full Hessian (all parameters,
    # priors included) at the plotted point. Galaxies whose Hessian is not positive
    # definite get no y error bar (counted in the legend).
    ps = post_sigmas[ok]
    has_post = np.isfinite(ps) & (ps > 0)
    yerr = np.where(has_post, ps, 0.0)
    if log_scale:  # keep the lower bar above zero on log axes
        yerr = np.vstack([np.minimum(yerr, 0.95 * fit_vals[ok]), yerr])
    ax_scatter.errorbar(prior_means[ok], fit_vals[ok], xerr=prior_sigmas[ok], yerr=yerr, fmt="o",
                         ms=3, color="#1f77b4", ecolor="#1f77b4", alpha=0.5, capsize=0,
                         label=f"x bar: prior $\\sigma$; y bar: posterior $\\sigma$ "
                               f"({has_post.sum()}/{ok.sum()} with finite posterior $\\sigma$)")
    if np.any(has_post):
        ratio = np.median(ps[has_post] / prior_sigmas[ok][has_post])
        ax_scatter.text(0.98, 0.02, f"median posterior/prior $\\sigma$ = {ratio:.2f}",
                        transform=ax_scatter.transAxes, ha="right", va="bottom", fontsize=7)
    lo = min(prior_means[ok].min(), fit_vals[ok].min())
    hi = max(prior_means[ok].max(), fit_vals[ok].max())
    ax_scatter.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="y = x")
    if log_scale:
        # D spans ~1.5 decades across the sample -- linear axes compress
        # the low end into the corner and make the spread hard to read.
        ax_scatter.set_xscale("log")
        ax_scatter.set_yscale("log")
        for axis in (ax_scatter.xaxis, ax_scatter.yaxis):
            axis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            axis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
            axis.set_minor_formatter(plt.NullFormatter())
    ax_scatter.set_xlabel(f"{label} prior mean")
    ax_scatter.set_ylabel(f"{label} fitted")
    ax_scatter.set_title(f"{label}: fitted vs. prior mean")
    ax_scatter.legend(fontsize=7, loc="upper left")


def _box_panel(ax, label, galaxy_val_pairs):
    """Histogram of log10(param) against the DIRECT search box. Returns the
    galaxies outside the box as (galaxy, log value), furthest first, for the
    page footer."""
    galaxies_arr = np.asarray([g for g, _ in galaxy_val_pairs])
    vals = np.asarray([v for _, v in galaxy_val_pairs], dtype=float)
    ok = np.isfinite(vals) & (vals != 0)
    if not np.any(ok):
        ax.set_title(f"{label}: no valid data")
        return []
    galaxies_ok = galaxies_arr[ok]
    # rho0/rs are always > 0 (get_pred rejects anything else), so plain
    # log10; shape params can be negative, so those keep sign*log10|x|.
    if np.all(vals[ok] > 0):
        x = np.log10(vals[ok])
        xlabel = f"log10({label})"
    else:
        x = np.sign(vals[ok]) * np.log10(np.abs(vals[ok]))
        xlabel = f"sign({label}) $\\times$ log10(|{label}|)"

    # Axis = search box + central 90% of fits, so a handful of runaway fits
    # (log10 ~ 50+) can't squash the real distribution into two bins; those
    # are stacked in red in the edge bins instead.
    lo_edge, hi_edge = DIRECT_LOG_BOUNDS
    q_lo, q_hi = np.percentile(x, [5, 95])
    xmin, xmax = min(lo_edge, q_lo) - 1.0, max(hi_edge, q_hi) + 1.0
    bins = np.linspace(xmin, xmax, 41)
    width = bins[1] - bins[0]
    inside = (x >= xmin) & (x <= xmax)
    ax.axvspan(xmin, lo_edge, color="0.93", lw=0)
    ax.axvspan(hi_edge, xmax, color="0.93", lw=0)
    ax.hist(x[inside], bins=bins, color="#2ca02c", alpha=0.8, label=f"N={ok.sum()}")
    n_low, n_high = int(np.sum(x < xmin)), int(np.sum(x > xmax))
    if n_low or n_high:
        ax.bar([xmin + width / 2, xmax - width / 2], [n_low, n_high], width=width, color="#d62728",
               label=f"{n_low + n_high} beyond axis (piled at edge)")
    for edge in DIRECT_LOG_BOUNDS:
        ax.axvline(edge, color="black", lw=1.0, ls="--")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(f"{xlabel}   (shaded = outside search box)")
    ax.set_ylabel("count")
    ax.set_title(f"{label}: {int(np.sum((x < lo_edge) | (x > hi_edge)))}/{ok.sum()} fits outside search box")
    ax.legend(fontsize=7, loc="upper right")

    outside = (x < lo_edge) | (x > hi_edge)
    dist = np.where(x < lo_edge, lo_edge - x, x - hi_edge)[outside]
    order = np.argsort(dist)[::-1]
    return list(zip(galaxies_ok[outside][order], x[outside][order]))


def _outlier_footer(ax, outliers_by_label, n_list=10):
    ax.axis("off")
    lines = []
    for label, outliers in outliers_by_label:
        if not outliers:
            continue
        items = [f"{g} ({v:.1f})" for g, v in outliers[:n_list]]
        if len(outliers) > n_list:
            items.append(f"+{len(outliers) - n_list} more")
        lines += textwrap.wrap(f"{label} outside search box [log10]: " + ", ".join(items), width=150,
                               subsequent_indent="    ")
    ax.text(0.0, 1.0, "\n".join(lines) or "All fits inside the search box.", transform=ax.transAxes,
            fontsize=7, va="top", ha="left", family="monospace")


def _posterior_sigmas(lk, fcn, nshape, theta, n_gal):
    """Marginal posterior sigma of the galaxy parameters (Inc, D[, f]) at theta =
    [shape..., (rho0, rs), Inc, D, (f)]: sqrt(diag(H^-1)) of the full Hessian of the
    same loss the fit minimises (include_priors=True), so correlations with the shape,
    rho0 and rs parameters are included. nan if the Hessian is not positive definite."""
    try:
        _, eq, integrated = lk.run_sympify(fcn, tmax=5, try_integration=False)
        eq_numpy = lambdify_equation(eq, nshape)
        hess = lk.get_loss(eq_numpy, integrated, value="hessian", include_priors=True)
        H = np.asarray(hess(jnp.asarray(theta, dtype=jnp.float64), lk.xvar, lk.yvar,
                            lk.yerr_lo, lk.yerr_hi), dtype=float)
        H = 0.5 * (H + H.T)
        d = np.diag(H)
        if not np.all(np.isfinite(H)) or np.any(d <= 0):
            return np.full(n_gal, np.nan)
        # Work in rescaled units (rho0 ~ 1e7, D ~ 1e5 make the raw eigenvalues span
        # ~20 decades, so a raw positivity test is just round-off).
        s = 1.0 / np.sqrt(d)
        Hs = H * np.outer(s, s)
        ev, V = np.linalg.eigh(Hs)
        tol = 1e-10 * ev.max()
        if ev.min() < -1e-6 * ev.max():  # genuinely not a minimum
            return np.full(n_gal, np.nan)
        keep = ev > tol
        # Exact degeneracies (e.g. pow(x,a0): only rho0*rs^-a0 is constrained) are
        # dropped -- pseudo-inverse -- as long as they don't involve Inc, D, f;
        # otherwise those galaxy parameters are unconstrained -> nan.
        null_weight_gal = np.sum(V[-n_gal:, ~keep] ** 2, axis=1)
        cov_s = (V[:, keep] / ev[keep]) @ V[:, keep].T
        sig = np.sqrt(np.diag(cov_s)) * s
        sig_gal = sig[-n_gal:]
        sig_gal[null_weight_gal > 1e-3] = np.nan
        return sig_gal
    except Exception as e:
        print(f"  posterior sigma failed for {lk.name if hasattr(lk, 'name') else ''} {fcn}: "
              f"{type(e).__name__}: {e}")
        return np.full(n_gal, np.nan)


def write_param_pages(pdf, picks, galaxies, use_physical_scale, data_file=DATA_FILE, fit_ml=False):
    """Append one page per pick (Inc/D prior pulls, search-box histograms for
    rho0/rs/shape params) to an open PdfPages. Shared with
    plot_top3_populations.py."""
    n_extra = 2 if use_physical_scale else 0
    max_param_total = MAX_FUN_PARAM + n_extra
    n_top = len(picks)
    top_fcn_strings = [p["fcn"] for p in picks]
    nshape_list = [simplifier.count_params([fs], MAX_FUN_PARAM)[0] for fs in top_fcn_strings]

    for k in range(n_top):
        per_galaxy_data = picks[k]["per_galaxy_data"]
        j = picks[k]["row"]
        nshape = nshape_list[k]

        inc_fit_vals, d_fit_vals, f_fit_vals = [], [], []
        inc_post_sig, d_post_sig, f_post_sig, f_prior_sig = [], [], [], []
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

            lk = MIGHTEELikelihood(data_file=data_file, name=galaxy, run_name="prior_plot_only",
                                    use_physical_scale=use_physical_scale, fit_ml=fit_ml)
            # f_ml runs: match.py writes f right after Inc, D in codelen_matches
            f_fit = float(row[3 + 2 * max_param_total + 2]) if fit_ml else None
            galaxy_vals = [inc_fit, d_fit] + ([f_fit] if fit_ml else [])
            theta = np.concatenate([params[:nshape], params[MAX_FUN_PARAM:MAX_FUN_PARAM + n_extra],
                                    galaxy_vals])
            if os.environ.get("NO_POSTERIOR_SIGMA") == "1":  # fast mode: prior error bars only
                sig = np.full(len(galaxy_vals), np.nan)
            else:
                sig = _posterior_sigmas(lk, top_fcn_strings[k], nshape, theta, len(galaxy_vals))
            inc_post_sig.append(sig[0])
            d_post_sig.append(sig[1])
            if fit_ml:
                f_fit_vals.append(f_fit)
                f_post_sig.append(sig[2])
                f_prior_sig.append(lk.sigma_frac_ml)

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
        n_pull_rows = 3 if fit_ml else 2  # Inc, D (, f) -- each row is (hist, scatter)
        n_box_rows = -(-n_box_panels // n_cols)  # ceil
        n_rows = n_pull_rows + n_box_rows

        fig = plt.figure(figsize=(11, 4 * n_rows + 1.3))
        gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.3])
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)])
        footer_ax = fig.add_subplot(gs[n_rows, :])

        _pull_panel(axes[0, 0], axes[0, 1], "Inc", inc_fit_vals, inc_prior_means, inc_prior_sigmas,
                    post_sigmas=inc_post_sig)
        _pull_panel(axes[1, 0], axes[1, 1], "D", d_fit_vals, d_prior_means, d_prior_sigmas, log_scale=True,
                    post_sigmas=d_post_sig)
        if fit_ml:  # prior TruncatedNormal(1, sigma_frac, low=0), same for every galaxy's mean
            _pull_panel(axes[2, 0], axes[2, 1], "f_ml", f_fit_vals, np.ones(len(f_fit_vals)), f_prior_sig,
                        post_sigmas=f_post_sig)

        outliers_by_label = []
        for bi, (blabel, bvals) in enumerate(zip(box_labels, box_series)):
            r, c = divmod(bi, n_cols)
            outliers_by_label.append((blabel, _box_panel(axes[n_pull_rows + r, c], blabel, bvals)))
        _outlier_footer(footer_ax, outliers_by_label)
        for bi in range(n_box_panels, n_box_rows * n_cols):
            r, c = divmod(bi, n_cols)
            axes[n_pull_rows + r, c].axis("off")

        fig.suptitle(f"comp={picks[k]['comp']}  rank #{k+1} (DL={picks[k]['DL']:.1f}): {top_fcn_strings[k]}")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
        pdf.savefig(fig)
        plt.close(fig)


def main():
    args = _parse_args()
    use_physical_scale = args.rho_tag == "rhoTrue"
    out_root = args.out_root or os.path.join(
        BASE_DIR, "output_glamdring", f"output_{args.run_name}_{args.rho_tag}")

    with open(GALAXY_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]

    picks, comp_label = build_picks(args.comp, args.comps, args.n_top, out_root, FN_DIR, galaxies,
                                     run_name=args.run_name)

    if args.pdf_out:
        pdf_out = args.pdf_out
    elif args.comps is not None:
        pdf_out = os.path.join(out_root, "GLOBAL", f"mightee_prior_separation_{comp_label}.pdf")
        os.makedirs(os.path.dirname(pdf_out), exist_ok=True)
    else:
        pdf_out = os.path.join(BASE_DIR, f"mightee_prior_separation_{comp_label}.pdf")

    with PdfPages(pdf_out) as pdf:
        write_param_pages(pdf, picks, galaxies, use_physical_scale,
                          fit_ml=args.run_name.endswith("_fml"))

    print(f"\nSaved: {pdf_out}")


if __name__ == "__main__":
    main()
