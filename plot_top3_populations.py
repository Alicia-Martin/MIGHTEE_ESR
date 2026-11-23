"""
For a given MIGHTEE complexity: find the top-N population-ranked functions
(reproducing combine_all_galaxies.py's own logic exactly, so the "winning
variant" per function matches what the GLOBAL ranking actually used), then
for every galaxy, plot that galaxy's own rotation-curve data against all N
functions' predicted curves (using that galaxy's own fitted params/Inc/D for
each function, read straight from its codelen_matches_comp{N}.dat -- no
re-optimization). One page per galaxy, saved to a single multi-page PDF,
preceded by one parameter-distribution page per function (shared with
plot_param_priors.py via write_param_pages).

Parameter uncertainties: Delta (saved per-parameter, cols after params) is
this codebase's MDL encoding resolution, Delta = sqrt(12)*Sigma -- so
Sigma = Delta/sqrt(12) recovers the usual 1-sigma uncertainty directly from
already-saved output, no extra Fisher/integral computation needed. The
shaded band per curve is a simple envelope (each active parameter perturbed
+/-1 sigma independently, pointwise min/max across all perturbations) -- not
a proper joint-covariance propagation, but a fast, honest visual indicator
using only what's already saved.

Requires codelen_matches_comp{N}.dat for every galaxy to be present locally
under OUT_ROOT (synced down from wherever the real run happened -- e.g.
Glamdring -- before running this).

Usage:
    python plot_top3_populations.py --comp 6 --n-top 3 --rho-tag rhoTrue
    python plot_top3_populations.py --comps 1,2,3,4,5,6 --n-top 3 --rho-tag rhoFalse
"""
import argparse
import os
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from esr.fitting.dm_likelihood import MIGHTEELikelihood
from esr.fitting.test_all import lambdify_equation
from esr.generation import simplifier

from _top_functions import build_picks
from plot_param_priors import write_param_pages

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


def _fmt_pm(value, sigma):
    """'1.23e+07 +/- 4.5e+05' style, or just the value if sigma is unusable."""
    if np.isfinite(sigma) and sigma > 0:
        return f"{value:.3g} \u00b1 {sigma:.2g}"
    return f"{value:.3g}"


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
    p.add_argument("--fn-library-dir", default=os.environ.get("ESR_FUNCTION_LIBRARY_DIR", FN_DIR),
                    help="function_library/core_maths directory (default: $ESR_FUNCTION_LIBRARY_DIR, "
                    "else <repo>/function_library/core_maths).")
    p.add_argument("--data-file", default=DATA_FILE, help="Catalogue the fits were run on.")
    p.add_argument("--pdf-out", default=None, help="Default: <this dir>/mightee_top<n-top>_comp<comp>.pdf, or "
                    "<out-root>/GLOBAL/mightee_top<n-top>_comp<lo>-<hi>.pdf for --comps")
    p.add_argument("--profiles", default="NFW", help="Comma-separated standard profiles from "
                    "fit_standard_profiles.py to overlay on every galaxy page (dashed), e.g. "
                    "'NFW,Burkert'. '' to skip.")
    p.add_argument("--profiles-csv", default=None,
                    help="fit_standard_profiles.py's per-galaxy table. Must come from the same "
                    "catalogue as --data-file (checked per galaxy via the number of points).")
    return p.parse_args()


def default_profiles_csv(use_physical_scale, fit_ml=False):
    """fit_standard_profiles.py's per-galaxy table for this mode (rhoFalse runs: _rhoFalse; f_ml runs: _fml)."""
    suffix = ("" if use_physical_scale else "_rhoFalse") + ("_fml" if fit_ml else "")
    return os.path.join(BASE_DIR, "posterior_check_output", f"standard_profiles_fit{suffix}.csv")


def load_profile_overlays(names, csv_path, fn_dir, galaxies, use_physical_scale=True):
    """Standard-profile fits (fit_standard_profiles.py) to draw next to the ESR picks.

    Returns a list of dicts with the profile's shape, lambdified function,
    per-galaxy fitted rows and population DL (same definition as that
    script's summary: sum of -logL + codelen over all galaxies, plus aifeyn;
    nan if any galaxy is missing or failed).
    """
    if not names:
        return []
    if not os.path.exists(csv_path):
        print(f"No standard-profile fits at {csv_path} -- run fit_standard_profiles.py first; "
              f"plotting without {names}.")
        return []
    import pandas as pd
    import fit_standard_profiles as fsp

    df = pd.read_csv(csv_path)
    prof = fsp.profiles(fn_dir, use_physical_scale)
    forms = fsp.FORMS if use_physical_scale else fsp.FORMS_FALSE
    styles = [("black", "--"), ("dimgray", ":"), ("gray", "-."), ("darkgray", (0, (5, 1)))]
    overlays = []
    for i, name in enumerate(names):
        if name not in prof:
            print(f"Unknown profile {name!r}; available: {list(prof)}")
            continue
        # Only the galaxies being plotted, so DL matches the ESR DLs of the same set
        sub = df[(df["profile"] == name) & df["galaxy"].isin(galaxies)].set_index("galaxy")
        shape, aifeyn, _ = prof[name]
        ok = all(g in sub.index for g in galaxies) and np.all(
            np.isfinite(sub["negloglike"]) & np.isfinite(sub["codelen"]))
        DL = float(sub["negloglike"].sum() + sub["codelen"].sum() + aifeyn) if ok else np.nan
        nshape = simplifier.count_params([shape], MAX_FUN_PARAM)[0]
        color, ls = styles[i % len(styles)]
        overlays.append({"name": name, "shape": shape, "nshape": nshape, "form": forms[name],
                         "rows": sub, "DL": DL, "color": color, "ls": ls,
                         "physical": use_physical_scale})
    return overlays


def _plot_density(ax, grid, vals, in_range, label, **kw):
    """Smooth rho(r): full strength inside the observed range, faint outside
    it, so the extrapolated part reads as extrapolation."""
    ax.plot(grid, vals, alpha=0.3, **kw)
    ax.plot(np.where(in_range, grid, np.nan), vals, label=label, **kw)


def _inc_factor(inc_a, inc_b):
    """sin(inc_a)/sin(inc_b), inclinations in degrees."""
    return float(np.sin(np.deg2rad(inc_a)) / np.sin(np.deg2rad(inc_b)))


def _reference_inc(picks, galaxy, max_param_total, inc_cat):
    """Fitted Inc of the best pick that has a finite fit for this galaxy (catalogue Inc if none)."""
    for p in picks:
        rows = p["per_galaxy_data"].get(galaxy)
        if rows is None:
            continue
        row = rows[p["row"]]
        if np.isfinite(row[0]) and row[0] != 0:
            return float(row[3 + 2 * max_param_total])
    return inc_cat


def write_galaxy_pages(pdf, picks, galaxies, use_physical_scale, data_file=DATA_FILE, overlays=(), fit_ml=False):
    """Append one page per galaxy (velocity data vs. each pick's smooth
    predicted curve, plus the extrapolated density profile with +/-1-sigma
    band) to an open PdfPages. `picks` is build_picks()'s list of
    {fcn, row, per_galaxy_data, DL, comp} dicts."""
    n_top = len(picks)
    n_extra = 2 if use_physical_scale else 0
    max_param_total = MAX_FUN_PARAM + n_extra
    top_fcn_strings = [p["fcn"] for p in picks]

    eq_numpy_list = []
    nshape_list = []
    for fs in top_fcn_strings:
        nshape = simplifier.count_params([fs], MAX_FUN_PARAM)[0]
        _lk = MIGHTEELikelihood(data_file=data_file, name=galaxies[0], run_name="parse_only",
                                 use_physical_scale=use_physical_scale)
        fcn_clean, eq, integrated = _lk.run_sympify(fs, try_integration=False)
        eq_numpy = lambdify_equation(eq, nshape)
        eq_numpy_list.append((eq_numpy, integrated))
        nshape_list.append(nshape)

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"][:n_top]

    galaxies_with_any_pick = [g for g in galaxies if any(g in p["per_galaxy_data"] for p in picks)]

    for gi, galaxy in enumerate(galaxies_with_any_pick):
        lk = MIGHTEELikelihood(data_file=data_file, name=galaxy, run_name="plot_only",
                                use_physical_scale=use_physical_scale, fit_ml=fit_ml)
        r = np.asarray(lk.xvar, dtype=float)
        v = np.asarray(lk.yvar, dtype=float)
        yerr_lo = np.asarray(lk.yerr_lo, dtype=float)
        yerr_hi = np.asarray(getattr(lk, "yerr_hi", lk.yerr_lo), dtype=float)

        # One inclination frame per panel: the data are deprojected at the best pick's fitted
        # Inc (i_ref), v_obs * sin(inc_cat)/sin(i_ref), errors alike. A model with its own Inc
        # i_k is drawn as v_circ * sin(i_k)/sin(i_ref), so the best pick (and its baryon / DM
        # components) is the physical curve and every model is compared with the same points.
        inc_ref = _reference_inc(picks, galaxy, max_param_total, float(lk.inc_true))
        corr_ref = _inc_factor(float(lk.inc_true), inc_ref)

        fig, (ax_v, ax_rho) = plt.subplots(1, 2, figsize=(12, 6))
        ax_v.errorbar(r, v, yerr=[yerr_lo, yerr_hi], fmt="o", mfc="none", color="gray",
                      ms=4, capsize=0, lw=0.6, label=f"data, catalogue Inc {float(lk.inc_true):.1f}",
                      zorder=4)
        ax_v.errorbar(r, v * corr_ref, yerr=[yerr_lo * corr_ref, yerr_hi * corr_ref], fmt="o",
                      color="black", ms=4, capsize=2, label=f"data, deprojected at Inc {inc_ref:.1f}",
                      zorder=5)

        r_min, r_max = r.min(), r.max()
        order = np.argsort(r)  # lk.xvar is already sorted; kept so lines never zig-zag
        rho_grid = np.logspace(np.log10(r_min) - 2, np.log10(r_max) + 2, 200)
        in_range = (rho_grid >= r_min) & (rho_grid <= r_max)
        main_curve_values = []  # best-fit curves only, NOT the perturbation bands -- a
                                 # +/-1-sigma band riding an exponential-type function out
                                 # to the extrapolated edge of rho_grid can reach absurd
                                 # values (1e21+) that are a real "unconstrained out here"
                                 # signal, not a bug, but must not set the axis scale, or
                                 # every well-behaved curve becomes unreadable.
        param_lines = []  # (color, text) per pick, drawn as a footer below both panels

        for k in range(n_top):
            galaxy_data_k = picks[k]["per_galaxy_data"].get(galaxy)
            if galaxy_data_k is None:
                continue
            j = picks[k]["row"]
            nshape = nshape_list[k]
            eq_numpy, integrated = eq_numpy_list[k]
            row = galaxy_data_k[j]
            negloglike_k, codelen_k, index_k = row[0], row[1], row[2]
            params = row[3:3 + max_param_total]
            deltas = row[3 + max_param_total:3 + 2 * max_param_total]
            inc_fit, d_fit = row[3 + 2 * max_param_total], row[3 + 2 * max_param_total + 1]
            # f_ml runs: match.py writes f right after Inc, D in codelen_matches
            f_fit = float(row[3 + 2 * max_param_total + 2]) if fit_ml else None

            if not np.isfinite(negloglike_k) or negloglike_k == 0:
                continue

            shape_params = params[:nshape]
            extra_params = params[MAX_FUN_PARAM:MAX_FUN_PARAM + n_extra]
            theta_model = np.concatenate([shape_params, extra_params])
            sigma_shape = deltas[:nshape] / np.sqrt(12.0)
            sigma_extra = deltas[MAX_FUN_PARAM:MAX_FUN_PARAM + n_extra] / np.sqrt(12.0)
            all_sigma = np.concatenate([sigma_shape, sigma_extra])

            # get_pred reads self.nparam_shape/self.eq_diff, set as a
            # side effect of run_sympify -- must be called on THIS lk
            # instance (parsing above used a throwaway instance).
            try:
                lk.run_sympify(top_fcn_strings[k], try_integration=False)
            except Exception as e:
                print(f"  {galaxy} fcn#{k+1} run_sympify failed: {type(e).__name__}: {e}")
                continue

            label = f"#{k+1} (comp={picks[k]['comp']}): {top_fcn_strings[k]}  (DL={picks[k]['DL']:.1f})"

            # Fitted values (+/- Delta/sqrt(12)) for the footer under the panels.
            fitted = [f"a{pidx} = {_fmt_pm(shape_params[pidx], sigma_shape[pidx])}" for pidx in range(nshape)]
            if n_extra == 2:
                fitted = [f"rho0 = {_fmt_pm(extra_params[0], sigma_extra[0])}",
                          f"rs = {_fmt_pm(extra_params[1], sigma_extra[1])}"] + fitted
            fitted += [f"Inc = {inc_fit:.1f} (prior {float(lk.inc_true):.1f})",
                       f"D = {d_fit:.4g} (prior {float(lk.distance_true):.4g})"]
            if fit_ml:
                fitted += [f"f_ml = {f_fit:.3f} (prior 1 +/- {lk.sigma_frac_ml:.3f})"]
            param_lines.append((colors[k], f"#{k+1}:  " + "   ".join(fitted)))

            # --- velocity panel: the model only at the data radii, where the
            # catalogue baryon term (lk.v_bar) exists -- exactly what the
            # likelihood scores. Lines between the points are a guide to the
            # eye. For the best function also show the decomposition.
            try:
                # physical curve (Inc=None), then into the i_ref frame
                v_pred, dm_term, baryon_v2 = lk.get_pred(
                    jnp.asarray(r), theta_model, eq_numpy, integrated=integrated,
                    D=d_fit, Inc=None, return_components=True, f_ml=f_fit,
                )
                to_ref = _inc_factor(inc_fit, inc_ref)
                v_pred = np.asarray(v_pred, dtype=float) * to_ref
                dm_term = np.asarray(dm_term, dtype=float) * to_ref ** 2
                baryon_v2 = np.asarray(baryon_v2, dtype=float) * to_ref ** 2
                ax_v.plot(r[order], v_pred[order], color=colors[k], label=label,
                          lw=1.2, marker="s", ms=3.5)
                if k == 0:
                    v_b = np.sqrt(np.maximum(np.asarray(baryon_v2, dtype=float), 0.0))
                    v_dm = np.sqrt(np.maximum(np.asarray(dm_term, dtype=float), 0.0))
                    ax_v.plot(r[order], v_b[order], color="gray", ls="--", lw=1.0,
                              marker=".", label="#1 baryons")
                    ax_v.plot(r[order], v_dm[order], color=colors[k], ls=":", lw=1.2,
                              marker=".", label="#1 dark matter")
            except Exception as e:
                print(f"  {galaxy} fcn#{k+1} velocity plot failed: {type(e).__name__}: {e}")

            # --- density panel: rho(r) = rho0*f(r/rs) (physical-scale
            # mode) or f(r) directly -- pure function of the ESR shape,
            # no baryon term, so a wide log-spaced grid is fine.
            try:
                if use_physical_scale:
                    rho0, rs = extra_params[0], extra_params[1]
                    u_grid = rho_grid / rs
                    f_vals = np.asarray(
                        eq_numpy(jnp.asarray(u_grid)) if nshape == 0
                        else eq_numpy(jnp.asarray(u_grid), *shape_params),
                        dtype=float,
                    )
                    rho_vals = rho0 * f_vals
                else:
                    rho_vals = np.asarray(
                        eq_numpy(jnp.asarray(rho_grid)) if nshape == 0
                        else eq_numpy(jnp.asarray(rho_grid), *shape_params),
                        dtype=float,
                    )
                main_curve_values.append(np.abs(rho_vals))
                _plot_density(ax_rho, rho_grid, np.abs(rho_vals), in_range, label,
                              color=colors[k], lw=1.8)

                # Same +/-1-sigma independent-perturbation envelope as
                # used for the shape codelen elsewhere this session --
                # fast, honest visual indicator, not a joint-covariance
                # propagation.
                curves = [rho_vals]
                for pidx in range(len(theta_model)):
                    s = all_sigma[pidx]
                    if not np.isfinite(s) or s <= 0:
                        continue
                    for sign in (-1, 1):
                        theta_pert = theta_model.copy()
                        theta_pert[pidx] = theta_pert[pidx] + sign * s
                        try:
                            if use_physical_scale:
                                rho0_p, rs_p = theta_pert[nshape], theta_pert[nshape + 1]
                                if rs_p <= 0 or rho0_p <= 0:
                                    continue
                                u_p = rho_grid / rs_p
                                f_p = np.asarray(
                                    eq_numpy(jnp.asarray(u_p)) if nshape == 0
                                    else eq_numpy(jnp.asarray(u_p), *theta_pert[:nshape]),
                                    dtype=float,
                                )
                                curves.append(rho0_p * f_p)
                            else:
                                f_p = np.asarray(
                                    eq_numpy(jnp.asarray(rho_grid)) if nshape == 0
                                    else eq_numpy(jnp.asarray(rho_grid), *theta_pert[:nshape]),
                                    dtype=float,
                                )
                                curves.append(f_p)
                        except Exception:
                            pass
                if len(curves) > 1:
                    stack = np.abs(np.vstack(curves))
                    lo = np.nanmin(stack, axis=0)
                    hi = np.nanmax(stack, axis=0)
                    ax_rho.fill_between(rho_grid, lo, hi, color=colors[k], alpha=0.15, lw=0)
            except Exception as e:
                print(f"  {galaxy} fcn#{k+1} density plot failed: {type(e).__name__}: {e}")

        # Standard profiles (fit_standard_profiles.py): each galaxy's own
        # fitted rho0, rs (and shape parameter, if any), Inc, D -- dashed.
        for ov in overlays:
            if galaxy not in ov["rows"].index:
                continue
            orow = ov["rows"].loc[galaxy]
            if not np.isfinite(orow["negloglike"]):
                continue
            if int(orow["n_data"]) != len(r):
                print(f"  {galaxy} {ov['name']}: fitted to {int(orow['n_data'])} points but this "
                      f"catalogue has {len(r)} -- different data file, not plotted")
                continue
            try:
                fcn_clean, eq_o, integ_o = lk.run_sympify(ov["shape"], try_integration=False)
                eq_o = lambdify_equation(eq_o, ov["nshape"])
                # the function's own parameters a0..; plus rho0, rs in physical-scale mode
                shape_o = [float(orow[f"a{i}"]) for i in range(ov["nshape"])]
                theta_o = np.array(shape_o + ([float(orow["rho0"]), float(orow["rs"])]
                                              if ov["physical"] else []))
                label_o = f"{ov['name']}: {ov['form']}  (DL={ov['DL']:.1f})"

                v_o = np.asarray(lk.get_pred(jnp.asarray(r), theta_o, eq_o, integrated=integ_o,
                                             D=float(orow["d_fit"]), Inc=None,
                                             return_components=True,
                                             f_ml=(float(orow["f_fit"]) if fit_ml and "f_fit" in orow else None))[0], dtype=float)
                v_o = v_o * _inc_factor(float(orow["inc_fit"]), inc_ref)
                ax_v.plot(r[order], v_o[order], color=ov["color"], ls=ov["ls"], lw=1.2,
                          marker="D", ms=3, label=label_o)

                if ov["physical"]:
                    f_o = np.asarray(eq_o(jnp.asarray(rho_grid / float(orow["rs"])), *shape_o), dtype=float)
                    rho_o = np.abs(float(orow["rho0"]) * f_o)
                else:
                    rho_o = np.abs(np.asarray(eq_o(jnp.asarray(rho_grid), *shape_o), dtype=float))
                main_curve_values.append(rho_o)
                _plot_density(ax_rho, rho_grid, rho_o, in_range, label_o,
                              color=ov["color"], ls=ov["ls"], lw=1.6)

                fitted_o = [f"a{i} = {v:.3g}" for i, v in enumerate(shape_o)]
                if ov["physical"]:
                    fitted_o += [f"rho0 = {float(orow['rho0']):.3g}", f"rs = {float(orow['rs']):.3g}"]
                fitted_o += [f"Inc = {float(orow['inc_fit']):.1f}", f"D = {float(orow['d_fit']):.4g}"]
                param_lines.append((ov["color"], f"{ov['name']}:  " + "   ".join(fitted_o)))
            except Exception as e:
                print(f"  {galaxy} {ov['name']} overlay failed: {type(e).__name__}: {e}")

        ax_v.set_xlabel("radius")
        ax_v.set_ylabel("v_circ")
        ax_v.set_title(f"{galaxy} -- velocity")
        ax_v.legend(fontsize=6, loc="best")

        ax_rho.set_xlabel("radius (wide range)")
        ax_rho.set_ylabel("|rho(r)|")
        ax_rho.set_xscale("log")
        ax_rho.set_yscale("log")
        ax_rho.set_title(f"{galaxy} -- density")
        ax_rho.axvspan(r_min, r_max, color="gray", alpha=0.1, label="observed range")
        ax_rho.legend(fontsize=6, loc="best")
        # Fixed floor: a function singular as r->0 or decaying towards 0
        # (inside or outside the observed range) would otherwise drag
        # the y-axis down far enough to squash the actually-constrained
        # density range into an unreadable sliver.
        Y_RHO_FLOOR = 1e-5
        # Ceiling: a couple of decades above the best-fit curves' own
        # peak (NOT the perturbation bands', which are free to run to
        # absurd values at the extrapolated edge -- see main_curve_values
        # above), so it tracks whatever's actually well-constrained
        # instead of matplotlib autoscaling to a +/-1-sigma pole.
        Y_RHO_CEILING_DECADES = 2
        # Only where there are data: an extrapolated curve can blow up far
        # outside them and would otherwise set an unreadable scale.
        all_main = (np.concatenate([c[in_range] for c in main_curve_values])
                    if main_curve_values else np.array([]))
        all_main = all_main[np.isfinite(all_main) & (all_main > 0)]
        # Cap before scaling: a curve can reach ~1e308 (e.g. a near-degenerate
        # fit); scaling that overflows to inf, and even ~1e300 overflows
        # matplotlib's log-axis tick maths. 1e200 is far above any real density.
        y_top = (min(all_main.max(), 1e200) * 10 ** Y_RHO_CEILING_DECADES
                 if all_main.size > 0 else None)
        ax_rho.set_ylim(bottom=Y_RHO_FLOOR, top=y_top)

        fig.suptitle(galaxy)
        footer_h = 0.035 * len(param_lines)
        fig.tight_layout(rect=(0, footer_h + 0.01, 1, 1))
        for li, (color, text) in enumerate(param_lines):
            fig.text(0.01, footer_h - 0.035 * li - 0.01, text, color=color, fontsize=8,
                     ha="left", va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        if (gi + 1) % 20 == 0:
            print(f"  ...{gi+1}/{len(galaxies_with_any_pick)} galaxies plotted")


def main():
    args = _parse_args()
    use_physical_scale = args.rho_tag == "rhoTrue"
    # Default: a local copy of the cluster output in output_glamdring/ if
    # present, else the pipeline's own output folder.
    out_root = args.out_root
    if out_root is None:
        out_root = os.path.join(BASE_DIR, "output_glamdring", f"output_{args.run_name}_{args.rho_tag}")
        if not os.path.isdir(out_root):
            out_root = os.path.join(BASE_DIR, "fitting", "output", f"output_{args.run_name}_{args.rho_tag}")
    print(f"Reading fits from: {out_root}")

    with open(GALAXY_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]

    picks, comp_label = build_picks(args.comp, args.comps, args.n_top, out_root, args.fn_library_dir, galaxies,
                                     run_name=args.run_name)

    if args.pdf_out:
        pdf_out = args.pdf_out
    elif args.comps is not None:
        pdf_out = os.path.join(out_root, "GLOBAL", f"mightee_top{args.n_top}_{comp_label}.pdf")
        os.makedirs(os.path.dirname(pdf_out), exist_ok=True)
    else:
        pdf_out = os.path.join(BASE_DIR, f"mightee_top{args.n_top}_{comp_label}.pdf")

    names = [n.strip().replace("_", " ") for n in args.profiles.split(",") if n.strip()]
    fit_ml = args.run_name.endswith("_fml")
    csv_path = args.profiles_csv or default_profiles_csv(use_physical_scale, fit_ml)
    overlays = load_profile_overlays(names, csv_path, args.fn_library_dir, galaxies,
                                     use_physical_scale)

    with PdfPages(pdf_out) as pdf:
        # Overview first: one parameter-distribution page per function
        # (same panels as plot_param_priors.py), then one page per galaxy.
        write_param_pages(pdf, picks, galaxies, use_physical_scale, data_file=args.data_file, fit_ml=fit_ml)
        write_galaxy_pages(pdf, picks, galaxies, use_physical_scale, data_file=args.data_file,
                           overlays=overlays, fit_ml=fit_ml)

    print(f"\nSaved: {pdf_out}")


if __name__ == "__main__":
    main()
