"""
For a given MIGHTEE complexity: find the top-N population-ranked functions
(reproducing combine_all_galaxies.py's own logic exactly, so the "winning
variant" per function matches what the GLOBAL ranking actually used), then
for every galaxy, plot that galaxy's own rotation-curve data against all N
functions' predicted curves (using that galaxy's own fitted params/Inc/D for
each function, read straight from its codelen_matches_comp{N}.dat -- no
re-optimization). One page per galaxy, saved to a single multi-page PDF.

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

FN_DIR = os.environ.get("ESR_FUNCTION_LIBRARY_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "function_library", "core_maths"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GALAXY_FILE = os.path.join(BASE_DIR, "galaxy_names.txt")
DATA_FILE = os.path.join(BASE_DIR, "ALL_gbar_gobs_RAR_direct_phot.txt")
MAX_FUN_PARAM = 4
# n_extra/max_param_total depend on use_physical_scale (rho0/rs only exist
# in physical-scale mode -- match.py's own column layout, n_extra=2 if
# use_physical_scale else 0) so they're computed per-run in main(), not
# fixed module constants.


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
    p.add_argument("--pdf-out", default=None, help="Default: <this dir>/mightee_top<n-top>_comp<comp>.pdf, or "
                    "<out-root>/GLOBAL/mightee_top<n-top>_comp<lo>-<hi>.pdf for --comps")
    return p.parse_args()


def main():
    args = _parse_args()
    n_top = args.n_top
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
        pdf_out = os.path.join(out_root, "GLOBAL", f"mightee_top{args.n_top}_{comp_label}.pdf")
        os.makedirs(os.path.dirname(pdf_out), exist_ok=True)
    else:
        pdf_out = os.path.join(BASE_DIR, f"mightee_top{args.n_top}_{comp_label}.pdf")

    top_fcn_strings = [p["fcn"] for p in picks]

    eq_numpy_list = []
    nshape_list = []
    for fs in top_fcn_strings:
        nshape = simplifier.count_params([fs], MAX_FUN_PARAM)[0]
        _lk = MIGHTEELikelihood(data_file=DATA_FILE, name=galaxies[0], run_name="parse_only",
                                 use_physical_scale=use_physical_scale)
        fcn_clean, eq, integrated = _lk.run_sympify(fs, try_integration=False)
        eq_numpy = lambdify_equation(eq, nshape)
        eq_numpy_list.append((eq_numpy, integrated))
        nshape_list.append(nshape)

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"][:n_top]

    galaxies_with_any_pick = [g for g in galaxies if any(g in p["per_galaxy_data"] for p in picks)]

    with PdfPages(pdf_out) as pdf:
        for gi, galaxy in enumerate(galaxies_with_any_pick):
            lk = MIGHTEELikelihood(data_file=DATA_FILE, name=galaxy, run_name="plot_only",
                                    use_physical_scale=use_physical_scale)
            r = np.asarray(lk.xvar, dtype=float)
            v = np.asarray(lk.yvar, dtype=float)
            yerr_lo = np.asarray(lk.yerr_lo, dtype=float)
            yerr_hi = np.asarray(getattr(lk, "yerr_hi", lk.yerr_lo), dtype=float)

            fig, (ax_v, ax_rho) = plt.subplots(1, 2, figsize=(12, 5))
            ax_v.errorbar(r, v, yerr=[yerr_lo, yerr_hi], fmt="o", color="black",
                          ms=4, capsize=2, label="data", zorder=5)

            r_min, r_max = r.min(), r.max()
            rho_grid = np.logspace(np.log10(r_min) - 2, np.log10(r_max) + 2, 200)
            main_curve_values = []  # best-fit curves only, NOT the perturbation bands -- a
                                     # +/-1-sigma band riding an exponential-type function out
                                     # to the extrapolated edge of rho_grid can reach absurd
                                     # values (1e21+) that are a real "unconstrained out here"
                                     # signal, not a bug, but must not set the axis scale, or
                                     # every well-behaved curve becomes unreadable.

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

                # --- velocity panel: native radii only (get_pred's baryon
                # term is index-aligned with the galaxy's own xvar; it
                # cannot be evaluated on an arbitrary different-length grid).
                try:
                    v_pred, _, _ = lk.get_pred(
                        jnp.asarray(r), theta_model, eq_numpy, integrated=integrated,
                        D=d_fit, Inc=inc_fit, return_components=True,
                    )
                    v_pred = np.asarray(v_pred, dtype=float)
                    order = np.argsort(r)
                    ax_v.plot(r[order], v_pred[order], color=colors[k], label=label, lw=1.8, marker="o", ms=3)
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
                    ax_rho.plot(rho_grid, np.abs(rho_vals), color=colors[k], label=label, lw=1.8)

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
            all_main = np.concatenate(main_curve_values) if main_curve_values else np.array([])
            all_main = all_main[np.isfinite(all_main) & (all_main > 0)]
            y_top = all_main.max() * 10 ** Y_RHO_CEILING_DECADES if all_main.size > 0 else None
            ax_rho.set_ylim(bottom=Y_RHO_FLOOR, top=y_top)

            fig.suptitle(galaxy)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            if (gi + 1) % 20 == 0:
                print(f"  ...{gi+1}/{len(galaxies_with_any_pick)} galaxies plotted")

    print(f"\nSaved: {pdf_out}")


if __name__ == "__main__":
    main()
