

import argparse
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

os.environ.setdefault(
    "ESR_FUNCTION_LIBRARY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "function_library", "core_maths"),
)

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import corner as corner_pkg

from esr.fitting.dm_likelihood import MIGHTEELikelihood
import esr.fitting.match as match_mod

import plot_posterior_check as grid  # reuse load_galaxy_row, build_fop, is_saturated

numpyro.set_host_device_count(1)

RESULTS_DIR = grid.RESULTS_DIR
GALAXY_NAMES_FILE = grid.GALAXY_NAMES_FILE
DATA_FILE = grid.DATA_FILE
SCRATCH_DATA_DIR = grid.SCRATCH_DATA_DIR
OUT_DIR = grid.OUT_DIR

HALF_WIDTH_DEX = 10.0    # uniform box half-width, in log10 decades -- fixed, NOT tied to Delta
NUM_WARMUP = 1500
NUM_SAMPLES = 2000
# check_density's hard 0/+inf penalty wall (dm_likelihood.py) has no gradient
# information right at its boundary, and rho0/rs are strongly correlated for
# most of these fits (an amplitude-scale degeneracy) -- numpyro's defaults
# (target_accept=0.8, diagonal mass matrix) badly mismatch that geometry,
# producing hundreds of divergences and r_hat>1.3 on the worst-constrained
# (typically N<4 datapoint) galaxies. Verified empirically on the single
# worst offender in the comp6 population (ID_720_49397, N=2 datapoints,
# 454 divergences/r_hat=1.30 at the old defaults): this combination gets it
# to 0 divergences/r_hat<=1.01.
NUTS_TARGET_ACCEPT_PROB = 0.98
NUTS_DENSE_MASS = True
NUTS_MAX_TREE_DEPTH = 12

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_DATA_DIR, exist_ok=True)

SANE_ABS_MAX = 1e15
MAX_FUN_PARAM = 4 


def is_sane_ml(val):
    """Finite AND nonzero -- log10(0) = -inf, so an exactly-zero ML value
    (a genuinely degenerate fit, or an unused MAX_FUN_PARAM padding slot
    read as if it were a real parameter) needs the same population-fallback
    treatment as +/-inf/NaN, not a -inf sampling-window center."""
    return np.isfinite(val) and val != 0


def active_param_names(nshape):
    """Names of every actively-sampled parameter for a function with this
    many free shape params, in the fixed order extract_theta_ml uses:
    a0..a{nshape-1}, then rho0, rs (n_extra=2 -- this script only ever runs
    in physical-scale mode). nshape=0 (e.g. 1/(x*(x+1/x)), 1/(x*pow(x,x)))
    is the common case for the current top-ranked functions, not an edge
    case -- there is no fixed parameter count across functions."""
    return [f"a{i}" for i in range(nshape)] + ["rho0", "rs"]


def extract_theta_ml(row, nshape):
    """theta_ml in the SAME order as active_param_names(nshape) -- only the
    first nshape of row["params"]'s MAX_FUN_PARAM shape slots are real for
    THIS function; the rest are padding and must not be sampled as if they
    were live parameters (that padding being silently treated as a real,
    always-finite a0 was the original bug this generalization fixes)."""
    return np.concatenate([row["params"][:nshape], row["params"][MAX_FUN_PARAM:MAX_FUN_PARAM + 2]])


def make_model(fop, signs, log_centers):
    """signs/log_centers: one entry per active parameter (see
    active_param_names), same fixed order used throughout this file. Each
    parameter is sampled as log10(|value|) in a flat uniform box of
    +/-HALF_WIDTH_DEX decades around its own MDL/ML point estimate -- a flat
    prior, not a Gaussian, so any Gaussian shape seen in the resulting
    posterior comes from the likelihood itself (the point of this check),
    not an inherited prior shape. Each parameter's sign is fixed at its ML
    estimate's own sign, not sampled: a sign flip is a discrete, different
    equation branch, not a continuous uncertainty on this one, and rho0/rs
    are physical amplitude/scale (always positive by construction) anyway."""
    n_active = len(log_centers)
    site_names = [f"log10_abs_p{i}" for i in range(n_active)]

    def model():
        log_vals = [
            numpyro.sample(
                site_names[i],
                dist.Uniform(log_centers[i] - HALF_WIDTH_DEX, log_centers[i] + HALF_WIDTH_DEX),
            )
            for i in range(n_active)
        ]
        theta = jnp.stack([signs[i] * 10.0 ** log_vals[i] for i in range(n_active)])
        nll = fop(theta)
        numpyro.factor("loglike", -nll)

    return model, site_names


def run_hmc_for_galaxy(fop, signs, log_centers, seed):
    model, site_names = make_model(fop, signs, log_centers)
    init_params = {name: jnp.asarray(c) for name, c in zip(site_names, log_centers)}
    kernel = NUTS(
        model, init_strategy=numpyro.infer.init_to_value(values=init_params),
        target_accept_prob=NUTS_TARGET_ACCEPT_PROB, dense_mass=NUTS_DENSE_MASS,
        max_tree_depth=NUTS_MAX_TREE_DEPTH,
    )
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    mcmc.print_summary()
    samples = mcmc.get_samples()
    # Captured as data (not just printed text) so it can be written to
    # hmc_convergence.txt alongside the plots -- print_summary()'s table is
    # otherwise only ever visible in whatever terminal/log happens to be
    # capturing stdout for this particular run, and is lost once that's gone.
    stats = numpyro.diagnostics.summary({k: v[None, ...] for k, v in samples.items()}, group_by_chain=True)
    n_divergent = int(np.sum(mcmc.get_extra_fields()["diverging"]))
    return samples, site_names, stats, n_divergent


def samples_to_array(samples, site_names):
    """(N, n_active) array in LOG10 units, columns matching site_names/
    active_param_names order.

    Plotted in log space, not linear -- these are sampled as log-magnitude
    uniforms and can genuinely span many decades (shape params especially,
    in their saturating/uninformative regime); exponentiating back to linear
    units for the corner plot lets a handful of rare tail draws blow up the
    visible axis to the point of squashing everything else into one bin.
    Log space is also the natural comparison point for the codelen's own
    log(|theta|)-based Rissanen encoding.
    """
    return np.stack([np.asarray(samples[name]) for name in site_names], axis=1)


def load_all_rows(galaxies, fcn_index):
    """First pass: load every galaxy's stored row (cheap, no likelihood
    build) so a sane population-level log10 anchor can be computed before
    any HMC run -- needed for the handful of galaxies whose OWN stored ML
    is itself saturated/garbage."""
    rows = {}
    for galaxy in galaxies:
        row = grid.load_galaxy_row(galaxy, fcn_index=fcn_index)
        if row is not None:
            rows[galaxy] = row
    return rows


def safe_fcn_name(fcn_string):
    """Filesystem-safe stand-in for an ESR function string, used in output
    filenames so different functions' HMC runs don't overwrite each other."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", fcn_string).strip("_")


RC_N_DRAWS = 400  # posterior draws pushed through the model for the RC band


def compute_rc_bands(likelihood, eq_numpy, integrated, inc_fit, d_fit, theta_ml,
                     used_fallback, active_names, arr, seed=0):
    """Rotation curve: data, optimiser best fit, and posterior percentiles of
    the model's total v_circ (same Inc/D the HMC held fixed).

    Evaluated at the OBSERVED radii only: get_pred's baryon term is the
    catalog v_bar, index-aligned with xvar and undefined between data
    points, so a finer grid would mean inventing baryon values."""
    x = likelihood.xvar
    order = np.argsort(np.asarray(x))

    def pred(theta):
        return likelihood.get_pred(x, theta, eq_numpy, integrated, D=d_fit, Inc=inc_fit)

    # Same sign convention the sampler used (make_model's docstring).
    signs = np.array([1.0 if name in used_fallback else float(np.sign(v))
                      for name, v in zip(active_names, theta_ml)])
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(arr), size=min(RC_N_DRAWS, len(arr)), replace=False)
    v_post = np.asarray(jax.vmap(pred)(jnp.asarray(signs * 10.0 ** arr[draws])))
    v_post = v_post[np.all(np.isfinite(v_post), axis=1)]

    v_ml = None if used_fallback else np.asarray(pred(jnp.asarray(theta_ml)))[order]
    return {
        "r": np.asarray(x)[order],
        "v": np.asarray(likelihood.yvar)[order],
        "err_lo": np.asarray(likelihood.yerr_lo)[order],
        "err_hi": np.asarray(likelihood.yerr_hi)[order],
        "v_ml": v_ml,
        "q": np.percentile(v_post, [2.5, 16, 50, 84, 97.5], axis=0)[:, order] if len(v_post) else None,
    }


def render_galaxy_corner_page(fcn_string, comp, galaxy, n_data, active_names,
                               theta_ml, used_fallback, arr, stats, site_names,
                               n_divergent, rc=None):
    """Builds one fully-annotated corner-plot page for a single galaxy --
    shared between main()'s own serial loop and merge_posterior_hmc.py (the
    Glamdring-parallelized workflow's merge step), so the annotation/title/
    trust-label logic lives in exactly one place regardless of which of
    those two calls it. `rc` (from compute_rc_bands) adds a rotation-curve
    panel under the summary table. Returns the matplotlib Figure; caller
    saves it."""
    labels = [f"log10({n})" if n in ("rho0", "rs") else f"log10|{n}|" for n in active_names]

    # The optimiser's own answer, drawn as an explicit colored line in every
    # panel (not corner's default thin/blended truths style) -- but ONLY for
    # a parameter whose own ML was sane enough to actually center the
    # sampling window on. When the population fallback anchor had to be used
    # instead, the old ML is off the plotted window by construction (that's
    # WHY it needed a fallback) -- a truth line there would be silently
    # invisible or misleading, so it's replaced with a visible annotation
    # instead. In log10 space, matching `arr`.
    truths = [
        np.log10(abs(theta_ml[k])) if active_names[k] not in used_fallback else None
        for k in range(len(active_names))
    ]

    # Per-panel titles off: the same numbers go in the summary table on the
    # right, and corner's titles crowd/overlap the small panels.
    fig = corner_pkg.corner(
        arr, labels=labels, truths=truths, truth_color="C1",
        show_titles=False, label_kwargs={"fontsize": 9},
    )

    # A4 landscape page. Squeeze corner's own axes into a box on the left so
    # nothing else on the page is drawn over them.
    fig.set_size_inches(11.7, 8.3)
    corner_axes = list(fig.axes)
    boxes = [ax.get_position() for ax in corner_axes]
    x0, y0 = min(b.x0 for b in boxes), min(b.y0 for b in boxes)
    x1, y1 = max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    tx0, ty0, tw, th = 0.07, 0.17, 0.38, 0.62
    for ax, b in zip(corner_axes, boxes):
        ax.set_position([
            tx0 + (b.x0 - x0) / (x1 - x0) * tw,
            ty0 + (b.y0 - y0) / (y1 - y0) * th,
            b.width / (x1 - x0) * tw,
            b.height / (y1 - y0) * th,
        ])
        ax.tick_params(labelsize=7)

    # Header, above the corner plot.
    fig.text(0.07, 0.95, galaxy, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.07, 0.905, f"{fcn_string}   (comp={comp})   N={n_data} datapoints",
             fontsize=11, ha="left", va="top")

    max_rhat = max(stats[site]["r_hat"] for site in site_names)

    # Full summary, on the right, in its own empty area.
    cols = ["param", "ML", "mean", "std", "median", "5%", "95%", "n_eff", "r_hat"]
    lines = ["Posterior summary (log10 units)", "",
             f"{cols[0]:<8}" + "".join(f"{c:>8}" for c in cols[1:])]
    for k, (name, site) in enumerate(zip(active_names, site_names)):
        s = stats[site]
        ml = f"{np.log10(abs(theta_ml[k])):8.3f}" if name not in used_fallback and theta_ml[k] != 0 else f"{'--':>8}"
        lines.append(
            f"{name:<8}{ml}"
            f"{s['mean']:8.3f}{s['std']:8.3f}{s['median']:8.3f}"
            f"{s['5.0%']:8.3f}{s['95.0%']:8.3f}{s['n_eff']:8.0f}{s['r_hat']:8.3f}"
        )
    lines += ["", f"divergences: {n_divergent}", f"max r_hat:   {max_rhat:.3f}"]
    if used_fallback:
        lines += ["", "ML unusable, sampling window centred on the",
                  "population fallback for: " + ", ".join(used_fallback)]
    lines += ["", "Orange lines: optimiser (MDL) best fit."]
    fig.text(0.49, 0.79, "\n".join(lines), fontsize=8.5, family="monospace",
             ha="left", va="top")

    if rc is not None:
        ax = fig.add_axes([0.55, 0.10, 0.40, 0.36])
        if rc["q"] is not None:
            q = rc["q"]
            ax.fill_between(rc["r"], q[0], q[4], color="C0", alpha=0.15, lw=0, label="posterior 95%")
            ax.fill_between(rc["r"], q[1], q[3], color="C0", alpha=0.35, lw=0, label="posterior 68%")
            ax.plot(rc["r"], q[2], color="C0", lw=1.5, label="posterior median")
        if rc["v_ml"] is not None:
            ax.plot(rc["r"], rc["v_ml"], color="C1", lw=1.5, ls="--", marker="o", ms=3,
                    label="optimiser best fit")
        ax.errorbar(rc["r"], rc["v"], yerr=[rc["err_lo"], rc["err_hi"]], fmt="o", color="k",
                    ms=4, capsize=2, zorder=5, label="data")
        ax.set_xlabel("R [kpc]", fontsize=9)
        ax.set_ylabel("v_circ [km/s]", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_title("Rotation curve (model evaluated at the observed radii)", fontsize=9)
        ax.legend(fontsize=7, loc="best")
    return fig


def compute_fallback_log_centers(rows, nshape):
    """Population median of log10|value| across galaxies whose OWN ML value
    is sane -- used to center the sampling window for the rare galaxy whose
    own ML is not. Only computed for the parameters actually active at this
    nshape (active_param_names(nshape)): a shape slot beyond nshape is
    padding (always 0 = always "sane" by is_sane_ml's own finite-only
    check), so blindly computing a fallback for every MAX_FUN_PARAM slot
    regardless of nshape would silently take log10(0) = -inf over an
    all-padding column for any function with nshape < MAX_FUN_PARAM."""
    fallbacks = {}
    param_cols = {f"a{i}": i for i in range(nshape)}
    param_cols["rho0"] = MAX_FUN_PARAM
    param_cols["rs"] = MAX_FUN_PARAM + 1
    defaults = {f"a{i}": 0.0 for i in range(nshape)}
    defaults["rho0"] = 7.0
    defaults["rs"] = 3.0
    for name, col in param_cols.items():
        vals = [row["params"][col] for row in rows.values() if is_sane_ml(row["params"][col])]
        fallbacks[name] = float(np.median(np.log10(np.abs(vals)))) if vals else defaults[name]
    return fallbacks


def main(comp, fcn_string, save_per_galaxy_png=False):
    grid.COMP = comp  # load_galaxy_row (defined in plot_posterior_check.py) reads COMP from
                      # THAT module's own globals, not this file's.
    fcn_index = grid.find_fcn_index(fcn_string, comp=comp)
    tag = f"comp{comp}_{safe_fcn_name(fcn_string)}"

    with open(GALAXY_NAMES_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]

    pdf_path = os.path.join(OUT_DIR, f"hmc_corners_all_galaxies_{tag}.pdf")
    if save_per_galaxy_png:
        per_galaxy_dir = os.path.join(OUT_DIR, f"hmc_corners_by_galaxy_{tag}")
        os.makedirs(per_galaxy_dir, exist_ok=True)
    summary_lines = []

    rows = load_all_rows(galaxies, fcn_index)

    # nshape depends only on fcn_string, not the galaxy -- determine it once
    # up front (needed by compute_fallback_log_centers before the per-galaxy
    # loop even starts) via a throwaway probe likelihood.
    probe_galaxy = next(iter(rows))
    probe_likelihood = MIGHTEELikelihood(
        data_file=DATA_FILE, name=probe_galaxy, run_name="posterior_hmc_probe",
        data_dir=SCRATCH_DATA_DIR, use_physical_scale=True,
    )
    probe_likelihood.run_sympify(fcn_string, tmax=60, try_integration=False)
    nshape = probe_likelihood.nparam_shape
    active_names = active_param_names(nshape)
    print(f"{fcn_string!r} (comp={comp}): nshape={nshape} free shape parameter(s); "
          f"active parameters = {active_names}")

    fallback_log10 = compute_fallback_log_centers(rows, nshape)
    print(f"Fallback log10 anchors (used only when a galaxy's own ML is itself "
          f"saturated): {fallback_log10}")

    n_ok, n_skip = 0, 0
    with PdfPages(pdf_path) as pdf:
        for gi, galaxy in enumerate(galaxies):
            row = rows.get(galaxy)
            if row is None:
                n_skip += 1
                continue

            try:
                likelihood = MIGHTEELikelihood(
                    data_file=DATA_FILE, name=galaxy,
                    run_name=f"posterior_hmc/{galaxy}/comp{comp}",
                    data_dir=SCRATCH_DATA_DIR, use_physical_scale=True,
                )
            except Exception as e:
                print(f"[{galaxy}] likelihood build failed: {e}")
                n_skip += 1
                continue

            n_data = len(likelihood.xvar)
            fcn_i, eq, integrated = likelihood.run_sympify(fcn_string, tmax=60, try_integration=False)
            eq_numpy = match_mod.make_lambdified_eq(eq, nshape)

            theta_ml = extract_theta_ml(row, nshape)

            fop = grid.build_fop(likelihood, eq_numpy, integrated, row["inc"], row["d"])

            # Center each param's log-magnitude uniform box on this galaxy's
            # own optimiser answer when it's sane, else on the population
            # fallback anchor (see compute_fallback_log_centers). Sign is
            # fixed at the ML estimate's own sign (make_model docstring).
            log_centers = []
            signs = []
            used_fallback = []
            for name, val in zip(active_names, theta_ml):
                if is_sane_ml(val):
                    log_centers.append(np.log10(abs(val)))
                    signs.append(float(np.sign(val)))
                else:
                    log_centers.append(fallback_log10[name])
                    signs.append(1.0)
                    used_fallback.append(name)

            try:
                samples, site_names, stats, n_divergent = run_hmc_for_galaxy(fop, signs, log_centers, seed=gi)
            except Exception as e:
                print(f"[{galaxy}] NUTS failed: {type(e).__name__}: {e}")
                n_skip += 1
                continue

            arr = samples_to_array(samples, site_names)
            rc = compute_rc_bands(likelihood, eq_numpy, integrated, row["inc"], row["d"],
                                  theta_ml, used_fallback, active_names, arr, seed=gi)
            fig = render_galaxy_corner_page(
                fcn_string, comp, galaxy, n_data, active_names, theta_ml,
                used_fallback, arr, stats, site_names, n_divergent, rc=rc,
            )
            pdf.savefig(fig)

            if save_per_galaxy_png:
                safe_name = galaxy.replace("/", "_").replace("+", "p")
                fig.savefig(os.path.join(per_galaxy_dir, f"{safe_name}.png"), dpi=140, bbox_inches="tight")

            plt.close(fig)

            n_ok += 1
            stat_bits = "; ".join(
                f"{name}: mean={stats[site]['mean']:.3g} std={stats[site]['std']:.3g} "
                f"n_eff={stats[site]['n_eff']:.0f} r_hat={stats[site]['r_hat']:.3f}"
                for name, site in zip(active_names, site_names)
            )
            summary_lines.append(
                f"{galaxy}\tN={n_data}\tused_fallback={used_fallback}\t"
                f"divergences={n_divergent}\t{stat_bits}"
            )
            if (gi + 1) % 10 == 0:
                print(f"{gi + 1}/{len(galaxies)} galaxies processed ({n_ok} ok, {n_skip} skipped)", flush=True)

    summary_path = os.path.join(OUT_DIR, f"hmc_summary_{tag}.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nDone: {n_ok} galaxies plotted, {n_skip} skipped.")
    print(f"PDF: {pdf_path}")
    print(f"Summary: {summary_path}")
    if save_per_galaxy_png:
        print(f"Per-galaxy PNGs: {per_galaxy_dir}/")


def _parse_args():
    p = argparse.ArgumentParser(
        description="HMC posterior check (flat uniform-prior Gaussianity test, centered "
                     "on the MDL/ML point estimate) for one ESR function's rho0/rs (and "
                     "any shape params) across all galaxies."
    )
    p.add_argument("--comp", type=int, default=7)
    p.add_argument("--fcn-string", default="1/(x*(x + 1/x))")
    p.add_argument("--save-per-galaxy-png", action="store_true",
                    help="Also save each galaxy's corner plot as its own PNG under "
                         "hmc_corners_by_galaxy_<tag>/ (default: off, just the combined PDF).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(comp=args.comp, fcn_string=args.fcn_string, save_per_galaxy_png=args.save_per_galaxy_png)
