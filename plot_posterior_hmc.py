

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
# The catalogue the ESR fits being checked were run on (default: the catalogue
# with the 5 arcsec point and Halpha), so the sampled likelihood matches the
# fits whose parameters seed it. Set MIGHTEE_DATA_FILE to use another one.
DATA_FILE = os.environ.get(
    "MIGHTEE_DATA_FILE", "ALL_gbar_gobs_RAR_direct_phot_with_5_arc_radius_plus_Halpha_for_ESR.txt")
SCRATCH_DATA_DIR = grid.SCRATCH_DATA_DIR
OUT_DIR = grid.OUT_DIR

HALF_WIDTH_DEX = 10.0    # "centred" prior: +/- this many decades around the ML value
# Default prior on log10|param|: the SAME fixed box for every galaxy and
# function, so posteriors are comparable and do not depend on where the
# optimiser stopped. Where the data only constrain a combination (rho0 rs or
# rho0 rs^2 ridges), the extent along the ridge is set by the box, whatever
# box is used -- so it is chosen to reach both asymptotic regimes of the
# data, not "as wide as possible": rs from 0.01 kpc (<< smallest radius,
# ~0.3 kpc) to 1e4 kpc (>> largest, ~60 kpc), and rho0 [Msun/kpc^3] wide
# enough to hold the measured combinations at those rs limits (rho0 rs^2 ~
# 1e8 at rs = 0.01 -> 1e12; rho0 rs ~ 1e7 at rs = 1e4 -> 1e3). Beyond those
# limits the likelihood is flat along the ridge: ML values like log rs = -20
# are the optimiser drifting, not something the data ask for. ML_CHECK_TOL
# flags a galaxy whose box does cut the optimum (see ml_check).
PRIOR_MODE = "fixed"     # "fixed" (PRIOR_BOX) or "centred" (ML +/- HALF_WIDTH_DEX)
PRIOR_BOX = {"rho0": (2.0, 13.0), "rs": (-2.0, 4.0), "shape": (-5.0, 5.0)}
ML_CHECK_TOL = 0.1
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


GALAXY_PARAM_NAMES = ["Inc", "D"]
GALAXY_PARAM_NAMES_ML = ["Inc", "D", "f_ml"]  # _fml runs: f also sampled


def n_galaxy_params(active_names):
    """How many trailing entries of active_names are galaxy parameters:
    3 (Inc, D, f_ml), 2 (Inc, D) or 0 (held fixed)."""
    if list(active_names[-3:]) == GALAXY_PARAM_NAMES_ML:
        return 3
    if list(active_names[-2:]) == GALAXY_PARAM_NAMES:
        return 2
    return 0
# Inc and D are sampled as z = (value - catalogue mean) / catalogue sigma in a
# flat box of +/-PRIOR_HALF_WIDTH_SIGMA, clipped to the prior's own support
# (Inc in [0, 90], D > 0). The catalogue truncated-Gaussian prior itself comes
# in through fop (include_priors=True), exactly as in the ESR fits.
PRIOR_HALF_WIDTH_SIGMA = 8.0
INC_LO, INC_HI = 0.0, 90.0


def galaxy_prior(likelihood):
    """Catalogue prior on (Inc, D): truncated Gaussians, D in kpc. With
    fit_ml also f ~ TruncatedNormal(1, sigma_frac, 0, inf), as in the fits."""
    prior = {"inc_mean": float(likelihood.inc_true), "inc_sd": float(likelihood.e_inc),
             "d_mean": float(likelihood.distance_true), "d_sd": float(likelihood.e_d)}
    if getattr(likelihood, "fit_ml", False):
        prior.update(f_mean=1.0, f_sd=float(likelihood.sigma_frac_ml))
    return prior


def _z_box(mean, sd, lo, hi):
    return (max((lo - mean) / sd, -PRIOR_HALF_WIDTH_SIGMA),
            min((hi - mean) / sd, PRIOR_HALF_WIDTH_SIGMA))


def build_fop_free(likelihood, eq_numpy, integrated):
    """-logL - log p(Inc) - log p(D) for a = [model params..., Inc, D]."""
    chi2_fcn = likelihood.get_wrapped_like(
        likelihood.get_loss(eq_numpy, integrated, value="evaluate", include_priors=True))

    def fop(theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)
        return chi2_fcn(theta, likelihood.xvar, likelihood.yvar, likelihood.yerr_lo, likelihood.yerr_hi)

    return fop


def param_boxes(active_names, log_centers, mode=None):
    """Prior box (lo, hi) on log10|param| for each active parameter."""
    mode = PRIOR_MODE if mode is None else mode
    if mode == "centred":
        return [(c - HALF_WIDTH_DEX, c + HALF_WIDTH_DEX) for c in log_centers]
    if mode != "fixed":
        raise ValueError(f"unknown prior mode {mode!r}")
    return [PRIOR_BOX.get(n, PRIOR_BOX["shape"]) for n in active_names]


def ml_check(samples, theta_ml_full, fop):
    """Lowest -logL reached by the posterior samples vs the stored best fit.
    > +ML_CHECK_TOL: the prior box excludes the optimum (box too tight here);
    < -ML_CHECK_TOL: the samples found a better point (optimiser missed it)."""
    nll_ml = float(fop(jnp.asarray(theta_ml_full)))
    nll_min = float(np.min(np.asarray(samples["nll"])))
    delta = nll_min - nll_ml
    flag = ("box cuts the optimum" if delta > ML_CHECK_TOL else
            "optimiser missed the optimum" if delta < -ML_CHECK_TOL else "")
    return {"nll_ml": nll_ml, "nll_min_samples": nll_min, "delta": delta, "flag": flag}


def make_model(fop, signs, log_centers, prior=None, galaxy_init=None, boxes=None):
    """signs/log_centers: one entry per active parameter (see
    active_param_names), same fixed order used throughout this file. Each
    parameter is sampled as log10(|value|) in a flat uniform box of
    +/-HALF_WIDTH_DEX decades around its own MDL/ML point estimate -- a flat
    prior, not a Gaussian, so any Gaussian shape seen in the resulting
    posterior comes from the likelihood itself (the point of this check),
    not an inherited prior shape. Each parameter's sign is fixed at its ML
    estimate's own sign, not sampled: a sign flip is a discrete, different
    equation branch, not a continuous uncertainty on this one, and rho0/rs
    are physical amplitude/scale (always positive by construction) anyway.

    prior: galaxy_prior(likelihood) to also sample Inc and D (fop must then
    be build_fop_free, taking [model params..., Inc, D]); None keeps them
    fixed inside fop. Returns the sampled site names (z-scores for Inc/D)
    and the plotted site names (Inc in deg, D in Mpc, as deterministic
    sites).

    boxes: (lo, hi) on log10|param| per active parameter (param_boxes);
    None = +/-HALF_WIDTH_DEX around log_centers. log_centers is then
    only the starting point, clipped inside the box. -logL is stored per
    sample as the deterministic site "nll" (for ml_check)."""
    n_active = len(log_centers)
    if boxes is None:
        boxes = [(c - HALF_WIDTH_DEX, c + HALF_WIDTH_DEX) for c in log_centers]
    site_names = [f"log10_abs_p{i}" for i in range(n_active)]
    plot_sites = list(site_names)
    fit_ml = prior is not None and "f_sd" in prior
    if prior is not None:
        inc_box = _z_box(prior["inc_mean"], prior["inc_sd"], INC_LO, INC_HI)
        d_box = _z_box(prior["d_mean"], prior["d_sd"], 0.0, np.inf)
        site_names = site_names + ["inc_z", "d_z"]
        plot_sites = plot_sites + ["Inc", "D_Mpc"]
    if fit_ml:
        f_box = _z_box(prior["f_mean"], prior["f_sd"], 0.0, np.inf)
        site_names = site_names + ["f_z"]
        plot_sites = plot_sites + ["f_ml"]

    def model():
        log_vals = [
            numpyro.sample(
                site_names[i],
                dist.Uniform(*boxes[i]),
            )
            for i in range(n_active)
        ]
        theta = jnp.stack([signs[i] * 10.0 ** log_vals[i] for i in range(n_active)])
        if prior is not None:
            inc = prior["inc_mean"] + prior["inc_sd"] * numpyro.sample("inc_z", dist.Uniform(*inc_box))
            d = prior["d_mean"] + prior["d_sd"] * numpyro.sample("d_z", dist.Uniform(*d_box))
            numpyro.deterministic("Inc", inc)
            numpyro.deterministic("D_Mpc", d / 1e3)
            gal = [inc, d]
            if fit_ml:
                f = prior["f_mean"] + prior["f_sd"] * numpyro.sample("f_z", dist.Uniform(*f_box))
                numpyro.deterministic("f_ml", f)
                gal.append(f)
            theta = jnp.concatenate([theta, jnp.stack(gal)])
        nll = fop(theta)
        numpyro.deterministic("nll", nll)
        numpyro.factor("loglike", -nll)

    # strictly inside the box, or init_to_value lands on a zero-density edge
    init = {name: float(np.clip(c, lo + 1e-3, hi - 1e-3))
            for name, c, (lo, hi) in zip(site_names, log_centers, boxes)}
    if prior is not None:
        inc0, d0 = galaxy_init[:2]
        if fit_ml:
            f0 = galaxy_init[2] if len(galaxy_init) > 2 and galaxy_init[2] is not None else 1.0
            init["f_z"] = float(np.clip((f0 - prior["f_mean"]) / prior["f_sd"],
                                        f_box[0] + 1e-3, f_box[1] - 1e-3))
        # strictly inside the box, or init_to_value lands on a zero-density edge
        init["inc_z"] = float(np.clip((inc0 - prior["inc_mean"]) / prior["inc_sd"],
                                      inc_box[0] + 1e-3, inc_box[1] - 1e-3))
        init["d_z"] = float(np.clip((d0 - prior["d_mean"]) / prior["d_sd"],
                                    d_box[0] + 1e-3, d_box[1] - 1e-3))
    return model, site_names, plot_sites, init


def run_hmc_for_galaxy(fop, signs, log_centers, seed, prior=None, galaxy_init=None, boxes=None):
    """Returns samples, plotted site names, stats, divergences."""
    model, site_names, plot_sites, init = make_model(fop, signs, log_centers, prior, galaxy_init, boxes)
    init_params = {name: jnp.asarray(c) for name, c in init.items()}
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
    return samples, plot_sites, stats, n_divergent


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
                     used_fallback, active_names, arr, seed=0, f_fit=None):
    """Rotation curve: data, optimiser best fit, and posterior percentiles of
    the model's total v_circ. When Inc/D were sampled (active_names ends in
    GALAXY_PARAM_NAMES), each draw uses its own Inc/D; otherwise the fixed
    best-fit Inc/D the HMC held.

    Evaluated at the OBSERVED radii only: get_pred's baryon term is the
    catalog v_bar, index-aligned with xvar and undefined between data
    points, so a finer grid would mean inventing baryon values.

    Inclination frame: the data are deprojected at the best-fit Inc,
    v_obs * sin(inc_cat)/sin(inc_fit) (errors alike), so the best-fit curve is
    the physical v_circ. A draw with its own Inc is drawn as
    v_circ * sin(inc_draw)/sin(inc_fit), i.e. where it falls on those data."""
    x = likelihood.xvar
    order = np.argsort(np.asarray(x))
    n_gal = n_galaxy_params(active_names)
    free_gal = n_gal > 0
    n_model = len(active_names) - n_gal
    sin_ref = jnp.sin(jnp.deg2rad(inc_fit))
    corr_data = float(np.sin(np.deg2rad(float(likelihood.inc_true))) / np.sin(np.deg2rad(inc_fit)))

    use_f = bool(getattr(likelihood, "fit_ml", False))  # v_star only exists with fit_ml

    def pred(theta, inc, d, f):
        v_circ = likelihood.get_pred(x, theta, eq_numpy, integrated, D=d, Inc=None,
                                     f_ml=f if use_f else None)
        return v_circ * jnp.sin(jnp.deg2rad(inc)) / sin_ref

    # Same sign convention the sampler used (make_model's docstring).
    signs = np.array([1.0 if name in used_fallback else float(np.sign(v))
                      for name, v in zip(active_names[:n_model], theta_ml[:n_model])])
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(arr), size=min(RC_N_DRAWS, len(arr)), replace=False)
    thetas = jnp.asarray(signs * 10.0 ** arr[draws, :n_model])
    f_ref = 1.0 if f_fit is None else float(f_fit)
    if free_gal:
        incs = jnp.asarray(arr[draws, n_model])
        ds = jnp.asarray(arr[draws, n_model + 1] * 1e3)  # sampled column is Mpc
    else:
        incs = jnp.full(len(draws), inc_fit)
        ds = jnp.full(len(draws), d_fit)
    fs = jnp.asarray(arr[draws, n_model + 2]) if n_gal == 3 else jnp.full(len(draws), f_ref)
    v_post = np.asarray(jax.vmap(pred)(thetas, incs, ds, fs))
    v_post = v_post[np.all(np.isfinite(v_post), axis=1)]

    v_ml = None if used_fallback else np.asarray(
        pred(jnp.asarray(theta_ml[:n_model]), inc_fit, d_fit, f_ref))[order]
    return {
        "r": np.asarray(x)[order],
        "v": np.asarray(likelihood.yvar)[order] * corr_data,
        "err_lo": np.asarray(likelihood.yerr_lo)[order] * corr_data,
        "err_hi": np.asarray(likelihood.yerr_hi)[order] * corr_data,
        "inc_ref": float(inc_fit),
        "corr_data": corr_data,
        "v_ml": v_ml,
        "q": np.percentile(v_post, [2.5, 16, 50, 84, 97.5], axis=0)[:, order] if len(v_post) else None,
    }


PRIOR_SHIFT_FLAG = 2.0  # |shift| in prior sigmas above which a galaxy is flagged


def prior_shifts(stats, site_names, prior):
    """How far the Inc/D (and f) posteriors sit from their priors, in prior
    sigmas, and how wide they are relative to it."""
    s_inc, s_d = stats["Inc"], stats["D_Mpc"]
    out = {
        "inc_shift": (s_inc["median"] - prior["inc_mean"]) / prior["inc_sd"],
        "inc_width": s_inc["std"] / prior["inc_sd"],
        "d_shift": (s_d["median"] * 1e3 - prior["d_mean"]) / prior["d_sd"],
        "d_width": s_d["std"] * 1e3 / prior["d_sd"],
    }
    if "f_sd" in prior and "f_ml" in stats:
        s_f = stats["f_ml"]
        out["f_shift"] = (s_f["median"] - prior["f_mean"]) / prior["f_sd"]
        out["f_width"] = s_f["std"] / prior["f_sd"]
    return out


def max_shift(r):
    return max(abs(r["inc_shift"]), abs(r["d_shift"]), abs(r.get("f_shift", 0.0)))


def shift_table_lines(records):
    """records: dicts with galaxy, n_data and prior_shifts' keys."""
    has_f = any("f_shift" in r for r in records)
    out = [f"{'galaxy':<26}{'N':>3}{'Inc shift':>11}{'Inc width':>11}{'D shift':>10}{'D width':>10}"
           + (f"{'f shift':>10}{'f width':>10}" if has_f else "") + "  flag"]
    for r in sorted(records, key=lambda r: -max_shift(r)):
        flag = "*" if max_shift(r) > PRIOR_SHIFT_FLAG else ""
        f_cols = (f"{r.get('f_shift', np.nan):+10.2f}{r.get('f_width', np.nan):10.2f}" if has_f else "")
        out.append(f"{r['galaxy']:<26}{r['n_data']:>3}{r['inc_shift']:+11.2f}{r['inc_width']:11.2f}"
                   f"{r['d_shift']:+10.2f}{r['d_width']:10.2f}{f_cols}  {flag}")
    return out


def render_prior_shift_page(records, fcn_string, comp):
    """Population page: every galaxy's Inc and D posterior shift from the
    catalogue prior, sorted, so the ones the data pull away stand out."""
    panels = [("inc", "Inc"), ("d", "D")]
    if all("f_shift" in r for r in records):
        panels.append(("f", "f_ML"))
    fig, axes = plt.subplots(1, len(panels), figsize=(11.7, max(8.3, 0.12 * len(records) + 1.5)),
                             sharey=True)
    recs = sorted(records, key=lambda r: r["inc_shift"])
    y = np.arange(len(recs))
    for ax, (key, title) in zip(axes, panels):
        shift = np.array([r[f"{key}_shift"] for r in recs])
        width = np.array([r[f"{key}_width"] for r in recs])
        bad = np.abs(shift) > PRIOR_SHIFT_FLAG
        ax.axvspan(-1, 1, color="C2", alpha=0.15, lw=0)
        ax.axvline(0, color="C2", lw=1)
        for v in (-PRIOR_SHIFT_FLAG, PRIOR_SHIFT_FLAG):
            ax.axvline(v, color="0.5", ls=":", lw=1)
        ax.errorbar(shift[~bad], y[~bad], xerr=width[~bad], fmt="o", ms=3, color="C0", lw=0.8)
        ax.errorbar(shift[bad], y[bad], xerr=width[bad], fmt="o", ms=3, color="C3", lw=0.8)
        ax.set_xlabel(f"({title} posterior median - prior mean) / prior sigma   [bar: posterior std]",
                      fontsize=9)
        ax.set_title(f"{title}: {int(bad.sum())} of {len(recs)} galaxies beyond "
                     f"{PRIOR_SHIFT_FLAG:g} sigma", fontsize=10)
        ax.tick_params(labelsize=7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{r['galaxy']} (N={r['n_data']})" for r in recs], fontsize=5)
    axes[0].set_ylim(-1, len(recs))
    fig.suptitle(f"{fcn_string} (comp={comp}): " + "/".join(t for _, t in panels)
                 + " posteriors vs prior (green: prior +/-1 sigma)", fontsize=12)
    fig.tight_layout()
    return fig


def render_galaxy_corner_page(fcn_string, comp, galaxy, n_data, active_names,
                               theta_ml, used_fallback, arr, stats, site_names,
                               n_divergent, rc=None, prior=None, boxes=None, mlc=None):
    prior_boxes = boxes  # "boxes" is reused below for the corner-axes layout
    """Builds one fully-annotated corner-plot page for a single galaxy --
    shared between main()'s own serial loop and merge_posterior_hmc.py (the
    Glamdring-parallelized workflow's merge step), so the annotation/title/
    trust-label logic lives in exactly one place regardless of which of
    those two calls it. `rc` (from compute_rc_bands) adds a rotation-curve
    panel under the summary table. `prior` (galaxy_prior) draws the
    catalogue Inc/D prior on those columns when they were sampled. Returns
    the matplotlib Figure; caller saves it."""
    n_gal = n_galaxy_params(active_names)
    free_gal = n_gal > 0
    n_model = len(active_names) - n_gal
    labels = [f"log10({n})" if n in ("rho0", "rs") else f"log10|{n}|" for n in active_names[:n_model]]
    if free_gal:
        labels += ["Inc [deg]", "D [Mpc]"] + (["$f_{ML}$"] if n_gal == 3 else [])

    # Values in the plotted units: log10|.| for the model parameters, Inc in
    # deg, D in Mpc, f as is.
    def plotted(k, val):
        if k < n_model:
            return np.log10(abs(val))
        return val / 1e3 if active_names[k] == "D" else val

    if free_gal and prior is not None:
        prior_mean = [prior["inc_mean"], prior["d_mean"] / 1e3]
        prior_sd = [prior["inc_sd"], prior["d_sd"] / 1e3]
        prior_support = [(INC_LO, INC_HI), (0.0, np.inf)]
        if n_gal == 3:
            prior_mean.append(prior["f_mean"])
            prior_sd.append(prior["f_sd"])
            prior_support.append((0.0, np.inf))
    # Axis ranges: the samples' own range, widened on Inc/D to show the
    # prior's +/-3 sigma, so a posterior pulled away from it is obvious.
    ranges = []
    for k in range(arr.shape[1]):
        lo, hi = float(np.min(arr[:, k])), float(np.max(arr[:, k]))
        if k >= n_model and prior is not None:
            j = k - n_model
            lo = max(min(lo, prior_mean[j] - 3 * prior_sd[j]), prior_support[j][0])
            hi = min(max(hi, prior_mean[j] + 3 * prior_sd[j]), prior_support[j][1])
        if k < n_model and prior_boxes is not None:
            # posterior running into a prior-box edge: show the edge
            blo, bhi = prior_boxes[k]
            if lo - blo < 0.5:
                lo = blo
            if bhi - hi < 0.5:
                hi = bhi
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        ranges.append((lo, hi))

    # The optimiser's own answer, drawn as an explicit colored line in every
    # panel (not corner's default thin/blended truths style) -- but ONLY for
    # a parameter whose own ML was sane enough to actually center the
    # sampling window on. When the population fallback anchor had to be used
    # instead, the old ML is off the plotted window by construction (that's
    # WHY it needed a fallback) -- a truth line there would be silently
    # invisible or misleading, so it's replaced with a visible annotation
    # instead. In log10 space, matching `arr`.
    truths = [
        plotted(k, theta_ml[k]) if active_names[k] not in used_fallback and theta_ml[k] != 0 else None
        for k in range(len(active_names))
    ]

    # Per-panel titles off: the same numbers go in the summary table on the
    # right, and corner's titles crowd/overlap the small panels.
    n_bins = 30
    fig = corner_pkg.corner(
        arr, labels=labels, truths=truths, truth_color="C1", bins=n_bins, range=ranges,
        show_titles=False, label_kwargs={"fontsize": 9},
    )

    # Catalogue prior on Inc/D: green line at its mean in every panel of
    # that parameter, and its truncated-Gaussian density (scaled to the
    # histogram counts) with a +/-1 sigma band on the diagonal.
    n_dim = arr.shape[1]
    grid_axes = np.array(fig.axes).reshape(n_dim, n_dim)
    if prior_boxes is not None:
        for k in range(n_model):
            for edge in prior_boxes[k]:
                if ranges[k][0] <= edge <= ranges[k][1]:
                    for i in range(k, n_dim):
                        grid_axes[i, k].axvline(edge, color="0.4", ls=":", lw=1.2)
                    for jj in range(k):
                        grid_axes[k, jj].axhline(edge, color="0.4", ls=":", lw=1.2)
    if free_gal and prior is not None:
        from scipy.stats import truncnorm
        for j in range(n_gal):
            k = n_model + j
            m, s = prior_mean[j], prior_sd[j]
            lo_s, hi_s = prior_support[j]
            pdf = truncnorm((lo_s - m) / s, (hi_s - m) / s, loc=m, scale=s)
            xs = np.linspace(*ranges[k], 300)
            scale = len(arr) * (ranges[k][1] - ranges[k][0]) / n_bins
            ax = grid_axes[k, k]
            ax.axvspan(m - s, m + s, color="C2", alpha=0.15, lw=0)
            ax.plot(xs, pdf.pdf(xs) * scale, color="C2", lw=1.5)
            ax.set_ylim(top=max(ax.get_ylim()[1], 1.05 * np.max(pdf.pdf(xs)) * scale))
            for i in range(k, n_dim):
                grid_axes[i, k].axvline(m, color="C2", ls="--", lw=1)
            for jj in range(k):
                grid_axes[k, jj].axhline(m, color="C2", ls="--", lw=1)

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
    unit_note = ("(log10 units; Inc in deg, D in Mpc" + (", f linear)" if n_gal == 3 else ")")
                 if free_gal else "(log10 units)")
    lines = [f"Posterior summary {unit_note}", "",
             f"{cols[0]:<8}" + "".join(f"{c:>8}" for c in cols[1:])]
    for k, (name, site) in enumerate(zip(active_names, site_names)):
        s = stats[site]
        fmt = "8.1f" if name == "D" else "8.3f"
        ml = f"{truths[k]:{fmt}}" if truths[k] is not None else f"{'--':>8}"
        lines.append(
            f"{name:<8}{ml}"
            + "".join(f"{s[key]:{fmt}}" for key in ("mean", "std", "median", "5.0%", "95.0%"))
            + f"{s['n_eff']:8.0f}{s['r_hat']:8.3f}"
        )
    lines += ["", f"divergences: {n_divergent}", f"max r_hat:   {max_rhat:.3f}"]
    if used_fallback:
        lines += ["", "ML unusable, sampling window centred on the",
                  "population fallback for: " + ", ".join(used_fallback)]
    lines += ["", "Orange lines: optimiser (MDL) best fit."]
    if prior_boxes is not None:
        lines += ["Grey dotted: prior-box edge (shown when reached). Box, log10:",
                  "  " + ", ".join(f"{n} [{lo:g}, {hi:g}]" for n, (lo, hi)
                                   in zip(active_names[:n_model], prior_boxes))]
    if mlc is not None:
        lines += [f"min -logL in posterior {mlc['nll_min_samples']:.2f}, best fit "
                  f"{mlc['nll_ml']:.2f} ({mlc['delta']:+.2f})"]
        if mlc["flag"]:
            lines += ["  -> " + mlc["flag"].upper()]
    if free_gal and prior is not None:
        lines += ["Green: catalogue prior (mean, +/-1 sigma, density)."]
        shifts = prior_shifts(stats, site_names, prior)
        flag = max_shift(shifts) > PRIOR_SHIFT_FLAG
        prior_lines = [
            "Catalogue prior (truncated Gaussian)       shift   width",
            f"  Inc = {prior['inc_mean']:6.2f} +/- {prior['inc_sd']:5.2f} deg"
            f"{shifts['inc_shift']:+15.2f}{shifts['inc_width']:8.2f}",
            f"  D   = {prior['d_mean'] / 1e3:6.1f} +/- {prior['d_sd'] / 1e3:5.1f} Mpc"
            f"{shifts['d_shift']:+15.2f}{shifts['d_width']:8.2f}",
        ]
        if "f_shift" in shifts:
            prior_lines += [f"  f   = {prior['f_mean']:6.2f} +/- {prior['f_sd']:5.2f}    "
                            f"{shifts['f_shift']:+15.2f}{shifts['f_width']:8.2f}"]
        prior_lines += [
            "shift = (posterior median - prior mean) / prior sigma",
            "width = posterior std / prior sigma (~1: prior-dominated)",
        ]
        if flag:
            prior_lines += [f"POSTERIOR PULLED > {PRIOR_SHIFT_FLAG:g} sigma FROM THE PRIOR"]
        fig.text(0.49, 0.95, "\n".join(prior_lines), fontsize=8.5, family="monospace",
                 ha="left", va="top", color="C3" if flag else "k")
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
        data_label = (f"data, deprojected at Inc {rc['inc_ref']:.1f}" if "inc_ref" in rc else "data")
        ax.errorbar(rc["r"], rc["v"], yerr=[rc["err_lo"], rc["err_hi"]], fmt="o", color="k",
                    ms=4, capsize=2, zorder=5, label=data_label)
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


def main(comp, fcn_string, save_per_galaxy_png=False, fixed_inc_d=False):
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
    shift_records = []
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

            if fixed_inc_d:
                fop = grid.build_fop(likelihood, eq_numpy, integrated, row["inc"], row["d"])
                prior = None
            else:
                fop = build_fop_free(likelihood, eq_numpy, integrated)
                prior = galaxy_prior(likelihood)

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
                boxes = param_boxes(active_names, log_centers)
                samples, site_names, stats, n_divergent = run_hmc_for_galaxy(
                    fop, signs, log_centers, seed=gi, prior=prior,
                    galaxy_init=(row["inc"], row["d"]), boxes=boxes)
            except Exception as e:
                print(f"[{galaxy}] NUTS failed: {type(e).__name__}: {e}")
                n_skip += 1
                continue

            arr = samples_to_array(samples, site_names)
            names_g, theta_g = active_names, theta_ml
            if prior is not None:
                names_g = active_names + GALAXY_PARAM_NAMES
                theta_g = np.concatenate([theta_ml, [row["inc"], row["d"]]])
                shift_records.append({"galaxy": galaxy, "n_data": n_data,
                                      **prior_shifts(stats, site_names, prior)})
            rc = compute_rc_bands(likelihood, eq_numpy, integrated, row["inc"], row["d"],
                                  theta_g, used_fallback, names_g, arr, seed=gi)
            mlc = ml_check(samples, theta_g if prior is not None else theta_ml, fop)
            fig = render_galaxy_corner_page(
                fcn_string, comp, galaxy, n_data, names_g, theta_g,
                used_fallback, arr, stats, site_names, n_divergent, rc=rc, prior=prior,
                boxes=boxes, mlc=mlc,
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
                for name, site in zip(names_g, site_names)
            )
            summary_lines.append(
                f"{galaxy}\tN={n_data}\tused_fallback={used_fallback}\t"
                f"divergences={n_divergent}\tmin_nll-ML={mlc['delta']:+.3f}"
                f"{' (' + mlc['flag'] + ')' if mlc['flag'] else ''}\t{stat_bits}"
            )
            if (gi + 1) % 10 == 0:
                print(f"{gi + 1}/{len(galaxies)} galaxies processed ({n_ok} ok, {n_skip} skipped)", flush=True)

        if shift_records:
            fig = render_prior_shift_page(shift_records, fcn_string, comp)
            pdf.savefig(fig)
            plt.close(fig)

    summary_path = os.path.join(OUT_DIR, f"hmc_summary_{tag}.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
        if shift_records:
            f.write("\n\nInc/D posterior vs catalogue prior (sorted by largest shift):\n")
            f.write("\n".join(shift_table_lines(shift_records)) + "\n")

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
    p.add_argument("--fixed-inc-d", action="store_true",
                    help="Hold Inc/D at the ESR best fit instead of sampling them "
                         "(the old behaviour).")
    p.add_argument("--prior-mode", choices=["fixed", "centred"], default=PRIOR_MODE,
                    help="fixed: the same PRIOR_BOX for every galaxy (default); centred: "
                         "ML +/- HALF_WIDTH_DEX decades (the old behaviour).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    PRIOR_MODE = args.prior_mode
    main(comp=args.comp, fcn_string=args.fcn_string, save_per_galaxy_png=args.save_per_galaxy_png,
         fixed_inc_d=args.fixed_inc_d)
