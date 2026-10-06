"""
Validate esr.fitting.test_all.optimise_fun_direct_nm (bounded DIRECT search +
hybrid Nelder-Mead multistart, then a stage-2 unbounded joint Inc/D polish)
against two independent brute-force baselines for a single ESR function on a
single galaxy:

  Arm A: the production pipeline itself (optimise_fun_direct_nm, unmodified).
  Arm B: 50 Nelder-Mead restarts from random points.
  Arm C: 50 BFGS restarts from the SAME random points as Arm B, using an
         analytically-correct gradient in internal (sign * 10**log-magnitude)
         space -- see _stage1_value_and_grad for why this isn't just
         jax.value_and_grad of the physical-space loss.

All three arms search the identical internal parameter space and the
identical stage-1 objective (no Inc/D, no priors), so stage-1 chi2 is
directly comparable across arms: it answers "did the pipeline actually find
the same optimum that many random restarts converge to?"

Usage:
    python validate_fit.py "a0/(a1+x)" J022128.8-042448
    python validate_fit.py "a0*pow(x,a1)/(a2+x)" J022128.8-042448 --n-random 100
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize

from esr.fitting.dm_likelihood import MIGHTEELikelihood
from esr.fitting.test_all import (
    optimise_fun_direct_nm,
    galaxy_params_polish,
    lambdify_equation,
)
import esr.fitting.match as match
import esr.generation.simplifier as simplifier

THIS_DIR = Path(__file__).resolve().parent
# ALL_gbar_gobs_RAR_direct_phot.txt is the fuller catalog (129 galaxies,
# whitespace-delimited) -- not rar_with_ml_direct_phot.csv (81 galaxies).
# dm_likelihood.py's _load_galaxy_data auto-detects the delimiter.
DEFAULT_DATA_FILE = THIS_DIR / "ALL_gbar_gobs_RAR_direct_phot.txt"
MAX_PARAM_SHAPE = 4  # matches optimise_fun_direct_nm's own default


def _random_signs(rng: np.random.Generator, nshape: int, use_physical_scale: bool) -> tuple:
    """Random +-1 sign per shape param; rho0/rs (if present) are always +1."""
    signs = rng.integers(0, 2, size=nshape) * 2 - 1
    if use_physical_scale:
        signs = np.concatenate([signs, [1, 1]])
    return tuple(int(s) for s in signs)


def _generate_random_starts(
    rng: np.random.Generator, nshape: int, nparam: int,
    use_physical_scale: bool, pmin: float, pmax: float, n_random: int,
) -> list[tuple[tuple, np.ndarray]]:
    """Shared (signs, p0) pairs -- Arms B and C restart from the SAME points."""
    starts = []
    for _ in range(n_random):
        signs = _random_signs(rng, nshape, use_physical_scale)
        p0 = rng.uniform(pmin, pmax, size=nparam)
        starts.append((signs, p0))
    return starts


def _stage1_objective(likelihood: MIGHTEELikelihood, eq_numpy, integrated: bool):
    """Same construction as optimise_fun_direct_nm's stage-1 objective: no
    Inc/D, no priors -- so stage-1 chi2 is directly comparable across arms."""
    loss_eval = likelihood.get_loss(eq_numpy, integrated, value="evaluate", include_priors=False)
    wrapped_eval = likelihood.get_wrapped_like(loss_eval)

    def objective(p_internal: np.ndarray, signs: tuple) -> float:
        val = float(wrapped_eval(
            np.asarray(p_internal, dtype=float),
            likelihood.xvar, likelihood.yvar, likelihood.yerr_lo, likelihood.yerr_hi,
            signs=list(signs), check_nans=True,
        ))
        return val if np.isfinite(val) else 1e30

    return objective


def _stage1_value_and_grad(likelihood: MIGHTEELikelihood, eq_numpy, integrated: bool):
    """
    Analytic gradient of the stage-1 objective w.r.t. the INTERNAL (sign *
    10**x) coordinates, for BFGS.

    likelihood.get_loss(..., value="value_and_grad") differentiates w.r.t.
    the PHYSICAL parameter vector p, not the internal x. Since
    p = sign * 10**x, the chain rule gives
        d(loss)/dx = d(loss)/dp * dp/dx = d(loss)/dp * p * ln(10)
    (the sign cancels: d/dx(s*10**x) = s*10**x*ln(10) = p*ln(10) regardless
    of s). Passing d(loss)/dp off as d(loss)/dx directly -- as the sibling
    testing_opt_mightee.py's BFGS path does -- is silently wrong by this
    missing p*ln(10) factor whenever log_opt=True.
    """
    loss_and_grad_phys = likelihood.get_loss(eq_numpy, integrated, value="value_and_grad", include_priors=False)

    def obj_grad(p_internal: np.ndarray, signs: tuple):
        signs_arr = np.asarray(signs, dtype=float)
        # BFGS is unconstrained and can propose internal coordinates well
        # outside the DIRECT box during its line search; 10**x can then
        # overflow to inf, which the isfinite guard below already handles --
        # errstate just silences the resulting (expected, harmless) warning.
        with np.errstate(over="ignore"):
            p_physical = signs_arr * (10.0 ** np.asarray(p_internal, dtype=float))
        value, grad_phys = loss_and_grad_phys(
            jnp.asarray(p_physical, dtype=jnp.float64),
            likelihood.xvar, likelihood.yvar, likelihood.yerr_lo, likelihood.yerr_hi,
        )
        value = float(value)
        grad_phys = np.asarray(grad_phys, dtype=float)
        with np.errstate(invalid="ignore"):
            grad_internal = grad_phys * p_physical * np.log(10.0)
        if not (np.isfinite(value) and np.all(np.isfinite(grad_internal))):
            return 1e30, np.zeros_like(p_internal, dtype=float)
        return value, grad_internal

    return obj_grad


def _run_nm_arm(starts, objective) -> Optional[dict]:
    best = None
    for signs, p0 in starts:
        res = scipy_minimize(lambda p, s=signs: objective(p, s), p0,
                              method="Nelder-Mead", options={"maxiter": 5000})
        if np.isfinite(res.fun) and (best is None or res.fun < best["stage1_chi2"]):
            active_params = (np.asarray(signs, dtype=float) * (10.0 ** np.asarray(res.x, dtype=float))).ravel()
            best = {"stage1_chi2": float(res.fun), "active_params": active_params, "signs": signs}
    return best


def _run_bfgs_arm(starts, obj_grad) -> Optional[dict]:
    best = None
    for signs, p0 in starts:
        res = scipy_minimize(lambda p, s=signs: obj_grad(p, s), p0,
                              method="BFGS", jac=True, options={"maxiter": 2000})
        if np.isfinite(res.fun) and (best is None or res.fun < best["stage1_chi2"]):
            active_params = (np.asarray(signs, dtype=float) * (10.0 ** np.asarray(res.x, dtype=float))).ravel()
            best = {"stage1_chi2": float(res.fun), "active_params": active_params, "signs": signs}
    return best


def _safe_filename(fcn_string: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in fcn_string).strip("_")[:80]


def _format_params(active_params: np.ndarray, nshape: int, use_physical_scale: bool) -> str:
    """Label each entry of active_params (shape a0..a{nshape-1}, then rho0/rs
    if present) instead of printing a bare array -- otherwise there's no way
    to tell which number is which without re-deriving nshape by hand."""
    labels = [f"a{i}" for i in range(nshape)]
    if use_physical_scale:
        labels += ["rho0", "rs"]
    return ", ".join(f"{lbl}={val:.6g}" for lbl, val in zip(labels, active_params))


def _pad_for_fisher(shape_compact: np.ndarray, nshape: int, n_extra: int,
                     max_fun_param: int = MAX_PARAM_SHAPE) -> np.ndarray:
    """match.compute_function_codelen expects theta_ML in the PADDED layout
    ([shape (nshape) | zero-padding to max_fun_param | rho0, rs]), not the
    compact [shape..., rho0, rs] layout everything in this script otherwise
    uses (galaxy_params_polish's own return format) -- same padded/compact
    split test_all.py's own full_params vs. get_pred input already has.
    optimise_fun_direct_nm's raw pipe_full_params is ALREADY in this padded
    layout (don't re-pad that one); only the compact shape_fit_* vectors
    from galaxy_params_polish (the NM/BFGS arms) need this."""
    padded = np.zeros(max_fun_param + n_extra)
    padded[:nshape] = shape_compact[:nshape]
    if n_extra > 0:
        padded[max_fun_param:max_fun_param + n_extra] = shape_compact[nshape:nshape + n_extra]
    return padded



def _plot_integral_profile(fcn_string, arm_label, param_label, p, j, Sigma_j,
                            fop, negloglike_here, boundary_left=None, boundary_right=None,
                            out_path=None, show=True):
    """
    Plots the SAME 1D profile likelihood match.get_sigma_from_integral
    integrates over for parameter j -- exp(-fop(theta) + negloglike_here),
    varying only theta[j] -- with the ML point, the +-Sigma 68% boundary,
    AND the log(1e3)-envelope boundaries find_boundary actually searched
    for (boundary_left/boundary_right, passed through from
    match.get_sigma_from_integral's boundaries_out) all marked. The 68%
    boundary is the narrower NARROWER width get_integral's quantile search
    found inside the envelope; the envelope boundaries are the wider outer
    limits get_integral integrated within (and, when one side fails, the
    only place to see whether that side never even had a valid outer limit
    to work with, vs. found one but failed some later step).

    If Sigma_j is finite, the grid spans p[j] +- 4*Sigma_j (padded past the
    boundary so its shape is visible), widened further if needed so any
    given boundary_left/boundary_right stay inside the plotted span. If
    Sigma_j is inf (the integral failed to find a boundary), spans p[j] +-
    max(3*|p[j]|, 1.0) instead (also widened for any given envelope
    boundary), so a failure still produces a plot showing why.
    """
    theta0 = float(p[j])
    boundary_found = np.isfinite(Sigma_j) and Sigma_j > 0
    half_span = 4.0 * Sigma_j if boundary_found else max(3.0 * abs(theta0), 1.0)
    for envelope_boundary in (boundary_left, boundary_right):
        if envelope_boundary is not None and np.isfinite(envelope_boundary):
            half_span = max(half_span, 1.2 * envelope_boundary)

    grid = np.linspace(theta0 - half_span, theta0 + half_span, 400)
    like = np.full(grid.shape, np.nan)
    for k, val in enumerate(grid):
        p_trial = p.copy()
        p_trial[j] = val
        nll = float(fop(p_trial))
        like[k] = np.exp(-nll + negloglike_here) if np.isfinite(nll) else np.nan

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, like, lw=1.5, color="k")
    ax.axvline(theta0, color="tab:blue", ls="--", lw=1.2, label=f"ML point ({theta0:.6g})")
    title = f"{fcn_string}\n{arm_label} -- {param_label}"
    if boundary_left is not None and np.isfinite(boundary_left):
        ax.axvline(theta0 - boundary_left, color="tab:green", ls="-.", lw=1.2,
                    label=f"log(1e3) envelope (left={boundary_left:.6g})")
    if boundary_right is not None and np.isfinite(boundary_right):
        ax.axvline(theta0 + boundary_right, color="tab:green", ls="-.", lw=1.2,
                    label=f"log(1e3) envelope (right={boundary_right:.6g})")
    if boundary_found:
        ax.axvline(theta0 - Sigma_j, color="tab:red", ls=":", lw=1.2,
                    label=f"68% boundary (Sigma={Sigma_j:.6g})")
        ax.axvline(theta0 + Sigma_j, color="tab:red", ls=":", lw=1.2)
    else:
        title += f"\n(integral did not find a boundary -- Sigma={Sigma_j:.6g})"
    ax.set_xlabel(param_label)
    ax.set_ylabel(r"$\exp(-\Delta\,\mathrm{NLL})$  (relative likelihood)")
    ax.set_title(title, fontsize=9, color=("black" if boundary_found else "tab:red"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"        {param_label}: profile-likelihood plot saved to {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _compute_production_codelen(fcn_string, eq_numpy, integrated, theta_ML_padded, likelihood,
                                  nshape, n_extra, inc_ML, d_ML,
                                  plot=False, arm_label="", verbose=True):
    """
    THE production codelen calculation for one already-fitted point --
    match.compute_function_codelen, the SAME function match.py::main() uses
    to write codelen_matches_comp{N}.dat, which combine_DL.py reads to
    build results_pretty_{N}.txt. Also prints every step along the way:
    which parameters the Fisher diagonal resolves on its own, which fall
    through to the profile-likelihood integral fallback (get_sigma_from_
    integral), and what 68%-credible boundary that integral finds for each.

    Used to be two separate functions here -- this one (match.py's own
    Fisher+integral-fallback+snap-exploration formula) and a _compute_codelen
    wrapping test_all_Fisher.convert_params directly, which has NO integral
    fallback (gives up with codelen=nan the instant the Fisher diagonal
    can't resolve a parameter on its own) and is NOT what generates
    results_pretty_{N}.txt despite its own former docstring claiming
    otherwise. Merged into this single function so there is one codelen
    number, computed one way, everywhere in this script.

    Inc and D are folded in as ordinary active parameters (like rho0/rs),
    not priced via a separate prior-aware term -- see
    match.compute_function_codelen's docstring for why. negloglike is
    exactly fop() at the returned point (include_priors=True, evaluated
    with Inc/D as free elements, not fixed) -- nothing is added to or
    subtracted from it.

    If plot (default False), pops up a 1D profile-likelihood plot (see
    _plot_integral_profile) immediately for every parameter that went
    through the integral fallback, marking the ML point and the boundary
    the integral found -- not saved to disk, just shown right away.

    Returns (negloglike, codelen, DL). DL = negloglike + codelen is the
    MODEL codelen contribution only -- it does NOT include aifeyn (the
    structural complexity prior loaded per-complexity from the function
    library), so it is not directly comparable to a real ranking DL from
    combine_DL.py (which is negloglike + codelen + aifeyn).
    """
    max_fun_param = MAX_PARAM_SHAPE
    max_param_total = max_fun_param + n_extra
    shape_idx, pad_idx, extra_idx, active_idx = match.build_layout(nshape, max_fun_param, n_extra)

    # Inc and D are folded in as two more "extra" (never-snap-eligible)
    # active parameters, exactly like rho0/rs -- see
    # match.compute_function_codelen's docstring. That means the Hessian
    # needs Inc/D as ordinary free elements of the differentiated vector
    # (include_priors=True, NOT fixed_galaxy_params), not held fixed the
    # way match.py::main() reads them from its derivs_comp{N}.dat cache
    # (which never had Inc/D curvature at all).
    num_galaxy_params = getattr(likelihood, "num_galaxy_params", 0)
    galaxy_idx = np.arange(max_param_total, max_param_total + num_galaxy_params)
    extra_idx_ext = np.concatenate([extra_idx, galaxy_idx]).astype(int)
    active_idx_ext = np.concatenate([shape_idx, extra_idx_ext]).astype(int)

    p = np.concatenate([np.asarray(theta_ML_padded, dtype=float)[:max_param_total], [inc_ML, d_ML]])

    hessian_template = likelihood.get_loss(eq_numpy, integrated, value="hessian", include_priors=True)
    Hmat_active = hessian_template(
        jnp.asarray(p[active_idx_ext], dtype=float),
        likelihood.xvar, likelihood.yvar, likelihood.yerr_lo, likelihood.yerr_hi,
    )
    fish_mat = np.zeros((len(p), len(p)), dtype=float)
    fish_mat[np.ix_(active_idx_ext, active_idx_ext)] = np.asarray(Hmat_active, dtype=float)
    diag_fish = np.diag(fish_mat)
    if verbose:
        print(f"      match.py Fisher diagonal: {diag_fish}")

    loss_template = likelihood.get_loss(eq_numpy, integrated, value="evaluate", include_priors=True)
    chi2_fcn = likelihood.get_wrapped_like(loss_template)

    def fop(theta):
        theta = jnp.asarray(theta, dtype=float)
        return chi2_fcn(theta[active_idx_ext], likelihood.xvar, likelihood.yvar,
                         likelihood.yerr_lo, likelihood.yerr_hi)

    negloglike_here = float(fop(p))

    def _label(j):
        if j < nshape:
            return f"a{j}"
        if len(extra_idx) > 0 and j == extra_idx[0]:
            return "rho0"
        if len(extra_idx) > 1 and j == extra_idx[1]:
            return "rs"
        if num_galaxy_params > 0 and j == galaxy_idx[0]:
            return "Inc"
        if num_galaxy_params > 1 and j == galaxy_idx[1]:
            return "D"
        return f"p{j}"

    if verbose:
        print(f"      match.py Fisher diagonal (per active parameter):")
        Sigma = np.full(len(p), np.inf, dtype=float)
        for j in active_idx_ext:
            label = _label(j)
            if j < len(diag_fish) and np.isfinite(diag_fish[j]) and diag_fish[j] > 0:
                Sigma[j] = 1.0 / np.sqrt(diag_fish[j])
                print(f"        {label}: Fisher_diag={diag_fish[j]:.6g}  "
                      f"-> Sigma(Fisher)={Sigma[j]:.6g}  RESOLVED by Fisher")
            else:
                print(f"        {label}: Fisher_diag={diag_fish[j]:.6g}  "
                      f"-> NOT resolved by Fisher, needs integral fallback")

        unresolved = active_idx_ext[(Sigma[active_idx_ext] <= 0) | np.isinf(Sigma[active_idx_ext]) | np.isnan(Sigma[active_idx_ext])]
        if len(unresolved) > 0:
            print(f"      Falling back to profile-likelihood integral for: "
                  f"{[_label(j) for j in unresolved]}")
            boundaries = {}
            Sigma = match.get_sigma_from_integral(
                p, Sigma, fop, negloglike_here, active_idx=active_idx_ext,
                fcn_i=fcn_string, number_points=10**3, boundaries_out=boundaries,
            )
            Sigma[np.isnan(Sigma)] = np.inf
            for j in unresolved:
                label = _label(j)
                boundary_left, boundary_right = boundaries.get(int(j), (None, None))
                if np.isfinite(Sigma[j]):
                    print(f"        {label}: integral found Sigma={Sigma[j]:.6g} "
                          f"(68% boundary at theta={p[j]:.6g} +/- {Sigma[j]:.6g}); "
                          f"log(1e3) envelope at -{boundary_left:.6g}/+{boundary_right:.6g}")
                else:
                    print(f"        {label}: integral FAILED to find a boundary -- Sigma=inf "
                          f"(log(1e3) envelope: left={boundary_left}, right={boundary_right})")
                if plot:
                    _plot_integral_profile(fcn_string, arm_label, label, p, j, Sigma[j],
                                            fop, negloglike_here, boundary_left=boundary_left,
                                            boundary_right=boundary_right, out_path=None, show=True)

    # The per-parameter Fisher/integral-fallback diagnostics above (when
    # verbose) are their own useful pass but are NOT what the final codelen
    # below is computed from -- that comes from match.compute_function_codelen,
    # the SAME function main() calls to write codelen_matches_comp{N}.dat
    # (and therefore combine_DL.py's results_pretty_{N}.txt): Delta capped
    # at |theta| for any unresolved/oversized entry, then a snap-vs-no-snap
    # exploration over shape parameters. negloglike_final is exactly fop()
    # at the returned point -- no term added or removed.
    codelen, negloglike_final, p_final, Delta_capped = match.compute_function_codelen(
        fcn_string, p, fish_mat,
        shape_idx, extra_idx_ext, active_idx_ext, pad_idx,
        fop, negloglike_here, number_points=10**3,
    )
    DL = negloglike_final + codelen if np.isfinite(codelen) else np.nan
    if verbose:
        print(f"      match.py codelen (production formula: Delta-capped + "
              f"snap-exploration, Inc/D as ordinary parameters) = {codelen:.6f}")
    return float(negloglike_final), float(codelen), float(DL)


def validate_fit(
    fcn_string: str,
    galaxy_name: str,
    data_file: str = str(DEFAULT_DATA_FILE),
    use_physical_scale: bool = True,
    pmin: float = -8.0,
    pmax: float = 8.0,
    n_random: int = 50,
    seed: int = 0,
    out_dir: Optional[str] = None,
    match_tol: float = 1e-3,
    direct_maxfun: int = 3000,
    direct_maxiter: int = 3000,
    landscape_grid_n: int = 40,
    compute_codelen: bool = True,
    check_match: bool = False,
    plot_integral_profiles: bool = False,
    codelen_pipeline_only: bool = True,
) -> dict:
    """
    Run the production DIRECT+NM pipeline and two random-restart baselines
    (Nelder-Mead, BFGS) on the same function/galaxy, report whether they
    agree, and plot the resulting fits. If compute_codelen (default True),
    also runs THE production codelen calculation on each arm's winning
    point -- match.compute_function_codelen, the same function
    match.py::main() uses to write codelen_matches_comp{N}.dat (and
    therefore combine_DL.py's results_pretty_{N}.txt): Fisher -> Sigma ->
    Delta (with a profile-likelihood integral fallback for any parameter
    the Fisher diagonal alone can't resolve), Delta capped at |theta| for
    any unresolved/oversized entry, a snap-vs-no-snap exploration over
    shape parameters, plus the separate Inc/D galaxy_params_codelen term --
    so the codelen/DL reported here is a faithful reproduction of the real
    pipeline output, not an approximation of it. If check_match (default
    False -- slower, opt in), ALSO prints the per-parameter diagnostic
    trace (which parameter the Fisher diagonal resolved on its own, which
    fell through to the integral fallback, and what boundary that fallback
    found). If plot_integral_profiles is ALSO True (ignored unless
    check_match is True), pops up a 1D profile-likelihood plot immediately
    (not saved) per integral-fallback parameter, marking the ML point and
    the 68% boundary the integral converged to. codelen_pipeline_only
    (default True) skips the codelen check for the two random-restart arms
    -- you almost always only care about the Fisher/codelen of the actual
    production (DIRECT) optimum, not the brute-force baselines; set False
    to check all three arms.

    Args:
        fcn_string: ESR-style function string, e.g. "a0/(a1+x)".
        galaxy_name: Galaxy name as it appears in the "Galaxy" column of
            data_file (this likelihood fits one galaxy's rotation curve at a
            time; there is no cluster-level object in this pipeline).
        match_tol: absolute chi2 tolerance below which the pipeline's stage-1
            optimum is considered to match the best random-restart optimum.
        direct_maxfun, direct_maxiter: DIRECT's evaluation budget, passed
            straight through to optimise_fun_direct_nm (production default
            3000/3000). Raise these to check whether a gap is a budget
            problem (more evaluations would close it) or a genuine blind
            spot (DIRECT plateaus regardless of budget) -- see the
            the development notes, where 3000->100000 made no
            difference for a0*pow(x,a1)/(a2+x).
        landscape_grid_n: resolution (per axis) of the likelihood-landscape
            plot's NLL grids.

    Returns:
        dict with per-arm results and the output plot paths.
    """
    likelihood = MIGHTEELikelihood(
        data_file=data_file, name=galaxy_name, run_name="validate_fit",
        use_physical_scale=use_physical_scale,
    )

    fcn_string, eq, integrated = likelihood.run_sympify(fcn_string, try_integration=False)
    nshape = simplifier.count_params([fcn_string], MAX_PARAM_SHAPE)[0]
    n_extra = 2 if use_physical_scale else 0
    nparam = nshape + n_extra
    eq_numpy = lambdify_equation(eq, nshape)

    r = np.asarray(likelihood.xvar)
    v = np.asarray(likelihood.yvar)
    print(f"Galaxy {galaxy_name}: {len(r)} points, R={r.min():.3f}-{r.max():.3f} kpc, "
          f"V={v.min():.3f}-{v.max():.3f} km/s, "
          f"Inc_true={likelihood.inc_true:.3f}+/-{likelihood.e_inc:.3f}, "
          f"D_true={likelihood.distance_true:.3f}+/-{likelihood.e_d:.3f}")
    print(f"Function {fcn_string}: nshape={nshape}, use_physical_scale={use_physical_scale}, "
          f"nparam={nparam}, DIRECT box=[{pmin},{pmax}]")

    # Computed early (not just before the fit-vs-data plot at the end) since
    # _compute_match_codelen's per-arm integral-profile plots need it too.
    out_path = Path(out_dir) if out_dir else THIS_DIR / "output" / "validate_fit" / galaxy_name
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- Arm A: the production pipeline ----
    print("\n[1/3] Running production pipeline (DIRECT+NM)...")
    diag = {}
    t0 = time.time()
    (_pipe_chi2_joint, pipe_full_params, _j_out, _count_lowest, _success,
     pipe_inc, pipe_d, pipe_stage2_chi2, pipe_flag_reason) = optimise_fun_direct_nm(
        fcn_string, likelihood, tmax=90, pmin=pmin, pmax=pmax, log_opt=True,
        direct_maxfun=direct_maxfun, direct_maxiter=direct_maxiter,
        diagnostics_out=diag,
    )
    pipe_wall = time.time() - t0
    # optimise_fun_direct_nm's own returned chi2_i (_pipe_chi2_joint, above) is NOT
    # a stage-1-only value -- by test_all.py's own design (see its "chi2_i
    # is the stage-2 ... likelihood" comment), chi2_i gets overwritten with
    # stage2_chi2 once stage 2 runs, so it's identical to pipe_stage2_chi2
    # here and would make the pipeline row's "stage1_chi2" column secretly
    # report the joint (Inc/D-optimized) loss -- not comparable to the NM/
    # BFGS arms' stage1_chi2 below, which genuinely IS shape-only (Inc/D
    # fixed at catalog). diagnostics_out is what actually carries the
    # genuine stage-1-only loss (best_fun, before that overwrite).
    pipe_stage1_chi2 = diag.get("stage1_chi2", np.nan)
    # optimise_fun_direct_nm returns the padded save layout
    # [shape | pad-to-4 | rho0, rs]; get_pred needs the compact
    # [shape..., rho0, rs] vector.
    pipe_active = np.concatenate([
        pipe_full_params[:nshape],
        pipe_full_params[MAX_PARAM_SHAPE:MAX_PARAM_SHAPE + n_extra],
    ])
    arms = {
        "pipeline (DIRECT+NM)": {
            "stage1_chi2": pipe_stage1_chi2, "active_params": pipe_active,
            "inc_fit": pipe_inc, "d_fit": pipe_d, "stage2_chi2": pipe_stage2_chi2,
            "wall_time": pipe_wall,
        }
    }
    print(f"      done in {pipe_wall:.2f}s: stage1_chi2={pipe_stage1_chi2:.6f}, "
          f"stage2_chi2={pipe_stage2_chi2:.6f}, Inc_fit={pipe_inc:.3f}, D_fit={pipe_d:.3f}")
    print(f"      {_format_params(pipe_active, nshape, use_physical_scale)}")
    if pipe_flag_reason:
        print(f"      PLAUSIBILITY FLAG: {pipe_flag_reason}")
    if compute_codelen:
        # pipe_full_params is ALREADY in the padded layout
        # compute_function_codelen expects (test_all.py's own save format)
        # -- don't re-pad it.
        nll_c, codelen, DL = _compute_production_codelen(
            fcn_string, eq_numpy, integrated, pipe_full_params, likelihood,
            nshape, n_extra, pipe_inc, pipe_d,
            plot=plot_integral_profiles, arm_label="pipeline", verbose=check_match,
        )
        arms["pipeline (DIRECT+NM)"].update(
            {"codelen": codelen, "negloglike_codelen": nll_c, "DL": DL})
        print(f"      codelen={codelen:.6f}  negloglike(codelen-consistent)={nll_c:.6f}  DL={DL:.6f}")

    # ---- Arms B & C: shared random starts, same internal space ----
    rng = np.random.default_rng(seed)
    starts = _generate_random_starts(rng, nshape, nparam, use_physical_scale, pmin, pmax, n_random)
    objective = _stage1_objective(likelihood, eq_numpy, integrated)
    obj_grad = _stage1_value_and_grad(likelihood, eq_numpy, integrated)

    print(f"\n[2/3] Running {n_random}x random-restart Nelder-Mead...")
    t0 = time.time()
    nm_best = _run_nm_arm(starts, objective)
    nm_wall = time.time() - t0
    if nm_best is not None:
        # shape_fit (not the stale pre-polish nm_best["active_params"]) is
        # the self-consistent post-polish shape -- see galaxy_params_polish's
        # own docstring on why pairing new Inc/D with old shape is wrong.
        shape_fit_nm, inc_nm, d_nm, stage2_nm = galaxy_params_polish(likelihood, eq_numpy, integrated, nm_best["active_params"])
        arms[f"{n_random}x random Nelder-Mead"] = {
            "stage1_chi2": nm_best["stage1_chi2"], "active_params": shape_fit_nm,
            "inc_fit": inc_nm, "d_fit": d_nm, "stage2_chi2": stage2_nm, "wall_time": nm_wall,
        }
        print(f"      done in {nm_wall:.2f}s: best stage1_chi2={nm_best['stage1_chi2']:.6f}")
        print(f"      {_format_params(shape_fit_nm, nshape, use_physical_scale)}")
        if compute_codelen and not codelen_pipeline_only:
            padded_nm = _pad_for_fisher(shape_fit_nm, nshape, n_extra)
            nll_c, codelen, DL = _compute_production_codelen(
                fcn_string, eq_numpy, integrated, padded_nm, likelihood,
                nshape, n_extra, inc_nm, d_nm,
                plot=plot_integral_profiles, arm_label="random_NM", verbose=check_match,
            )
            arms[f"{n_random}x random Nelder-Mead"].update(
                {"codelen": codelen, "negloglike_codelen": nll_c, "DL": DL})
            print(f"      codelen={codelen:.6f}  negloglike(codelen-consistent)={nll_c:.6f}  DL={DL:.6f}")
    else:
        print(f"      done in {nm_wall:.2f}s: every restart failed (no finite chi2 found)")

    print(f"\n[3/3] Running {n_random}x random-restart BFGS (same starting points as NM)...")
    t0 = time.time()
    bfgs_best = _run_bfgs_arm(starts, obj_grad)
    bfgs_wall = time.time() - t0
    if bfgs_best is not None:
        shape_fit_bfgs, inc_bfgs, d_bfgs, stage2_bfgs = galaxy_params_polish(likelihood, eq_numpy, integrated, bfgs_best["active_params"])
        arms[f"{n_random}x random BFGS"] = {
            "stage1_chi2": bfgs_best["stage1_chi2"], "active_params": shape_fit_bfgs,
            "inc_fit": inc_bfgs, "d_fit": d_bfgs, "stage2_chi2": stage2_bfgs, "wall_time": bfgs_wall,
        }
        print(f"      done in {bfgs_wall:.2f}s: best stage1_chi2={bfgs_best['stage1_chi2']:.6f}")
        print(f"      {_format_params(shape_fit_bfgs, nshape, use_physical_scale)}")
        if compute_codelen and not codelen_pipeline_only:
            padded_bfgs = _pad_for_fisher(shape_fit_bfgs, nshape, n_extra)
            nll_c, codelen, DL = _compute_production_codelen(
                fcn_string, eq_numpy, integrated, padded_bfgs, likelihood,
                nshape, n_extra, inc_bfgs, d_bfgs,
                plot=plot_integral_profiles, arm_label="random_BFGS", verbose=check_match,
            )
            arms[f"{n_random}x random BFGS"].update(
                {"codelen": codelen, "negloglike_codelen": nll_c, "DL": DL})
            print(f"      codelen={codelen:.6f}  negloglike(codelen-consistent)={nll_c:.6f}  DL={DL:.6f}")
    else:
        print(f"      done in {bfgs_wall:.2f}s: every restart failed (no finite chi2 found)")

    # ---- summary table (all arms side by side) ----
    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    if compute_codelen:
        print(f"{'arm':>28} {'stage1_chi2':>14} {'stage2_chi2':>14} {'Inc_fit':>9} {'D_fit':>10} "
              f"{'wall_s':>8} {'nll_codelen':>12} {'codelen':>12} {'DL':>12}")
    else:
        print(f"{'arm':>28} {'stage1_chi2':>14} {'stage2_chi2':>14} {'Inc_fit':>9} {'D_fit':>10} {'wall_s':>8}")
    for name, res in arms.items():
        if compute_codelen:
            codelen_str = f"{res['codelen']:.4f}" if "codelen" in res else "n/a"
            DL_str = f"{res['DL']:.4f}" if "DL" in res else "n/a"
            nll_c_str = f"{res['negloglike_codelen']:.4f}" if "negloglike_codelen" in res else "n/a"
            print(f"{name:>28} {res['stage1_chi2']:>14.6f} {res['stage2_chi2']:>14.6f} "
                  f"{res['inc_fit']:>9.3f} {res['d_fit']:>10.3f} {res['wall_time']:>8.2f} "
                  f"{nll_c_str:>12} {codelen_str:>12} {DL_str:>12}")
            # nll_codelen (negloglike_codelen) can differ from stage2_chi2:
            # compute_function_codelen's snap-vs-no-snap exploration may
            # find that setting a shape parameter to exactly 0 gives a
            # LOWER total DL than the optimizer's own converged point --
            # when it does, codelen/DL (and this column) are evaluated at
            # that snapped point, not at stage2_chi2's point. DL is always
            # nll_codelen + codelen, NOT stage2_chi2 + codelen -- flag it
            # explicitly whenever the two points disagree by more than
            # floating-point noise, so this is never silently confusing.
            if "negloglike_codelen" in res and np.isfinite(res["negloglike_codelen"]):
                gap = abs(res["negloglike_codelen"] - res["stage2_chi2"])
                if gap > 1e-3:
                    print(f"{'':>28} NOTE: nll_codelen != stage2_chi2 (gap={gap:.4f}). "
                          f"nll_codelen is fop() with Inc/D as FREE elements (no "
                          f"fixed_galaxy_params); stage2_chi2 is galaxy_params_polish's own "
                          f"fixed-Inc/D evaluation -- these can differ by tiny floating-point "
                          f"amounts from that alone. A LARGE gap instead means the snap-vs-"
                          f"no-snap exploration inside compute_function_codelen found a "
                          f"genuinely different, lower-DL point (typically a shape parameter "
                          f"snapped to exactly 0) than the optimizer's own converged point -- "
                          f"check p_final vs active_params above to see which parameter moved. "
                          f"DL is always nll_codelen + codelen, never stage2_chi2 + codelen.")
        else:
            print(f"{name:>28} {res['stage1_chi2']:>14.6f} {res['stage2_chi2']:>14.6f} "
                  f"{res['inc_fit']:>9.3f} {res['d_fit']:>10.3f} {res['wall_time']:>8.2f}")
        print(f"{'':>28} {_format_params(res['active_params'], nshape, use_physical_scale)}")

    if compute_codelen:
        dl_by_arm = {name: res["DL"] for name, res in arms.items() if "DL" in res and np.isfinite(res["DL"])}
        if dl_by_arm:
            best_dl_name = min(dl_by_arm, key=dl_by_arm.get)
            print(f"\nLowest DL: '{best_dl_name}' (DL={dl_by_arm[best_dl_name]:.4f}) -- "
                  "note this is negloglike+codelen only, NOT the full ranking DL "
                  "(no aifeyn structural-complexity term included).")
        else:
            print("\nAll arms' codelen came back nan/inf -- see the per-parameter Fisher/"
                  "integral-fallback prints above (pass check_match=True for the full trace) "
                  "for why.")

    best_random_name = min(
        (name for name in arms if name != "pipeline (DIRECT+NM)"),
        key=lambda name: arms[name]["stage1_chi2"],
        default=None,
    )
    best_random_chi2 = arms[best_random_name]["stage1_chi2"] if best_random_name else np.inf
    pipe_chi2_val = arms["pipeline (DIRECT+NM)"]["stage1_chi2"]
    gap = pipe_chi2_val - best_random_chi2
    print("-" * 78)
    if gap > match_tol:
        print(f">>> PIPELINE BEATEN by '{best_random_name}': "
              f"{pipe_chi2_val:.6f} vs {best_random_chi2:.6f} (gap={gap:.6f}, tol={match_tol})")
    else:
        print(f">>> MATCH: pipeline={pipe_chi2_val:.6f}, best random={best_random_chi2:.6f} "
              f"(gap={gap:.6f} <= tol={match_tol})")

    # ---- plot ----
    plot_file = out_path / f"{_safe_filename(fcn_string)}.png"
    _plot_arms(likelihood, eq_numpy, integrated, fcn_string, arms, plot_file)
    print(f"\nPlot saved to {plot_file}")

    landscape_file = out_path / f"{_safe_filename(fcn_string)}_landscape.png"
    _plot_likelihood_landscape(
        likelihood, eq_numpy, integrated, fcn_string, diag, arms,
        grid_n=landscape_grid_n, outfile=landscape_file,
    )

    return {"arms": arms, "plot_file": str(plot_file), "landscape_file": str(landscape_file)}


def _plot_arms(likelihood, eq_numpy, integrated, fcn_string, arms, plot_file: Path) -> None:
    r = np.asarray(likelihood.xvar)
    order = np.argsort(r)
    vobs = np.asarray(likelihood.yvar)
    err_lo = np.asarray(likelihood.yerr_lo)
    err_hi = np.asarray(likelihood.yerr_hi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.errorbar(r, vobs, yerr=[err_lo, err_hi], fmt="o", capsize=3, color="black",
                markersize=4, label="Data", zorder=1)
    for name, res in arms.items():
        v_model = np.asarray(likelihood.get_pred(
            likelihood.xvar, res["active_params"], eq_numpy, integrated=integrated,
            D=res["d_fit"], Inc=res["inc_fit"],
        ))
        ax.plot(r[order], v_model[order], lw=2,
                label=f"{name} (chi2={res['stage2_chi2']:.1f})")
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Velocity (km/s)")
    ax.set_title(fcn_string)
    ax.legend(fontsize=8)

    ax = axes[1]
    for name, res in arms.items():
        shape_params = res["active_params"][:len(res["active_params"]) - (2 if likelihood.use_physical_scale else 0)]
        if likelihood.use_physical_scale:
            rho0, rs = res["active_params"][-2], res["active_params"][-1]
            u = r[order] / rs
            f_u = eq_numpy(u, *shape_params) if len(shape_params) > 0 else eq_numpy(u)
            rho_model = rho0 * np.asarray(f_u)
        else:
            rho_model = np.asarray(eq_numpy(r[order], *shape_params) if len(shape_params) > 0 else eq_numpy(r[order]))
        # A shape function that doesn't symbolically depend on x at all
        # (e.g. a snapped/degenerate candidate like "0", "1", or oo**a0
        # with a0 the only free symbol) lambdifies to something that
        # ignores its array input entirely -- sympy's lambdify has nothing
        # to broadcast against when x never appears in the expression, so
        # eq_numpy(u, ...) returns a bare scalar regardless of u's shape.
        # Broadcast it out to r[order]'s shape before plotting; a real,
        # x-dependent shape function is already the right shape and this
        # is a no-op for it.
        rho_model = np.broadcast_to(rho_model, r[order].shape)
        ax.plot(r[order], rho_model, lw=2, label=name)
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Density (model units)")
    ax.set_yscale("log")
    ax.set_title("Density profile")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _trace_polish_trajectory(likelihood, eq_numpy, integrated, signs, p0, maxiter=5000):
    """
    Re-run the polish (NM and, for comparison, BFGS) from a specific
    internal-coordinate starting point, recording every iterate via a
    callback -- not just start and end.

    Motivation: a straight line between a polish's start and end point can
    look like it crosses uniformly bad territory even when the optimum is
    completely genuine (confirmed on a0/(a1+x)+(1-a0)/(a2+x): nll stays in
    the tens of thousands along >90% of the straight-line interpolation,
    yet Nelder-Mead reaches the true optimum in a few thousand ordinary,
    monotonically-improving steps). The actual path curves through
    parameter space -- here, along a1->a2 while a0 grows -- in a way no
    straight line or fixed-third-coordinate 2D slice can represent. Tracing
    the real path is the only way to show what the optimizer actually did.

    Returns (path_array, method_name) for whichever of NM/BFGS reaches the
    lower final value from this start (matching optimise_fun_direct_nm's
    own tie-break), where path_array is (n_iter, nparam) in internal coords.
    """
    xvar, yvar = likelihood.xvar, likelihood.yvar
    yerr_lo, yerr_hi = likelihood.yerr_lo, likelihood.yerr_hi
    loss_eval = likelihood.get_loss(eq_numpy, integrated, value="evaluate", include_priors=False)
    wrapped_eval = likelihood.get_wrapped_like(loss_eval)

    def obj(p):
        val = float(wrapped_eval(np.asarray(p, dtype=float), xvar, yvar, yerr_lo, yerr_hi,
                                  signs=list(signs), check_nans=True))
        return val if np.isfinite(val) else 1e30

    p0 = np.asarray(p0, dtype=float)
    candidates = []

    path_nm = [p0.copy()]
    res_nm = scipy_minimize(obj, p0, method="Nelder-Mead", options={"maxiter": maxiter},
                             callback=lambda xk: path_nm.append(np.array(xk, dtype=float)))
    candidates.append((res_nm.fun, np.asarray(path_nm), "Nelder-Mead"))

    path_bfgs = [p0.copy()]
    res_bfgs = scipy_minimize(obj, p0, method="BFGS",
                               callback=lambda xk: path_bfgs.append(np.array(xk, dtype=float)))
    candidates.append((res_bfgs.fun, np.asarray(path_bfgs), "BFGS"))

    candidates = [c for c in candidates if np.isfinite(c[0])]
    if not candidates:
        return None, None
    _, path, method = min(candidates, key=lambda c: c[0])
    return path, method


def _plot_likelihood_landscape(
    likelihood, eq_numpy, integrated, fcn_string, diag: dict, arms: dict,
    grid_n: int = 40, delta_clip: float = 60.0, zoom_span: float = 1.5,
    outfile: Optional[Path] = None,
) -> None:
    """
    Pairwise 2D NLL grids (stage-1 objective, internal sign*10**x
    coordinates) for the pipeline's own winning sign combo -- shows DIRECT's
    actual sampled points (from diag["global_samples"]) alongside each arm's
    converged point, so a gap between arms is visible as "the good point
    sits somewhere DIRECT's own search never sampled densely," not just a
    number in a table. Directly answers "why doesn't DIRECT find it?": if
    the true optimum is off in a region with few/no DIRECT dots nearby,
    that's the answer -- DIRECT's box-partitioning search simply never
    prioritised that region within its evaluation budget.

    Only arms whose converged point shares the SAME sign combo as the
    reference are plotted on these axes -- a different sign combo is a
    genuinely different corner of parameter space (not reachable by moving
    continuously within this grid), so it's skipped with a printed note
    rather than silently mis-plotted.

    Two columns per parameter pair: a "wide" panel sized to cover both the
    marked arms AND DIRECT's own sampled cloud (not just the marked arms --
    when only one arm survives the sign-combo filter, as happens whenever
    the pipeline's own answer is the outlier being diagnosed, the old
    marked-only span collapsed to its 0.5-internal-unit floor, leaving a
    background grid too small to be visible next to DIRECT's box-bounded
    samples: mostly white/unplotted, with one small saturated patch -- that
    was the "all white and yellow" bug, not a real feature of the
    landscape), and a "zoom" panel at a small fixed span around the best
    marked point, showing local curvature the wide view compresses away.
    """
    signs = diag.get("best_signs")
    if signs is None:
        print("No sign combo recorded (diagnostics unavailable) -- skipping likelihood-landscape plot.")
        return
    signs_arr = np.asarray(signs, dtype=float)
    nparam = len(signs_arr)

    xvar, yvar = likelihood.xvar, likelihood.yvar
    yerr_lo, yerr_hi = likelihood.yerr_lo, likelihood.yerr_hi
    loss_eval = likelihood.get_loss(eq_numpy, integrated, value="evaluate", include_priors=False)
    wrapped_eval = likelihood.get_wrapped_like(loss_eval)

    def nll_from_internal(p_internal: np.ndarray) -> float:
        val = float(wrapped_eval(
            np.asarray(p_internal, dtype=float), xvar, yvar, yerr_lo, yerr_hi,
            signs=list(signs), check_nans=True,
        ))
        return val if np.isfinite(val) else np.nan

    def to_internal(active_params: np.ndarray) -> Optional[np.ndarray]:
        ap = np.asarray(active_params, dtype=float)
        if not np.all(np.sign(ap) == signs_arr):
            return None
        return np.log10(ap * signs_arr)

    marked = {}
    for name, res in arms.items():
        p_int = to_internal(res["active_params"])
        if p_int is None:
            print(f"  (landscape plot: '{name}' uses a different sign combo, skipping)")
        else:
            marked[name] = p_int

    if not marked:
        print("No arm shares the plotted sign combo -- skipping likelihood-landscape plot.")
        return

    # Hold the "other" coordinates fixed at the BEST (lowest stage1_chi2)
    # marked point's own values in every panel -- that point's neighbourhood
    # is what each 2D slice actually shows; other arms are overlaid as
    # (i,j)-projected markers for comparison, not full re-slices.
    best_name = min(marked, key=lambda n: arms[n]["stage1_chi2"])
    p_center = marked[best_name].copy()

    direct_pts = np.array([
        s["p_internal"] for s in diag.get("global_samples", [])
        if s["signs"] is not None and tuple(int(v) for v in s["signs"]) == tuple(int(v) for v in signs)
    ])

    # Which single point the winning polish (NM or BFGS) actually started
    # from, and where it ended up -- the star can legitimately sit far from
    # every DIRECT sample (a local polish walks there through many
    # unplotted intermediate steps DIRECT never touched), which otherwise
    # looks unexplained/suspicious. Only meaningful for the pipeline arm
    # itself (this diag dict comes from its one optimise_fun_direct_nm
    # call); absent for the random-restart arms, and gracefully skipped if
    # an older diagnostics dict without it is passed in.
    best_start = diag.get("best_start")
    start_p_int = None
    pipeline_p_int = marked.get("pipeline (DIRECT+NM)")
    if best_start is not None and pipeline_p_int is not None:
        start_p_int = np.asarray(best_start["p_internal"], dtype=float)

    # Which point DIRECT's own box search itself ranked best for this sign
    # combo, before any polish -- shown separately from `best_start` because
    # they're often NOT the same point: `best_start` is whichever of the
    # ~20-30 hybrid starts (best-direct, global pool, per-combo, random) won
    # the polish loop's strict "<" comparison, which can be a `source="random"`
    # point that beat DIRECT's own best by a rounding-level margin (fine for
    # ranking, since only the final chi2 matters there) -- but that makes a
    # diagnostic plot showing only `best_start` look like DIRECT never found
    # anything useful, when its own top candidate may have reached
    # essentially the same answer. Marking both lets the reader see that
    # directly instead of taking it on faith.
    direct_best_p_int = None
    for s in diag.get("hybrid_starts", []):
        if s.get("source") == "best-direct":
            direct_best_p_int = np.asarray(s["p_internal"], dtype=float)
            break

    # The actual optimizer path from start_p_int to the star -- not a
    # straight line, the real iterate-by-iterate trajectory (see
    # _trace_polish_trajectory's docstring for why this matters: a straight
    # line between the same two endpoints can look like it crosses
    # uniformly bad territory even when the true path is a smooth,
    # monotonically-improving curve through parameter space).
    trajectory = None
    if start_p_int is not None:
        trajectory, traj_method = _trace_polish_trajectory(
            likelihood, eq_numpy, integrated, signs, start_p_int,
        )
        if trajectory is not None:
            print(f"  landscape plot: traced {trajectory.shape[0]} {traj_method} iterates "
                  f"from the winning polish start")

    pairs = [(i, j) for i in range(nparam) for j in range(i + 1, nparam)]
    all_pts = np.array(list(marked.values()))
    colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, len(marked)))

    def span_for(idx, floor=0.5, pad=1.8):
        # Span must cover DIRECT's own sampled cloud, not just the marked
        # arms -- with only one marked arm (common: it's the pipeline's own
        # outlier being diagnosed), all_pts alone gives zero spread and the
        # span collapses to `floor`, producing a background grid too small
        # to overlap DIRECT's box-bounded samples at all.
        vals = all_pts[:, idx]
        if direct_pts.size > 0:
            vals = np.concatenate([vals, direct_pts[:, idx]])
        return max(np.abs(vals - p_center[idx]).max(), floor) * pad

    def draw_panel(ax, i, j, span_i, span_j):
        xgrid = np.linspace(p_center[i] - span_i, p_center[i] + span_i, grid_n)
        ygrid = np.linspace(p_center[j] - span_j, p_center[j] + span_j, grid_n)

        Z = np.full((grid_n, grid_n), np.nan)
        p_trial = p_center.copy()
        for yy_i, yy in enumerate(ygrid):
            for xx_i, xx in enumerate(xgrid):
                p_trial[i], p_trial[j] = xx, yy
                Z[yy_i, xx_i] = nll_from_internal(p_trial)

        # p_center itself is essentially never exactly a grid node (it's a
        # continuous coordinate, the grid is a fixed-resolution lattice), so
        # its own true nll can silently be missing from Z entirely -- and if
        # the grid's own minimum happens to land in some unrelated, worse
        # region (this landscape has near-flat escape directions with huge
        # dynamic range), the colour scale ends up anchored to a point worse
        # than the marked optimum itself, making that optimum look bad by
        # comparison to a reference it was never actually compared against.
        # Fold the star's own real value in explicitly so this can't happen.
        center_nll = nll_from_internal(p_center)
        nll_best = np.nanmin(np.append(Z.ravel(), center_nll))
        dZ = np.minimum(Z - nll_best, delta_clip)
        cf = ax.contourf(xgrid, ygrid, dZ, levels=np.linspace(0, delta_clip, 30), cmap="viridis")

        if direct_pts.size > 0:
            in_view = (
                (direct_pts[:, i] >= xgrid[0]) & (direct_pts[:, i] <= xgrid[-1]) &
                (direct_pts[:, j] >= ygrid[0]) & (direct_pts[:, j] <= ygrid[-1])
            )
            if np.any(in_view):
                ax.scatter(direct_pts[in_view, i], direct_pts[in_view, j], color="red", s=4,
                           alpha=0.15, edgecolors="none", rasterized=True, label="DIRECT samples")

        for (name, p_int), color in zip(marked.items(), colors):
            if not (xgrid[0] <= p_int[i] <= xgrid[-1] and ygrid[0] <= p_int[j] <= ygrid[-1]):
                continue  # outside this panel's window -- don't let it silently expand the axes
            ax.scatter(p_int[i], p_int[j], color=color, s=70, edgecolors="black",
                       marker="*" if name == best_name else "o", label=name, zorder=5)

        if direct_best_p_int is not None:
            db_in = xgrid[0] <= direct_best_p_int[i] <= xgrid[-1] and ygrid[0] <= direct_best_p_int[j] <= ygrid[-1]
            if db_in:
                ax.scatter(direct_best_p_int[i], direct_best_p_int[j], color="orange", s=45,
                           edgecolors="black", marker="D", label="DIRECT's own best (pre-polish)",
                           zorder=6)

        if start_p_int is not None:
            start_in = xgrid[0] <= start_p_int[i] <= xgrid[-1] and ygrid[0] <= start_p_int[j] <= ygrid[-1]
            if start_in:
                start_label = f"winning polish start (source={best_start.get('source', '?')})"
                ax.scatter(start_p_int[i], start_p_int[j], color="white", s=45, edgecolors="black",
                           marker="X", label=start_label, zorder=6)

        if trajectory is not None:
            tp = trajectory[:, [i, j]]
            in_view = (
                (tp[:, 0] >= xgrid[0]) & (tp[:, 0] <= xgrid[-1]) &
                (tp[:, 1] >= ygrid[0]) & (tp[:, 1] <= ygrid[-1])
            )
            if np.any(in_view):
                # Real iterate-by-iterate path, not start->end -- this is
                # what actually shows the curved, multi-parameter route
                # (e.g. a0 growing while a1 converges to a2) that no
                # straight line or fixed-slice grid can represent.
                ax.plot(tp[:, 0], tp[:, 1], color="cyan", lw=1.2, alpha=0.8,
                        label=f"actual optimizer path ({traj_method})", zorder=5.5)

        ax.set_xlabel(f"internal coord {i}")
        ax.set_ylabel(f"internal coord {j}")
        return cf

    nrows = len(pairs)
    fig, axes = plt.subplots(nrows, 2, figsize=(10.0, 4.5 * nrows), squeeze=False)

    cf = None
    for row, (i, j) in enumerate(pairs):
        cf = draw_panel(axes[row, 0], i, j, span_for(i), span_for(j))
        axes[row, 0].set_title("wide (DIRECT's box + escape route)", fontsize=9)
        draw_panel(axes[row, 1], i, j, zoom_span, zoom_span)
        axes[row, 1].set_title(f"zoomed ({best_name}, ±{zoom_span:g})", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.tight_layout(rect=(0.0, 0.05, 0.9, 0.94))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center",
               ncol=min(4, len(by_label)), fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.colorbar(cf, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02, label="delta NLL (clipped)")
    fig.suptitle(f"{fcn_string}\nred dots = every point DIRECT actually evaluated (this sign combo)")

    if outfile is not None:
        fig.savefig(str(outfile), dpi=300, bbox_inches="tight")
        print(f"Likelihood-landscape plot saved to {outfile}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fcn_string", help='ESR function string, e.g. "a0/(a1+x)"')
    parser.add_argument("galaxy_name", help='Galaxy name as in the "Galaxy" column of the data file')
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--use-physical-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pmin", type=float, default=-20.0)
    parser.add_argument("--pmax", type=float, default=20.0)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--match-tol", type=float, default=1e-3)
    parser.add_argument("--direct-maxfun", type=int, default=3000,
                         help="DIRECT's evaluation budget (production default 3000). Raise this "
                              "to check whether a gap is a budget problem or a genuine blind spot.")
    parser.add_argument("--direct-maxiter", type=int, default=3000)
    parser.add_argument("--landscape-grid-n", type=int, default=40,
                         help="Resolution per axis of the likelihood-landscape plot's NLL grids.")
    parser.add_argument("--compute-codelen", action=argparse.BooleanOptionalAction, default=True,
                         help="Also run the production codelen calculation "
                              "(match.compute_function_codelen -- the same formula that produces "
                              "results_pretty_{N}.txt) on each arm's winning point. "
                              "Default on; pass --no-compute-codelen to skip (faster).")
    parser.add_argument("--check-match", action=argparse.BooleanOptionalAction, default=False,
                         help="Also print the per-parameter Fisher/integral-fallback diagnostic "
                              "trace (which parameter the Fisher diagonal resolved on its own, "
                              "which fell through to a profile-likelihood integral search, and "
                              "what value it found). Slower; default off.")
    parser.add_argument("--plot-integral-profiles", action=argparse.BooleanOptionalAction, default=False,
                         help="With --check-match: pop up a 1D profile-likelihood plot "
                              "immediately (not saved) per integral-fallback parameter, "
                              "marking the ML point and the 68% boundary the integral "
                              "converged to. Ignored without --check-match.")
    parser.add_argument("--codelen-pipeline-only", action=argparse.BooleanOptionalAction, default=True,
                         help="Only run the codelen/match checks on the production (DIRECT) "
                              "arm, not the two random-restart baselines. Default on -- pass "
                              "--no-codelen-pipeline-only to check all three arms.")
    args = parser.parse_args()

    validate_fit(
        fcn_string=args.fcn_string,
        galaxy_name=args.galaxy_name,
        data_file=args.data_file,
        use_physical_scale=args.use_physical_scale,
        pmin=args.pmin,
        pmax=args.pmax,
        n_random=args.n_random,
        seed=args.seed,
        out_dir=args.out_dir,
        match_tol=args.match_tol,
        direct_maxfun=args.direct_maxfun,
        direct_maxiter=args.direct_maxiter,
        landscape_grid_n=args.landscape_grid_n,
        compute_codelen=args.compute_codelen,
        check_match=args.check_match,
        plot_integral_profiles=args.plot_integral_profiles,
        codelen_pipeline_only=args.codelen_pipeline_only,
    )


if __name__ == "__main__":
    main()
