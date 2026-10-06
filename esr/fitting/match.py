
import math
import os
import sys
import itertools
import warnings

import numpy as np
import sympy
import jax
import jax.numpy as jnp
from jax import vmap
import scipy
from scipy.optimize import brentq
from mpi4py import MPI

import esr.fitting.test_all as test_all
import esr.fitting.test_all_Fisher as test_all_Fisher
from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def build_layout(nshape, max_fun_params=4, n_extra=0):
    """Return indices for the shape block, padding, extra physical params, and active params."""
    nshape = int(nshape)
    max_fun_params = int(max_fun_params)
    n_extra = int(n_extra)

    shape_idx = np.arange(0, nshape, dtype=int)
    pad_idx = np.arange(nshape, max_fun_params, dtype=int)
    extra_idx = np.arange(max_fun_params, max_fun_params + n_extra, dtype=int)
    active_idx = np.concatenate([shape_idx, extra_idx]) if n_extra > 0 else shape_idx.copy()
    return shape_idx, pad_idx, extra_idx, active_idx


def flat_upper_to_matrix(flat, n):
    """Reconstruct a symmetric matrix from its flattened upper triangle."""
    flat = np.asarray(flat, dtype=float).ravel()
    n = int(n)
    out = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n)
    m = min(len(flat), len(iu[0]))
    out[iu[0][:m], iu[1][:m]] = flat[:m]
    out = out + np.triu(out, 1).T
    return out


def safe_cumtrapz(y, x):
    """Compatibility wrapper for scipy.integrate.cumtrapz / cumulative_trapezoid."""
    if hasattr(scipy.integrate, "cumulative_trapezoid"):
        return scipy.integrate.cumulative_trapezoid(y, x, initial=0.0)
    return scipy.integrate.cumtrapz(y, x, initial=0.0)


def make_lambdified_eq(eq, nshape):
    """Lambdify a sympy expression with nshape ESR parameters."""
    nshape = int(nshape)
    try:
        if nshape == 0:
            return sympy.lambdify([x], eq, modules=["jax"])
        elif nshape == 1:
            return sympy.lambdify([x, a0], eq, modules=["jax"])
        else:
            all_a = list(sympy.symbols(" ".join([f"a{i}" for i in range(nshape)]), real=True))
            return sympy.lambdify([x] + all_a, eq, modules=["jax"])
    except KeyError:
        # sympy.lambdify can raise KeyError on a degenerate constant
        # expression (e.g. one that simplifies to ComplexInfinity).
        def _always_inf(xv, *_a):
            return jnp.full(jnp.shape(jnp.asarray(xv)), jnp.inf)
        return _always_inf


def codelen_from_vector(p, Delta, active_idx):
    """Description-length contribution from the active parameters only.

    A parameter contributes codelen only if it's either been snapped to
    exactly zero (p[j]==0 -- genuinely not needed by the model, correctly
    free) or has a resolved (finite, positive) Delta. A parameter that is
    actually used (p[j]!=0) but whose uncertainty could not be resolved by
    either the Fisher matrix or the integral fallback (e.g. two parameters
    that only enter the model through their combined product, so neither
    is individually identifiable) is floored at Delta=|theta| rather than propagating nan.
    """
    p = np.asarray(p, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    active_idx = np.asarray(active_idx, dtype=int)

    use = []
    for j in active_idx:
        if j < 0 or j >= len(p):
            continue
        if not np.isfinite(p[j]):
            continue
        if p[j] == 0:
            continue  # genuinely snapped/not needed -- correctly free
        if not (np.isfinite(Delta[j]) and Delta[j] > 0):
            continue  # unresolved precision -- floored at Delta=|theta|, costs 0, same as snapped
        use.append(j)

    if len(use) == 0:
        return 0.0

    use = np.asarray(use, dtype=int)
    return len(use) * math.log(2.0) + float(np.sum(np.log(np.abs(p[use]) / Delta[use])))


def get_sigma_from_integral(p, Sigma, fop, negloglike_all, active_idx, fcn_i="", number_points=10**3,
                             boundaries_out=None):
    """Estimate per-parameter sigma by integrating the 1D likelihood along each active direction.

    boundaries_out: optional dict, populated in-place as
    {j: (boundary_left, boundary_right)} for every parameter this function
    runs the integral fallback on (the log(1e3)-envelope steps found by
    find_boundary, i.e. the outer limits get_integral integrates within --
    NOT the returned Sigma, which is the narrower 68%-credible half-width
    found inside that envelope). Lets a caller that wants to plot/inspect
    the search (e.g. validate_fit.py's diagnostic plot) see both the
    envelope this function integrated over and the 68% width it found,
    without duplicating find_boundary's own search.

    Boundary-finding (where deltaNLL along this 1D direction first reaches
    log(1e3), a generous envelope well outside the eventual 68% region --
    see get_integral below) is a log10-step-size, grid-scan-then-bracket
    root find, done INDEPENDENTLY for the two directions (factor=-1 and
    +1). Previously this used multi-start Nelder-Mead minimization of
    |deltaNLL - log(1e3)| from ~7 fixed coarse initial guesses, with the
    two directions coupled (a failed factor=-1 search aborted factor=+1
    without even trying it, via an early `break`). Both were real, verified
    bugs: (1) a genuinely easy-to-find crossing on one side could be
    skipped entirely just because the coarse guesses happened not to
    converge on the OTHER side; (2) 1D Nelder-Mead from a handful of fixed
    starting points routinely overshot a real, sharp root straight into
    the invalid region (e.g. rho0 going negative) rather than converging
    to it -- confirmed by grid-scanning the actual objective at a case
    that reported "couldn't find integral limits" despite a clean, visible
    root in the profile-likelihood plot. A grid scan for the first sign
    change of (deltaNLL - log(1e3)), refined by bisection (brentq), finds
    the same root reliably without depending on where a fixed initial
    guess happens to land.
    """
    p = np.asarray(p, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    active_idx = np.asarray(active_idx, dtype=int)

    def signed_gap(x, pvec, j, factor):
        """deltaNLL - log(1e3) at step 10**x in direction `factor`; +inf
        (as a large finite sentinel, so brentq/bisection stay well-defined)
        wherever fop is non-finite -- i.e. past the parameter's valid
        domain, which is unambiguously "too far", the correct sign for a
        monotonically-departing-from-ML search direction.
        """
        params = np.asarray(pvec, dtype=float).copy()
        try:
            step = factor * (10.0 ** x)
        except OverflowError:
            # x this large only arises when theta_abs itself is already
            # near the float range's edge -- same "too far" case as a
            # non-finite fop below, so it gets the same sentinel.
            return 1e6
        params[j] = params[j] + step
        negloglike = fop(params)
        if np.isfinite(negloglike):
            return float(negloglike - negloglike_all - np.log(1e3))
        return 1e6

    def find_boundary(pvec, j, factor, theta_abs, n_grid=300):
        """Scan x=log10(step) from very small to a generous upper bound,
        return 10**x at the first sign change of signed_gap (refined by
        brentq), or None if signed_gap never crosses zero in range.
        """
        x_hi = max(np.log10(theta_abs) + 4.0, 6.0)
        x_grid = np.linspace(-12.0, x_hi, n_grid)
        prev_x, prev_g = x_grid[0], signed_gap(x_grid[0], pvec, j, factor)
        if prev_g >= 0:
            # Already past target at the smallest step -- no valid bracket
            # (this direction's deltaNLL doesn't start below the target).
            return None
        for x in x_grid[1:]:
            g = signed_gap(x, pvec, j, factor)
            if g >= 0:
                root_x = brentq(signed_gap, prev_x, x, args=(pvec, j, factor), xtol=1e-6)
                return 10.0 ** root_x
            prev_x, prev_g = x, g
        return None

    def compute_like(theta_vec):
        return jnp.exp(-fop(theta_vec) + negloglike_all)

    def get_integral(theta, boundary_left, boundary_right, param, number_points=5 * 10**2):
        """Cumulative probability mass in the symmetric interval
        [theta-d, theta+d], as a function of d, combining each side's OWN
        cumulative mass -- computed on its OWN dedicated grid, at its OWN
        (possibly much smaller, cutoff-truncated) extent, so a hard cutoff
        on one side no longer dilutes that side's sampling resolution the
        way sharing a single max(boundary_left, boundary_right) grid over
        both sides used to (most of that grid's points would land past a
        cutoff, in exp(-inf)=0 territory that contributes nothing).

        Each side's mass(d) is only defined out to its own boundary; past
        that, mass is held flat at its final (already ~100% of that side's
        contribution, by construction of `find_boundary`'s log(1e3)
        envelope) value via np.interp's `right=` fill -- exactly equivalent
        to the old shared-grid version's implicit truncation (exp(-inf)=0
        beyond a cutoff contributes nothing either way), just without
        spending samples to rediscover that.
        """
        d_max = max(boundary_left, boundary_right)
        d_common = np.linspace(0.0, d_max, number_points)

        a_range_left = np.linspace(theta, theta - boundary_left, number_points)
        thetas_left = np.tile(param, (len(a_range_left), 1))
        thetas_left[:, j] = a_range_left
        like_left = np.asarray(vmap(compute_like)(thetas_left), dtype=float)
        # cumtrapz over decreasing x yields a negative-growing array; negate
        # so mass_left(d) is a positive, increasing function of distance d.
        mass_left = -safe_cumtrapz(like_left, a_range_left)
        d_left = theta - a_range_left

        a_range_right = np.linspace(theta, theta + boundary_right, number_points)
        thetas_right = np.tile(param, (len(a_range_right), 1))
        thetas_right[:, j] = a_range_right
        like_right = np.asarray(vmap(compute_like)(thetas_right), dtype=float)
        mass_right = safe_cumtrapz(like_right, a_range_right)
        d_right = a_range_right - theta

        mass_left_interp = np.interp(d_common, d_left, mass_left, right=mass_left[-1])
        mass_right_interp = np.interp(d_common, d_right, mass_right, right=mass_right[-1])

        return mass_left_interp + mass_right_interp, d_common

    arg_integral = np.where((Sigma <= 0) | np.isinf(Sigma) | np.isnan(Sigma))[0]
    arg_integral = np.array([j for j in arg_integral if j in set(active_idx)], dtype=int)

    param = p.copy()

    for j in arg_integral:
        theta = param[j]
        if not np.isfinite(theta):
            # Scanning around an undefined ML point can never succeed --
            # skip immediately rather than running the full grid search
            # only to fail and print a "couldn't find limits" message that
            # reads like a search failure rather than an invalid input.
            print(f"Skipping integral search for {fcn_i}, parameter idx {j}: "
                  f"ML value itself is non-finite ({theta}) -- nothing to resolve.", flush=True)
            Sigma[j] = np.inf
            continue
        theta_abs = max(abs(theta), 1e-12)

        # Each direction is searched independently -- one side failing must
        # never stop the other side's (possibly trivially easy) search.
        boundary_left = find_boundary(param, j, -1, theta_abs)
        boundary_right = find_boundary(param, j, 1, theta_abs)

        if boundaries_out is not None:
            boundaries_out[int(j)] = (boundary_left, boundary_right)

        if boundary_left is None or boundary_right is None:
            print(f"Couldn't find integral limits for function {fcn_i}, parameter {theta}", flush=True)
            Sigma[j] = np.inf
            continue

        integral, d_range = get_integral(theta, boundary_left, boundary_right, param, number_points=number_points)

        if len(integral) < 2 or integral[-1] == 0:
            Sigma[j] = np.inf
            continue

        arg_min = np.argmin(np.abs(0.68 - integral / integral[-1]))
        arg_min = int(np.clip(arg_min + 1, 0, len(d_range) - 1))
        Sigma[j] = d_range[arg_min]

    return Sigma


def compute_function_codelen(
    fcn_i, p, fish_mat,
    shape_idx, extra_idx, active_idx, pad_idx,
    fop, negloglike_at_p, number_points=10**3,
    galaxy_idx=None, inc_d_cache=None, cache_key=None,
    stats=None,
):
    """
    Full production codelen for one function at one already-fitted point,
    given its joint Fisher matrix -- the SAME logic main() below uses to
    write codelen_matches_comp{N}.dat (and therefore combine_DL.py's
    results_pretty_{N}.txt): Fisher -> Sigma -> Delta, the
    get_sigma_from_integral fallback, Delta capped at |theta| for any
    unresolved/oversized entry (see codelen_from_vector's own docstring for
    why -- an uncapped oversized Delta gives log(|theta|/Delta) < 0, an
    unbounded-negative-codelen pathology), then a snap-vs-no-snap
    combinatorial exploration over shape parameters.

    Inc and D are NOT a special case here -- the caller folds them into
    extra_idx/active_idx (and into p/fish_mat) exactly like rho0/rs, so
    they go through this same Fisher/Sigma/Delta/integral-fallback/cap
    pipeline and the same codelen_from_vector formula, log(2) +
    log(|theta|/Delta), with no separate prior-density term. fop must
    already price -log(p(Inc))-log(p(D)) via its own include_priors=True
    evaluation over the FULL (Inc/D not fixed) vector -- that is the only
    place that term is ever counted; nothing here adds or subtracts it
    again. This replaces the previous design, which fixed Inc/D out of the
    Hessian, priced them via a separate galaxy_params_codelen term that DID
    include -log(p(theta_ML)), and subtracted that same amount back out of
    negloglike to avoid double-counting -- correct in total, but two
    parameters treated by a different rule than every other parameter for
    no benefit, and confusing to read results from (e.g. "negloglike"
    silently not matching what fop() at the same point returns).

    Factored out of main()'s per-row loop so any OTHER caller (e.g.
    validate_fit.py's diagnostics) gets an identical, non-drifting
    reproduction of the real production number instead of a hand-rolled
    approximation.

    Args:
        p: padded parameter vector at the point to price (shape params,
            padding, rho0/rs, Inc, D -- whatever active_idx/extra_idx say
            is active) -- mutated copy is returned (see below).
        fish_mat: the joint Hessian at p (same shape as len(p) squared),
            evaluated with include_priors=True and Inc/D as ordinary FREE
            elements of the differentiated vector, not fixed constants.
        fop: closure computing negloglike (include_priors=True, Inc/D free
            elements of its input, not fixed) at a given padded p vector --
            same contract get_sigma_from_integral expects.
        negloglike_at_p: fop(p), the reference negloglike this function's
            Sigma/Delta search measures distance from.
        galaxy_idx, inc_d_cache, cache_key: optional. When given, Inc/D's
            resolved Sigma (whether it came from the cheap Fisher-diagonal
            path or the expensive get_sigma_from_integral fallback) is
            cached in inc_d_cache under cache_key (the caller's canonical
            function index) and reused on subsequent calls with the same
            key, instead of being recomputed. This is sound because Inc/D's
            own Fisher/Sigma depends only on the physical predicted curve at
            the joint optimum -- identical for every duplicate-group row
            sharing a canonical index, regardless of which row-specific
            (possibly redundant) shape-parameter labeling reproduces that
            curve (verified: a coordinate's Hessian diagonal, with all
            others held at fixed NUMERIC values, cannot depend on how those
            fixed values are symbolically decomposed). It specifically
            targets get_sigma_from_integral's cost (a 300-point grid scan
            plus brentq bisection per direction), which Inc/D trigger often
            in practice (e.g. a function with no real Inc dependence, like a
            constant profile, has an exactly-zero Fisher diagonal there
            every time). The cache lookup happens BEFORE the
            bad_joint_hessian check below and is unconditionally overridden
            if THIS row's own joint Hessian is indefinite -- a fresh
            per-row indefiniteness finding is never overridden by a cached
            value from a different row, even though that finding should
            itself be index-invariant for a genuinely equivalent duplicate.

    Returns (codelen_total, negloglike_final, p_final, Delta_capped).
    p_final may differ from the input p if the snap-vs-no-snap exploration
    found a lower-DL snapped combination (shape parameters only -- Inc/D,
    like rho0/rs, are never snap-eligible). negloglike_final is exactly
    fop() at the returned p_final -- no term added or removed.
    """
    p = np.asarray(p, dtype=float).copy()
    active_idx = np.asarray(active_idx, dtype=int)
    max_param_total = len(p)
    diag_fish = np.diag(fish_mat)

    Sigma = np.full(max_param_total, np.inf, dtype=float)
    for j in active_idx:
        if j < len(diag_fish) and np.isfinite(diag_fish[j]) and diag_fish[j] > 0:
            Sigma[j] = 1.0 / np.sqrt(diag_fish[j])

    use_inc_d_cache = (
        galaxy_idx is not None and inc_d_cache is not None and cache_key is not None
    )
    cached_idx = set()
    if use_inc_d_cache:
        cached_sigma = inc_d_cache.get(cache_key)
        if cached_sigma is not None:
            for gi, sv in zip(galaxy_idx, cached_sigma):
                if gi < len(Sigma):
                    Sigma[gi] = sv
                    cached_idx.add(int(gi))

    fish_active = fish_mat[np.ix_(active_idx, active_idx)]
    bad_participation = np.zeros(len(active_idx), dtype=bool)
    if fish_active.size > 0:
        if not np.all(np.isfinite(fish_active)):
            # Can't even eigendecompose a non-finite matrix meaningfully --
            # conservatively fall back to marking the whole block, as before.
            bad_participation[:] = True
        else:
            eigvals_active, eigvecs_active = np.linalg.eigh(fish_active)
            tol = 1e-8 * np.max(np.abs(eigvals_active))
            eigenvector_weight_tol = 1e-6
            for be in np.where(eigvals_active < -tol)[0]:
                bad_participation |= np.abs(eigvecs_active[:, be]) > eigenvector_weight_tol
    if np.any(bad_participation):
        bad_active_idx = active_idx[bad_participation]
        # A fresh indefiniteness finding for THIS row overrides any cached
        # trust from a different row -- conservative, see docstring -- but
        # only for the parameters actually implicated, not the whole cache.
        Sigma[bad_active_idx] = np.inf
        cached_idx -= set(int(j) for j in bad_active_idx)

    check_idx = np.array([j for j in active_idx if int(j) not in cached_idx], dtype=int)
    if check_idx.size > 0 and np.any(
        (Sigma[check_idx] <= 0) | np.isinf(Sigma[check_idx]) | np.isnan(Sigma[check_idx])
    ):
        print(f"{fcn_i}: Fisher unusable for param idx {check_idx.tolist()}, running integral fallback", flush=True)
        if stats is not None:
            stats['n_integral_fallback'] = stats.get('n_integral_fallback', 0) + 1
        Sigma = get_sigma_from_integral(
            p, Sigma, fop, negloglike_at_p, active_idx=check_idx,
            fcn_i=fcn_i, number_points=number_points,
        )
        Sigma[np.isnan(Sigma)] = np.inf

    if use_inc_d_cache and cache_key not in inc_d_cache:
        inc_d_cache[cache_key] = tuple(
            float(Sigma[gi]) for gi in galaxy_idx if gi < len(Sigma)
        )

    Delta = np.zeros(max_param_total, dtype=float)
    finite = np.isfinite(Sigma) & (Sigma > 0)
    Delta[finite] = np.sqrt(12.0) * Sigma[finite]
    Delta[~finite] = np.inf

    if len(extra_idx) > 0:
        Delta[extra_idx] = np.where(Delta[extra_idx] == 0, np.abs(p[extra_idx]), Delta[extra_idx])

    p_orig = p.copy()
    Delta_orig = Delta.copy()

    def _snap_eligible(j):
        d = Delta[j]
        if not np.isfinite(d):
            return True
        if d <= 0:
            return False
        return (np.abs(p[j]) / d) < 1

    snap_idx = np.array(
        [j for j in shape_idx if j < len(p) and _snap_eligible(j)], dtype=int
    )
    extra_unresolved_idx = np.array(
        [j for j in extra_idx if j < len(p) and _snap_eligible(j)], dtype=int
    )

    Delta_capped = Delta_orig.copy()
    for _j in snap_idx:
        Delta_capped[_j] = np.abs(p_orig[_j])
    for _j in extra_unresolved_idx:
        Delta_capped[_j] = np.abs(p_orig[_j])

    def eval_total(p_candidate):
        try:
            neglog = float(fop(np.asarray(p_candidate, dtype=float)))
            if not np.isfinite(neglog):
                return np.inf, np.inf, np.inf
            code = codelen_from_vector(p_candidate, Delta_capped, active_idx)
            return neglog, code, neglog + code
        except Exception as e:
            print(f"match.py: eval_total failed for {fcn_i} on candidate "
                  f"{p_candidate}: {type(e).__name__}: {e}", flush=True)
            return np.inf, np.inf, np.inf

    if len(snap_idx) > 0:
        best_p = p_orig.copy()
        best_neglog, best_code, best_total = eval_total(best_p)

        p_snap_all = p_orig.copy()
        p_snap_all[snap_idx] = 0.0
        neglog_snap_all, code_snap_all, total_snap_all = eval_total(p_snap_all)
        if np.isfinite(total_snap_all) and total_snap_all < best_total - 1e-8:
            best_p = p_snap_all
            best_neglog, best_code, best_total = neglog_snap_all, code_snap_all, total_snap_all

        if len(snap_idx) > 1:
            for r in range(len(snap_idx) - 1, 0, -1):
                for idx_comb in itertools.combinations(snap_idx, r):
                    p_trial = p_orig.copy()
                    p_trial[list(idx_comb)] = 0.0
                    neglog_trial, code_trial, total_trial = eval_total(p_trial)
                    if np.isfinite(total_trial) and total_trial < best_total - 1e-8:
                        best_p = p_trial
                        best_neglog, best_code, best_total = neglog_trial, code_trial, total_trial

        p = best_p
        negloglike_final = best_neglog
        codelen_total = best_code
    else:
        negloglike_final = negloglike_at_p
        codelen_total = codelen_from_vector(p, Delta_capped, active_idx)

    if len(pad_idx) > 0:
        p[pad_idx] = 0.0
        Delta_capped[pad_idx] = 0.0

    return float(codelen_total), float(negloglike_final), p, Delta_capped


def get_functions(comp, likelihood, unique=True):
    if unique:
        unifn_file = likelihood.fn_dir + "/compl_%i/unique_equations_%i.txt" % (comp, comp)
    else:
        unifn_file = likelihood.fn_dir + "/compl_%i/all_equations_%i.txt" % (comp, comp)

    if comp >= 8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

    if rank == 0:
        for dirname in [likelihood.base_out_dir, likelihood.out_dir, likelihood.temp_dir]:
            if not os.path.isdir(dirname):
                print("Making dir:", dirname)
                os.makedirs(dirname, exist_ok=True)
    comm.Barrier()

    if rank == 0:
        print("Number of cores:", size, flush=True)

    with open(unifn_file, "r") as f:
        fcn_list = f.readlines()

    nLs = int(np.ceil(len(fcn_list) / float(size)))
    while nLs * (size - 1) > len(fcn_list):
        if rank == 0:
            print("Correcting for many cores.", flush=True)
        nLs -= 1

    if rank == 0:
        print("Total number of functions:", len(fcn_list), flush=True)
        print("Number of test points per proc:", nLs, flush=True)

    data_start = rank * nLs
    data_end = (rank + 1) * nLs
    if rank == size - 1:
        data_end = len(fcn_list)

    return fcn_list[data_start:data_end], data_start, data_end


# -----------------------------------------------------------------------------
# Main matching routine
# -----------------------------------------------------------------------------
def main(comp, likelihood, tmax=5, print_frequency=1000, try_integration=False):
    """
    Match the fitted functions and compute Fisher-based description lengths.

    Parameter ordering is always:
        [shape params up to 4 | padding to 4 | rho0, rs]
    """
    if likelihood.is_mse:
        raise ValueError("Cannot use MSE with description length")

    if rank == 0:
        print("\nMatching", flush=True)

    xvar = likelihood.xvar
    yvar = likelihood.yvar
    yerr_lo = likelihood.yerr_lo
    yerr_hi = likelihood.yerr_hi

    invsubs_file = likelihood.fn_dir + "/compl_%i/inv_subs_%i.txt" % (comp, comp)
    match_file = likelihood.fn_dir + "/compl_%i/matches_%i.txt" % (comp, comp)

    fcn_list_proc, data_start, data_end = test_all.get_functions(comp, likelihood, unique=False)

    fcn_list_unique, _, _ = test_all.get_functions(comp, likelihood, unique=True)
    (
        negloglike, params_meas, Nconv, Niter, times,
        inc_fit, d_fit, stage2_chi2,
    ) = test_all_Fisher.load_loglike(
        comp, likelihood, data_start, data_end, split=False
    )

    max_param_total = params_meas.shape[1]
    max_fun_params = 4
    use_physical_scale = getattr(likelihood, "use_physical_scale", False)
    n_extra = 2 if use_physical_scale else 0

    all_inv_subs_proc = simplifier.load_subs(invsubs_file, max_param_total)[data_start:data_end]
    matches_proc = np.atleast_1d(np.loadtxt(match_file).astype(int))[data_start:data_end]

    # derivs_comp{N}.dat holds the shape/extra Hessian upper-triangle
    # (get_deriv's layout, n_shape_cols columns) followed by Inc/D's own
    # Fisher-diagonal and cross-terms with the shape block (galaxy_hessian_extra's
    # layout, appended by test_all_Fisher.py's main() -- see its own comment
    # there for the exact combined column layout). Both blocks are keyed by
    # the same row index as matches_proc[i]. The galaxy block is used only to
    # SEED _inc_d_sigma_cache below (each row still runs its own fresh joint
    # Hessian for the shape block and the joint indefiniteness check -- this
    # only lets the FIRST row of a canonical function's duplicate group also
    # skip get_sigma_from_integral if this value already resolves it, instead
    # of only the 2nd-and-later rows benefiting from the cache this loop
    # itself populates). A derivs_comp{N}.dat from before this block was
    # appended (shape-only, exactly n_shape_cols columns) parses with
    # all_galaxy_fisher = None, same as when the data was simply absent.
    all_fish_combined = np.atleast_2d(np.loadtxt(likelihood.out_dir + "/derivs_comp" + str(comp) + ".dat"))
    n_shape_cols = max_param_total * (max_param_total + 1) // 2
    all_fish = all_fish_combined[:, :n_shape_cols]
    all_galaxy_fisher = (
        all_fish_combined[:, n_shape_cols:] if all_fish_combined.shape[1] > n_shape_cols else None
    )

    codelen = np.zeros(len(fcn_list_proc))
    negloglike_all = np.zeros(len(fcn_list_proc))
    index_arr = np.zeros(len(fcn_list_proc))
    params = np.zeros([len(fcn_list_proc), max_param_total])
    Nconv_all = np.zeros(len(fcn_list_proc))
    Niter_all = np.zeros(len(fcn_list_proc))
    times_all = np.zeros(len(fcn_list_proc))
    Deltas = np.zeros([len(fcn_list_proc), max_param_total])
    inc_fit_all = np.full(len(fcn_list_proc), np.nan)
    d_fit_all = np.full(len(fcn_list_proc), np.nan)

    _joint_polish_cache = {}

    _full_result_cache = {}

    # Tally of how many rows' Fisher couldn't resolve at least one active
    # parameter on its own and fell through to get_sigma_from_integral --
    # shared across every compute_function_codelen call this rank makes,
    # reported once at the end (per rank; not reduced across ranks).
    _fallback_stats = {'n_integral_fallback': 0}

    _inc_d_sigma_cache = {}

    if all_galaxy_fisher is not None:
        for _idx in range(all_galaxy_fisher.shape[0]):
            _f_inc, _f_d = all_galaxy_fisher[_idx, 0], all_galaxy_fisher[_idx, 1]
            _sigma_inc = 1.0 / np.sqrt(_f_inc) if (np.isfinite(_f_inc) and _f_inc > 0) else np.inf
            _sigma_d = 1.0 / np.sqrt(_f_d) if (np.isfinite(_f_d) and _f_d > 0) else np.inf
            _inc_d_sigma_cache[_idx] = (_sigma_inc, _sigma_d)

    for i in range(len(fcn_list_proc)):
        if rank == 0 and ((i == 0) or ((i + 1) % print_frequency == 0)):
            print(f"{i + 1} of {len(fcn_list_proc)}", flush=True)

        fcn_i = fcn_list_proc[i].replace("'", "")

        nshape = simplifier.count_params([fcn_i], max_fun_params)[0]
        shape_idx, pad_idx, extra_idx, active_idx = build_layout(nshape, max_fun_params, n_extra)

        index = matches_proc[i]
        index_arr[i] = index
        negloglike_all[i] = negloglike[index]
        Nconv_all[i] = Nconv[index]
        Niter_all[i] = Niter[index]
        times_all[i] = times[index]
        inc_fit_all[i] = inc_fit[index]
        d_fit_all[i] = d_fit[index]

        if np.isnan(negloglike[index]) or np.isinf(negloglike[index]):
            codelen[i] = np.nan
            continue

        full_cache_key = (fcn_i, int(index))
        if full_cache_key in _full_result_cache:
            codelen[i], negloglike_all[i], p_cached, Delta_capped_cached = _full_result_cache[full_cache_key]
            params[i, :] = p_cached
            Deltas[i, :] = Delta_capped_cached
            continue

        if index not in _joint_polish_cache:
            canonical_fcn = fcn_list_unique[index].replace("'", "")
            canonical_fcn_i, canonical_eq, canonical_integrated = likelihood.run_sympify(
                canonical_fcn, tmax=tmax, try_integration=try_integration
            )

            canonical_nshape = likelihood.nparam_shape
            measured_canonical = np.asarray(params_meas[index, :], dtype=float)
            shape_fit_joint = np.concatenate([
                measured_canonical[:canonical_nshape],
                measured_canonical[max_fun_params:max_fun_params + n_extra],
            ])
            inc_fit_joint, d_fit_joint = inc_fit[index], d_fit[index]
            _joint_polish_cache[index] = (shape_fit_joint, inc_fit_joint, d_fit_joint, canonical_nshape)

        shape_fit_joint, inc_fit_joint, d_fit_joint, canonical_nshape = _joint_polish_cache[index]

        fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
        eq_numpy = make_lambdified_eq(eq, nshape)

        loss_template = likelihood.get_loss(
            eq_numpy, integrated, value="evaluate", include_priors=True,
            fixed_galaxy_params=(inc_fit_joint, d_fit_joint),
        )
        chi2_fcn = likelihood.get_wrapped_like(loss_template)

        def fop(theta):
            theta = jnp.asarray(theta, dtype=float)
            return chi2_fcn(theta[active_idx], xvar, yvar, yerr_lo, yerr_hi)

        fish_flat = np.asarray(all_fish[index, :], dtype=float)
        fish_mat = flat_upper_to_matrix(fish_flat, max_param_total)

        p = np.zeros(max_param_total, dtype=float)
        jinv_shape = None
        try:
            if canonical_nshape > 0:
                p_shape, _, jinv_shape = simplifier.convert_params(
                    shape_fit_joint[:canonical_nshape],
                    fish_flat,
                    all_inv_subs_proc[i],
                    n=max_param_total,
                )
                p_shape = np.asarray(p_shape, dtype=float).ravel()
                n_assign = min(nshape, len(p_shape))
                p[:n_assign] = p_shape[:n_assign]
        except Exception as e:
            # No zero-padding fallback here -- matching CLASH's own
            # match.py, which excludes a row outright (there: codelen=inf)
            # whenever this reparametrization raises, rather than trying to
            # salvage a partial answer. The most common trigger is
            # structurally unrecoverable, not a transient glitch: nshape >
            # canonical_nshape (this row's own function has genuinely more
            # real shape parameters than the canonical form it's a duplicate
            # of), which has no valid transform at all -- there's no
            # legitimate way to fill in parameters the canonical fit never
            # had. Silently zero-filling those slots (the old behaviour)
            # gave every such row an identical, fabricated point rather than
            # a real fit, which is worse than excluding it.
            print(f"match.py: shape reparametrization failed for row {i} "
                  f"({fcn_i}): {type(e).__name__}: {e} -- excluding (no valid transform, "
                  f"matching CLASH's own except-and-exclude behaviour).", flush=True)
            print(f"  nshape={nshape}  canonical_nshape={canonical_nshape}  "
                  f"inv_subs={all_inv_subs_proc[i]!r}", flush=True)
            codelen[i] = np.nan
            continue

        # simplifier.convert_params can produce nan/inf WITHOUT raising --
        # e.g. a substitution that divides by a canonical shape symbol
        # (a0 -> 1/a0, a0 -> a0/a1, ...) evaluated via sympy.lambdify
        # compiles to plain numpy arithmetic, and numpy does not raise on
        # 1.0/0.0 or 0.0/0.0 (silently inf/nan, just a RuntimeWarning) --
        # and test_all_Fisher.py's own snap-to-zero step can legitimately
        # leave a canonical shape parameter at EXACTLY 0.0, so dividing by
        # it is a real, reachable case, not a hypothetical one. The
        # except-and-exclude block above only catches a raised exception;
        # this catches the silent case, before it ever reaches fop(p) or
        # the (expensive, and by construction unresolvable) integral
        # fallback for a parameter whose own ML value is already garbage.
        if not np.all(np.isfinite(p[shape_idx])):
            print(f"match.py: shape reparametrization for row {i} ({fcn_i}) produced a "
                  f"non-finite value with no exception raised -- p={p}, "
                  f"inv_subs={all_inv_subs_proc[i]!r} -- excluding.", flush=True)
            codelen[i] = np.nan
            continue

        # Zero the padding block always.
        if len(pad_idx) > 0:
            p[pad_idx] = 0.0

        # Keep any extra parameters (rho0, rs) at the end -- from
        # shape_fit_joint's own extras (the TRUE joint rho0/rs), not
        # test_all.py's raw pre-joint saved values.
        if n_extra > 0 and len(extra_idx) > 0:
            p[extra_idx] = shape_fit_joint[canonical_nshape:canonical_nshape + n_extra]

        negloglike_all[i] = float(fop(p))

        if not np.isfinite(negloglike_all[i]):
            codelen[i] = np.nan
            continue

        num_galaxy_params = getattr(likelihood, "num_galaxy_params", 0)
        if num_galaxy_params > 0:
            galaxy_idx = np.arange(max_param_total, max_param_total + num_galaxy_params)
            extra_idx_ext = np.concatenate([extra_idx, galaxy_idx]).astype(int)
            active_idx_ext = np.concatenate([shape_idx, extra_idx_ext]).astype(int)
            p_ext = np.concatenate([p, [inc_fit_joint, d_fit_joint]])

            # Build this row's own FULL [shape, extra, Inc, D] Hessian via a
            # cheap linear (Jacobian congruence) transform of the CANONICAL
            # function's own full joint Hessian (test_all_Fisher's saved
            # shape+extra block plus the Inc/D block and shape/extra-to-Inc/D
            # cross terms, both read from derivs_comp{N}.dat -- see all_fish/
            # all_galaxy_fisher's column split above), instead of a second,
            # expensive per-row JAX autodiff Hessian call.
            #
            # Why this is valid: inv_subs only ever substitutes the SHAPE
            # symbols (a0..a3) -- it never touches rho0/rs or Inc/D -- so
            # the full Jacobian relating this row's own [shape,extra,Inc,D]
            # to the canonical function's is BLOCK-DIAGONAL: jinv_shape on
            # the shape block, identity everywhere else. A block-diagonal
            # congruence transform of the full canonical Hessian correctly
            # propagates cross-terms too (ordinary multivariable chain
            # rule) -- verified numerically on a toy case with a genuine
            # rescaling substitution (not just a permutation), reproducing
            # a fresh row-level Hessian bit-for-bit, shape-Inc cross term
            # included. Only valid when canonical_nshape == nshape (always
            # true whenever inv_subs is non-NaN-sentinel, i.e. whenever
            # this code path is even reached) and when jinv_shape/the saved
            # galaxy Hessian data are actually available.
            fish_mat_ext = None
            n_model_compact = canonical_nshape + n_extra
            galaxy_row = (
                all_galaxy_fisher[index, :]
                if all_galaxy_fisher is not None and index < all_galaxy_fisher.shape[0]
                else None
            )
            if (
                galaxy_row is not None
                and np.all(np.isfinite(galaxy_row))
                and canonical_nshape == nshape
                and (canonical_nshape == 0 or jinv_shape is not None)
            ):
                try:
                    compact_shape_idx = np.arange(canonical_nshape)
                    compact_extra_idx = np.arange(canonical_nshape, n_model_compact)
                    padded_shape_idx = np.arange(canonical_nshape)
                    padded_extra_idx = np.arange(max_fun_params, max_fun_params + n_extra)
                    padded_combo_idx = np.concatenate([padded_shape_idx, padded_extra_idx]).astype(int)
                    compact_combo_idx = np.concatenate([compact_shape_idx, compact_extra_idx]).astype(int)

                    Hmat_canon_full = np.zeros((n_model_compact + 2, n_model_compact + 2))
                    Hmat_canon_full[np.ix_(compact_combo_idx, compact_combo_idx)] = (
                        fish_mat[np.ix_(padded_combo_idx, padded_combo_idx)]
                    )

                    fisher_inc_canon, fisher_d_canon, fisher_inc_d_canon = galaxy_row[0], galaxy_row[1], galaxy_row[2]
                    cross_inc = galaxy_row[3:3 + n_model_compact]
                    cross_d = galaxy_row[3 + max_param_total:3 + max_param_total + n_model_compact]

                    inc_pos, d_pos = n_model_compact, n_model_compact + 1
                    Hmat_canon_full[compact_combo_idx, inc_pos] = cross_inc
                    Hmat_canon_full[inc_pos, compact_combo_idx] = cross_inc
                    Hmat_canon_full[compact_combo_idx, d_pos] = cross_d
                    Hmat_canon_full[d_pos, compact_combo_idx] = cross_d
                    Hmat_canon_full[inc_pos, inc_pos] = fisher_inc_canon
                    Hmat_canon_full[d_pos, d_pos] = fisher_d_canon
                    Hmat_canon_full[inc_pos, d_pos] = fisher_inc_d_canon
                    Hmat_canon_full[d_pos, inc_pos] = fisher_inc_d_canon

                    Jinv_full = np.eye(n_model_compact + 2)
                    if canonical_nshape > 0:
                        Jinv_full[:canonical_nshape, :canonical_nshape] = jinv_shape

                    Hmat_row_full_compact = Jinv_full.T @ Hmat_canon_full @ Jinv_full

                    fish_mat_ext = np.zeros((len(p_ext), len(p_ext)))
                    fish_mat_ext[np.ix_(active_idx_ext, active_idx_ext)] = Hmat_row_full_compact
                except Exception as e:
                    print(f"match.py: Hessian transform failed for row {i} ({fcn_i}), "
                          f"falling back to a fresh Hessian call: {type(e).__name__}: {e}", flush=True)
                    fish_mat_ext = None

            if fish_mat_ext is None:
                hessian_template_ext = likelihood.get_loss(
                    eq_numpy, integrated, value="hessian", include_priors=True,
                )
                Hmat_ext_active = hessian_template_ext(
                    jnp.asarray(p_ext[active_idx_ext], dtype=float), xvar, yvar, yerr_lo, yerr_hi,
                )
                fish_mat_ext = np.zeros((len(p_ext), len(p_ext)))
                fish_mat_ext[np.ix_(active_idx_ext, active_idx_ext)] = np.asarray(Hmat_ext_active, dtype=float)

            loss_template_ext = likelihood.get_loss(
                eq_numpy, integrated, value="evaluate", include_priors=True,
            )
            chi2_fcn_ext = likelihood.get_wrapped_like(loss_template_ext)

            def fop_ext(theta):
                theta = jnp.asarray(theta, dtype=float)
                return chi2_fcn_ext(theta[active_idx_ext], xvar, yvar, yerr_lo, yerr_hi)

            negloglike_ext_at_p = float(fop_ext(p_ext))

            codelen[i], negloglike_all[i], p_ext_final, Delta_capped_ext = compute_function_codelen(
                fcn_i, p_ext, fish_mat_ext,
                shape_idx, extra_idx_ext, active_idx_ext, pad_idx,
                fop_ext, negloglike_ext_at_p, number_points=10**3,
                galaxy_idx=galaxy_idx, inc_d_cache=_inc_d_sigma_cache, cache_key=int(index),
                stats=_fallback_stats,
            )
            p = p_ext_final[:max_param_total]
            Delta_capped = Delta_capped_ext[:max_param_total]
        else:
            codelen[i], negloglike_all[i], p, Delta_capped = compute_function_codelen(
                fcn_i, p, fish_mat,
                shape_idx, extra_idx, active_idx, pad_idx,
                fop, negloglike_all[i], number_points=10**3,
                stats=_fallback_stats,
            )

        params[i, :] = p
        Deltas[i, :] = Delta_capped

        _full_result_cache[full_cache_key] = (
            float(codelen[i]), float(negloglike_all[i]),
            np.array(p, dtype=float), np.array(Delta_capped, dtype=float),
        )

    print(f"rank {rank}: {_fallback_stats['n_integral_fallback']} of {len(fcn_list_proc)} rows "
          f"hit the integral fallback (get_sigma_from_integral) for at least one parameter", flush=True)

    fcn_list_proc = np.array(fcn_list_proc)
    print(fcn_list_proc[np.argwhere(codelen == np.inf)])
    print(fcn_list_proc[np.argwhere(codelen == -np.inf)])

    out_arr = np.vstack(
        [negloglike_all, codelen, index_arr]
        + [params[:, i] for i in range(max_param_total)]
        + [Deltas[:, i] for i in range(max_param_total)]
        + [inc_fit_all, d_fit_all]
        + [Nconv_all, Niter_all, times_all]
    )
    out_arr = np.transpose(out_arr)

    np.savetxt(
        likelihood.temp_dir + "/codelen_matches_" + str(comp) + "_" + str(rank) + ".dat",
        out_arr,
        fmt="%.7e",
    )

    comm.Barrier()

    if rank == 0:
        string = (
            "cat `find "
            + likelihood.temp_dir
            + '/ -name "codelen_matches_'
            + str(comp)
            + '_*.dat" | sort -V` > '
            + likelihood.out_dir
            + "/codelen_matches_comp"
            + str(comp)
            + ".dat"
        )
        os.system(string)
        string = "rm " + likelihood.temp_dir + "/codelen_matches_" + str(comp) + "_*.dat"
        os.system(string)

    comm.Barrier()
    return
