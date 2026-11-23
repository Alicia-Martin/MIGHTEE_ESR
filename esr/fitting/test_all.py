
import re
import numpy as np
import jax
import jax.numpy as jnp
import sympy
import itertools
from scipy.optimize import direct as scipy_direct
from scipy.optimize import minimize as scipy_minimize

from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier

import time
import os
import sys

try:
    import mpi4py.MPI as MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    # optimise_fun_direct_nm (and the helpers around it) never touch
    # comm/rank/size -- only get_functions() and main() do, and both are
    # only ever called on a cluster where mpi4py is installed. This fallback
    # just lets the module (and optimise_fun_direct_nm specifically) be
    # imported standalone, e.g. from validate_fit.py, on a machine without
    # mpi4py/MPI.
    comm = None
    rank = 0
    size = 1


# The shared function library has literal zoo/nan/oo/-oo tokens baked into
# some candidate strings at every complexity from 4 upward (a generation/
# simplification artifact, not a meaningful physical density profile --
# oo**a0 is 0 for any a0<0, 1 at a0=0, infinite for a0>0: a genuine
# step-function likelihood surface with no usable gradient anywhere except
# a measure-zero point, and its second-derivative autodiff is not usable
# either). Matched on
# word boundaries so this can't accidentally hit a real identifier that
# merely contains "oo"/"nan" as a substring (none exist in this function
# library's alphabet, but the boundary costs nothing and removes any risk).
DEGENERATE_TOKEN_RE = re.compile(r'\b(oo|zoo|nan)\b')


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_direct_bounds(nparam, log_opt=False, direct_bounds=None, pmin=-5, pmax=5):
    """
    Build DIRECT bounds for the internal optimiser coordinates.

    Both the log_opt and non-log_opt branches respect pmin/pmax exactly as
    given, including asymmetric pairs -- the bounds are not mirrored or
    forced symmetric around zero.
    """
    if nparam <= 0:
        return []

    if direct_bounds is not None:
        bounds = [tuple(b) for b in direct_bounds]
        if len(bounds) != nparam:
            raise ValueError(
                f"direct_bounds has length {len(bounds)} but nparam={nparam}"
            )
        return bounds

    return [(float(pmin), float(pmax))] * nparam

def pick_diverse_starts(samples, k, delta_ll, min_sep):
    """
    Choose up to k samples from those with fun <= best + delta_ll,
    enforcing a minimum separation in parameter space.
    """
    finite = [s for s in samples if np.isfinite(s["fun"]) and s["fun"] < 1e29]
    if len(finite) == 0:
        return []

    finite = sorted(finite, key=lambda s: s["fun"])
    best_fun = finite[0]["fun"]
    pool = [s for s in finite if s["fun"] <= best_fun + delta_ll]

    selected = []
    for s in pool:
        p = np.asarray(s["p_internal"], dtype=float)
        too_close = False
        for s0 in selected:
            p0 = np.asarray(s0["p_internal"], dtype=float)
            if np.linalg.norm(p - p0) < min_sep:
                too_close = True
                break
        if not too_close:
            selected.append(s)
        if len(selected) >= k:
            break

    return selected


def add_unique_start(dest, sample, min_sep):
    p = np.asarray(sample["p_internal"], dtype=float)
    for existing in dest:
        p0 = np.asarray(existing["p_internal"], dtype=float)
        if np.linalg.norm(p - p0) < min_sep:
            return False
    dest.append(sample)
    return True


def sign_to_text(signs):
    if signs is None:
        return "None"
    return str(np.asarray(signs, dtype=int))


def physical_params_from_internal(p_internal, signs=None, log_opt=False):
    """
    Convert internal optimiser coordinates to physical parameters.
    """
    p_internal = np.asarray(p_internal, dtype=float)

    if log_opt:
        if signs is None:
            signs = np.ones_like(p_internal)
        signs = np.asarray(signs, dtype=float)
        return signs * 10.0 ** p_internal

    return p_internal.copy()


def get_functions(comp, likelihood, unique=True):
    """Load all functions for a given complexity and distribute them among ranks."""
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


def split_layout(nshape, max_fun_params=4, n_extra=0):
    """
    Saved/full layout is always:
        [active ESR shape params | padding to 4 | rho0, rs]
    """
    nshape = int(nshape)
    max_fun_params = int(max_fun_params)
    n_extra = int(n_extra)

    shape_idx = np.arange(0, nshape, dtype=int)
    pad_idx = np.arange(nshape, max_fun_params, dtype=int)
    extra_idx = np.arange(max_fun_params, max_fun_params + n_extra, dtype=int)
    active_idx = np.concatenate([shape_idx, extra_idx]) if n_extra > 0 else shape_idx.copy()
    return shape_idx, pad_idx, extra_idx, active_idx


def compact_from_internal(x_internal, signs=None, log_opt=False):
    """Return active physical parameters from internal coordinates."""
    if log_opt:
        return physical_params_from_internal(x_internal, signs=signs, log_opt=True)
    return np.asarray(x_internal, dtype=float).copy()


def full_from_compact(theta_active, nshape, max_fun_params=4, n_extra=0):
    """Convert [shape..., extras] to [shape..., pad..., extras]."""
    nshape = int(nshape)
    max_fun_params = int(max_fun_params)
    n_extra = int(n_extra)
    theta_active = np.asarray(theta_active, dtype=float).ravel()

    full = np.zeros(max_fun_params + n_extra, dtype=float)
    if nshape > 0:
        full[:nshape] = theta_active[:nshape]
    if n_extra > 0:
        full[max_fun_params:max_fun_params + n_extra] = theta_active[nshape:nshape + n_extra]
    return full


def lambdify_equation(eq, nshape):
    """Lambdify sympy expression using the standard ESR parameter naming."""
    try:
        if nshape > 1:
            all_a = list(sympy.symbols(" ".join([f"a{i}" for i in range(nshape)]), real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
        elif nshape == 1:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
        else:
            eq_numpy = sympy.lambdify(x, eq, modules=["jax"])
        return eq_numpy
    except KeyError:
        # sympy.lambdify can raise KeyError when the expression evaluates to
        # sympy's ComplexInfinity/NaN ("zoo") for some functions in the
        # library. Rather than let that crash the optimiser, treat the
        # function as always-infinite loss so it is scored as unfittable
        # and the sweep continues.
        def _always_inf(xv, *_a):
            return jnp.full(jnp.shape(jnp.asarray(xv)), jnp.inf)
        return _always_inf


def galaxy_params_polish(likelihood, eq_numpy, integrated, active_params, f_seeds=(1.0,)):
    """
    Stage-2 unbounded joint polish over [active_params (from stage 1,
    compact shape+rho0/rs vector), Inc, D]. Every candidate function in the
    per-complexity sweep gets its own Inc/D polish, not just a single chosen
    winner.

    Inc/D are NOT part of the bounded DIRECT search (see
    optimise_fun_direct_nm) -- they have real, informative priors
    (catalog inclination/distance), unlike the flat-prior shape/rho0/rs
    parameters, so they get this separate, unbounded local polish instead.

    Inc is multi-started from a small, fixed set of seeds rather than a
    single seed at the catalog mean. A single seed can converge to a
    non-stationary point: for functions whose stage-1 shape fit is already
    difficult, Nelder-Mead can report "converged" because its mixed-scale
    simplex (dominated by huge rho0/rs values) satisfies its own tolerance
    while barely moving Inc at all. The fixed spread of seeds is tried
    unconditionally (not gated behind a convergence check, since a false
    "looks converged" result would never trigger a conditional retry), and
    whichever seed converges to the lowest actual loss is kept. D is not
    multi-started, since D consistently converges to essentially the same
    value regardless of the Inc seed.

    The seeds used are the catalog Inc mean plus two fixed values (45 and
    80 degrees), avoiding the extreme edges of the [0, 90] range: Inc's
    truncated-Gaussian prior support ends there, and seeds placed right at
    the boundary tend to get stuck against it. The seed near 80 degrees is
    the most load-bearing of the two non-catalog seeds, since it is close
    to the boundary where the true optimum sits for functions with a
    degenerate-DM-vanishing pathology; the seed at 45 degrees is kept as a
    middle-of-range check. This multi-seed approach does not guarantee the
    true global optimum, but multi-seed agreement is a much stronger
    practical signal than any single seed's result on its own.

    With likelihood.fit_ml, the stellar M/L factor f is polished jointly with
    Inc and D, starting from each of f_seeds (default: its prior mean f = 1
    only; the extra f-seeded stage 1 adds its own f, see stage1_f_seed).

    Returns (shape_fit, inc_fit, d_fit, stage2_chi2, f_fit), f_fit nan without
    fit_ml. shape_fit is the
    jointly-optimized shape+rho0/rs vector this polish actually landed on
    (same compact/unpadded format as the input active_params). Callers
    needing self-consistent galaxy Fisher/codelen must use shape_fit here,
    not their own pre-polish shape values, when calling
    galaxy_params_codelen -- pairing the new inc_fit/d_fit with a stale,
    pre-polish shape vector is not a stationary point of anything, so its
    Hessian has no reason to be positive.

    If likelihood.num_galaxy_params is 0 (no Inc/D on this likelihood),
    returns (nan, nan, nan, nan) -- caller should skip this call entirely
    in that case; kept simple here since test_all.py's main() only calls
    this when num_galaxy_params > 0.
    """
    loss_eval_priors = likelihood.get_loss(
        eq_numpy, integrated, value="evaluate", include_priors=True
    )

    def obj(a):
        val = float(loss_eval_priors(
            jnp.asarray(a, dtype=jnp.float64),
            likelihood.xvar, likelihood.yvar, likelihood.yerr_lo, likelihood.yerr_hi,
        ))
        return val if np.isfinite(val) else 1e30

    # x0 mixes raw physical-scale rho0/rs (~1e12-1e18) with Inc/D (~1e1-1e4),
    # so Nelder-Mead's default absolute xatol/fatol (1e-4) essentially never
    # triggers on its own; convergence in practice is reached well within
    # maxiter=500, which keeps a wide safety margin at a fraction of the
    # cost of a much larger maxiter.
    active_params = np.asarray(active_params, dtype=float)
    # Fixed Inc seeds: catalog mean, plus 45 and 80 degrees (see this
    # function's docstring for why these three).
    inc_seeds = sorted(set([float(likelihood.inc_true), 45.0, 80.0]))
    fit_ml = getattr(likelihood, "fit_ml", False)

    best_res = None
    for inc0 in inc_seeds:
        for f0 in (list(dict.fromkeys(float(f) for f in f_seeds)) if fit_ml else [None]):
            x0 = np.concatenate([active_params, [inc0, likelihood.distance_true] + ([f0] if fit_ml else [])])
            res = scipy_minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 500})
            if best_res is None or (np.isfinite(res.fun) and res.fun < best_res.fun):
                best_res = res

    n_model = len(active_params)
    shape_fit = np.asarray(best_res.x[:n_model], dtype=float)
    inc_fit = float(best_res.x[n_model])
    d_fit = float(best_res.x[n_model + 1])
    stage2_chi2 = float(best_res.fun)
    f_fit = float(best_res.x[n_model + 2]) if fit_ml else np.nan
    return shape_fit, inc_fit, d_fit, stage2_chi2, f_fit


def check_fit_plausibility(likelihood, eq_numpy, integrated, model_params, inc_fit, d_fit,
                            dm_frac_threshold=0.05, inc_boundary_margin=1.0,
                            d_sigma_threshold=5.0, f_fit=np.nan):
    """
    Post-hoc, non-scoring plausibility check on a function's finished stage-2
    fit -- diagnostic only, never affects chi2/codelen/DL. Flags a
    degenerate-fit pattern (rho0->0 or rs->huge, forcing Inc/D to compensate
    by leaning entirely on the fixed baryon term). This is deliberately kept
    out of the loss itself: folding it in as a penalty would bias
    model-selection ranking against candidate functions the data genuinely
    supports finding little or no DM in, which runs against the point of
    using ESR to let the data determine the DM density functional form (or
    its absence). Instead this flags the pattern for human review.

    Plain numpy, not JAX-traced -- runs once per function after optimisation
    finishes (never called inside a jax.jit/grad trace).

    Returns a short "; "-joined reason string (each check that fires appends
    its own clause), or "" if nothing looks wrong. Never raises -- a failure
    in this diagnostic must not fail the fit itself.
    """
    if getattr(likelihood, "num_galaxy_params", 0) == 0:
        return ""
    if not (np.isfinite(inc_fit) and np.isfinite(d_fit)):
        return ""

    reasons = []

    try:
        _, dm_v2, baryon_v2 = likelihood.get_pred(
            likelihood.xvar, model_params, eq_numpy, integrated=integrated,
            D=d_fit, Inc=inc_fit, return_components=True,
            f_ml=f_fit if getattr(likelihood, "fit_ml", False) else None,
        )
        dm_v2 = np.asarray(dm_v2, dtype=float)
        baryon_v2 = np.asarray(baryon_v2, dtype=float)
        safe_dm_v2 = np.where(np.isfinite(dm_v2), dm_v2, 0.0)
        dm_frac = safe_dm_v2 / (safe_dm_v2 + baryon_v2 + 1e-10)
        max_frac = float(np.max(dm_frac))
        if max_frac < dm_frac_threshold:
            reasons.append(f"DM negligible everywhere (max fraction={max_frac:.3f})")
    except Exception as e:
        reasons.append(f"DM-fraction check failed: {type(e).__name__}: {e}")

    if inc_fit <= inc_boundary_margin or inc_fit >= 90.0 - inc_boundary_margin:
        reasons.append(f"Inc at prior boundary ({inc_fit:.2f} deg)")

    e_d = getattr(likelihood, "e_d", None)
    if e_d is not None and e_d > 0:
        d_sigma = abs(d_fit - likelihood.distance_true) / e_d
        if d_sigma > d_sigma_threshold:
            reasons.append(f"D {d_sigma:.1f} sigma from catalog")

    return "; ".join(reasons)


# -----------------------------------------------------------------------------
# DIRECT + Nelder-Mead optimiser
# -----------------------------------------------------------------------------
def _optimise_fun_direct_nm_once(
    fcn_i,
    likelihood,
    tmax,
    pmin,
    pmax,
    comp=0,
    try_integration=False,
    log_opt=False,
    max_param_shape=4,
    method="Nelder-Mead",
    direct_bounds=None,
    # vol_tol=1e-30/len_tol=1e-10 (below) essentially never trigger DIRECT's
    # early stopping on their own for a search this low-dimensional (2-6
    # dims), so direct_maxfun/direct_maxiter are the effective budget. Final
    # precision comes from the Nelder-Mead polish afterward, not from DIRECT
    # itself, so DIRECT's budget only needs to be large enough to find the
    # right basin, not to converge tightly.
    direct_maxfun=3000,
    direct_maxiter=3000,
    hybrid_global_n=10,
    global_pool_delta_ll=5.0,
    global_pool_seed=42,
    start_min_separation=1e-6,
    n_top_combos_for_pool=4,
    hybrid_per_combo_n=2,
    per_combo_pool_delta_ll=5.0,
    # Multiple distinct polished shape basins are carried into stage 2,
    # rather than just the single global-best one. hybrid_starts already
    # collects several genuinely different starting points (best-direct,
    # global pool, per-sign bests, per-combo pool) and polishes every one of
    # them; a basin that scores slightly worse on stage 1 alone (shape-only,
    # Inc/D at catalog defaults) can still be substantially better once
    # Inc/D are jointly polished in stage 2, since the shape-only ranking
    # and the joint ranking are not the same ordering. This costs nothing
    # extra on the DIRECT/polish side (already computed); the only added
    # cost is trying stage 2 K times instead of once. min_sep=2.0 decades is
    # roughly DIRECT's own typical sample spacing, rounded up so two
    # candidates count as "the same basin" only when they really are close.
    # No loss-magnitude cutoff is applied when selecting candidates
    # (pick_diverse_starts is called with delta_ll=inf below), since a
    # candidate that looks worse on the shape-only stage-1 loss is exactly
    # the case this mechanism is meant to rescue.
    n_shape_candidates=3,
    shape_candidate_min_sep=2.0,
    # Optional out-parameter (populated in-place, not returned) so a caller
    # can inspect the raw DIRECT sample points / hybrid starts for plotting
    # or debugging without changing this function's return signature -- every
    # existing 8-tuple-unpacking call site stays unaffected.
    diagnostics_out=None,
    # Lets main()'s batch loop show progress through the function list
    # directly on the per-function header, instead of only every
    # print_frequency functions. Both None (the default, used by every
    # direct/scratchpad call site outside main()) just omits the "(i/N)"
    # suffix.
    fcn_index=None,
    fcn_total=None,
    # Inc held fixed during stage 1 (DIRECT + polish), D at catalog. None =
    # catalog Inc, the original behaviour. See optimise_fun_direct_nm.
    stage1_inc=None,
    # fit_ml only: f held fixed at this value during stage 1 (None = f = 1,
    # the original behaviour), and stage 2 also starts f from it. See
    # stage1_f_seed.
    stage1_f=None,
):
    """
    Optimise one function by:
      1) DIRECT global search in internal space
      2) selecting several good DIRECT points
      3) polishing each with Nelder-Mead
      4) returning the best polished result

    DIRECT is the only global search mechanism used here; there is no
    random-restart component anywhere in this function. Any case
    DIRECT+polish misses is fixed by extending DIRECT's own search (budget,
    box, or an added search dimension), not by bolting a random-restart
    supplement onto it.
    """
    xvar = likelihood.xvar
    yvar = likelihood.yvar
    yerr_lo = getattr(likelihood, "yerr_lo", getattr(likelihood, "yerr", None))
    yerr_hi = getattr(likelihood, "yerr_hi", yerr_lo)

    if yerr_lo is None:
        raise ValueError("Could not find yerr, yerr_lo, or yerr_hi on likelihood.")

    use_physical_scale = bool(getattr(likelihood, "use_physical_scale", False))
    n_extra = 2 if use_physical_scale else 0

    # Parse the function and, if possible, integrate it analytically
    fcn_i = fcn_i.replace("\n", "").replace("'", "")
    fcn_i, eq, integrated = likelihood.run_sympify(
        fcn_i,
        tmax=tmax,
        try_integration=try_integration,
    )

    # Count active ESR shape params only
    nshape = simplifier.count_params([fcn_i], max_param_shape)[0]
    nparam = nshape + n_extra  # compact active vector length
    nparam_out = max_param_shape + n_extra  # saved/full vector length

    # Build lambdified shape function
    eq_numpy = lambdify_equation(eq, nshape)

    # Scalar loss for DIRECT / Nelder-Mead. include_priors=False (the
    # default): Inc/D are NOT part of this bounded search -- they get their
    # own unbounded joint polish afterward (see the stage-2 step below),
    # matching testing_opt_mightee.py's fit_galaxy_params_after_direct
    # pattern. The vector this search produces is only nparam long
    # (shape+rho0/rs); passing include_priors=True here would silently
    # read Inc/D past the end of that vector (JAX clamps out-of-bounds
    # indices rather than raising), corrupting both the prior term and
    # the D-based radius rescaling inside get_pred.
    loss_eval = likelihood.get_loss(
        eq_numpy,
        integrated,
        value="evaluate",
        include_priors=False,
        fixed_galaxy_params=(
            None if stage1_inc is None and stage1_f is None else
            (float(likelihood.inc_true if stage1_inc is None else stage1_inc),
             float(likelihood.distance_true))
            + (() if stage1_f is None else (float(stage1_f),))
        ),
    )
    wrapped_eval = likelihood.get_wrapped_like(loss_eval)
    f_seeds = (1.0,) if stage1_f is None else (1.0, float(stage1_f))

    # Differentiable version of the same internal (log-magnitude + sign)
    # coordinates, for a BFGS polish alongside Nelder-Mead. On some
    # candidate functions with a narrow, badly-scaled ridge in parameter
    # space, Nelder-Mead's rank-only simplex comparisons can get pulled into
    # a degenerate escape route that BFGS's gradient avoids, tracking the
    # ridge down instead. Neither algorithm wins universally (the reverse
    # happens on other functions), so both are tried from every start and
    # whichever converges better is kept -- no new search points, just more
    # thorough use of the ones DIRECT/the pools already found.
    def _loss_of_internal(x_internal, signs_arr):
        # log_opt is a plain Python bool closed over from this function's own
        # argument, constant for every call within one optimise_fun_direct_nm
        # invocation -- resolved at JIT trace time, not a runtime branch.
        # The sign*10**x transform is only applied when log_opt=True; when
        # log_opt=False, x_internal already IS the physical value directly
        # (matching dm_likelihood.py::wrapped_like's own
        # `if signs is None: p = x.copy()` branch).
        if log_opt:
            p = signs_arr * (10.0 ** x_internal)
        else:
            p = x_internal
        return loss_eval(p, xvar, yvar, yerr_lo, yerr_hi)

    _value_and_grad_internal = jax.jit(jax.value_and_grad(_loss_of_internal))

    progress = f" ({fcn_index}/{fcn_total})" if fcn_index is not None and fcn_total is not None else ""
    print(f"\nOptimising function: {fcn_i.strip()}{progress}")

    if nparam == 0:
        val = wrapped_eval(
            np.array([]),
            xvar, yvar, yerr_lo, yerr_hi,
            signs=None,
            check_nans=True,
        )
        num_galaxy_params = getattr(likelihood, "num_galaxy_params", 0)
        if num_galaxy_params > 0:
            _shape_fit, inc_fit, d_fit, stage2_chi2, f_fit = galaxy_params_polish(
                likelihood, eq_numpy, integrated, np.array([]), f_seeds=f_seeds
            )
        else:
            inc_fit, d_fit, stage2_chi2, f_fit = np.nan, np.nan, np.nan, np.nan
        flag_reason = check_fit_plausibility(
            likelihood, eq_numpy, integrated, np.array([]), inc_fit, d_fit, f_fit=f_fit,
        )
        return (float(val), np.zeros(nparam_out), 0, 0, True,
                inc_fit, d_fit, stage2_chi2, flag_reason, f_fit)

    # DIRECT bounds in internal coordinates
    bounds = make_direct_bounds(
        nparam,
        log_opt=log_opt,
        direct_bounds=direct_bounds,
        pmin=pmin,
        pmax=pmax,
    )

    # Sign combinations
    if log_opt and nparam > 0:
        if use_physical_scale:
            # branch only over the ESR-shape parameters; extras are fixed positive
            shape_signs = list(itertools.product([1, -1], repeat=nshape))
            sign_list = [tuple(s) + (1, 1) for s in shape_signs]
            if len(sign_list) == 0:
                sign_list = [(1, 1)]
        else:
            sign_list = list(itertools.product([1, -1], repeat=nparam))
    else:
        sign_list = [None]

    global_samples = []
    direct_results = []

    # ---- DIRECT over sign branches ----
    for i, signs in enumerate(sign_list, start=1):

        def obj_direct(p_internal):
            p_internal = np.asarray(p_internal, dtype=float)
            val = wrapped_eval(
                p_internal,
                xvar,
                yvar,
                yerr_lo,
                yerr_hi,
                signs=signs,
                check_nans=True,
            )
            val = float(val)
            val_safe = val if np.isfinite(val) else 1e30

            global_samples.append({
                "fun": val_safe,
                "p_internal": p_internal.copy(),
                "signs": signs,
            })
            return val_safe

        res_direct = scipy_direct(
            obj_direct,
            bounds,
            maxfun=direct_maxfun,
            maxiter=direct_maxiter,
        )

        direct_results.append({
            "fun": float(res_direct.fun) if np.isfinite(res_direct.fun) else 1e30,
            "p_internal": np.asarray(res_direct.x, dtype=float),
            "signs": signs,
            "res": res_direct,
        })

    direct_results = sorted(direct_results, key=lambda d: d["fun"])
    global_samples = sorted(global_samples, key=lambda d: d["fun"])

    best_direct = direct_results[0]

    # ---- choose starts for Nelder-Mead ----
    hybrid_starts = []

    # Always include the best DIRECT point
    hybrid_starts.append({
        "fun": best_direct["fun"],
        "p_internal": np.asarray(best_direct["p_internal"], dtype=float),
        "signs": best_direct["signs"],
        "source": "best-direct",
    })

    # Add a few diverse points from the good DIRECT pool
    global_candidates = pick_diverse_starts(
        global_samples,
        hybrid_global_n,
        global_pool_delta_ll,
        start_min_separation,
    )
    for sample in global_candidates:
        sample_copy = dict(sample)
        sample_copy["source"] = "global"
        add_unique_start(hybrid_starts, sample_copy, start_min_separation)

    # Also include the best endpoint from each sign branch
    for sample in direct_results:
        if not np.isfinite(sample["fun"]) or sample["fun"] >= 1e29:
            continue
        sample_copy = dict(sample)
        sample_copy["source"] = "per-sign"
        add_unique_start(hybrid_starts, sample_copy, start_min_separation)

    # Per-sign-combo diverse pool: the pool above is filtered against the
    # GLOBAL best, so it contributes zero diversity to any sign combo whose
    # own best is far from the global best -- which is common, since
    # different sign combos routinely land at very different raw
    # likelihoods, and the combo holding the true optimum can otherwise be
    # represented by only its single per-sign point. So for the top few sign
    # combos (by their OWN best nll -- bounded, so this doesn't blow up for
    # functions with many sign combos), pull a few diverse points measured
    # against THAT combo's own best, not the global one.
    if nshape > 0:
        samples_by_combo = {}
        for s in global_samples:
            key = tuple(s["signs"]) if s["signs"] is not None else None
            samples_by_combo.setdefault(key, []).append(s)
        combo_bests = sorted(
            ((key, min(s["fun"] for s in samples)) for key, samples in samples_by_combo.items()),
            key=lambda kv: kv[1],
        )
        for key, _ in combo_bests[:n_top_combos_for_pool]:
            combo_candidates = pick_diverse_starts(
                samples_by_combo[key],
                hybrid_per_combo_n,
                per_combo_pool_delta_ll,
                start_min_separation,
            )
            for sample in combo_candidates:
                sample_copy = dict(sample)
                sample_copy["source"] = "per-combo"
                add_unique_start(hybrid_starts, sample_copy, start_min_separation)

    # Polish is fully unbounded. Bounding rho0/rs to stop them reaching
    # large values would cost real minima, not just make them look nicer --
    # some functions' genuine best joint solution sits at a physically huge
    # rho0/rs value, past any bound short of the true, data-dependent ridge
    # edge (which isn't predictable in advance). The optimiser's job is to
    # find the minimum; whether that minimum looks physical is a separate
    # concern for priors/bounds to handle, not something to solve by
    # constraining the optimiser itself.

    # ---- Nelder-Mead polish ----
    best_polish = None
    best_fun = np.inf
    best_signs = None
    best_start = None
    # Every start's own best-of-(NM, L-BFGS-B) result, kept alongside the
    # single global best above -- feeds the multi-basin selection for stage 2
    # after the loop (see n_shape_candidates comment above).
    polish_results = []

    for j, start in enumerate(hybrid_starts, start=1):
        p0 = np.asarray(start["p_internal"], dtype=float)
        signs = start["signs"]

        def obj_polish(p_internal):
            p_internal = np.asarray(p_internal, dtype=float)
            val = wrapped_eval(
                p_internal,
                xvar,
                yvar,
                yerr_lo,
                yerr_hi,
                signs=signs,
                check_nans=True,
            )
            val = float(val)
            return val if np.isfinite(val) else 1e30

        res_polish = scipy_minimize(
            obj_polish,
            p0,
            method=method if method in ("Nelder-Mead",) else "Nelder-Mead",
            options={"maxiter": 5000},
        )

        # Plain unbounded BFGS: keep whichever of NM/BFGS converges better
        # for this start, no new search points involved. signs is None
        # whenever log_opt=False (every DIRECT-derived start); jnp.asarray(
        # None, ...) raises, so fall back to a same-length dummy --
        # _loss_of_internal's own log_opt branch above never reads
        # signs_arr in that case anyway (x_internal already IS the physical
        # value), this just needs to be a JIT-shape-compatible placeholder.
        signs_arr = (
            jnp.asarray(signs, dtype=jnp.float64)
            if signs is not None
            else jnp.ones(len(p0), dtype=jnp.float64)
        )

        def obj_bfgs(p_internal, signs_arr=signs_arr):
            val, grad = _value_and_grad_internal(
                jnp.asarray(p_internal, dtype=jnp.float64), signs_arr
            )
            val = float(val)
            grad = np.nan_to_num(np.asarray(grad, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(val):
                val = 1e30
            return val, grad

        res_bfgs = scipy_minimize(obj_bfgs, p0, jac=True, method="BFGS")

        if np.isfinite(res_bfgs.fun) and res_bfgs.fun < res_polish.fun:
            res_polish = res_bfgs

        polish_results.append({
            "fun": float(res_polish.fun) if np.isfinite(res_polish.fun) else 1e30,
            "p_internal": np.asarray(res_polish.x, dtype=float),
            "signs": signs,
            "source": start.get("source"),
        })

        if np.isfinite(res_polish.fun) and res_polish.fun < best_fun:
            best_fun = float(res_polish.fun)
            best_polish = res_polish
            best_signs = signs
            best_start = start

    num_galaxy_params = getattr(likelihood, "num_galaxy_params", 0)

    if best_polish is None:
        # Stage 1 itself failed to find any finite polished result -- no
        # sensible starting point for the stage-2 Inc/D polish either, so
        # report the catalog defaults with an infinite stage-2 chi2
        # (consistent with success=False / chi2=inf already signalling
        # this function failed).
        if num_galaxy_params > 0:
            inc_fit, d_fit, stage2_chi2 = likelihood.inc_true, likelihood.distance_true, np.inf
        else:
            inc_fit, d_fit, stage2_chi2 = np.nan, np.nan, np.nan
        if diagnostics_out is not None:
            diagnostics_out.update({
                "global_samples": global_samples, "direct_results": direct_results,
                "hybrid_starts": hybrid_starts, "nshape": nshape, "bounds": bounds,
                "best_signs": None, "active_params": None,
            })
        # No sensible fitted shape/Inc/D to check plausibility of here --
        # chi2=inf already signals this function failed outright.
        return (np.inf, np.zeros(nparam_out), 0, 0, False,
                inc_fit, d_fit, stage2_chi2, "", 1.0 if getattr(likelihood, "fit_ml", False) else np.nan)

    # ---- convert back to physical params ----
    # Whether sign*10**x conversion is needed is decided by best_signs being
    # not None, not by the global log_opt flag. best_signs is None exactly
    # when log_opt was False for the winning start (since only log_opt=True
    # branches ever populate real sign tuples), so checking best_signs
    # directly gives the correct conversion regardless of which start won.
    if best_signs is not None and nparam > 0:
        active_params = np.asarray(best_signs, dtype=float) * (10.0 ** np.asarray(best_polish.x, dtype=float))
    else:
        active_params = np.asarray(best_polish.x, dtype=float)

    active_params = np.asarray(active_params, dtype=float).ravel()

    # full saved layout: [shape | padding to 4 | extras]
    full_params = full_from_compact(active_params, nshape, max_fun_params=max_param_shape, n_extra=n_extra)

    chi2_i = best_fun
    j_out = int(getattr(best_polish, "nit", 0))
    count_lowest = len(hybrid_starts)
    success = bool(getattr(best_polish, "success", False))

    # The console summary is printed further down, after winning_k is
    # determined, so that what's printed always matches the full_params/
    # chi2_i that actually get saved for this function (rather than
    # printing candidate 0's shape/loss before stage 2 has picked a winner).

    # ---- select up to n_shape_candidates distinct polished shape basins ----
    # See the n_shape_candidates comment near this function's signature.
    # delta_ll=np.inf: no loss-magnitude gate, only the min_sep diversity
    # constraint -- a candidate with a worse shape-only loss is exactly
    # what this is meant to rescue. The single overall-best result
    # (active_params/best_signs/best_polish.x, already selected above) is
    # always included since it is by construction the lowest-fun entry in
    # polish_results.
    shape_candidate_starts = pick_diverse_starts(
        polish_results, n_shape_candidates, np.inf, shape_candidate_min_sep,
    )
    shape_candidates = []
    for cand in shape_candidate_starts:
        if log_opt and cand["signs"] is not None and nparam > 0:
            cand_params = np.asarray(cand["signs"], dtype=float) * (
                10.0 ** np.asarray(cand["p_internal"], dtype=float)
            )
        else:
            cand_params = np.asarray(cand["p_internal"], dtype=float)
        shape_candidates.append(np.asarray(cand_params, dtype=float).ravel())
    if not shape_candidates:
        shape_candidates = [active_params]

    # ---- stage 2: unbounded joint Inc/D polish (every function gets one) ----
    if num_galaxy_params > 0:
        # Tried once per shape candidate (see n_shape_candidates above),
        # keeping whichever (shape candidate x Inc seed) combination gives
        # the lowest joint loss -- a shape basin that looked slightly worse
        # on stage 1 alone can win here once Inc/D are free to adapt to it,
        # which is the whole point of carrying more than one through.
        best_stage2 = None
        for k, cand_params in enumerate(shape_candidates):
            _sf, _inc, _d, _s2chi2, _f = galaxy_params_polish(
                likelihood, eq_numpy, integrated, cand_params, f_seeds=f_seeds
            )
            if best_stage2 is None or (np.isfinite(_s2chi2) and _s2chi2 < best_stage2[3]):
                best_stage2 = (_sf, _inc, _d, _s2chi2, k, _f)
        _shape_fit, inc_fit, d_fit, stage2_chi2, winning_k, f_fit = best_stage2

        # shape_fit (stage 2's own joint-optimum output) is saved directly
        # and unconditionally, not the pre-polish shape_candidates[winning_k]
        # input. galaxy_params_polish is not a fixed point of its own output
        # (re-seeding it from shape_fit can land on a nearby but slightly
        # different optimum), so downstream consumers must read shape_fit
        # off disk as the authoritative joint-optimum shape rather than
        # re-deriving it by re-running the polish themselves.
        active_params = np.asarray(_shape_fit, dtype=float).ravel()
        full_params = full_from_compact(
            active_params, nshape, max_fun_params=max_param_shape, n_extra=n_extra
        )
        # chi2_i is the stage-2 (joint shape+Inc+D) likelihood, matching
        # full_params (also stage 2's own output, saved above) -- so the
        # saved negloglike and the saved params always describe the same
        # point, and downstream code can use this column directly instead
        # of re-deriving it.
        chi2_i = stage2_chi2

        print(
            f"    stage 2: Inc fit={inc_fit:.4f} (catalog {likelihood.inc_true:.4f}), "
            f"D fit={d_fit:.4f} (catalog {likelihood.distance_true:.4f}), "
            + (f"f_ml fit={f_fit:.4f} (prior 1 +/- {likelihood.sigma_frac_ml:.4f}), "
               if getattr(likelihood, "fit_ml", False) else "")
            + f"stage2_chi2={stage2_chi2:.6f} (winning shape candidate "
            f"{winning_k+1}/{len(shape_candidates)})",
            flush=True,
        )
    else:
        inc_fit, d_fit, stage2_chi2, f_fit = np.nan, np.nan, np.nan, np.nan

    # full_params/chi2_i here are whatever will actually be returned/saved --
    # shape_fit (stage 2's joint-optimum output) if num_galaxy_params > 0,
    # candidate 0's stage-1 result otherwise (see the stage-2 block above).
    print(
        "params",
        full_params,
        chi2_i,
        "(",
        count_lowest,
        "/",
        j_out,
        ")",
        method,
        flush=True,
    )

    if diagnostics_out is not None:
        diagnostics_out.update({
            "global_samples": global_samples, "direct_results": direct_results,
            "hybrid_starts": hybrid_starts, "nshape": nshape, "bounds": bounds,
            "best_signs": best_signs, "active_params": active_params,
            "best_start": best_start, "shape_candidates": shape_candidates,
            # The genuine stage-1-only (shape params, Inc/D fixed at catalog)
            # loss -- NOT what the returned chi2_i ends up holding once stage
            # 2 runs (chi2_i is overwritten with stage2_chi2 above, by design,
            # so it stays paired with full_params). A caller that wants a
            # true stage-1-vs-stage-1 comparison (e.g. against a shape-only
            # random-restart baseline) needs this, not the returned tuple's
            # first element.
            "stage1_chi2": best_fun,
        })

    flag_reason = check_fit_plausibility(
        likelihood, eq_numpy, integrated, active_params, inc_fit, d_fit, f_fit=f_fit,
    )

    return (chi2_i, full_params, j_out, count_lowest, success,
            inc_fit, d_fit, stage2_chi2, flag_reason, f_fit)


# main(): save this rank's partial results every CHECKPOINT_EVERY functions
# (resumed automatically on restart), and clear JAX's compile cache after each
# function (memory otherwise grows with every function fitted).
CHECKPOINT_EVERY = int(os.environ.get("ESR_CHECKPOINT_EVERY", 200))
CLEAR_JAX_CACHES = os.environ.get("ESR_CLEAR_JAX_CACHES", "1") != "0"
SKIP_PREVIOUS_EQNS = os.environ.get("ESR_SKIP_PREVIOUS", "1") != "0"


# Extra stage-1 inclinations for galaxies where the catalogue baryons exceed
# the observed velocity. "req": only the Inc at which v_bar <= v_obs at every
# radius; "req+sigma": also catalog Inc - 1 and - 2 sigma; None: off.
STAGE1_INC_MODE = "req"


def stage1_inc_seeds(likelihood, mode=STAGE1_INC_MODE):
    """Extra fixed inclinations for stage 1, or [] when not needed.

    Stage 1 fits the shape with Inc at the catalogue value. If v_bar > v_obs
    somewhere at that Inc, any dark matter makes the fit worse there, so
    stage 1 drives the DM to ~0 (rho0 rs^2 ~ 1e-10). Stage 2 then polishes
    rho0/rs in linear units from that point, where -logL is flat over many
    decades, and only lowers Inc -- it cannot find the joint optimum with
    lower Inc AND substantial DM (MGTH_J095717.1+020853, 1/(x**2 + x):
    -logL 27.6 stored vs 19.8 true). Seeding stage 1 at an Inc where the
    baryons fit under the data lets it find the DM-on basin.
    """
    if not mode or getattr(likelihood, "num_galaxy_params", 0) == 0:
        return []
    v = np.asarray(likelihood.yvar, dtype=float)
    vb = np.asarray(likelihood.v_bar, dtype=float)
    if not np.any(vb > v):
        return []
    inc0, sig = float(likelihood.inc_true), float(likelihood.e_inc)
    ratio = float(np.min(v[vb > 0] / vb[vb > 0]))
    inc_req = float(np.degrees(np.arcsin(np.clip(np.sin(np.radians(inc0)) * ratio, 0.0, 1.0))))
    seeds = [inc_req]
    if mode == "req+sigma":
        seeds += [inc0 - sig, inc0 - 2.0 * sig]
    out = []
    for s in seeds:
        s = float(np.clip(s, 1.0, 89.0))
        if abs(s - inc0) > 0.5 and all(abs(s - t) > 0.5 for t in out):
            out.append(s)
    return out


def stage1_f_seed(likelihood, mode=STAGE1_INC_MODE):
    """fit_ml only: the f at which v_bar <= v_obs at every radius with Inc and
    D at their catalogue values, or None when not needed / not possible.

    The f counterpart of stage1_inc_seeds: with f free, the baryons can be
    brought under the data either by lowering Inc or by lowering f
    (v_bar^2 = f v_star^2 + v_gas^2). Seeding only the low-Inc way would
    miss a DM-on basin at the catalogue Inc with lower f, which can cost
    less prior (galaxies with a narrow f prior vs a broad Inc prior).
        f_req = min over radii with v_star > 0 of (v_obs^2 - v_gas^2) / v_star^2
    None when f_req <= 0 (the gas alone exceeds v_obs: no f can fix it) or
    f_req >= 1 (v_bar already fits under the data).
    """
    if not mode or not getattr(likelihood, "fit_ml", False) \
            or getattr(likelihood, "num_galaxy_params", 0) == 0:
        return None
    v = np.asarray(likelihood.yvar, dtype=float)
    vb = np.asarray(likelihood.v_bar, dtype=float)
    vs = np.asarray(likelihood.v_star, dtype=float)
    if not np.any(vb > v):
        return None
    m = vs > 0
    if not np.any(m):
        return None
    # v_gas^2 = v_bar^2 - v_star^2 (catalogue: v_bar^2 = v_star^2 + v_gas^2)
    f_req = float(np.min((v[m] ** 2 - (vb[m] ** 2 - vs[m] ** 2)) / vs[m] ** 2))
    if not (0.0 < f_req < 1.0):
        return None
    return f_req


def optimise_fun_direct_nm(fcn_i, likelihood, *args, stage1_inc_mode=None, **kwargs):
    """_optimise_fun_direct_nm_once at the catalogue Inc, plus -- only for
    galaxies with v_bar > v_obs somewhere (stage1_inc_seeds) -- one more run
    per extra stage-1 inclination, and with fit_ml one more at the catalogue
    Inc with f lowered to bring v_bar under the data (stage1_f_seed). Keeps
    the run with the lowest final (stage-2, joint) loss; same arguments and
    return tuple as before."""
    best = _optimise_fun_direct_nm_once(fcn_i, likelihood, *args, **kwargs)
    mode = STAGE1_INC_MODE if stage1_inc_mode is None else stage1_inc_mode
    for inc_s in stage1_inc_seeds(likelihood, mode):
        print(f"    extra stage 1 at Inc={inc_s:.2f} (catalog {float(likelihood.inc_true):.2f}): "
              "v_bar > v_obs at the catalogue Inc", flush=True)
        res = _optimise_fun_direct_nm_once(fcn_i, likelihood, *args, stage1_inc=inc_s, **kwargs)
        if np.isfinite(res[0]) and (not np.isfinite(best[0]) or res[0] < best[0]):
            best = res
    f_s = stage1_f_seed(likelihood, mode)
    if f_s is not None:
        print(f"    extra stage 1 at f={f_s:.3f}, catalogue Inc: v_bar > v_obs at f = 1", flush=True)
        res = _optimise_fun_direct_nm_once(fcn_i, likelihood, *args, stage1_f=f_s, **kwargs)
        if np.isfinite(res[0]) and (not np.isfinite(best[0]) or res[0] < best[0]):
            best = res
    return best


# -----------------------------------------------------------------------------
# Main batch routine
# -----------------------------------------------------------------------------
def main(comp, likelihood, tmax=90, pmin=-6, pmax=10, print_frequency=50,
         try_integration=False, log_opt=False, Niter_params=[40,60],
         Nconv_params=[-5,20], method="DIRECT"):
    """
    Optimise all functions for a given complexity and save results to file.

    pmin/pmax default to an asymmetric [-6, 10] rather than a symmetric
    range. Good-fit parameters that go out of range essentially never need
    more room on the negative-log10 side -- they need it on the positive
    side, since rho0/rs needs to be a physically large number far more
    often than a physically tiny one. A same-width box shifted toward the
    positive side reaches those cases without paying the resolution cost
    of a wider symmetric box.

    The box is also kept relatively narrow rather than very wide: DIRECT's
    own resolution within its box degrades as the box widens, since a
    fixed evaluation budget has to cover more decades in up to several
    dimensions. The unbounded polish that follows can walk from a modest
    starting point all the way to whatever magnitude the true optimum
    needs -- DIRECT's own job is finding the right basin/sign combo, not
    the exact magnitude, so a narrower box that finds the right basin
    reliably outperforms a wider box with degraded resolution. This
    depends on make_direct_bounds (see its own docstring) correctly
    respecting an asymmetric pmin/pmax pair.
    """
    if likelihood.is_mse:
        raise ValueError("Cannot use MSE with description length")

    if rank == 0:
        print("\nRunning fits", flush=True)

    fcn_list_proc, _, _ = get_functions(comp, likelihood)

    if rank == 0:
        previous_unifn_list = []
        if comp > 1:
            for compl in range(1, comp):
                unifn_file_i = likelihood.fn_dir + "/compl_%i/unique_equations_%i.txt" % (compl, compl)
                with open(unifn_file_i, "r") as f:
                    fcn_list_i = f.readlines()
                previous_unifn_list += fcn_list_i
        previous_unifn_list = np.array(previous_unifn_list)
        np.savetxt(
            likelihood.fn_dir + "/compl_" + str(comp) + "/previous_eqns_" + str(comp) + ".txt",
            previous_unifn_list,
            fmt="%s",
        )
    comm.Barrier()

    # Skip functions already seen at a lower complexity: an exact text match
    # against previous_eqns_{comp}.txt. Their lower-comp copy has a shorter
    # tree and the same likelihood, so it always has the smaller DL; refitting
    # them only costs time (17% of comp 9, 28% of comp 8). Recorded as
    # chi2=inf, which test_all_Fisher.py, match.py and combine_DL.py skip.
    # ESR_SKIP_PREVIOUS=0 turns it off.
    previous_fns = set()
    if comp > 1 and SKIP_PREVIOUS_EQNS:
        with open(likelihood.fn_dir + "/compl_" + str(comp) + "/previous_eqns_" + str(comp) + ".txt") as f:
            previous_fns = {line.replace("\n", "").replace("'", "").strip() for line in f}
        previous_fns.discard("")
    n_skipped_previous = 0

    chi2 = np.zeros(len(fcn_list_proc))
    Niter_opt = np.zeros(len(fcn_list_proc))
    Nconv_opt = np.zeros(len(fcn_list_proc))
    times = np.zeros(len(fcn_list_proc))
    inc_arr = np.full(len(fcn_list_proc), np.nan)
    d_arr = np.full(len(fcn_list_proc), np.nan)
    stage2_chi2_arr = np.full(len(fcn_list_proc), np.nan)
    f_arr = np.full(len(fcn_list_proc), np.nan)

    max_param_shape = int(max(4, np.floor((comp - 1) / 2)))
    n_extra = 2 if getattr(likelihood, "use_physical_scale", False) else 0
    max_param_total = max_param_shape + n_extra
    params = np.zeros([len(fcn_list_proc), max_param_total])

    # Non-scoring plausibility flags (check_fit_plausibility, above) --
    # sparse and text, so a plain list rather than one more homogeneous
    # float column in out_arr/params. Written to its own file below, never
    # fed into chi2/codelen/DL.
    flagged_rows = []

    print(len(fcn_list_proc), flush=True)

    # Checkpoint / resume, so a job killed late in a long run (comp 9 takes
    # days per galaxy) does not lose its work. Every CHECKPOINT_EVERY functions this rank's
    # arrays are saved to temp_dir (atomic replace); a restarted job with the
    # same function list and MPI size resumes after the last saved function.
    # The checkpoint is deleted once the final per-rank file is written.
    ckpt_path = os.path.join(likelihood.temp_dir, f"ckpt_comp{comp}_rank{rank}of{size}.npz")
    import hashlib  # not hash(): str hashes are randomised per process
    fcn_key = hashlib.md5("\n".join(s.strip() for s in fcn_list_proc).encode()).hexdigest() \
        + f"_{len(fcn_list_proc)}"
    start_i = 0
    if os.path.exists(ckpt_path):
        try:
            ck = np.load(ckpt_path, allow_pickle=False)
            if str(ck["fcn_key"]) == fcn_key and ck["params"].shape == params.shape:
                start_i = int(ck["next_i"])
                chi2[:], params[:], inc_arr[:], d_arr[:] = ck["chi2"], ck["params"], ck["inc"], ck["d"]
                stage2_chi2_arr[:], f_arr[:] = ck["stage2"], ck["f"]
                Nconv_opt[:], Niter_opt[:], times[:] = ck["nconv"], ck["niter"], ck["times"]
                flagged_rows = [s for s in ck["flagged"].tolist() if s]
                print(f"RESUMING from checkpoint {ckpt_path}: functions 0..{start_i - 1} already done",
                      flush=True)
            else:
                print(f"Ignoring checkpoint {ckpt_path}: different function list or layout", flush=True)
        except Exception as e:
            print(f"Ignoring unreadable checkpoint {ckpt_path}: {type(e).__name__}: {e}", flush=True)

    def _save_checkpoint(next_i):
        tmp = ckpt_path + ".tmp.npz"
        np.savez(tmp, fcn_key=np.array(fcn_key), next_i=np.array(next_i), chi2=chi2, params=params,
                 inc=inc_arr, d=d_arr, stage2=stage2_chi2_arr, f=f_arr, nconv=Nconv_opt,
                 niter=Niter_opt, times=times, flagged=np.array(flagged_rows + [""], dtype=str))
        os.replace(tmp, ckpt_path)

    for i in range(start_i, len(fcn_list_proc)):
        if rank == 0 and ((i == 0) or ((i + 1) % print_frequency == 0)):
            print(f"{i + 1} of {len(fcn_list_proc)}", flush=True)

        start = time.time()

        if fcn_list_proc[i].replace("\n", "").replace("'", "").strip() in previous_fns:
            chi2[i] = np.inf
            j, count_lowest = 0, 0
            flag_reason = ""
            n_skipped_previous += 1
        elif DEGENERATE_TOKEN_RE.search(fcn_list_proc[i]):
            # Never even attempt to fit a literal oo/zoo/nan candidate --
            # see DEGENERATE_TOKEN_RE's own comment for why these aren't
            # meaningful functions regardless of complexity. Recorded the
            # same way an unfittable function already is (chi2=nan), so
            # every downstream stage (test_all_Fisher.py, match.py,
            # combine_DL.py) excludes it exactly as it already does for
            # nan chi2 -- no new code path needed there.
            print(f"SKIPPING function {i} ({fcn_list_proc[i].strip()!r}): "
                  f"literal oo/zoo/nan token, not a meaningful physical "
                  f"density profile -- recording as unfittable (chi2=nan), "
                  f"never fit.", flush=True)
            chi2[i] = np.nan
            j, count_lowest = 0, 0
            flag_reason = ""
        else:
            # A handful of library functions can crash the optimiser -- e.g.
            # a sympy parse resolving to ComplexInfinity/NaN, sometimes
            # triggered by sympy-internal-cache state left over from a
            # previous function in the same process rather than by the
            # failing string itself. This guard treats such a failure the
            # same as any other unfittable function: chi2=nan, which every
            # downstream stage (test_all_Fisher.py, match.py) already
            # short-circuits on -- rather than letting one bad function
            # crash a run that may represent hours of prior work on
            # thousands of others.
            try:
                (
                    chi2[i], params[i, :], j, count_lowest, success,
                    inc_arr[i], d_arr[i], stage2_chi2_arr[i], flag_reason, f_arr[i],
                ) = optimise_fun_direct_nm(
                    fcn_list_proc[i],
                    likelihood,
                    tmax,
                    pmin,
                    pmax,
                    comp=comp,
                    try_integration=try_integration,
                    log_opt=log_opt,
                    max_param_shape=max_param_shape,
                    method="Nelder-Mead" if method is None else method,
                    fcn_index=i + 1,
                    fcn_total=len(fcn_list_proc),
                )
            except Exception as e:
                print(f"WARNING: function {i} ({fcn_list_proc[i].strip()!r}) raised "
                      f"{type(e).__name__}: {e} -- recording as unfittable (chi2=nan).",
                      flush=True)
                chi2[i] = np.nan
                j, count_lowest = 0, 0
                flag_reason = ""

        if flag_reason:
            flagged_rows.append(f"{fcn_list_proc[i].strip()};{flag_reason}")

        end = time.time()
        time_taken = end - start
        Niter_opt[i] = j
        Nconv_opt[i] = count_lowest
        times[i] = time_taken

        print(
            i,
            fcn_list_proc[i],
            chi2[i],
            time_taken,
            "s",
            "(",
            count_lowest,
            "/",
            j,
            ")",
            method,
            flush=True,
        )

        # Every function builds its own jax.jit programs, and JAX keeps every
        # compiled program in a process-wide cache: over ~28k comp-9
        # functions that grows until the job is OOM-killed. Nothing compiled
        # for one function is reused by the next, so clearing it costs nothing.
        if CLEAR_JAX_CACHES and hasattr(jax, "clear_caches"):
            jax.clear_caches()
        if (i + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(i + 1)

    print(f"rank {rank}: skipped {n_skipped_previous} of {len(fcn_list_proc)} functions already "
          f"present at a lower complexity (chi2 = inf)", flush=True)

    # Column layout: [stage1_chi2, shape+rho0/rs params (max_param_total),
    # Inc, D, stage2_chi2, (f if fit_ml), Nconv, Niter, times].
    out_arr = np.vstack(
        [chi2]
        + [params[:, i] for i in range(max_param_total)]
        + [inc_arr, d_arr, stage2_chi2_arr]
        + ([f_arr] if getattr(likelihood, "fit_ml", False) else [])
        + [Nconv_opt, Niter_opt, times]
    )
    out_arr = np.transpose(out_arr)

    np.savetxt(
        likelihood.temp_dir + "/chi2_comp" + str(comp) + "weights_" + str(rank) + ".dat",
        out_arr,
        fmt="%.7e",
    )
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    comm.Barrier()

    if rank == 0:
        string = (
            "cat `find "
            + likelihood.temp_dir
            + '/ -name "chi2_comp'
            + str(comp)
            + 'weights_*.dat" | sort -V` > '
            + likelihood.out_dir
            + "/negloglike_comp"
            + str(comp)
            + ".dat"
        )
        os.system(string)
        string = "rm " + likelihood.temp_dir + "/chi2_comp" + str(comp) + "weights_*.dat"
        os.system(string)

    comm.Barrier()

    # Plausibility-flag list (check_fit_plausibility, above) -- same
    # per-rank-write-then-rank-0-concatenate pattern as negloglike_comp
    # above, into its own file so it never touches the existing format
    # test_all_Fisher.py/match.py already parse. Written every rank (even
    # if empty) so the find/cat/sort below behaves uniformly.
    with open(likelihood.temp_dir + "/flagged_comp" + str(comp) + "_" + str(rank) + ".dat", "w") as f:
        for row in flagged_rows:
            f.write(row + "\n")

    comm.Barrier()

    if rank == 0:
        string = (
            "cat `find "
            + likelihood.temp_dir
            + '/ -name "flagged_comp'
            + str(comp)
            + '_*.dat" | sort -V` > '
            + likelihood.out_dir
            + "/flagged_comp"
            + str(comp)
            + ".dat"
        )
        os.system(string)
        string = "rm " + likelihood.temp_dir + "/flagged_comp" + str(comp) + "_*.dat"
        os.system(string)

    comm.Barrier()
    return chi2
