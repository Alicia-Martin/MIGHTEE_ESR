import numpy as np
import sympy
from scipy.optimize import minimize
import itertools
import jax.numpy as jnp
from esr.fitting.dm_likelihood import MIGHTEELikelihood
import esr.generation.simplifier as simplifier

import scipy
from sympy import *
import sympy
import matplotlib.pyplot as plt
import jax.numpy as jnp

import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.plot
from esr.fitting.sympy_symbols import *
from scipy.optimize import direct

# -------------------------------------------------
# Configuración
# -------------------------------------------------
LOG_OPT = True
USE_SIGN_COMBINATIONS = True
DATA_FILE = "rar_with_ml_direct_phot.csv"
GALAXY_NAME = "J022128.8-042448"
# GALAXY_NAME = "J021719.4-052851"
RUN_NAME = "test_mightee"

# FUNCTIONS_TO_TRY = [
#     # "a0/(1 + x/a1)",
#     # "a0*exp(-x/a1)",
#     # "a0/(x + a1)",
#     # "a0/(1 + pow(x, a1))",
#     # "a0*pow(x, a1)",
#     "a0/(x*(a1 + x)**2)",
# ]

FUNCTIONS_TO_TRY = [
    # "a0/(pow(a1+x,2)*x)",
    # "a0/(pow(a1+x,a2)*x)",
    # "a0/((a1+x)**a2*x)",
    # "a0/(x*(a1+x)*(a2+x))",
    "a0*pow(x,a1)/(a2+x)",
]

METHODS_TO_TRY = [
    # "Nelder-Mead",
    # "Powell",
    "DIRECT+Nelder-Mead",
]

TRY_INTEGRATION = False
N_STARTS = 5
USE_SIGN_COMBINATIONS = True
MAX_PARAM = 4

# -------------------------------------------------
# Configuración direct
# -------------------------------------------------

USE_DIRECT_PIPELINE = True

DIRECT_MAXFUN = 20000
DIRECT_MAXITER = 2000
DIRECT_BOUND_LIMIT = 30
# DIRECT_BOUND_LIMIT = [(8, 12), (1, 5), (1, 3)]  # per parameter bounds for DIRECT when log_opt=True

HYBRID_GLOBAL_N = 10
GLOBAL_POOL_DELTA_LL = 10.0
GLOBAL_POOL_SEED = 42
START_MIN_SEPARATION = 1e-6

# -------------------------------------------------
# Stage 2: distance/inclination optimization (post-DIRECT)
# -------------------------------------------------
# Inc and D are deliberately NOT part of the bounded DIRECT search above --
# they are optimized in a second, unbounded stage after the shape parameters
# are fixed by DIRECT+polish, mirroring CLASH_SPARC's two-stage structure
# (bounded DIRECT over shape params, then an unbounded joint Nelder-Mead
# polish over shape+galaxy params -- see CLASH_SPARC/direct_fit.py's res_7d_i).
FIT_GALAXY_PARAMS_STAGE2 = True


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def make_direct_bounds(nparam, log_opt=True, direct_bound_limit=10.0):
    """
    Build bounds for DIRECT.

    direct_bound_limit can be either:
      - a single number, e.g. 10
        -> [(-10, 10), (-10, 10), ...]
      - a list of tuples, e.g. [(8, 15), (1, 10), (-2, 2)]
        -> one bound per parameter
    """
    if nparam == 0:
        return []

    if log_opt:
        if isinstance(direct_bound_limit, (list, tuple, np.ndarray)):
            bounds = [tuple(b) for b in direct_bound_limit]

            if len(bounds) != nparam:
                raise ValueError(
                    f"DIRECT_BOUND_LIMIT has {len(bounds)} bounds, "
                    f"but this function has {nparam} parameters."
                )

            return bounds

        return [(-float(direct_bound_limit), float(direct_bound_limit))] * nparam

    else:
        if isinstance(direct_bound_limit, (list, tuple, np.ndarray)):
            bounds = [tuple(b) for b in direct_bound_limit]

            if len(bounds) != nparam:
                raise ValueError(
                    f"DIRECT_BOUND_LIMIT has {len(bounds)} bounds, "
                    f"but this function has {nparam} parameters."
                )

            return bounds

        return [(-10.0, 10.0)] * nparam

def build_equation(eq_string, nparam):
    """
    Parse the ESR string and build a JAX-lambdified function.
    """
    eq_string = eq_string.replace("\n", "").replace("'", "")

    eq = sympy.sympify(
        eq_string,
        locals={
            "inv": inv,
            "square": square,
            "cube": cube,
            "sqrt": sqrt,
            "log": log,
            "exp": exp,
            "pow": pow,
            "x": x,
            "a0": a0,
            "a1": a1,
            "a2": a2,
        }
    )

    if nparam > 1:
        all_a = list(sympy.symbols(" ".join([f"a{i}" for i in range(nparam)]), real=True))
        eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    elif nparam == 1:
        eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
    else:
        eq_numpy = sympy.lambdify(x, eq, modules=["jax"])

    return eq, eq_numpy

def lambdify_equation(eq, nparam):
    if nparam > 1:
        all_a = list(sympy.symbols(" ".join([f"a{i}" for i in range(nparam)]), real=True))
        eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["jax"])
    elif nparam == 1:
        eq_numpy = sympy.lambdify([x, a0], eq, modules=["jax"])
    else:
        eq_numpy = sympy.lambdify(x, eq, modules=["jax"])

    return eq_numpy


def random_initial_guess(nparam, low=-2.0, high=2.0):
    return np.random.uniform(low, high, size=nparam)


def run_scipy_optimize(chi2_fcn, x0, method, likelihood, signs=None):
    args = (
        likelihood.xvar,
        likelihood.yvar,
        likelihood.yerr_lo,
        likelihood.yerr_hi,
        signs,
    )

    if method in ["BFGS", "L-BFGS-B", "CG", "Newton-CG", "trust-constr"]:
        res = minimize(
            fun=lambda p: float(chi2_fcn(p, *args)[0]),
            x0=x0,
            jac=lambda p: np.array(chi2_fcn(p, *args)[1]),
            method=method,
            options={"maxiter": 2000}
        )
    else:
        res = minimize(
            fun=lambda p: float(chi2_fcn(p, *args)[0]),
            x0=x0,
            method=method,
            options={"maxiter": 2000}
        )

    return res

def generate_starts(nparam, n_starts, seed=0, low=-2.0, high=2.0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(low, high, size=nparam) for _ in range(n_starts)]


def best_of_many_starts(chi2_fcn, starts, method, likelihood, log_opt=False, use_signs=True):
    best_res = None
    best_fun = np.inf
    best_signs = None

    # inside best_of_many_starts(...)
    nparam = len(starts[0])
    if log_opt and use_signs and nparam > 0:
        sign_list = list(itertools.product([1, -1], repeat=nparam))
    else:
        sign_list = [None]


    for signs in sign_list:
        for x0_raw in starts:
            x0 = np.array(x0_raw, copy=True)

            print(f"Trying start: {x0} | signs = {signs} | method = {method} | log_opt = {log_opt}")

            res = run_scipy_optimize(
                chi2_fcn=chi2_fcn,
                x0=x0,
                method=method,
                likelihood=likelihood,
                signs=signs
            )

            if res.fun < best_fun:
                best_fun = res.fun
                best_res = res
                best_signs = signs

    best_res.best_signs = best_signs
    return best_res


# -------------------------------------------------
# Main fitting loop
# -------------------------------------------------
# def fit_one_function(likelihood, fcn_string, method="BFGS", try_integration=True, n_starts=5):
#     """
#     Fit one ESR function to one galaxy.
#     """
#     nparam = simplifier.count_params([fcn_string], MAX_PARAM)[0]

#     fcn_string, eq, integrated = likelihood.run_sympify(
#         fcn_string,
#         try_integration=try_integration
#     )

#     # Build symbolic function/lambdified version
#     _, eq_numpy = build_equation(fcn_string, nparam)

#     # Loss with asymmetric errors and density positivity check
#     loss_template = likelihood.get_loss(eq_numpy, integrated, value="value_and_grad")

#     def loss_fn(params):
#         return loss_template(
#             jnp.array(params),
#             likelihood.xvar,
#             likelihood.yvar,
#             likelihood.yerr_lo,
#             likelihood.yerr_hi
#         )

#     # Optimize
#     res = best_of_many_starts(
#         loss_fn=loss_fn,
#         nparam=nparam,
#         method=method,
#         n_starts=n_starts,
#         use_signs=USE_SIGN_COMBINATIONS
#     )

#     return {
#         "function": fcn_string,
#         "method": method,
#         "nparam": nparam,
#         "integrated": integrated,
#         "chi2": float(res.fun),
#         "params": np.array(res.x),
#         "success": bool(res.success),
#         "message": res.message if hasattr(res, "message") else "",
#     }

def fit_one_function(likelihood, fcn_string, method="BFGS", try_integration=True, n_starts=5, log_opt=False):
    """
    Fit one ESR function to one galaxy.
    """
    # nparam = simplifier.count_params([fcn_string], MAX_PARAM)[0]
    nshape = simplifier.count_params([fcn_string], MAX_PARAM)[0]
    n_extra = 2 if getattr(likelihood, "use_physical_scale", False) else 0
    n_active = nshape + n_extra

    fcn_string, eq, integrated = likelihood.run_sympify(
        fcn_string,
        try_integration=try_integration
    )

    # _, eq_numpy = build_equation(fcn_string, nparam)
    eq_numpy = lambdify_equation(eq, nshape)

    loss_template = likelihood.get_loss(eq_numpy, integrated, value="value_and_grad")
    chi2_fcn = likelihood.get_wrapped_like(loss_template)

    starts = generate_starts(n_active, n_starts, seed=0, low=-10.0, high=10.0)

    res = best_of_many_starts(
        chi2_fcn=chi2_fcn,
        starts=starts,
        method=method,
        likelihood=likelihood,
        log_opt=log_opt,
        use_signs=USE_SIGN_COMBINATIONS,
    )

    if log_opt and res.best_signs is not None:
        active_params = np.array(res.best_signs) * 10.0**np.array(res.x)
    else:
        active_params = np.array(res.x)

    full_params = np.zeros(MAX_PARAM + n_extra)
    full_params[:nshape] = active_params[:nshape]
    if n_extra > 0:
        full_params[MAX_PARAM:MAX_PARAM + n_extra] = active_params[nshape:nshape + n_extra]

    return {
        "function": fcn_string,
        "method": method,
        "nparam": nparam,
        "integrated": integrated,
        "chi2": float(res.fun),
        "params": active_params,      # compact active vector
        "full_params": full_params,   # shape | pad | rho0, rs
        "success": bool(res.success),
        "message": res.message if hasattr(res, "message") else "",
    }


def fit_many_functions(likelihood, functions, methods, fit_galaxy_params=False):
    results = []

    for fcn in functions:
        for method in methods:
            print(f"\nFitting: {fcn} | method = {method}")

            if method == "DIRECT+Nelder-Mead":
                out = fit_one_function_direct_pipeline(
                    likelihood=likelihood,
                    fcn_string=fcn,
                    try_integration=TRY_INTEGRATION,
                    log_opt=LOG_OPT,
                    use_signs=USE_SIGN_COMBINATIONS,
                    polish_method="Nelder-Mead",
                )
                if fit_galaxy_params:
                    out = fit_galaxy_params_after_direct(
                        likelihood=likelihood,
                        best_result=out,
                        try_integration=TRY_INTEGRATION,
                        polish_method="Nelder-Mead",
                    )
            else:
                out = fit_one_function(
                    likelihood=likelihood,
                    fcn_string=fcn,
                    method=method,
                    try_integration=TRY_INTEGRATION,
                    n_starts=N_STARTS,
                    log_opt=LOG_OPT,
                )

            results.append(out)

            print("  chi2 =", out["chi2"])
            print("  signs =", out.get("signs"))
            print("  params =", out["params"])

    return results

def physical_params_from_internal(p_internal, signs=None, log_opt=False):
    """
    Convert an internal DIRECT/Nelder-Mead search vector into physical
    parameter values.

    If `signs` is shorter than `p_internal` (e.g. `signs` covers only the
    ESR shape parameters from get_sign_combinations, while `p_internal`
    additionally covers rho0/rs when likelihood.use_physical_scale is True),
    the missing trailing entries are padded with +1 -- mirroring
    MIGHTEELikelihood.get_wrapped_like's own auto-extension convention
    (rho0, rs are always forced positive via the sign convention, never
    sign-enumerated).
    """
    p_internal = np.asarray(p_internal, dtype=float)

    if log_opt:
        if signs is None:
            signs = np.ones_like(p_internal)
        else:
            signs = np.asarray(signs, dtype=float)
            if len(signs) < len(p_internal):
                pad = np.ones(len(p_internal) - len(signs), dtype=float)
                signs = np.concatenate([signs, pad])
            elif len(signs) != len(p_internal):
                raise ValueError(
                    f"signs has length {len(signs)}, longer than p_internal "
                    f"(length {len(p_internal)}); cannot broadcast."
                )
        return signs * 10.0**p_internal

    return p_internal.copy()


def sign_to_text(signs):
    if signs is None:
        return "None"
    return str(np.asarray(signs, dtype=int))


def get_sign_combinations(nparam, log_opt=True, use_signs=True):
    if (not log_opt) or (not use_signs) or nparam == 0:
        return [None]
    return [np.array(s, dtype=float) for s in itertools.product([1.0, -1.0], repeat=nparam)]


def get_internal_point(sample):
    return np.asarray(sample["p_internal"], dtype=float)


def pick_global_pool_starts(samples, k, delta_ll, seed, include_best=True):
    finite = [s for s in samples if np.isfinite(s["fun"])]

    if len(finite) == 0:
        return []

    finite = sorted(finite, key=lambda s: s["fun"])
    best_fun = finite[0]["fun"]
    pool = [s for s in finite if s["fun"] <= best_fun + delta_ll]

    selected = []

    if include_best:
        selected.append(pool[0])

    rng = np.random.default_rng(seed)
    remaining = pool[1:]

    need = k - len(selected)
    if need > 0 and len(remaining) > 0:
        if need >= len(remaining):
            selected.extend(remaining)
        else:
            idx = rng.choice(len(remaining), size=need, replace=False)
            selected.extend([remaining[i] for i in idx])

    return selected[:k]


def add_unique_start(dest, sample, min_sep, verbose=True):
    p = get_internal_point(sample)

    for existing in dest:
        p0 = get_internal_point(existing)
        dist = np.linalg.norm(p - p0)

        if dist < min_sep:
            if verbose:
                print("Rejected near-duplicate start:")
                print("  new source =", sample.get("source", "unknown"))
                print("  old source =", existing.get("source", "unknown"))
                print("  distance   =", dist)
                print("  new nll    =", sample.get("fun"))
                print("  old nll    =", existing.get("fun"))
                print("  new p      =", p)
                print("  old p      =", p0)
            return False

    dest.append(sample)

    if verbose:
        print("Accepted start:")
        print("  source =", sample.get("source", "unknown"))
        print("  nll    =", sample.get("fun"))
        print("  p      =", p)

    return True

def fit_one_function_direct_pipeline(
    likelihood,
    fcn_string,
    try_integration=True,
    log_opt=True,
    use_signs=True,
    polish_method="Nelder-Mead",
):
    nparam = simplifier.count_params([fcn_string], MAX_PARAM)[0]

    fcn_string, eq, integrated = likelihood.run_sympify(
        fcn_string,
        try_integration=try_integration
    )

    # `nparam` (shape-params-only, from the function string) stays the
    # lambdify/sign-enumeration count. The DIRECT/polish SEARCH
    # dimensionality additionally covers rho0, rs when
    # likelihood.use_physical_scale is True -- they are core physical
    # parameters (uninformed, can span orders of magnitude, just like a0),
    # not external nuisance parameters like Inc/D (which get their own,
    # later, unbounded polish stage in fit_galaxy_params_after_direct).
    # likelihood.nparam already equals nparam + likelihood.nparam_extra
    # (0 or 2), set by run_sympify.
    assert likelihood.nparam_shape == nparam, (
        f"likelihood.nparam_shape ({likelihood.nparam_shape}) != local shape "
        f"param count ({nparam}); MAX_PARAM here and run_sympify's internal "
        f"max_param=4 must stay in sync."
    )
    n_search = likelihood.nparam

    eq_numpy = lambdify_equation(eq, nparam)

    loss_eval = likelihood.get_loss(eq_numpy, integrated, value="evaluate")
    wrapped_eval = likelihood.get_wrapped_like(loss_eval)

    sign_combos = get_sign_combinations(
        nparam,
        log_opt=log_opt,
        use_signs=use_signs
    )

    # if log_opt:
    #     bounds = [(-DIRECT_BOUND_LIMIT, DIRECT_BOUND_LIMIT)] * nparam
    # else:
    #     bounds = [(-10.0, 10.0)] * nparam

    # Bounds cover the full search dimensionality (n_search), so rho0/rs get
    # the same generic box as the shape parameters when use_physical_scale
    # is True. Rho0/rs are always forced positive via the sign convention
    # (see get_sign_combinations above, scoped to `nparam` only, and
    # MIGHTEELikelihood.get_wrapped_like's own auto-extension of a
    # shape-only `signs` list with [1, 1]) -- never sign-enumerated.
    bounds = make_direct_bounds(
        n_search,
        log_opt=log_opt,
        direct_bound_limit=DIRECT_BOUND_LIMIT,
    )

    print("    DIRECT bounds =", bounds)
    print(f"    search dimensionality = {n_search} (shape={nparam}, extra={n_search - nparam})")

    global_samples = []
    direct_results = []

    print("\n  DIRECT search")

    for i, signs in enumerate(sign_combos, start=1):
        print(f"    sign combo {i}/{len(sign_combos)}: {sign_to_text(signs)}")

        def obj_direct(p_internal):
            val = wrapped_eval(
                np.asarray(p_internal, dtype=float),
                likelihood.xvar,
                likelihood.yvar,
                likelihood.yerr_lo,
                likelihood.yerr_hi,
                signs=signs,
                check_nans=True,
            )

            val = float(val)
            val_safe = val if np.isfinite(val) else 1e30

            global_samples.append({
                "fun": val_safe,
                "p_internal": np.asarray(p_internal, dtype=float).copy(),
                "signs": signs,
            })

            return val_safe

        res_direct = direct(
            obj_direct,
            bounds,
            maxfun=DIRECT_MAXFUN,
            maxiter=DIRECT_MAXITER,
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

    print("    best DIRECT nll =", best_direct["fun"])
    print("    best DIRECT signs =", sign_to_text(best_direct["signs"]))
    print(
        "    best DIRECT physical params =",
        physical_params_from_internal(
            best_direct["p_internal"],
            best_direct["signs"],
            log_opt=log_opt,
        )
    )

    # Choose polish starts from good DIRECT samples
    hybrid_starts = []

    global_candidates = pick_global_pool_starts(
        global_samples,
        HYBRID_GLOBAL_N,
        GLOBAL_POOL_DELTA_LL,
        GLOBAL_POOL_SEED,
        include_best=True,
    )

    finite_samples = [s for s in global_samples if np.isfinite(s["fun"])]
    best_direct_fun = min(s["fun"] for s in finite_samples)

    pool = [
        s for s in finite_samples
        if s["fun"] <= best_direct_fun + GLOBAL_POOL_DELTA_LL
    ]

    print("\nDIRECT start diagnostics:")
    print("  total DIRECT samples  =", len(global_samples))
    print("  finite DIRECT samples =", len(finite_samples))
    print("  best DIRECT nll       =", best_direct_fun)
    print("  GLOBAL_POOL_DELTA_LL  =", GLOBAL_POOL_DELTA_LL)
    print("  samples in pool       =", len(pool))
    print("  requested global N    =", HYBRID_GLOBAL_N)
    print("  direct_results        =", len(direct_results))
    print("  sign combos           =", len(sign_combos))
    print("  min separation        =", START_MIN_SEPARATION)

    for sample in global_candidates:
        sample = dict(sample)
        sample["source"] = "global"
        add_unique_start(hybrid_starts, sample, START_MIN_SEPARATION)

    # Also include best endpoint per sign combo
    for sample in direct_results[:len(sign_combos)]:
        sample = dict(sample)
        sample["source"] = "per-sign"
        add_unique_start(hybrid_starts, sample, START_MIN_SEPARATION)

    print(f"    polishing {len(hybrid_starts)} starts with {polish_method}")

    print("\nFinal hybrid starts:")
    for i, s in enumerate(hybrid_starts):
        print(
            i,
            "source =", s.get("source", "unknown"),
            "nll =", s.get("fun"),
            "p =", s.get("p_internal"),
            "signs =", s.get("signs"),
        )
    print("Total hybrid starts =", len(hybrid_starts))

    best_polish = None
    best_fun = np.inf
    best_signs = None
    best_nm_path = None

    for j, start in enumerate(hybrid_starts, start=1):
        nm_path = []

        def callback_nm(xk):
            nm_path.append(np.asarray(xk, dtype=float).copy())
        p0 = start["p_internal"]
        signs = start["signs"]

        print(
            f"      polish start {j}/{len(hybrid_starts)} "
            f"nll={start['fun']:.6f} signs={sign_to_text(signs)}"
        )

        def obj_polish(p_internal):
            val = wrapped_eval(
                np.asarray(p_internal, dtype=float),
                likelihood.xvar,
                likelihood.yvar,
                likelihood.yerr_lo,
                likelihood.yerr_hi,
                signs=signs,
                check_nans=True,
            )

            val = float(val)
            return val if np.isfinite(val) else 1e30

        res_polish = minimize(
            obj_polish,
            p0,
            method=polish_method,
            callback=callback_nm,
            options={"maxiter": 5000}
        )

        print("        polish nll =", res_polish.fun)

        if np.isfinite(res_polish.fun) and res_polish.fun < best_fun:
            best_fun = float(res_polish.fun)
            best_polish = res_polish
            best_signs = signs
            best_nm_path = np.array(nm_path)

    if best_polish is None:
        raise RuntimeError("DIRECT pipeline failed: no finite polished result.")

    physical_params = physical_params_from_internal(
        best_polish.x,
        best_signs,
        log_opt=log_opt,
    )

    return {
        "function": fcn_string,
        "method": f"DIRECT+{polish_method}",
        "nparam": nparam,
        "integrated": integrated,
        "log_opt": log_opt,
        "signs": best_signs,
        "chi2": best_fun,
        "opt_params": np.asarray(best_polish.x),          # Nelder-Mead internal params
        "params": physical_params,
        "success": bool(best_polish.success),
        "message": best_polish.message if hasattr(best_polish, "message") else "",
        "direct_chi2": best_direct["fun"],
        "direct_opt_params": np.asarray(best_direct["p_internal"]),   # DIRECT internal params
        "direct_params": physical_params_from_internal(
            best_direct["p_internal"],
            best_direct["signs"],
            log_opt=log_opt,
        ),
        "direct_samples": global_samples,
        "nm_path": best_nm_path,
    }


def fit_galaxy_params_after_direct(
    likelihood,
    best_result,
    try_integration=True,
    polish_method="Nelder-Mead",
):
    """
    Second-stage joint polish over [model params..., Inc, D], run AFTER the
    bounded DIRECT/log-space search (fit_one_function_direct_pipeline) has
    already found the best-fit shape parameters. This stage is plain,
    unbounded, physical-space Nelder-Mead -- it does NOT go through
    likelihood.get_wrapped_like's sign*10**x log-space convention, since Inc
    and D are ordinary real-valued nuisance parameters, not ESR shape
    parameters. Their truncated-Gaussian priors (see
    MIGHTEELikelihood.get_loss(include_priors=True)) are what keep them
    identifiable/well-behaved in this unbounded polish.
    """
    fcn_string = best_result["function"]
    fcn_string, eq, integrated = likelihood.run_sympify(
        fcn_string,
        try_integration=try_integration
    )
    eq_numpy = lambdify_equation(eq, likelihood.nparam_shape)

    model_params0 = np.asarray(best_result["params"], dtype=float)
    if model_params0.shape[0] != likelihood.nparam:
        raise ValueError(
            f"Stage-1 'params' has length {model_params0.shape[0]}, but "
            f"likelihood.nparam={likelihood.nparam}; cannot build the joint "
            f"[model params..., Inc, D] vector for stage 2. This is expected "
            f"(pre-existing, unrelated to Inc/D) when likelihood.use_physical_scale "
            f"is True: fit_one_function_direct_pipeline's DIRECT search only ever "
            f"covers the ESR shape parameters (counted from the function string), "
            f"never the extra rho0/rs physical-scale parameters, so 'params' comes "
            f"back short by likelihood.nparam_extra. Use use_physical_scale=False, "
            f"or fix the DIRECT search's dimensionality separately, before running "
            f"stage 2 in physical-scale mode."
        )

    x0 = np.concatenate([model_params0, [likelihood.inc_true, likelihood.distance_true]])

    loss_eval = likelihood.get_loss(eq_numpy, integrated, value="evaluate", include_priors=True)

    def obj(a):
        val = float(loss_eval(
            jnp.asarray(a, dtype=jnp.float64),
            likelihood.xvar, likelihood.yvar, likelihood.yerr_lo, likelihood.yerr_hi,
        ))
        return val if np.isfinite(val) else 1e30

    res = minimize(obj, x0, method=polish_method, options={"maxiter": 5000})

    n_model = likelihood.nparam
    full_params = np.asarray(res.x, dtype=float)
    model_params = full_params[:n_model]
    inc_fit = float(full_params[n_model])
    d_fit = float(full_params[n_model + 1])

    print("\n  Stage 2: joint [model params, Inc, D] polish")
    print("    x0      =", x0)
    print("    result  =", full_params)
    print(f"    chi2    = {res.fun:.6f} (stage 1 was {best_result['chi2']:.6f})")
    print(f"    Inc fit = {inc_fit:.4f} (catalog {likelihood.inc_true:.4f} +/- {likelihood.e_inc:.4f})")
    print(f"    D fit   = {d_fit:.4f} (catalog {likelihood.distance_true:.4f} +/- {likelihood.e_d:.4f})")

    out = dict(best_result)
    out["stage1_result"] = best_result
    out["stage1_chi2"] = best_result["chi2"]
    out["chi2"] = float(res.fun)
    out["model_params"] = model_params
    out["params"] = model_params  # keep "params" model-only, so existing get_pred()/plot call sites need no changes
    out["full_params_with_galaxy"] = full_params
    out["inc"] = inc_fit
    out["distance"] = d_fit
    out["success"] = bool(res.success)
    out["message"] = res.message if hasattr(res, "message") else ""
    out["method"] = f"{best_result.get('method', '')}+galaxy_params"
    return out


def plot_likelihood_corner_grid(
    best_result,
    likelihood,
    try_integration=True,
    log_opt=True,
    grid_n=80,
    spans=0.5,
    delta_clip=100.0,
    contour_levels=(1, 2, 5, 10, 25, 50),
    outfile=None,
):
    """
    Pairwise likelihood panels for any number of parameters.

    This plots one panel per unique parameter pair:
        (a0,a1), (a0,a2), ..., (a_{n-2}, a_{n-1})

    So:
      nparam=2 -> 1 panel
      nparam=3 -> 3 panels
      nparam=4 -> 6 panels
      etc.
    """
    fcn_string = best_result["function"]
    signs = best_result.get("signs", None)
    p_best = np.asarray(best_result["opt_params"], dtype=float)
    p_direct = np.asarray(best_result.get("direct_opt_params", p_best), dtype=float)
    direct_samples = best_result.get("direct_samples", [])
    nm_path = best_result.get("nm_path", None)

    nparam = simplifier.count_params([fcn_string], MAX_PARAM)[0]
    if nparam < 2:
        print("Pairwise grid skipped: need at least 2 parameters.")
        return

    fcn_string, eq, integrated = likelihood.run_sympify(
        fcn_string,
        try_integration=try_integration
    )
    eq_numpy = lambdify_equation(eq, nparam)

    loss_eval = likelihood.get_loss(eq_numpy, integrated, value="evaluate")
    wrapped_eval = likelihood.get_wrapped_like(loss_eval)

    def nll_from_internal(p_internal):
        val = wrapped_eval(
            np.asarray(p_internal, dtype=float),
            likelihood.xvar,
            likelihood.yvar,
            likelihood.yerr_lo,
            likelihood.yerr_hi,
            signs=signs,
            check_nans=True,
        )
        val = float(val)
        return val if np.isfinite(val) else np.nan

    if np.isscalar(spans):
        spans = np.full(nparam, float(spans))
    else:
        spans = np.asarray(spans, dtype=float)
        if len(spans) != nparam:
            raise ValueError(f"spans has length {len(spans)}, but nparam={nparam}")

    nll_best = float(best_result["chi2"])
    print("\nPairwise likelihood grid")
    print("  nll best =", nll_best)
    print("  p_best internal =", p_best)

    pair_list = list(itertools.combinations(range(nparam), 2))
    n_pairs = len(pair_list)

    ncols = min(3, n_pairs)   # compact layout
    nrows = int(np.ceil(n_pairs / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.8 * nrows),
    )
    axes = np.atleast_1d(axes).ravel()

    # DIRECT samples with same sign choice
    same_sign_samples = []
    if len(direct_samples) > 0:
        if signs is None:
            same_sign_samples = direct_samples
        else:
            same_sign_samples = [
                s for s in direct_samples
                if s.get("signs") is not None
                and np.all(np.asarray(s["signs"]) == np.asarray(signs))
            ]

    if len(same_sign_samples) > 0:
        direct_pts = np.array([s["p_internal"] for s in same_sign_samples], dtype=float)
        direct_vals = np.array([s["fun"] for s in same_sign_samples], dtype=float)
        good_direct = np.isfinite(direct_vals) & (direct_vals < 1e29)
        direct_pts = direct_pts[good_direct]
        direct_vals = direct_vals[good_direct]
    else:
        direct_pts = np.empty((0, nparam))
        direct_vals = np.empty(0)

    best_grid_val = np.inf
    best_grid_p = None
    cf = None

    for ax, (i, j) in zip(axes, pair_list):
        xgrid = np.linspace(p_best[i] - spans[i], p_best[i] + spans[i], grid_n)
        ygrid = np.linspace(p_best[j] - spans[j], p_best[j] + spans[j], grid_n)

        # ax.set_xlim(xgrid.min(), xgrid.max())
        # ax.set_ylim(ygrid.min(), ygrid.max())

        Z = np.full((grid_n, grid_n), np.nan, dtype=float)
        p_trial = p_best.copy()

        for yy_i, yy in enumerate(ygrid):
            for xx_i, xx in enumerate(xgrid):
                p_trial[:] = p_best
                p_trial[i] = xx
                p_trial[j] = yy
                Z[yy_i, xx_i] = nll_from_internal(p_trial)

        finite = np.isfinite(Z)
        if np.any(finite):
            local_idx = np.nanargmin(Z)
            iy, ix = np.unravel_index(local_idx, Z.shape)
            local_best = Z[iy, ix]

            if local_best < best_grid_val:
                best_grid_val = local_best
                best_grid_p = p_best.copy()
                best_grid_p[i] = xgrid[ix]
                best_grid_p[j] = ygrid[iy]

        dZ = Z - nll_best
        dZ_plot = np.minimum(dZ, delta_clip)

        cf = ax.contourf(
            xgrid,
            ygrid,
            dZ_plot,
            levels=np.linspace(0, delta_clip, 40),
            cmap="viridis",
        )

        try:
            cs = ax.contour(
                xgrid,
                ygrid,
                dZ,
                levels=contour_levels,
                colors="white",
                linewidths=0.6,
            )
            ax.clabel(cs, fmt="%.0f", fontsize=6)
        except Exception:
            pass

        if len(direct_pts) > 0:
            ax.scatter(
                direct_pts[:, i],
                direct_pts[:, j],
                color="red",
                s=3,
                alpha=0.12,
                edgecolors="none",
                rasterized=True,
            )

        if nm_path is not None and len(nm_path) > 0:
            ax.plot(
                nm_path[:, i],
                nm_path[:, j],
                "-",
                color="red",
                lw=1.0,
                alpha=0.8,
            )
            ax.scatter(
                nm_path[0, i],
                nm_path[0, j],
                marker="o",
                s=35,
                color="red",
                edgecolors="black",
            )

        ax.plot(p_direct[i], p_direct[j], "ws", ms=5, mec="k")
        ax.plot(p_best[i], p_best[j], "r*", ms=10)

        ax.set_xlabel(f"internal a{i}" if log_opt else f"a{i}")
        ax.set_ylabel(f"internal a{j}" if log_opt else f"a{j}")
        ax.set_title(f"a{i} vs a{j}")

    # Hide any extra empty axes if n_pairs does not fill the grid
    for ax in axes[n_pairs:]:
        ax.axis("off")

    if cf is not None:
        cbar = fig.colorbar(cf, ax=axes[:n_pairs].tolist(), shrink=0.8, pad=0.02)
        cbar.set_label(r"$\Delta$NLL clipped")

    # fig.suptitle(f"{GALAXY_NAME}", y=0.98)
    # fig.subplots_adjust(top=0.92)

    print("\nBest value found on plotted grid:")
    print("  grid nll =", best_grid_val)
    print("  grid ΔNLL =", best_grid_val - nll_best)
    if best_grid_p is not None:
        print("  grid internal params =", best_grid_p)
        print(
            "  grid physical params =",
            physical_params_from_internal(best_grid_p, signs=signs, log_opt=log_opt)
        )

    print("\nReference:")
    print("  polished nll =", best_result["chi2"])
    print("  polished internal =", p_best)
    print("  polished physical =", best_result["params"])
    print("  DIRECT nll =", best_result.get("direct_chi2"))
    print("  DIRECT internal =", p_direct)
    print("  DIRECT physical =", best_result.get("direct_params"))

    if outfile is not None:
        fig.savefig(outfile)

    # plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Example usage
# -------------------------------------------------
likelihood = MIGHTEELikelihood(
    data_file=DATA_FILE,
    name=GALAXY_NAME,
    run_name=RUN_NAME,
    use_physical_scale=True,
)

results = fit_many_functions(
    likelihood, FUNCTIONS_TO_TRY, METHODS_TO_TRY, fit_galaxy_params=FIT_GALAXY_PARAMS_STAGE2
)

# Best result
best = min(results, key=lambda d: d["chi2"])
# print("\nBEST:")
# print(best)

# -------------------------------------------------
# Plot best fit against data: velocity + density
# -------------------------------------------------

best_function = best["function"]
best_params = np.asarray(best["params"])

nparam = simplifier.count_params([best_function], MAX_PARAM)[0]

# Rebuild the integrated expression for the velocity model
fcn_string, eq_int, integrated = likelihood.run_sympify(
    best_function,
    try_integration=TRY_INTEGRATION
)
eq_numpy_int = lambdify_equation(eq_int, nparam)

# Rebuild the ORIGINAL density expression for the density plot
eq_density, eq_numpy_density = build_equation(best_function, nparam)

# Model predictions
v_model = likelihood.get_pred(
    likelihood.xvar,
    best_params,
    eq_numpy_int,
    integrated=integrated,
    D=best.get("distance"),
    Inc=best.get("inc"),
)

r = np.asarray(likelihood.xvar)
vobs = np.asarray(likelihood.yvar)
err_lo = np.asarray(likelihood.yerr_lo)
err_hi = np.asarray(likelihood.yerr_hi)

v_model = np.asarray(v_model)

if nparam == 1:
    rho_model = eq_numpy_density(likelihood.xvar, best_params[0])
elif nparam == 2:
    rho_model = eq_numpy_density(likelihood.xvar, best_params[0], best_params[1])
elif nparam == 3:
    rho_model = eq_numpy_density(likelihood.xvar, best_params[0], best_params[1], best_params[2])
else:
    rho_model = eq_numpy_density(likelihood.xvar, *best_params)

# print(eq_density)
# print('rho_model', rho_model)
# print((best_params[1] + r))
# print((best_params[1] + r)**best_params[2])
# print(best_params[0]/((best_params[1] + r)**best_params[2]))

rho_model = np.asarray(rho_model)

order = np.argsort(r)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: velocity
ax = axes[0]
ax.errorbar(
    r,
    vobs,
    yerr=[err_lo, err_hi],
    fmt="o",
    capsize=3,
    color="black",
    label="Data"
)
ax.plot(
    r[order],
    v_model[order],
    lw=2,
    label="Best-fit ESR"
)
ax.set_xlabel("Radius (kpc)")
ax.set_ylabel("Velocity (km/s)")
ax.set_title("Velocity profile")
ax.legend()

# Right: density
ax = axes[1]
ax.plot(
    r[order],
    rho_model[order],
    lw=2,
    label="Best-fit ESR density"
)
ax.set_xlabel("Radius (kpc)")
ax.set_ylabel("Density")
ax.set_yscale("log")
ax.set_title("Density profile")
ax.legend()

fig.suptitle(
    f"{GALAXY_NAME}\n"
    f"{best_function}\n"
    f"NLL = {best['chi2']:.3f}",
    y=1.05
)

fig.tight_layout()
fig.savefig(f"best_fit_{GALAXY_NAME}.pdf", bbox_inches="tight")
plt.show()

plot_likelihood_corner_grid(
    best_result=best,
    likelihood=likelihood,
    try_integration=TRY_INTEGRATION,
    log_opt=LOG_OPT,
    grid_n=80,
    spans=0.8,
    delta_clip=30.0,
    outfile=f"likelihood_pairwise_grid_{GALAXY_NAME}.pdf",
)