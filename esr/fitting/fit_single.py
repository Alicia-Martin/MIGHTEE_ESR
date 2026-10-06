
#!/usr/bin/env python3
"""
fit_single.py

Single-function fitting + codelength evaluation for ESR-style models.

Parameter layout used throughout:
    [shape params up to 4 | zero padding to 4 | rho0, rs]

If use_physical_scale=False:
    active parameters are just the ESR shape parameters.

If use_physical_scale=True:
    the model is interpreted as
        rho(r) = rho0 * f(r / rs)
    and rho0, rs are appended at the end of the active parameter vector.

This file is intentionally self-contained, but it expects the user's existing
likelihood object to provide:
    - xvar, yvar, yerr_lo, yerr_hi
    - run_sympify(fcn_i, tmax=..., try_integration=...)
    - get_loss(eq_numpy, integrated, value=...)
    - get_wrapped_like(loss_template)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import sympy
import jax
import jax.numpy as jnp
from scipy.optimize import direct as scipy_direct
from scipy.optimize import minimize as scipy_minimize

from esr.fitting.sympy_symbols import *  # noqa: F401,F403
import esr.generation.simplifier as simplifier

warnings.filterwarnings("ignore")
jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# Layout helpers
# -----------------------------------------------------------------------------
def build_layout(nshape: int, max_fun_params: int = 4, n_extra: int = 0):
    """
    Return index arrays for the parameter layout:
        [active ESR shape params | padding up to max_fun_params | rho0, rs]
    """
    nshape = int(nshape)
    max_fun_params = int(max_fun_params)
    n_extra = int(n_extra)

    shape_idx = np.arange(0, nshape, dtype=int)
    pad_idx = np.arange(nshape, max_fun_params, dtype=int)
    extra_idx = np.arange(max_fun_params, max_fun_params + n_extra, dtype=int)
    active_idx = np.concatenate([shape_idx, extra_idx]) if n_extra > 0 else shape_idx.copy()
    return shape_idx, pad_idx, extra_idx, active_idx


def make_lambdified_eq(eq, nshape: int):
    """Lambdify the ESR shape f(x) with nshape parameters."""
    nshape = int(nshape)
    if nshape == 0:
        return sympy.lambdify([x], eq, modules=["jax"])
    elif nshape == 1:
        return sympy.lambdify([x, a0], eq, modules=["jax"])
    else:
        all_a = list(sympy.symbols(" ".join([f"a{i}" for i in range(nshape)]), real=True))
        return sympy.lambdify([x] + all_a, eq, modules=["jax"])


def make_direct_bounds(nparam: int, log_opt: bool = False, direct_bounds=None, pmin=-5, pmax=5):
    """Build DIRECT bounds."""
    nparam = int(nparam)
    if nparam <= 0:
        return []

    if direct_bounds is not None:
        bounds = [tuple(b) for b in direct_bounds]
        if len(bounds) != nparam:
            raise ValueError(f"direct_bounds has length {len(bounds)} but nparam={nparam}")
        return bounds

    if log_opt:
        return [(-float(pmax), float(pmax))] * nparam

    return [(float(pmin), float(pmax))] * nparam


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _to_numpy(x):
    try:
        return np.asarray(x, dtype=float)
    except Exception:
        return np.array(x, dtype=float)


def physical_params_from_internal(p_internal, signs=None, log_opt=False):
    """Convert internal optimiser coordinates to physical parameters."""
    p_internal = _to_numpy(p_internal)

    if log_opt:
        if signs is None:
            signs = np.ones_like(p_internal)
        signs = np.asarray(signs, dtype=float)
        return signs * (10.0 ** p_internal)

    return p_internal.copy()


def compact_to_full_params(theta_active: Sequence[float], nshape: int, max_fun_params: int = 4, n_extra: int = 0):
    """
    Convert compact active params [shape..., extras] to the full saved layout:
        [shape..., padding to 4, extras]
    """
    nshape = int(nshape)
    max_fun_params = int(max_fun_params)
    n_extra = int(n_extra)

    theta_active = _to_numpy(theta_active).ravel()
    full = np.zeros(max_fun_params + n_extra, dtype=float)

    # shape block
    if nshape > 0:
        full[:nshape] = theta_active[:nshape]

    # extra block
    if n_extra > 0:
        full[max_fun_params:max_fun_params + n_extra] = theta_active[nshape:nshape + n_extra]

    return full


def full_to_compact_params(full_params: Sequence[float], nshape: int, max_fun_params: int = 4, n_extra: int = 0):
    """Convert full layout params back to compact active layout."""
    nshape = int(nshape)
    max_fun_params = int(max_fun_params)
    n_extra = int(n_extra)

    full_params = _to_numpy(full_params).ravel()
    compact = np.zeros(nshape + n_extra, dtype=float)

    if nshape > 0:
        compact[:nshape] = full_params[:nshape]
    if n_extra > 0:
        compact[nshape:nshape + n_extra] = full_params[max_fun_params:max_fun_params + n_extra]

    return compact


def flat_upper_to_matrix(flat, n):
    """Reconstruct symmetric matrix from flattened upper triangle."""
    flat = _to_numpy(flat).ravel()
    n = int(n)
    out = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n)
    m = min(len(flat), len(iu[0]))
    out[iu[0][:m], iu[1][:m]] = flat[:m]
    out = out + np.triu(out, 1).T
    return out


def safe_cumtrapz(y, x):
    """Compatibility wrapper for scipy.integrate.cumulative_trapezoid/cumtrapz."""
    import scipy.integrate
    if hasattr(scipy.integrate, "cumulative_trapezoid"):
        return scipy.integrate.cumulative_trapezoid(y, x, initial=0.0)
    return scipy.integrate.cumtrapz(y, x, initial=0.0)


# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------
@dataclass
class FitResult:
    fcn: str
    success: bool
    log_opt: bool
    use_physical_scale: bool
    nshape: int
    n_extra: int
    negloglike: float
    codelen: float
    compact_params: np.ndarray
    full_params: np.ndarray
    Delta: np.ndarray
    fisher_diag: np.ndarray
    hessian: np.ndarray
    niter: int
    nstarts: int
    direct_fun: float
    direct_signs: Optional[Tuple[int, ...]]


def fit_single_function(
    fcn_i: str,
    likelihood,
    tmax: float = 60,
    try_integration: bool = False,
    log_opt: bool = False,
    max_fun_params: int = 4,
    pmin: float = -20,
    pmax: float = 20,
    direct_bounds=None,
    direct_maxfun: int = 100000,
    direct_maxiter: int = 10000,
    polish_maxiter: int = 5000,
) -> FitResult:
    """
    Fit a single candidate function and compute a codelength estimate.

    Returns a FitResult with both the compact active parameter vector and the
    full saved-layout vector.
    """
    xvar = likelihood.xvar
    yvar = likelihood.yvar
    yerr_lo = getattr(likelihood, "yerr_lo", getattr(likelihood, "yerr", None))
    yerr_hi = getattr(likelihood, "yerr_hi", yerr_lo)

    if yerr_lo is None:
        raise ValueError("Could not find yerr, yerr_lo, or yerr_hi on likelihood.")

    use_physical_scale = bool(getattr(likelihood, "use_physical_scale", False))
    n_extra = 2 if use_physical_scale else 0

    # parse expression
    fcn_i = fcn_i.replace("\n", "").replace("'", "")
    fcn_i, eq, integrated = likelihood.run_sympify(
        fcn_i,
        tmax=tmax,
        try_integration=try_integration,
    )

    # count shape parameters only
    nshape = simplifier.count_params([fcn_i], max_fun_params)[0]
    nparam = nshape + n_extra  # compact active length
    full_width = max_fun_params + n_extra

    # build lambdified shape
    eq_numpy = make_lambdified_eq(eq, nshape)

    # likelihood wrappers
    loss_eval = likelihood.get_loss(eq_numpy, integrated, value="evaluate")
    wrapped_eval = likelihood.get_wrapped_like(loss_eval)

    def fop(theta_internal):
        theta_internal = np.asarray(theta_internal, dtype=float)
        return wrapped_eval(
            theta_internal,
            xvar, yvar, yerr_lo, yerr_hi,
            signs=None,
            check_nans=True,
        )

    # If no active parameters at all, just return zeros
    if nparam == 0:
        full_params = np.zeros(full_width, dtype=float)
        Delta = np.zeros(full_width, dtype=float)
        return FitResult(
            fcn=fcn_i,
            success=True,
            log_opt=log_opt,
            use_physical_scale=use_physical_scale,
            nshape=nshape,
            n_extra=n_extra,
            negloglike=float(fop(np.array([]))),
            codelen=0.0,
            compact_params=np.zeros(0, dtype=float),
            full_params=full_params,
            Delta=Delta,
            fisher_diag=np.zeros(0, dtype=float),
            hessian=np.zeros((0, 0), dtype=float),
            niter=0,
            nstarts=0,
            direct_fun=float(fop(np.array([]))),
            direct_signs=None,
        )

    # optimizer bounds
    bounds = make_direct_bounds(
        nparam,
        log_opt=log_opt,
        direct_bounds=direct_bounds,
        pmin=pmin,
        pmax=pmax,
    )

    # sign branches
    if log_opt:
        if use_physical_scale:
            # Only branch over shape params; rho0 and rs stay positive (+1)
            sign_list = [tuple(s) + (1, 1) for s in itertools.product([1, -1], repeat=nshape)]
        else:
            sign_list = list(itertools.product([1, -1], repeat=nparam))
    else:
        sign_list = [None]

    best_direct = None
    best_polish = None
    best_fun = np.inf
    best_signs = None

    # global/direct phase
    for signs in sign_list:
        def obj_direct(p_internal):
            p_internal = np.asarray(p_internal, dtype=float)
            try:
                val = wrapped_eval(
                    p_internal,
                    xvar, yvar, yerr_lo, yerr_hi,
                    signs=signs,
                    check_nans=True,
                )
                val = float(val)
            except Exception:
                val = np.inf
            return val if np.isfinite(val) else 1e30

        res_direct = scipy_direct(
            obj_direct,
            bounds,
            maxfun=direct_maxfun,
            maxiter=direct_maxiter,
        )

        fun_val = float(res_direct.fun) if np.isfinite(res_direct.fun) else 1e30
        if best_direct is None or fun_val < best_direct["fun"]:
            best_direct = {
                "fun": fun_val,
                "x": np.asarray(res_direct.x, dtype=float),
                "signs": signs,
                "res": res_direct,
            }

    if best_direct is None:
        raise RuntimeError("DIRECT failed to return any result.")

    # polish phase
    p0 = np.asarray(best_direct["x"], dtype=float)
    signs = best_direct["signs"]

    def obj_polish(p_internal):
        p_internal = np.asarray(p_internal, dtype=float)
        try:
            val = wrapped_eval(
                p_internal,
                xvar, yvar, yerr_lo, yerr_hi,
                signs=signs,
                check_nans=True,
            )
            val = float(val)
        except Exception:
            val = np.inf
        return val if np.isfinite(val) else 1e30

    res_polish = scipy_minimize(
        obj_polish,
        p0,
        method="Nelder-Mead",
        options={"maxiter": polish_maxiter},
    )

    if np.isfinite(res_polish.fun):
        best_fun = float(res_polish.fun)
        best_polish = res_polish
        best_signs = signs
    else:
        best_fun = float(best_direct["fun"])
        best_polish = best_direct["res"]
        best_signs = best_direct["signs"]

    # convert optimizer variables to physical active parameters
    if log_opt:
        theta_active = physical_params_from_internal(best_polish.x, signs=best_signs, log_opt=True)
    else:
        theta_active = np.asarray(best_polish.x, dtype=float).copy()

    theta_active = np.asarray(theta_active, dtype=float).ravel()

    # build the full saved-layout vector
    full_params = compact_to_full_params(theta_active, nshape, max_fun_params=max_fun_params, n_extra=n_extra)

    # Hessian / codelen on the compact active parameter vector
    hessian_template = likelihood.get_loss(eq_numpy, integrated, value="hessian")

    # For the Hessian and codelen we use the active compact vector (shape + extras)
    try:
        H = np.asarray(
            hessian_template(
                theta_active,
                xvar, yvar, yerr_lo, yerr_hi,
            ),
            dtype=float,
        )
    except Exception:
        # If the Hessian fails, return a partial result
        Delta = np.full(full_width, np.inf, dtype=float)
        return FitResult(
            fcn=fcn_i,
            success=bool(getattr(res_polish, "success", False)),
            log_opt=log_opt,
            use_physical_scale=use_physical_scale,
            nshape=nshape,
            n_extra=n_extra,
            negloglike=float(best_fun),
            codelen=np.nan,
            compact_params=theta_active,
            full_params=full_params,
            Delta=Delta,
            fisher_diag=np.full(nparam, np.nan, dtype=float),
            hessian=np.full((nparam, nparam), np.nan, dtype=float),
            niter=int(getattr(res_polish, "nit", 0)),
            nstarts=len(sign_list),
            direct_fun=float(best_direct["fun"]),
            direct_signs=best_signs,
        )

    fisher_diag = np.diag(H).copy()
    if np.any(~np.isfinite(fisher_diag)) or np.any(fisher_diag <= 0):
        Delta_active = np.full(nparam, np.inf, dtype=float)
        codelen = np.nan
    else:
        Delta_active = np.sqrt(12.0 / fisher_diag)

        # Simple codelength estimate:
        #   -k/2 log(3) + sum(0.5 log(Fisher_diag) + log(|theta|))
        # over the active fitted parameters.
        if np.any(theta_active == 0):
            codelen = np.nan
        else:
            k = int(np.sum(np.isfinite(fisher_diag) & (fisher_diag > 0)))
            codelen = float(
                -k / 2.0 * math.log(3.0)
                + np.sum(0.5 * np.log(fisher_diag) + np.log(np.abs(theta_active)))
            )

    # Expand Delta to full layout, padding block stays zero
    Delta_full = np.zeros(full_width, dtype=float)
    Delta_full[:nshape] = Delta_active[:nshape] if nshape > 0 else np.array([], dtype=float)
    if n_extra > 0:
        Delta_full[max_fun_params:max_fun_params + n_extra] = Delta_active[nshape:nshape + n_extra]

    return FitResult(
        fcn=fcn_i,
        success=bool(getattr(res_polish, "success", False)),
        log_opt=log_opt,
        use_physical_scale=use_physical_scale,
        nshape=nshape,
        n_extra=n_extra,
        negloglike=float(best_fun),
        codelen=float(codelen),
        compact_params=theta_active,
        full_params=full_params,
        Delta=Delta_full,
        fisher_diag=fisher_diag,
        hessian=H,
        niter=int(getattr(res_polish, "nit", 0)),
        nstarts=len(sign_list),
        direct_fun=float(best_direct["fun"]),
        direct_signs=best_signs,
    )


# -----------------------------------------------------------------------------
# Convenience CLI-ish entry point
# -----------------------------------------------------------------------------
def main_single_function(fcn_i: str, likelihood, **kwargs) -> FitResult:
    """Convenience wrapper for interactive use."""
    return fit_single_function(fcn_i, likelihood, **kwargs)


if __name__ == "__main__":
    print(
        "This module is meant to be imported. "
        "Call fit_single_function(fcn_i, likelihood, ...) from your pipeline."
    )