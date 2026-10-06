import os
import warnings
import numpy as np
import pandas as pd
import sympy
from sympy import *
import astropy.constants
import astropy.units as apu
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import InterpolatedUnivariateSpline

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.stats import norm

from esr.fitting.likelihood import Likelihood
from esr.generation.simplifier import time_limit
from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier


# Fiducial flat LambdaCDM cosmology used (a) to convert Radius_arcsec -> kpc
# and (b) to set each galaxy's fiducial angular-diameter distance D_true, so
# that R and distance_true stay self-consistent when D is later varied as a
# free nuisance parameter (R * D/distance_true == R exactly at D=distance_true).
COSMO_H0 = 70.0   # km/s/Mpc
COSMO_OM0 = 0.3
_COSMOLOGY = FlatLambdaCDM(H0=COSMO_H0, Om0=COSMO_OM0)

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

# Distance uncertainty derived from peculiar velocities (no per-galaxy
# distance-uncertainty column exists in this catalog). v_obs ~= cz = H0*D +
# v_pec, so inverting D=cz/H0 without knowing v_pec gives sigma_D =
# sigma_v_pec/H0, i.e. sigma_D/D = sigma_v_pec/(c*z) -- H0 cancels.
# sigma_v_pec = 300 km/s is the standard fiducial RMS peculiar velocity used
# throughout the local-universe distance literature (Tully-Fisher/SPARC-type
# work). For this sample's z-range this comes out to roughly 2-5%
# (highest-z galaxies) up to ~20% (lowest-z galaxies) -- peculiar velocities
# matter proportionally most for the nearest galaxies.
PECULIAR_VELOCITY_KMS = 300.0
_C_KMS = astropy.constants.c.to(apu.km / apu.s).value

# Target absolute radial step size (kpc) for _add_points_fixed. Point count
# per segment scales with segment width relative to this target, rather
# than a single flat count: a flat n_insert is scale-dependent, since the
# same point count spread over a wider segment resolves it more coarsely
# than it does over a narrower one. See _add_points_fixed.
FIXED_GRID_TARGET_STEP_KPC = 1.43091072 / 50


def _angular_diameter_distance_kpc(z):
    """Angular-diameter distance (kpc) at redshift z, fiducial flat LCDM."""
    return float(_COSMOLOGY.angular_diameter_distance(z).to(apu.kpc).value)


class MIGHTEELikelihood(Likelihood):
    """
    Likelihood for MIGHTEE rotation-curve data.

    Modes:
      1) use_physical_scale = False
         ESR function is interpreted directly as rho(r) = f(r)

      2) use_physical_scale = True
         ESR function is interpreted as a dimensionless shape f(u),
         with rho(r) = rho0 * f(r / rs)

    In the second mode, the fitted parameters are:
        [ESR shape params..., rho0, rs]
    and rho0, rs are enforced positive by the sign/log wrapper.
    """

    def __init__(
        self,
        data_file,
        name,
        run_name,
        data_dir=None,
        fn_set='core_maths',
        use_physical_scale=False,
        fn_library_dir=None,
    ):
        # G in units of kpc (km/s)^2 / Msun
        self.G = astropy.constants.G.to(
            apu.kpc * (apu.km / apu.s) ** 2 / apu.Msun
        ).value

        self.use_physical_scale = use_physical_scale

        self.galaxy_data = self._load_galaxy_data(data_file, name)

        self.xvar = jnp.array(self.galaxy_data["R"], dtype=jnp.float64)
        self.yvar = jnp.array(self.galaxy_data["Vobs"], dtype=jnp.float64)
        self.yerr_lo = jnp.array(self.galaxy_data["e_Vobs_lo"], dtype=jnp.float64)
        self.yerr_hi = jnp.array(self.galaxy_data["e_Vobs_hi"], dtype=jnp.float64)

        # self.v_bar (catalog v_bar_total) is the FIXED, non-fit baryon
        # velocity contribution added in quadrature inside get_pred (see its
        # docstring). Stays index-aligned with self.xvar by construction:
        # both are built from the same sorted `galaxy` dataframe slice in
        # _load_galaxy_data. self.e_v_bar remains unused -- MIGHTEE's baryon
        # term is treated as noiseless.
        self.v_bar = jnp.array(self.galaxy_data["Vbar"], dtype=jnp.float64)
        self.e_v_bar = jnp.array(self.galaxy_data["e_Vbar"], dtype=jnp.float64)

        # Distance/inclination nuisance-parameter machinery (stage-2 optimization
        # only -- see get_loss(include_priors=True)). num_galaxy_params counts Inc, D.
        self.z = self.galaxy_data["z"]
        self.inc_true = self.galaxy_data["inc_true"]
        self.e_inc = self.galaxy_data["e_inc"]
        self.distance_true = self.galaxy_data["distance_true"]
        self.e_d = self.galaxy_data["e_d"]
        self.num_galaxy_params = 2

        super().__init__(data_file, run_name, data_dir=data_dir, fn_set=fn_set)

        # Overrides the base class's own esr_dir-derived fn_dir: the function
        # library lives in the separate, shared dm-esr/WL/ESR checkout, not
        # inside this project's own esr/ copy. This path is machine-specific,
        # so it is resolved in priority order: explicit fn_library_dir arg
        # (what run_esr_mightee.py's --fn-library-dir bakes into the addqueue
        # command line, since addqueue jobs can't be relied on to inherit the
        # submitting shell's env) > ESR_FUNCTION_LIBRARY_DIR env var (for
        # interactive/notebook use) > a default local path, so interactive
        # use needs neither set if the library is in <repo>/function_library/core_maths.
        self.fn_dir = fn_library_dir or os.environ.get(
            "ESR_FUNCTION_LIBRARY_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "function_library", "core_maths"),
        )
        self.ylabel = r'$V_{\rm obs}$'

    def _load_galaxy_data(self, data_file, name):
        # sep=None + engine="python" auto-sniffs the delimiter, so this reads
        # both the comma-separated rar_with_ml_direct_phot.csv (81 galaxies)
        # and the whitespace-separated ALL_gbar_gobs_RAR_direct_phot.txt (129
        # galaxies, the fuller catalog) without needing a format-specific path.
        df = pd.read_csv(data_file, sep=None, engine="python")
        galaxy = df[df["Galaxy"] == name].copy()

        if galaxy.empty:
            raise ValueError(f"No data found for galaxy '{name}'")

        galaxy = galaxy.sort_values("Radius_arcsec")

        z = float(galaxy["z"].iloc[0])
        inc_true = float(galaxy["inc"].iloc[0])
        e_inc = float(galaxy["inc_err"].iloc[0])

        distance_true = _angular_diameter_distance_kpc(z)
        e_d = (PECULIAR_VELOCITY_KMS / (_C_KMS * z)) * distance_true

        # Recompute R (kpc) from Radius_arcsec using the SAME cosmology-derived
        # distance_true, so R and distance_true are self-consistent: at D ==
        # distance_true, R * D/distance_true == R exactly. Do NOT use the CSV's
        # own Radius_kpc column (computed with a different/unknown cosmology).
        R_kpc = galaxy["Radius_arcsec"].to_numpy() * ARCSEC_TO_RAD * distance_true

        return {
            "R": R_kpc,
            # v_circ (not v_rot) is the fit target -- it's the catalog's own
            # exact quadrature reconstruction (v_circ = sqrt(v_bar_total^2 +
            # v_DM^2)), the theoretically clean circular velocity this
            # model's own vcirc prediction should be compared against. v_rot
            # is close (median residual ~0.2 km/s) but not identical, so
            # v_circ is used specifically. Error bars still come from
            # v_rot's columns, since v_circ has no error columns of its own.
            "Vobs": galaxy["v_circ"].to_numpy(),
            "e_Vobs_lo": galaxy["v_rot_err_lo"].to_numpy(),
            "e_Vobs_hi": galaxy["v_rot_err_hi"].to_numpy(),
            "Vbar": galaxy["v_bar_total"].to_numpy(),
            "e_Vbar": galaxy["v_bar_total_err"].to_numpy(),
            "z": z,
            "inc_true": inc_true,
            "e_inc": e_inc,
            "distance_true": distance_true,
            "e_d": e_d,
        }

    @staticmethod
    def _cumtrapz_jax(y, x):
        """
        Cumulative trapezoidal integral with initial=0.
        Returns an array with the same length as x.
        """
        y = jnp.asarray(y, dtype=jnp.float64)
        x = jnp.asarray(x, dtype=jnp.float64)

        if y.shape != x.shape:
            raise ValueError("x and y must have the same shape for _cumtrapz_jax.")

        if x.shape[0] < 2:
            return jnp.zeros_like(x)

        dx = x[1:] - x[:-1]
        avg = 0.5 * (y[1:] + y[:-1])
        cum = jnp.cumsum(dx * avg)
        return jnp.concatenate([jnp.array([0.0], dtype=y.dtype), cum])

    def add_points(
        self,
        x,
        n_insert=50,
        r0=1e-6,
        *,
        y=None,
        yerr=None,
        vel_error=0.01,
        min_points_adaptive=4,
        num_points_min=1,
        num_points_max=500,
    ):
        """
        Create a denser radius grid for numerical integration.

        Hybrid scheme:
          - If `y` and `yerr` are both given AND there are at least
            `min_points_adaptive` radius points, use a curvature-adaptive
            grid (see `_add_points_adaptive`).
          - Otherwise (y/yerr not supplied -- so `add_points(x)` can still
            be called with just a radius array -- or fewer than
            `min_points_adaptive` points), fall back to the fixed,
            uniform-linear grid (`_add_points_fixed`).

            51 of 81 MIGHTEE galaxies in rar_with_ml_direct_phot.csv have
            < 4 radius bins -- too few for a cubic spline (k=3 needs
            >= k+1 = 4 points) or a meaningful third-derivative curvature
            estimate -- so this fallback is the common case for this
            dataset, not a rare edge case.

        Important: the grid must start close to zero, not at the first
        observed radius, otherwise M(<r_min) is forced to be zero. Do not
        use exactly zero because some density functions are singular at
        r=0 (ESR searches many functional forms, e.g. containing log(x)
        or 1/x terms).

        Args:
            x: observed radii (kpc). Need not be pre-sorted -- sorted
                internally via argsort, and y/yerr (if given) are permuted
                with the SAME permutation to stay index-aligned with x.
            n_insert: interior points per segment in the fixed-grid
                fallback, and the per-segment fallback count used inside
                the adaptive branch whenever that segment's own
                curvature-based estimate is non-finite or non-positive
                (see `_add_points_adaptive`). Kept as the 2nd positional
                parameter so positional calls are unaffected by the
                keyword-only arguments below.
            r0: absolute ceiling on the near-zero starting radius.
            y: observed velocities (km/s), same length as x. Together
                with `yerr`, enables the adaptive scheme. Keyword-only so
                no existing positional call can accidentally bind to it.
            yerr: per-point reference velocity uncertainty (km/s), same
                length as x. Pass the tighter (smaller) of MIGHTEE's
                asymmetric yerr_lo/yerr_hi, e.g.
                `np.minimum(self.yerr_lo, self.yerr_hi)`, so the grid is
                resolved at least as finely as the most precisely measured
                point demands.
            vel_error: target integration error, as a fraction of the
                tightest velocity measurement uncertainty across the
                galaxy (default 0.01 = 1%).
            min_points_adaptive: minimum radius-point count required to
                attempt the adaptive scheme (default 4 = k+1 for a cubic
                spline; scipy raises "(m>k) failed" below this).
            num_points_min: floor on the adaptive scheme's per-segment
                inserted-point count after rounding (default 1: never
                leave a single trapezoidal segment represented only by
                its two endpoints).
            num_points_max: ceiling on the adaptive scheme's per-segment
                inserted-point count (default 500 = 10*n_insert: generous
                headroom for genuinely high-curvature segments, while
                bounding the cost of an ill-conditioned curvature
                estimate from silently blowing up the grid size).

        Returns:
            jnp.ndarray of extended, sorted radii (float64).
        """
        x = np.asarray(x, dtype=float)

        if len(x) == 0:
            return jnp.asarray([], dtype=jnp.float64)

        if y is not None and len(y) != len(x):
            raise ValueError(f"add_points: len(y)={len(y)} != len(x)={len(x)}")
        if yerr is not None and len(yerr) != len(x):
            raise ValueError(f"add_points: len(yerr)={len(yerr)} != len(x)={len(x)}")

        # Sort x, and carry y/yerr along with the SAME permutation (do not
        # sort x independently of y/yerr, or their alignment breaks).
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = np.asarray(y, dtype=float)[sort_idx] if y is not None else None
        yerr = np.asarray(yerr, dtype=float)[sort_idx] if yerr is not None else None

        # Include a point close to zero.
        # Do not use exactly zero because some density functions are singular at r=0.
        r_start = min(r0, 1e-6 * x[0])

        use_adaptive = (
            y is not None and yerr is not None and len(x) >= min_points_adaptive
        )

        if use_adaptive:
            try:
                extended_x = self._add_points_adaptive(
                    x, y, yerr, r_start,
                    n_insert=n_insert,
                    vel_error=vel_error,
                    num_points_min=num_points_min,
                    num_points_max=num_points_max,
                )
                return jnp.asarray(extended_x, dtype=jnp.float64)
            except Exception as exc:
                warnings.warn(
                    f"add_points: curvature-adaptive grid failed ({exc!r}); "
                    f"falling back to the fixed uniform grid (n_insert="
                    f"{n_insert}) for this galaxy.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        extended_x = self._add_points_fixed(x, r_start, n_insert=n_insert)
        return jnp.asarray(extended_x, dtype=jnp.float64)

    def _add_points_fixed(self, x_sorted, r_start, n_insert=50, max_insert=2000):
        """
        Uniform-linear per-segment grid, width-scaled -- used when the
        curvature-adaptive scheme isn't available (see add_points).
        `x_sorted` must already be sorted ascending.

        n_insert=50: the enclosed-mass integrand 4*pi*r^2*rho(r) for a
        typical density profile rises from ~0 near r=0 and peaks around
        the profile's own scale radius (e.g. r=1 for a0/(x*(1+x)**2))
        before declining -- i.e. the region needing fine resolution is
        near the *upper* end of the first segment [r_start, x_sorted[0]],
        not spread across the ~6 decades down to r_start. Linear spacing
        is used rather than log (geomspace) spacing for this reason: log
        spacing concentrates points where the integrand is negligible and
        starves the actual peak, giving a worse error.

        Per-segment point count is scaled to FIXED_GRID_TARGET_STEP_KPC
        (a target absolute step size), not held flat at n_insert, because
        a flat count is scale-dependent: the same point count resolves a
        wider segment more coarsely than a narrower one. n_insert is kept
        as a floor (never fewer points than that for narrow segments);
        max_insert bounds the cost for unusually wide ones.
        """
        x_0 = np.concatenate([[r_start], x_sorted])
        widths = np.diff(x_0)

        n_per_segment = np.clip(
            np.ceil(widths / FIXED_GRID_TARGET_STEP_KPC).astype(int),
            n_insert,
            max_insert,
        )

        new_points = [
            np.linspace(x_0[i], x_0[i + 1], n_per_segment[i] + 2)[1:-1]
            for i in range(len(x_0) - 1)
        ]

        if new_points:
            return np.sort(np.concatenate([x_0] + new_points))
        return x_0

    def _add_points_adaptive(
        self, x_sorted, y_sorted, yerr_sorted, r_start,
        n_insert=50, vel_error=0.01, num_points_min=1, num_points_max=500,
    ):
        """
        Curvature-adaptive grid. All args are already sorted by radius
        (ascending), aligned index-for-index.

        Core idea: mass = y**2*x/G treats the OBSERVED velocity data as
        an implied mass profile M_obs(r) = v_obs(r)^2 * r / G -- a
        data-driven proxy for how much resolution the mass-vs-radius
        curve needs, used once per galaxy (shared across whichever ESR
        candidate function is being tested this call), before any
        specific candidate function or its parameters are known. A cubic
        spline is fit through M_obs(r) (using the observed radii as
        knots); its third derivative at each segment's midpoint gives a
        local curvature estimate. A target absolute error (`vel_error` *
        the tightest velocity measurement uncertainty across the galaxy,
        divided by the number of data points) is converted into a
        required step size via the standard quadrature-error-bound
        rearrangement step_size = sqrt(6*error/(midpoint*der_fit)), and
        num_points = round(interval_width/step_size) gives that
        interval's inserted-point count.

        Notes on this implementation:
          - Segments are built from r_start (near zero), not literal 0,
            since some density functions are singular at r=0.
          - Uses plain scipy.interpolate.InterpolatedUnivariateSpline: this
            method is never called inside a jax.jit/grad trace (called
            once per galaxy from run_sympify), so JAX-differentiability
            isn't load-bearing here.
          - yerr_sorted is expected to already be the tighter (smaller) of
            MIGHTEE's asymmetric yerr_lo/yerr_hi per point -- reduced by
            the caller (run_sympify), not here.
          - Includes explicit non-finite / degenerate-curvature handling
            and num_points clamps, needed because the smallest
            adaptive-eligible galaxies have only 4 points, which makes the
            cubic-spline curvature estimate more prone to being
            ill-conditioned.
        """
        mass = y_sorted**2 * x_sorted / self.G

        x_0 = np.concatenate([[r_start], x_sorted])
        middle_points = (x_0[1:] + x_0[:-1]) / 2
        widths = np.diff(x_0)

        # Cubic spline through the OBSERVED (x, mass) pairs only -- r_start
        # is deliberately NOT a spline knot. The first segment's midpoint
        # (between r_start and x_sorted[0]) is therefore outside the
        # spline's fitted domain and evaluated via scipy's default
        # extrapolation.
        spline = InterpolatedUnivariateSpline(x_sorted, mass, k=3)
        der_fit = spline(middle_points, nu=3)

        # Tightest (smallest) finite, positive per-point velocity
        # uncertainty across the galaxy, as the reference precision.
        # Guarded explicitly: a single reported yerr of exactly 0 would
        # otherwise drive error_vel, and therefore EVERY segment's
        # step_size, to exactly 0 -- i.e. would poison the whole galaxy's
        # adaptive grid, not just one segment.
        abs_yerr = np.abs(yerr_sorted)
        finite_positive = np.isfinite(abs_yerr) & (abs_yerr > 0)
        if not np.any(finite_positive):
            raise ValueError(
                "no finite, positive yerr values -- cannot form a "
                "reference measurement precision for the adaptive grid"
            )
        min_abs_yerr = np.min(abs_yerr[finite_positive])

        error_vel = min_abs_yerr / len(x_sorted) * vel_error
        error = 2 * error_vel * y_sorted * x_sorted / self.G

        # error/middle_points/der_fit/widths all have length
        # len(x_sorted) (x_0 has len(x_sorted)+1 entries -> len(x_sorted)
        # segments) -- elementwise division below is shape-safe.
        denom = middle_points * der_fit
        with np.errstate(divide="ignore", invalid="ignore"):
            step_size = np.sqrt(np.abs(6 * error / denom))

        step_size_ok = np.isfinite(step_size) & (step_size > 0)
        safe_step = np.where(step_size_ok, step_size, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            num_points_raw = widths / safe_step

        # Per-segment fallback: non-finite/non-positive step_size (der_fit
        # exactly 0 -> division by zero; der_fit and error both 0 -> 0/0
        # nan; a spline ill-conditioned by fitting a cubic through as few
        # as 4 points) falls back to n_insert for THAT segment, not the
        # whole galaxy.
        num_points_raw = np.where(
            step_size_ok & np.isfinite(num_points_raw), num_points_raw, n_insert
        )

        num_points = np.clip(np.round(num_points_raw), num_points_min, num_points_max)

        if not np.all(np.isfinite(num_points)):
            # Should be unreachable given the substitution above; a hard,
            # loud failure here (caught by add_points's try/except, which
            # falls back to the fixed grid for the whole galaxy) is
            # preferable to silently casting a NaN/inf to int.
            raise RuntimeError("num_points contains non-finite values after clamping")

        num_points = num_points.astype(int)

        new_points = [
            np.linspace(x_0[i], x_0[i + 1], num_points[i] + 2, endpoint=True)[1:-1]
            for i in range(len(x_0) - 1)
        ]

        if new_points:
            return np.sort(np.concatenate([x_0] + new_points))
        return x_0

    def _make_lambdify(self, expr, nparam):
        """
        Create a JAX-compatible lambdified function for expr(x, a0, a1, ...).
        """
        try:
            if nparam == 0:
                return sympy.lambdify(x, expr, modules=["jax"])

            elif nparam == 1:
                return sympy.lambdify([x, a0], expr, modules=["jax"])

            else:
                all_a = list(sympy.symbols(' '.join([f'a{i}' for i in range(nparam)]), real=True))
                return sympy.lambdify([x] + all_a, expr, modules=["jax"])
        except KeyError:
            # Some function-library entries are literal degenerate constants
            # (e.g. "zoo"/ComplexInfinity, which can appear when ESR's own
            # generation/simplification collapses some tree to a symbolic
            # infinity). sympy's pycode/jax printer has no registered
            # handler for these special constants (Abs/ComplexInfinity/nan)
            # and raises a bare KeyError inside sympy.lambdify itself --
            # not a normal "bad function" failure mode, and since
            # run_sympify calls this unconditionally, an unhandled KeyError
            # would crash the whole batch run instead of just this one
            # candidate. The correct semantic treatment is that this
            # candidate is undefined/divergent everywhere, so it should
            # always report infinite loss (ranked last, same as any other
            # always-fails function), not crash.
            def _always_inf(xv, *_a):
                return jnp.full(jnp.shape(jnp.asarray(xv)), jnp.inf)
            return _always_inf

    def run_sympify(self, fcn_i, tmax=5, try_integration=True):
        """
        Parse the ESR function.

        If use_physical_scale=False:
            interpret ESR directly as rho(r)

        If use_physical_scale=True:
            interpret ESR as f(u), with u = r/rs
            and use rho(r) = rho0 * f(r/rs)
        """
        fcn_i = fcn_i.replace('\n', '')
        fcn_i = fcn_i.replace("'", '')

        try:
            eq = sympy.sympify(
                fcn_i,
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
        except Exception:
            # Same failure class as the "zoo"/ComplexInfinity KeyError
            # _make_lambdify already guards against (see its docstring),
            # but one level earlier: a small number of function-library
            # strings fail to sympify at all. Falling back to sympy.zoo is
            # guaranteed to hit _make_lambdify's existing KeyError ->
            # _always_inf path below, giving this candidate the same
            # "always fails, ranked last" treatment as any other
            # degenerate entry instead of crashing run_sympify's caller.
            eq = sympy.zoo

        max_param = 4
        # max_param is hardcoded here, but test_all.py/test_all_Fisher.py/
        # match.py all size their output columns using the comp-dependent
        # max_param_shape = max(4, floor((comp-1)/2)), which only exceeds 4
        # starting at comp=11. count_params only ever scans for "a0".."a3"
        # when max_param=4, so a genuine 5th shape parameter (a4) would be
        # silently undercounted rather than raising -- refuse loudly instead.
        # This pipeline is not yet safe past comp=10.
        if "a4" in fcn_i or "a5" in fcn_i:
            raise ValueError(
                f"run_sympify: {fcn_i!r} appears to need a 5th+ shape "
                f"parameter (a4/a5) but max_param is hardcoded to 4 here -- "
                f"this pipeline is not yet safe past comp=10."
            )
        nparam_shape = simplifier.count_params([fcn_i], max_param)[0]
        self.nparam_shape = nparam_shape

        if self.use_physical_scale:
            self.nparam_extra = 2  # rho0 and rs
        else:
            self.nparam_extra = 0

        self.nparam = self.nparam_shape + self.nparam_extra

        # Density function for positivity checks
        self.eq_diff = self.get_diff(fcn_i, eq, self.nparam_shape)

        integrated = False

        if try_integration:
            try:
                with time_limit(tmax):
                    # Antiderivative of 4*pi*x^2 * f(x) dx
                    eq2 = 4 * jnp.pi * integrate(eq * x**2, x)

                    if eq2.has(sympy.Integral):
                        raise ValueError(
                            "Analytic integration returned an unevaluated Integral."
                        )

                    # Test lambdify once to catch obvious issues early
                    eq_test = self._make_lambdify(eq2, self.nparam_shape)

                    if self.nparam_shape == 0:
                        _ = eq_test(self.xvar)
                    elif self.nparam_shape == 1:
                        _ = eq_test(self.xvar, 1)
                    else:
                        _ = eq_test(self.xvar, *([1] * self.nparam_shape))

                    eq = eq2
                    integrated = True

            except Exception:
                integrated = False

        if not integrated:
            # Denser grid for numerical integration. Pass y/yerr so
            # add_points can use the curvature-adaptive scheme wherever
            # there are enough points (>=4); falls back to the fixed
            # uniform grid otherwise (the majority case for this dataset:
            # 51/81 galaxies have < 4 radius bins -- see add_points docstring).
            if not hasattr(self, "extended_R"):
                yerr_tight = np.minimum(
                    np.asarray(self.yerr_lo), np.asarray(self.yerr_hi)
                )
                self.extended_R = self.add_points(
                    self.xvar, y=self.yvar, yerr=yerr_tight
                )

        return fcn_i, eq, integrated

    def get_diff(self, fcn_i, eq, nparam_shape):
        """
        Lambdified density-shape function.

        This returns the shape function f(x), not the full physical density
        when use_physical_scale=True.
        """
        return self._make_lambdify(eq, nparam_shape)

    def get_pred(self, x, a, eq_numpy, integrated, D=None, Inc=None, return_components=False):
        """
        Convert the ESR density model into the TOTAL predicted circular
        velocity v_circ(r), for direct comparison against self.yvar (v_circ,
        the catalog's TOTAL observed rotation velocity -- baryons + DM):

            v_circ(r)^2 = inc_ratio2 * baryon_term(r) + dm_term(r)

        where:
          - dm_term(r) = G * M_DM(<r) / r, with M_DM(<r) the enclosed mass of
            the ESR-searched density profile (see Modes below) -- the only
            part of the model with free/fitted shape parameters.
          - baryon_term(r) = (D_eff/self.distance_true) * self.v_bar(r)**2, a
            FIXED (non-fit) contribution from the catalog's v_bar_total
            column (self.v_bar, index-aligned with x via self.xvar). MIGHTEE
            has no mass-to-light (upsilon) free parameters, so there is no
            per-component fit scaling here -- only a (D/distance_true)
            geometric rescale of the fixed baryon term.
          - inc_ratio2 = (sin(deg2rad(Inc_eff))/sin(deg2rad(self.inc_true)))**2
            applies ONLY to baryon_term, not to dm_term: the inclination
            correction rescales the baryonic v^2, not the dark-matter term,
            since inclination affects how the observed line-of-sight
            velocity relates to the baryonic disk geometry, not the
            spherically-symmetric DM enclosed-mass term.

        Modes (apply to the DM term only):
          - use_physical_scale=False:
                rho(r) = f(r)

          - use_physical_scale=True:
                rho(r) = rho0 * f(r/rs)

        Parameter order in physical-scale mode:
            a = [shape params..., rho0, rs]

        Args:
            D: distance nuisance parameter (kpc), optional. When given, the
                physical radius used for the DM term throughout is
                r = x * D/self.distance_true, and D_eff = D also feeds the
                baryon term's (D/distance_true) rescale. Leave as None
                (default) to use the DM-term radius directly (r == x), as
                the shape-parameter-only DIRECT search does; the baryon
                term's distance rescale then becomes a no-op
                (D_eff = self.distance_true, so D_eff/distance_true == 1).
                None does NOT skip the baryon term -- it is always added,
                only its D-dependent rescale collapses to 1.
            Inc: inclination nuisance parameter (degrees), optional. When
                given, inc_ratio2 (see above) rescales baryon_term only.
                Leave as None (default) to use Inc_eff = self.inc_true, i.e.
                inc_ratio2 == 1 -- again NOT "skip the correction": the
                baryon term is still added, just at its catalog-fiducial
                inclination (the same convention D=None uses).
            return_components: when True, additionally return
                (dm_v2, baryon_v2) -- the two terms' own contribution to
                v_circ^2 (dm_v2 = dm_term, baryon_v2 = inc_ratio2*baryon_term,
                i.e. v_circ^2 = dm_v2 + baryon_v2 exactly) -- as
                (v_circ, dm_v2, baryon_v2) instead of just v_circ. f_loss
                requests return_components=True and discards dm_v2/
                baryon_v2, purely so that vpred is computed identically
                regardless of which call mode is used elsewhere. Default
                False preserves the plain single-return signature for
                every other caller.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        a = jnp.atleast_1d(a)

        r_obs = x if D is None else x * (D / self.distance_true)
        r_obs_safe = jnp.where(r_obs > 0, r_obs, 1e-8)

        nshape = self.nparam_shape

        if self.use_physical_scale:
            shape_params = a[:nshape]
            rho0 = a[nshape]
            rs = a[nshape + 1]

            valid_scale = (
                jnp.isfinite(rho0) &
                jnp.isfinite(rs) &
                (rho0 > 0) &
                (rs > 0)
            )

            # rs_safe, same idiom as r_obs_safe above: mass=jnp.where(valid_scale,
            # mass, jnp.inf) below already masks the FORWARD value correctly when
            # rs<=0, but computing u=r_obs/rs unguarded still runs through
            # eq_numpy at u=inf (or worse, nan if rs is itself nan), and
            # jnp.where evaluates both branches -- so a NaN produced here still
            # poisons the GRADIENT even though the value looks fine, since
            # masking the forward value alone does not mask the gradient
            # computed through the masked-out branch.
            rs_safe = jnp.where(rs > 0, rs, 1.0)
            u = r_obs / rs_safe

            if integrated:
                # eq_numpy is antiderivative of 4*pi*u^2*f(u) du
                # so M(<r) = rho0 * rs^3 * F(r/rs)
                if nshape == 0:
                    F_u = eq_numpy(u) - eq_numpy(0.0)
                else:
                    F_u = eq_numpy(u, *shape_params) - eq_numpy(0.0, *shape_params)

                mass = rho0 * rs**3 * F_u

            else:
                # Numerical integration on a denser grid in physical radius r
                r_dense = self.extended_R if D is None else self.extended_R * (D / self.distance_true)
                u_dense = r_dense / rs_safe

                if nshape == 0:
                    f_dense = eq_numpy(u_dense)
                else:
                    f_dense = eq_numpy(u_dense, *shape_params)

                rho_dense = rho0 * jnp.asarray(f_dense, dtype=jnp.float64)

                integrand = 4.0 * jnp.pi * r_dense**2 * rho_dense
                mass_dense = self._cumtrapz_jax(integrand, r_dense)

                # Interpolate back to observed radii
                mass = jnp.interp(r_obs, r_dense, mass_dense)

            mass = jnp.where(valid_scale, mass, jnp.inf)

        else:
            # use_physical_scale=False: rho(r) = f(r) directly.
            if integrated:
                if len(a) == 0:
                    mass = eq_numpy(r_obs) - eq_numpy(0.0)
                else:
                    mass = eq_numpy(r_obs, *a) - eq_numpy(0.0, *a)

            else:
                r_dense = self.extended_R if D is None else self.extended_R * (D / self.distance_true)

                if len(a) == 0:
                    rho_dense = eq_numpy(r_dense)
                else:
                    rho_dense = eq_numpy(r_dense, *a)

                rho_dense = jnp.asarray(rho_dense, dtype=jnp.float64)
                integrand = 4.0 * jnp.pi * r_dense**2 * rho_dense
                mass_dense = self._cumtrapz_jax(integrand, r_dense)

                # Interpolate back to the observed radii
                mass = jnp.interp(r_obs, r_dense, mass_dense)

        # Step 2: combine the ESR-searched DM term with the FIXED (non-fit)
        # baryon term, in quadrature. MIGHTEE has no upsilon (mass-to-light)
        # free parameters, so baryon_term is a single fixed
        # per-radius-point value (self.v_bar, index-aligned with x via
        # self.xvar) rather than a fit-scaled sum of disk/gas/bulge terms.
        dm_term = jnp.where(r_obs_safe > 0, self.G * mass / r_obs_safe, 0.0)

        # D/Inc default to the catalog's fiducial values (NOT "skip the
        # correction") when None -- the baryon term must ALWAYS be added,
        # since the fit target (self.yvar/v_circ) is the TOTAL observed
        # velocity in every call path, including the shape-only stage-1
        # DIRECT search. At these defaults inc_ratio2==1 and
        # (D_eff/distance_true)==1, i.e. baryon_term == self.v_bar**2
        # unscaled -- the catalog's own baryon curve, verbatim.
        D_eff = self.distance_true if D is None else D
        Inc_eff = self.inc_true if Inc is None else Inc

        sin_inc_true = jnp.sin(jnp.deg2rad(self.inc_true))
        sin_inc_true_safe = jnp.where(jnp.abs(sin_inc_true) > 1e-8, sin_inc_true, 1e-8)
        inc_ratio2 = (jnp.sin(jnp.deg2rad(Inc_eff)) / sin_inc_true_safe) ** 2

        baryon_term = (D_eff / self.distance_true) * (self.v_bar ** 2)
        baryon_v2 = inc_ratio2 * baryon_term

        v2 = baryon_v2 + dm_term
        v_circ = jnp.sqrt(jnp.maximum(v2, 0.0))
        if return_components:
            return v_circ, dm_term, baryon_v2
        return v_circ

    @staticmethod
    def neg_log_asymmetric_gaussian(y_true, y_pred, err_lo, err_hi):
        """
        Piecewise Gaussian with different lower/upper errors.
        Uses err_hi if model lies above the data, err_lo otherwise.
        """
        err_lo = jnp.asarray(err_lo, dtype=jnp.float64)
        err_hi = jnp.asarray(err_hi, dtype=jnp.float64)

        sigma = jnp.where(y_pred >= y_true, err_hi, err_lo)
        sigma = jnp.where(sigma > 0, sigma, 1e-12)

        return 0.5 * ((y_true - y_pred) / sigma) ** 2 + jnp.log(sigma)

    def _neg_log_truncated_gaussian(self, val, mean, sd, lower, upper):
        """
        -log p(val) for Normal(mean, sd) truncated to [lower, upper].
        """
        a_std = (lower - mean) / sd
        b_std = (upper - mean) / sd
        Z = jnp.maximum(norm.cdf(b_std) - norm.cdf(a_std), jnp.finfo(jnp.float64).tiny)
        log_pdf = norm.logpdf(val, loc=mean, scale=sd)
        log_trunc_pdf = jnp.where((val < lower) | (val > upper), -jnp.inf, log_pdf - jnp.log(Z))
        return -log_trunc_pdf

    def get_loss(self, eq_numpy, integrated, verbose=False, value='value_and_grad',
                 include_priors=False, fixed_galaxy_params=None, **kwargs):
        """
        Negative log-likelihood with:
        - asymmetric errors on Vobs
        - positivity penalty for rho(r) >= 0

        In physical-scale mode:
            rho(r) = rho0 * f(r/rs)
            with rho0 > 0 and rs > 0

        Args:
            include_priors: when False (default), `a` is the model-parameter
                vector only (length self.nparam). get_pred always adds the
                fixed baryon term (D_eff/Inc_eff default to
                distance_true/inc_true when D/Inc aren't supplied here), so
                vpred -- and therefore this stage-1 loss -- reflects the
                total (baryon + DM) predicted velocity for every
                galaxy/function, consistent with the fact that the fit
                target (v_circ) is itself the total observed velocity. When
                True, `a` is [model params..., Inc, D] (length self.nparam +
                self.num_galaxy_params); D rescales the DM-term radius and
                the baryon term's (D/distance_true) factor via
                get_pred(..., D=D); Inc rescales ONLY the baryon term via
                get_pred(..., Inc=Inc) (inc_ratio2 = (sin(Inc)/sin(inc_true))**2,
                since inclination affects only the baryonic v^2 term, not the
                DM term); and both Inc and D additionally get a
                truncated-Gaussian prior anchored to their catalog values.
            fixed_galaxy_params: (Inc, D) tuple, optional. Lets a caller
                evaluate the pure data likelihood at a specific Inc/D (e.g.
                the true jointly-optimized value) while keeping `a` as the
                model-parameter-only vector, instead of being limited to
                either catalog-default Inc/D (D=None, Inc=None) or treating
                Inc/D as free elements of `a` (include_priors=True). This
                matters for model-selection ranking: `-logL` and the
                parameter cost charged for Inc/D need to be evaluated at the
                same point, not at catalog defaults for one and the true
                joint value for the other. When given, get_pred uses these
                Inc/D instead of catalog defaults. Whether the prior cost for
                these fixed values gets added follows include_priors, same
                as always: with include_priors=False (default) it's not
                added (a caller can charge it separately downstream, so
                nothing is double-counted there); with include_priors=True
                the prior IS added, evaluated at these fixed values --
                letting a caller get "-logL(data) - log p(Inc) - log p(D)"
                for a shape-only optimization (Inc/D held fixed, not
                searched over) in one get_loss call, instead of needing a
                separate, duplicate evaluation of the same prior terms
                afterward. Default None reproduces the behavior of every
                caller that doesn't pass it.
        """

        def check_density(params, eq_diff, r):
            r = jnp.asarray(r, dtype=jnp.float64)
            params = jnp.atleast_1d(params)

            if self.use_physical_scale:
                nshape = self.nparam_shape
                shape_params = params[:nshape]
                rho0 = params[nshape]
                rs = params[nshape + 1]

                valid_scale = (
                    jnp.isfinite(rho0) &
                    jnp.isfinite(rs) &
                    (rho0 > 0) &
                    (rs > 0)
                )

                u = r / rs

                if nshape == 0:
                    f = eq_diff(u)
                else:
                    f = eq_diff(u, *shape_params)

                rho = rho0 * jnp.asarray(f, dtype=jnp.float64)
                ok = valid_scale & jnp.all(jnp.isfinite(rho) & (rho >= 0))

            else:
                if len(params) == 0:
                    rho = eq_diff(r)
                else:
                    rho = eq_diff(r, *params)

                rho = jnp.asarray(rho, dtype=jnp.float64)
                ok = jnp.all(jnp.isfinite(rho) & (rho >= 0))

            return jnp.where(ok, 0.0, jnp.inf)

        def f_loss(a, xvar, yvar, yerr_lo, yerr_hi=None):
            if yerr_hi is None:
                yerr_hi = yerr_lo

            a = jnp.atleast_1d(a)

            if include_priors:
                if fixed_galaxy_params is not None:
                    # Inc/D held fixed but still priced by the prior --
                    # see fixed_galaxy_params in this method's docstring.
                    model_params = a
                    Inc, D = fixed_galaxy_params
                else:
                    n_model = self.nparam
                    model_params = a[:n_model]
                    Inc = a[n_model]
                    D = a[n_model + 1]
            else:
                model_params = a
                if fixed_galaxy_params is not None:
                    Inc, D = fixed_galaxy_params
                else:
                    D = None
                    Inc = None

            # Use denser grid for the density positivity check when available,
            # rescaled by D the same way get_pred rescales it internally.
            # (Inc doesn't affect physical radius, only velocity amplitude, so
            # it plays no role in this density check.)
            r_check = self.extended_R if hasattr(self, "extended_R") else xvar
            if D is not None:
                r_check = r_check * (D / self.distance_true)
            density_penalty = check_density(model_params, self.eq_diff, r_check)

            # return_components=True is used here (rather than the plain
            # vpred-only call) even though dm_v2/baryon_v2 go unused, purely
            # so that get_pred's vpred computation is guaranteed identical
            # regardless of which call mode is used elsewhere.
            vpred, _, _ = self.get_pred(
                xvar, model_params, eq_numpy, integrated=integrated, D=D, Inc=Inc,
                return_components=True,
            )

            nll = jnp.sum(
                self.neg_log_asymmetric_gaussian(yvar, vpred, yerr_lo, yerr_hi)
            )

            if include_priors:
                priors = (
                    self._neg_log_truncated_gaussian(Inc, self.inc_true, self.e_inc, 0.0, 90.0) +
                    self._neg_log_truncated_gaussian(D, self.distance_true, self.e_d, 0.0, np.inf)
                )
                nll = nll + priors

            return density_penalty + nll

        if value == 'hessian':
            return jax.hessian(f_loss)
        elif value == 'evaluate':
            # JIT-compiled: this path is called from every DIRECT-search and
            # Nelder-Mead-polish objective across test_all.py,
            # testing_opt_mightee.py, match.py, and test_all_Fisher.py --
            # tens of thousands of calls per function during a DIRECT search.
            # Un-jitted, each call re-traces through JAX eagerly, which is
            # far too slow for an exhaustive per-function sweep. Same
            # eq_numpy/shapes are reused across an entire optimization run,
            # so the one-time compile cost amortizes immediately.
            return jax.jit(f_loss)
        elif value == 'grad':
            return jax.grad(f_loss)
        else:
            return jax.jit(jax.value_and_grad(f_loss))

    def get_wrapped_like(self, loss_template):
        """
        Wrapper compatible with optimizers.
        Supports either symmetric or asymmetric errors.

        In physical-scale mode, the parameter vector is:
            [shape params..., rho0, rs]
        and rho0, rs are forced positive by using sign +1.
        """

        def handle_nans(negloglike):
            try:
                return jnp.where(jnp.isnan(negloglike), jnp.inf, negloglike)
            except Exception:
                return (jnp.where(jnp.isnan(negloglike[0]), jnp.inf, negloglike[0]),) + negloglike[1:]

        def wrapped_like(x, xvar, yvar, yerr_lo, yerr_hi=None, signs=None, check_nans=True):
            if yerr_hi is None:
                yerr_hi = yerr_lo

            if signs is None:
                p = x.copy()
            else:
                signs = list(signs)

                # If only the original ESR signs were passed in,
                # append positive signs for rho0 and rs.
                if self.use_physical_scale and len(signs) == self.nparam_shape:
                    signs = signs + [1, 1]

                if len(signs) != self.nparam:
                    raise ValueError(
                        f"Expected {self.nparam} signs, but got {len(signs)}. "
                        f"nparam_shape={self.nparam_shape}, "
                        f"use_physical_scale={self.use_physical_scale}"
                    )

                p = x.copy()
                try:
                    p[:len(signs)] = [s * 10**xi for s, xi in zip(signs, x)]
                except Exception:
                    p = p.at[:len(signs)].set(
                        [s * 10**xi for s, xi in zip(signs, x[:len(signs)])]
                    )

            loss = loss_template(p, xvar, yvar, yerr_lo, yerr_hi)

            if check_nans:
                loss = handle_nans(loss)

            return loss

        return wrapped_like


