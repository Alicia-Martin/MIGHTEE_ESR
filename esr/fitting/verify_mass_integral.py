"""
Standalone numerical-accuracy check for MIGHTEELikelihood's enclosed-mass
integral: the add_points-built dense radius grid + cumulative-trapezoidal
quadrature used inside get_pred whenever ESR's analytic integration is
skipped (which is always true on the live pipeline, since
testing_opt_mightee.TRY_INTEGRATION = False).

Compares get_pred's implied enclosed mass (recovered from v_circ) against
scipy.integrate.quad (adaptive, high-precision) at each of a galaxy's
observed radii, for several representative test density shapes, and reports
a PASS/FAIL table against a relative tolerance.

Run directly (from any cwd):
    python esr/fitting/verify_mass_integral.py
"""
import sys
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import numpy as np
from scipy.integrate import quad

# `esr` is a package rooted at MIGHTEE_ESR/, two directories up from this
# file (MIGHTEE_ESR/esr/fitting/). Running this script directly only puts
# its own directory on sys.path, so add the package root explicitly rather
# than requiring the caller to `cd` there first or use `python -m`.
_MIGHTEE_ESR_ROOT = Path(__file__).resolve().parents[2]
if str(_MIGHTEE_ESR_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIGHTEE_ESR_ROOT))

from esr.fitting.dm_likelihood import MIGHTEELikelihood  # noqa: E402

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_FILE = Path(__file__).resolve().parents[2] / "rar_with_ml_direct_phot.csv"
GALAXY_NAME = "J022128.8-042448"
RUN_NAME = "verify_mass_integral"

# 0.1% relative -- roughly two orders of magnitude below MIGHTEE's typical
# rotation-velocity measurement uncertainty (several percent), i.e. a
# "numerics are not the bottleneck" bar. See quality-gates.md.
RELATIVE_TOLERANCE = 1.0e-3


class TestDensity(NamedTuple):
    label: str
    fcn_string: str                    # ESR-syntax density string, rho(r) = f(r; params)
    params: Sequence[float]            # numeric values for a0, a1, ...
    rho_scalar: Callable[..., float]   # same function, plain Python, for scipy.integrate.quad


TEST_DENSITIES = [
    TestDensity(
        label="NFW-like cusp: a0/(x*(1+x)**2)",
        fcn_string="a0/(x*(1+x)**2)",
        params=[3.5e6],
        rho_scalar=lambda r, a0: a0 / (r * (1.0 + r) ** 2),
    ),
    TestDensity(
        label="Cored: a0/(1+x**2)",
        fcn_string="a0/(1+x**2)",
        params=[3.5e6],
        rho_scalar=lambda r, a0: a0 / (1.0 + r ** 2),
    ),
    TestDensity(
        label="ESR-searched form: a0*pow(x,a1)/(a2+x)",
        fcn_string="a0*pow(x,a1)/(a2+x)",
        params=[3.5e6, -1.0, 1.0],
        rho_scalar=lambda r, a0, a1, a2: a0 * r ** a1 / (a2 + r),
    ),
]


def quad_enclosed_mass(rho_scalar: Callable[..., float], params: Sequence[float], r: float) -> float:
    """Reference M(<r) = 4*pi*int_0^r r'^2 rho(r') dr', via scipy.integrate.quad."""
    def integrand(rp: float) -> float:
        return 4.0 * np.pi * rp ** 2 * rho_scalar(rp, *params)

    value, _ = quad(integrand, 0.0, r, limit=200)
    return value


def pipeline_enclosed_mass(likelihood, params, eq_numpy, integrated, r_obs):
    """
    M(<r) at each r in r_obs as actually computed inside get_pred (the
    add_points dense grid -> 4*pi*r^2*rho(r) -> cumulative trapezoid ->
    interpolate back to r_obs), recovered by inverting get_pred's own
    v_circ^2 = v_bar**2 + G*M/r (D=None/Inc=None here, so the baryon term's
    geometric/inclination rescale factors are both 1 -- see get_pred's
    docstring, 2026-08-06 baryon-term fix). self.v_bar**2 is subtracted
    before inverting, to isolate the DM-only mass this check actually cares
    about (integration-grid numerics), not the fixed baryon contribution
    get_pred now always adds. Requires r_obs == likelihood.xvar (true for
    this script's only caller) so likelihood.v_bar stays index-aligned with
    r_obs. use_physical_scale=False here, so get_pred's `a` is the density
    params directly (no rho0/rs).
    """
    assert len(r_obs) == len(likelihood.v_bar), (
        "pipeline_enclosed_mass assumes r_obs is index-aligned with "
        "likelihood.v_bar (i.e. r_obs == likelihood.xvar)."
    )
    v_circ = np.asarray(
        # np.asarray(params): this installed jax version's jnp.atleast_1d
        # (called inside get_pred) rejects a bare Python list -- pre-existing
        # incompatibility, unrelated to the baryon-term fix above.
        likelihood.get_pred(r_obs, np.asarray(params), eq_numpy, integrated=integrated)
    )
    dm_only_v2 = v_circ ** 2 - np.asarray(likelihood.v_bar) ** 2
    return dm_only_v2 * np.asarray(r_obs) / likelihood.G


def run_check(likelihood, test_density: TestDensity, tolerance: float) -> bool:
    fcn_string, eq, integrated = likelihood.run_sympify(
        test_density.fcn_string, try_integration=False
    )
    if integrated:
        raise RuntimeError(
            f"{test_density.label}: try_integration=False should force "
            f"integrated=False; got integrated=True. This check needs the "
            f"numerical add_points/cumtrapz path to actually be exercised."
        )

    eq_numpy = likelihood._make_lambdify(eq, likelihood.nparam_shape)

    r_obs = np.asarray(likelihood.xvar)
    mass_pipeline = pipeline_enclosed_mass(
        likelihood, test_density.params, eq_numpy, integrated, r_obs
    )
    mass_quad = np.array([
        quad_enclosed_mass(test_density.rho_scalar, test_density.params, r)
        for r in r_obs
    ])

    rel_err = np.abs(mass_pipeline - mass_quad) / np.abs(mass_quad)

    print(f"\n{test_density.label}")
    print(f"{'r [kpc]':>10} {'M_pipeline':>16} {'M_quad':>16} {'rel err':>10}  status")
    all_pass = True
    for r, mp, mq, e in zip(r_obs, mass_pipeline, mass_quad, rel_err):
        status = "PASS" if e <= tolerance else "FAIL"
        all_pass &= bool(e <= tolerance)
        print(f"{r:10.4f} {mp:16.6e} {mq:16.6e} {e:10.4%}  {status}")

    worst = rel_err.max()
    print(
        f"  worst relative error = {worst:.4%}  "
        f"({'PASS' if worst <= tolerance else 'FAIL'} at {tolerance:.2%} tolerance)"
    )
    return all_pass


def main() -> int:
    likelihood = MIGHTEELikelihood(
        data_file=str(DATA_FILE),
        name=GALAXY_NAME,
        run_name=RUN_NAME,
        use_physical_scale=False,
    )

    print(f"Galaxy: {GALAXY_NAME}")
    print(f"Observed radii (kpc): {np.asarray(likelihood.xvar)}")
    print(f"Tolerance: {RELATIVE_TOLERANCE:.2%}")

    overall_pass = True
    for test_density in TEST_DENSITIES:
        overall_pass &= run_check(likelihood, test_density, RELATIVE_TOLERANCE)

    print("\n" + "=" * 60)
    print("OVERALL:", "PASS" if overall_pass else "FAIL")
    print("=" * 60)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
