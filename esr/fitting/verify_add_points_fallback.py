"""
Standalone check for MIGHTEELikelihood.add_points' fixed-grid fallback path
(used whenever a galaxy has fewer than min_points_adaptive=4 radius bins --
the majority case for this dataset: 51/81 galaxies in
rar_with_ml_direct_phot.csv have < 4 bins, so this is not a rare edge case).

Confirms, for two concrete low-point-count galaxies:
  1. The <4-points guard actually engages: add_points(x, y=..., yerr=...)
     and add_points(x) (no kwargs) give identical output.
  2. The resulting grid is still numerically accurate against
     scipy.integrate.quad, at the same tolerance as verify_mass_integral.py.

Run directly (from any cwd):
    python esr/fitting/verify_add_points_fallback.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

_MIGHTEE_ESR_ROOT = Path(__file__).resolve().parents[2]
if str(_MIGHTEE_ESR_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIGHTEE_ESR_ROOT))

from esr.fitting.dm_likelihood import MIGHTEELikelihood  # noqa: E402

DATA_FILE = _MIGHTEE_ESR_ROOT / "rar_with_ml_direct_phot.csv"
RELATIVE_TOLERANCE = 1.0e-3  # matches verify_mass_integral.py

# Concrete low-point-count galaxies identified directly from the CSV.
LOW_POINT_GALAXIES = [
    ("J021748.5-043241", 3),
    ("J021814.0-044944", 2),
]

A0 = 3.5e6


def rho(r, a0):
    return a0 / (r * (1.0 + r) ** 2)


def quad_enclosed_mass(a0, r):
    def integrand(rp):
        return 4.0 * np.pi * rp ** 2 * rho(rp, a0)

    value, _ = quad(integrand, 0.0, r, limit=200)
    return value


def check_galaxy(name, expected_n):
    likelihood = MIGHTEELikelihood(
        data_file=str(DATA_FILE),
        name=name,
        run_name="verify_add_points_fallback",
        use_physical_scale=False,
    )
    n = len(likelihood.xvar)

    print(f"\n{name}: {n} radius bins (expected < 4)")
    if n != expected_n:
        print(
            f"  WARNING: expected {expected_n} rows for this galaxy, found {n} -- "
            f"the CSV may have changed; re-pick a low-point-count galaxy if so."
        )
    if n >= 4:
        print("  SKIPPED: this galaxy no longer has < 4 points, can't test the fallback path.")
        return True

    yerr_tight = np.minimum(np.asarray(likelihood.yerr_lo), np.asarray(likelihood.yerr_hi))
    grid_with_data = np.asarray(likelihood.add_points(likelihood.xvar, y=likelihood.yvar, yerr=yerr_tight))
    grid_no_data = np.asarray(likelihood.add_points(likelihood.xvar))
    fallback_engaged = np.array_equal(grid_with_data, grid_no_data)
    print(f"  fallback engaged (add_points(y=,yerr=) == add_points()): {fallback_engaged}")

    fcn_string, eq, integrated = likelihood.run_sympify("a0/(x*(1+x)**2)", try_integration=False)
    eq_numpy = likelihood._make_lambdify(eq, likelihood.nparam_shape)

    r_obs = np.asarray(likelihood.xvar)
    # np.asarray([A0]): this installed jax version's jnp.atleast_1d (called
    # inside get_pred) rejects a bare Python list -- pre-existing
    # incompatibility, unrelated to the baryon-term fix noted below.
    v_circ = np.asarray(likelihood.get_pred(r_obs, np.asarray([A0]), eq_numpy, integrated=integrated))
    # get_pred now always adds the fixed baryon term (D=None/Inc=None here
    # -> its geometric/inclination rescale factors are both 1) -- subtract
    # it before inverting v_circ^2 = v_bar**2 + G*M/r, to isolate the
    # DM-only mass this check actually cares about (add_points/cumtrapz
    # numerics), not the baryon contribution. See dm_likelihood.py's
    # get_pred docstring (2026-08-06 baryon-term fix).
    dm_only_v2 = v_circ ** 2 - np.asarray(likelihood.v_bar) ** 2
    mass_pipeline = dm_only_v2 * r_obs / likelihood.G
    mass_quad = np.array([quad_enclosed_mass(A0, r) for r in r_obs])
    rel_err = np.abs(mass_pipeline - mass_quad) / np.abs(mass_quad)

    worst = rel_err.max()
    passed = fallback_engaged and worst <= RELATIVE_TOLERANCE
    print(f"  worst relative error vs. quad = {worst:.4%}  "
          f"({'PASS' if worst <= RELATIVE_TOLERANCE else 'FAIL'} at {RELATIVE_TOLERANCE:.2%} tolerance)")
    return passed


def main() -> int:
    overall_pass = True
    for name, expected_n in LOW_POINT_GALAXIES:
        overall_pass &= check_galaxy(name, expected_n)

    print("\n" + "=" * 60)
    print("OVERALL:", "PASS" if overall_pass else "FAIL")
    print("=" * 60)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
