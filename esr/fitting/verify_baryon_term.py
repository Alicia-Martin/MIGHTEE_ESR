"""
Standalone correctness check for the 2026-08-06 baryon-term fix to
MIGHTEELikelihood.get_pred: v_circ(r)^2 = inc_ratio2*baryon_term(r) +
dm_term(r), compared against self.yvar (v_rot, the TOTAL observed velocity).
Before this fix, get_pred returned a DM-only v_circ compared directly
against v_rot -- see get_pred's docstring and the development notes (not included).

Three checks, in increasing order of how much of get_pred they exercise:

  Check A: near-zero DM amplitude -> vpred should collapse to the catalog's
           own baryon curve (self.v_bar), not to ~0 (what the pre-fix code
           would give).
  Check B: the strong correctness check. Exploits linearity of
           rho(r) = a0*f(r) in a0 to solve in closed form for the a0 that
           reproduces the catalog's own precomputed v_DM residual at one
           radius point, then confirms get_pred at that a0 reproduces
           sqrt(v_bar_total^2 + v_DM^2) -- the catalog's own v_circ -- to
           within the same tolerance verify_mass_integral.py already uses.
  Check C: get_pred(D=None, Inc=None) agrees with get_pred(D=distance_true,
           Inc=inc_true) for the same params (validates the new defaulting
           logic added by the fix).

Run directly (from any cwd):
    python esr/fitting/verify_baryon_term.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MIGHTEE_ESR_ROOT = Path(__file__).resolve().parents[2]
if str(_MIGHTEE_ESR_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIGHTEE_ESR_ROOT))

from esr.fitting.dm_likelihood import MIGHTEELikelihood  # noqa: E402

DATA_FILE = _MIGHTEE_ESR_ROOT / "rar_with_ml_direct_phot.csv"
GALAXY_NAME = "J022128.8-042448"
RUN_NAME = "verify_baryon_term"

# Matches verify_mass_integral.py's own integration-accuracy budget.
RELATIVE_TOLERANCE = 1.0e-3

FCN_STRING = "a0/(x*(1+x)**2)"  # NFW-like cusp, linear in a0
RADIUS_INDEX = 0  # first observed radius point


def check_a_near_zero_dm(likelihood, eq_numpy, integrated) -> bool:
    """Near-zero DM amplitude -> vpred should collapse to self.v_bar."""
    vpred = np.asarray(
        likelihood.get_pred(likelihood.xvar, np.asarray([1e-30]), eq_numpy, integrated=integrated)
    )
    v_bar = np.asarray(likelihood.v_bar)
    rel_err = np.abs(vpred - v_bar) / np.maximum(np.abs(v_bar), 1e-30)
    worst = rel_err.max()
    passed = worst <= RELATIVE_TOLERANCE
    print(f"\nCheck A -- near-zero DM amplitude collapses to baryon curve")
    print(f"  worst |vpred - v_bar| / |v_bar| = {worst:.4%}  "
          f"({'PASS' if passed else 'FAIL'} at {RELATIVE_TOLERANCE:.2%} tolerance)")
    return passed


def check_b_reproduces_catalog_v_dm(likelihood, eq_numpy, integrated) -> bool:
    """
    Solve in closed form (via linearity in a0) for the amplitude that
    reproduces the catalog's own v_DM at RADIUS_INDEX, then confirm get_pred
    reproduces the catalog's own v_circ there.
    """
    df = pd.read_csv(DATA_FILE)
    galaxy = df[df["Galaxy"] == GALAXY_NAME].sort_values("Radius_arcsec")
    v_dm_catalog = galaxy["v_DM"].to_numpy()[RADIUS_INDEX]
    v_bar_catalog = galaxy["v_bar_total"].to_numpy()[RADIUS_INDEX]
    v_circ_catalog = np.sqrt(v_bar_catalog ** 2 + v_dm_catalog ** 2)

    k = RADIUS_INDEX
    v_bar_k = float(np.asarray(likelihood.v_bar)[k])

    # rho(r) = a0*f(r) is linear in a0, so dm_term(a0) = a0 * dm_term(a0=1).
    v1 = np.asarray(
        likelihood.get_pred(likelihood.xvar, np.asarray([1.0]), eq_numpy, integrated=integrated)
    )
    dm_term_at_a0_1 = v1[k] ** 2 - v_bar_k ** 2
    a0_target = v_dm_catalog ** 2 / dm_term_at_a0_1

    vpred_final = np.asarray(
        likelihood.get_pred(
            likelihood.xvar, np.asarray([a0_target]), eq_numpy, integrated=integrated,
            D=likelihood.distance_true, Inc=likelihood.inc_true,
        )
    )
    rel_err = abs(vpred_final[k] - v_circ_catalog) / abs(v_circ_catalog)
    passed = rel_err <= RELATIVE_TOLERANCE
    print(f"\nCheck B -- reproduces catalog's own v_DM decomposition at radius index {k}")
    print(f"  catalog v_bar_total={v_bar_catalog:.4f}  v_DM={v_dm_catalog:.4f}  "
          f"v_circ=sqrt(v_bar^2+v_DM^2)={v_circ_catalog:.4f}")
    print(f"  solved a0_target={a0_target:.6e}  get_pred vpred={vpred_final[k]:.4f}")
    print(f"  relative error = {rel_err:.4%}  "
          f"({'PASS' if passed else 'FAIL'} at {RELATIVE_TOLERANCE:.2%} tolerance)")
    return passed


def check_c_none_defaults_match_explicit_fiducial(likelihood, eq_numpy, integrated) -> bool:
    """get_pred(D=None, Inc=None) should agree with the explicit fiducial call."""
    params = np.asarray([3.5e6])
    v_none = np.asarray(
        likelihood.get_pred(likelihood.xvar, params, eq_numpy, integrated=integrated)
    )
    v_explicit = np.asarray(
        likelihood.get_pred(
            likelihood.xvar, params, eq_numpy, integrated=integrated,
            D=likelihood.distance_true, Inc=likelihood.inc_true,
        )
    )
    passed = bool(np.allclose(v_none, v_explicit, rtol=1e-8, atol=1e-8))
    print(f"\nCheck C -- D=None/Inc=None matches explicit fiducial D/Inc")
    print(f"  max |diff| = {np.abs(v_none - v_explicit).max():.3e}  "
          f"({'PASS' if passed else 'FAIL'})")
    return passed


def main() -> int:
    likelihood = MIGHTEELikelihood(
        data_file=str(DATA_FILE),
        name=GALAXY_NAME,
        run_name=RUN_NAME,
        use_physical_scale=False,
    )

    fcn_string, eq, integrated = likelihood.run_sympify(FCN_STRING, try_integration=False)
    eq_numpy = likelihood._make_lambdify(eq, likelihood.nparam_shape)

    print(f"Galaxy: {GALAXY_NAME}")
    print(f"Function: {fcn_string}")
    print(f"Tolerance: {RELATIVE_TOLERANCE:.2%}")

    overall_pass = True
    overall_pass &= check_a_near_zero_dm(likelihood, eq_numpy, integrated)
    overall_pass &= check_b_reproduces_catalog_v_dm(likelihood, eq_numpy, integrated)
    overall_pass &= check_c_none_defaults_match_explicit_fiducial(likelihood, eq_numpy, integrated)

    print("\n" + "=" * 60)
    print("OVERALL:", "PASS" if overall_pass else "FAIL")
    print("=" * 60)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
