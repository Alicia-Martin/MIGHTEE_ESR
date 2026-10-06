"""
Posterior-shape diagnostic for the codelen Gaussian approximation.

The codelen calculation (test_all_Fisher.galaxy_params_codelen, match.py's
codelen_from_vector) assumes each parameter's uncertainty is well described
by a local Gaussian around the ML point (Fisher-matrix / Laplace
approximation). MIGHTEE galaxies have very few datapoints per galaxy
(median 3, several with only 2) -- this script checks whether that
assumption actually holds by computing the REAL joint posterior (on a grid,
flat priors on the ESR/physical params, real truncated-Gaussian priors on
Inc/D held fixed at their joint-fit values) for the population-level
winning function at a given comp, for every galaxy, and comparing each
galaxy's marginal posterior shape against the Gaussian the codelen
calculation actually used.

Usage:
    python3 plot_posterior_check.py

Reads comp=6's "previous run" results (output_glamdring/output_mightee_rhoTrue/),
not the post-fix rerun -- this is a QC check on the DATA, not on the recent
codelen bugfixes.
"""
import os
import sys
import tempfile
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

os.environ.setdefault(
    "ESR_FUNCTION_LIBRARY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "function_library", "core_maths"),
)

import jax
import jax.numpy as jnp
from jax import vmap

from esr.fitting.dm_likelihood import MIGHTEELikelihood
import esr.fitting.match as match_mod

# -----------------------------------------------------------------------------
# Configuration -- comp=6, "previous run" (pre-fix) results, winning function.
# -----------------------------------------------------------------------------
COMP = 6
RESULTS_DIR = "output_glamdring/output_mightee_rhoTrue"  # current, verified-stable
                    # copy -- NOT "new new results" (a one-off snapshot folder found
                    # this session to be missing the Inc/D prior term for ~75/129
                    # galaxies; see the comp1-7 rerun investigation). On any machine
                    # other than this laptop, override via --results-dir (see
                    # prepare_posterior_hmc_meta.py/run_posterior_hmc_single.py) --
                    # this default is a Mac-local path with no reason to exist
                    # elsewhere.
FCN_STRING = "1/(x*pow(x,x))"   # comp=6 rank-0 global winner, comp1-7 cross-complexity ranking
USE_PHYSICAL_SCALE = True
MAX_FUN_PARAMS = 4
GALAXY_NAMES_FILE = "galaxy_names.txt"
DATA_FILE = "ALL_gbar_gobs_RAR_direct_phot.txt"
GRID_N = 25          # points per active dimension
OUT_DIR = "posterior_check_output"
SCRATCH_DATA_DIR = os.path.join(tempfile.gettempdir(), "mightee_posterior_check_scratch")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_DATA_DIR, exist_ok=True)


def find_fcn_index(fcn_string, comp=COMP):
    """0-based row of fcn_string in unique_equations_{comp}.txt -- derived
    from the string itself rather than a hand-maintained constant, so
    changing FCN_STRING can never silently desync from which row actually
    gets loaded (this was a real bug: FCN_INDEX used to be hardcoded
    separately from FCN_STRING, so editing one without the other loaded the
    WRONG function's fitted params/Delta while evaluating a DIFFERENT
    function's likelihood)."""
    unifn_file = os.path.join(
        os.environ["ESR_FUNCTION_LIBRARY_DIR"], f"compl_{comp}", f"unique_equations_{comp}.txt"
    )
    with open(unifn_file) as f:
        unique_fcn = [line.strip() for line in f]
    target = fcn_string.strip()
    for i, fcn in enumerate(unique_fcn):
        if fcn == target:
            return i
    raise ValueError(f"'{fcn_string}' not found in {unifn_file}")


FCN_INDEX = find_fcn_index(FCN_STRING)


def load_galaxy_row(galaxy, fcn_index=FCN_INDEX):
    """Load this galaxy's codelen_matches_comp{COMP}.dat row for fcn_index
    (defaults to FCN_STRING's own index -- pass an explicit fcn_index, from
    find_fcn_index(your_own_fcn_string), when calling from a DIFFERENT
    module that overrides FCN_STRING itself, e.g. plot_posterior_hmc.py)."""
    path = os.path.join(RESULTS_DIR, galaxy, f"comp{COMP}", f"codelen_matches_comp{COMP}.dat")
    if not os.path.exists(path):
        return None
    data = np.atleast_2d(np.genfromtxt(path))
    idx_col = data[:, 2].astype(int)
    matches = np.where(idx_col == fcn_index)[0]
    if len(matches) == 0:
        return None
    row = data[matches[0]]
    negloglike, codelen, index = row[0], row[1], int(row[2])
    max_param_total = 6  # 4 shape slots + rho0, rs (USE_PHYSICAL_SCALE=True)
    params = row[3:3 + max_param_total]
    deltas = row[3 + max_param_total:3 + 2 * max_param_total]
    inc_fit, d_fit = row[3 + 2 * max_param_total], row[3 + 2 * max_param_total + 1]
    if not np.isfinite(negloglike):
        return None
    return {
        "negloglike": negloglike, "codelen": codelen,
        "params": params, "deltas": deltas,
        "inc": inc_fit, "d": d_fit,
    }


def auto_range(fop_1d, theta0, delta0, other_fixed_negloglike, n_tries=8):
    """Geometrically bracket a +/- window where negloglike rises by ~15 nats.

    Seeded from the stored Delta (Sigma = Delta/sqrt(12)) when finite, else
    from a fraction of |theta0|. Falls back to a fixed absolute window if
    theta0 == 0.
    """
    if np.isfinite(delta0) and delta0 > 0:
        halfwidth = max(5.0 * delta0 / np.sqrt(12.0), 1e-6)
    else:
        halfwidth = max(3.0 * abs(theta0), 1.0)

    target_rise = 15.0
    for _ in range(n_tries):
        lo_val = fop_1d(theta0 - halfwidth)
        hi_val = fop_1d(theta0 + halfwidth)
        rise = min(lo_val, hi_val) - other_fixed_negloglike
        if not np.isfinite(rise):
            halfwidth *= 0.5
            continue
        if rise < target_rise:
            halfwidth *= 2.0
        elif rise > 5.0 * target_rise:
            halfwidth *= 0.6
        else:
            break
    return halfwidth


def build_fop(likelihood, eq_numpy, integrated, inc_fit, d_fit):
    loss_template = likelihood.get_loss(
        eq_numpy, integrated, value="evaluate", include_priors=True,
        fixed_galaxy_params=(inc_fit, d_fit),
    )
    chi2_fcn = likelihood.get_wrapped_like(loss_template)

    def fop(theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)
        return chi2_fcn(theta, likelihood.xvar, likelihood.yvar, likelihood.yerr_lo, likelihood.yerr_hi)

    return fop


SATURATION_ABS = 1e8


def is_saturated(theta_val, delta_val):
    """A parameter driven to a numerically-extreme value with unresolved
    Delta isn't "a well-posed but wide" posterior -- it's the function
    degenerating to not using that parameter at all (e.g. pow(Abs(a0),-x)
    -> 0 as a0 -> inf). Gridding it linearly is numerically meaningless
    (float64 overflow) and physically misleading (there is no finite-width
    peak to show). Detect and fix it at its stored value instead of
    including it as a grid dimension.
    """
    return abs(theta_val) > SATURATION_ABS and not np.isfinite(delta_val)


def compute_joint_posterior(fop, theta_ml, ranges, grid_n, free_dims):
    """Evaluate -logL on a grid over `free_dims` only (others fixed at
    theta_ml), return grid axes (length 3, None for fixed dims) + density
    (shape (grid_n,)*len(free_dims))."""
    if len(free_dims) == 0:
        return [None, None, None], None

    axes_free = [np.linspace(theta_ml[k] - ranges[k], theta_ml[k] + ranges[k], grid_n) for k in free_dims]
    mesh = np.meshgrid(*axes_free, indexing="ij")
    flat_free = np.stack([m.ravel() for m in mesh], axis=1)

    n_pts = flat_free.shape[0]
    full = np.tile(theta_ml, (n_pts, 1))
    for col, k in enumerate(free_dims):
        full[:, k] = flat_free[:, col]

    def fop_row(row):
        return fop(row)

    negloglike_flat = np.asarray(vmap(fop_row)(jnp.asarray(full)))
    negloglike_grid = negloglike_flat.reshape([grid_n] * len(free_dims))

    finite = np.isfinite(negloglike_grid)
    if not np.any(finite):
        return [None, None, None], None

    nll_min = np.min(negloglike_grid[finite])
    unnorm = np.where(finite, np.exp(-(negloglike_grid - nll_min)), 0.0)

    dvols = [axes_free[c][1] - axes_free[c][0] for c in range(len(free_dims))]
    total = unnorm.sum() * np.prod(dvols)
    if total <= 0 or not np.isfinite(total):
        return [None, None, None], None
    density = unnorm / total

    axes = [None, None, None]
    for col, k in enumerate(free_dims):
        axes[k] = axes_free[col]
    return axes, density


def marginal_1d(density, axes, free_dims, keep):
    """Integrate out every free dim except `keep` (original param index).
    Returns None if `keep` was fixed (not in free_dims) for this galaxy."""
    if keep not in free_dims:
        return None
    other_positions = [p for p, k in enumerate(free_dims) if k != keep]
    dvols = [axes[free_dims[p]][1] - axes[free_dims[p]][0] for p in other_positions]
    m = density.sum(axis=tuple(other_positions)) * np.prod(dvols) if other_positions else density
    return m


def marginal_2d(density, axes, free_dims, keep_pair):
    """2D marginal over (keep_pair[0], keep_pair[1]), in that axis order.
    Returns None if either is fixed for this galaxy."""
    if keep_pair[0] not in free_dims or keep_pair[1] not in free_dims:
        return None
    other_positions = [p for p, k in enumerate(free_dims) if k not in keep_pair]
    m = density
    for p in sorted(other_positions, reverse=True):
        dvol = axes[free_dims[p]][1] - axes[free_dims[p]][0]
        m = m.sum(axis=p) * dvol
    pos0, pos1 = free_dims.index(keep_pair[0]), free_dims.index(keep_pair[1])
    if pos0 > pos1:
        m = m.T
    return m


def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def main():
    with open(GALAXY_NAMES_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]

    param_labels = ["a0", "rho0", "rs"]
    results = {}  # galaxy -> dict of axes, density, theta_ml, sigma_fisher, n_data

    for gi, galaxy in enumerate(galaxies):
        row = load_galaxy_row(galaxy)
        if row is None:
            continue

        try:
            likelihood = MIGHTEELikelihood(
                data_file=DATA_FILE, name=galaxy,
                run_name=f"posterior_check/{galaxy}/comp{COMP}",
                data_dir=SCRATCH_DATA_DIR, use_physical_scale=USE_PHYSICAL_SCALE,
            )
        except Exception as e:
            print(f"[{galaxy}] likelihood build failed: {e}")
            continue

        n_data = len(likelihood.xvar)

        fcn_i, eq, integrated = likelihood.run_sympify(FCN_STRING, tmax=60, try_integration=False)
        nshape = likelihood.nparam_shape
        eq_numpy = match_mod.make_lambdified_eq(eq, nshape)

        theta_ml = np.array([row["params"][0], row["params"][4], row["params"][5]])
        delta_ml = np.array([row["deltas"][0], row["deltas"][4], row["deltas"][5]])

        fop = build_fop(likelihood, eq_numpy, integrated, row["inc"], row["d"])
        nll_ml = float(fop(theta_ml))
        if not np.isfinite(nll_ml):
            print(f"[{galaxy}] ML point itself gives non-finite -logL, skipping")
            continue

        free_dims = [k for k in range(3) if not is_saturated(theta_ml[k], delta_ml[k])]
        saturated_dims = [k for k in range(3) if k not in free_dims]

        ranges = [0.0, 0.0, 0.0]
        for k in free_dims:
            def fop_1d(val, k=k):
                theta = theta_ml.copy()
                theta[k] = val
                return float(fop(theta))
            ranges[k] = auto_range(fop_1d, theta_ml[k], delta_ml[k], nll_ml)

        axes, density = compute_joint_posterior(fop, theta_ml, ranges, GRID_N, free_dims)
        if density is None:
            print(f"[{galaxy}] posterior grid degenerate (free_dims={free_dims}), skipping")
            continue

        sigma_fisher = np.where(np.isfinite(delta_ml) & (delta_ml > 0), delta_ml / np.sqrt(12.0), np.nan)

        results[galaxy] = {
            "axes": axes, "density": density, "theta_ml": theta_ml,
            "sigma_fisher": sigma_fisher, "n_data": n_data,
            "free_dims": free_dims, "saturated_dims": saturated_dims,
        }
        if (gi + 1) % 20 == 0:
            print(f"{gi + 1}/{len(galaxies)} galaxies processed", flush=True)

    print(f"Computed posteriors for {len(results)}/{len(galaxies)} galaxies.")

    # ---------------------------------------------------------------------
    # Figure 1: overlay every galaxy's 1D marginal (in Fisher-sigma units)
    # against the standard normal the codelen calc assumes.
    # ---------------------------------------------------------------------
    fig, axarr = plt.subplots(1, 3, figsize=(15, 4.5))
    n_data_all = np.array([r["n_data"] for r in results.values()])
    norm = matplotlib.colors.Normalize(vmin=n_data_all.min(), vmax=n_data_all.max())
    cmap = matplotlib.colormaps["viridis"]

    n_saturated_by_dim = {k: 0 for k in range(3)}
    for k, label in enumerate(param_labels):
        ax = axarr[k]
        n_plotted = 0
        for galaxy, r in results.items():
            if k in r["saturated_dims"]:
                n_saturated_by_dim[k] += 1
                continue
            sigma = r["sigma_fisher"][k]
            if not (np.isfinite(sigma) and sigma > 0):
                continue
            m = marginal_1d(r["density"], r["axes"], r["free_dims"], k)
            if m is None:
                continue
            x = (r["axes"][k] - r["theta_ml"][k]) / sigma
            ax.plot(x, m * sigma, color=cmap(norm(r["n_data"])), alpha=0.35, lw=0.8)
            n_plotted += 1
        xg = np.linspace(-6, 6, 400)
        ax.plot(xg, gaussian_pdf(xg, 0, 1), "r--", lw=2.5, label="Gaussian (assumed by codelen)")
        ax.set_xlabel(f"({label} - ML) / sigma_Fisher")
        ax.set_ylabel("density x sigma_Fisher (dimensionless)")
        ax.set_xlim(-6, 6)
        ax.set_title(f"{label}  ({n_plotted} shown, {n_saturated_by_dim[k]} saturated/excluded)")
        if k == 0:
            ax.legend(fontsize=8)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axarr, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("N datapoints for this galaxy")
    fig.suptitle(
        f"comp={COMP} winner \"{FCN_STRING}\": per-galaxy marginal posteriors vs the Fisher/Gaussian "
        f"approximation codelen assumes ({len(results)} galaxies)"
    )
    fig.savefig(os.path.join(OUT_DIR, "all_galaxies_marginals_vs_gaussian.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Figure 2: full corner plots for a few representative galaxies
    # (fewest datapoints, and the best-sampled galaxy for contrast).
    # ---------------------------------------------------------------------
    by_n = sorted(results.items(), key=lambda kv: kv[1]["n_data"])
    picks = []
    seen_n = set()
    for galaxy, r in by_n:
        if r["n_data"] not in seen_n:
            picks.append((galaxy, r))
            seen_n.add(r["n_data"])
        if len(picks) >= 3:
            break
    if by_n:
        picks.append(by_n[-1])  # best-sampled galaxy, for contrast

    for galaxy, r in picks:
        fig, axarr = plt.subplots(3, 3, figsize=(9, 9))
        axes, density, theta_ml, sigma_fisher = r["axes"], r["density"], r["theta_ml"], r["sigma_fisher"]
        free_dims = r["free_dims"]

        for i in range(3):
            for j in range(3):
                ax = axarr[i, j]
                if j > i:
                    ax.axis("off")
                    continue
                if i not in free_dims or (j != i and j not in free_dims):
                    ax.axis("off")
                    if i == j:
                        ax.text(0.5, 0.5, f"{param_labels[i]}\nsaturated\n(={theta_ml[i]:.2e})",
                                ha="center", va="center", fontsize=7, transform=ax.transAxes)
                    continue
                if i == j:
                    m = marginal_1d(density, axes, free_dims, i)
                    ax.plot(axes[i], m, "k-")
                    if np.isfinite(sigma_fisher[i]) and sigma_fisher[i] > 0:
                        ax.plot(axes[i], gaussian_pdf(axes[i], theta_ml[i], sigma_fisher[i]), "r--")
                    ax.set_yticks([])
                else:
                    m2 = marginal_2d(density, axes, free_dims, (j, i))
                    if m2 is None:
                        ax.axis("off")
                        continue
                    ax.contourf(axes[j], axes[i], m2.T, levels=12, cmap="viridis")
                    ax.axvline(theta_ml[j], color="w", lw=0.5, ls=":")
                    ax.axhline(theta_ml[i], color="w", lw=0.5, ls=":")
                if i == 2:
                    ax.set_xlabel(param_labels[j])
                if j == 0 and i != 0:
                    ax.set_ylabel(param_labels[i])
        fig.suptitle(f"{galaxy}  (N={r['n_data']} datapoints)")
        fig.tight_layout()
        safe_name = galaxy.replace("/", "_").replace("+", "p")
        fig.savefig(os.path.join(OUT_DIR, f"corner_{safe_name}.png"), dpi=140)
        plt.close(fig)
        print(f"Saved corner plot for {galaxy} (N={r['n_data']})")

    print(f"\nAll figures saved under {OUT_DIR}/")


if __name__ == "__main__":
    main()
