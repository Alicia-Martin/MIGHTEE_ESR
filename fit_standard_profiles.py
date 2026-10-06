"""
Fit standard, literature-standard dark matter halo density profiles (NFW,
Einasto, Burkert, pseudo-isothermal) directly to every MIGHTEE galaxy, as a
baseline to compare against the ESR-discovered functions' fit quality.

"Without rho0/rs": each profile is fit in use_physical_scale=False mode --
the formula itself IS rho(r), with its own native amplitude (a0) and scale
radius (a1) as free params, NOT routed through the pipeline's separate
rho0/rs physical-scale convention used for ESR shape-only functions.

Reuses test_all.py's own optimise_fun_direct_nm (stage 1: DIRECT+polish over
the profile's shape params; stage 2: galaxy_params_polish, joint Inc/D) --
confirmed to have no dependency on the ESR function-library bookkeeping
files, so an arbitrary hand-written formula string fits normally. Its
returned chi2_i is the final, joint (shape+Inc+D) negloglike -- see
test_all.py:846-849 (`chi2_i = stage2_chi2`).

DIRECT search box (pmin=-30, pmax=30, log_opt=True, maxfun/maxiter=60000)
mirrors the convention this codebase already uses for amplitude/scale-type
parameters (SPARC's rho0/rs mixed-bounds fix) -- rho_s/r_s need the same
huge-dynamic-range treatment, not ESR's own small shape-parameter box.

Also computes codelen (parameter-cost only, NOT a full DL) for each fit.
negloglike is stage 2's own chi2_i (include_priors=True) taken directly, no
reconstruction -- it already prices -log(p(Inc))-log(p(D)) once, so codelen
must not re-add a prior-density term for Inc/D. Inc and D are therefore NOT
treated as a separate case: they are appended to the same parameter vector
as the shape params (a0, a1) and go through match.py's codelen_from_vector
exactly like any other parameter -- Delta from the Fisher diagonal
(Delta=sqrt(12/Fisher), falling back to match.py's get_sigma_from_integral
when the Fisher diagonal is non-finite/non-positive), then the same
log(|theta|/Delta) precision term. This is deliberately NOT combined with an
"aifeyn" complexity prior into a full DL -- a hand-picked profile has no
aifeyn derived from the same
generative combinatorics as the ESR search (Einasto's fractional power
isn't even expressible in that basis), so bolting one on wouldn't be
apples-to-apples. negloglike + codelen (no aifeyn) IS directly comparable to
the ESR table's own (-logL, Codelen) columns with AIFeyn subtracted back
out -- that comparison is what the summary table below reports.

Parallelized with mpi4py -- same rank/size convention as match.py's own
get_functions() -- so this can run locally (mpirun -np N, N up to your core
count) or on Glamdring unchanged, not tied to a local-only mechanism.

Usage:
    mpirun -np 6 python3 fit_standard_profiles.py   # 6 of 8 local cores, e.g.
    python3 fit_standard_profiles.py                # single-process fallback
Output:
    posterior_check_output/standard_profiles_fit.csv        (per galaxy x profile)
    posterior_check_output/standard_profiles_summary.txt    (population-summed)
"""
import os
import warnings

import numpy as np
import pandas as pd
import jax.numpy as jnp
from mpi4py import MPI

warnings.filterwarnings("ignore")

os.environ.setdefault(
    "ESR_FUNCTION_LIBRARY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "function_library", "core_maths"),
)

import esr.fitting.test_all as test_all
import esr.fitting.match as match_mod
import esr.generation.simplifier as simplifier
from esr.fitting.dm_likelihood import MIGHTEELikelihood

GALAXY_NAMES_FILE = "galaxy_names.txt"
DATA_FILE = "ALL_gbar_gobs_RAR_direct_phot.txt"
OUT_DIR = "posterior_check_output"
SCRATCH_DATA_DIR = os.path.join(OUT_DIR, "standard_profiles_scratch")

# a0 (amplitude/rho_s) and a1 (scale radius/rs) are wrapped in Abs() --
# they're physically positive quantities, but nothing in these formulas
# forces that on its own. check_density only verifies the OUTPUT rho(r) is
# non-negative, which a0 and a1 BOTH going negative can satisfy by sign
# cancellation (e.g. NFW: a0/((x/a1)*(1+x/a1)**2) with both negative still
# gives a positive output) -- a degenerate, physically meaningless
# parametrization that isn't a real fit, just an exploited sign loophole.
# Confirmed empirically: without Abs(), fits converged to a0/a1 pairs like
# [-6.8e-16, -5.25e22] and [-5.86e23, -3.88e-7]. Abs() is the same
# defensive convention nearly every ESR-discovered function in this
# session already uses (pow(Abs(a0),...), etc.) for exactly this reason.
PROFILES = {
    "NFW": "Abs(a0)/((x/Abs(a1))*(1+x/Abs(a1))**2)",
    "Burkert": "Abs(a0)/((1+x/Abs(a1))*(1+(x/Abs(a1))**2))",
    "pseudo-isothermal": "Abs(a0)/(1+(x/Abs(a1))**2)",
    # alpha fixed at 0.18 (a standard literature value) so this stays a
    # clean 2-parameter fit, directly comparable in complexity to the other
    # three -- a free-alpha 3-parameter version is a documented, not yet
    # implemented, follow-up.
    "Einasto (alpha=0.18)": "Abs(a0)*exp(-2/0.18*((x/Abs(a1))**0.18 - 1))",
}

DIRECT_KWARGS = dict(
    tmax=60, pmin=-30, pmax=30, log_opt=True,
    max_param_shape=4, method="Nelder-Mead",
    direct_maxfun=60000, direct_maxiter=60000,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DATA_DIR, exist_ok=True)
comm.Barrier()


def reference_aifeyn(nparam, comp=6):
    """Median aifeyn among REAL ESR comp={comp} functions with exactly
    `nparam` shape parameters -- a grounded reference point for a rough
    full-DL comparison, NOT a computed aifeyn for these specific profiles.
    An exact aifeyn requires ESR's own combinatorial-generation tree
    representation (generation/generator.py's aifeyn_complexity), which
    only exists for functions the generator actually built -- there's no
    reverse parser from an arbitrary hand-written string back to that tree,
    and Einasto's fractional power (x/a1)**0.18 isn't even expressible in
    ESR's basis-function grammar in the first place. Returns (median, n_found).
    """
    fn_dir = os.environ["ESR_FUNCTION_LIBRARY_DIR"]
    with open(f"{fn_dir}/compl_{comp}/unique_equations_{comp}.txt") as f:
        unique_fcns = [l.strip() for l in f if l.strip()]
    with open(f"{fn_dir}/compl_{comp}/all_equations_{comp}.txt") as f:
        all_fcns = [l.strip() for l in f if l.strip()]
    aifeyn_all = np.genfromtxt(f"{fn_dir}/compl_{comp}/aifeyn_{comp}.txt")

    nshapes = simplifier.count_params(unique_fcns, 4)
    matching = [f for f, n in zip(unique_fcns, nshapes) if n == nparam]
    vals = []
    for f in matching:
        try:
            vals.append(aifeyn_all[all_fcns.index(f)])
        except ValueError:
            continue
    vals = np.array(vals)
    return (float(np.median(vals)) if len(vals) else float("nan")), len(vals)


def compute_codelen(likelihood, profile_string, shape_params, inc_fit, d_fit):
    """Parameter-cost codelen (shape params + Inc/D), no aifeyn. negloglike
    is stage 2's own chi2_i (include_priors=True), taken directly with no
    reconstruction -- it already prices the Inc/D prior-density term once.

    Inc and D are NOT a separate case: they're appended to the shape-param
    vector and go through match.py's codelen_from_vector as ordinary
    parameters, exactly like a0/a1 -- same Delta=sqrt(12/Fisher) precision
    term, with match.py's own get_sigma_from_integral fallback when the
    Fisher diagonal doesn't resolve a parameter. There is no separate
    prior-density addend to worry about double-counting: codelen_from_vector
    never added one in the first place (log(|theta|/Delta) is already just
    a precision cost), so nothing here can double-count against chi2_i's
    own -log(p(Inc))-log(p(D)).
    """
    fcn_i, eq, integrated = likelihood.run_sympify(profile_string, tmax=60, try_integration=False)
    nshape = likelihood.nparam_shape
    eq_numpy = test_all.lambdify_equation(eq, nshape)
    n_total = nshape + 2

    p_shape = jnp.asarray(shape_params[:nshape], dtype=jnp.float64)
    theta_full = jnp.concatenate([p_shape, jnp.array([inc_fit, d_fit], dtype=jnp.float64)])
    p = np.asarray(theta_full, dtype=float)
    active_idx = np.arange(n_total)

    hessian_template = likelihood.get_loss(eq_numpy, integrated, value="hessian", include_priors=True)
    Hmat = hessian_template(theta_full, likelihood.xvar, likelihood.yvar, likelihood.yerr_lo, likelihood.yerr_hi)
    fisher_diag = np.asarray(jnp.diag(Hmat), dtype=float)

    Sigma = np.full(n_total, np.inf, dtype=float)
    for j in range(n_total):
        fd = fisher_diag[j]
        if np.isfinite(fd) and fd > 0:
            Sigma[j] = 1.0 / np.sqrt(fd)

    if np.any(~np.isfinite(Sigma[active_idx]) | (Sigma[active_idx] <= 0)):
        loss_template = likelihood.get_loss(eq_numpy, integrated, value="evaluate", include_priors=True)
        chi2_fcn = likelihood.get_wrapped_like(loss_template)

        def fop(theta):
            theta = jnp.asarray(theta, dtype=float)
            return chi2_fcn(theta, likelihood.xvar, likelihood.yvar, likelihood.yerr_lo, likelihood.yerr_hi)

        negloglike_at_p = float(fop(p))
        Sigma = match_mod.get_sigma_from_integral(
            p, Sigma, fop, negloglike_at_p, active_idx=active_idx, fcn_i=fcn_i, number_points=10**3,
        )
        Sigma[np.isnan(Sigma)] = np.inf

    Delta = np.where(np.isfinite(Sigma) & (Sigma > 0), np.sqrt(12.0) * Sigma, np.inf)

    return float(match_mod.codelen_from_vector(p, Delta, active_idx))


def fit_one(galaxy, profile_name, profile_string):
    likelihood = MIGHTEELikelihood(
        data_file=DATA_FILE, name=galaxy,
        run_name=f"standard_profiles/{profile_name}/{galaxy}",
        data_dir=SCRATCH_DATA_DIR, use_physical_scale=False,
    )
    (
        chi2_i, full_params, j_out, count_lowest, success,
        inc_fit, d_fit, stage2_chi2, flag_reason,
    ) = test_all.optimise_fun_direct_nm(profile_string, likelihood, **DIRECT_KWARGS)

    codelen = compute_codelen(likelihood, profile_string, full_params, inc_fit, d_fit)

    return {
        "galaxy": galaxy, "profile": profile_name,
        "negloglike": chi2_i, "codelen": codelen,
        "a0": full_params[0], "a1": full_params[1],
        "inc_fit": inc_fit, "d_fit": d_fit, "n_data": len(likelihood.xvar),
    }


def _fit_one_safe(args):
    galaxy, profile_name, profile_string = args
    try:
        return fit_one(galaxy, profile_name, profile_string)
    except Exception as e:
        print(f"[rank {rank}] [{galaxy} / {profile_name}] FAILED: {type(e).__name__}: {e}", flush=True)
        return {
            "galaxy": galaxy, "profile": profile_name,
            "negloglike": np.nan, "codelen": np.nan, "a0": np.nan, "a1": np.nan,
            "inc_fit": np.nan, "d_fit": np.nan, "n_data": np.nan,
        }


def main():
    with open(GALAXY_NAMES_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]

    tasks = [
        (galaxy, profile_name, profile_string)
        for galaxy in galaxies
        for profile_name, profile_string in PROFILES.items()
    ]
    n_total = len(tasks)

    # Same rank/size slicing convention as match.py's get_functions() --
    # embarrassingly parallel (every (galaxy, profile) fit is independent),
    # so a straight contiguous split is enough; no inter-rank communication
    # needed until the final gather. Also means this can run unchanged on
    # Glamdring via `mpirun -np N python3 fit_standard_profiles.py`, not
    # just locally.
    nLs = int(np.ceil(n_total / float(size)))
    data_start = rank * nLs
    data_end = min((rank + 1) * nLs, n_total)
    my_tasks = tasks[data_start:data_end]

    if rank == 0:
        print(f"{n_total} total fits across {size} ranks (~{len(my_tasks)} per rank)", flush=True)

    rows = []
    for i, task in enumerate(my_tasks):
        rows.append(_fit_one_safe(task))
        if rank == 0 and (i + 1) % 5 == 0:
            print(f"rank 0: {i + 1}/{len(my_tasks)} of its own share done", flush=True)

    my_df = pd.DataFrame(rows)
    part_path = os.path.join(SCRATCH_DATA_DIR, f"standard_profiles_part_{rank}.csv")
    my_df.to_csv(part_path, index=False)

    comm.Barrier()
    if rank != 0:
        return

    df = pd.concat(
        [pd.read_csv(os.path.join(SCRATCH_DATA_DIR, f"standard_profiles_part_{r}.csv")) for r in range(size)],
        ignore_index=True,
    )
    for r in range(size):
        os.remove(os.path.join(SCRATCH_DATA_DIR, f"standard_profiles_part_{r}.csv"))

    csv_path = os.path.abspath(os.path.join(OUT_DIR, "standard_profiles_fit.csv"))
    df.to_csv(csv_path, index=False)

    both_finite = np.isfinite(df["negloglike"]) & np.isfinite(df["codelen"])
    n_finite = df[both_finite].groupby("profile")["negloglike"].size()
    summed_nll = df[both_finite].groupby("profile")["negloglike"].sum()
    summed_codelen = df[both_finite].groupby("profile")["codelen"].sum()

    # All 4 profiles have exactly 2 free shape params (a0, a1) -- Einasto's
    # alpha is fixed, not free -- so one reference value covers all of them.
    ref_aifeyn, ref_n = reference_aifeyn(nparam=2, comp=6)

    summary = pd.DataFrame({
        "n_finite": n_finite,
        "summed_negloglike": summed_nll,
        "summed_codelen": summed_codelen,
        "summed_DL_no_aifeyn": summed_nll + summed_codelen,
        "summed_DL_with_ref_aifeyn": summed_nll + summed_codelen + ref_aifeyn,
    }).sort_values("summed_DL_no_aifeyn")

    summary_lines = [
        f"Standard DM halo profile fits, {len(galaxies)} galaxies, use_physical_scale=False",
        "(population-summed over galaxies with both a finite negloglike AND codelen;",
        " summed_DL_no_aifeyn = summed_negloglike + summed_codelen, lower is better)",
        "",
        summary.to_string(),
        "",
        f"summed_DL_with_ref_aifeyn adds {ref_aifeyn:.3f} once (not per galaxy, matching "
        f"combine_all_galaxies.py's own convention -- aifeyn prices the function FORM, not "
        f"each galaxy's fit) -- the median aifeyn among {ref_n} real ESR comp=6 functions "
        "with the same 2-parameter count as these profiles. This is a grounded REFERENCE "
        "point, not a computed aifeyn for these specific profiles -- an exact aifeyn needs "
        "ESR's own combinatorial-generation tree, which has no reverse parser from an "
        "arbitrary string, and Einasto's fractional power isn't even expressible in ESR's "
        "basis-function grammar in the first place. Use summed_DL_no_aifeyn as the primary, "
        "trustworthy comparison; treat the _with_ref_aifeyn column as illustrative only.",
        "",
        "Compare against the ESR comp=6 winner's own (-logL, Codelen) from the global "
        "ranking table -- subtract its AIFeyn to get the same no-aifeyn DL for a fair comparison.",
    ]
    summary_txt = "\n".join(summary_lines)
    print("\n" + summary_txt)

    summary_path = os.path.abspath(os.path.join(OUT_DIR, "standard_profiles_summary.txt"))
    with open(summary_path, "w") as f:
        f.write(summary_txt + "\n")

    # Absolute paths, explicitly -- OUT_DIR is a relative path, so where
    # this actually lands depends entirely on the cwd `mpirun` was launched
    # from (an easy thing to lose track of on a cluster, e.g. Glamdring).
    print(f"\nFull per-galaxy table (params, negloglike, codelen for EVERY galaxy x profile): {csv_path}")
    print(f"Summary table: {summary_path}")


if __name__ == "__main__":
    main()
