"""
Fit standard dark matter halo profiles (NFW, Burkert, pseudo-isothermal,
Einasto with free alpha, pure power law) to every MIGHTEE galaxy through EXACTLY the
production ESR path, so their DL can be compared like for like with the
population ranking from combine_all_galaxies.py / combine_across_comp_global.py.

What is identical to an ESR function in the production run:
  * mode: use_physical_scale=True -- each profile is a parameter-free shape
    f(u), rho(r) = rho0 * f(r/rs), i.e. exactly how the pipeline fits an ESR
    function with nshape=0. rho0, rs, Inc, D are the free parameters.
  * fit: test_all.optimise_fun_direct_nm with the settings run_esr_mightee.py
    passes via test_all.main (tmax=60, pmin=-6, pmax=10, log_opt=True,
    method="DIRECT", max_param_shape=4, default DIRECT budget), including the
    stage-2 joint Inc/D polish.
  * -logL and codelen: match.compute_function_codelen on the same padded
    [shape | pad | rho0, rs | Inc, D] vector and joint Hessian
    (include_priors=True) that match.py builds for a row -- same
    Fisher/eigenvalue check, integral fallback, Delta capping and snapping.
  * population DL: sum over galaxies of (-logL + codelen), plus aifeyn once,
    for the full sample and the same >= N point subsets as
    combine_all_galaxies.py. A profile that fails on any galaxy gets DL = nan,
    as an ESR function would in combine_all_galaxies.py.

aifeyn (function-form cost):
  * pseudo-isothermal 1/(1+u^2) is in the core_maths library itself, first at
    comp 7 as 1/(x*(x + 1/x)) -- its library aifeyn is used directly.
  * NFW and Burkert are not in the library, so their trees are written by hand
    with integers as nodes of their own (aifeyn adds log|n| for each, as for
    library functions like 1/(x**2 + 2*x)): NFW = 1/(x*pow((x + 1),2)), 8 nodes;
    Burkert = 1/((x + 1)*(pow(x,2) + 1)), 10 nodes. With --no-physical-scale the
    amplitude a0 sits inside the formula, so NFW = a0/(x*pow((a1 + x),2)) is 9.
  * Einasto, rho0 exp(-u^alpha) with alpha free, needs exp, which core_maths lacks: aifeyn = nan,
    so only its DL without aifeyn is comparable.

Usage (MPI: each rank fits all profiles to its own share of the galaxies,
rank 0 gathers and writes the outputs; also runs as a single process):
    mpirun -np 16 python3 fit_standard_profiles.py --fn-library-dir <core_maths dir>
    python3 fit_standard_profiles.py
Outputs (in --out-dir):
    standard_profiles_fit.csv       per galaxy x profile
    standard_profiles_summary.txt   population sums, full sample and subsets
"""
import argparse
import os
import warnings

# One CPU thread per MPI rank. Otherwise every rank's JAX/BLAS starts a thread
# pool the size of the whole node, N ranks oversubscribe the cores, and even
# trivial XLA compiles take minutes ("Very slow compile?" warnings). Must be
# set before jax/numpy are imported.
for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np
import pandas as pd
import jax.numpy as jnp
from mpi4py import MPI

warnings.filterwarnings("ignore")

import esr.fitting.test_all as test_all
import esr.fitting.match as match
import esr.generation.generator as generator
from esr.fitting.dm_likelihood import MIGHTEELikelihood
from combine_all_galaxies import DEFAULT_MIN_POINTS, galaxy_subsets, resolve_fn_dir

NEW_DATA_FILE = "ALL_gbar_gobs_RAR_direct_phot_with_5_arc_radius_plus_Halpha_for_ESR.txt"

# Written out in full for the summary table (u = r/rs)
FORMS = {
    "NFW": "rho0 / [u (1+u)^2]",
    "Burkert": "rho0 / [(1+u)(1+u^2)]",
    "pseudo-isothermal": "rho0 / (1+u^2)",
    "Einasto": "rho0 exp[-u^alpha], alpha free",
    "power law": "rho0 u^a0, a0 free",
}
# --no-physical-scale: the formula IS rho(r), with its own amplitude and scale
FORMS_FALSE = {
    "NFW": "a0 / [r (a1+r)^2]",
    "Burkert": "a0 / [(a1+r)(a1^2+r^2)]",
    "pseudo-isothermal": "a0 / (a1+r^2)",
    "Einasto": "a0 exp[-a1 r^a2]",
    "power law": "a0 r^a1",
}

PARAM_LIST = ["a0", "a1", "a2", "a3"]
MAX_FUN_PARAMS = 4


def n_extra(physical):
    return 2 if physical else 0  # rho0, rs only exist in physical-scale mode

# Production settings: run_esr_mightee.py -> test_all.main -> optimise_fun_direct_nm
FIT_KWARGS = dict(tmax=60, pmin=-6, pmax=10, try_integration=False, log_opt=True,
                  max_param_shape=MAX_FUN_PARAMS, method="DIRECT")


def library_aifeyn(fn_dir, comp, row):
    return float(np.genfromtxt(f"{fn_dir}/compl_{comp}/aifeyn_{comp}.txt")[row])


def tree_aifeyn(labels):
    return float(generator.aifeyn_complexity(labels, PARAM_LIST))


def profiles(fn_dir, physical=True):
    if not physical:
        return profiles_no_physical_scale(fn_dir)
    # Integers are nodes of their own (aifeyn adds log|n| for each), as in
    # library functions such as 1/(x**2 + 2*x):
    #   NFW     1/(x*pow((x + 1),2))        inv * x pow + x 1 2           8 nodes
    #   Burkert 1/((x + 1)*(pow(x,2) + 1))  inv * + x 1 + pow x 2 1      10 nodes
    nfw_tree = ["inv", "*", "x", "pow", "+", "x", "1", "2"]
    burkert_tree = ["inv", "*", "+", "x", "1", "+", "pow", "x", "2", "1"]
    return {
        "NFW": ("1/(x*(x + 1)**2)", tree_aifeyn(nfw_tree),
                "hand-built 8-node tree 1/(x*pow((x + 1),2))"),
        "Burkert": ("1/((x + 1)*(x**2 + 1))", tree_aifeyn(burkert_tree),
                    "hand-built 10-node tree 1/((x + 1)*(pow(x,2) + 1))"),
        "pseudo-isothermal": ("1/(x*(x + 1/x))", library_aifeyn(fn_dir, 7, 747),
                              "exact: library comp 7, row 747"),
        # rho ∝ exp(-A r^alpha): A and rs are one degree of freedom (A = rs^-alpha),
        # so in physical-scale form this is rho0 * exp(-u^alpha) with alpha = a0
        # free -- the same family as rho_-2 exp[-(2/alpha)((r/r_-2)^alpha - 1)].
        "Einasto": ("exp(-pow(x, a0))", float("nan"),
                    "not expressible in core_maths (no exp)"),
        # What free-alpha Einasto tends to as alpha -> 0, rs -> 0. Only rho0*rs^-a0
        # is constrained, as for this ESR function in the production runs.
        "power law": ("pow(x,a0)", library_aifeyn(fn_dir, 3, 9),
                      "exact: library comp 3, row 9"),
    }


def profiles_no_physical_scale(fn_dir):
    """--no-physical-scale forms: rho(r) itself, matching how such an ESR run
    writes these profiles (e.g. its rank-1 comp-7 a0/(a1 + x**2)).

    NFW and Burkert need a repeated parameter (a1) and the constant 2, so they are
    not in the library up to comp 10; their aifeyn comes from hand-built trees
    with a1 reused, an upper bound in the same sense as the physical-scale ones.
    """
    # a0/(x*pow((a1 + x),2)): 9 nodes, the integer 2 a node of its own (aifeyn
    # adds log|2| for it, as ESR's aifeyn_complexity does for any integer)
    nfw_tree = ["/", "a0", "*", "x", "pow", "+", "a1", "x", "2"]
    burkert_tree = ["/", "a0", "*", "+", "a1", "x", "+", "*", "a1", "a1", "*", "x", "x"]
    return {
        "NFW": ("a0/(x*(a1 + x)**2)", tree_aifeyn(nfw_tree),
                "hand-built 9-node tree a0/(x*pow((a1 + x),2)), a1 reused"),
        "Burkert": ("a0/((a1 + x)*(a1**2 + x**2))", tree_aifeyn(burkert_tree),
                    "upper bound: hand-built 13-node core_maths tree, a1 reused"),
        "pseudo-isothermal": ("a0/(a1 + x**2)", library_aifeyn(fn_dir, 7, 4598),
                              "exact: library comp 7, row 4598"),
        "Einasto": ("a0*exp(-a1*pow(x, a2))", float("nan"),
                    "not expressible in core_maths (no exp)"),
        "power law": ("a0*pow(x,a1)", library_aifeyn(fn_dir, 5, 216),
                      "exact: library comp 5, row 216"),
    }


def fit_one(galaxy, shape, args, fn_dir):
    likelihood = MIGHTEELikelihood(data_file=args.data_file, name=galaxy,
                                   run_name=f"standard_profiles/{galaxy}", data_dir=os.getcwd(),
                                   use_physical_scale=args.physical, fn_library_dir=fn_dir,
                                   fit_ml=getattr(args, "fit_ml", False))
    (chi2_i, full_params, _, _, _, inc_fit, d_fit, _, _, f_fit) = test_all.optimise_fun_direct_nm(
        shape, likelihood, **FIT_KWARGS)
    if not np.isfinite(chi2_i):
        return np.nan, np.nan, full_params, inc_fit, d_fit, len(likelihood.xvar), f_fit

    # Same vector/indices/Hessian as match.py builds for one row (its
    # fresh-Hessian path), then the same compute_function_codelen call.
    fcn_i, eq, integrated = likelihood.run_sympify(shape, tmax=5, try_integration=False)
    nshape = likelihood.nparam_shape
    eq_numpy = match.make_lambdified_eq(eq, nshape)
    shape_idx, pad_idx, extra_idx, _ = match.build_layout(nshape, MAX_FUN_PARAMS, n_extra(args.physical))
    mpt = MAX_FUN_PARAMS + n_extra(args.physical)
    galaxy_idx = np.arange(mpt, mpt + likelihood.num_galaxy_params)
    extra_idx_ext = np.concatenate([extra_idx, galaxy_idx]).astype(int)
    active_idx_ext = np.concatenate([shape_idx, extra_idx_ext]).astype(int)
    p_ext = np.concatenate([np.asarray(full_params, dtype=float),
                            [inc_fit, d_fit] + ([f_fit] if likelihood.fit_ml else [])])
    # Production saves test_all's fits with fmt="%.7e" and match.py reads them
    # back, so its -logL/codelen are evaluated at parameters rounded to 8
    # significant figures. The asymmetric-error likelihood has a step where
    # the model crosses a data point, and fits often sit exactly on one, so
    # that rounding can move -logL by ~0.3 nats -- round the same way here.
    p_ext = np.array([float(f"{v:.7e}") for v in p_ext])

    xvar, yvar, lo, hi = likelihood.xvar, likelihood.yvar, likelihood.yerr_lo, likelihood.yerr_hi
    H = likelihood.get_loss(eq_numpy, integrated, value="hessian", include_priors=True)(
        jnp.asarray(p_ext[active_idx_ext], dtype=float), xvar, yvar, lo, hi)
    fish = np.zeros((len(p_ext), len(p_ext)))
    fish[np.ix_(active_idx_ext, active_idx_ext)] = np.asarray(H, dtype=float)

    chi2_fcn = likelihood.get_wrapped_like(
        likelihood.get_loss(eq_numpy, integrated, value="evaluate", include_priors=True))

    def fop(theta):
        return chi2_fcn(jnp.asarray(theta, dtype=float)[active_idx_ext], xvar, yvar, lo, hi)

    codelen, negloglike, p_final, _ = match.compute_function_codelen(
        fcn_i, p_ext, fish, shape_idx, extra_idx_ext, active_idx_ext, pad_idx,
        fop, float(fop(p_ext)), number_points=10**3, galaxy_idx=galaxy_idx,
    )
    return negloglike, codelen, p_final, inc_fit, d_fit, len(xvar), f_fit


def fit_one_safe(task, args, fn_dir, rank):
    galaxy, name, shape = task
    try:
        nll, cl, p, inc, d, n, f_ml = fit_one(galaxy, shape, args, fn_dir)
    except Exception as e:
        print(f"[rank {rank}] [{galaxy} / {name}] FAILED: {type(e).__name__}: {e}", flush=True)
        nll, cl, p, inc, d, n, f_ml = (np.nan, np.nan, np.full(MAX_FUN_PARAMS + n_extra(args.physical), np.nan),
                                       np.nan, np.nan, np.nan, np.nan)
    phys = args.physical
    return {"galaxy": galaxy, "profile": name, "negloglike": nll, "codelen": cl,
            "rho0": p[MAX_FUN_PARAMS] if phys else np.nan,
            "rs": p[MAX_FUN_PARAMS + 1] if phys else np.nan,
            # the function's own parameters a0..a2 (physical-scale: alpha for
            # Einasto, the exponent for the power law; otherwise all of them)
            "a0": p[0], "a1": p[1], "a2": p[2],
            "inc_fit": inc, "d_fit": d, "f_fit": f_ml, "n_data": n}


def summarise(df, subsets, prof, forms):
    lines = []
    for min_points, gal_subset in subsets:
        label = "all galaxies" if min_points is None else f">= {min_points} points"
        sub = df[df["galaxy"].isin(gal_subset)]
        rows = []
        for name, (_, aifeyn, _) in prof.items():
            s = sub[sub["profile"] == name]
            ok = np.isfinite(s["negloglike"]) & np.isfinite(s["codelen"])
            nll, cl = s["negloglike"].sum(), s["codelen"].sum()
            # Same rule as combine_all_galaxies.py: any failed galaxy -> no DL
            DL_no = nll + cl if ok.all() and len(s) == len(gal_subset) else np.nan
            rows.append({"profile": name, "form": forms[name], "n_ok": f"{int(ok.sum())}/{len(gal_subset)}",
                         "sum_negloglike": nll, "sum_codelen": cl, "aifeyn": aifeyn,
                         "DL_no_aifeyn": DL_no, "DL": DL_no + aifeyn})
        # Ranked by full DL like the ESR tables; profiles without an aifeyn
        # (no full DL) go last, ordered by DL_no_aifeyn.
        table = pd.DataFrame(rows).sort_values(["DL", "DL_no_aifeyn"], na_position="last").to_string(
            index=False, float_format="%.2f")
        lines += [f"=== {label} ({len(gal_subset)} galaxies) ===", table, ""]
    return lines


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--galaxy-file", default="galaxy_names.txt")
    p.add_argument("--data-file", default=NEW_DATA_FILE,
                   help="Catalogue (default: the new one with the 5 arcsec point and Halpha). "
                        "Compare only against ESR runs made on the same catalogue.")
    p.add_argument("--fn-library-dir", default=os.environ.get("ESR_FUNCTION_LIBRARY_DIR"))
    p.add_argument("--min-points", default=",".join(map(str, DEFAULT_MIN_POINTS)),
                   help="Also sum over galaxies with >= N points (same subsets as "
                        "combine_all_galaxies.py). '' to skip.")
    p.add_argument("--out-dir", default="posterior_check_output")
    p.add_argument("--fit-ml", action="store_true",
                   help="Fit the stellar M/L factor f too. Outputs get a _fml suffix.")
    p.add_argument("--use-physical-scale", dest="physical", action="store_true", default=True,
                   help="rho0 * shape(r/rs) (default; compare with rhoTrue ESR runs).")
    p.add_argument("--no-physical-scale", dest="physical", action="store_false",
                   help="rho(r) with its own parameters, no rho0/rs (compare with rhoFalse "
                        "ESR runs). Outputs get a _rhoFalse suffix.")
    return p.parse_args()


def main():
    args = _parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    fn_dir = resolve_fn_dir(args.fn_library_dir)
    prof = profiles(fn_dir, args.physical)
    forms = FORMS if args.physical else FORMS_FALSE
    suffix = ("" if args.physical else "_rhoFalse") + ("_fml" if args.fit_ml else "")

    with open(args.galaxy_file) as f:
        galaxies = [l.strip() for l in f if l.strip()]
    # Each MPI rank takes every size-th galaxy and fits all profiles to it;
    # rank 0 gathers everything and writes the table and summary.
    my_galaxies = galaxies[rank::size]
    my_tasks = [(g, name, shape) for g in my_galaxies for name, (shape, _, _) in prof.items()]
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"{len(galaxies)} galaxies x {len(prof)} profiles over {size} rank(s), "
              f"{'physical-scale' if args.physical else 'no physical scale'}", flush=True)
    print(f"rank {rank}/{size}: {len(my_galaxies)} galaxies", flush=True)

    rows = []
    for i, task in enumerate(my_tasks):
        rows.append(fit_one_safe(task, args, fn_dir, rank))
        if (i + 1) % len(prof) == 0:
            print(f"rank {rank}: {(i + 1) // len(prof)}/{len(my_galaxies)} galaxies done", flush=True)

    all_rows = comm.gather(rows, root=0)
    if rank != 0:
        return
    df = pd.DataFrame([r for part in all_rows for r in part])
    csv_path = os.path.abspath(os.path.join(args.out_dir, f"standard_profiles_fit{suffix}.csv"))
    df.to_csv(csv_path, index=False)

    min_points_list = [int(n) for n in args.min_points.split(",") if n.strip()]
    subsets = galaxy_subsets(galaxies, args.data_file, min_points_list)
    lines = [
        "Standard profiles fitted through the production ESR path ("
        + ("physical-scale shapes" if args.physical else "no physical scale: rho(r) with its own parameters") + ",",
        "production optimiser settings, match.compute_function_codelen).",
        "DL = sum(-logL + codelen) + aifeyn, directly comparable to the DL column of",
        "global_final_all_comps.dat for the same galaxy sample and catalogue.",
        "",
    ] + summarise(df, subsets, prof, forms) + ["aifeyn sources:"] + [
        f"  {name}: {aifeyn:.4f} ({note})" for name, (_, aifeyn, note) in prof.items()
    ]
    text = "\n".join(lines)
    print("\n" + text)
    summary_path = os.path.abspath(os.path.join(args.out_dir, f"standard_profiles_summary{suffix}.txt"))
    with open(summary_path, "w") as f:
        f.write(text + "\n")
    print(f"\nPer-galaxy table: {csv_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
