"""
Glamdring array-job worker for the HMC posterior check (see
submit_posterior_hmc_glamdring.sh): runs the (expensive) NUTS sampling for
ONE galaxy and saves the raw result -- samples + convergence stats -- to a
small file. A separate, cheap, serial step (merge_posterior_hmc.py) later
reads every galaxy's raw result, in galaxy_names.txt order, and does the
(fast) plotting into one final PDF -- so the actual sampling, the only
expensive part, is what gets parallelized across cluster cores.

Requires prepare_posterior_hmc_meta.py to have already been run once for this
comp/fcn-string (fallback_log10/nshape/active_names must be IDENTICAL across
every galaxy's job, and computing them needs every galaxy's own row, so it
can't be redone independently -- and inconsistently -- per job).

Usage:
    python run_posterior_hmc_single.py --galaxy ID_20_44925 \
        --meta posterior_check_output/hmc_raw_comp6_1_x_pow_x_x/meta.json \
        --out-dir posterior_check_output/hmc_raw_comp6_1_x_pow_x_x \
        --fn-library-dir /path/to/function_library/core_maths --seed 0
"""
import argparse
import json
import os
import shutil

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--galaxy", required=True)
    p.add_argument("--meta", required=True, help="meta.json from prepare_posterior_hmc_meta.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fn-library-dir", required=True)
    p.add_argument("--results-dir", default=None, help="Base dir holding "
                    "<galaxy>/comp<N>/codelen_matches_comp<N>.dat -- overrides "
                    "plot_posterior_check.py's own RESULTS_DIR default. Must match "
                    "whatever --results-dir prepare_posterior_hmc_meta.py used for "
                    "this same meta.json.")
    p.add_argument("--seed", type=int, default=0, help="Submit script passes the "
                    "galaxy's own index in galaxy_names.txt, so every galaxy gets a "
                    "distinct seed -- matches main()'s own seed=gi convention in the "
                    "serial script.")
    return p.parse_args()


def main():
    args = _parse_args()
    os.environ["ESR_FUNCTION_LIBRARY_DIR"] = args.fn_library_dir
    import plot_posterior_hmc as hmc  # deferred: must import AFTER the env var is set
    import esr.fitting.match as match_mod

    with open(args.meta) as f:
        meta = json.load(f)

    comp = meta["comp"]
    fcn_string = meta["fcn_string"]
    nshape = meta["nshape"]
    active_names = meta["active_names"]
    fallback_log10 = meta["fallback_log10"]
    fcn_index = meta["fcn_index"]

    hmc.grid.COMP = comp
    if args.results_dir is not None:
        hmc.grid.RESULTS_DIR = args.results_dir
    row = hmc.grid.load_galaxy_row(args.galaxy, fcn_index=fcn_index)
    os.makedirs(args.out_dir, exist_ok=True)
    safe_name = args.galaxy.replace("/", "_").replace("+", "p")
    result_path = os.path.join(args.out_dir, f"{safe_name}.json")
    samples_path = os.path.join(args.out_dir, f"{safe_name}.npz")

    # A per-galaxy subdirectory, not hmc.SCRATCH_DATA_DIR directly -- unlike
    # the serial script (one galaxy fully finishes before the next starts,
    # so sharing one scratch dir is harmless), up to ~129 of these run
    # concurrently as separate array jobs and can land on the same compute
    # node. All racing to create the same fitting/ subdirectory under one
    # shared path threw FileExistsError in practice on Glamdring.
    #
    # Wiped and recreated fresh (not just os.makedirs(..., exist_ok=True))
    # because ESR's own Likelihood.__init__ does a bare os.mkdir(like_dir)
    # with no exist_ok -- it needs data_dir to already exist, but also
    # can't tolerate data_dir/fitting/ already existing from a prior
    # attempt at this same galaxy (a resubmitted/retried job). This scratch
    # data is disposable, so wiping it first makes every run idempotent
    # regardless of what a previous attempt left behind.
    scratch_dir = os.path.join(hmc.SCRATCH_DATA_DIR, safe_name)
    shutil.rmtree(scratch_dir, ignore_errors=True)
    os.makedirs(scratch_dir, exist_ok=True)

    if row is None:
        with open(result_path, "w") as f:
            json.dump({"galaxy": args.galaxy, "status": "skipped_no_row"}, f)
        print(f"[{args.galaxy}] no row for this function -- skipped.")
        return

    try:
        likelihood = hmc.MIGHTEELikelihood(
            data_file=hmc.DATA_FILE, name=args.galaxy,
            run_name=f"posterior_hmc/{args.galaxy}/comp{comp}",
            data_dir=scratch_dir, use_physical_scale=True,
        )
    except Exception as e:
        with open(result_path, "w") as f:
            json.dump({"galaxy": args.galaxy, "status": "skipped_likelihood_build_failed",
                       "error": f"{type(e).__name__}: {e}"}, f)
        print(f"[{args.galaxy}] likelihood build failed: {e}")
        return

    n_data = len(likelihood.xvar)
    fcn_i, eq, integrated = likelihood.run_sympify(fcn_string, tmax=60, try_integration=False)
    eq_numpy = match_mod.make_lambdified_eq(eq, nshape)
    theta_ml = hmc.extract_theta_ml(row, nshape)
    fop = hmc.grid.build_fop(likelihood, eq_numpy, integrated, row["inc"], row["d"])

    log_centers, signs, used_fallback = [], [], []
    for name, val in zip(active_names, theta_ml):
        if hmc.is_sane_ml(val):
            log_centers.append(float(np.log10(abs(val))))
            signs.append(float(np.sign(val)))
        else:
            log_centers.append(float(fallback_log10[name]))
            signs.append(1.0)
            used_fallback.append(name)

    try:
        samples, site_names, stats, n_divergent = hmc.run_hmc_for_galaxy(
            fop, signs, log_centers, seed=args.seed)
    except Exception as e:
        with open(result_path, "w") as f:
            json.dump({"galaxy": args.galaxy, "status": "skipped_nuts_failed",
                       "error": f"{type(e).__name__}: {e}"}, f)
        print(f"[{args.galaxy}] NUTS failed: {type(e).__name__}: {e}")
        return

    arr = hmc.samples_to_array(samples, site_names)
    np.savez_compressed(samples_path, arr=arr)

    stats_plain = {site: {k: float(v) for k, v in s.items()} for site, s in stats.items()}
    result = {
        "galaxy": args.galaxy,
        "status": "ok",
        "n_data": n_data,
        "theta_ml": [float(v) for v in theta_ml],
        "active_names": active_names,
        "site_names": site_names,
        "used_fallback": used_fallback,
        "n_divergent": n_divergent,
        "stats": stats_plain,
        "inc": float(row["inc"]),
        "d": float(row["d"]),
    }
    with open(result_path, "w") as f:
        json.dump(result, f)
    max_rhat = max(s["r_hat"] for s in stats_plain.values())
    print(f"[{args.galaxy}] done: N={n_data} divergences={n_divergent} max_r_hat={max_rhat:.3f}")


if __name__ == "__main__":
    main()
