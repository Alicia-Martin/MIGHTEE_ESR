"""
One-time prep step for a Glamdring-parallelized HMC posterior check (see
submit_posterior_hmc_glamdring.sh): computes the population-level setup that
must be IDENTICAL across every per-galaxy array job -- nshape, the active
parameter list, and the fallback log10 anchors (compute_fallback_log_centers
needs every galaxy's OWN row to take a population median, so it can't be
redone independently -- and inconsistently -- inside each of the 129
parallel jobs) -- and saves it to a small JSON that run_posterior_hmc_single.py
and merge_posterior_hmc.py both read.

Usage:
    python prepare_posterior_hmc_meta.py --comp 6 --fcn-string "1/(x*pow(x,x))" \
        --fn-library-dir /path/to/function_library/core_maths \
        --meta-out posterior_check_output/hmc_raw_comp6_1_x_pow_x_x/meta.json
"""
import argparse
import json
import os


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comp", type=int, required=True)
    p.add_argument("--fcn-string", required=True)
    p.add_argument("--fn-library-dir", required=True, help="Absolute path to "
                    "function_library/core_maths -- required, no Mac-path fallback "
                    "(matches run_esr_mightee.py/submit_mightee_comp_glamdring.sh's "
                    "own convention: addqueue jobs aren't guaranteed to inherit the "
                    "submitting shell's environment).")
    p.add_argument("--meta-out", required=True)
    p.add_argument("--results-dir", default=None, help="Base dir holding "
                    "<galaxy>/comp<N>/codelen_matches_comp<N>.dat -- overrides "
                    "plot_posterior_check.py's own RESULTS_DIR default (a Mac-local "
                    "snapshot-folder path with no reason to exist on any other "
                    "machine). Required on Glamdring: point it at wherever "
                    "run_esr_mightee.py actually wrote this run's output there, "
                    "e.g. fitting/output/output_mightee_rhoTrue.")
    return p.parse_args()


def main():
    args = _parse_args()
    os.environ["ESR_FUNCTION_LIBRARY_DIR"] = args.fn_library_dir
    import plot_posterior_hmc as hmc  # deferred: must import AFTER the env var is set,
                                       # since hmc's own os.environ.setdefault (a Mac
                                       # path) would otherwise win if imported first.

    hmc.grid.COMP = args.comp
    if args.results_dir is not None:
        hmc.grid.RESULTS_DIR = args.results_dir
    fcn_index = hmc.grid.find_fcn_index(args.fcn_string, comp=args.comp)

    with open(hmc.GALAXY_NAMES_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]
    rows = hmc.load_all_rows(galaxies, fcn_index)
    if not rows:
        raise SystemExit(
            f"No galaxy had a usable row under RESULTS_DIR={hmc.grid.RESULTS_DIR!r} "
            f"(comp={args.comp}, fcn_index={fcn_index}). Pass --results-dir pointing "
            f"at wherever this run's codelen_matches_comp{args.comp}.dat files "
            f"actually live on this machine."
        )

    probe_galaxy = next(iter(rows))
    probe_likelihood = hmc.MIGHTEELikelihood(
        data_file=hmc.DATA_FILE, name=probe_galaxy, run_name="posterior_hmc_probe",
        data_dir=hmc.SCRATCH_DATA_DIR, use_physical_scale=True,
    )
    probe_likelihood.run_sympify(args.fcn_string, tmax=60, try_integration=False)
    nshape = int(probe_likelihood.nparam_shape)  # numpy int64 otherwise -- not JSON serializable
    active_names = hmc.active_param_names(nshape)
    fallback_log10 = hmc.compute_fallback_log_centers(rows, nshape)

    meta = {
        "comp": args.comp,
        "fcn_string": args.fcn_string,
        "nshape": nshape,
        "active_names": active_names,
        "fallback_log10": fallback_log10,
        "fcn_index": fcn_index,
        "n_galaxies_with_rows": len(rows),
        "n_galaxies_total": len(galaxies),
    }
    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"nshape={nshape}  active_names={active_names}")
    print(f"fallback_log10={fallback_log10}")
    print(f"{len(rows)}/{len(galaxies)} galaxies have a usable row for this function.")
    print(f"Saved: {args.meta_out}")


if __name__ == "__main__":
    main()
