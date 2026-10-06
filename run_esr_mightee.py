import argparse
import os

import pandas as pd

import esr.fitting.test_all as test_all
import esr.fitting.test_all_Fisher as test_all_Fisher
import esr.fitting.match as match
import esr.fitting.combine_DL as combine_DL

from esr.fitting.dm_likelihood import MIGHTEELikelihood


def run_mightee_pipeline_for_galaxy(
    galaxy_name,
    comp,
    data_file,
    run_name,
    fn_set="core_maths",
    try_integration=False,
    log_opt=False,
    method="DIRECT",
    use_physical_scale=True,
    fn_library_dir=None,
):
    """
    Run:
      1) test_all
      2) test_all_Fisher
      3) match
      4) combine_DL

    for a single MIGHTEE galaxy.
    """

    likelihood = MIGHTEELikelihood(data_file=data_file, name=galaxy_name, run_name=run_name, data_dir=os.getcwd(), fn_set=fn_set, use_physical_scale=use_physical_scale, fn_library_dir=fn_library_dir,)

    print(f"Running test_all for {galaxy_name} ...")
    test_all.main(comp, likelihood, tmax=60, try_integration=try_integration, log_opt=log_opt, method=method,)

    print(f"Running test_all_Fisher for {galaxy_name} ...")
    test_all_Fisher.main(comp, likelihood, tmax=5, try_integration=try_integration,)

    print(f"Running match for {galaxy_name} ...")
    match.main( comp, likelihood, tmax=5, try_integration=try_integration,)

    print(f"Running combine_DL for {galaxy_name} ...")
    combine_DL.main(comp, likelihood)

    print("Done.")

def run_mightee_pipeline_all_galaxies(
    data_file,
    run_name,
    fn_set="core_maths",
    try_integration=False,
    log_opt=False,
    method="DIRECT",
    comp=3,
    use_physical_scale=True,
    fn_library_dir=None,
):
    """
    Run the full 4-step pipeline (test_all -> test_all_Fisher -> match ->
    combine_DL) once per galaxy in data_file.

    MIGHTEELikelihood is fundamentally single-galaxy (_load_galaxy_data
    filters the dataframe by Galaxy == name; every array it builds --
    self.xvar, self.inc_true, etc. -- is shaped for exactly one galaxy's
    rotation curve), so there is no "load every galaxy into one likelihood"
    mode to ask for -- this loops the existing, working single-galaxy
    driver instead. Each galaxy gets its OWN run_name (base run_name,
    suffixed with the galaxy name and nested by comp): run_name maps
    directly to an output_{run_name}/ directory (via os.makedirs, so the
    nesting itself is safe), so galaxies/comps sharing one run_name would
    silently overwrite each other's chi2_comp{N}weights_{rank}.dat as the
    loop progressed, not just risk a naming collision. The rho0/rs setting
    is folded into run_name too (rhoTrue/rhoFalse), for the same reason --
    two runs of the same galaxy/comp that differ only in use_physical_scale
    would otherwise land in the identical output_{run_name}/ directory.
    Layout: output_{run_name}_{rhoTrue|rhoFalse}/{galaxy}/comp{comp}/ -- one
    top-level folder per galaxy, comps nested inside it, instead of a
    separate top-level folder per (galaxy, comp) pair.
    """

    df = pd.read_csv(data_file, sep=None, engine="python")
    galaxy_names = df["Galaxy"].unique().tolist()
    print(f"Running pipeline for {len(galaxy_names)} galaxies in {data_file} ...")

    rho_tag = "rhoTrue" if use_physical_scale else "rhoFalse"

    for i, galaxy_name in enumerate(galaxy_names, 1):
        print(f"\n=== Galaxy {i}/{len(galaxy_names)}: {galaxy_name} ===")
        run_mightee_pipeline_for_galaxy(
            galaxy_name=galaxy_name,
            comp=comp,
            data_file=data_file,
            run_name=f"{run_name}_{rho_tag}/{galaxy_name}/comp{comp}",
            fn_set=fn_set,
            try_integration=try_integration,
            log_opt=log_opt,
            method=method,
            use_physical_scale=use_physical_scale,
            fn_library_dir=fn_library_dir,
        )

    print("\nAll galaxies done.")

def _parse_args():
    p = argparse.ArgumentParser(
        description="Run the MIGHTEE ESR pipeline (test_all -> test_all_Fisher -> "
                     "match -> combine_DL) for one galaxy at one complexity. "
                     "Intended to be called once per cluster array-job task "
                     "(see submit_mightee_comp.sh) as well as for local runs."
    )
    p.add_argument("--galaxy", required=True, help="Galaxy name, must match the "
                    "'Galaxy' column in --data-file.")
    p.add_argument("--comp", required=True, type=int, help="Complexity to run.")
    p.add_argument("--data-file", default="ALL_gbar_gobs_RAR_direct_phot.txt",
                    help="Rotation-curve catalogue (see README for the expected columns).")
    p.add_argument("--run-name", default="mightee", help="Base run name; the actual "
                    "run_name used is '{run-name}_rho{True|False}/{galaxy}/comp{comp}', "
                    "matching run_mightee_pipeline_all_galaxies's own nesting so output "
                    "directories look identical either way -- one top-level folder per "
                    "galaxy, with each comp nested inside it as output_{run-name}_"
                    "rho{True|False}/{galaxy}/comp{comp}/, instead of a separate "
                    "top-level folder per (galaxy, comp) pair.")
    p.add_argument("--fn-set", default="core_maths")
    p.add_argument("--method", default="DIRECT")
    p.add_argument("--log-opt", dest="log_opt", action="store_true", default=True)
    p.add_argument("--no-log-opt", dest="log_opt", action="store_false")
    p.add_argument("--use-physical-scale", dest="use_physical_scale", action="store_true",
                    default=True, help="Fit rho0/rs as physical-scale amplitude/scale "
                    "parameters (production default). Encoded into run_name as "
                    "'_rhoTrue' so output dirs never collide with a --no-physical-scale "
                    "run of the same galaxy/comp.")
    p.add_argument("--no-physical-scale", dest="use_physical_scale", action="store_false",
                    help="Shape-only fit, no rho0/rs. Encoded into run_name as "
                    "'_rhoFalse'.")
    p.add_argument("--fn-library-dir", default=os.environ.get("ESR_FUNCTION_LIBRARY_DIR"),
                    help="Absolute path to the function_library/core_maths directory "
                    "(the ESR function library, downloaded separately -- see README). "
                    "Defaults to $ESR_FUNCTION_LIBRARY_DIR if set, otherwise to "
                    "<repo>/function_library/core_maths. Pass this explicitly in "
                    "cluster submit scripts rather than relying on addqueue/sbatch to "
                    "inherit the submitting shell's env.")
    p.add_argument("--rerun", action="store_true", help="Run even if this galaxy/comp "
                    "already has a results_pretty_{comp}.txt (combine_DL.py's own final "
                    "output) under its output directory. Without this flag, an "
                    "already-done galaxy is skipped -- lets a resubmitted array job "
                    "not re-burn allocation on finished galaxies.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rho_tag = "rhoTrue" if args.use_physical_scale else "rhoFalse"
    run_name = f"{args.run_name}_{rho_tag}/{args.galaxy}/comp{args.comp}"

    # Mirrors likelihood.py's own out_dir construction (like_dir = data_dir/fitting/,
    # out_dir = like_dir/output/output_{run_name}), since combine_DL.py writes
    # results_pretty_{comp}.txt there only once the full 4-stage pipeline for this
    # galaxy/comp has actually finished.
    results_marker = os.path.join(
        os.getcwd(), "fitting", "output", f"output_{run_name}",
        f"results_pretty_{args.comp}.txt",
    )
    if not args.rerun and os.path.exists(results_marker):
        print(f"Skipping {args.galaxy} comp={args.comp}: already done "
              f"({results_marker}). Pass --rerun to force.")
    else:
        run_mightee_pipeline_for_galaxy(
            galaxy_name=args.galaxy,
            comp=args.comp,
            data_file=args.data_file,
            run_name=run_name,
            fn_set=args.fn_set,
            log_opt=args.log_opt,
            method=args.method,
            use_physical_scale=args.use_physical_scale,
            fn_library_dir=args.fn_library_dir,
        )