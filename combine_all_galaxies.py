import argparse
import csv
import os

import numpy as np
import pandas as pd
from prettytable import PrettyTable


# Every run also saves the ranking restricted to galaxies with at least this
# many radial points (i.e. more than 3 and more than 4), each in its own
# GLOBAL_minpts{N}/ folder, to check how much the preferred functions depend
# on the poorly sampled galaxies. Only the full-sample table is printed.
DEFAULT_MIN_POINTS = [4, 5]


def global_dir_name(min_points):
    # Filtered combinations get their own folder so they never overwrite the
    # full-sample GLOBAL ranking (combine_across_comp_global.py reads the same name).
    return "GLOBAL" if not min_points else f"GLOBAL_minpts{min_points}"


def galaxy_subsets(galaxies, data_file, min_points_list):
    """[(None, all galaxies), (N, galaxies with >= N points), ...].

    data_file must be the catalogue the per-galaxy fits were run on, since the
    point count is what each galaxy's -logL was computed from.
    """
    subsets = [(None, galaxies)]
    if not min_points_list:
        return subsets
    counts = pd.read_csv(data_file, sep=None, engine="python").groupby("Galaxy").size()
    not_in_file = [g for g in galaxies if g not in counts.index]
    if not_in_file:
        print(f"WARNING: {len(not_in_file)} galaxies not in {data_file}, left out of the "
              f"min-points subsets: {not_in_file}")
    for n in min_points_list:
        kept = [g for g in galaxies if counts.get(g, 0) >= n]
        print(f">= {n} points: {len(kept)}/{len(galaxies)} galaxies (counts from {data_file}).")
        subsets.append((n, kept))
    return subsets


DATA_FILE = "ALL_gbar_gobs_RAR_direct_phot_with_5_arc_radius_plus_Halpha_for_ESR.txt"


def resolve_fn_dir(fn_library_dir):
    # explicit > $ESR_FUNCTION_LIBRARY_DIR > <repo>/function_library/core_maths
    return fn_library_dir or os.environ.get("ESR_FUNCTION_LIBRARY_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "function_library", "core_maths")


def galaxy_out_dir(data_dir, base_run_name, rho_tag, galaxy, comp):
    # Mirrors run_esr_mightee.py's own run_name/out_dir construction exactly
    # (output_{base}_{rho}/{galaxy}/comp{comp}/) -- both must stay in sync.
    run_name = f"{base_run_name}_{rho_tag}/{galaxy}/comp{comp}"
    return os.path.join(data_dir, "fitting", "output", "output_" + run_name)


def max_param_total(comp, use_physical_scale):
    # Matches combine_DL.py's own formula exactly -- not used for column
    # slicing here (we only need columns 0:3 of codelen_matches), kept for
    # reference/future use if per-galaxy params are ever wanted in this report.
    max_param_shape = int(max(4, np.floor((comp - 1) / 2)))
    n_extra = 2 if use_physical_scale else 0
    return max_param_shape + n_extra


def combine_one_comp(comp, galaxies, data_dir, base_run_name, use_physical_scale, fn_dir,
                     cache=None):
    rho_tag = "rhoTrue" if use_physical_scale else "rhoFalse"
    unifn_file = os.path.join(fn_dir, f"compl_{comp}", f"unique_equations_{comp}.txt")
    allfn_file = os.path.join(fn_dir, f"compl_{comp}", f"all_equations_{comp}.txt")
    aifeyn_file = os.path.join(fn_dir, f"compl_{comp}", f"aifeyn_{comp}.txt")

    with open(unifn_file) as f:
        unique_fcn = f.read().splitlines()
    with open(allfn_file) as f:
        all_fcn = f.read().splitlines()
    # aifeyn is per ALL_equations row (one value per variant, repeated across
    # sign-combo variants of the same unique function) -- combine_DL.py itself
    # indexes it with the same per-variant boolean mask it uses for
    # negloglike/codelen, so it must be the same length/row-order as
    # codelen_matches, not len(unique_fcn). Mirrored here, not re-derived.
    aifeyn = np.atleast_1d(np.genfromtxt(aifeyn_file))

    n_variants = len(all_fcn)
    negloglike_sum = np.zeros(n_variants)
    codelen_sum = np.zeros(n_variants)
    index_ref = None
    n_included = 0
    missing_galaxies = []
    mismatched_galaxies = []

    for galaxy in galaxies:
        out_dir = galaxy_out_dir(data_dir, base_run_name, rho_tag, galaxy, comp)
        matches_file = os.path.join(out_dir, f"codelen_matches_comp{comp}.dat")
        if cache is not None and galaxy in cache:
            data = cache[galaxy]
        elif not os.path.exists(matches_file):
            data = None
        else:
            # Only columns 0:3 are used; caching just those lets the min-points
            # subsets reuse this read instead of re-parsing every file.
            data = np.atleast_2d(np.genfromtxt(matches_file))[:, :3]
        if cache is not None:
            cache[galaxy] = data
        if data is None:
            missing_galaxies.append(galaxy)
            continue

        if data.shape[0] != n_variants:
            mismatched_galaxies.append((galaxy, f"{data.shape[0]} rows, expected {n_variants}"))
            continue

        negloglike, codelen, index = data[:, 0], data[:, 1], data[:, 2]

        if index_ref is None:
            index_ref = index
        elif not np.array_equal(index_ref, index):
            mismatched_galaxies.append((galaxy, "variant index order does not match other galaxies"))
            continue

        negloglike_sum += negloglike
        codelen_sum += codelen
        n_included += 1

    return {
        "unique_fcn": unique_fcn,
        "all_fcn": all_fcn,
        "aifeyn": aifeyn,
        "index_ref": index_ref,
        "negloglike_sum": negloglike_sum,
        "codelen_sum": codelen_sum,
        "n_included": n_included,
        "n_total": len(galaxies),
        "missing_galaxies": missing_galaxies,
        "mismatched_galaxies": mismatched_galaxies,
    }


def find_best_variant_per_function(result):
    unique_fcn = result["unique_fcn"]
    all_fcn = result["all_fcn"]
    aifeyn = result["aifeyn"]
    index_ref = result["index_ref"]
    negloglike_sum = result["negloglike_sum"]
    codelen_sum = result["codelen_sum"]

    # DL summed over every included galaxy's negloglike+codelen, but aifeyn
    # (the function-complexity prior) added exactly once per variant row --
    # it's a property of the function form, not of any individual galaxy fit.
    DL = negloglike_sum + codelen_sum + aifeyn

    n_unique = len(unique_fcn)
    fcn_min = [None] * n_unique
    DL_min = np.full(n_unique, np.nan)
    negloglike_min = np.full(n_unique, np.nan)
    codelen_min = np.full(n_unique, np.nan)
    aifeyn_min = np.full(n_unique, np.nan)

    if index_ref is None:
        # No galaxy had this comp -- nothing to rank.
        return fcn_min, DL_min, negloglike_min, codelen_min, aifeyn_min

    for i in range(n_unique):
        mask = index_ref == i
        if not np.any(mask):
            continue
        DL_i = DL[mask]
        if np.sum(~np.isnan(DL_i)) == 0:
            continue
        best_local = np.nanargmin(DL_i)
        fcn_all_i = [all_fcn[j] for j in range(len(mask)) if mask[j]]
        fcn_min[i] = fcn_all_i[best_local]
        DL_min[i] = DL_i[best_local]
        negloglike_min[i] = negloglike_sum[mask][best_local]
        codelen_min[i] = codelen_sum[mask][best_local]
        aifeyn_min[i] = aifeyn[mask][best_local]

    return fcn_min, DL_min, negloglike_min, codelen_min, aifeyn_min


def write_report(comp, result, fcn_min, DL_min, negloglike_min, codelen_min, aifeyn_min,
                  data_dir, base_run_name, rho_tag, min_points=None, verbose=True):
    out_dir = os.path.join(data_dir, "fitting", "output", f"output_{base_run_name}_{rho_tag}",
                            global_dir_name(min_points), f"comp{comp}")
    os.makedirs(out_dir, exist_ok=True)

    mask = ~np.isnan(DL_min)
    order = np.argsort(DL_min[mask])
    idx_sorted = np.where(mask)[0][order]

    ptab = PrettyTable()
    ptab.field_names = ["Rank", "Function", "DL (summed)", "-logL (summed)", "Codelen (summed)", "AIFeyn"]

    dat_path = os.path.join(out_dir, f"global_final_{comp}.dat")
    with open(dat_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        for rank, i in enumerate(idx_sorted):
            row = [rank, fcn_min[i], DL_min[i], negloglike_min[i], codelen_min[i], aifeyn_min[i]]
            writer.writerow(row)
            if rank < 200:
                ptab.add_row([rank, fcn_min[i], f"{DL_min[i]:.2f}", f"{negloglike_min[i]:.2f}",
                              f"{codelen_min[i]:.2f}", f"{aifeyn_min[i]:.2e}"])

    if verbose:
        print(ptab)
    with open(os.path.join(out_dir, f"global_results_pretty_{comp}.txt"), "w") as f:
        print(ptab, file=f)
    print(f"Saved: {dat_path}")
    if not verbose:
        best = idx_sorted[0] if len(idx_sorted) else None
        if best is not None:
            print(f"  >= {min_points} points: {result['n_included']}/{result['n_total']} galaxies, "
                  f"best = {fcn_min[best]} (DL {DL_min[best]:.2f})")
        return

    n_missing = len(result["missing_galaxies"])
    n_mismatched = len(result["mismatched_galaxies"])
    print(f"\ncomp={comp}: {result['n_included']}/{result['n_total']} galaxies included "
          f"({n_missing} missing, {n_mismatched} mismatched/corrupted).")
    if result["missing_galaxies"]:
        print(f"  Missing (not yet run or still running): {result['missing_galaxies']}")
    if result["mismatched_galaxies"]:
        print(f"  Mismatched/corrupted (excluded from sum): {result['mismatched_galaxies']}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Combine per-galaxy MIGHTEE ESR fits into one population-level "
                     "ranking per complexity: sum negloglike+codelen across all "
                     "galaxies for each function variant, add the aifeyn complexity "
                     "prior exactly once, and rank by the summed DL. Also reports "
                     "which galaxies are missing or produced a mismatched/corrupted "
                     "codelen_matches file for that comp, as a run-status/QC check."
    )
    p.add_argument("--comps", required=True, help="Comma-separated list of complexities, e.g. 3,4,5,6,7,8")
    p.add_argument("--galaxy-file", default="galaxy_names.txt")
    p.add_argument("--data-dir", default=os.getcwd())
    p.add_argument("--run-name", default="mightee", help="Base run name (before the "
                    "rho tag), matching what was passed to run_esr_mightee.py's own --run-name.")
    p.add_argument("--use-physical-scale", dest="use_physical_scale", action="store_true", default=True)
    p.add_argument("--no-physical-scale", dest="use_physical_scale", action="store_false")
    p.add_argument("--fn-library-dir", default=os.environ.get("ESR_FUNCTION_LIBRARY_DIR"))
    p.add_argument("--min-points", default=",".join(map(str, DEFAULT_MIN_POINTS)),
                    help="Comma-separated point thresholds: besides the full sample (GLOBAL/), "
                    "also save the ranking over galaxies with at least N points to "
                    "GLOBAL_minpts{N}/ for each N. Default '4,5' (more than 3, more than 4); "
                    "pass '' to skip.")
    p.add_argument("--data-file", default=DATA_FILE, help="Catalogue "
                    "used to count points for --min-points -- must be the one the fits ran on.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fn_dir = resolve_fn_dir(args.fn_library_dir)
    rho_tag = "rhoTrue" if args.use_physical_scale else "rhoFalse"

    with open(args.galaxy_file) as f:
        galaxies = [line.strip() for line in f if line.strip()]
    min_points_list = [int(n) for n in args.min_points.split(",") if n.strip()]
    subsets = galaxy_subsets(galaxies, args.data_file, min_points_list)

    comps = [int(c) for c in args.comps.split(",")]
    for comp in comps:
        print(f"\n=== comp={comp} ===")
        cache = {}
        for min_points, gal_subset in subsets:
            result = combine_one_comp(comp, gal_subset, args.data_dir, args.run_name,
                                       args.use_physical_scale, fn_dir, cache=cache)
            fcn_min, DL_min, negloglike_min, codelen_min, aifeyn_min = find_best_variant_per_function(result)
            write_report(comp, result, fcn_min, DL_min, negloglike_min, codelen_min, aifeyn_min,
                         args.data_dir, args.run_name, rho_tag, min_points,
                         verbose=min_points is None)
