import argparse
import csv
import os
import re
from itertools import combinations

import numpy as np

from combine_all_galaxies import galaxy_out_dir, max_param_total, resolve_fn_dir


# Step 1 of the global/local scheme (Martin et al. 2026, CLASH paper Sec. on
# local and global parameters): using only the existing all-local per-galaxy
# fits, bound how much the summed DL could drop if a subset of a function's
# SHAPE parameters were shared by every galaxy, and keep the (function,
# subset) pairs that could then reach the top-K population ranking. Only
# those get the expensive nested re-fit (step 2).
#
# Per-parameter cost is match.py's own codelen_from_vector: log(2) +
# log(|theta|/Delta) for a used, resolved parameter, 0 if snapped to zero or
# unresolved. Globalising parameter j replaces N per-galaxy costs with one
# global cost whose minimum is log(2) (|theta*| = Delta*), so
#     max_gain_j = sum_a cost_{a,j} - log(2)
# which is the paper's Eq. (globalise) written with each galaxy's actual cost.
# Optimistic because it assumes -logL does not change after sharing.
#
# Only shape parameters a0..a{n-1} are candidates: with --use-physical-scale
# the profile is rho0 * f(r/rs; a), so the a_i are dimensionless shape
# parameters and sharing them is meaningful, while rho0/rs (mass, size) and
# Inc/D (per-galaxy geometry) stay local.


def n_shape_params(fcn):
    return len(set(re.findall(r"a(\d+)", fcn)))


def load_allfcn_and_aifeyn(fn_dir, comp):
    with open(os.path.join(fn_dir, f"compl_{comp}", f"all_equations_{comp}.txt")) as f:
        all_fcn = [s.replace("'", "") for s in f.read().splitlines()]
    aifeyn = np.atleast_1d(np.genfromtxt(os.path.join(fn_dir, f"compl_{comp}", f"aifeyn_{comp}.txt")))
    return all_fcn, aifeyn


def accumulate_galaxies(comp, galaxies, data_dir, base_run_name, use_physical_scale, n_variants):
    """Sum negloglike, codelen and per-shape-parameter cost over galaxies, per variant row.

    Mirrors combine_all_galaxies.combine_one_comp's inclusion rules (missing
    file / wrong row count / mismatched index order -> excluded), so the
    all-local DL here is the same number that ranking uses.
    """
    rho_tag = "rhoTrue" if use_physical_scale else "rhoFalse"
    mpt = max_param_total(comp, use_physical_scale)
    n_extra = 2 if use_physical_scale else 0
    max_fun_params = mpt - n_extra
    n_cols = 3 + 2 * mpt + 5  # negloglike, codelen, index | params | Deltas | inc, d, Nconv, Niter, times

    negloglike_sum = np.zeros(n_variants)
    codelen_sum = np.zeros(n_variants)
    shape_cost_sum = np.zeros((n_variants, max_fun_params))
    index_ref = None
    included, skipped = [], []

    for galaxy in galaxies:
        path = os.path.join(galaxy_out_dir(data_dir, base_run_name, rho_tag, galaxy, comp),
                            f"codelen_matches_comp{comp}.dat")
        if not os.path.exists(path):
            skipped.append((galaxy, "missing"))
            continue
        data = np.atleast_2d(np.genfromtxt(path))
        # _fml runs have one extra column (f, after inc, d); only the first 3 + 2*mpt are read here
        if data.shape[0] != n_variants or data.shape[1] not in (n_cols, n_cols + 1):
            skipped.append((galaxy, f"shape {data.shape}, expected {(n_variants, n_cols)} or +1 col (f_ml)"))
            continue
        index = data[:, 2]
        if index_ref is None:
            index_ref = index
        elif not np.array_equal(index_ref, index):
            skipped.append((galaxy, "variant index order differs"))
            continue

        p = data[:, 3:3 + max_fun_params]
        Delta = data[:, 3 + mpt:3 + mpt + max_fun_params]
        with np.errstate(divide="ignore", invalid="ignore"):
            cost = np.log(2.0) + np.log(np.abs(p) / Delta)
        used = np.isfinite(p) & (p != 0) & np.isfinite(Delta) & (Delta > 0)
        cost = np.where(used, cost, 0.0)

        negloglike_sum += data[:, 0]
        codelen_sum += data[:, 1]
        shape_cost_sum += cost
        included.append(galaxy)

    return negloglike_sum, codelen_sum, shape_cost_sum, index_ref, included, skipped


def ranking_threshold(DL_by_comp, top_k):
    """DL of the top_k-th entry of the all-local ranking across the given comps.

    Same de-duplication as combine_across_comp_global.py (identical summed
    negloglike == same fit padded with a no-op term, keep the lowest DL).
    """
    best = {}
    for DL, negloglike in DL_by_comp:
        key = round(negloglike, 6)
        if key not in best or DL < best[key]:
            best[key] = DL
    ranked = np.sort(np.array(list(best.values())))
    return ranked[min(top_k, len(ranked)) - 1]


def screen_comp(comp, all_fcn, aifeyn, negloglike_sum, codelen_sum, shape_cost_sum, threshold,
                max_subset_size):
    DL_local = negloglike_sum + codelen_sum + aifeyn
    gain_per_param = shape_cost_sum - np.log(2.0)

    candidates = []
    for row in np.where(np.isfinite(DL_local))[0]:
        nshape = n_shape_params(all_fcn[row])
        for r in range(1, min(max_subset_size, nshape) + 1):
            for subset in combinations(range(nshape), r):
                max_gain = float(np.sum(gain_per_param[row, list(subset)]))
                if max_gain <= 0:
                    continue
                DL_opt = DL_local[row] - max_gain
                if DL_opt < threshold:
                    candidates.append({
                        "comp": comp, "row": int(row), "function": all_fcn[row],
                        "global_idx": list(subset), "DL_local": float(DL_local[row]),
                        "negloglike_sum": float(negloglike_sum[row]),
                        "max_gain": max_gain, "DL_optimistic": float(DL_opt),
                    })
    return candidates


def _parse_args():
    p = argparse.ArgumentParser(
        description="Screen which (function, shape-parameter subset) pairs could "
                    "enter the top-K population ranking if those parameters were "
                    "shared (global) across all galaxies, using only the existing "
                    "all-local codelen_matches_comp{N}.dat files. Step 1 of the "
                    "global/local scheme; survivors go to the nested re-fit."
    )
    p.add_argument("--comps", required=True, help="Comma-separated complexities, e.g. 1,2,3,4,5,6,7")
    p.add_argument("--top-k", type=int, default=10, help="Keep candidates whose optimistic "
                   "DL would beat the top-K-th all-local DL across --comps.")
    p.add_argument("--max-subset-size", type=int, default=4)
    p.add_argument("--galaxy-file", default="galaxy_names.txt")
    p.add_argument("--data-dir", default=os.getcwd())
    p.add_argument("--out-dir", default=None, help="Defaults to "
                   "fitting/output/output_{run}_{rho}/MIXED under --data-dir.")
    p.add_argument("--run-name", default="mightee")
    p.add_argument("--use-physical-scale", dest="use_physical_scale", action="store_true", default=True)
    p.add_argument("--no-physical-scale", dest="use_physical_scale", action="store_false")
    p.add_argument("--fn-library-dir", default=os.environ.get("ESR_FUNCTION_LIBRARY_DIR"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fn_dir = resolve_fn_dir(args.fn_library_dir)
    rho_tag = "rhoTrue" if args.use_physical_scale else "rhoFalse"
    out_dir = args.out_dir or os.path.join(args.data_dir, "fitting", "output",
                                           f"output_{args.run_name}_{rho_tag}", "MIXED")
    os.makedirs(out_dir, exist_ok=True)

    with open(args.galaxy_file) as f:
        galaxies = [line.strip() for line in f if line.strip()]
    comps = [int(c) for c in args.comps.split(",")]

    per_comp = {}
    for comp in comps:
        all_fcn, aifeyn = load_allfcn_and_aifeyn(fn_dir, comp)
        sums = accumulate_galaxies(comp, galaxies, args.data_dir, args.run_name,
                                   args.use_physical_scale, len(all_fcn))
        negloglike_sum, codelen_sum, shape_cost_sum, _, included, skipped = sums
        print(f"comp={comp}: {len(included)}/{len(galaxies)} galaxies included"
              + (f", skipped: {skipped}" if skipped else ""))
        per_comp[comp] = (all_fcn, aifeyn, negloglike_sum, codelen_sum, shape_cost_sum, len(included))

    n_incl = {c: v[-1] for c, v in per_comp.items()}
    if len(set(n_incl.values())) > 1:
        print(f"WARNING: different galaxy counts per comp {n_incl} -- summed DLs are not "
              f"comparable across comps until every comp has the same galaxies.")

    DL_pairs = []
    for all_fcn, aifeyn, nll, cl, _, _ in per_comp.values():
        DL = nll + cl + aifeyn
        ok = np.isfinite(DL)
        DL_pairs.extend(zip(DL[ok], nll[ok]))
    threshold = ranking_threshold(DL_pairs, args.top_k)
    print(f"Top-{args.top_k} all-local DL threshold across comps {comps}: {threshold:.2f}")

    candidates = []
    for comp, (all_fcn, aifeyn, nll, cl, shape_cost, _) in per_comp.items():
        candidates += screen_comp(comp, all_fcn, aifeyn, nll, cl, shape_cost, threshold,
                                  args.max_subset_size)
    candidates.sort(key=lambda c: c["DL_optimistic"])

    # Variant rows of one function (sign/relabelling duplicates, e.g. a0 + 1/x
    # vs -a0 + 1/x) give identical summed fits and gains -- keep one per
    # (comp, DL_local, max_gain) so the nested re-fit isn't run twice.
    n_before = len(candidates)
    seen, unique = set(), []
    for c in candidates:
        key = (c["comp"], round(c["DL_local"], 4), round(c["max_gain"], 4))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    candidates = unique
    print(f"De-duplicated {n_before} -> {len(candidates)} candidates.")

    out_path = os.path.join(out_dir, f"global_local_candidates_top{args.top_k}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["comp", "row", "function", "global_idx", "DL_local",
                         "negloglike_sum", "max_gain", "DL_optimistic"])
        for c in candidates:
            writer.writerow([c["comp"], c["row"], c["function"],
                             ",".join(str(j) for j in c["global_idx"]),
                             f"{c['DL_local']:.4f}", f"{c['negloglike_sum']:.4f}",
                             f"{c['max_gain']:.4f}", f"{c['DL_optimistic']:.4f}"])

    print(f"{len(candidates)} (function, global-subset) candidates; "
          f"{len({(c['comp'], c['row']) for c in candidates})} distinct variant rows.")
    for c in candidates[:20]:
        print(f"  comp={c['comp']} row={c['row']} {c['function']}  global={c['global_idx']}  "
              f"DL_local={c['DL_local']:.2f}  max_gain={c['max_gain']:.2f}  "
              f"DL_opt={c['DL_optimistic']:.2f}")
    print(f"Saved: {out_path}")
