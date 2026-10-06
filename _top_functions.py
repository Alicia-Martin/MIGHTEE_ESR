"""
Shared top-N population-ranked function selection, factored out of
plot_top3_populations.py so export_top3_params.py and plot_param_priors.py
select exactly the same functions (same ranking logic, same "winning
variant" per function) without duplicating the ~80-line block.
"""
import csv
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class TopFunctionsResult:
    all_fcn: list
    per_galaxy_data: dict
    index_ref: np.ndarray
    n_variants: int
    DL_min: np.ndarray
    top_indices: list
    top_rows: list
    top_fcn_strings: list


@dataclass
class GlobalPick:
    comp: int
    function_string: str
    DL: float
    row: int


@dataclass
class GlobalTopFunctionsResult:
    picks: list  # list[GlobalPick], sorted by DL ascending, len <= n_top
    per_galaxy_data_by_comp: dict  # {comp: per_galaxy_data dict}


def select_top_functions(comp, n_top, out_root, fn_dir, galaxies, run_name="mightee"):
    with open(os.path.join(fn_dir, f"compl_{comp}", f"all_equations_{comp}.txt")) as f:
        all_fcn = [l.strip() for l in f.readlines()]
    aifeyn = np.atleast_1d(np.genfromtxt(os.path.join(fn_dir, f"compl_{comp}", f"aifeyn_{comp}.txt")))
    n_variants = len(all_fcn)

    negloglike_sum = np.zeros(n_variants)
    codelen_sum = np.zeros(n_variants)
    index_ref = None
    per_galaxy_data = {}

    for galaxy in galaxies:
        fpath = os.path.join(out_root, galaxy, f"comp{comp}", f"codelen_matches_comp{comp}.dat")
        if not os.path.exists(fpath):
            continue
        data = np.atleast_2d(np.genfromtxt(fpath))
        if data.shape[0] != n_variants:
            print(f"SKIP {galaxy}: {data.shape[0]} rows, expected {n_variants}")
            continue
        negloglike, codelen, index = data[:, 0], data[:, 1], data[:, 2]
        if index_ref is None:
            index_ref = index
        elif not np.array_equal(index_ref, index):
            print(f"SKIP {galaxy}: variant index order mismatch")
            continue
        negloglike_sum += negloglike
        codelen_sum += codelen
        per_galaxy_data[galaxy] = data

    print(f"Included {len(per_galaxy_data)}/{len(galaxies)} galaxies in the population sum.")
    if index_ref is None:
        raise RuntimeError(f"No codelen_matches_comp{comp}.dat files found under {out_root}")

    DL = negloglike_sum + codelen_sum + aifeyn
    n_unique = int(index_ref.max()) + 1
    DL_min = np.full(n_unique, np.nan)
    winning_row = np.full(n_unique, -1, dtype=int)
    for i in range(n_unique):
        mask = index_ref == i
        if not np.any(mask):
            continue
        DL_i = DL[mask]
        if np.sum(~np.isnan(DL_i)) == 0:
            continue
        rows_i = np.where(mask)[0]
        best_local = np.nanargmin(DL_i)
        DL_min[i] = DL_i[best_local]
        winning_row[i] = rows_i[best_local]

    order = np.argsort(DL_min)
    top_indices = [i for i in order if np.isfinite(DL_min[i])][:n_top]
    top_rows = [winning_row[i] for i in top_indices]
    top_fcn_strings = [all_fcn[j] for j in top_rows]

    # Cap what gets PRINTED separately from what gets RETURNED -- callers
    # (e.g. select_global_top_functions) sometimes pass a huge n_top just to
    # get the full per-comp ranking back, and printing thousands of lines in
    # that case would bury the actually-useful summary in noise.
    n_print = min(n_top, 10)
    print(f"\nTop {n_print} population-ranked functions (comp={comp}):"
          if n_print == n_top else
          f"\nTop {n_print} of {len(top_indices)} population-ranked functions (comp={comp}):")
    for rank, (i, j, fs) in enumerate(zip(top_indices[:n_print], top_rows[:n_print], top_fcn_strings[:n_print])):
        print(f"  {rank+1}. row={j} unique_idx={i} DL={DL_min[i]:.2f}  {fs}")

    return TopFunctionsResult(
        all_fcn=all_fcn,
        per_galaxy_data=per_galaxy_data,
        index_ref=index_ref,
        n_variants=n_variants,
        DL_min=DL_min,
        top_indices=top_indices,
        top_rows=top_rows,
        top_fcn_strings=top_fcn_strings,
    )


def select_global_top_functions(comps, n_top, out_root, fn_dir, galaxies, run_name="mightee"):
    """
    Cross-complexity version of select_top_functions: ranks functions from
    ALL of `comps` against each other by population DL (which already
    includes each comp's own aifeyn complexity penalty), so a genuinely
    simpler function that wins once complexity is fairly accounted for is
    not hidden by only ever comparing within a single comp.

    Reads each comp's global_final_{comp}.dat (already produced by the
    pipeline's combine_all_galaxies.py, at out_root/GLOBAL/comp{N}/) rather
    than re-summing the raw per-galaxy .dat files.
    """
    all_rows = []
    for comp in comps:
        dat_path = os.path.join(out_root, "GLOBAL", f"comp{comp}", f"global_final_{comp}.dat")
        if not os.path.exists(dat_path):
            print(f"SKIP comp{comp}: no global_final_{comp}.dat found at {dat_path}")
            continue
        with open(dat_path, newline="") as f:
            for row in csv.reader(f, delimiter=";"):
                all_rows.append({
                    "comp": comp,
                    "function": row[1],
                    "DL": float(row[2]),
                    "negloglike": float(row[3]),
                })

    if not all_rows:
        raise RuntimeError(f"No global_final_{{comp}}.dat found for any of comps={comps} under {out_root}/GLOBAL")

    # De-duplicate by negloglike, keeping the lowest-DL occurrence -- mirrors
    # combine_across_comp_global.py's convention: a function re-appearing at
    # a higher complexity with an identical negloglike is the same fit
    # padded with a no-op term, so only its lowest-complexity (lowest-DL)
    # occurrence should compete in the ranking.
    best_by_negloglike = {}
    for row in all_rows:
        key = round(row["negloglike"], 6)
        if key not in best_by_negloglike or row["DL"] < best_by_negloglike[key]["DL"]:
            best_by_negloglike[key] = row
    deduped = sorted(best_by_negloglike.values(), key=lambda r: r["DL"])[:n_top]

    print(f"\nTop {n_top} GLOBAL population-ranked functions (comps={comps}):")
    for rank, row in enumerate(deduped):
        print(f"  {rank+1}. comp={row['comp']} DL={row['DL']:.2f}  {row['function']}")

    # Resolve each pick's row (variant index within its own comp) by looking
    # it up in that comp's own full per-comp ranking -- only compute this
    # once per distinct comp actually needed, not for every comp requested.
    needed_comps = sorted({row["comp"] for row in deduped})
    per_comp_lookup = {}       # comp -> {function_string: row}
    per_galaxy_data_by_comp = {}
    for comp in needed_comps:
        comp_result = select_top_functions(comp, 10**9, out_root, fn_dir, galaxies, run_name=run_name)
        lookup = dict(zip(comp_result.top_fcn_strings, comp_result.top_rows))
        lookup_stripped = {k.strip(): v for k, v in lookup.items()}
        per_comp_lookup[comp] = (lookup, lookup_stripped)
        per_galaxy_data_by_comp[comp] = comp_result.per_galaxy_data

    picks = []
    for row in deduped:
        comp = row["comp"]
        lookup, lookup_stripped = per_comp_lookup[comp]
        fs = row["function"]
        if fs in lookup:
            pick_row = lookup[fs]
        elif fs.strip() in lookup_stripped:
            pick_row = lookup_stripped[fs.strip()]
        else:
            print(f"WARNING: could not match function {fs!r} (comp={comp}) to a row "
                  f"in its per-galaxy data -- skipping this pick.")
            continue
        picks.append(GlobalPick(comp=comp, function_string=fs, DL=row["DL"], row=pick_row))

    return GlobalTopFunctionsResult(picks=picks, per_galaxy_data_by_comp=per_galaxy_data_by_comp)


def build_picks(comp, comps, n_top, out_root, fn_dir, galaxies, run_name="mightee"):
    """Normalizes the single-comp (--comp) and cross-comp (--comps) selection
    paths into one list of {fcn, row, per_galaxy_data, DL, comp} dicts, plus
    a short label for output filenames -- shared by all three MIGHTEE
    top-function scripts (plot_top3_populations.py, export_top3_params.py,
    plot_param_priors.py) so they don't each reimplement this branch.

    Exactly one of `comp` (int) / `comps` (comma-separated str) must be given.
    """
    if comp is not None:
        top = select_top_functions(comp, n_top, out_root, fn_dir, galaxies, run_name=run_name)
        picks = [
            {"fcn": fs, "row": row, "per_galaxy_data": top.per_galaxy_data,
             "DL": top.DL_min[idx], "comp": comp}
            for idx, row, fs in zip(top.top_indices, top.top_rows, top.top_fcn_strings)
        ]
        label = f"comp{comp}"
    else:
        comps_list = [int(c) for c in comps.split(",")]
        gtop = select_global_top_functions(comps_list, n_top, out_root, fn_dir, galaxies, run_name=run_name)
        picks = [
            {"fcn": p.function_string, "row": p.row,
             "per_galaxy_data": gtop.per_galaxy_data_by_comp[p.comp], "DL": p.DL, "comp": p.comp}
            for p in gtop.picks
        ]
        label = f"comp{min(comps_list)}-{max(comps_list)}"
    return picks, label
