"""
One-stop cross-complexity summary of a MIGHTEE ESR run. Reads every
complexity's per-galaxy codelen_matches_comp{N}.dat and, from those files
alone (so the ranking, curves and parameters are all mutually consistent):

  1. Ranks every function from every complexity by population DL
     (sum over galaxies of negloglike + codelen, plus aifeyn once -- the
     same formula as combine_all_galaxies.py), keeping the best sign
     variant per function.
  2. Removes duplicates, keeping only the lowest-DL occurrence: the same
     function string found at several comps, and functions that give the
     same fit galaxy by galaxy -- summed -logL within --dup-tol-sum AND
     median per-galaxy |delta -logL| <= --dup-tol-gal. The second test
     catches equivalent models such as 1/(x**2+x), 1/(x*(x+1)) and
     1/(x**2+2*x) (identical once rho0/rs absorb the rescaling), whose
     separate fits differ by optimizer noise of up to ~0.5 per galaxy, while
     genuinely different functions differ by several units per galaxy even
     when their sums happen to be close. The comps a kept function also
     appeared at are listed in the table.
  3. Writes the de-duplicated ranking as a CSV (all rows) and a pretty
     table (top --table-n rows).
  4. top<N>_curves.pdf: one page per galaxy with the data, the top-N
     functions' smooth velocity curves and their density profiles, with the
     fitted rho0, rs, shape params, Inc and D printed under the panels.
  5. top<N>_params.pdf: one page per top function -- fitted Inc and D
     against their (initial) catalogue priors, and the rho0/rs/shape-param
     distributions across galaxies.

Everything above is repeated for the galaxies with at least N radial points,
for each N in --min-points (default 4,5 -- the same subsets and folder names
as combine_all_galaxies.py's GLOBAL_minpts{N}), to check how much the
preferred functions depend on poorly sampled galaxies. Each comp's files
are read once and ranked for every subset.

A comp is only ranked if every galaxy has a complete file for it (summed
DLs over different galaxy sets are not comparable); incomplete comps are
skipped with a warning unless --allow-partial, which instead restricts
every comp to the galaxies they all share.

Outputs go to <out-root>/GLOBAL/summary_comp<lo>-<hi>/ (all galaxies) and
<out-root>/GLOBAL_minpts<N>/summary_comp<lo>-<hi>/ (subsets).

Usage:
    python summarize_global_top.py --rho-tag rhoTrue
    python summarize_global_top.py --rho-tag rhoTrue --comps 1,2,3,4,5,6,7 --n-top 3
    python summarize_global_top.py --rho-tag rhoTrue --min-points ''   # all galaxies only
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from prettytable import PrettyTable

from esr.generation import simplifier

from combine_all_galaxies import DEFAULT_MIN_POINTS, galaxy_subsets, global_dir_name

from plot_param_priors import write_param_pages
from plot_top3_populations import default_profiles_csv, load_profile_overlays, write_galaxy_pages

# ESR function library: $ESR_FUNCTION_LIBRARY_DIR if set, else
# <repo>/function_library/core_maths.
FN_DIR = os.environ.get("ESR_FUNCTION_LIBRARY_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "function_library", "core_maths")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GALAXY_FILE = os.path.join(BASE_DIR, "galaxy_names.txt")
DATA_FILE = os.path.join(BASE_DIR, "ALL_gbar_gobs_RAR_direct_phot_with_5_arc_radius_plus_Halpha_for_ESR.txt")
MAX_FUN_PARAM = 4
MAX_COMP_SEARCH = 20


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comps", default=None, help="Comma-separated complexities (default: every comp found "
                    "under --out-root).")
    p.add_argument("--n-top", type=int, default=3, help="How many functions to plot.")
    p.add_argument("--table-n", type=int, default=50, help="Rows in the pretty table (the CSV has all).")
    p.add_argument("--dup-tol-sum", type=float, default=0.01, help="Duplicates must have summed -logL "
                    "within this (default 0.01: the same to 2 decimal places).")
    p.add_argument("--dup-tol-gal", type=float, default=0.05, help="...and median per-galaxy |delta -logL| "
                    "at most this.")
    p.add_argument("--keep-per-comp", type=int, default=300, help="Per comp, how many best functions keep "
                    "their full per-galaxy rows in memory for plotting.")
    p.add_argument("--allow-partial", action="store_true", help="Use comps with missing galaxies by "
                    "restricting ALL comps to the shared galaxy set.")
    p.add_argument("--rho-tag", default="rhoTrue", choices=["rhoTrue", "rhoFalse"])
    p.add_argument("--run-name", default="mightee")
    p.add_argument("--out-root", default=None, help="Base dir holding <galaxy>/comp<N>/codelen_matches_comp<N>.dat "
                    "(default: <this dir>/output_glamdring/output_<run-name>_<rho-tag>, else fitting/output/...)")
    p.add_argument("--fn-library-dir", default=os.environ.get("ESR_FUNCTION_LIBRARY_DIR", FN_DIR))
    p.add_argument("--data-file", default=DATA_FILE)
    p.add_argument("--min-points", default=",".join(map(str, DEFAULT_MIN_POINTS)),
                    help="Comma-separated thresholds: also summarize the galaxies with >= N radial points "
                    "(default '4,5'); pass '' for all galaxies only.")
    p.add_argument("--profiles", default="NFW", help="Standard profiles from fit_standard_profiles.py "
                    "to overlay (dashed) on the curve pages, e.g. 'NFW,Burkert'. '' to skip.")
    p.add_argument("--profiles-csv", default=None, help="Default: posterior_check_output/"
                    "standard_profiles_fit.csv (rhoTrue) or ..._rhoFalse.csv (rhoFalse).")
    p.add_argument("--save-dir", default=None, help="Default: <out-root>/GLOBAL/summary_comp<lo>-<hi> "
                    "(subsets: GLOBAL_minpts<N>/...). If given, subsets go to <save-dir>_minpts<N>.")
    return p.parse_args()


def _dat_path(out_root, galaxy, comp):
    return os.path.join(out_root, galaxy, f"comp{comp}", f"codelen_matches_comp{comp}.dat")


def _load_dat(path):
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64).to_numpy()


def choose_comps_and_galaxies(requested, galaxies, out_root, allow_partial):
    candidates = requested or list(range(1, MAX_COMP_SEARCH + 1))
    present = {c: [g for g in galaxies if os.path.exists(_dat_path(out_root, g, c))] for c in candidates}
    present = {c: gs for c, gs in present.items() if gs}
    if not present:
        raise RuntimeError(f"No codelen_matches_comp*.dat files found under {out_root}")

    if allow_partial:
        shared = set(galaxies)
        for gs in present.values():
            shared &= set(gs)
        used_galaxies = [g for g in galaxies if g in shared]
        comps = sorted(present)
        if len(used_galaxies) < len(galaxies):
            print(f"--allow-partial: restricting every comp to the {len(used_galaxies)}/{len(galaxies)} "
                  f"galaxies present in all of comps {comps}.")
        return comps, used_galaxies

    comps = []
    for c in sorted(present):
        if len(present[c]) == len(galaxies):
            comps.append(c)
        else:
            print(f"WARNING: skipping comp{c}: only {len(present[c])}/{len(galaxies)} galaxies have results "
                  f"(DL would not be comparable). Use --allow-partial to rank it on the shared subset.")
    if not comps:
        raise RuntimeError("No complexity has results for every galaxy.")
    return comps, list(galaxies)


def load_comp(comp, galaxies, out_root, fn_dir):
    """Read one comp's function list, aifeyn and every galaxy's results."""
    with open(os.path.join(fn_dir, f"compl_{comp}", f"all_equations_{comp}.txt")) as f:
        all_fcn = [l.strip() for l in f.readlines()]
    aifeyn = np.atleast_1d(np.genfromtxt(os.path.join(fn_dir, f"compl_{comp}", f"aifeyn_{comp}.txt")))
    n_variants = len(all_fcn)

    index_ref = None
    per_galaxy = {}
    for galaxy in galaxies:
        data = np.atleast_2d(_load_dat(_dat_path(out_root, galaxy, comp)))
        if data.shape[0] != n_variants:
            raise RuntimeError(f"comp{comp} {galaxy}: {data.shape[0]} rows, expected {n_variants} "
                               f"(incomplete match.py run?)")
        if index_ref is None:
            index_ref = data[:, 2]
        elif not np.array_equal(index_ref, data[:, 2]):
            raise RuntimeError(f"comp{comp} {galaxy}: variant index order differs from other galaxies")
        per_galaxy[galaxy] = data
    return all_fcn, aifeyn, index_ref, per_galaxy


def rank_comp(comp, all_fcn, aifeyn, index_ref, per_galaxy, galaxies, keep_per_comp, label):
    """Best variant per unique function for one comp, summed over `galaxies`.
    Returns (records, kept_rows_by_galaxy, kept_row_lookup)."""
    negloglike_sum = np.sum([per_galaxy[g][:, 0] for g in galaxies], axis=0)
    codelen_sum = np.sum([per_galaxy[g][:, 1] for g in galaxies], axis=0)
    DL = negloglike_sum + codelen_sum + aifeyn
    DL = np.where(np.isfinite(DL), DL, np.inf)

    # Best variant per unique function: sort by (index, DL), take first of each index.
    order = np.lexsort((DL, index_ref))
    first = np.ones(len(order), dtype=bool)
    first[1:] = index_ref[order][1:] != index_ref[order][:-1]
    best_rows = order[first]
    best_rows = best_rows[np.isfinite(DL[best_rows])]
    best_rows = best_rows[np.argsort(DL[best_rows])]

    nll_by_galaxy = np.stack([per_galaxy[g][:, 0] for g in galaxies], axis=1)  # (n_variants, n_gal)
    records = [{
        "comp": comp, "row": int(j), "function": all_fcn[j], "DL": float(DL[j]),
        "negloglike": float(negloglike_sum[j]), "codelen": float(codelen_sum[j]), "aifeyn": float(aifeyn[j]),
        "nll_vec": nll_by_galaxy[j],
    } for j in best_rows]

    kept = best_rows[:keep_per_comp]
    kept_rows_by_galaxy = {g: per_galaxy[g][kept] for g in galaxies}
    kept_row_lookup = {int(j): i for i, j in enumerate(kept)}
    if records:
        print(f"  comp{comp} [{label}]: {len(records)} functions with finite DL; best "
              f"{records[0]['function']} (DL {records[0]['DL']:.2f})")
    else:
        print(f"  comp{comp} [{label}]: no finite-DL functions")
    return records, kept_rows_by_galaxy, kept_row_lookup


def deduplicate(records, tol_sum, tol_gal):
    """Collapse duplicates onto their lowest-DL member. Records are visited
    in DL order, so the first of each group becomes its representative and
    later members are only ever compared with representatives (no chaining).
    A record is a duplicate of a representative if it has the same function
    string, or the same fit galaxy by galaxy (see module docstring)."""
    reps, rep_sums, rep_vecs, rep_by_name = [], [], [], {}
    for rec in sorted(records, key=lambda r: r["DL"]):
        name = rec["function"].replace(" ", "")
        match = rep_by_name.get(name)
        if match is None and reps:
            close = np.flatnonzero(np.abs(np.asarray(rep_sums) - rec["negloglike"]) <= tol_sum)
            for i in close:
                if np.median(np.abs(rep_vecs[i] - rec["nll_vec"])) <= tol_gal:
                    match = i
                    break
        if match is None:
            rep_by_name[name] = len(reps)
            reps.append(dict(rec, n_duplicates=0, comps_seen={rec["comp"]}))
            rep_sums.append(rec["negloglike"])
            rep_vecs.append(rec["nll_vec"])
        else:
            rep = reps[match]
            rep_by_name.setdefault(name, match)
            rep["n_duplicates"] += 1
            rep["comps_seen"].add(rec["comp"])

    for r in reps:
        r["also_at_comps"] = sorted(r.pop("comps_seen") - {r["comp"]})
        r.pop("nll_vec")
    return reps


def write_tables(deduped, save_dir, table_n):
    csv_path = os.path.join(save_dir, "ranking_all_comps.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "function", "comp", "DL", "delta_DL", "negloglike", "codelen", "aifeyn",
                    "n_duplicates", "also_at_comps"])
        DL0 = deduped[0]["DL"]
        for rank, r in enumerate(deduped, start=1):
            w.writerow([rank, r["function"], r["comp"], f"{r['DL']:.4f}", f"{r['DL'] - DL0:.4f}",
                        f"{r['negloglike']:.4f}", f"{r['codelen']:.4f}", f"{r['aifeyn']:.4f}",
                        r["n_duplicates"], " ".join(map(str, r["also_at_comps"]))])

    ptab = PrettyTable()
    ptab.field_names = ["Rank", "Function", "Comp", "N par", "DL", "ΔDL", "-logL", "Codelen", "AIFeyn",
                        "Dupl. at comps"]
    ptab.align["Function"] = "l"
    for c in ["DL", "ΔDL", "-logL", "Codelen", "AIFeyn"]:
        ptab.align[c] = "r"
    for rank, r in enumerate(deduped[:table_n], start=1):
        n_par = simplifier.count_params([r["function"]], MAX_FUN_PARAM)[0]
        ptab.add_row([rank, r["function"], r["comp"], n_par, f"{r['DL']:.2f}", f"{r['DL'] - DL0:.2f}",
                      f"{r['negloglike']:.2f}", f"{r['codelen']:.2f}", f"{r['aifeyn']:.2f}",
                      ",".join(map(str, r["also_at_comps"])) or "-"])
    txt_path = os.path.join(save_dir, "ranking_all_comps.txt")
    with open(txt_path, "w") as f:
        print(ptab, file=f)
    return ptab, csv_path, txt_path


def summarize_subset(state, galaxies, label, save_dir, args, use_physical_scale):
    """Dedup, table and both PDFs for one galaxy set."""
    deduped = deduplicate(state["records"], args.dup_tol_sum, args.dup_tol_gal)
    os.makedirs(save_dir, exist_ok=True)
    ptab, csv_path, txt_path = write_tables(deduped, save_dir, args.table_n)
    print(f"\n===== {label}: {len(galaxies)} galaxies =====")
    print(f"{len(state['records'])} functions across comps -> {len(deduped)} after removing duplicates "
          f"(same string, or |Δ summed -logL| <= {args.dup_tol_sum} with median per-galaxy "
          f"|Δ -logL| <= {args.dup_tol_gal}).\n")
    print(ptab)

    picks = []
    for r in deduped[:args.n_top]:
        local = state["lookup"][r["comp"]].get(r["row"])
        if local is None:
            raise RuntimeError(f"{r['function']} (comp{r['comp']}) is not among that comp's best "
                               f"{args.keep_per_comp}; rerun with a larger --keep-per-comp.")
        picks.append({"fcn": r["function"], "row": local, "per_galaxy_data": state["kept"][r["comp"]],
                      "DL": r["DL"], "comp": r["comp"]})

    names = [n.strip().replace("_", " ") for n in args.profiles.split(",") if n.strip()]
    fit_ml = args.run_name.endswith("_fml")
    overlays = load_profile_overlays(names, args.profiles_csv or default_profiles_csv(use_physical_scale, fit_ml),
                                     args.fn_library_dir, galaxies, use_physical_scale)
    for ov in overlays:
        print(f"  overlay {ov['name']}: DL = {ov['DL']:.2f} over these {len(galaxies)} galaxies")

    curves_pdf = os.path.join(save_dir, f"top{args.n_top}_curves.pdf")
    with PdfPages(curves_pdf) as pdf:
        write_galaxy_pages(pdf, picks, galaxies, use_physical_scale, data_file=args.data_file,
                           overlays=overlays, fit_ml=fit_ml)
    params_pdf = os.path.join(save_dir, f"top{args.n_top}_params.pdf")
    with PdfPages(params_pdf) as pdf:
        write_param_pages(pdf, picks, galaxies, use_physical_scale, data_file=args.data_file, fit_ml=fit_ml)

    print(f"\nSaved ({label}):\n  {csv_path}\n  {txt_path}\n  {curves_pdf}\n  {params_pdf}")


def main():
    args = _parse_args()
    use_physical_scale = args.rho_tag == "rhoTrue"
    out_root = args.out_root
    if out_root is None:
        out_root = os.path.join(BASE_DIR, "output_glamdring", f"output_{args.run_name}_{args.rho_tag}")
        if not os.path.isdir(out_root):
            out_root = os.path.join(BASE_DIR, "fitting", "output", f"output_{args.run_name}_{args.rho_tag}")
    print(f"Reading fits from: {out_root}")

    with open(GALAXY_FILE) as f:
        all_galaxies = [l.strip() for l in f if l.strip()]
    requested = [int(c) for c in args.comps.split(",")] if args.comps else None
    comps, galaxies = choose_comps_and_galaxies(requested, all_galaxies, out_root, args.allow_partial)
    min_points_list = [int(n) for n in args.min_points.split(",") if n.strip()]
    subsets = galaxy_subsets(galaxies, args.data_file, min_points_list)
    print(f"Ranking comps {comps}.\n")

    states = {mp: {"records": [], "kept": {}, "lookup": {}} for mp, _ in subsets}
    for comp in comps:
        all_fcn, aifeyn, index_ref, per_galaxy = load_comp(comp, galaxies, out_root, args.fn_library_dir)
        for mp, gal_subset in subsets:
            label = "all galaxies" if mp is None else f">= {mp} points"
            recs, states[mp]["kept"][comp], states[mp]["lookup"][comp] = rank_comp(
                comp, all_fcn, aifeyn, index_ref, per_galaxy, gal_subset, args.keep_per_comp, label)
            states[mp]["records"].extend(recs)
        del per_galaxy

    comp_tag = f"summary_comp{min(comps)}-{max(comps)}"
    for mp, gal_subset in subsets:
        label = "all galaxies" if mp is None else f">= {mp} points"
        if args.save_dir:
            save_dir = args.save_dir if mp is None else f"{args.save_dir.rstrip('/')}_minpts{mp}"
        else:
            save_dir = os.path.join(out_root, global_dir_name(mp), comp_tag)
        summarize_subset(states[mp], gal_subset, label, save_dir, args, use_physical_scale)


if __name__ == "__main__":
    main()
