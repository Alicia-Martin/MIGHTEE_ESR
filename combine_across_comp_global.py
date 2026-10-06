import argparse
import csv
import os

import numpy as np
from prettytable import PrettyTable


def load_global_final(data_dir, base_run_name, rho_tag, comp):
    out_dir = os.path.join(data_dir, "fitting", "output", f"output_{base_run_name}_{rho_tag}",
                            "GLOBAL", f"comp{comp}")
    dat_path = os.path.join(out_dir, f"global_final_{comp}.dat")
    if not os.path.exists(dat_path):
        return None

    rows = []
    with open(dat_path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            # rank, function, DL, negloglike, codelen, aifeyn
            rows.append({
                "comp": comp,
                "function": row[1],
                "DL": float(row[2]),
                "negloglike": float(row[3]),
                "codelen": float(row[4]),
                "aifeyn": float(row[5]),
            })
    return rows


def _parse_args():
    p = argparse.ArgumentParser(
        description="Rank the population-level best function from EACH complexity "
                     "(combine_all_galaxies.py's own global_final_{comp}.dat, one row "
                     "set per comp) against each other, across all complexities, by "
                     "DL -- the Occam-penalized answer to 'which complexity actually "
                     "wins overall', not just within one comp."
    )
    p.add_argument("--comps", required=True, help="Comma-separated list of complexities "
                    "to combine, e.g. 3,4,5,6,7,8,9,10 -- must already have a "
                    "global_final_{comp}.dat from combine_all_galaxies.py.")
    p.add_argument("--data-dir", default=os.getcwd())
    p.add_argument("--run-name", default="mightee")
    p.add_argument("--use-physical-scale", dest="use_physical_scale", action="store_true", default=True)
    p.add_argument("--no-physical-scale", dest="use_physical_scale", action="store_false")
    p.add_argument("--top-n", type=int, default=200, help="How many functions to keep "
                    "in the final combined ranking (still writes every finite-DL row "
                    "to the .dat file; only the printed/pretty table is truncated).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rho_tag = "rhoTrue" if args.use_physical_scale else "rhoFalse"
    comps = [int(c) for c in args.comps.split(",")]

    all_rows = []
    missing_comps = []
    for comp in comps:
        rows = load_global_final(args.data_dir, args.run_name, rho_tag, comp)
        if rows is None:
            missing_comps.append(comp)
            continue
        all_rows.extend(rows)

    if missing_comps:
        print(f"Warning: no global_final_{{comp}}.dat found for comp(s) {missing_comps} "
              f"-- run combine_all_galaxies.py for those first. Continuing with the rest.")

    if not all_rows:
        print("No global results found for any requested comp -- nothing to combine.")
        raise SystemExit(1)

    # Deduplicate by negloglike (same convention as CLASH's combine_comp.py):
    # a function re-appearing at a higher complexity with an identical
    # negloglike is the same underlying fit padded with a no-op term, so
    # keep only the lowest-DL (== lowest-complexity, since DL includes the
    # aifeyn complexity prior) occurrence.
    best_by_negloglike = {}
    for row in all_rows:
        key = round(row["negloglike"], 6)
        if key not in best_by_negloglike or row["DL"] < best_by_negloglike[key]["DL"]:
            best_by_negloglike[key] = row

    deduped_rows = sorted(best_by_negloglike.values(), key=lambda r: r["DL"])

    out_dir = os.path.join(args.data_dir, "fitting", "output", f"output_{args.run_name}_{rho_tag}", "GLOBAL")
    os.makedirs(out_dir, exist_ok=True)

    ptab = PrettyTable()
    ptab.field_names = ["Rank", "Function", "Comp", "DL", "-logL", "Codelen", "AIFeyn"]

    dat_path = os.path.join(out_dir, "global_final_all_comps.dat")
    with open(dat_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        for rank, row in enumerate(deduped_rows):
            writer.writerow([rank, row["function"], row["comp"], row["DL"],
                              row["negloglike"], row["codelen"], row["aifeyn"]])
            if rank < args.top_n:
                ptab.add_row([rank, row["function"], row["comp"], f"{row['DL']:.2f}",
                              f"{row['negloglike']:.2f}", f"{row['codelen']:.2f}", f"{row['aifeyn']:.2e}"])

    print(ptab)
    with open(os.path.join(out_dir, "global_results_pretty_all_comps.txt"), "w") as f:
        print(ptab, file=f)
    print(f"\nSaved: {dat_path}")
    print(f"Combined {len(comps) - len(missing_comps)}/{len(comps)} complexities, "
          f"{len(all_rows)} candidate rows -> {len(deduped_rows)} after de-duplication.")
