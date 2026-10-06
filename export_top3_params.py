"""
For a given MIGHTEE complexity: find the top-N population-ranked functions
(same selection as plot_top3_populations.py, via _top_functions.py), then
for every galaxy write one CSV row per (galaxy, function-rank) with that
galaxy's own fitted parameter values, 1-sigma uncertainties (MDL Delta ->
Sigma = Delta/sqrt(12), same conversion used for the plot's shaded bands),
and the Inc/D fit values alongside their per-galaxy priors (inc_true/e_inc,
distance_true/e_d from MIGHTEELikelihood) -- everything plot_param_priors.py
needs to compute prior-separation, in one place a human can also just open.

Usage:
    python export_top3_params.py --comp 6 --n-top 3 --rho-tag rhoTrue
    python export_top3_params.py --comps 1,2,3,4,5,6 --n-top 3 --rho-tag rhoFalse
"""
import argparse
import csv
import os

import numpy as np

from esr.fitting.dm_likelihood import MIGHTEELikelihood
from esr.generation import simplifier

from _top_functions import build_picks

FN_DIR = os.environ.get("ESR_FUNCTION_LIBRARY_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "function_library", "core_maths"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GALAXY_FILE = os.path.join(BASE_DIR, "galaxy_names.txt")
DATA_FILE = os.path.join(BASE_DIR, "ALL_gbar_gobs_RAR_direct_phot.txt")
MAX_FUN_PARAM = 4
# n_extra/max_param_total depend on use_physical_scale (rho0/rs only exist
# in physical-scale mode -- match.py's own column layout, n_extra=2 if
# use_physical_scale else 0) so they're computed per-run in main(), not
# fixed module constants.


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    comp_group = p.add_mutually_exclusive_group(required=True)
    comp_group.add_argument("--comp", type=int, help="Single complexity.")
    comp_group.add_argument("--comps", help="Comma-separated complexities to rank AGAINST EACH OTHER "
                             "by population DL (e.g. 1,2,3,4,5,6) -- picks the true cross-complexity "
                             "winner(s), not just the best within one comp.")
    p.add_argument("--n-top", type=int, default=3)
    p.add_argument("--rho-tag", default="rhoTrue", choices=["rhoTrue", "rhoFalse"])
    p.add_argument("--run-name", default="mightee")
    p.add_argument("--out-root", default=None, help="Base dir holding <galaxy>/comp<N>/codelen_matches_comp<N>.dat "
                    "(default: <this dir>/output_glamdring/output_<run-name>_<rho-tag>)")
    p.add_argument("--csv-out", default=None, help="Default: <this dir>/export_top<n-top>_comp<comp>.csv, or "
                    "<out-root>/GLOBAL/export_top<n-top>_comp<lo>-<hi>.csv for --comps")
    return p.parse_args()


FIELDNAMES = [
    "galaxy", "rank", "comp", "function_string", "DL", "negloglike", "codelen",
    "param_0", "param_1", "param_2", "param_3", "rho0", "rs",
    "delta_param_0", "delta_param_1", "delta_param_2", "delta_param_3",
    "delta_rho0", "delta_rs",
    "Inc_fit", "D_fit",
    "Inc_prior_mean", "Inc_prior_sigma", "D_prior_mean", "D_prior_sigma",
]


def main():
    args = _parse_args()
    use_physical_scale = args.rho_tag == "rhoTrue"
    out_root = args.out_root or os.path.join(
        BASE_DIR, "output_glamdring", f"output_{args.run_name}_{args.rho_tag}")
    n_extra = 2 if use_physical_scale else 0
    max_param_total = MAX_FUN_PARAM + n_extra

    with open(GALAXY_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]

    picks, comp_label = build_picks(args.comp, args.comps, args.n_top, out_root, FN_DIR, galaxies,
                                     run_name=args.run_name)
    n_top = len(picks)

    if args.csv_out:
        csv_out = args.csv_out
    elif args.comps is not None:
        csv_out = os.path.join(out_root, "GLOBAL", f"export_top{args.n_top}_{comp_label}.csv")
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    else:
        csv_out = os.path.join(BASE_DIR, f"export_top{args.n_top}_{comp_label}.csv")

    top_fcn_strings = [p["fcn"] for p in picks]
    nshape_list = [simplifier.count_params([fs], MAX_FUN_PARAM)[0] for fs in top_fcn_strings]

    galaxies_with_any_pick = [g for g in galaxies if any(g in p["per_galaxy_data"] for p in picks)]

    n_rows_written = 0
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for galaxy in galaxies_with_any_pick:
            lk = MIGHTEELikelihood(data_file=DATA_FILE, name=galaxy, run_name="export_only",
                                    use_physical_scale=use_physical_scale)

            for k in range(n_top):
                galaxy_data_k = picks[k]["per_galaxy_data"].get(galaxy)
                if galaxy_data_k is None:
                    continue
                j = picks[k]["row"]
                nshape = nshape_list[k]
                row = galaxy_data_k[j]
                negloglike_k, codelen_k = row[0], row[1]
                params = row[3:3 + max_param_total]
                deltas = row[3 + max_param_total:3 + 2 * max_param_total]
                inc_fit, d_fit = row[3 + 2 * max_param_total], row[3 + 2 * max_param_total + 1]

                if not np.isfinite(negloglike_k) or negloglike_k == 0:
                    continue

                # Shape params past nshape are inactive for this function --
                # leave them NaN rather than reporting meaningless zeros.
                param_shape = [params[i] if i < nshape else np.nan for i in range(MAX_FUN_PARAM)]
                delta_shape = [deltas[i] / np.sqrt(12.0) if i < nshape else np.nan for i in range(MAX_FUN_PARAM)]
                # rho0/rs only exist in physical-scale mode (n_extra=2) --
                # NaN, not a bogus read past the params block, otherwise.
                if n_extra == 2:
                    rho0, rs = params[MAX_FUN_PARAM], params[MAX_FUN_PARAM + 1]
                    delta_rho0, delta_rs = (deltas[MAX_FUN_PARAM] / np.sqrt(12.0),
                                             deltas[MAX_FUN_PARAM + 1] / np.sqrt(12.0))
                else:
                    rho0 = rs = delta_rho0 = delta_rs = np.nan

                writer.writerow({
                    "galaxy": galaxy,
                    "rank": k + 1,
                    "comp": picks[k]["comp"],
                    "function_string": top_fcn_strings[k],
                    "DL": picks[k]["DL"],
                    "negloglike": negloglike_k,
                    "codelen": codelen_k,
                    "param_0": param_shape[0], "param_1": param_shape[1],
                    "param_2": param_shape[2], "param_3": param_shape[3],
                    "rho0": rho0, "rs": rs,
                    "delta_param_0": delta_shape[0], "delta_param_1": delta_shape[1],
                    "delta_param_2": delta_shape[2], "delta_param_3": delta_shape[3],
                    "delta_rho0": delta_rho0, "delta_rs": delta_rs,
                    "Inc_fit": inc_fit, "D_fit": d_fit,
                    "Inc_prior_mean": lk.inc_true, "Inc_prior_sigma": lk.e_inc,
                    "D_prior_mean": lk.distance_true, "D_prior_sigma": lk.e_d,
                })
                n_rows_written += 1

    print(f"\nWrote {n_rows_written} rows ({len(galaxies_with_any_pick)} galaxies x up to {n_top} ranks) "
          f"to: {csv_out}")


if __name__ == "__main__":
    main()
