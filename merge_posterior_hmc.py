"""
Merges per-galaxy raw HMC results (from run_posterior_hmc_single.py, one
.json + .npz per galaxy under --raw-dir) into the same final PDF +
hmc_summary_<tag>.txt that plot_posterior_hmc.py produces when run serially
-- the fast, must-be-ordered plotting step that follows the parallelized,
expensive sampling step on the cluster. Reuses render_galaxy_corner_page
from plot_posterior_hmc.py so the annotation/title/trust-label logic lives
in exactly one place, shared with the serial script.

Usage:
    python merge_posterior_hmc.py \
        --meta posterior_check_output/hmc_raw_comp6_1_x_pow_x_x/meta.json \
        --raw-dir posterior_check_output/hmc_raw_comp6_1_x_pow_x_x \
        --out-dir posterior_check_output \
        --fn-library-dir /path/to/function_library/core_maths
"""
import argparse
import json
import os

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fn-library-dir", required=True)
    p.add_argument("--results-dir", default=None, help="Base dir holding "
                    "<galaxy>/comp<N>/codelen_matches_comp<N>.dat -- only read for "
                    "Inc/D when a galaxy's .json predates run_posterior_hmc_single.py "
                    "saving them itself. Defaults to plot_posterior_check.py's "
                    "RESULTS_DIR.")
    return p.parse_args()


def main():
    args = _parse_args()
    os.environ["ESR_FUNCTION_LIBRARY_DIR"] = args.fn_library_dir
    import plot_posterior_hmc as hmc  # deferred: must import AFTER the env var is set
    import esr.fitting.match as match_mod
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    with open(args.meta) as f:
        meta = json.load(f)
    comp = meta["comp"]
    fcn_string = meta["fcn_string"]
    active_names = meta["active_names"]
    nshape = meta["nshape"]
    fit_ml = bool(meta.get("fit_ml", False))
    hmc.grid.COMP = comp
    if args.results_dir is not None:
        hmc.grid.RESULTS_DIR = args.results_dir

    with open(hmc.GALAXY_NAMES_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]

    # _fml suffix: never overwrite the PDF/summary of the run without f
    tag = f"comp{comp}_{hmc.safe_fcn_name(fcn_string)}" + ("_fml" if fit_ml else "")
    os.makedirs(args.out_dir, exist_ok=True)
    pdf_path = os.path.join(args.out_dir, f"hmc_corners_all_galaxies_{tag}.pdf")
    summary_path = os.path.join(args.out_dir, f"hmc_summary_{tag}.txt")

    n_ok, n_skip = 0, 0
    summary_lines = []
    shift_records = []
    ml_flags = []
    with PdfPages(pdf_path) as pdf:
        for galaxy in galaxies:
            safe_name = galaxy.replace("/", "_").replace("+", "p")
            result_path = os.path.join(args.raw_dir, f"{safe_name}.json")
            samples_path = os.path.join(args.raw_dir, f"{safe_name}.npz")
            if not os.path.exists(result_path):
                print(f"[{galaxy}] no raw result found -- missing/not yet run.")
                n_skip += 1
                continue
            with open(result_path) as f:
                result = json.load(f)
            if result.get("status") != "ok":
                print(f"[{galaxy}] {result.get('status')}: {result.get('error', '')}")
                n_skip += 1
                continue

            theta_ml = np.asarray(result["theta_ml"])
            # Older raw results (Inc/D fixed) carry neither key.
            names_g = result.get("active_names", active_names)
            prior = result.get("prior")
            site_names = result["site_names"]
            used_fallback = result["used_fallback"]
            stats = result["stats"]
            n_divergent = result["n_divergent"]
            n_data = result["n_data"]
            arr = np.load(samples_path)["arr"]

            if "inc" in result:
                inc_fit, d_fit = result["inc"], result["d"]
            else:
                row = hmc.grid.load_galaxy_row(galaxy, fcn_index=meta["fcn_index"])
                if row is None:
                    print(f"[{galaxy}] no Inc/D in its .json and no codelen_matches row under "
                          f"{hmc.grid.RESULTS_DIR} -- pass --results-dir; skipping.")
                    n_skip += 1
                    continue
                inc_fit, d_fit = row["inc"], row["d"]
            f_fit = result.get("f") if fit_ml else None
            likelihood = hmc.MIGHTEELikelihood(
                data_file=hmc.DATA_FILE, name=galaxy,
                run_name=f"posterior_hmc_merge/{galaxy}/comp{comp}",
                data_dir=hmc.SCRATCH_DATA_DIR, use_physical_scale=True, fit_ml=fit_ml,
            )
            _, eq, integrated = likelihood.run_sympify(fcn_string, tmax=60, try_integration=False)
            eq_numpy = match_mod.make_lambdified_eq(eq, nshape)
            rc = hmc.compute_rc_bands(likelihood, eq_numpy, integrated, inc_fit, d_fit,
                                      theta_ml, used_fallback, names_g, arr, f_fit=f_fit)

            fig = hmc.render_galaxy_corner_page(
                fcn_string, comp, galaxy, n_data, names_g, theta_ml,
                used_fallback, arr, stats, site_names, n_divergent, rc=rc, prior=prior,
                boxes=result.get("boxes"), mlc=result.get("ml_check"),
            )
            if result.get("ml_check") and result["ml_check"]["flag"]:
                ml_flags.append(f"{galaxy}: {result['ml_check']['flag']} "
                                f"({result['ml_check']['delta']:+.2f})")
            if prior is not None:
                shift_records.append({"galaxy": galaxy, "n_data": n_data,
                                      **hmc.prior_shifts(stats, site_names, prior)})
            pdf.savefig(fig)
            plt.close(fig)

            n_ok += 1
            stat_bits = "; ".join(
                f"{name}: mean={stats[site]['mean']:.3g} std={stats[site]['std']:.3g} "
                f"n_eff={stats[site]['n_eff']:.0f} r_hat={stats[site]['r_hat']:.3f}"
                for name, site in zip(names_g, site_names)
            )
            summary_lines.append(
                f"{galaxy}\tN={n_data}\tused_fallback={used_fallback}\t"
                f"divergences={n_divergent}\t{stat_bits}"
            )

        if shift_records:
            fig = hmc.render_prior_shift_page(shift_records, fcn_string, comp)
            pdf.savefig(fig)
            plt.close(fig)

    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
        if shift_records:
            f.write("\n\nInc/D" + ("/f" if fit_ml else "")
                    + " posterior vs prior (sorted by largest shift):\n")
            f.write("\n".join(hmc.shift_table_lines(shift_records)) + "\n")
    if shift_records:
        flagged = [r["galaxy"] for r in shift_records if hmc.max_shift(r) > hmc.PRIOR_SHIFT_FLAG]
        print(f"{len(flagged)} galaxies with Inc, D" + (" or f" if fit_ml else "")
              + f" pulled > {hmc.PRIOR_SHIFT_FLAG:g} sigma from the prior: {', '.join(flagged)}")

    if ml_flags:
        print(f"{len(ml_flags)} galaxies where the posterior's lowest -logL differs from the "
              f"best fit by > {hmc.ML_CHECK_TOL:g}:\n  " + "\n  ".join(ml_flags))
        with open(summary_path, "a") as f:
            f.write("\nPosterior min -logL vs best fit:\n" + "\n".join(ml_flags) + "\n")
    print(f"\nDone: {n_ok} galaxies plotted, {n_skip} skipped.")
    print(f"PDF: {pdf_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
