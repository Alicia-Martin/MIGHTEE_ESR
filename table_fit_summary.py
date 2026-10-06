"""
Per-galaxy summary table: fitted Inc/D (with their catalog priors in
parens), fitted rho0/rs, and the fit's -logL -- comp=6's "previous run"
winner (same source as plot_posterior_check.py / check_inc_prior_vs_fit.py).

Usage:
    python3 table_fit_summary.py
Output:
    posterior_check_output/fit_summary.csv    (full precision, for further analysis)
    prints a compact, human-readable table to stdout
"""
import os

import numpy as np
import pandas as pd

import plot_posterior_check as grid
from esr.fitting.dm_likelihood import _angular_diameter_distance_kpc

DATA_FILE = grid.DATA_FILE
GALAXY_NAMES_FILE = grid.GALAXY_NAMES_FILE
OUT_DIR = grid.OUT_DIR


def load_catalog(galaxy_names):
    df = pd.read_csv(DATA_FILE, sep=None, engine="python")
    cat = df[["Galaxy", "z", "inc", "inc_err"]].drop_duplicates().set_index("Galaxy")
    cat = cat.loc[cat.index.intersection(galaxy_names)].copy()
    cat["distance_prior"] = cat["z"].apply(_angular_diameter_distance_kpc)
    return cat


def build_table():
    with open(GALAXY_NAMES_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]
    cat = load_catalog(galaxies)

    rows = []
    for galaxy in galaxies:
        row = grid.load_galaxy_row(galaxy)
        if row is None or galaxy not in cat.index:
            continue
        rows.append({
            "galaxy": galaxy,
            "inc_fit": row["inc"], "inc_prior": float(cat.loc[galaxy, "inc"]),
            "d_fit": row["d"], "d_prior": float(cat.loc[galaxy, "distance_prior"]),
            "rho0": row["params"][4], "rs": row["params"][5],
            "negloglike": row["negloglike"],
        })
    return pd.DataFrame(rows)


def format_table(df):
    lines = []
    header = f"{'Galaxy':<24}{'Inc (prior)':<20}{'D (prior)':<22}{'rho0':<14}{'rs':<12}{'-logL':<10}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in df.iterrows():
        inc_str = f"{r['inc_fit']:.2f} ({r['inc_prior']:.2f})"
        d_str = f"{r['d_fit']:.1f} ({r['d_prior']:.1f})"
        rho0_str = f"{r['rho0']:.3e}"
        rs_str = f"{r['rs']:.3e}"
        nll_str = f"{r['negloglike']:.3f}"
        lines.append(f"{r['galaxy']:<24}{inc_str:<20}{d_str:<22}{rho0_str:<14}{rs_str:<12}{nll_str:<10}")
    return "\n".join(lines)


def main():
    df = build_table()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "fit_summary.csv")
    df.to_csv(out_csv, index=False)

    print(f"comp=6 winner \"{grid.FCN_STRING}\": {len(df)} galaxies\n")
    print(format_table(df))
    print(f"\nFull table (full precision): {out_csv}")


if __name__ == "__main__":
    main()
