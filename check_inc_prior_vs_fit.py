"""
Prior (catalog) Inc vs. jointly-fit Inc, for every galaxy -- comp=6's
"previous run" winner (same source as plot_posterior_check.py's own
all_galaxies_marginals_vs_gaussian.png).

Answers: which galaxies moved furthest from their catalog inclination
during the joint shape+Inc+D fit, and is it suspicious that most sit near
the prior (the 1:1 line)?

Usage:
    python3 check_inc_prior_vs_fit.py                  # full ranked table + plot
    python3 check_inc_prior_vs_fit.py GALAXY_NAME       # just print one galaxy
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_posterior_check as grid

DATA_FILE = grid.DATA_FILE
GALAXY_NAMES_FILE = grid.GALAXY_NAMES_FILE
OUT_DIR = grid.OUT_DIR


def load_catalog_inc():
    df = pd.read_csv(DATA_FILE, sep=None, engine="python")
    cat = df[["Galaxy", "inc", "inc_err"]].drop_duplicates().set_index("Galaxy")
    return cat


def build_table():
    with open(GALAXY_NAMES_FILE) as f:
        galaxies = [l.strip() for l in f if l.strip()]
    cat = load_catalog_inc()

    rows = []
    for galaxy in galaxies:
        row = grid.load_galaxy_row(galaxy)
        if row is None or galaxy not in cat.index:
            continue
        inc_prior = float(cat.loc[galaxy, "inc"])
        e_inc = float(cat.loc[galaxy, "inc_err"])
        inc_fit = float(row["inc"])
        diff = inc_fit - inc_prior
        rows.append({
            "galaxy": galaxy, "inc_prior": inc_prior, "e_inc": e_inc,
            "inc_fit": inc_fit, "diff": diff, "diff_sigma": diff / e_inc if e_inc > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def print_one(galaxy, df):
    match = df[df["galaxy"] == galaxy]
    if match.empty:
        print(f"No data for '{galaxy}' (not in galaxy list, no codelen_matches row, "
              f"or missing from the catalog).")
        return
    r = match.iloc[0]
    print(f"{galaxy}:")
    print(f"  prior (catalog) Inc = {r['inc_prior']:.3f} deg  (catalog sigma = {r['e_inc']:.3f} deg)")
    print(f"  final (fitted)  Inc = {r['inc_fit']:.3f} deg")
    print(f"  difference          = {r['diff']:+.3f} deg  ({r['diff_sigma']:+.2f} sigma)")


def main():
    df = build_table()

    if len(sys.argv) > 1:
        print_one(sys.argv[1], df)
        return

    df_sorted = df.reindex(df["diff"].abs().sort_values(ascending=False).index)
    out_csv = os.path.join(OUT_DIR, "inc_prior_vs_fit.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    df_sorted.to_csv(out_csv, index=False)

    print(f"{len(df)} galaxies with both a catalog Inc and a comp=6 fit.")
    print(f"Full ranked table (most-changed first): {out_csv}\n")
    print("Top 15 by |Inc_fit - Inc_prior|:")
    print(df_sorted.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(df["inc_prior"], df["inc_fit"], c=df["diff_sigma"].abs(),
                     cmap="viridis", s=25, vmin=0, vmax=5)
    lims = [0, 90]
    ax.plot(lims, lims, "r--", lw=1, label="1:1 (no change from prior)")
    # label the most-deviant galaxies directly on the plot
    for _, r in df_sorted.head(10).iterrows():
        ax.annotate(r["galaxy"], (r["inc_prior"], r["inc_fit"]), fontsize=6,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Prior (catalog) Inc [deg]")
    ax.set_ylabel("Fitted Inc [deg]")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("|deviation| in catalog-sigma units")
    fig.suptitle(f"comp=6 winner \"{grid.FCN_STRING}\": prior vs. fitted Inc ({len(df)} galaxies)")
    out_png = os.path.join(OUT_DIR, "inc_prior_vs_fit.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot: {out_png}")


if __name__ == "__main__":
    main()
