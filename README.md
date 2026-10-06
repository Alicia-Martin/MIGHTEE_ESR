# MIGHTEE_ESR

Data-driven dark-matter density profiles from **MIGHTEE-HI galaxy rotation curves**,
using **Exhaustive Symbolic Regression (ESR)**.

Rather than assuming a halo profile (NFW, Burkert, …), the code fits *every* candidate
function in an ESR function library up to a given complexity. Each candidate is a
profile shape `ρ(r) = ρ₀ f(r/r_s)`. The candidates are ranked by **minimum description
length** (MDL), which trades off goodness of fit against the information needed to
specify the function and its parameters. The best-ranked forms are the profiles the
data actually support.

> Status: active PhD research (University of Oxford). The accompanying paper is in
> preparation.

## How it works

For each galaxy and complexity (number of nodes in the function tree):

1. **Fit** (`esr/fitting/test_all.py`). Fit every candidate function to the rotation
   curve. This uses a global DIRECT search over parameter-sign combinations, followed
   by Nelder-Mead/BFGS polishing and a joint polish of the nuisance parameters
   (inclination, distance).
2. **Description length** (`esr/fitting/test_all_Fisher.py`). Compute the Fisher
   matrix and the MDL code length for each unique function, snapping parameters that
   are consistent with zero.
3. **Match** (`esr/fitting/match.py`). Extend the results to every algebraically
   equivalent rewrite of each function.
4. **Rank** (`esr/fitting/combine_DL.py`). Rank by total description length.

The physics is in `esr/fitting/dm_likelihood.py` (`MIGHTEELikelihood`). It predicts
`v_circ(r)` from the dark-matter profile plus the baryonic contribution, with priors on
inclination and distance. [`PIPELINE_EXPLAINED.md`](PIPELINE_EXPLAINED.md) walks through
each stage in detail, including the non-obvious numerical behaviour.

Further analysis:

- `combine_all_galaxies.py`, `combine_across_comp_global.py`: population-level
  rankings across galaxies and complexities.
- `fit_standard_profiles.py`, `fit_nfw_all_galaxies.py`: fits of standard profiles
  (NFW, etc.) for comparison.
- `prepare_posterior_hmc_meta.py`, `run_posterior_hmc_single.py`,
  `merge_posterior_hmc.py`, `plot_posterior_hmc.py`: HMC (NumPyro NUTS) posteriors for
  the top functions.
- `validate_fit.py`, `plot_posterior_check.py`, `check_inc_prior_vs_fit.py`:
  validation of the fits.
- `submit_*.sh`: SLURM (`sbatch`) and `addqueue` cluster submission scripts, one job
  per galaxy.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The pipeline also needs the **ESR function library** (`core_maths`), which is the list
of candidate functions at each complexity. It's distributed with ESR; see the
[ESR documentation](https://esr.readthedocs.io). Put it in `function_library/core_maths/`
in this repo, or point to it with:

```bash
export ESR_FUNCTION_LIBRARY_DIR=/path/to/function_library/core_maths
```

## Usage

Run from the repository root:

```bash
# one galaxy at complexity 5
python run_esr_mightee.py --galaxy <NAME> --comp 5 --data-file <catalogue.txt>

# every galaxy on a cluster (one job each)
COMP=5 PYTHON_BIN=$(which python3) bash submit_mightee_comp_glamdring.sh
```

Results are written to `fitting/output/output_<run>/<galaxy>/comp<N>/`. The final
ranking is in `results_pretty_<N>.txt`.

## Data

The MIGHTEE-HI rotation-curve catalogue is **not included**, as it is not yet public.
The code expects a text table (CSV or whitespace-separated; the delimiter is auto-detected)
with one row per radial point and these columns:

| Column | Meaning |
|---|---|
| `Galaxy` | galaxy identifier |
| `z` | redshift |
| `inc`, `inc_err` | inclination and its uncertainty (deg) |
| `Radius_arcsec` | radius (arcsec) |
| `v_circ` | observed circular velocity (km/s) |
| `v_rot_err_lo`, `v_rot_err_hi` | asymmetric velocity uncertainties |
| `v_bar_total`, `v_bar_total_err` | baryonic velocity contribution and uncertainty |

Any rotation-curve sample in this format can be used.

## Repository layout

```
esr/                  modified copy of ESR (see NOTICE) with the MIGHTEE likelihood
  fitting/            fitting, Fisher/MDL, matching, ranking, verification scripts
  generation/         ESR function generation (upstream)
notebooks/            data exploration and checks (outputs cleared)
legacy/               earlier standalone drivers, kept for reference
*.py, submit_*.sh     drivers, analysis, plotting and cluster scripts
```

## Acknowledgements

- **ESR**: this project builds on ESR by Deaglan Bartlett and Harry Desmond
  ([github.com/DeaglanBartlett/ESR](https://github.com/DeaglanBartlett/ESR)); see
  [`NOTICE`](NOTICE). Please also cite Bartlett, Desmond & Ferreira (2023),
  *Exhaustive Symbolic Regression*, IEEE TEVC
  ([arXiv:2211.11461](https://arxiv.org/abs/2211.11461)).
- **MIGHTEE**: Andreea Varasteanu, for the MIGHTEE-HI rotation-curve data and
  collaboration.

## Citation

Please cite this code using [`CITATION.cff`](CITATION.cff); GitHub's "Cite this
repository" button does this for you. A paper reference will be added on publication.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
