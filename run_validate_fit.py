"""
Edit the settings below, then just run:  python run_validate_fit.py

This is the same as calling validate_fit.py from the command line, just with
the arguments set as plain variables here instead of typed as flags every
time. See validate_fit.py's own module docstring for what this actually does
(compares the production optimizer against random-restart baselines on one
function/galaxy).
"""
from validate_fit import validate_fit, DEFAULT_DATA_FILE

# ---- EDIT THESE ----

FCN_STRING = "pow(oo,a0)"          # ESR-style function string
GALAXY_NAME = "MGTH_J095717.1+020853"  # must match the "Galaxy" column in DATA_FILE

USE_PHYSICAL_SCALE = True        # fit rho0/rs as well as the shape params?

DATA_FILE = str(DEFAULT_DATA_FILE)  # path to the galaxy catalog
PMIN = -20.0                      # DIRECT search box, internal log-magnitude units
PMAX = 20.0
N_RANDOM = 100                    # number of random-restart starting points (NM and BFGS each)
SEED = 0                          # random seed, for reproducible random-restart starts
OUT_DIR = None                    # where plots go; None -> output/validate_fit/<galaxy>/
MATCH_TOL = 1e-3                  # chi2 tolerance for "pipeline matches best random restart"
DIRECT_MAXFUN = 3000              # DIRECT's evaluation budget (production default)
DIRECT_MAXITER = 3000
LANDSCAPE_GRID_N = 40             # resolution of the likelihood-landscape plot
COMPUTE_CODELEN = True            # also run the real production codelen calculation
                                   # (test_all_Fisher.convert_params) on each arm's winning
                                   # point? Set False to skip (faster, chi2-only check).
CHECK_MATCH = True               # ALSO run match.py's own Sigma/codelen mechanism
                                               
PLOT_INTEGRAL_PROFILES = True     
                                   
CODELEN_PIPELINE_ONLY = True      
                                   

# ---- end of settings ----

if __name__ == "__main__":
    validate_fit(
        fcn_string=FCN_STRING,
        galaxy_name=GALAXY_NAME,
        data_file=DATA_FILE,
        use_physical_scale=USE_PHYSICAL_SCALE,
        pmin=PMIN,
        pmax=PMAX,
        n_random=N_RANDOM,
        seed=SEED,
        out_dir=OUT_DIR,
        match_tol=MATCH_TOL,
        direct_maxfun=DIRECT_MAXFUN,
        direct_maxiter=DIRECT_MAXITER,
        landscape_grid_n=LANDSCAPE_GRID_N,
        compute_codelen=COMPUTE_CODELEN,
        check_match=CHECK_MATCH,
        plot_integral_profiles=PLOT_INTEGRAL_PROFILES,
        codelen_pipeline_only=CODELEN_PIPELINE_ONLY,
    )
