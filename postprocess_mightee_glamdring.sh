#!/bin/bash
# Combine + all plots for one MIGHTEE run, as ONE addqueue job (steps run in order).
#
#   addqueue -q redwood -n 1 -m 16 /bin/bash postprocess_mightee_glamdring.sh
#
# Settings (edit here or pass as env vars before addqueue):
RUN_NAME="${RUN_NAME:-mightee_fml}"     # mightee_fml = run with the M/L factor f; mightee = without f
COMPS="${COMPS:-1,2,3,4,5,6,7,8}"
RHO_TAG="${RHO_TAG:-rhoTrue}"            # rhoTrue or rhoFalse
N_TOP="${N_TOP:-3}"
SKIP_PROFILES="${SKIP_PROFILES:-0}"     # 1 = standard_profiles_fit csv already exists, don't refit
SKIP_COMBINE="${SKIP_COMBINE:-0}"       # 1 = combine_all_galaxies / combine_across_comp_global already done
: "${ESR_FUNCTION_LIBRARY_DIR:?set ESR_FUNCTION_LIBRARY_DIR=/path/to/function_library/core_maths}"
export ESR_FUNCTION_LIBRARY_DIR
PY="${PYTHON_BIN:-$(command -v python3)}"

cd "${MIGHTEE_DIR:-$(dirname "$(readlink -f "$0")")}" || exit 1   # repo directory
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

if [ "$RHO_TAG" = "rhoTrue" ]; then SCALE=--use-physical-scale; else SCALE=--no-physical-scale; fi
FML=""; case "$RUN_NAME" in *_fml) FML=--fit-ml ;; esac

failed=()
# The ESR modules import mpi4py. Under addqueue/srun one job slot allows only
# one process to call MPI_Init, so later python steps would fail in
# MPI_Init_thread. Every step is run without the scheduler's PMI/Slurm
# variables, so each python starts as a standalone (singleton) MPI process.
MPI_CLEAN=$(env | grep -oE '^(PMIX|PMI|OMPI|SLURM)[A-Za-z0-9_]*' | sed 's/^/-u /' | tr '\n' ' ')
step() {  # step <name> <command...>: run, log, keep going
    echo; echo "######## $1  ($(date))"; shift
    env $MPI_CLEAN "$@" || { echo "######## FAILED: $*"; failed+=("$*"); }
}

echo "run=$RUN_NAME comps=$COMPS rho=$RHO_TAG python=$PY"

# 1. NFW/Burkert/... with the same likelihood (needed by the overlays and the Pareto plot)
if [ "$SKIP_PROFILES" != "1" ]; then
    step "standard profiles" "$PY" fit_standard_profiles.py --fn-library-dir "$ESR_FUNCTION_LIBRARY_DIR" $FML \
        $([ "$RHO_TAG" = "rhoFalse" ] && echo --no-physical-scale)
fi
if [ "$SKIP_COMBINE" != "1" ]; then
    # 2. per-comp population ranking (also lists galaxies that are missing)
    step "combine per comp" "$PY" combine_all_galaxies.py --comps "$COMPS" --run-name "$RUN_NAME" $SCALE
    # 3. ranking across comps
    step "combine across comps" "$PY" combine_across_comp_global.py --comps "$COMPS" --run-name "$RUN_NAME" $SCALE
fi
# 4. Pareto plot
step "pareto" "$PY" pareto_plot_mightee.py --comps "$COMPS" --rho-tag "$RHO_TAG" --run-name "$RUN_NAME"
# 5. de-duplicated table + top-N curves pdf + params pdf (Inc, D, rho0, rs, shape params)
step "summary + plots" "$PY" summarize_global_top.py --rho-tag "$RHO_TAG" --run-name "$RUN_NAME" \
    --comps "$COMPS" --n-top "$N_TOP"

echo
if [ ${#failed[@]} -eq 0 ]; then echo "ALL STEPS OK ($(date))"; else
    echo "FAILED STEPS:"; printf '  %s\n' "${failed[@]}"; fi
