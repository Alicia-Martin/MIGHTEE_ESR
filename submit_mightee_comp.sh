#!/bin/bash -l
# submit_mightee_comp.sh
# Submit ONE DiRAC cluster job per MIGHTEE galaxy, all at a given complexity --
# each array task runs the full 4-stage pipeline (test_all -> test_all_Fisher ->
# match -> combine_DL) for exactly one galaxy via run_esr_mightee.py's CLI.
# A bad/stuck galaxy only kills its own array task, not the whole batch.
#
# Usage (run from inside dm-esr/MIGHTEE_ESR/, same directory as this script,
# galaxy_names.txt, and run_esr_mightee.py):
#     sbatch --export=ALL,COMP=5 submit_mightee_comp.sh
#
# USE_PHYSICAL_SCALE controls whether rho0/rs are fit as physical amplitude/
# scale parameters (run_esr_mightee.py's --use-physical-scale/--no-physical-
# scale). Defaults to 1 (on, matching the production default). Set to 0 to
# turn it off:
#     sbatch --export=ALL,COMP=5,USE_PHYSICAL_SCALE=0 submit_mightee_comp.sh
# The setting is folded into run_name (rhoTrue/rhoFalse) by run_esr_mightee.py
# itself, so the two modes never collide. Output layout is one top-level
# folder per galaxy, with each comp nested inside it:
#     fitting/output/output_mightee_rhoTrue/<galaxy>/comp<N>/
# not a separate top-level folder per (galaxy, comp) pair.
#
# COMP is required -- there is no default, so an accidental bare `sbatch
# submit_mightee_comp.sh` fails fast (in the array task, see the check below)
# rather than silently running the wrong complexity.
#
# Re-running: run_esr_mightee.py's own default already skips a galaxy that
# has a results_pretty_{comp}.txt from a previous run, so resubmitting this
# same command after some tasks failed only redoes the unfinished ones -- no
# flag needed for that case. To force EVERY galaxy to redo from scratch, set
# RERUN=1:
#     sbatch --export=ALL,COMP=5,RERUN=1 submit_mightee_comp.sh
#
# NOTE: -t 08:00:00 is an UNTUNED placeholder -- there's no timing data yet for
# how long one galaxy takes at a real MIGHTEE complexity (function count varies
# a lot by comp). Lower it once you've seen real wall-clock time for a comp, both
# to fail faster on a stuck job and to avoid over-requesting allocation.
#
# NOTE: --exclusive + --ntasks 1 reserves a whole cclake node for one MPI rank
# (carried over verbatim from the working template this was built from) --
# ~128x more core-hours charged than used, unless your DiRAC queue policy
# requires exclusive node allocation on this account. Left as given, not
# second-guessed here.
#SBATCH --ntasks 1
#SBATCH --job-name=mightee
#SBATCH -p cclake
#SBATCH -A YOUR-ACCOUNT        # set to your allocation
#SBATCH --exclusive
#SBATCH -t 08:00:00
#SBATCH --array=0-128
#SBATCH -o logs/mightee_%A_%a.out
#SBATCH -e logs/mightee_%A_%a.err

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
# 2026-08-26: plain `module purge` was NOT clearing a pre-loaded
# intel-oneapi-compilers/2023.2.4/..., which then conflicted with the
# 2022.1.0 build rhel8/default-icl needs (job 34375468's .err, task 47) --
# looks like a sticky module on this system. Unload it explicitly first,
# per that error's own HINT. Safe no-op if it isn't loaded.
module unload intel-oneapi-compilers/2023.2.4/gcc/4lbvg4hv 2>/dev/null || true
module load rhel8/default-icl              # REQUIRED - loads the basic environment

echo "DEBUG: cwd=$(pwd)  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
source "${VENV:-../esr_WL_env}/bin/activate"

mkdir -p logs

if [ -z "${COMP:-}" ]; then
    echo "Error: COMP not set. Submit with: sbatch --export=ALL,COMP=<n> submit_mightee_comp.sh"
    exit 1
fi

USE_PHYSICAL_SCALE="${USE_PHYSICAL_SCALE:-1}"
if [ "$USE_PHYSICAL_SCALE" = "1" ] || [ "$USE_PHYSICAL_SCALE" = "true" ]; then
    PHYS_FLAG="--use-physical-scale"
else
    PHYS_FLAG="--no-physical-scale"
fi

RERUN="${RERUN:-0}"

GALAXY=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" galaxy_names.txt)
if [ -z "$GALAXY" ]; then
    echo "Error: no galaxy at line $((SLURM_ARRAY_TASK_ID + 1)) of galaxy_names.txt"
    exit 1
fi

echo "Array task $SLURM_ARRAY_TASK_ID: galaxy=$GALAXY comp=$COMP use_physical_scale=$USE_PHYSICAL_SCALE rerun=$RERUN"
PY_ARGS=(--galaxy "$GALAXY" --comp "$COMP" --run-name "mightee" "$PHYS_FLAG")
if [ "$RERUN" = "1" ] || [ "$RERUN" = "true" ]; then
    PY_ARGS+=(--rerun)
fi
mpirun -np 1 python3 run_esr_mightee.py "${PY_ARGS[@]}"
