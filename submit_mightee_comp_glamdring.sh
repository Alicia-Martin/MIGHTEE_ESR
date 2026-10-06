#!/bin/bash
# submit_mightee_comp_glamdring.sh
# Submit ONE Glamdring cluster job per MIGHTEE galaxy, all at a given complexity --
# each job runs the full 4-stage pipeline (test_all -> test_all_Fisher -> match ->
# combine_DL) for exactly one galaxy via run_esr_mightee.py's CLI. Glamdring
# equivalent of submit_mightee_comp.sh (DiRAC/SLURM) -- same CLI, same per-galaxy
# granularity, addqueue instead of sbatch (matching the existing
# dm-esr/CLASH_SPARC/submit_galaxies.sh convention for this cluster). A bad/stuck
# galaxy only kills its own job, not the whole batch.
#
# Usage (run from inside dm-esr/MIGHTEE_ESR/, same directory as this script,
# galaxy_names.txt, and run_esr_mightee.py):
#     COMP=5 ./submit_mightee_comp_glamdring.sh
#
# USE_PHYSICAL_SCALE controls whether rho0/rs are fit as physical amplitude/
# scale parameters (run_esr_mightee.py's --use-physical-scale/--no-physical-
# scale). Defaults to 1 (on, matching the production default). Set to 0 to
# turn it off:
#     COMP=5 USE_PHYSICAL_SCALE=0 ./submit_mightee_comp_glamdring.sh
# The setting is folded into run_name (rhoTrue/rhoFalse) by run_esr_mightee.py
# itself, so the two modes never collide. Output layout is one top-level
# folder per galaxy, with each comp nested inside it:
#     fitting/output/output_mightee_rhoTrue/<galaxy>/comp<N>/
# not a separate top-level folder per (galaxy, comp) pair.
#
# Re-running: run_esr_mightee.py's own default already skips a galaxy that
# has a results_pretty_{comp}.txt from a previous run, so re-running this
# script after some jobs failed only redoes the unfinished ones -- no flag
# needed for that case. To force EVERY galaxy to redo from scratch (ignore
# existing results), set RERUN=1:
#     COMP=5 RERUN=1 ... bash submit_mightee_comp_glamdring.sh
#
# NOTE: -n 1 -m 10 (1 core, 10 GB) is carried over untuned from
# dm-esr/CLASH_SPARC's own addqueue jobs -- no MIGHTEE-specific timing/memory
# data exists yet. Adjust once you've seen real usage for a comp.
#
# NOTE: -q redwood is the only Glamdring queue referenced anywhere in this repo
# (CLASH_SPARC's scripts) -- carried over as the default, not confirmed
# MIGHTEE-appropriate.
#
# NOTE: no mpirun wrapper -- addqueue execs its argument directly (it doesn't
# resolve a bare command through a login-shell PATH the way an interactive
# shell does), so `mpirun -np 1 python3 ...` fails with "The file .../mpirun
# does not exist" as soon as mpirun isn't already resolvable as a real path.
# Matches Alicia's own working addqueue pattern (redwood queue):
#     addqueue -q redwood -n 1 -m 10 /path/to/python3 script.py args
# mpi4py runs fine single-process without an mpirun launcher (COMM_WORLD size
# 1, rank 0) -- the launcher was never actually required for -np 1.
#
# NOTE: PYTHON_BIN defaults to whatever `python3` resolves to in *this*
# submitting shell, expanded to its absolute path via `command -v` for the
# same reason mpirun failed above -- addqueue needs a real path, not a bare
# name it has to search PATH for on the worker. If that's not the right
# interpreter (e.g. you need the esr_WL_env venv), pass it explicitly:
#     PYTHON_BIN=/path/to/esr_WL_env/bin/python3 COMP=5 bash submit_mightee_comp_glamdring.sh
#
# NOTE: no explicit -o/log-redirect flag -- neither existing CLASH_SPARC addqueue
# script (submit_galaxies.sh/submit_functions.sh) uses one, so this matches that
# convention and relies on addqueue's own default log handling rather than a
# flag I couldn't confirm exists.
#
# ESR_FUNCTION_LIBRARY_DIR is REQUIRED here (no fallback): dm_likelihood.py's
# own default for this path is Alicia's Mac OneDrive path, which doesn't
# exist on Glamdring. It's baked into each job's command line as
# --fn-library-dir rather than left as an inherited env var, since addqueue
# jobs aren't guaranteed to inherit the submitting shell's environment.
#     ESR_FUNCTION_LIBRARY_DIR=/path/to/function_library/core_maths \
#         COMP=5 bash submit_mightee_comp_glamdring.sh
set -u

comp="${COMP:-}"
if [ -z "$comp" ]; then
    echo "Error: COMP not set. Usage: COMP=<n> ./submit_mightee_comp_glamdring.sh"
    exit 1
fi

galaxy_names_file="${GALAXY_FILE:-galaxy_names.txt}"
python_bin="${PYTHON_BIN:-$(command -v python3)}"
if [ -z "$python_bin" ]; then
    echo "Error: could not resolve python3 to an absolute path, and PYTHON_BIN " \
         "not set. Pass PYTHON_BIN=/full/path/to/python3 explicitly."
    exit 1
fi
queue="${QUEUE:-redwood}"
n_cores="${N_CORES:-1}"
mem_gb="${MEM_GB:-10}"
use_physical_scale="${USE_PHYSICAL_SCALE:-1}"
if [ "$use_physical_scale" = "1" ] || [ "$use_physical_scale" = "true" ]; then
    phys_flag="--use-physical-scale"
    rho_tag="rhoTrue"
else
    phys_flag="--no-physical-scale"
    rho_tag="rhoFalse"
fi
group_name="mightee_comp${comp}_${rho_tag}"
rerun="${RERUN:-0}"
fn_library_dir="${ESR_FUNCTION_LIBRARY_DIR:-}"
if [ -z "$fn_library_dir" ]; then
    echo "Error: ESR_FUNCTION_LIBRARY_DIR not set. Required on Glamdring -- the" \
         "code's own default is a Mac-only path. Example:"
    echo "  ESR_FUNCTION_LIBRARY_DIR=/path/to/function_library/core_maths"
    exit 1
fi

[ -f "$galaxy_names_file" ] || { echo "Error: '$galaxy_names_file' not found."; exit 1; }

n_gal=$(grep -cve '^[[:space:]]*$' "$galaxy_names_file")
gal_idx=0

while IFS= read -r galaxy || [ -n "$galaxy" ]; do
    [ -z "${galaxy// }" ] && continue
    gal_idx=$((gal_idx + 1))
    echo "submitting galaxy $gal_idx/$n_gal: $galaxy (comp=$comp, use_physical_scale=$use_physical_scale, rerun=$rerun)"
    # Built as an array (never expanded empty under `set -u`) so --rerun is
    # appended only when requested, rather than risking an empty-string
    # argument reaching argparse when it's not.
    py_args=(--galaxy "$galaxy" --comp "$comp" --run-name "mightee"
             --fn-library-dir "$fn_library_dir" "$phys_flag")
    if [ "$rerun" = "1" ] || [ "$rerun" = "true" ]; then
        py_args+=(--rerun)
    fi
    addqueue -q "$queue" -n "$n_cores" -m "$mem_gb" --group "$group_name" \
        "$python_bin" run_esr_mightee.py "${py_args[@]}"
done < "$galaxy_names_file"

echo "Submitted $gal_idx galaxy jobs at comp=$comp, use_physical_scale=$use_physical_scale, rerun=$rerun."
