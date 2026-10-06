#!/bin/bash
# submit_posterior_hmc_glamdring.sh
# Submit ONE Glamdring cluster job per MIGHTEE galaxy to run the HMC
# posterior-Gaussianity check (plot_posterior_hmc.py's own model, settings,
# and trust diagnostic) for a single function, in parallel -- the local/
# laptop version runs 129 galaxies serially (NUM_WARMUP=1500, dense_mass,
# target_accept=0.98 -- see plot_posterior_hmc.py's own comments for why --
# takes hours per function); this fans the same per-galaxy NUTS sampling out
# across cluster cores, one galaxy per job, so wall-clock time drops to
# roughly one galaxy's own sampling time instead of 129x that.
#
# Three-stage workflow (stages 1 and 3 are cheap and run once; stage 2 is
# the expensive part, parallelized):
#   1. prepare_posterior_hmc_meta.py   -- ONE run, computes nshape/fallback
#      anchors shared by every galaxy's job (this script runs it for you).
#   2. run_posterior_hmc_single.py     -- ONE array job PER galaxy, the
#      actual NUTS sampling (this script submits the whole array).
#   3. merge_posterior_hmc.py          -- ONE run, after every array job has
#      finished, assembles the final PDF + hmc_summary_<tag>.txt (this
#      script prints the exact command to run once you've confirmed the
#      array has finished -- addqueue has no dependency flag this repo's
#      other scripts rely on, so it isn't submitted automatically).
#
# Usage (run from inside dm-esr/MIGHTEE_ESR/, same directory as this script,
# galaxy_names.txt, and plot_posterior_hmc.py). RESULTS_DIR must point at
# wherever run_esr_mightee.py actually wrote this run's per-galaxy output on
# THIS machine (plot_posterior_check.py's own RESULTS_DIR default is a
# Mac-local snapshot-folder path with no reason to exist anywhere else):
#     COMP=6 FCN_STRING="1/(x*pow(x,x))" \
#         RESULTS_DIR=fitting/output/output_mightee_rhoTrue \
#         ESR_FUNCTION_LIBRARY_DIR=/path/to/function_library/core_maths \
#         bash submit_posterior_hmc_glamdring.sh
#
# Conventions carried over from submit_mightee_comp_glamdring.sh (see that
# script's own header for the reasoning): addqueue instead of sbatch, no
# mpirun wrapper, PYTHON_BIN resolved to an absolute path via `command -v`,
# ESR_FUNCTION_LIBRARY_DIR required with no Mac-path fallback (addqueue jobs
# aren't guaranteed to inherit the submitting shell's environment, so it's
# baked into each job's command line as --fn-library-dir), -q redwood/-n 1/
# -m 10 untuned defaults.
set -u

comp="${COMP:-}"
fcn_string="${FCN_STRING:-}"
if [ -z "$comp" ] || [ -z "$fcn_string" ]; then
    echo "Error: COMP and FCN_STRING must both be set. Usage:"
    echo '  COMP=6 FCN_STRING="1/(x*pow(x,x))" ESR_FUNCTION_LIBRARY_DIR=... bash submit_posterior_hmc_glamdring.sh'
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
fn_library_dir="${ESR_FUNCTION_LIBRARY_DIR:-}"
if [ -z "$fn_library_dir" ]; then
    echo "Error: ESR_FUNCTION_LIBRARY_DIR not set. Required on Glamdring -- the" \
         "code's own default is a Mac-only path. Example:"
    echo "  ESR_FUNCTION_LIBRARY_DIR=/path/to/function_library/core_maths"
    exit 1
fi
results_dir="${RESULTS_DIR:-}"
if [ -z "$results_dir" ]; then
    echo "Error: RESULTS_DIR not set. plot_posterior_check.py's own default is a" \
         "Mac-local snapshot-folder path with no reason to exist on any other" \
         "machine. Point it at wherever run_esr_mightee.py actually wrote this" \
         "run's per-galaxy output here, e.g.:"
    echo "  RESULTS_DIR=fitting/output/output_mightee_rhoTrue"
    exit 1
fi

[ -f "$galaxy_names_file" ] || { echo "Error: '$galaxy_names_file' not found."; exit 1; }

# safe_fcn_name's own sanitization, reproduced here only to name the raw-dir
# consistently with what merge_posterior_hmc.py will independently derive
# from meta.json's own fcn_string field -- not load-bearing for correctness
# (meta.json is the source of truth both scripts actually read), just for a
# human-readable, collision-free directory name.
tag="comp${comp}_$(echo "$fcn_string" | sed -E 's/[^0-9a-zA-Z]+/_/g; s/^_+|_+$//g')"
raw_dir="posterior_check_output/hmc_raw_${tag}"
meta_path="${raw_dir}/meta.json"
group_name="posterior_hmc_${tag}"

echo "=== Stage 1/3: preparing shared meta (nshape, fallback anchors) ==="
"$python_bin" prepare_posterior_hmc_meta.py --comp "$comp" --fcn-string "$fcn_string" \
    --fn-library-dir "$fn_library_dir" --results-dir "$results_dir" --meta-out "$meta_path"
if [ $? -ne 0 ]; then
    echo "Error: prepare_posterior_hmc_meta.py failed -- not submitting any array jobs."
    exit 1
fi

echo
echo "=== Stage 2/3: submitting one array job per galaxy ==="
n_gal=$(grep -cve '^[[:space:]]*$' "$galaxy_names_file")
gal_idx=0

while IFS= read -r galaxy || [ -n "$galaxy" ]; do
    [ -z "${galaxy// }" ] && continue
    gal_idx=$((gal_idx + 1))
    echo "submitting galaxy $gal_idx/$n_gal: $galaxy (comp=$comp, fcn='$fcn_string')"
    addqueue -q "$queue" -n "$n_cores" -m "$mem_gb" --group "$group_name" \
        "$python_bin" run_posterior_hmc_single.py --galaxy "$galaxy" \
        --meta "$meta_path" --out-dir "$raw_dir" \
        --fn-library-dir "$fn_library_dir" --results-dir "$results_dir" --seed "$gal_idx"
done < "$galaxy_names_file"

echo
echo "Submitted $gal_idx galaxy jobs under group '$group_name'."
echo
echo "=== Stage 3/3: once every job in that group has finished (check with"
echo "    e.g. 'qstat -u \$USER' or your cluster's own job-status command),"
echo "    run the merge step to produce the final PDF + summary: ==="
echo
echo "$python_bin merge_posterior_hmc.py --meta $meta_path --raw-dir $raw_dir \\"
echo "    --out-dir posterior_check_output --fn-library-dir $fn_library_dir"
