#!/bin/bash
#
# Redraw figures and re-send reports from the data cache.
#
# For a change of title, label or layout: nothing is recomputed. Each
# experiment rebuilds its figures and its email from the tables its last
# finished run left in data_cache/, in minutes.
#
#   prune        prune audit        P1, P2     run_prune_audit.py --report-only
#   geometry     field geometry     G1, G2, G4 run_field_geometry.py --report-only
#   validation   extent validation  V1-V7      run_extent_validation.py --report-only
#   scale        scale distribution S1-S3      run_scale_distribution.py --use-cache
#
# The first three read CSVs only. `scale` is the exception: it reuses its
# cached field libraries but still reads each arena's dataset and refits the
# distributions, so it is the slow one and is not in the default.
#
# USAGE
#
#   sbatch slurm/replot.sh                        # prune + geometry
#   sbatch slurm/replot.sh prune                  # one of them
#   sbatch slurm/replot.sh prune geometry validation
#   sbatch slurm/replot.sh all                    # all four, scale included
#   sbatch slurm/replot.sh --no-email geometry    # figures only, no mail
#   bash   slurm/replot.sh --dry-run all          # print what would run, run nothing
#
# Any other --option is passed to every experiment's script. Each experiment
# mails its own report, so expect one email per name. One that fails does not
# stop the others; the job exits non-zero if any did, and mails the log.
#
# It needs a finished run of each experiment in data_cache/. An experiment
# with nothing cached says so and is counted as failed.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=replot
#SBATCH --partition=general
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# No GPU is requested, because none is used. If the partition refuses a job
# without one, add it on the command line and nothing else changes:
#
#   sbatch --gres=gpu:1 slurm/replot.sh
#
# 32G and 4 h are for `scale`, which loads one arena's features at a time. The
# other three finish in minutes inside a few GB.
# --------------------------------------------------------------------------

set -euo pipefail

DRY=0
NAMES=()
PASS=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --*)       PASS+=("$a") ;;
        all)       NAMES+=(prune geometry validation scale) ;;
        prune|geometry|validation|scale) NAMES+=("$a") ;;
        *) echo "replot: unknown name '$a' (prune, geometry, validation, scale, all)" >&2
           exit 2 ;;
    esac
done
[[ ${#NAMES[@]} -eq 0 ]] && NAMES=(prune geometry)

A=analysis/experiment_channel_isolation
command_for() {
    case "$1" in
        prune)      echo "$A/run_prune_audit.py --report-only" ;;
        geometry)   echo "$A/run_field_geometry.py --report-only" ;;
        validation) echo "$A/run_extent_validation.py --report-only" ;;
        scale)      echo "$A/run_scale_distribution.py --use-cache" ;;
    esac
}

if [[ $DRY -eq 1 ]]; then
    for n in "${NAMES[@]}"; do
        echo "python $(command_for "$n") ${PASS[*]+"${PASS[*]}"}"
    done
    exit 0
fi

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm/logs

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-realm-vpce}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

JOB_ID="${SLURM_JOB_ID:-local}"
export REALM_LOG_PATH="slurm/logs/${SLURM_JOB_NAME:-replot}-${JOB_ID}.out"

echo "===================================================================="
echo "Job     : ${JOB_ID}   node $(hostname)"
echo "Redraw  : ${NAMES[*]}"
echo "Options : ${PASS[*]:-(none)}"
echo "Email   : ${EMAIL_TO:-(EMAIL_TO unset - reports will not send)}"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

# One experiment failing must not cost the others their figures, so each
# status is kept and the job reports the lot at the end.
FAILED=()
for n in "${NAMES[@]}"; do
    echo
    echo "---------------------------------------------------------- ${n}"
    set +e
    # shellcheck disable=SC2046
    python $(command_for "$n") ${PASS[@]+"${PASS[@]}"}
    st=$?
    set -e
    echo "---- ${n}: exit ${st}"
    [[ $st -ne 0 ]] && FAILED+=("$n")
done

echo
echo "Finished : $(date -Is)"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All redrawn: ${NAMES[*]}"
    exit 0
fi

echo "FAILED: ${FAILED[*]}"
if [[ -n "${EMAIL_TO:-}" ]]; then
    python - "${JOB_ID}" "${FAILED[*]}" "${REALM_LOG_PATH}" <<'PY' \
        || echo "(failure mailer failed — job status unchanged)"
import sys
from realm_tools.experiment_lib.reporting import send_email
job, failed, log = sys.argv[1:4]
try:
    tail = ''.join(open(log, errors='replace').readlines()[-80:])
except OSError:
    tail = '(log unavailable)'
send_email(f'[REALM-VPCE] replot FAILED for {failed} (job {job})',
           f'These could not be redrawn from the cache: {failed}.\n\n'
           f'Last 80 log lines:\n\n{tail}', attachments=[log])
PY
fi
exit 1
