#!/bin/bash
#
# Which channel reconstructs a known place field, and does that survive a
# change of room shape?
#
# Runs the recovery test across three environments that hold area (~314 m^2)
# and landmark count (8) fixed while varying shape -- disc, rectangle,
# corridor -- for every feature channel, under the new sigma definition only
# (SIGMA_MODE='quantile', EXTENT_PCTL=65). Ideal place cells vary in size
# (0.5-3.0 m) and in location (contours at fixed distance from the wall).
#
# Usage:
#   sbatch slurm/geometry_recovery.sh
#   sbatch slurm/geometry_recovery.sh --channels hog,color --sites 4
#   sbatch slurm/geometry_recovery.sh --envs corr_lm8_r0 --verify
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=geo-recov
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# No --gres. Unlike the other analysis jobs this one never builds a full
# pairwise distance matrix: recovery evaluates one centroid against every
# position, which is a matvec, and the sigma statistic only ever touches the
# <=512 sampled members of one group. Nothing here calls
# rules.feature_sq_distances, so a GPU would sit idle -- and a CPU-only job
# schedules on far more of the cluster, GPU6 included.
#
# 64G, not the 128G the other scripts ask for. Peak working set is the
# feature matrix for the widest channel (`all`, ~7.2 GB) alongside the
# per-channel blocks it was assembled from (~7.2 GB), so ~16 GB with room to
# spare. The larger request would exclude GPU6 for no reason.
# --------------------------------------------------------------------------

set -euo pipefail

EXTRA_ARGS=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm/logs

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate realm-vpce

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

JOB_ID="${SLURM_JOB_ID:-local}"

echo "===================================================================="
echo "Job      : ${JOB_ID}"
echo "Extra    : ${EXTRA_ARGS[*]:-(none)}"
echo "Started  : $(date -Is)"
echo "Git      : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

set +e
python analysis/experiment_channel_isolation/run_geometry_recovery.py \
    "${EXTRA_ARGS[@]}"
STATUS=$?
set -e

echo "Finished : $(date -Is)  (exit ${STATUS})"

# Results are files, not an emailed report: metrics.csv and masks.npz in
# data_cache/geometry_recovery/ are the deliverable, and initial_results.txt
# is echoed into the log above. Mail only on failure, so a dead run is not
# silent.
if [[ ${STATUS} -ne 0 && -n "${EMAIL_TO:-}" ]]; then
    python - "${JOB_ID}" "${STATUS}" \
             "slurm/logs/${SLURM_JOB_NAME:-geo-recov}-${JOB_ID}.out" <<'PY' \
        || echo "(failure mailer failed — job status unchanged)"
import sys
from realm_tools.experiment_lib.reporting import send_email
job, status, log = sys.argv[1:4]
try:
    tail = ''.join(open(log, errors='replace').readlines()[-60:])
except OSError:
    tail = '(log unavailable)'
send_email(f'[REALM-VPCE] geometry-recovery FAILED (exit {status}, job {job})',
           f'The run exited {status} before it could finish.\n\n'
           f'Last 60 log lines:\n\n{tail}',
           attachments=[log])
PY
fi

exit ${STATUS}
