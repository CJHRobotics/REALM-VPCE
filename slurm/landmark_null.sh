#!/bin/bash
#
# Do landmarks organise where our fields sit, or how large they are?
#
# Reproduces Eliav et al. 2021's five landmark tests over circ_lm4/6/8/10_r0,
# which vary interlandmark spacing from 15.7 m to 6.3 m by construction. Our
# model is driven purely by view appearance and so has an obvious route to
# over-predicting landmark dependence; this is the sharpest available test of
# whether it produces a spatial code or a landmark-proximity detector.
#
# `lidar` is the control: it cannot see the panels, so any landmark
# relationship it shows is arena geometry rather than landmark appearance.
#
# Usage:
#   sbatch slurm/landmark_null.sh                          # all four arenas
#   sbatch slurm/landmark_null.sh --envs circ_lm8_r0       # one, in parallel
#   sbatch slurm/landmark_null.sh --use-cache              # reuse field banks
#
# One arena per job is worth it if the queue allows: the cost is dominated by
# building a field library per channel, and the arenas are independent.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=lm-null
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# GPU, unlike the collection jobs: this one builds a full pairwise feature
# distance matrix per channel through rules.feature_sq_distances, which is
# what the card is for. Any GPU will do -- the wide channels (visual 7.2 GB,
# all 7.3 GB) do not fit an 11 GB 1080 Ti alongside their working set, but
# feature_sq_distances checks free VRAM and falls back to the CPU for that
# stage, so a small card costs time rather than correctness. Pinning a type
# queues behind those nodes for more than it saves.
#
# 128G because the pipeline holds the feature matrix, its pairwise block and
# the candidate responses at once. No container: this reads the collected
# HDF5 datasets and never touches Webots.
#
# The experiment mails its own report with figures through
# realm_tools.experiment_lib.reporting; it needs EMAIL_TO exported.
# --------------------------------------------------------------------------

set -euo pipefail

EXTRA_ARGS=("$@")

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
export REALM_LOG_PATH="slurm/logs/${SLURM_JOB_NAME:-lm-null}-${JOB_ID}.out"

echo "===================================================================="
echo "Job     : ${JOB_ID}   node $(hostname)"
echo "Extra   : ${EXTRA_ARGS[*]:-(none)}"
echo "Email   : ${EMAIL_TO:-(EMAIL_TO unset - report will not send)}"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "GPU     : none visible (will run on CPU)"
echo "===================================================================="

# `set -e` would abort before the report could be sent for a failing run --
# exactly the run worth hearing about. Capture the status by hand.
set +e
python analysis/experiment_channel_isolation/run_landmark_null.py \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
STATUS=$?
set -e

echo "Finished : $(date -Is)  (exit ${STATUS})"

if [[ ${STATUS} -ne 0 && -n "${EMAIL_TO:-}" ]]; then
    python - "${JOB_ID}" "${STATUS}" "${REALM_LOG_PATH}" <<'PY' \
        || echo "(failure mailer failed — job status unchanged)"
import sys
from realm_tools.experiment_lib.reporting import send_email
job, status, log = sys.argv[1:4]
try:
    tail = ''.join(open(log, errors='replace').readlines()[-60:])
except OSError:
    tail = '(log unavailable)'
send_email(f'[REALM-VPCE] landmark-null FAILED (exit {status}, job {job})',
           f'The run exited {status} before it could report.\n\n'
           f'Last 60 log lines:\n\n{tail}', attachments=[log])
PY
fi

exit ${STATUS}
