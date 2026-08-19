#!/bin/bash
#
# Field recovery and discrimination on GAIVI.
#
# Hands the extent rule a known place field -- a disc of floor -- and measures
# what it returns, then hands it things that are not place fields and measures
# whether it rejects them. The second half is the point: a rule that returns a
# tidy compact field from any input scores perfectly on recovery alone.
#
# Compares the current width statistic (90th percentile of within-group
# pairwise distance) against SIGMA_MODE='quantile', and calibrates
# EXTENT_PCTL on the rate at which real fields are admitted minus the rate at
# which non-fields are.
#
# Usage:
#   sbatch slurm/field_recovery.sh [env_name] [extra args...]
#
# Examples:
#   sbatch slurm/field_recovery.sh circ_lm8_r0
#   sbatch slurm/field_recovery.sh circ_lm8_r0 --tests 1,2 --channels color,hog
#   sbatch slurm/field_recovery.sh circ_lm8_r0 --pctls 50,65,80,90
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=fieldrec
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:A40:1
# GPU types on `general` (exact GRES strings, from `sinfo -p general -N -o "%N %G"`):
#   GPU6   gpu:1080Ti:4    11 GB, sm_61, no TF32
#   GPU42  gpu:TitanRTX:4  24 GB, sm_75, no TF32
#   GPU43  gpu:A40:4       48 GB, sm_86, TF32
#   GPU44  gpu:A40:4       48 GB, sm_86, TF32
# Arm 3 holds a feature matrix and its full pairwise block at once (7.2 GB +
# 3.6 GB for `all`), so the 11 GB 1080 Ti is not enough. Type strings are
# case-sensitive: `A40`, not `a40`.
# Override with a plain `--gres=gpu:1` to take whatever is free.
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chamilton4@usf.edu
# --------------------------------------------------------------------------
# The rich report (summary + figures + metrics.csv) is sent by the experiment
# itself through realm_tools.experiment_lib.reporting, not by this script.
# It needs EMAIL_TO exported; set the defaults once in ~/.bashrc on GAIVI:
#   export EMAIL_TO=chamilton4@usf.edu EMAIL_FROM=chamilton4@usf.edu
#   export EMAIL_SMTP=smtp.usf.edu EMAIL_SMTP_PORT=587
# Without EMAIL_TO the send is a silent no-op and the job still succeeds.

set -euo pipefail

ENV="${1:-circ_lm8_r0}"
shift || true
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
echo "Env      : ${ENV}"
echo "Extra    : ${EXTRA_ARGS[*]:-(none)}"
echo "Started  : $(date -Is)"
echo "Git      : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "GPU      : none visible (will run on CPU)"
python - <<'PY' 2>/dev/null || echo "Torch    : import failed"
import torch
print(f"Torch    : {torch.__version__}  cuda_build={torch.version.cuda}  "
      f"available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
PY
echo "===================================================================="

# `set -e` would abort before the report could be sent for a failing run —
# exactly the run worth hearing about. Capture the status by hand instead.
set +e
python analysis/experiment_channel_isolation/run_field_recovery.py \
    "${ENV}" "${EXTRA_ARGS[@]}"
STATUS=$?
set -e

echo "Finished : $(date -Is)  (exit ${STATUS})"

# The experiment mails its own report on success. If it died before reaching
# that point, send a bare failure notice with the log so the failure is not
# silent.
if [[ ${STATUS} -ne 0 && -n "${EMAIL_TO:-}" ]]; then
    python - "${ENV}" "${JOB_ID}" "${STATUS}" \
             "slurm/logs/${SLURM_JOB_NAME:-fieldrec}-${JOB_ID}.out" <<'PY' \
        || echo "(failure mailer failed — job status unchanged)"
import sys
from realm_tools.experiment_lib.reporting import send_email
env, job, status, log = sys.argv[1:5]
try:
    tail = ''.join(open(log, errors='replace').readlines()[-60:])
except OSError:
    tail = '(log unavailable)'
send_email(f'[REALM-VPCE] {env} field-recovery FAILED (exit {status}, job {job})',
           f'The run exited {status} before it could report.\n\n'
           f'Last 60 log lines:\n\n{tail}',
           attachments=[log])
PY
fi

exit ${STATUS}
