#!/bin/bash
#
# Submit the three-way feature-selection comparison for one environment.
# Usage:
#   sbatch slurm/compare_feature_sets.sh <env_name>
# Default env is circ_lm8_r0.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=cmp-feat
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chamilton4@usf.edu
# ---- fill in GAIVI-specific lines before first submission ---------------
# #SBATCH --partition=<partition>
# #SBATCH --account=<account>
# #SBATCH --qos=<qos>
# --------------------------------------------------------------------------
# For the rich report (metrics + figures attached), export EMAIL_TO before
# submitting:  EMAIL_TO=chamilton4@usf.edu sbatch slurm/compare_feature_sets.sh
# If GAIVI's localhost mail relay refuses, set EMAIL_SMTP / EMAIL_SMTP_USER
# / EMAIL_SMTP_PASS (see slurm/send_report.py).

set -euo pipefail

ENV="${1:-circ_lm8_r0}"

# Locate the repo (this script's dir is <repo>/slurm)
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p slurm/logs

# ---- environment ---------------------------------------------------------
# GAIVI: activate the conda env (create it once with:
#   conda create -n realm-vpce python=3.11 -y
#   conda activate realm-vpce
#   pip install -r setup/requirements.txt )
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate realm-vpce

# BLAS thread count — match the SBATCH cpus-per-task
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Unbuffered stdout so the SLURM log we email later is complete when read.
export PYTHONUNBUFFERED=1

JOB_NAME="${SLURM_JOB_NAME:-cmp-feat}"
JOB_ID="${SLURM_JOB_ID:-local}"
LOG_PATH="slurm/logs/${JOB_NAME}-${JOB_ID}.out"

echo "===================================================================="
echo "Job     : ${JOB_ID}"
echo "Env     : ${ENV}"
echo "Repo    : ${REPO_DIR}"
echo "Threads : ${OMP_NUM_THREADS}"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

python analysis/experiment_feature_selection/compare_feature_sets.py "${ENV}"
STATUS=$?

echo "Finished: $(date -Is)  (exit ${STATUS})"

# ---- email the summary + comparison figure ------------------------------
# Silent no-op if EMAIL_TO isn't exported; won't fail the job either way.
if [[ -n "${EMAIL_TO:-}" ]]; then
    python slurm/send_report.py \
        "REALM-VPCE ${ENV} compare_feature_sets (job ${JOB_ID}, exit ${STATUS})" \
        "${LOG_PATH}" \
        "analysis/experiment_feature_selection/figures/${ENV}__compare.png" \
        || echo "(mailer failed — job status unchanged)"
fi

exit ${STATUS}
