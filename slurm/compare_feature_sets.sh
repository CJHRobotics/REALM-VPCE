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
# ---- fill in GAIVI-specific lines before first submission ---------------
# #SBATCH --partition=<partition>
# #SBATCH --account=<account>
# #SBATCH --qos=<qos>
# --------------------------------------------------------------------------

set -euo pipefail

ENV="${1:-circ_lm8_r0}"

# Locate the repo (this script's dir is <repo>/slurm)
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p slurm/logs

# ---- environment ---------------------------------------------------------
# module load python/3.11        # uncomment + adjust if GAIVI uses modules
# module load anaconda           # or your preferred toolchain

# Activate the project venv (create it once on GAIVI:
#   python -m venv realm_venv && source realm_venv/bin/activate
#   pip install -r setup/requirements.txt )
if [[ -f "realm_venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source realm_venv/bin/activate
fi

# BLAS thread count — match the SBATCH cpus-per-task
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "===================================================================="
echo "Job     : ${SLURM_JOB_ID:-local}"
echo "Env     : ${ENV}"
echo "Repo    : ${REPO_DIR}"
echo "Threads : ${OMP_NUM_THREADS}"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

python analysis/experiment_feature_selection/compare_feature_sets.py "${ENV}"

echo "Finished: $(date -Is)"
