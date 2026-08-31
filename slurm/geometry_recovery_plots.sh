#!/bin/bash
#
# Figures for the geometry x channel recovery experiment.
#
# Reads what run_geometry_recovery.py saved (metrics.csv, masks.npz) and
# writes the figures, a correlation table, and the ranking notes. Does no
# modelling of its own, so it is cheap -- but it still belongs in the
# scheduler rather than on a login node: it loads the full mask archive and
# computes second moments over every reference field.
#
# Usage:
#   sbatch slurm/geometry_recovery_plots.sh
#   sbatch slurm/geometry_recovery_plots.sh --in /path/to/results
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=geo-plots
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# No GPU, and modest CPU and memory: this is matplotlib plus one pass over
# the saved masks. Sized small on purpose so it schedules immediately.
# --------------------------------------------------------------------------

set -euo pipefail

EXTRA_ARGS=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm/logs

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate realm-vpce

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "===================================================================="
echo "Job      : ${SLURM_JOB_ID:-local}"
echo "Extra    : ${EXTRA_ARGS[*]:-(none)}"
echo "Started  : $(date -Is)"
echo "Git      : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

python analysis/experiment_channel_isolation/plot_geometry_recovery.py \
    "${EXTRA_ARGS[@]}"

echo "Finished : $(date -Is)"
