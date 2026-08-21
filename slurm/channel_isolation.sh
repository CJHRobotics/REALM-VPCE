#!/bin/bash
#
# Channel isolation x spatial-weighting sweep on GAIVI.
#
# Builds a rule-governed place-field bank for every (feature channel, lambda)
# pair: 6 channels x 4 lambdas = 24 runs by default.
#
# Usage:
#   sbatch slurm/channel_isolation.sh [env_name] [extra args...]
#
# Examples:
#   sbatch slurm/channel_isolation.sh circ_lm8_r0
#   sbatch slurm/channel_isolation.sh corridor_lm10 --lambdas 0,0.5
#   sbatch slurm/channel_isolation.sh circ_lm8_r0 --subsample 15000
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=chan-iso
#SBATCH --partition=general
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
# Any GPU will do. The wide channels (visual 7.2 GB, all 7.3 GB) do not fit an
# 11 GB 1080 Ti alongside their working set, but rules.feature_sq_distances and
# rules.environment_readout both check free VRAM and fall back to the CPU for
# that stage, so a small card costs time, not correctness. Pinning the GRES to
# a named type (`gpu:A40:1`) instead sits the job behind whatever is queued for
# those four nodes, which in practice has cost far more than the CPU fallback
# ever does. Add `--gres=gpu:A40:1` at submit time if a run needs the speed:
#   sbatch --gres=gpu:A40:1 slurm/<job>.sh circ_lm8_r0
# GPU types on `general` (exact strings, from `sinfo -p general -N -o "%N %G"`):
#   GPU6   gpu:1080Ti:4    11 GB, sm_61, no TF32
#   GPU42  gpu:TitanRTX:4  24 GB, sm_75, no TF32
#   GPU43  gpu:A40:4       48 GB, sm_86, TF32
#   GPU44  gpu:A40:4       48 GB, sm_86, TF32
# Type strings are case-sensitive: `A40`, not `a40`.
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
# Note: this SLURM does NOT substitute %h. Path is repo-relative, so run
# `mkdir -p slurm/logs` in the repo ONCE before your first submission.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chamilton4@usf.edu
# --------------------------------------------------------------------------
# Resource notes for N = 30147 locations (circ_lm8_r0):
#   GPU  — the N x N feature Gram (~5e16 MACs for the 59632-d 'all' config)
#          and the environment readout. Needs ~11 GB of device memory for
#          the widest config; falls back to CPU automatically if no GPU is
#          visible, at roughly 20x the wall time.
#   RAM  — peak ~20 GB: feature matrix (7.2 GB) + float32 distance matrix
#          (3.6 GB) + float64 condensed vector for linkage (3.6 GB) and its
#          internal copy. 128G leaves comfortable headroom.
#   CPU  — Ward linkage is sequential; the cores are used by BLAS on the
#          CPU fallback path and by the per-candidate mask work.
#
# Rich report (metrics + figures attached) is opt-in via EMAIL_TO. Set the
# defaults once on GAIVI in ~/.bashrc:
#   EMAIL_TO=chamilton4@usf.edu  EMAIL_FROM=chamilton4@usf.edu
#   EMAIL_SMTP=smtp.usf.edu      EMAIL_SMTP_PORT=587  (STARTTLS auto)

set -euo pipefail

ENV="${1:-circ_lm8_r0}"
shift || true
EXTRA_ARGS=("$@")

# SLURM copies the batch script to spool before running it, so BASH_SOURCE
# points into /var/spool/slurm on the compute node. SLURM_SUBMIT_DIR is the
# directory sbatch was called from — our workflow submits from the repo root.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

mkdir -p slurm/logs

# ---- environment ---------------------------------------------------------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate realm-vpce

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

JOB_NAME="${SLURM_JOB_NAME:-chan-iso}"
JOB_ID="${SLURM_JOB_ID:-local}"
LOG_PATH="slurm/logs/${JOB_NAME}-${JOB_ID}.out"

echo "===================================================================="
echo "Job      : ${JOB_ID}"
echo "Env      : ${ENV}"
echo "Extra    : ${EXTRA_ARGS[*]:-(none)}"
echo "Repo     : ${REPO_DIR}"
echo "Threads  : ${OMP_NUM_THREADS}"
echo "Started  : $(date -Is)"
echo "Git      : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "GPU      : none visible (will run on CPU)"
# A GPU can be allocated while torch is still a CPU-only build, which looks
# identical in the log until the readout runs slowly. Print the build.
python - <<'PY' 2>/dev/null || echo "Torch    : import failed"
import torch
print(f"Torch    : {torch.__version__}  cuda_build={torch.version.cuda}  "
      f"available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
PY
echo "===================================================================="

# `set -e` would abort the script the moment python returned non-zero, so
# the report below would never be sent for exactly the runs you most want
# to see. Disable it around the call and capture the status by hand.
set +e
python analysis/experiment_channel_isolation/run_channel_isolation.py \
    "${ENV}" "${EXTRA_ARGS[@]}"
STATUS=$?
set -e

echo "Finished : $(date -Is)  (exit ${STATUS})"

# ---- email the summary + figures -----------------------------------------
# Silent no-op if EMAIL_TO isn't exported; won't fail the job either way.
if [[ -n "${EMAIL_TO:-}" ]]; then
    FIG_DIR="analysis/experiment_channel_isolation/figures/${ENV}"
    python slurm/send_report.py \
        "REALM-VPCE ${ENV} channel_isolation (job ${JOB_ID}, exit ${STATUS})" \
        "${LOG_PATH}" \
        "${FIG_DIR}/F1_field_maps.png" \
        "${FIG_DIR}/F2_lambda_effect.png" \
        "${FIG_DIR}/F3_anisotropy.png" \
        "${FIG_DIR}/F7_size_vs_wall.png" \
        || echo "(mailer failed — job status unchanged)"
fi

exit ${STATUS}
