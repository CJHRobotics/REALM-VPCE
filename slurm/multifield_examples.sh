#!/bin/bash
#
# Multi-field responses: draw the clusters that contiguity rejects.
#
# The model admits a cluster only if its field is one connected patch of
# floor. The ones that fail often respond in several separate places -- a
# multi-field place cell, which the single-field model produces and then
# throws away. This rebuilds ONE field library (one arena, one channel), keeps
# the candidates that passed the size rule and failed contiguity, and draws
# their responses on the floor: one figure per cluster, plus one overview
# sheet, mailed with a count of how many rejects really are multi-field.
#
# A "subfield" is a connected patch of the field at least as large as the
# smallest admissible field. A reject with one subfield and some fragments is
# not a multi-field cell, and the report counts those separately.
#
# It has to rebuild the library: a rejected cluster's response is saved
# nowhere. One library is one Gram matrix, one tree and one readout -- minutes
# on a GPU.
#
# USAGE
#
#   sbatch slurm/multifield_examples.sh                       # corridor with landmarks, HOG
#   sbatch slurm/multifield_examples.sh --env circ_lm8_r6 --channel color
#   sbatch slurm/multifield_examples.sh --n 20                # more clusters
#   sbatch slurm/multifield_examples.sh --min-subfields 3     # three places or more
#
# The default pair lost the most candidates to contiguity among the arenas
# with landmarks in the prune audit. An arena without landmarks gives more
# multi-field clusters still, but many of those are the arena's own symmetry.
#
# Figures: analysis/experiment_channel_isolation/figures/multifield/<env>_<channel>/
#   M00_overview, then M01.. one per cluster; PNG at 300 dpi and PDF.
# Table:   data_cache/multifield/<env>_<channel>_rejected.csv, every reject.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=multifield
#SBATCH --partition=general
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# 64G: one arena's features (~7 GB), their pairwise block (~4 GB) and the
# candidate responses, live at once. Any GPU; a small card falls back to the
# CPU for the widest channels, which costs time, not correctness.
#
# The report is mailed by the script itself; it needs EMAIL_TO exported.
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
export REALM_LOG_PATH="slurm/logs/${SLURM_JOB_NAME:-multifield}-${JOB_ID}.out"

echo "===================================================================="
echo "Job     : ${JOB_ID}   node $(hostname)"
echo "Extra   : ${EXTRA_ARGS[*]:-(none)}"
echo "Email   : ${EMAIL_TO:-(EMAIL_TO unset - report will not send)}"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "GPU     : none visible (will run on CPU)"
echo "===================================================================="

# `set -e` would abort before a failure could be mailed. Capture the status.
set +e
python analysis/experiment_channel_isolation/run_multifield_examples.py \
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
send_email(f'[REALM-VPCE] multifield-examples FAILED (exit {status}, job {job})',
           f'The run exited {status} before it could report.\n\n'
           f'Last 60 log lines:\n\n{tail}', attachments=[log])
PY
fi

exit ${STATUS}
