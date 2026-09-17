#!/bin/bash
#
# Where do a channel's fields die? A per-scale audit of the admission rules.
#
# Experiment 2 reports what survived. This reports what did not, and at which
# rule, for one (arena, channel) pair at a time. It exists because several
# libraries collapse in ways the field counts alone cannot explain:
#
#   corr_lm0_l10w2   hog, spatial, visual   0 fields, where the same corridor
#                                           with panels gives 370, 266, 432
#   corr_lm8_l10w10  color                  7 fields, all at one coarse scale
#   corr_lm0_l10w10  spatial, visual        18 and 7 fields, coarse only
#
# For every candidate node the audit records the stage it died at:
#
#   no candidate    the tree never produced a node whose field lands at this
#                   scale -- the channel cannot localise to that size at all
#   rule 8/9 size   the mask fell below the floor or above the ceiling
#   rule 1          the mask was in pieces; the largest connected patch held
#                   less than CC_FRAC_MIN of it
#   rule 11         a larger field of the same scale already claimed the spot
#   rule 12         the scale survived competition but its fields covered less
#                   than TILING_FRAC_MIN of the floor, so the scale was dropped
#
# The counts come from the rules engine itself, which now records which
# candidates survived each stage, rather than from a second implementation of
# the same rules that could drift from it.
#
# Configuration is Experiment 2's operating point exactly -- EXTENT_PCTL 65,
# ACT_THRESH 0.5, Rule 2 off, LAMBDA 0, seed 0 -- so the audit describes the
# same libraries the reports do.
#
# Usage:
#   sbatch slurm/prune_audit.sh                       # the collapsed pairs, plus controls
#   sbatch slurm/prune_audit.sh --pairs corr_lm0_l10w2:hog
#   sbatch slurm/prune_audit.sh --pairs corr_lm8_l10w10:color,circ_lm8_r6:color
#   sbatch slurm/prune_audit.sh --envs corr_lm0_l10w2  # every channel of one arena
#
# Cost is a field library per pair: the Gram matrix, the Ward tree and the
# readout, the same work Experiment 2 does per channel. There is no cache to
# reuse -- the banks it writes are the survivors, and this needs the candidates
# that did not become survivors, which are never stored. Twelve pairs is
# roughly two arenas of Experiment 2, well inside the walltime.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=prune-audit
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# Any GPU: rules.feature_sq_distances checks free VRAM and falls back to the
# CPU for the widest channels, so a small card costs time, not correctness.
# 128G because the feature matrix, its pairwise block and the candidate
# responses are all live at once, as in Experiment 2.
#
# The experiment mails its own report through
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
export REALM_LOG_PATH="slurm/logs/${SLURM_JOB_NAME:-prune-audit}-${JOB_ID}.out"

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
python analysis/experiment_channel_isolation/run_prune_audit.py \
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
send_email(f'[REALM-VPCE] prune-audit FAILED (exit {status}, job {job})',
           f'The run exited {status} before it could report.\n\n'
           f'Last 60 log lines:\n\n{tail}', attachments=[log])
PY
fi

exit ${STATUS}
