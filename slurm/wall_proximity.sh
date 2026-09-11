#!/bin/bash
#
# Experiment 3 — do large place fields sit further from the walls and landmarks?
#
# Reads the field libraries Experiment 2 built, in the same four arenas and
# under the same place-field configuration, and asks where fields of a given
# size are located:
#
#   circ_lm8_r3      r = 3     28.27 m^2   small
#   circ_lm8_r6      r = 6    113.10 m^2   medium
#   circ_lm8_r10     r = 10   314.16 m^2   mega
#   corr_lm8_l10w2   10 x 2 m  20.00 m^2   corridor
#
# Same configuration means exactly Experiment 2's: EXTENT_PCTL 65, ACT_THRESH
# 0.5, Rule 2 off, LAMBDA 0, seed 0. None of it is exposed as an option. The
# libraries are read from Experiment 2's cache (data_cache/scale_distribution)
# when they are there and built into it when they are not, with the same code
# and the same cache key, so the two experiments describe the same fields.
#
# Fields are split by area within each library: bottom 50% small, top 10%
# large. Each field clear of the wall is compared with where its own shape
# lands at random, placed only where it fits whole, and the test is whether
# large fields sit further out than that while small ones do not. (Fields the
# wall cut are left out of the wall test: the library records an ellipse, so
# their shape beyond the wall is unknown, and guessing it biased the test.) Two random baselines: uniform, and one
# that also obeys Rule 11's spacing, as the real library had to. Landmarks are
# tested by position along the wall relative to the panels, so they cannot
# stand in for wall distance. See the module docstring of
# analysis/experiment_channel_isolation/run_wall_proximity.py.
#
# This replaced a correlation of size with wall distance over the whole
# library, which found nothing: the small fields, about 60% of every library,
# sit at every distance and swamp any correlation.
#
# Usage:
#   sbatch slurm/wall_proximity.sh                          # all four arenas
#   sbatch slurm/wall_proximity.sh --envs circ_lm8_r3       # one arena
#   sbatch slurm/wall_proximity.sh --n-null 200             # a quick look
#   sbatch slurm/wall_proximity.sh --rebuild                # ignore the cache
#
# If Experiment 2's libraries are cached, one job does everything in well
# under an hour: no feature blocks are loaded and the GPU goes unused. If any
# are missing, fan the builds out one arena per job as Experiment 2 does, then
# run once over all four for the combined figures and report:
#
#   for e in circ_lm8_r3 circ_lm8_r6 circ_lm8_r10 corr_lm8_l10w2; do
#       sbatch --job-name=wall-prox-build slurm/wall_proximity.sh --envs "$e" --no-email
#   done
#   sbatch slurm/wall_proximity.sh
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=wall-prox
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
# The header is sized for the worst case, where every library has to be built:
# that is Experiment 2's cost and needs its GPU and memory. With the cache in
# place the job needs neither, and a lighter request schedules sooner:
#
#   sbatch --mem=32G --time=2:00:00 slurm/wall_proximity.sh
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
export REALM_LOG_PATH="slurm/logs/${SLURM_JOB_NAME:-wall-prox}-${JOB_ID}.out"

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
python analysis/experiment_channel_isolation/run_wall_proximity.py \
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
send_email(f'[REALM-VPCE] wall-proximity FAILED (exit {status}, job {job})',
           f'The run exited {status} before it could report.\n\n'
           f'Last 60 log lines:\n\n{tail}', attachments=[log])
PY
fi

exit ${STATUS}
