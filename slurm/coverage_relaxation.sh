#!/bin/bash
#
# Side analysis on Experiment 2 — what Rule 12's coverage test was deleting.
#
# Same eight arenas, same channels, same operating point, same code. One knob
# moved: TILING_FRAC_MIN, the coverage threshold in Rule 12 (tiling stop). A
# scale whose admitted fields, unioned, cover less than that fraction of the
# floor is dropped whole, and the surviving ladder is the contiguous run of
# qualifying scales around the best-covered one. The default is 0.50.
#
# Swept: 0.50 (the operating point), 0.35, 0.20, 0.00 (Rule 12 off).
#
# A NOTE ON THE RULE NUMBER. Rule 4 in this codebase is spatial weighting --
# merge cost = feature distance + LAMBDA * space -- and it is already off at
# the operating point (LAMBDA 0), so toggling it changes nothing. Coverage is
# Rule 12. This sweeps Rule 12.
#
# Why this is a clean one-rule change
# -----------------------------------
# Rule 12 runs LAST, after Rule 11's competition, and only deletes whole
# scales from a ladder that is already settled. Nothing upstream reads the
# threshold: not the tree, not the candidate set, not the size window, not
# which field beats which in competition. So lowering it can only ADD scales
# back, and each library is a strict superset of the one above it. That is
# asserted per library (`nested` in the summary), not assumed — if it ever
# fails, something upstream is reading the knob and the comparison is void.
#
# The same fact makes the sweep nearly free once the tree exists:
# prepare_candidates is built once per arena and channel and every threshold
# is scored against an identical tree, so a difference between thresholds
# cannot be a difference in the tree.
#
# It also checks itself against Experiment 2. At 0.50 the bank it builds must
# be identical, field for field, to the one in Experiment 2's cache — same
# code, same settings, same seed. `matches_exp2` says whether it was. A
# mismatch means the operating-point column is not Experiment 2's library and
# nothing in the comparison can be read.
#
# THIS CANNOT REUSE EXPERIMENT 2'S BANKS
# --------------------------------------
# Those hold the survivors, and the whole question is about the scales Rule 12
# dropped. So the pipeline runs in full — feature blocks, Gram matrix, Ward
# tree — at Experiment 2's cost, once per arena and channel. It writes to its
# own cache under its own key (`data_cache/coverage_relaxation`, with the
# threshold in the filename); Experiment 2's cache is read for the self-check
# and never written.
#
# Usage — fan out one arena per job, as Experiment 2 does:
#
#   for e in circ_lm8_r3 circ_lm8_r6 circ_lm0_r3 circ_lm0_r6 \
#            corr_lm8_l10w10 corr_lm0_l10w10 corr_lm8_l10w2 corr_lm0_l10w2; do
#       sbatch slurm/coverage_relaxation.sh --envs "$e" --no-email
#   done
#
# then, once those finish, one combining pass over all eight for the
# cross-arena figures and the single report. That pass reads this script's own
# cache, loads no feature blocks and needs no GPU:
#
#   sbatch --mem=32G --time=2:00:00 slurm/coverage_relaxation.sh
#
# The combining pass is not optional: C1 through C4 put the arenas side by
# side, and every single-arena job writes the same summary files, so one that
# finishes late would overwrite the combined output.
#
# Other options:
#   --tiling 0.5,0.4,0.3,0.2,0.1,0   a finer sweep (cheap: same tree)
#   --channels hog,color             fewer channels
#   --rebuild                        ignore this script's cache
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=cov-relax
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
# Sized as Experiment 2 is, because building the sweep IS Experiment 2's work:
# the feature Gram matrix per channel is what the GPU is for, and 128G holds
# the feature matrix, its pairwise block and the candidate responses at once.
# The wide channels (visual 7.2 GB, all 7.3 GB) do not fit an 11 GB card
# alongside their working set, but feature_sq_distances checks free VRAM and
# falls back to the CPU for that stage, so a small card costs time rather than
# correctness.
#
# The four thresholds are nearly free: they share one tree and one candidate
# context, and only admit_fields repeats. A finer sweep costs almost nothing.
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
export REALM_LOG_PATH="slurm/logs/${SLURM_JOB_NAME:-cov-relax}-${JOB_ID}.out"

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
python analysis/experiment_channel_isolation/run_coverage_relaxation.py \
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
send_email(f'[REALM-VPCE] coverage-relaxation FAILED (exit {status}, job {job})',
           f'The run exited {status} before it could report.\n\n'
           f'Last 60 log lines:\n\n{tail}', attachments=[log])
PY
fi

exit ${STATUS}
