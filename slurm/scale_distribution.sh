#!/bin/bash
#
# Experiment 2 — where our scale distribution sits.
#
# What shape is our field-size distribution, how does it compare to the three
# forms in the literature, and how does it move with environment scale?
#
# The default env list is all eight arenas, each with one role:
#
#   circ_lm8_r3       r = 3       28.27 m^2   area sweep, small
#   circ_lm8_r6       r = 6      113.10 m^2   area sweep, medium
#   circ_lm8_r10      r = 10     314.16 m^2   area sweep, mega
#   corr_lm8_l10w2    10 x 2 m    20.00 m^2   Eliav corridor
#   corr_lm8_l10w10   10 x 10 m  100.00 m^2   square, two panels on each wall
#   circ_lm0_r3       r = 3       28.27 m^2   no landmarks
#   circ_lm0_r6       r = 6      113.10 m^2   no landmarks
#   corr_lm0_l10w10   10 x 10 m  100.00 m^2   no landmarks
#
# Only the three lm8 discs enter the trend against area: 11.1x, against
# Harland's 8.8x. That is the axis Harland vary, and two of the three targets
# (Fig 3F-G's scale-dependent form, Fig 6E's CV against area) cannot be read
# on any other. Each no-landmark arena is compared with its lm8 twin.
#
# Takes the admitted field library per arena and channel and reports, for each:
#
#   * the best of log-normal, negative exponential and Gaussian, each fitted
#     within the Rule 8/9 size window, with goodness of fit for all three --
#     the two source papers disagree about the form, so fitting only a
#     favoured one would beg the question;
#   * coefficient of variation of field size (Harland Fig 6E: ~70/85/101);
#   * min, median, max and max/min ratio;
#   * scale occupancy, scales 0 (finest) to 5 (coarsest);
#   * fraction of the arena covered per field (Harland ~9-13%).
#
# Analysis only: no Webots, no collection. It reads the HDF5 datasets and
# rebuilds field libraries through the same rules the rest of the experiment
# series uses.
#
# THE THRESHOLD CAVEAT. Harland's exponential fit goes quasi-linear at a
# lower detection threshold, so every fit is reported across a sweep of our
# analogous knob. That knob is EXTENT_PCTL, *not* ACT_THRESH: under
# SIGMA_MODE=quantile the activation threshold cancels exactly from the mask
# boundary and sweeping it changes nothing (see RETIRED.md). The default
# settings list carries one off-threshold point as a standing invariance
# check, and the report states the measured difference rather than asserting
# the algebra.
#
# Usage:
#   sbatch slurm/scale_distribution.sh                        # all eight arenas
#   sbatch slurm/scale_distribution.sh --envs circ_lm8_r3     # one, in parallel
#   sbatch slurm/scale_distribution.sh --use-cache            # reuse banks
#   sbatch slurm/scale_distribution.sh --settings 50:0.5,65:0.5,80:0.5
#                                                             # re-open the sweep
# Rule 2 (--split-half-iou-min) is currently UNUSABLE and the run refuses it:
# at the lattice bin the two split-half maps occupy disjoint bins, so every
# IoU is exactly 0 and any threshold rejects the whole library. It needs the
# halves scored on a coarser grid first.
#
# One arena per job is the better pattern here, as for the other analysis
# jobs: cost is dominated by building a field library per channel, the arenas
# are independent, and eight of them serially is eight times the walltime for
# no benefit. Submit the fan-out with:
#
#   for e in circ_lm8_r3 circ_lm8_r6 circ_lm8_r10 corr_lm8_l10w2 \
#            corr_lm8_l10w10 circ_lm0_r3 circ_lm0_r6 corr_lm0_l10w10; do
#       sbatch slurm/scale_distribution.sh --envs "$e"
#   done
#
# then re-run once over all eight with --use-cache to get the cross-arena
# figures and the single combined report. The area trend, S2b, S3 and the
# landmark-pair comparison need their arenas in one run, so that combining
# pass is not optional. Let the fan-out finish first: every run writes the
# same summary files and figures, so a single-arena job that finishes after
# the combined pass overwrites it.
#
# THE THRESHOLD SWEEP IS NOT IN THE DEFAULT. EXTENT_PCTL saturates at 65
# (run_field_recovery, against ideal cells of known size), and the ACT_THRESH
# invariance check was exact to 0. The first full run also found log-normal
# winning at 50, 65 and 80 alike, but those fits ignored the Rule 8/9 size
# window and pick log-normal whatever the shape, so whether the shape holds
# across settings is still open. Re-open the sweep with --settings to check.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=scale-dist
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
# 24h was sized for six arenas run serially. The default is now eight, so do
# not rely on a bare `sbatch` building every library inside it: fan out one
# arena per job as above, and keep the all-arena run for the --use-cache
# combining pass, which reuses every library. A single --envs job finishes in
# a small fraction of 24h.
#
# GPU: the pipeline builds a full pairwise feature distance matrix per
# channel through rules.feature_sq_distances, which is what the card is for.
# Any GPU will do — the wide channels (visual 7.2 GB, all 7.3 GB) do not fit
# an 11 GB 1080 Ti alongside their working set, but feature_sq_distances
# checks free VRAM and falls back to the CPU for that stage, so a small card
# costs time rather than correctness. Pinning a GRES type queues behind those
# four nodes for more than it saves.
#
# The EXTENT_PCTL sweep is nearly free in GPU terms: the Gram matrix and the
# Ward tree depend on neither swept parameter, so they are computed once per
# channel and reused. Only prepare_candidates' sigma loop and the readout
# repeat per setting. That reuse is also what makes the sweep strictly
# like-for-like — every setting is scored against an identical tree.
#
# 128G because the pipeline holds the feature matrix, its pairwise block and
# the candidate responses at once. No container: this never touches Webots.
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
export REALM_LOG_PATH="slurm/logs/${SLURM_JOB_NAME:-scale-dist}-${JOB_ID}.out"

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
python analysis/experiment_channel_isolation/run_scale_distribution.py \
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
send_email(f'[REALM-VPCE] scale-distribution FAILED (exit {status}, job {job})',
           f'The run exited {status} before it could report.\n\n'
           f'Last 60 log lines:\n\n{tail}', attachments=[log])
PY
fi

exit ${STATUS}
