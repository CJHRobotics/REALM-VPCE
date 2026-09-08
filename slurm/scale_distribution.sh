#!/bin/bash
#
# Experiment 2 — where our scale distribution sits.
#
# What shape is our field-size distribution, how does it compare to the three
# forms in the literature, and how does it move with environment scale?
#
# The default env list is the AREA SWEEP -- circ_lm8_rad1p25 .. rad10p0, 4.91
# to 314.16 m^2, a 64x range at fixed shape and landmark count. That is the
# axis Harland vary, and two of the three targets (Fig 3F-G's scale-dependent
# form, Fig 6E's CV against area) cannot be read on any other. The six
# same-area datasets are the control: pass them with --envs.
#
# Takes the admitted field library per channel and reports, for each:
#
#   * fits against log-normal, negative exponential AND Gaussian, with
#     goodness of fit for all three — the two source papers disagree about
#     the form, so fitting only a favoured one would beg the question;
#   * coefficient of variation of field size (Harland Fig 6E: ~70/85/101);
#   * min, median, max and max/min ratio;
#   * scale-band occupancy, bands 0-5;
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
#   sbatch slurm/scale_distribution.sh                        # the area sweep
#   sbatch slurm/scale_distribution.sh --envs circ_lm8_r0     # one, in parallel
#   sbatch slurm/scale_distribution.sh --use-cache            # reuse banks
#   sbatch slurm/scale_distribution.sh --settings 50:0.5,65:0.5,80:0.5
#                                                             # re-open the sweep
#
# One arena per job is the better pattern here, as for the other analysis
# jobs: cost is dominated by building a field library per channel, the arenas
# are independent, and six of them serially is six times the walltime for no
# benefit. Submit the fan-out with:
#
#   for e in circ_lm8_rad1p25 circ_lm8_rad2p0 circ_lm8_r0 \
#            circ_lm8_rad3p5 circ_lm8_rad6p0 circ_lm8_rad10p0; do
#       sbatch slurm/scale_distribution.sh --envs "$e"
#   done
#
# then re-run once over all six with --use-cache to get the cross-environment
# figures and the single combined report. S2 needs every arena in one run to
# draw its axis, so that combining pass is not optional here.
#
# THE THRESHOLD SWEEP IS RETIRED FROM THE DEFAULT. EXTENT_PCTL saturates at 65
# (run_field_recovery, against ideal cells of known size) and the first full
# run found log-normal winning on AIC at 50, 65 and 80 alike across all 24
# libraries, with the ACT_THRESH invariance check exact to 0. Re-open either
# with --settings if something upstream changes.
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
# 24h for the all-six default, which is six times the per-arena cost. A
# single --envs job finishes in a small fraction of that; the wall clock is
# sized for the serial worst case so the default invocation cannot be killed
# mid-run.
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
