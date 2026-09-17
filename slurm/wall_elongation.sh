#!/bin/bash
#
# Experiment 4 — are place fields more elongated near a wall, and do they line
# up with it?
#
# Reads the field libraries Experiment 2 built, in all eight collected arenas
# and under the same place-field configuration, and asks two things of every
# field: how elongated it is, and which way it points relative to the nearest
# wall. Both against wall distance, per channel.
#
#   circ_lm8_r3       r = 3       28.27 m^2   disc
#   circ_lm8_r6       r = 6      113.10 m^2   disc
#   circ_lm0_r3       r = 3       28.27 m^2   disc, no panels
#   circ_lm0_r6       r = 6      113.10 m^2   disc, no panels
#   corr_lm8_l10w10   10 x 10 m  100.00 m^2   square
#   corr_lm0_l10w10   10 x 10 m  100.00 m^2   square, no panels
#   corr_lm8_l10w2    10 x 2 m    20.00 m^2   corridor
#   corr_lm0_l10w2    10 x 2 m    20.00 m^2   corridor, no panels
#
# Four discs against two squares and two corridors — the round-against-
# straight-walled contrast this experiment was asked for. circ_lm8_r10 is in
# Experiment 2's arena list but was never collected, so it is not here.
#
# Same configuration means exactly Experiment 2's: EXTENT_PCTL 65, ACT_THRESH
# 0.5, Rule 2 off, LAMBDA 0, seed 0. None of it is exposed as an option. The
# libraries are read from Experiment 2's cache (data_cache/scale_distribution)
# when they are there and built into it when they are not, with the same code
# and the same cache key, so all three experiments describe the same fields.
#
# READ THIS BEFORE READING ANY NUMBER IT PRODUCES
# -----------------------------------------------
# Rule 7 fits a field's ellipse to the second moments of its MASK, and the
# mask stops at the wall. A field reaching past the wall is cut, and a cut
# blob's moments are elongated ALONG the wall. So both of this experiment's
# measures come out positive near a wall in a pipeline with no anisotropy in
# it at all: a raw curve of elongation against wall distance is a picture of
# the arena's outline, not a result. Everything in the analysis is machinery
# for measuring the distance between the raw curve and what clipping alone
# predicts. Figures E1 and E4 show both curves side by side; start there.
#
# Two arms, for the two populations:
#
#   Arm A  only the fields whose recorded ellipse, grown by one bin, lies
#          wholly on the floor. Nothing clipped them, so nothing needs
#          correcting. The null keeps each shape rigid and moves it somewhere
#          it still fits whole — Experiment 3's placement null, uniform and
#          Rule 11 tiling. Unbiased in development; this is the arm to
#          believe. Its limit is who it can speak for: near the wall, only
#          small fields fit whole.
#   Arm B  every field, against a null that clips the way the data was
#          clipped. The library's own clear fields become donors, re-laid at
#          12 orientations, dropped at random, cut by the floor and
#          re-measured; each real field is matched to placements of the same
#          VISIBLE area at the same wall distance. Keeps the near-wall fields
#          Arm A has to drop, and carries a residual bias that the
#          calibration below measures.
#
# THE CALIBRATION IS NOT OPTIONAL FOR ARM B
# -----------------------------------------
# --synthetic replaces every library with fields of known shape — areas
# resampled from the real library, same arena, same channel, same count — and
# reruns the whole pipeline:
#
#   round    circular fields at random orientations. Every scrap of
#            elongation and alignment in that data is the wall cutting a
#            circle, so whatever the arms report is bias. This is the
#            false-positive check and the source of the correction.
#   planted  elongation rising toward the wall with the major axis on the
#            wall tangent. The effect is real and known, so what the arms
#            report is the power.
#
# In development, at a deliberately coarsened bin size and on resampled
# log-normal areas, Arm A came out clean on 6 of 6 round tests (p 0.27-0.81)
# and found the planted effect in 6 of 6. Arm B found the planted effect in
# 6 of 6 but kept a residual false positive in two places: near-minus-far
# elongation in the disc (p 0.0008) and alignment in the 2 m corridor
# (p 0.03). Those are development numbers on synthetic banks at a coarse
# grid, NOT measurements on this data — run the calibration here and read its
# own numbers.
#
# --calibration takes a round run's summary.csv and subtracts its measured
# bias from every statistic, re-deriving each p from the corrected value
# against the same null standard deviation. The raw values stay in summary.csv
# as *_excess_raw.
#
# Usage — the full sequence, in order:
#
#   # 1. the real run, uncorrected: builds or reuses every library
#   sbatch slurm/wall_elongation.sh
#
#   # 2. the two calibration runs (need the libraries, so run them after 1)
#   sbatch slurm/wall_elongation.sh --synthetic round   --no-email
#   sbatch slurm/wall_elongation.sh --synthetic planted --no-email
#
#   # 3. the run to read, with the bias subtracted
#   sbatch slurm/wall_elongation.sh \
#       --calibration data_cache/wall_elongation_synthetic_round/summary.csv
#
# Step 3 is the one that mails the report worth reading. Step 1 exists to get
# the libraries cached and to see the uncorrected numbers; if Experiment 2's
# cache is already populated you can run 1 and 2 concurrently.
#
# Other options:
#   --envs circ_lm8_r3      one arena (the shape contrast needs them together)
#   --channels hog,color    fewer channels
#   --n-null 200            a quick look; the default is 1000
#   --rebuild               ignore Experiment 2's cache
#
# Unlike Experiment 2 there is no reason to fan out one arena per job when the
# libraries are cached: no feature blocks are loaded, the GPU goes unused, and
# the shape contrast, E3 and E7 all need their arenas in one run. Fan out only
# if the libraries have to be built, then re-run once over all eight.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=wall-elong
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
# place this job needs neither, and a lighter request schedules sooner:
#
#   sbatch --mem=32G --time=6:00:00 --gres=gpu:0 slurm/wall_elongation.sh \
#       --calibration data_cache/wall_elongation_synthetic_round/summary.csv
#
# Cost with the cache in place is the null machinery, which is CPU and memory
# bound: per library, two rigid placement nulls of 1000 draws each (Arm A) and
# a pool of 150k clipped, re-measured placements (Arm B). Eight arenas x six
# channels. The pool is built donor by donor in chunks of a few million
# template x placement elements, so peak memory is modest and 32G is
# comfortable; the walltime is dominated by re-ranking a template per donor
# and orientation, which is worst in the arenas with the largest fields.
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
export REALM_LOG_PATH="slurm/logs/${SLURM_JOB_NAME:-wall-elong}-${JOB_ID}.out"

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
python analysis/experiment_channel_isolation/run_wall_elongation.py \
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
send_email(f'[REALM-VPCE] wall-elongation FAILED (exit {status}, job {job})',
           f'The run exited {status} before it could report.\n\n'
           f'Last 60 log lines:\n\n{tail}', attachments=[log])
PY
fi

exit ${STATUS}
