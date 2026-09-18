#!/bin/bash
#
# Experiment 4 — field shape against scale and wall proximity.
#
# Takes every field Experiment 2 admitted, in all eight collected arenas, and
# writes down four things about each: how elongated it is, which way it points,
# how far it sits from the nearest wall, and the angle between its long axis
# and that wall. Then correlates the pairs.
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
# circ_lm8_r10 is in Experiment 2's arena list but was never collected.
#
# Three questions, Spearman on each — scale is ordinal and elongation is
# heavy-tailed:
#
#   scale         vs elongation       do coarser fields come out longer?
#   wall distance vs elongation       are fields near a wall longer?
#   wall distance vs angle to wall    do fields near a wall point at it?
#
# Reported pooled, per arena, per arena x channel, and per arena x scale (which
# holds field size still). Descriptive: no null model, no permutation, no
# resampling. q is Benjamini-Hochberg across the rows of a table, which is not
# a null model — just the correction for asking the same question of every
# arena, channel and scale.
#
# Libraries are Experiment 2's, unchanged and from its cache under its cache
# key: EXTENT_PCTL 65, ACT_THRESH 0.5, Rule 2 off, LAMBDA 0, seed 0. None of it
# is exposed as an option. Only the position arrays are read from the HDF5
# datasets — never the feature blocks — so with the cache in place this is a
# small job.
#
# ONE THING TO READ BEFORE THE NUMBERS
# ------------------------------------
# Rule 7 fits a field's ellipse to the second moments of its MASK, and the mask
# is intersected with the floor. A field whose shape reaches past the wall is
# cut, and a cut blob's moments are elongated ALONG the wall — so both wall
# correlations are partly the arena's outline rather than the fields.
#
# So every correlation is reported twice: over all admitted fields, and over
# the fields whose recorded ellipse does not reach the wall (dist_to_wall_m
# below reach_to_wall_m, the ellipse's own support function toward it), whose
# shapes nothing cut. Where the two agree the result stands on its own. Where
# they disagree, the trend is in the cut fields, and that is what the
# correlation found.
#
# On synthetic test libraries with elongation planted against SCALE and nothing
# planted against the wall, this split did its job: wall distance vs elongation
# came out rho -0.064 at p = 3e-10 over all fields and rho -0.011 at p = 0.30
# over the uncut ones. The apparently significant wall effect was 4% of the
# fields, and the ones the wall had cut.
#
# Usage:
#   sbatch --mem=32G --time=2:00:00 slurm/field_geometry.sh
#   sbatch slurm/field_geometry.sh --envs circ_lm8_r3    # one arena
#   sbatch slurm/field_geometry.sh --channels hog,color  # fewer channels
#   sbatch slurm/field_geometry.sh --rebuild             # ignore the cache
#
# Run it over all eight in one job: the figures put the arenas side by side and
# the pooled correlations need them together. Fan out one arena per job only if
# the libraries have to be built, then re-run once over all eight.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=field-geom
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# The header is sized for the worst case, where every library has to be built:
# that is Experiment 2's cost and needs its GPU and memory. With the cache in
# place the job reads eight CSVs per arena, computes per-field geometry and
# runs a few hundred Spearmans — minutes, no GPU, and a much lighter request
# schedules sooner:
#
#   sbatch --mem=32G --time=2:00:00 slurm/field_geometry.sh
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
export REALM_LOG_PATH="slurm/logs/${SLURM_JOB_NAME:-field-geom}-${JOB_ID}.out"

echo "===================================================================="
echo "Job     : ${JOB_ID}   node $(hostname)"
echo "Extra   : ${EXTRA_ARGS[*]:-(none)}"
echo "Email   : ${EMAIL_TO:-(EMAIL_TO unset - report will not send)}"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

# `set -e` would abort before the report could be sent for a failing run --
# exactly the run worth hearing about. Capture the status by hand.
set +e
python analysis/experiment_channel_isolation/run_field_geometry.py \
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
send_email(f'[REALM-VPCE] field-geometry FAILED (exit {status}, job {job})',
           f'The run exited {status} before it could report.\n\n'
           f'Last 60 log lines:\n\n{tail}', attachments=[log])
PY
fi

exit ${STATUS}
