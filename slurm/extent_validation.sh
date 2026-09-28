#!/bin/bash
#
# Is q = 65 the right field extent? The validation behind EXTENT_PCTL.
#
# q sets where a field's edge is drawn: the boundary encloses q% of the group
# of positions the field was built from. It was chosen as 65 on 21 August 2026
# in the r = 10 disc, which no longer exists. This asks again, in the eight
# arenas the papers use now, and is written so the answer can come back "no".
#
# The model is handed place fields whose answer is known -- a disc of floor,
# one per scale, at 24 sites from the wall to the open floor -- and its own
# extent machinery draws each one at every q from 5 to 100. It is also handed
# groups that are not fields (scattered, shuffled, oversized, two lobes, a
# ring), which a good q refuses. Three criteria, fixed in the code before the
# run: accuracy (overlap with the true field), calibration (drawn area over
# true area) and discrimination (true fields admitted minus non-fields
# admitted). Each gets a best q, a 95% bootstrap interval and a range that does
# nearly as well, and the report says whether 65 is inside each.
#
# The pipeline stage then rebuilds every real library at q = 50, 65 and 80, so
# the report can say how much of what the papers report rests on the value.
#
# FIGURES (analysis/experiment_channel_isolation/figures/extent_validation/)
#
#   V1  what q does, on one representative field
#   V2  the three criteria against q -- the main result
#   V3  robustness: arena x channel, scale, wall distance
#   V4  what q = 65 draws in every arena, one figure per channel (one mailed)
#   V5  the libraries at q = 50, 65, 80
#   V6  discrimination, control by control
#
# Each as PNG (300 dpi) and PDF (editable text, 174 mm double-column width),
# and all of them in extent_validation_figures.pdf, which the mail attaches.
#
# USAGE
#
#   bash   slurm/extent_validation.sh --submit        # the usual way
#   bash   slurm/extent_validation.sh --submit circ_lm8_r3,circ_lm8_r6
#   sbatch slurm/extent_validation.sh                 # every arena in one job
#   sbatch slurm/extent_validation.sh --stages recovery   # skip the libraries
#
# --submit runs on the login node and only calls sbatch: one job per arena,
# each computing and caching without mailing, then one report job that waits
# for all of them (afterany, so a failed arena still gets a report on the
# rest) and sends the one email. Its job ids are printed.
#
# RE-SENDING THE REPORT
#
# Everything the report says is in data_cache/extent_validation/, so a wording
# or figure change needs no recompute -- seconds, no dataset read:
#
#   sbatch --mem=16G --time=1:00:00 slurm/extent_validation.sh --report-only
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=extent-val
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
# Cost per arena: the recovery sweep is minutes (one distance computation per
# group, reused for every q). The pipeline stage is the rest -- per channel one
# Gram matrix and Ward tree, then a readout and admission per q -- about three
# times Experiment 2's cost for one library. Every arena in one job should
# finish in a few hours; --submit returns them in the time the slowest takes.
#
# Memory and GPU as Experiment 2: the feature matrix, its pairwise block and
# the candidate responses are live at once. Any GPU; a small card falls back to
# the CPU for the widest channels and costs time, not correctness.
#
# The experiment mails its own report through
# realm_tools.experiment_lib.reporting; it needs EMAIL_TO exported.
# --------------------------------------------------------------------------

set -euo pipefail

SUBMIT_ENVS=circ_lm8_r3,circ_lm0_r3,circ_lm8_r6,circ_lm0_r6,corr_lm8_l10w10,corr_lm0_l10w10,corr_lm8_l10w2,corr_lm0_l10w2
if [[ "${1:-}" == "--submit" ]]; then
    # sbatch records the working directory as SLURM_SUBMIT_DIR, which the job
    # uses as the repo, so submit from the repo root wherever this is run.
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
    mkdir -p slurm/logs
    list="${2:-$SUBMIT_ENVS}"
    ids=()
    for e in ${list//,/ }; do
        id=$(sbatch --parsable --job-name="extent-val-$e" \
             slurm/extent_validation.sh --envs "$e" --no-report)
        echo "  $e -> job $id"
        ids+=("$id")
    done
    dep=$(IFS=:; echo "${ids[*]}")
    rid=$(sbatch --parsable --job-name=extent-val-report \
          --dependency="afterany:${dep}" --mem=16G --time=2:00:00 \
          slurm/extent_validation.sh --report-only)
    echo "  report -> job $rid (starts when all ${#ids[@]} finish, mails the result)"
    exit 0
fi

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
export REALM_LOG_PATH="slurm/logs/${SLURM_JOB_NAME:-extent-val}-${JOB_ID}.out"

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
python analysis/experiment_channel_isolation/run_extent_validation.py \
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
send_email(f'[REALM-VPCE] extent-validation FAILED (exit {status}, job {job})',
           f'The run exited {status} before it could report.\n\n'
           f'Last 60 log lines:\n\n{tail}', attachments=[log])
PY
fi

exit ${STATUS}
