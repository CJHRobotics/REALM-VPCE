#!/bin/bash
#
# Webots data collection on GAIVI, headless, inside the Singularity image.
#
# Renders the POV images for one or more arenas and writes the feature
# datasets to data/vpce/collect_data/. This is the job the container was
# built for: no display, no GPU, Mesa llvmpipe behind Xvfb.
#
# Usage:
#   sbatch slurm/collect_data.sh                          # the maze_files list in the controller
#   sbatch slurm/collect_data.sh circ_lm8_rad2p0          # one arena
#   sbatch slurm/collect_data.sh rect_lm8_r0,corr_lm8_r0  # several, serially
#
# One arena per job is the better pattern: they are independent, they queue
# in parallel, and a single serial session over four arenas is four times the
# walltime for no benefit.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=collect
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# No --gres. The camera is 224x224 and the lidar 360x1, so llvmpipe carries
# the render and a GPU would sit unused -- see container/README.md. CPUs do
# matter: llvmpipe is threaded, and feature extraction runs a thread pool.
#
# 24 h is a guess for one arena of ~30k positions x 8 headings under software
# rendering; the Mac took ~2 h with a real GPU. Collection is resumable --
# an arena whose .h5 already exists is skipped -- so a timeout costs only the
# arena in flight.
# --------------------------------------------------------------------------

set -euo pipefail

MAZES="${1:-}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm/logs data/vpce/collect_data

SIF="${WEBOTS_SIF:-$REPO_DIR/webots_r2025a.sif}"
SINGULARITY=/apps/singularity/bin/singularity

# shellcheck disable=SC1091
source "$REPO_DIR/slurm/_webots_env.sh"
resolve_python
container_binds || exit 1
WORLD="$REPO_DIR/simulation/worlds/collect_data.wbt"

[[ -f "$SIF"   ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
[[ -f "$WORLD" ]] || { echo "ERROR: world not found: $WORLD" >&2; exit 1; }

echo "===================================================================="
echo "Job     : ${SLURM_JOB_ID:-local}   node $(hostname)"
echo "Mazes   : ${MAZES:-(controller default)}"
echo "Image   : $SIF"
echo "Python  : $PYTHON_BIN"
echo "Binds   : ${BINDS[*]+${BINDS[*]}}${BINDS[0]:-(none needed - auto-mounted)}"
T_START=$(date +%s)
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

preflight || { echo 'PREFLIGHT FAILED - not launching Webots' >&2; exit 1; }
write_runtime_ini

# --- run ------------------------------------------------------------------
export PYTHONUNBUFFERED=1
[[ -n "$MAZES" ]] && export REALM_MAZES="$MAZES"

# Webots does not exit when a controller dies -- it keeps the simulation
# running, so a crashed controller would sit here until the walltime expires.
# Cap it: the controllers call simulationQuit(0) on success, so hitting this
# timeout always means something went wrong, and the log above says what.
WEBOTS_TIMEOUT="${WEBOTS_TIMEOUT:-84000}"

set +e
timeout --signal=TERM --kill-after=60 "$WEBOTS_TIMEOUT" \
"$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" --env REALM_MAZES="${REALM_MAZES:-}" "$SIF" \
    xvfb-run -a webots --batch --stdout --stderr --mode=fast --minimize "$WORLD"
STATUS=$?
set -e
[[ ${STATUS} -eq 124 ]] && echo "TIMED OUT after ${WEBOTS_TIMEOUT}s -- Webots does not quit on a controller crash; check for a traceback above."

echo "Finished : $(date -Is)  (exit ${STATUS})"
ls -lh data/vpce/collect_data/ || true

# Report what was actually collected, by mail. SLURM's own --mail-type=END
# says the job finished; it does not say whether the datasets are usable.
# This reports arena geometry, sample counts, feature blocks, and the
# blank-frame verdict -- the failure that otherwise stays silent, since a
# headless renderer returning blank frames writes a dataset of exactly the
# right shape that nothing downstream would question.
ELAPSED=$(( $(date +%s) - T_START ))

# Which arenas to report on: the explicit list if one was given, otherwise
# whatever datasets this run actually wrote or refreshed.
if [[ -n "$MAZES" ]]; then
    MAZE_LIST="${MAZES//,/ }"
else
    MAZE_LIST=$(find data/vpce/collect_data -name '*.h5' -newermt "@$T_START" \
        -exec basename {} .h5 \; | tr '\n' ' ')
fi

if [[ -n "${MAZE_LIST// /}" ]]; then
    # shellcheck disable=SC2086
    "$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" \
        --env EMAIL_TO="${EMAIL_TO:-}" --env EMAIL_FROM="${EMAIL_FROM:-}" \
        --env EMAIL_SMTP="${EMAIL_SMTP:-}" --env EMAIL_SMTP_PORT="${EMAIL_SMTP_PORT:-}" \
        --env EMAIL_SMTP_USER="${EMAIL_SMTP_USER:-}" --env EMAIL_SMTP_PASS="${EMAIL_SMTP_PASS:-}" \
        --env SLURM_JOB_ID="${SLURM_JOB_ID:-}" --env SLURMD_NODENAME="${SLURMD_NODENAME:-}" \
        "$SIF" "$PYTHON_BIN" \
        slurm/report_collection.py $MAZE_LIST --elapsed "$ELAPSED" \
        || { echo "REPORT FLAGGED A PROBLEM - see above"; exit 1; }
else
    echo "no datasets written or updated by this run"
fi

exit ${STATUS}
