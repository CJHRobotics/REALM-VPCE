#!/bin/bash
#
# Smallest possible proof that Webots renders on GAIVI.
#
# Captures a camera image and a lidar scan at three poses, writes them to
# data_cache/render_check/, and emails the pictures. Run this before any
# collection job: a headless renderer returning blank frames writes datasets
# of exactly the right shape whose features are identical everywhere, and
# looking at one picture settles in a glance what no summary statistic does.
#
# Usage:
#   sbatch slurm/render_check.sh                              # baseline
#   sbatch slurm/render_check.sh circ_lm8_r0 --no-rendering   # test a flag
#
# Extra arguments go straight to Webots. The report includes a timing
# estimate measured on the arena's own collection grid, so two runs can be
# compared directly -- which is the only way to settle whether a flag that
# skips the main 3D view also skips the offscreen camera render. Watch the
# pixel std alongside the timing: a flag that makes capture instant by
# returning blank frames is not a speedup.
#
# Needs EMAIL_TO exported to receive the images; without it they are still
# written to data_cache/render_check/ and the send is a silent no-op. Set the
# defaults once in ~/.bashrc on GAIVI:
#   export EMAIL_TO=chamilton4@usf.edu EMAIL_FROM=chamilton4@usf.edu
#   export EMAIL_SMTP=smtp.usf.edu EMAIL_SMTP_PORT=587
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=rendchk
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
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

MAZE="${1:-circ_lm8_r0}"
shift || true
# Anything further is passed straight to Webots, so a flag can be tested
# without editing this script:
#   sbatch slurm/render_check.sh circ_lm8_r0 --no-rendering
WEBOTS_EXTRA=("$@")
# Tags the output files and the mail subject so two configurations can be
# compared rather than overwriting one another.
RUN_TAG="${RUN_TAG:-$(printf '%s' "${WEBOTS_EXTRA[*]:-baseline}" | tr -cd '[:alnum:]')}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm/logs data_cache/render_check

SIF="${WEBOTS_SIF:-$REPO_DIR/webots_r2025a.sif}"
SINGULARITY=/apps/singularity/bin/singularity

# shellcheck disable=SC1091
source "$REPO_DIR/slurm/_webots_env.sh"
resolve_python
container_binds || exit 1
WORLD="$REPO_DIR/simulation/worlds/render_check.wbt"

[[ -f "$SIF"   ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
[[ -f "$WORLD" ]] || { echo "ERROR: world not found: $WORLD" >&2; exit 1; }

echo "===================================================================="
echo "Job     : ${SLURM_JOB_ID:-local}   node $(hostname)"
echo "Maze    : ${MAZE}"
echo "Webots  : extra flags [${WEBOTS_EXTRA[*]:-none}]   run tag '${RUN_TAG}'"
echo "Email   : ${EMAIL_TO:-(EMAIL_TO unset - files written, no send)}"
echo "Image   : $SIF"
echo "Python  : $PYTHON_BIN"
echo "Binds   : ${BINDS[*]+${BINDS[*]}}${BINDS[0]:-(none needed - auto-mounted)}"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

preflight || { echo 'PREFLIGHT FAILED - not launching Webots' >&2; exit 1; }
write_runtime_ini

# --- run ------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export REALM_MAZE="$MAZE"
export REALM_RUN_TAG="$RUN_TAG"

# Webots does not exit when a controller dies -- it keeps the simulation
# running, so a crashed controller would sit here until the walltime expires.
# Cap it: the controllers call simulationQuit(0) on success, so hitting this
# timeout always means something went wrong, and the log above says what.
WEBOTS_TIMEOUT="${WEBOTS_TIMEOUT:-900}"

set +e
timeout --signal=TERM --kill-after=60 "$WEBOTS_TIMEOUT" \
"$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" \
    --env REALM_MAZE="$MAZE" --env REALM_RUN_TAG="$RUN_TAG" \
    --env REALM_TIMED_N="${REALM_TIMED_N:-20}" \
    --env EMAIL_TO="${EMAIL_TO:-}" --env EMAIL_FROM="${EMAIL_FROM:-}" \
    --env EMAIL_SMTP="${EMAIL_SMTP:-}" --env EMAIL_SMTP_PORT="${EMAIL_SMTP_PORT:-}" \
    --env EMAIL_SMTP_USER="${EMAIL_SMTP_USER:-}" --env EMAIL_SMTP_PASS="${EMAIL_SMTP_PASS:-}" \
    "$SIF" \
    xvfb-run -a webots --batch --stdout --stderr --mode=fast --minimize \
        "${WEBOTS_EXTRA[@]+"${WEBOTS_EXTRA[@]}"}" "$WORLD"
STATUS=$?
set -e
[[ ${STATUS} -eq 124 ]] && echo "TIMED OUT after ${WEBOTS_TIMEOUT}s -- Webots does not quit on a controller crash; check for a traceback above."

echo "Finished : $(date -Is)  (exit ${STATUS})"

echo "--- files produced ---"
ls -lh data_cache/render_check/ || echo "(nothing written - check the log above)"

exit ${STATUS}
