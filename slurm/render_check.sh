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
#   sbatch slurm/render_check.sh                 # circ_lm8_r0
#   sbatch slurm/render_check.sh rect_lm8_r0
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

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm/logs data_cache/render_check

SIF="${WEBOTS_SIF:-$REPO_DIR/webots_r2025a.sif}"
SINGULARITY=/apps/singularity/bin/singularity
VENV="${REALM_VENV:-$REPO_DIR/realm_venv_gaivi}"
WORLD="$REPO_DIR/simulation/worlds/render_check.wbt"

[[ -f "$SIF"   ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
[[ -f "$WORLD" ]] || { echo "ERROR: world not found: $WORLD" >&2; exit 1; }

echo "===================================================================="
echo "Job     : ${SLURM_JOB_ID:-local}   node $(hostname)"
echo "Maze    : ${MAZE}"
echo "Email   : ${EMAIL_TO:-(EMAIL_TO unset - files written, no send)}"
echo "Image   : $SIF"
echo "Venv    : $VENV"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

# --- venv, built inside the container -------------------------------------
# It lives on the host filesystem but is created by the container's Python,
# so every wheel links against the container's libraries rather than
# GAIVI's. Idempotent: only built the first time.
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "building venv at $VENV (first run only)"
    "$SINGULARITY" exec "$SIF" python3.11 -m venv "$VENV"
    "$SINGULARITY" exec "$SIF" "$VENV/bin/pip" install --upgrade pip
    "$SINGULARITY" exec "$SIF" "$VENV/bin/pip" install -r setup/requirements.txt
    # opencv needs libGL at import; the image has Mesa, but headless wheels
    # avoid pulling a second GUI stack into the venv.
    "$SINGULARITY" exec "$SIF" "$VENV/bin/pip" install --force-reinstall \
        "opencv-python-headless<=4.9.0.80"
    echo "$REPO_DIR" > "$VENV/lib/python3.11/site-packages/realm_repo.pth"
fi

# --- runtime.ini ----------------------------------------------------------
# Webots launches the controller itself and reads the interpreter from here,
# so it must name the container-side venv rather than whatever built the
# repo on a laptop. Rewritten every run: the file is gitignored and machine
# specific, and a stale one fails in a way that looks like a Webots problem.
for d in simulation/controllers/*/; do
    printf '[python]\nCOMMAND = %s/bin/python3\n' "$VENV" > "$d/runtime.ini"
done
echo "runtime.ini -> $VENV/bin/python3"

# --- run ------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export REALM_MAZE="$MAZE"

set +e
"$SINGULARITY" exec \
    --env REALM_MAZE="$MAZE" \
    --env EMAIL_TO="${EMAIL_TO:-}" --env EMAIL_FROM="${EMAIL_FROM:-}" \
    --env EMAIL_SMTP="${EMAIL_SMTP:-}" --env EMAIL_SMTP_PORT="${EMAIL_SMTP_PORT:-}" \
    --env EMAIL_SMTP_USER="${EMAIL_SMTP_USER:-}" --env EMAIL_SMTP_PASS="${EMAIL_SMTP_PASS:-}" \
    "$SIF" \
    xvfb-run -a webots --batch --stdout --stderr --mode=fast --minimize "$WORLD"
STATUS=$?
set -e

echo "Finished : $(date -Is)  (exit ${STATUS})"

echo "--- files produced ---"
ls -lh data_cache/render_check/ || echo "(nothing written - check the log above)"

exit ${STATUS}
