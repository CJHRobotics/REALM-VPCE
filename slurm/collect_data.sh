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
VENV="${REALM_VENV:-$REPO_DIR/realm_venv_gaivi}"
WORLD="$REPO_DIR/simulation/worlds/collect_data.wbt"

[[ -f "$SIF"   ]] || { echo "ERROR: image not found: $SIF" >&2; exit 1; }
[[ -f "$WORLD" ]] || { echo "ERROR: world not found: $WORLD" >&2; exit 1; }

echo "===================================================================="
echo "Job     : ${SLURM_JOB_ID:-local}   node $(hostname)"
echo "Mazes   : ${MAZES:-(controller default)}"
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
[[ -n "$MAZES" ]] && export REALM_MAZES="$MAZES"

set +e
"$SINGULARITY" exec --env REALM_MAZES="${REALM_MAZES:-}" "$SIF" \
    xvfb-run -a webots --batch --stdout --stderr --mode=fast --minimize "$WORLD"
STATUS=$?
set -e

echo "Finished : $(date -Is)  (exit ${STATUS})"
ls -lh data/vpce/collect_data/ || true

# Guard against a headless renderer that returns blank frames: the dataset
# would be the right size with every feature identical, and nothing else in
# the pipeline would notice.
if [[ ${STATUS} -eq 0 ]]; then
    "$SINGULARITY" exec "$SIF" "$VENV/bin/python3" slurm/check_dataset.py \
        || { echo "SANITY CHECK FAILED - see above"; exit 1; }
fi

exit ${STATUS}
