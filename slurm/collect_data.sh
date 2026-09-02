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
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# --gres=gpu:1 and USE_GPU=1 by default. Measured per position on
# circ_lm8_r0 (perf_probe):
#
#   32-core node, llvmpipe    1154 ms    9.7 h per arena
#   64-core node, llvmpipe     288 ms    2.4 h per arena
#   32-core node, GPU           22 ms    0.2 h per arena
#
# 75x on the same node class, and faster than the Mac that produced the
# original datasets. This contradicts the reasoning in container/README.md,
# which argued a bound GPU would go unused because Webots renders through
# GLX and Xvfb is a software X server. Empirically it does get used; the
# reasoning was wrong and the note has been corrected.
#
# Because the GPU makes node choice irrelevant, the request drops back to 16
# CPUs and any GPU. Do not pin a GPU type: per the slurm README that queues
# behind the named nodes for no gain here, since a 224x224 render asks
# almost nothing of the card.
#
# gl_info aborts the job if GL falls back to software, rather than letting a
# 12-minute arena silently become a ten-hour one.
#
# Memory: ~241k rows x 7,364 float32 features is ~7.1 GB held live, plus the
# image batch, hence 32G. Resumability is per arena, not within one --
# collect_data writes the .h5 only at the end -- so a timeout loses that
# arena's whole run.
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
# 1280x1024, the size hardware GL has actually been verified at. The 320x240
# screen was worth 2.4x under llvmpipe on the slow CPU nodes and nothing on
# the GPU path, and was never tested with hardware GL -- it was briefly
# blamed for a GLX failure that turned out to be a display-number collision.
# Set XVFB_SCREEN explicitly when running with USE_GPU=0.
XVFB_SCREEN="${XVFB_SCREEN:-1280x1024x24}"
echo "xvfb screen: $XVFB_SCREEN"
# Webots binds a control port, default 1234, and several of these land on the
# same node. It auto-increments on a clash, but that is a race between
# concurrent starts rather than a fix, so give each job its own.
WEBOTS_PORT=$(( 10000 + (${SLURM_JOB_ID:-0} % 20000) ))
# `xvfb-run -a` probes for a free display and races when several jobs start
# together on one node: two can pick the same number, and the loser fails
# with "BadValue (integer parameter out of range)" from GLX, which reads as a
# driver fault rather than a collision. Job ids are unique, so derive the
# display from one instead of probing.
XVFB_DISPLAY=$(( 100 + (${SLURM_JOB_ID:-0} % 400) ))
echo "webots port: $WEBOTS_PORT   xvfb display: :$XVFB_DISPLAY"

USE_GPU="${USE_GPU:-1}"
gpu_args
gl_info || exit 1
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
# llvmpipe caps its own worker count (32 in recent Mesa), and without this it
# picks a number we never see. Setting it explicitly makes the thread count a
# measured variable rather than an assumption -- and shows in the log whether
# a larger allocation is actually being used for rasterisation.
# Xvfb's default screen is 1280x1024 -- 1.3 million pixels, 26x the camera's
# 224x224. If Webots renders its main 3D view at screen size then that view,
# not the camera, is the bulk of every step under llvmpipe. Shrinking the
# virtual screen costs nothing and is one of two ways to find out; passing
# --no-rendering to Webots is the other.
export LP_NUM_THREADS="${LP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONUNBUFFERED=1
echo "llvmpipe threads: $LP_NUM_THREADS (of ${SLURM_CPUS_PER_TASK:-?} allocated CPUs)"
[[ -n "$MAZES" ]] && export REALM_MAZES="$MAZES"

# Webots does not exit when a controller dies -- it keeps the simulation
# running, so a crashed controller would sit here until the walltime expires.
# Cap it: the controllers call simulationQuit(0) on success, so hitting this
# timeout always means something went wrong, and the log above says what.
WEBOTS_TIMEOUT="${WEBOTS_TIMEOUT:-84000}"

set +e
timeout --signal=TERM --kill-after=60 "$WEBOTS_TIMEOUT" \
"$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" --env REALM_MAZES="${REALM_MAZES:-}" --env LP_NUM_THREADS="$LP_NUM_THREADS" \
    "$SIF" \
    xvfb-run -n "$XVFB_DISPLAY" -s "-screen 0 $XVFB_SCREEN" webots --batch --stdout --stderr --mode=fast --minimize --port="$WEBOTS_PORT" "$WORLD"
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
