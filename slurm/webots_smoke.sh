#!/bin/bash
#SBATCH --job-name=webots-smoke
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Does Webots render offscreen on a GAIVI compute node at all?
#
# Deliberately asks for NO GPU. The HamBot camera is 224x224 and the lidar
# is 360x1, so Mesa llvmpipe should carry them -- and CPU-only jobs
# schedule far faster than gpu ones. If this passes, the render path works
# and the remaining work is all repo-side.
#
#   sbatch slurm/webots_smoke.sh                     # uses the default below
#   sbatch slurm/webots_smoke.sh /path/to/other.sif   # or override it
#
# Escalation if it fails: read the .err first. A Qt/xcb "could not connect
# to display" means Xvfb never came up; a GLX/GL version error means
# llvmpipe is too old or LIBGL_ALWAYS_SOFTWARE did not take.

set -euo pipefail

SIF="${1:-/home/c/chamilton4/REALM-VPCE/webots_r2025a.sif}"
if [[ ! -f "$SIF" ]]; then
    echo "ERROR: image not found: $SIF" >&2
    exit 1
fi
echo "=== image: $SIF ==="
SINGULARITY=/apps/singularity/bin/singularity

echo "=== node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK:-?} ==="

# --- 1. Does the image run, and is Webots the version we pinned? ---
echo "=== [1/3] webots --version ==="
"$SINGULARITY" exec "$SIF" webots --version

# --- 2. Is there a working software GL stack behind Xvfb? ---
# glxinfo failing here localises the problem to rendering rather than to
# Webots, which is a much cheaper thing to debug.
echo "=== [2/3] offscreen GL ==="
"$SINGULARITY" exec "$SIF" xvfb-run -a glxinfo -B 2>&1 | \
    grep -Ei 'opengl (vendor|renderer|version)|error' || \
    echo "WARN: glxinfo produced no recognisable output"

# --- 3. Can Webots actually start, step a world, and exit cleanly? ---
# --batch suppresses the dialogs that would otherwise hang an unattended
# job until the walltime runs out; --mode=fast drops real-time sync.
echo "=== [3/3] headless Webots world load ==="
WORLD="${WEBOTS_WORLD:-/usr/local/webots/projects/samples/devices/worlds/camera.wbt}"
timeout 300 "$SINGULARITY" exec "$SIF" \
    xvfb-run -a webots --batch --stdout --stderr --mode=fast --minimize "$WORLD" \
    && echo "PASS: webots exited cleanly" \
    || echo "NOTE: exit $? -- a timeout here can be benign (world with no self-terminating controller); judge by the log above"

echo "=== smoke test done ==="
