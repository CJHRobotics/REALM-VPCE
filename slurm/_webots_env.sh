# Shared setup for the Webots jobs. Source, do not execute.
#
# Resolves the Python that Webots will launch the controller with, makes it
# visible inside the Singularity image, and checks it actually works there
# before a world is loaded.
#
# The container supplies Webots, Xvfb and Mesa. It does NOT supply the Python
# packages -- those come from the existing `realm-vpce` conda environment,
# the same one every other job in this directory uses. An earlier version of
# these scripts built a second venv inside the container on the theory that
# wheels should link against container libraries; in practice that duplicated
# several GB (torch alone is ~3 GB), filled the home quota, and bought
# nothing that the preflight below does not verify directly.
#
# DEFINE EACH FUNCTION EXACTLY ONCE. This file carried three stacked copies of
# write_runtime_ini, gpu_args and gl_info, newest first, because two successive
# fixes were prepended instead of replacing the body they fixed. Bash keeps the
# LAST definition, so both fixes were dead code and the original -- the one
# they were written to correct -- was the one that actually ran, for weeks.

CONDA_ENV_NAME="${CONDA_ENV_NAME:-realm-vpce}"

resolve_python() {
    if [[ -n "${REALM_PY:-}" ]]; then
        PYTHON_BIN="$REALM_PY"
        return
    fi
    local base
    base="$(conda info --base 2>/dev/null || echo /apps/anaconda3)"
    # shellcheck disable=SC1091
    source "$base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    PYTHON_BIN="${CONDA_PREFIX:?conda activate $CONDA_ENV_NAME failed}/bin/python3"
}

# Singularity auto-mounts $HOME, $PWD and /tmp but nothing else, so a conda
# environment living under, say, /apps is invisible inside the image unless
# bound. Only ever bind a NON-system top-level directory: binding /usr or
# /lib mounts the host's system tree over the container's own and destroys
# it -- the visible symptom is singularity failing to open /bin/sh.
SYSTEM_DIRS=" /usr /bin /sbin /lib /lib64 /etc /var /proc /sys /dev / "

container_binds() {
    BINDS=()
    local top
    top="/$(printf '%s' "${PYTHON_BIN#/}" | cut -d/ -f1)"
    case "$PYTHON_BIN" in
        "$HOME"/*|/tmp/*|"$REPO_DIR"/*)
            return 0 ;;                     # auto-mounted already
    esac
    if [[ " $SYSTEM_DIRS " == *" $top "* ]]; then
        echo "ERROR: $PYTHON_BIN lives under the system directory $top." >&2
        echo "       Binding that into the image would overwrite the" >&2
        echo "       container's own system tree. Point REALM_PY at a conda" >&2
        echo "       environment instead." >&2
        return 1
    fi
    BINDS=(--bind "$top")
    return 0
}

# Fail here, clearly, rather than inside Webots. A missing interpreter or an
# unimportable binding surfaces from Webots as a controller that exits
# immediately with no useful message.
preflight() {
    echo "--- preflight: $PYTHON_BIN inside the image ---"
    "$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" "$SIF" "$PYTHON_BIN" - <<'PYCHK'
import sys
print('  python', sys.version.split()[0], 'at', sys.executable)
missing = []
for m in ('numpy', 'pandas', 'h5py', 'cv2', 'matplotlib', 'PIL', 'tqdm'):
    try:
        __import__(m)
    except Exception as e:
        missing.append(f'{m} ({e.__class__.__name__})')
try:
    import controller                      # Webots bindings, from PYTHONPATH
    print('  controller module OK')
except Exception as e:
    missing.append(f'controller ({e})')
if missing:
    print('  MISSING:', ', '.join(missing))
    sys.exit(1)
print('  all imports OK')
PYCHK
}

write_runtime_ini() {
    # Webots launches the controller itself and takes the interpreter from
    # here, so a stale path fails looking like a Webots fault. Rewritten every
    # run; the file is gitignored and machine specific.
    local d
    for d in simulation/controllers/*/; do
        printf '[python]\nCOMMAND = %s\n' "$PYTHON_BIN" > "$d/runtime.ini"
    done
    echo "runtime.ini -> $PYTHON_BIN"
}


# --- optional GPU passthrough --------------------------------------------
# `--nv` binds the host NVIDIA driver into the image. On its own that is not
# enough here for two reasons, and both are worth knowing before reading the
# result:
#
#   1. The image pins LIBGL_ALWAYS_SOFTWARE=1 and GALLIUM_DRIVER=llvmpipe at
#      build time, so GL stays software unless those are overridden.
#   2. Webots renders through GLX, and GLX acceleration comes from the X
#      server. Xvfb is a software X server with no NVIDIA GLX extension, so
#      the vendor dispatch is expected to land back on Mesa regardless.
#
# The honest expectation is therefore "no change, or a GL error". gl_info()
# prints the renderer actually in use, which settles it in one line rather
# than by inference from a timing.
gpu_args() {
    GPU_ARGS=()
    [[ "${USE_GPU:-0}" == "1" ]] || return 0
    GPU_ARGS=(--nv
              --env LIBGL_ALWAYS_SOFTWARE=0
              --env GALLIUM_DRIVER=
              --env __GLX_VENDOR_LIBRARY_NAME=nvidia)
    return 0
}

gl_info() {
    # Reports the renderer actually in use and, when hardware GL was asked for
    # but is not available, FALLS BACK to software rather than failing the job.
    #
    # This used to abort, on the reasoning that a silent fallback turns a
    # 12-minute arena into as much as ten hours. That reasoning is sound and
    # the guard was still wrong: it blocked collection entirely on a cluster
    # where software rendering demonstrably works, twice, for a GPU path that
    # is an optimisation rather than a requirement. A loud warning in the log
    # and in the mailed report carries the same information without costing a
    # day. Set STRICT_GPU=1 to abort instead.
    local out r
    echo "--- GL renderer actually in use ---"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    if [[ "${USE_GPU:-0}" == "1" ]]; then
        "$SINGULARITY" exec "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" "$SIF" \
            nvidia-smi -L 2>&1 | head -5 || echo "  (no nvidia-smi in image)"
    fi

    # One call, not two, and `-a` rather than `-n`. Two calls on the same
    # display number is the bug this function kept regressing to: `xvfb-run -n`
    # fails outright when the number is still held, so the second call returned
    # nothing and the guard reported a broken GL stack on a machine whose GL
    # was fine. `-a` probes for a free number instead, which also survives a
    # stale /tmp/.X<n>-lock left behind by a killed job. Webots itself keeps
    # its deterministic -n display, assigned in the job script.
    out="$("$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" \
        "$SIF" xvfb-run -a \
        -s "-screen 0 ${XVFB_SCREEN:-1280x1024x24}" glxinfo -B 2>&1 || true)"
    echo "$out" | grep -Ei 'vendor|renderer|version|error' | head -8 \
        || echo "  (no glxinfo output)"

    [[ "${USE_GPU:-0}" == "1" ]] || return 0

    r="$(echo "$out" | grep -i 'OpenGL renderer' || true)"
    if [[ -n "$r" && "$r" != *llvmpipe* && "$r" != *softpipe* && "$r" != *swrast* ]]; then
        echo "  hardware GL confirmed:$r"
        return 0
    fi

    echo "WARNING: hardware GL unavailable (${r:-glxinfo returned no renderer})." >&2
    echo "         Falling back to software rendering: expect roughly 2.5-10 h" >&2
    echo "         per arena instead of 0.2 h, depending on the node." >&2
    if [[ "${STRICT_GPU:-0}" == "1" ]]; then
        echo "         STRICT_GPU=1 set -- aborting instead." >&2
        return 1
    fi
    GPU_ARGS=()
    USE_GPU=0
    # The small screen is worth ~2.4x under llvmpipe on the slow nodes and
    # nothing on fast ones, so it is only applied on this path.
    XVFB_SCREEN="320x240x24"
    echo "         XVFB_SCREEN set to $XVFB_SCREEN for the software path." >&2
    return 0
}
