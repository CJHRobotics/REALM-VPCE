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
    echo "--- GL renderer actually in use ---"
    "$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" \
        "$SIF" xvfb-run -a -s "-screen 0 ${XVFB_SCREEN:-1280x1024x24}" \
        glxinfo -B 2>&1 | grep -Ei 'vendor|renderer|version|error' | head -8 \
        || echo "  (glxinfo failed)"
    if [[ "${USE_GPU:-0}" == "1" ]]; then
        "$SINGULARITY" exec "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" "$SIF" \
            nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
            2>&1 | head -3 || echo "  (no nvidia-smi in image)"
        # Fail rather than fall back. Hardware GL renders a step in 2.1 ms and
        # llvmpipe in 15-158 ms depending on the node, so a silent fallback
        # turns a 12-minute arena into anywhere up to ten hours -- and looks
        # like nothing but a slow run.
        local r
        r="$("$SINGULARITY" exec "${BINDS[@]+"${BINDS[@]}"}" "${GPU_ARGS[@]+"${GPU_ARGS[@]}"}" \
            "$SIF" xvfb-run -a -s "-screen 0 ${XVFB_SCREEN:-1280x1024x24}" \
            glxinfo -B 2>/dev/null | grep -i 'OpenGL renderer' || true)"
        if [[ "$r" == *llvmpipe* || "$r" == *softpipe* || "$r" == *swrast* ]]; then
            echo "ERROR: USE_GPU=1 but GL fell back to software: $r" >&2
            echo "       Set USE_GPU=0 to run on llvmpipe deliberately." >&2
            return 1
        fi
    fi
    return 0
}
