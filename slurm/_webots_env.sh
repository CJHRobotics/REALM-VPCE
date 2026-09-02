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
# environment living under /apps is invisible inside the image unless bound.
container_binds() {
    local binds=()
    case "$PYTHON_BIN" in
        "$HOME"/*) ;;                       # already visible
        *) binds+=(--bind "$(echo "$PYTHON_BIN" | cut -d/ -f1-2)") ;;
    esac
    printf '%s\n' "${binds[@]:-}"
}

# Fail here, clearly, rather than inside Webots. A missing interpreter or an
# unimportable binding surfaces from Webots as a controller that exits
# immediately with no useful message.
preflight() {
    echo "--- preflight: $PYTHON_BIN inside the image ---"
    "$SINGULARITY" exec "${BINDS[@]}" "$SIF" "$PYTHON_BIN" - <<'PYCHK'
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
