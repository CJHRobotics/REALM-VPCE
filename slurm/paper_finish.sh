#!/bin/bash
#
# The last step of the paper rerun: every number and every figure, from the
# caches the analysis jobs left, mailed in one go.
#
# Submitted by slurm/paper_rerun.sh once the analysis jobs are done (afterany:
# it runs even if one of them failed, so whatever did finish still reaches
# you, and a number that could not be computed says why in the CSV). It can
# also be submitted by hand to redo this step alone:
#
#   sbatch slurm/paper_finish.sh
#
# Steps, each allowed to fail without stopping the next:
#   1. prune audit   join the per-arena parts into its two tables
#   2. validation    report from its per-arena caches (draws V1-V7)
#   3. numbers       paper_numbers.py -> figures/paper/paper_numbers.csv
#   4. figures       paper_figures.py -> figures/paper/<paper names>
#   5. appendix      V3 and V4 copied in under the paper's names
#   6. mail          the CSV, the log and every figure, split over as many
#                    messages as the attachment budget needs
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=paper_finish
#SBATCH --partition=general
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# No GPU: nothing is built here. The validation report's bootstraps are the
# slow part (1000 draws over every hand-crafted cluster), hence 6 h.
# --------------------------------------------------------------------------

set -o pipefail

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

A=analysis/experiment_channel_isolation
OUT=$A/figures/paper
JOB_ID="${SLURM_JOB_ID:-local}"
LOG="slurm/logs/${SLURM_JOB_NAME:-paper_finish}-${JOB_ID}.out"

echo "===================================================================="
echo "Job     : ${JOB_ID}   node $(hostname)"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

FAILED=()
step() {                       # step NAME COMMAND... -- run, record, go on
    local name=$1; shift
    echo; echo "---------------------------------------------------------- $name"
    "$@"
    local st=$?
    echo "---- $name: exit $st"
    [[ $st -ne 0 ]] && FAILED+=("$name")
    return 0
}

rm -f "$OUT"/*.png "$OUT"/*.pdf "$OUT"/*.csv
mkdir -p "$OUT"

step "prune audit" python $A/run_prune_audit.py --merge-parts --no-email
step "validation"  python $A/run_extent_validation.py --report-only --no-email
step "numbers"     python $A/paper_numbers.py --out "$OUT/paper_numbers.csv"
step "figures"     python $A/paper_figures.py --out "$OUT"

copy_in() {                    # the appendix's validation figures, renamed
    local ok=0
    for pair in V3_robustness:valid_V3_robustness V4_gallery_all:valid_V4_gallery_all; do
        local src=${pair%%:*} dst=${pair##*:}
        for ext in png pdf; do
            if [[ -f "$A/figures/extent_validation/$src.$ext" ]]; then
                cp "$A/figures/extent_validation/$src.$ext" "$OUT/$dst.$ext"
            else
                echo "  missing $A/figures/extent_validation/$src.$ext"; ok=1
            fi
        done
    done
    return $ok
}
step "appendix" copy_in

STATUS=0
[[ ${#FAILED[@]} -gt 0 ]] && STATUS=1
echo
echo "Finished : $(date -Is)"
echo "${FAILED[*]:+FAILED: ${FAILED[*]}}"
echo "numbers  -> $OUT/paper_numbers.csv"
echo "figures  -> $OUT"

if [[ -n "${EMAIL_TO:-}" ]]; then
    python $A/paper_figures.py --out "$OUT" --mail "${JOB_ID}" "${STATUS}" "${LOG}" \
        || echo "(mailer failed - job status unchanged)"
fi
exit "${STATUS}"
