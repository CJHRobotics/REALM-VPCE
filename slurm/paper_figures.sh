#!/bin/bash
#
# Draw the paper's figures from the data cache, in one job.
#
# Runs analysis/experiment_channel_isolation/paper_figures.py, which draws each
# figure with its experiment's own plotting function from the tables a finished
# run left in data_cache/, and names the files as the manuscript includes them.
# Nothing is recomputed: no dataset, no GPU, minutes.
#
#   valid-q      fig:valid-q (Fig. 3)       valid_V1_what_q_does
#   funnel       fig:funnel (Fig. 4)        prune_P1_rule_shares
#   scale-maps   fig:scale-maps (Fig. 5)    scale_S2a_<env>, all eight arenas
#   size-dist    fig:size-dist (Fig. 6)     scale_S1_sizes_by_arena
#   outlines     fig:supp-outlines (Fig. 13) scale_S2b_field_outlines
#   size-fits    fig:supp-size-fits (Fig. 12) scale_S1_sizes_with_fits
#   elong-scale  fig:elong-scale (Fig. 7)   geom_G1_elongation_by_scale
#   elong-wall   fig:elong-wall (Fig. 8)    geom_G2_elongation_vs_wall
#   geom-rho     fig:geom-rho (Fig. 9)      geom_G4_correlation_summary
#   multifield   fig:multifield (Fig. 10)   multifield_M00_overview
#
# `multifield` draws every multifield unit -- two or more selected subfields
# -- from the scale run's libraries, all feature spaces pooled.
#
# `python paper_figures.py --list` prints the current set; figures are added
# there, and this script needs no change when one is.
#
# Check a layout locally first, on synthetic input, through the same code:
#
#   python analysis/experiment_channel_isolation/paper_figures.py --fake
#
# USAGE
#
#   sbatch slurm/paper_figures.sh                 # every figure
#   sbatch slurm/paper_figures.sh valid-q         # any subset, by name
#   sbatch slurm/paper_figures.sh --no-email      # figures only, no mail
#   bash   slurm/paper_figures.sh --dry-run       # print what would run
#
# OUTPUT
#
#   analysis/experiment_channel_isolation/figures/paper/<paper name>.png|.pdf
#
# With EMAIL_TO set, the PNGs and the log are mailed -- in as many messages
# as the attachment budget needs, the paper's figures first -- so they can
# go straight into vpce-paper/figures/. A figure whose cache is missing, or was
# built at a different q than rules.py holds now, is reported and skipped; the
# others are still drawn, and the job exits non-zero.
#
# ------------------------------------------------------------- SLURM header
#SBATCH --job-name=paper_figures
#SBATCH --partition=general
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chamilton4@usf.edu
#
# No GPU is requested, because none is used. If the partition refuses a job
# without one: sbatch --gres=gpu:1 slurm/paper_figures.sh
# --------------------------------------------------------------------------

set -euo pipefail

DRY=0
EMAIL=1
ARGS=()
for a in "$@"; do
    case "$a" in
        --dry-run)  DRY=1 ;;
        --no-email) EMAIL=0 ;;
        *)          ARGS+=("$a") ;;
    esac
done

SCRIPT=analysis/experiment_channel_isolation/paper_figures.py
OUT=analysis/experiment_channel_isolation/figures/paper

if [[ $DRY -eq 1 ]]; then
    echo "python $SCRIPT ${ARGS[*]+"${ARGS[*]}"}"
    exit 0
fi

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p slurm/logs

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-realm-vpce}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

JOB_ID="${SLURM_JOB_ID:-local}"
LOG="slurm/logs/${SLURM_JOB_NAME:-paper_figures}-${JOB_ID}.out"

echo "===================================================================="
echo "Job     : ${JOB_ID}   node $(hostname)"
echo "Figures : ${ARGS[*]:-(all)}"
echo "Email   : $([[ $EMAIL -eq 1 ]] && echo "${EMAIL_TO:-(EMAIL_TO unset - nothing will send)}" || echo off)"
echo "Started : $(date -Is)"
echo "Git     : $(git rev-parse --short HEAD 2>/dev/null || echo 'no git')"
echo "===================================================================="

# Last run's files are cleared first, so what is mailed and what is in the
# folder is exactly this run's.
rm -f "$OUT"/*.png "$OUT"/*.pdf

set +e
python "$SCRIPT" ${ARGS[@]+"${ARGS[@]}"}
STATUS=$?
set -e

echo
echo "Finished : $(date -Is)  (exit ${STATUS})"

if [[ $EMAIL -eq 1 && -n "${EMAIL_TO:-}" ]]; then
    # As many messages as the attachment budget needs: in one, the files
    # that did not fit were dropped, and Figure 13 was one of them.
    python "$SCRIPT" --out "$OUT" --mail "${JOB_ID}" "${STATUS}" "${LOG}" \
        || echo "(mailer failed - job status unchanged)"
fi
exit "${STATUS}"
