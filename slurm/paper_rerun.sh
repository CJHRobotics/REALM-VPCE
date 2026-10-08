#!/bin/bash
#
# Rerun every analysis the paper reports, under the subfield rule, and mail
# back the paper's numbers (one CSV row each) and its figures.
#
# RUN THIS WITH bash, NOT sbatch. It does no work itself: it only submits the
# jobs below, chained so each starts when what it reads is ready. Submitting
# is all that happens on the login node.
#
#   bash slurm/paper_rerun.sh            # submit the whole chain
#   bash slurm/paper_rerun.sh --dry-run  # print what would be submitted
#
# What changed (rules.FIELD_UNIT = 'subfield', 8 October 2026): a cluster's
# field is split into its subfields, and each subfield is a field of its own
# -- its own area, ellipse and scale, tested on size and competing at its
# scale -- with no contiguity rule. A unit with two or more selected
# subfields is multifield. Every library is rebuilt; libraries built the old
# way are named differently (no `_sub`) and are never read.
#
# The chain (every analysis job fans out one arena per job, as their own
# headers recommend):
#
#   scale distribution  8 x --envs E            libraries, sizes, fits, coverage
#                       then 1 x --use-cache    the cross-arena tables (afterok)
#   prune audit         8 x --envs E --part E   field-selection counts
#   extent validation   8 x --envs E --no-report  q validation, q = 50/65/80 sets
#   field geometry      1 x                     elongation (after the scale pass)
#   paper_finish.sh     1 x                     numbers, figures, mail (afterany)
#
# Every analysis job runs with --no-email; paper_finish.sh sends the one
# message (or as many as its attachments need). A job that fails still mails
# its failure through SLURM, and paper_finish runs anyway with what exists.
# ---------------------------------------------------------------------------

set -euo pipefail

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p slurm/logs

ENVS=(circ_lm8_r3 circ_lm8_r6 circ_lm0_r3 circ_lm0_r6
      corr_lm8_l10w2 corr_lm0_l10w2 corr_lm8_l10w10 corr_lm0_l10w10)

sub() {                        # sub [sbatch options...] -- script args...
    if [[ $DRY -eq 1 ]]; then
        echo "sbatch $*" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}
join() { local IFS=:; echo "$*"; }

# Old prune-audit parts would be merged with the new ones.
if [[ $DRY -eq 0 ]]; then
    rm -f data_cache/prune_audit/parts/*.csv
fi

SD=() PA=() EV=()
for e in "${ENVS[@]}"; do
    SD+=("$(sub slurm/scale_distribution.sh --envs "$e" --no-email)")
    PA+=("$(sub slurm/prune_audit.sh --envs "$e" --part "$e" --no-email)")
    EV+=("$(sub slurm/extent_validation.sh --envs "$e" --no-report --no-email)")
done
SDC=$(sub --dependency="afterok:$(join "${SD[@]}")" \
          slurm/scale_distribution.sh --use-cache --no-email)
FG=$(sub --dependency="afterok:$SDC" slurm/field_geometry.sh --no-email)
FIN=$(sub --dependency="afterany:$(join "$SDC" "$FG" "${PA[@]}" "${EV[@]}")" \
          slurm/paper_finish.sh)

cat <<EOF

Submitted:
  scale distribution  ${SD[*]}
                      combining pass $SDC
  prune audit         ${PA[*]}
  extent validation   ${EV[*]}
  field geometry      $FG
  paper_finish        $FIN   <- mails the numbers CSV and the figures

Watch with:  squeue -u \$USER
Results in:  analysis/experiment_channel_isolation/figures/paper/
EOF
