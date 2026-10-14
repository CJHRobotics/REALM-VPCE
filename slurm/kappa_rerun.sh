#!/bin/bash
#
# Figure 3 and the kappa numbers on the model's own clusters.
#
# RUN THIS WITH bash, NOT sbatch: it only submits jobs.
#
#   bash slurm/kappa_rerun.sh            # submit
#   bash slurm/kappa_rerun.sh --dry-run  # print what would be submitted
#
# Eight extent-validation jobs, one per arena, run only the real-cluster
# stage (--stages members): candidate clusters of the real hierarchy, every
# kappa, each field scored against its own members. They write to
# data_cache/extent_validation/members/ and leave the hand-crafted cache, the
# libraries and everything else as they are -- kappa stays at 80, so nothing
# downstream changes.
#
# Then paper_finish.sh (afterany) recomputes the numbers CSV -- now with the
# kappa.* rows -- redraws every figure, the new Figure 3 and the pooled
# real-cluster figure (valid_V2_member_extent) among them, and mails them.
# ---------------------------------------------------------------------------

set -euo pipefail

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p slurm/logs

ENVS=(circ_lm8_r3 circ_lm8_r6 circ_lm0_r3 circ_lm0_r6
      corr_lm8_l10w2 corr_lm0_l10w2 corr_lm8_l10w10 corr_lm0_l10w10)

sub() {
    if [[ $DRY -eq 1 ]]; then
        echo "sbatch $*" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

EV=()
for e in "${ENVS[@]}"; do
    EV+=("$(sub slurm/extent_validation.sh --envs "$e" --stages members \
            --no-report --no-email)")
done
DEP=$(IFS=:; echo "${EV[*]}")
FIN=$(sub --dependency="afterany:$DEP" slurm/paper_finish.sh)

cat <<EOF

Submitted:
  real-cluster stage  ${EV[*]}
  paper_finish        $FIN   <- mails the numbers CSV and every figure

Watch with:  squeue -u \$USER
Results in:  analysis/experiment_channel_isolation/figures/paper/
EOF
