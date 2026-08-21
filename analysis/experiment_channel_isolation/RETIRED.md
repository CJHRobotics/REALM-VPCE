# Retired experiments

## `run_threshold_sweep.py` / `slurm/threshold_sweep.sh`

Removed 21 August 2026 (last present at `0149cd9`).

The experiment swept `ACT_THRESH`, the fraction of a field's own peak at which
its boundary is drawn, to ask two things: how much reported field sizes change
between our 0.50 and the 0.20 convention used in the recording literature, and
whether a lower threshold lowers the smallest field we can find.

**`ACT_THRESH` no longer affects anything.** Under `SIGMA_MODE = 'quantile'`,
sigma is solved so that the threshold contour lands at `Q`, the `EXTENT_PCTL`
percentile of a cluster's member-to-centroid distances:

    sigma = sqrt( (Q^2 - d_min^2) / (2 ln(1/T)) )

Substituting that back into the boundary gives

    sqrt( d_min^2 + 2 sigma^2 ln(1/T) )  =  sqrt( d_min^2 + Q^2 - d_min^2 )  =  Q

for any `T`. The threshold cancels exactly. Sweeping it now varies sigma while
leaving every mask, every field and every admission decision untouched, so the
experiment cannot produce a result.

The question it was asking has moved rather than gone away. Field extent is now
set by `EXTENT_PCTL`, and the equivalent sweep is over that, which
`run_field_recovery.py` already runs -- with the advantage that it scores each
setting against ideal place cells of known size rather than against each other.

Recover the scripts with:

    git show 0149cd9:analysis/experiment_channel_isolation/run_threshold_sweep.py
    git show 0149cd9:slurm/threshold_sweep.sh
