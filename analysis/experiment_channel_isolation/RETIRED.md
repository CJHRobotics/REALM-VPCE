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

---

## Scripts kept for the record, but not runnable

Removed 9 September 2026 from the active set: their SLURM wrappers are gone
and the arenas they need no longer exist, so they will fail on a missing world
file. They are kept because each documents a decision the current experiment
still depends on, and deleting them would leave those constants asserted
rather than justified.

| script | what it established | arenas it needs |
|---|---|---|
| `run_field_recovery.py` | `SIGMA_MODE = 'quantile'` and `EXTENT_PCTL = 65`, measured against ideal place cells of known size | `circ_lm8_r0` (the old r = 10 disc) |
| `run_pruning_sweep.py` | `SAME_SCALE_SEPARATION = 0.35`, the point at which the smallest admissible field size saturates | `circ_lm8_r0` |
| `run_landmark_null.py` | the landmark-independence null, Eliav's five tests — reported 7 September 2026 | `circ_lm2/4/8/12_r0` |
| `run_geometry_recovery.py`, `plot_geometry_recovery.py` | recovery across disc / rectangle / corridor at matched area | `rect_lm8_r0`, `corr_lm8_r0` |
| `run_channel_isolation.py`, `plots.py` | the per-channel × λ field banks the series was built on | `circ_lm8_r0` |
| `run_locality_test.py`, `check_channel_locality.py` | whether a channel carries location information at all | `circ_lm8_r0`, `circ_lm12_r0` |

To run any of them again, restore its arena and grid from history:

```bash
git show 37735d2:simulation/worlds/environments/vpce/circ_lm8_r0.xml > /tmp/circ_lm8_r0.xml
```

`37735d2` is the last commit in which the full environment set existed. Note
that the old `circ_lm8_r0` was the **r = 3** disc at that commit and the
**r = 10** disc before `fa1b965`; check which one a given result used before
comparing anything to it.
