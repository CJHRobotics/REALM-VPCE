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

---

## `run_wall_elongation.py` / `slurm/wall_elongation.sh`

Removed 18 September 2026 (last present at `887947b`).

Asked the same question Experiment 4 now asks — is a field more elongated near
a wall, and does it line up with the wall — but answered it with a null-model
apparatus: two arms, four nulls, a 150k-placement donor pool per library, a
vectorised reimplementation of Rule 7 to measure null placements, and a
synthetic calibration mode with a subtractable bias offset.

It worked. The calibration was not decoration: it caught the lattice's
inflation of the axis ratio being counted twice in the donor pool, a donor pool
offering the wrong size mixture near a wall, an empty near-wall bin in the
corridor from normalising by half-width, and an Arm A elongation level with no
null at all. Arm A came out clean on 6 of 6 no-effect tests and found a planted
effect in 6 of 6.

**It was the wrong instrument for the question.** Nobody had yet looked at what
the libraries' shapes do — the distributions of elongation by scale, of
elongation against wall distance, of orientation against wall proximity had
never been plotted or correlated. Building a null model to defend a result
before anyone had seen the result put 2100 lines and a three-job submission
sequence between the question and its answer. `run_field_geometry.py` replaces
it with a per-field table and Spearman correlations.

The one idea worth keeping came across intact, and cost one column rather than
an arm: Rule 7 fits the ellipse to the clipped mask, so a field reaching past
the wall has a cut shape on record and its elongation is partly the arena's
outline. Experiment 4 carries `reach_to_wall_m` and `crosses_wall` from the
ellipse's own support function and reports every correlation over all fields
and over the uncut ones. If the two agree the result needs no null; if they
disagree, that is the finding.

Recover the scripts with:

    git show 887947b:analysis/experiment_channel_isolation/run_wall_elongation.py
    git show 887947b:slurm/wall_elongation.sh

---

## `run_coverage_relaxation.py` / `slurm/coverage_relaxation.sh`

Removed 18 September 2026 (last present at `d72897d`). Never run.

Built to answer "what happens to the field distributions when the coverage
requirement is relaxed" and answered it with a bespoke comparison script: its
own sweep of `TILING_FRAC_MIN` over four values, a paired on/off table, a
superset check across thresholds, a self-check against Experiment 2's cached
banks, and four figures of its own.

The question did not need any of that. It was "run Experiment 2 with the
coverage requirement disabled", and Experiment 2 already computes every
statistic and draws every figure that answers it. What was missing was one
option. `run_scale_distribution.py` now takes `--tiling-frac-min`, which sends
its outputs to a parallel `..._tf<value>` cache, figure directory and bank
name so it cannot overwrite the operating point, and the report leads with a
section saying Rule 12 is not at its default.

    sbatch slurm/scale_distribution.sh --tiling-frac-min 0

Recover the scripts with:

    git show d72897d:analysis/experiment_channel_isolation/run_coverage_relaxation.py
    git show d72897d:slurm/coverage_relaxation.sh
