# Experiment: channel isolation × spatial weighting

Builds a rule-governed place-field bank for every (feature channel, λ) pair
and asks two questions:

1. **Which sensory channel produces place-like fields?** Each of `hog`,
   `color`, `spatial` and `lidar` is run in isolation, alongside `visual`
   (the three camera channels) and `all`.
2. **How much does positional binding change the answer?** Rule 4's λ term
   is swept from 0 (features only) upward.

The agglomeration is governed by explicit, individually reportable rules —
see `Models/Agglomeration Rules.md` in the VPCE-Brain vault for the
biological case behind each one.

## Rules in force

| Rule | Name | Where |
|------|------|-------|
| 1  | contiguity — a field is one connected patch of floor | `rules.largest_component_fraction` |
| 2  | reliability — a field must reproduce across split halves | `rules.mask_iou` |
| 4  | spatial weighting — merge cost = feature distance + λ · space | `rules.ward_linkage` |
| 7  | anisotropy — fields have two axes and an orientation | `rules.field_shape` |
| 8  | size floor — nothing smaller than the smallest measured field | `RULE8_AREA_FRAC` |
| 9  | size ceiling — nothing larger than the largest measured field | `RULE9_AREA_FRAC` |
| 11 | competition — same-scale neighbours compete, nesting allowed | `rules.rule11_competition` |
| 12 | tiling stop — drop scale bands that cannot cover the floor | `rules.rule12_tiling` |

Rules 3, 5, 6 and 10 are not implemented.

**Distance to the wall is never an input to any rule.** It is computed and
reported only. The Paper 1 claim is that wall-dependent field size *emerges*
from the features, so putting wall distance into the model would assume the
result. Figure F7 is therefore a measurement, not a check that a constraint
was applied.

Scale bands (geometric, ratio 1.6) group fields for Rules 11 and 12 but
never admit or reject one — the ladder spacing is measured, not imposed,
because Rule 10 is out of force.

## Range-limited lidar

The agent's distance perception is bounded at 5.0 m. Any beam returning
beyond that, and any non-finite beam (the sensor's "no hit" code), is set to
a `-1` sentinel and flagged in a companion binary in-range channel. In
`circ_lm8_r0` this masks **62% of all beams** (53% beyond range, 9%
non-finite), so it is a substantial restriction rather than a formality.

The mask channel matters: the sentinel alone puts a discontinuity in the
feature space — a wall at 4.99 m and one at 5.01 m sit 6 m apart — and with
most beams out of range that jump would dominate every distance. The
companion channel makes "I cannot see that far" an explicit, comparable
signal.

## Running

```bash
python analysis/experiment_channel_isolation/run_channel_isolation.py circ_lm8_r0
```

On GAIVI:

```bash
sbatch slurm/channel_isolation.sh circ_lm8_r0
```

Options: `--channels`, `--lambdas`, `--subsample N`, `--bin-m`,
`--lidar-max-range`, `--no-gpu`, `--no-plots`.

## Outputs

`data_cache/channel_isolation/<env>/<channel>/lam<λ>/`

| file | contents |
|------|----------|
| `bank.csv` | one row per surviving field: area, equivalent radius, semi-major/minor axes, orientation, centroid, scale band, contiguity fraction, split-half IoU, wall distance |
| `bank_mu.npy` | feature-space centres, aligned to `bank.csv` |
| `report.json` | admission funnel, per-band coverage, size window, rule diagnostics |
| `diagnostics.npz` | per-candidate arrays behind the F4/F5/F6 distributions |

`data_cache/channel_isolation/<env>/metrics.csv` — one row per run.

## Figures

`figures/<env>/`

| figure | shows | rules |
|--------|-------|-------|
| F1 | field maps, channel × λ grid, fields drawn as ellipses | 4, 7 |
| F2 | field count, size, elongation, reliability vs λ | 4 |
| F3 | elongation vs wall distance; near-wall axis alignment | 7 |
| F4 | fragmentation distribution and rejection rate | 1 |
| F5 | split-half IoU distribution and pass rate | 2 |
| F6 | admission funnel, per-band coverage, size window | 8, 9, 11, 12 |
| F7 | field size vs wall distance | measurement only |

## Compute

Two steps dominate, both GPU-accelerated with automatic CPU fallback:

- the N × N feature Gram matrix (~5 × 10¹⁶ MACs for the 59632-d `all`
  configuration at N = 30147), and
- the environment readout, an (N × D) @ (D × n_cand) product per run.

Ward linkage is sequential and stays in scipy on the CPU. The feature
distance matrix is computed **once per channel** and reused across every λ,
since Rule 4 only adds a positional term — so the 24-run default costs 6
Gram computations, not 24.

Peak RAM at full N is roughly 20 GB; GPU memory roughly 11 GB for the widest
configuration.

## Note on subsampling

`--subsample` changes results, not just runtime. A node with a given member
count is a much tighter cluster when drawn from 30147 locations than from
6000, so small subsamples inflate field size. At `--subsample 6000` the
`hog` channel produces **zero** fields — every candidate exceeds the Rule 9
ceiling — while at 14000 it produces 28. Use the full dataset for any result
you intend to report.

---

# Experiment 2: where our scale distribution sits

`run_scale_distribution.py` — analysis only, no collection.

What shape is our field-size distribution, and how does it compare to the
three forms reported in the literature? Descriptive rather than decisive: it
establishes our baseline in Eliav's and Harland's own units and is a
prerequisite for reading Experiment 3.

## Targets

| source | claim |
|--------|-------|
| Eliav 2021 | field size log-normal |
| Harland Fig 3F–G | negative exponential in the megaspace (r = 0.995, 78% of fields ≤ 1 m²); Gaussian in the small environments (r = 0.985) |
| Harland Fig 6E | CV of field size rises with arena area, ~70 → 85 → 101 |
| Harland | one field covers ~9–13% of the arena |

The two papers report **different forms for the same quantity**, so all three
are fitted for every library and all three goodness-of-fit numbers are
reported side by side. Reporting only a favoured form is the one thing this
experiment cannot support.

## Reported per (environment × channel × setting)

- fits against log-normal, negative exponential and Gaussian
- CV of field area and of field radius
- min, median, max, max/min ratio
- scale-band occupancy, bands 0–5 (plus a 6+ overflow)
- fraction of the arena covered per field

Goodness of fit is three numbers, because Harland's statistic and a
model-selection statistic answer different questions:

| statistic | what it is | how to read it |
|-----------|-----------|----------------|
| `r_hist` | Pearson r between the binned density and the fitted pdf at bin centres | **Harland's own statistic** — the only one comparable to their 0.995 and 0.985. A weak discriminator: all three forms clear 0.9 on a heavy-fine-end sample, so never read it alone |
| `ks`, `ks_p_boot` | KS distance, p from a parametric bootstrap that refits each synthetic sample | the analytic p is anticonservative when parameters came from the same sample. Expect **every** form to be rejected past ~1000 fields; that is normal, not a failure |
| `aic`, `d_aic` | 2k − 2 logL | the discriminator. `winner` is its argmin |

## The threshold caveat — settled, no longer swept

Harland show their exponential fit becomes quasi-linear at a lower
field-detection threshold, so distribution shape is not threshold-independent
and the comparison is meaningless without checking ours. It has been checked:

- `EXTENT_PCTL` saturates at 65 — `run_field_recovery.py`, against ideal place
  cells of known size.
- The first full run of this experiment swept 50 / 65 / 80 and found
  log-normal winning on AIC at **every** setting, in all 24 environment ×
  channel libraries.
- The `ACT_THRESH` invariance check ran at 24 paired runs: identical field
  counts, maximum area difference exactly **0**.

The default is therefore the single operating point, `65:0.5`. Re-open either
check with `--settings 50:0.5,65:0.5,80:0.5` or `--settings 65:0.5,65:0.2` if
something upstream changes.

**Our analogous knob is `EXTENT_PCTL`, not `ACT_THRESH`.** Under
`SIGMA_MODE = 'quantile'` sigma is solved so the threshold contour lands at
`Q`, the `EXTENT_PCTL` percentile of a cluster's centroid-distance
distribution, and substituting it back into the mask boundary gives `Q` for
any `T` — the activation threshold cancels exactly. Sweeping `ACT_THRESH`
varies sigma and leaves every mask, field and admission decision untouched.
See `RETIRED.md`.

`ACT_THRESH` was never the knob and cannot be — the algebra above means
sweeping it varies sigma and leaves every mask, field and admission decision
untouched. `--settings 65:0.5,65:0.2` re-runs the invariance check, which
compares the two banks and reports the measured difference rather than
asserting the algebra.

## Two caveats the numbers cannot carry on their own

**Truncation.** Rules 8 and 9 bound field size by construction — floor at
Harland's smallest measured field, ceiling at 20% of arena area — so every fit
is to a doubly-truncated sample. `frac_at_floor` and `frac_at_ceiling` say how
much of the distribution is the rule rather than the model. Agreement at the
fine end is partly assumed rather than found.

**Scale is the primary axis.** Two of the three targets are claims about
environment *scale*: Fig 3F–G is a scale-dependent shape claim (exponential in
the megaspace, Gaussian in the small environments) and Fig 6E is CV against
enclosure area. Neither can be read from datasets that hold area constant.

So the default env list is `AREA_ENVS` — the area sweep, six discs from 4.91
to 314.16 m², a 64× range at fixed shape and fixed landmark count, each
sampled at ~`N_TARGET` positions so sample count is not a covariate.

`ENVS`, the six same-area datasets, is the **control**: it says whether cue
density (2/4/8/12 landmarks) or arena shape (disc/rectangle/corridor) move the
distribution, which is what licenses reading the area sweep as being about
area. Run it with `--envs "$(...)"` or by passing the names.

S2 picks its x axis from the data: arena area when the runs span at least
`AREA_SPAN_MIN` (2×), otherwise landmark count and aspect. The corridor is
28.224 m² against the discs' 28.274, so the test is a span ratio rather than
`nunique() > 1` — a 0.2% rounding difference must not be read as an area
axis.

## Running

```bash
sbatch slurm/scale_distribution.sh                     # the area sweep (default)
sbatch slurm/scale_distribution.sh --envs circ_lm8_r0  # one, in parallel
```

The control run, over the six same-area datasets:

```bash
sbatch slurm/scale_distribution.sh --envs circ_lm2_r0,circ_lm4_r0,circ_lm8_r0,circ_lm12_r0,rect_lm8_r0,corr_lm8_r0
```

Fan out one arena per job, then re-run over the whole list with `--use-cache`
for the cross-environment figures and the combined report. S2 needs every
arena in one run to draw its axis, so the combining pass is not optional when
the area sweep is the point.

Options: `--envs`, `--channels`, `--settings P:T,...`, `--lam`, `--subsample`,
`--n-boot`, `--use-cache`, `--no-gpu`, `--no-email`.

## Outputs

`data_cache/scale_distribution/`

| file | contents |
|------|----------|
| `summary.csv` | one row per env × channel × setting: CV, extremes, ratio, band occupancy, coverage, truncation |
| `fits.csv` | one row per env × channel × setting × variable × form: params, `r_hist`, `ks`, `ks_p_boot`, `aic`, `d_aic`, `winner` |
| `threshold_invariance.csv` | the `ACT_THRESH` check, per paired run |
| `<env>/<channel>_p<P>_t<T>_bank.csv` | the field library behind each row |

Fits are run on **area** (Harland's unit) and on **equivalent diameter** (the
closest thing we have to Eliav's 1D field width).

Figures — `figures/scale_distribution/`

| figure | shows |
|--------|-------|
| S1 | size histogram per env × channel with all three fits drawn |
| S2 | CV of field size — against arena area (Harland Fig 6E) when area varies, otherwise against landmark count and aspect |
| S3 | per-field arena coverage against the 9–13% band |

Scale-band occupancy is still written to `summary.csv` (`band0_frac` …
`band6plus_frac`) but no longer plotted: it was a coarsened S1 and harder to
read than the histogram it summarised. No log axes anywhere — every panel is
linear and zero-based.

## Compute

The Gram matrix and the Ward tree depend on neither swept parameter, so both
are computed **once per channel** and reused across every setting — the sweep
costs one `prepare_candidates` and one `admit_fields` per setting, not a full
rebuild. That reuse is also what makes the sweep strictly like-for-like: every
setting is scored against an identical tree and identical candidates.
