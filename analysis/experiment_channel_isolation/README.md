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

So the default env list is `AREA_ENVS` — small, medium, mega at fixed shape
and landmark count, each sampled at ~`N_TARGET` positions so sample count is
not a covariate:

| arena | r | area | span | wall cover |
|---|---|---|---|---|
| `circ_lm8_rad2p0` | 2 m | 12.57 m² | small | 48% |
| `circ_lm8_r0` | 3 m | 28.27 m² | medium | 32% |
| `circ_lm8_rad6p0` | 6 m | 113.10 m² | mega | 16% |

**9.0×, matching Harland's 8.8×.** We match their *ratio*, not their absolute
areas, and cannot do otherwise: their megaspace is 18.6 m² — smaller than our
medium arena — and matching it absolutely would put the small environment at
~2.1 m², a disc of radius 0.82 m. The robot's circumscribing radius is 0.31 m
and the keep-out 0.2 m, so it would barely fit, and eight 0.75 m panels need
6 m of wall against a 5.2 m circumference. The agent is larger relative to its
arena than a rat is to a room, so ordering and ratio are what transfer.

`circ_lm8_rad1p25`, `rad3p5` and `rad10p0` are still built and available via
`--envs` for a wider span; `rad1p25` is the weakest, with panels covering 76%
of its circumference.

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
| `band_summary.csv` | one row per env × channel × scale band: field count, share of the library, median area and coverage, tiling multiple, CV, median split-half IoU |
| `scale_trends.csv` | one row per tracked quantity: value at small and mega, mega/small ratio, pooled Spearman against area, the expected direction and its source, and whether ours agrees |
| `threshold_invariance.csv` | the `ACT_THRESH` check, per paired run |
| `<env>/<channel>_p<P>_t<T>_bank.csv` | the field library behind each row |

Fits are run on **area** (Harland's unit) and on **equivalent diameter** (the
closest thing we have to Eliav's 1D field width).

Figures — `figures/scale_distribution/`

Every figure but S1 is indexed by arena area, so the experiment reads as
"what changes as scale changes".

| figure | shows |
|--------|-------|
| S1 | size histogram per arena × channel, arenas in scale order, all three fits drawn |
| S2 | admitted fields drawn on the arena as their Rule 7 ellipses, coloured by scale band, each panel to its own arena with a 1 m bar |
| S3 | median field size, max/min spread and bands occupied, against area |

**CV is measured but not plotted, and not compared to Fig 6E.** Pooled across
bands it describes a six-band mixture spanning two orders of magnitude, and
its trend across area tracks how many bands are occupied rather than any field
size. Within a band it is fixed by the band definition — bands are geometric
in radius at ratio 1.6, so areas span 2.56× and a uniform spread gives
CV ≈ 25%, which is what we measure (23–32%). Harland's 70–101 sits between the
two. Neither is comparable until there is a model of how a recording samples
cells from this library. The numbers are in `band_summary.csv`.

The three-form fits are likewise kept in `fits.csv` and the report rather than
plotted against area: the pooled distribution they fit is the tiling spectrum
(N(>s) ∝ s⁻¹·¹), not a recorded population, so "which form wins" inherits the
same non-comparability. S1 shows the raw distribution, which is the honest
version of that picture.

Scale-band occupancy is in `summary.csv` (`band0_frac` … `band6plus_frac`) and
per-band detail in `band_summary.csv`; S3 plots only the count of occupied
bands. No log axes anywhere — every panel is linear and zero-based, and the
report attaches only the figures the run actually wrote.

### Read the bands, not the pool

A field library is a **tiling at every scale**, not a sample of cells. A
tiling at scale *s* needs ~arena/*s* tiles, so the finest band necessarily
holds most of the library and necessarily sets any pooled median, mean or CV.
Measured: band 0 is 61–65% of every channel's library, and the pooled median
coverage (0.26%) is just band 0's.

Per band the picture is different — bands 4 and 5 sit at ~8% and ~16% of the
arena, bracketing Harland's 9–13% per cell. **The model does reach their
scale; the pooled statistic hides it.** `band_summary.csv` and the report's
per-band table are the numbers comparable to a recorded sample.

Two things follow. Raising the Rule 8 floor is not the fix: the median lands
at about **2× the floor wherever the floor is put** (measured at 0.12%, 1%, 2%
and 4% of arena), so choosing the floor chooses the answer. And Rule 2
(split-half reliability) is the principled way to thin the fine end if you
want to — reliability rises monotonically with band (median IoU 0.45 → 0.69),
so a threshold removes fine fields for being unreproducible rather than for
being small, and approximates an experimenter's detection criterion:

```bash
sbatch slurm/scale_distribution.sh --split-half-iou-min none,0.4,0.5,0.6
```

**Rule 2 cannot be applied by filtering a finished bank**, so it does need the
banks rebuilt — but only the cheap stage. It sits upstream of Rule 11, whose
competition ordering is already tie-broken on reliability, and upstream of
Rule 12, which decides which bands survive on the coverage the survivors
reach. Remove a field before competition and a different one claims that
territory; remove enough and a whole band stops tiling.

So `--split-half-iou-min` takes a **comma list** and scores every threshold in
one job: the Gram matrix and Ward tree are built once per channel,
`prepare_candidates` once per setting, and only `admit_fields` re-runs per
threshold. The first entry is the one the figures and report are drawn at; the
rest lands in the CSVs, which carry a `split_half_iou_min` column throughout.

### What the model cannot answer

Harland give **four** quantities against area. Two are within-cell and this
model cannot produce them at all: subfields per cell (linear in area,
R² = 0.9776) and summed subfield area per cell (exponential, r = 0.996). A
single-centroid cluster owns exactly one field, so it has no subfield count
and no sum over subfields — those wait on multi-field place cells.

Of the two that transfer, coverage is still not quite like for like: theirs is
per **cell**, summed over that cell's subfields; ours is per **field**. The
same caution applies to the max/min ratio, where Eliav's 4.4 → 1.6 is
within-neuron and ours is across the population. `scale_trends.csv` records
the source of every expectation (`Harland`, `Eliav`, `ours`) so an
Eliav-derived direction is never scored as agreement with Harland.

## Compute

The Gram matrix and the Ward tree depend on neither swept parameter, so both
are computed **once per channel** and reused across every setting — the sweep
costs one `prepare_candidates` and one `admit_fields` per setting, not a full
rebuild. That reuse is also what makes the sweep strictly like-for-like: every
setting is scored against an identical tree and identical candidates.
