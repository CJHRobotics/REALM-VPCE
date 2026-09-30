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
| 12 | tiling stop — **measured, not enforced**: coverage per scale is reported, and admits nothing | `rules.rule12_tiling` |

**Three rules admit a field, and they are the whole of admission:**

| | rule | what it asks |
|---|---|---|
| **size range** | 8 and 9 | is the field neither smaller than the smallest nor larger than the largest measured field? |
| **contiguity** | 1 | is the field one connected patch of floor? |
| **competition** | 11 | had no larger field of the same scale already claimed the ground? |

Rules 2 and 12 are both **measured and reported but no longer filter**. Rule 2
(split-half reliability) was dropped after the first full run — it rejected at
most 1% of candidates — and its IoU is still written to every bank. Rule 12
(coverage) was dropped after the eight-arena review: it deleted whole scales
for covering too little of the floor, and the scales it deleted turned out to
be ones worth keeping and describing. `TILING_FRAC_MIN` defaults to **0**, and
coverage is still computed for every scale and returned in the report.

Setting `TILING_FRAC_MIN` above 0 restores the filter for a comparison run.
Every script that reads it carries the value in its output paths and bank
names, so a filtered run cannot be mistaken for the standard one.

Rules 3, 5, 6 and 10 are not implemented.

**Distance to the wall is never an input to any rule.** It is computed and
reported only. The Paper 1 claim is that wall-dependent field size *emerges*
from the features, so putting wall distance into the model would assume the
result. Figure F7 is therefore a measurement, not a check that a constraint
was applied.

Scale bands (geometric, ratio 1.6) group fields for Rule 11 and for reporting
coverage, but never admit or reject one — the ladder spacing is measured, not
imposed, because Rule 10 is out of force.

## Range-limited lidar

The agent's distance perception is bounded at 5.0 m. Any beam returning
beyond that, and any non-finite beam (the sensor's "no hit" code), is set to
a `-1` sentinel and flagged in a companion binary in-range channel. In
`circ_lm8_r3` this masks **62% of all beams** (53% beyond range, 9%
non-finite), so it is a substantial restriction rather than a formality.

The mask channel matters: the sentinel alone puts a discontinuity in the
feature space — a wall at 4.99 m and one at 5.01 m sit 6 m apart — and with
most beams out of range that jump would dominate every distance. The
companion channel makes "I cannot see that far" an explicit, comparable
signal.

## Running

```bash
python analysis/experiment_channel_isolation/run_channel_isolation.py circ_lm8_r3
```

On GAIVI:

```bash
sbatch slurm/channel_isolation.sh circ_lm8_r3
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
| F6 | admission funnel, per-band coverage, size window | 8, 9, 11 (+ 12 as measured) |
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
are fitted for every library, and the best fit is reported beside how well the
other two do. Reporting only a favoured form is the one thing this experiment
cannot support.

**Every fit knows the size window.** Rules 8 and 9 admit a field only between
the floor and the ceiling, so each form's density is renormalised to that
window before it is fitted — a truncated fit. Without it the comparison is
rigged: an exponential fitted from zero puts its peak exactly where the floor
has removed every field, and loses for it. Checked on synthetic libraries of
known shape, 1500 fields each, cut at the r = 3 window:

| true shape | untruncated fit (50 libraries) | truncated fit (30 libraries) |
|---|---|---|
| exponential | log-normal wins 50 | exponential wins 30 |
| log-normal | not run | log-normal wins 30 |
| Gaussian | Gaussian wins 50 | Gaussian wins 30 |

The Gaussian's mean is held inside the window. Unconstrained, a Gaussian
peaking far below the floor shows only its falling tail inside it — an
exponential under another name — and took 6 of 30 exponential libraries that
way, with a mean of −43 m².

## Reported per (environment × channel × setting)

- the best of log-normal, negative exponential and Gaussian, with fit quality for all three
- CV of field area and of field radius
- min, median, max, max/min ratio
- scale occupancy, scales 0–5 (plus a 6+ overflow)
- fraction of the arena covered per field

Goodness of fit is three numbers, because Harland's statistic and a
model-selection statistic answer different questions:

| statistic | what it is | how to read it |
|-----------|-----------|----------------|
| `r_hist` | Pearson r between the binned density and the fitted pdf at bin centres | **Harland's own statistic** — the only one comparable to their 0.995 and 0.985. A weak discriminator: all three forms clear 0.9 on a heavy-fine-end sample, so never read it alone |
| `ks`, `ks_p_boot` | KS distance from the truncated form, p from a parametric bootstrap that draws from and refits that same truncated form | the analytic p is anticonservative when parameters came from the same sample. Expect **every** form to be rejected past ~1000 fields; that is normal, not a failure |
| `aic`, `d_aic`, `aic_weight` | 2k − 2 logL, its gap to the best, and the Akaike weight | the discriminator. `winner` is the argmin; an `aic_weight` near 1 is a decisive win, near 1/3 means the three forms fit about equally |

## The threshold caveat — not swept by default

Harland show their exponential fit becomes quasi-linear at a lower
field-detection threshold, so distribution shape is not threshold-independent
and the comparison is meaningless without checking ours. What is known:

- `EXTENT_PCTL` is **80**, set by `run_extent_validation.py` against ideal
  place cells of known size in all eight arenas (see *Extent validation*
  below). It was 65 — chosen by `run_field_recovery.py` in the retired r = 10
  disc — until 29 September 2026. Library filenames carry the value (`_p80_`),
  so libraries built at 65 are never read back as current ones.
- The first full run of this experiment swept 50 / 65 / 80 and found
  log-normal winning on AIC at every setting, in all 24 environment × channel
  libraries. **Those fits ignored the size window** and would have picked
  log-normal whatever the true shape, so that result says nothing about
  whether the shape holds across settings. Re-open the sweep to check it.
- The `ACT_THRESH` invariance check ran at 24 paired runs: identical field
  counts, maximum area difference exactly **0**.

The default is therefore the single operating point, read from `rules.py`
(`80:0.5`). Every experiment that shares these libraries takes it from
`SETTINGS[0]` here, so changing `EXTENT_PCTL` in the rules moves them all. Re-open
either check with `--settings 65:0.5,80:0.5,95:0.5` or `--settings 80:0.5,80:0.2`
if something upstream changes.

**Our analogous knob is `EXTENT_PCTL`, not `ACT_THRESH`.** Under
`SIGMA_MODE = 'quantile'` sigma is solved so the threshold contour lands at
`Q`, the `EXTENT_PCTL` percentile of a cluster's centroid-distance
distribution, and substituting it back into the mask boundary gives `Q` for
any `T` — the activation threshold cancels exactly. Sweeping `ACT_THRESH`
varies sigma and leaves every mask, field and admission decision untouched.
See `RETIRED.md`.

`ACT_THRESH` was never the knob and cannot be — the algebra above means
sweeping it varies sigma and leaves every mask, field and admission decision
untouched. `--settings 80:0.5,80:0.2` re-runs the invariance check, which
compares the two banks and reports the measured difference rather than
asserting the algebra.

## Two caveats the numbers cannot carry on their own

**Truncation.** Rules 8 and 9 bound field size by construction — floor at
Harland's smallest measured field, ceiling at 20% of arena area. The fits
account for that window, but the window is still the rule speaking:
`frac_at_floor` and `frac_at_ceiling` say how much of the distribution rests on
a bound, and agreement at the fine end is partly assumed rather than found.

**Scale is the primary axis.** Two of the three targets are claims about
environment *scale*: Fig 3F–G is a scale-dependent shape claim (exponential in
the megaspace, Gaussian in the small environments) and Fig 6E is CV against
enclosure area. Neither can be read from datasets that hold area constant.

The default env list is all eight arenas, each sampled at ~`N_TARGET` positions
so sample count is not a covariate. Each has one **declared role**, and the
role alone decides which comparisons it enters:

| arena | size | area | role | cue cover |
|---|---|---|---|---|
| `circ_lm8_r3` | r = 3 m | 28.27 m² | area sweep, small | 32% |
| `circ_lm8_r6` | r = 6 m | 113.10 m² | area sweep, medium | 16% |
| `corr_lm8_l10w2` | 10 × 2 m | 20.00 m² | **Eliav corridor** | 25% |
| `corr_lm8_l10w10` | 10 × 10 m | 100.00 m² | square, two panels on each wall | 15% |
| `circ_lm0_r3` | r = 3 m | 28.27 m² | no landmarks, twin of `circ_lm8_r3` | — |
| `circ_lm0_r6` | r = 6 m | 113.10 m² | no landmarks, twin of `circ_lm8_r6` | — |
| `corr_lm0_l10w10` | 10 × 10 m | 100.00 m² | no landmarks, twin of `corr_lm8_l10w10` | — |
| `corr_lm0_l10w2` | 10 × 2 m | 20.00 m² | no landmarks, twin of `corr_lm8_l10w2` | — |

Only the area sweep enters the trend against area, and only the Eliav corridor
is scored on field length. Roles are declared in `ROLES` rather than read off
aspect ratio, which would have put the square (aspect 1) on the disc trend and
into the corridor comparison. Each no-landmark arena has exactly its twin's
walls and position grid, so the report's landmark-pair section isolates what
the landmarks do.

All eight are generated by
`simulation/worlds/environments/vpce/make_envs.py`. Names are
`<shape>_lm<landmarks>_<size>`, where size is `r<radius>` for a disc and
`l<length>w<width>` for a box.

**11.1× span, against Harland's 8.8×.** We match their *ratio*, not their
absolute areas, and cannot do otherwise: their megaspace is 18.6 m², smaller
than our small arena. The agent is larger relative to its arena than a rat is
to a room, so ordering and ratio are what transfer.

**The corridor is the Eliav comparison, not an area control.** Eliav report
field sizes along a 200 m tunnel and, more usefully, along a 6 m segment of
it, where mean field size fell from 5.9 m to 1.5 m and the within-neuron size
ratio from 4.4 to 1.6 — their evidence that a spread of scales belongs to a
large space rather than to the hippocampus. A 10 × 2 m corridor is the nearest
this series gets to that short segment. It is deliberately *not* area-matched;
that is the rectangle's job.

Because Eliav measure a one-dimensional width rather than an area, the
corridor is scored on **field length along the long axis** — the projection of
each field's ellipse onto that axis — and written to `eliav_lengths.csv`.
Circular arenas have no long axis and are skipped.

**Cue salience is confounded with area** across the sweep, and worst at the
top: a fixed 0.75 m panel subtends fewer pixels the larger the arena gets. The
colour channel once collapsed to single-digit field counts in a disc larger
than these; it did not at r = 6 (511 fields). With the sweep now spanning only
r = 3 to r = 6 the confound is milder than it was, and colour is still the
channel to check first in any new run.

**The no-landmark twins are the landmark control.** A difference between an
lm8 arena and its lm0 twin — in field count, median size, scales occupied or
best-fitting form — can only come from the panels, since walls and positions
are identical. The report tabulates each pair per channel.

## Running

Fan out one arena per job:

```bash
for e in circ_lm8_r3 circ_lm8_r6 corr_lm8_l10w2 corr_lm8_l10w10 circ_lm0_r3 circ_lm0_r6 corr_lm0_l10w10 corr_lm0_l10w2; do sbatch slurm/scale_distribution.sh --envs "$e"; done
```

When all eight have finished, run once over every arena, reusing their
libraries, for the combined figures and report:

```bash
sbatch slurm/scale_distribution.sh --use-cache
```

The combining pass is not optional: the area trend, S2b, S3 and the
landmark-pair comparison need their arenas in one run. Let the fan-out finish
first — every run writes the same summary files and figures, so a single-arena
job that finishes after the combining pass overwrites its output.

Options: `--envs`, `--channels`, `--settings P:T,...`, `--lam`, `--subsample`,
`--n-boot`, `--use-cache`, `--no-gpu`, `--no-email`.

## Outputs

`data_cache/scale_distribution/`

| file | contents |
|------|----------|
| `summary.csv` | one row per env × channel × setting: role, CV, extremes, ratio, scale occupancy (`scale0_frac` … `scale6plus_frac`, `n_scales_occupied`), coverage, truncation |
| `fits.csv` | one row per env × channel × setting × variable × form: params, the window it was truncated to (`trunc_lo`, `trunc_hi`), `r_hist`, `ks`, `ks_p_boot`, `aic`, `d_aic`, `aic_weight`, `winner` |
| `scale_summary.csv` | one row per env × channel × scale: field count, share of the library, median area and coverage, tiling multiple, CV, median split-half IoU |
| `redundancy.csv` | one row per env × channel, **scales combined**: field count, total field area, tiling multiple, fraction of floor covered, **R = total area / covered area**, the same ratio within a scale, max and median depth, and the fraction of floor at depth 0 / 1 / 2 / 3+ |
| `scale_trends.csv` | one row per tracked quantity, over the area sweep only: value at small and mega, mega/small ratio, pooled Spearman against area, the expected direction and its source, and whether ours agrees |
| `threshold_invariance.csv` | the `ACT_THRESH` check, per paired run |
| `<env>/<channel>_p<P>_t<T>_bank.csv` | the field library behind each row; its `scale_band` column is the scale |

Fits are run on **area** (Harland's unit) and on **equivalent diameter** (the
closest thing we have to Eliav's 1D field width).

Figures — `figures/scale_distribution/`

| figure | shows |
|--------|-------|
| S1 | `S1_sizes_by_arena` (main text): one panel per arena, all six channels overlaid as outline histograms, no fits — the shared shape, many small fields and few large; journal width, PNG and PDF. `S1_sizes_with_fits` (appendix): a panel per arena × channel on linear axes, each stopped at its own 95th percentile (the count beyond is printed, and the tail stays in the fit), all three truncated fits drawn with the best one heavier |
| S2a | one figure per arena, `S2a_scales_<env>.png`: channels on the rows, scales 0–5 on the columns, each column headed with its radius range in metres |
| S2b | every field as a near-transparent fill under a strong outline, both coloured by the field's own radius on one logarithmic ramp shared by every panel — green for small, through teal and blue, to deep purple for large — with a colourbar in metres; line width still grows with scale and coarse fields are drawn on top; arena on the row, channel on the column, each arena filling its own panel with a scale bar |
| S3 | median field size, max/min spread and scales occupied, against area; only the area sweep is joined |

**CV is measured but not plotted, and not compared to Fig 6E.** Pooled across
scales it describes a six-scale mixture spanning two orders of magnitude, and
its trend across area tracks how many scales are occupied rather than any
field size. Within a scale it is fixed by the scale definition — scales are
geometric in radius at ratio 1.6, so areas span 2.56× and a uniform spread
gives CV ≈ 25%, which is what we measure (23–32%). Harland's 70–101 sits
between the two. Neither is comparable until there is a model of how a
recording samples cells from this library. The numbers are in
`scale_summary.csv`.

**The best-fitting form is reported for comparison with both papers.** The
report sets it against Harland's megaspace (exponential, read at our largest
disc, r = 6), Harland's small environments (Gaussian, at the r = 3 disc) and
Eliav (log-normal). Note that our span is 4.0× where Harland's is 8.8×, so
their trend is being read over less than half the range they measured it on. Carry one caveat into that comparison: the pooled distribution
being fitted is the tiling spectrum
(N(>s) ∝ s⁻¹·¹), enumerated from a hierarchy rather than recorded from a sample of
cells, so a matching form is a shared shape, not evidence of a shared process.

Scale occupancy is in `summary.csv` (`scale0_frac` … `scale6plus_frac`) and
per-scale detail in `scale_summary.csv`; S2a draws each scale on its own and S3
plots the count of occupied scales. No log axes anywhere — every panel is
linear, and the report attaches only the figures the run actually wrote.

### Read the scales, not the pool

A field library is a **tiling at every scale**, not a sample of cells. A
tiling at scale *s* needs ~arena/*s* tiles, so the finest scale necessarily
holds most of the library and necessarily sets any pooled median, mean or CV.
Measured: scale 0 is 61–65% of every channel's library, and the pooled median
coverage (0.26%) is just scale 0's.

Per scale the picture is different — scales 4 and 5 sit at ~8% and ~16% of
the arena, bracketing Harland's 9–13% per cell. **The model does reach their
size; the pooled statistic hides it.** `scale_summary.csv` and the report's
per-scale table are the numbers comparable to a recorded sample.

Two things follow. Raising the Rule 8 floor is not the fix: the median lands
at about **2× the floor wherever the floor is put** (measured at 0.12%, 1%, 2%
and 4% of arena), so choosing the floor chooses the answer. **Rule 2 (split-half reliability) is currently unusable**, and the run refuses
it rather than returning an empty library. At the lattice bin each bin holds
one sample, so the two half-maps occupy disjoint bins, every split-half IoU is
exactly 0, and any threshold rejects everything. It was informative under the
old 0.25 m binning — median IoU rose from 0.45 at scale 0 to 0.69 at scale 5,
which is what made it the principled way to thin the fine end — so the measure
is sound and only its resolution is wrong. Restoring it means scoring the
halves on a deliberately coarser grid than the one used for field extent.

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

---

# Prune audit: which rule empties a library

`run_prune_audit.py` — analysis only, no collection.

Experiment 2 reports what survived. This reports what did not, and at which
rule. It exists because several libraries collapse in ways a field count
cannot explain:

| library | fields | the same channel elsewhere |
|---|---|---|
| `corr_lm0_l10w2` hog, spatial, visual | 0, 0, 0 | 370, 266, 432 in the same corridor **with** panels |
| `corr_lm8_l10w10` color | 7, one coarse scale | 291 in the same square **without** panels |
| `corr_lm0_l10w10` spatial, visual | 18, 7 | 485, 534 with panels |

A count says a library is empty. It does not say whether the tree never built
a candidate that size, whether the candidates were built and came out in
pieces, or whether they were whole and lost to a later rule. Those have
different causes and different fixes.

## What it follows

Every candidate node is followed to the stage it died at:

| outcome | what it means |
|---|---|
| no candidate | the tree produced no field of that size at all. Not a rule at work: the channel cannot localise to it, and nothing downstream could have saved it |
| **size** | the field fell below the floor or above the ceiling, so it never had a scale of its own. Size therefore shows only at the two ends of the scale axis |
| **contiguity** | the field came out in pieces — the largest connected patch held under `CC_FRAC_MIN` of it. This is what an incoherent response looks like: a channel that cannot separate two distant places responds in both, and the field fragments |
| **competition** | a larger field of the same scale already claimed the ground. Routine, and the main reason counts fall with scale |
| **coverage** | *not an admission rule.* How much of the floor a scale's survivors cover, unioned, is measured and reported for every scale but admits nothing — `TILING_FRAC_MIN` defaults to 0. Set it above 0 and Rule 12 deletes short scales again, the audit grows a fourth rule, and it says so |

The counts come from the rules engine itself — `rules.admit_fields` records
which candidates survived each rule — rather than from a second
implementation that could drift from it. Those records are arrays, and the
JSON report Experiment 2 writes already drops arrays, so they cost nothing
there.

Configuration is Experiment 2's operating point exactly (EXTENT_PCTL 80,
ACT_THRESH 0.5, Rule 2 off, LAMBDA 0, seed 0), so the audit describes the same
libraries those reports describe.

## Re-sending a report without re-auditing

There is no field cache here, deliberately: the audit needs the candidates the
field banks throw away, so it rebuilds the tree for every library every time.
But everything the *report* says is in the two CSVs a finished run writes, so

```bash
sbatch --mem=8G --time=0:20:00 slurm/prune_audit.sh --report-only
```

rebuilds the report, the figures and the mail from those in seconds — no GPU,
no dataset read. Use it after a wording or figure change, or when a run's
numbers were fine but its mail was not.

## Running

Fan out one job per arena — the usual way. Run it with `bash`, not `sbatch`:
it only calls `sbatch`, once per arena.

```bash
bash slurm/prune_audit.sh --submit
```

That submits the eight arenas that have datasets, six libraries each, and they
finish in the time the slowest one takes. A bare `sbatch slurm/prune_audit.sh`
audits every arena against every channel in a single job instead — 48
libraries, around ninety minutes, well inside its 24 h. Narrow it with
`--envs` or `--pairs`:

```bash
sbatch slurm/prune_audit.sh --envs corr_lm0_l10w2
```

```bash
sbatch slurm/prune_audit.sh --pairs corr_lm0_l10w2:hog,corr_lm8_l10w2:hog
```

Cost is one field library per pair — the Gram matrix, the Ward tree and the
readout, the same work Experiment 2 does per channel. There is no cache to
reuse: the banks Experiment 2 writes hold the survivors, and this needs the
candidates that never became survivors, which are never stored. Measured on
GAIVI: two arenas, twelve libraries, twenty minutes. The CSVs are rewritten
after every library, so a job that stops early still leaves behind the
libraries it finished.

## Outputs

`data_cache/prune_audit/`

| file | contents |
|------|----------|
| `prune_audit_scales.csv` | one row per library × scale: candidates built at that scale, then how many survived size, contiguity and competition, the number admitted, the coverage the scale reached (measured, not a threshold it had to pass), and a plain-language verdict |
| `prune_audit_pairs.csv` | one row per library: the totals through the three admission rules and the number admitted, the candidate radius range against the size window, fragmentation rate, median `sigma_ratio`, and which scales survived |

Figures — `figures/prune_audit/`

**`P1_rule_shares.png` is the one the report mails.** One row per arena,
channels pooled, segments summing to 100: what size, contiguity and
competition each cut as a share of the candidates the tree offered, and what
was admitted. It answers the question the audit exists for — which rule is
producing the library — in one panel, where the per-arena funnels take eight.
Arenas run discs, then squares, then corridors, each panelled arena above its
no-panel twin.

**`P2_prune_funnels_<env>.png`, one per arena,** is written to the figure
directory but *not* mailed: one panel per channel, four bars per scale — the
candidates built there, then what survives size, contiguity and competition,
each rule in its own colour, with the floor each scale covers annotated
beneath it as the measurement it is. Counts on a linear axis, because the
question is whether anything came through at all. Per arena rather than one
sheet of 48 panels, since six channels of one arena is the comparison being
made.

P2 is the only view of **which scale** a rule emptied, and that is where the
interesting failures are — contiguity is inert nearly everywhere and then
takes a whole scale in the 2 m corridor. It stays on disk beside the CSVs for
exactly that.

The axis starts at scale 0. Candidates below the size floor are counted in
`prune_audit_scales.csv` but not drawn — a tree produces them in the
thousands, and a first bar that tall flattens every scale beside it.

**`sigma_ratio` near 1** means a node's members are as far apart in feature
space as two random locations are: the response is flat, and a flat response
makes a mask that either fills the arena or breaks into pieces. It is the
input-side number to read when Rule 1 is taking everything.


---

# Experiment 4: field shape against scale and wall proximity

`run_field_geometry.py` — every field Experiment 2 admitted, described by four
things: how elongated it is, which way it points, how far it sits from the
nearest wall, and the angle between its long axis and that wall. Then the
pairs are correlated.

Runs on all eight collected arenas — four discs (r = 3 and r = 6, with and
without panels), two 10 × 10 m squares, two 10 × 2 m corridors — reading
Experiment 2's libraries unchanged, from its cache under its cache key. Only
the position arrays are read from the HDF5 datasets, never the feature blocks.

Three questions, Spearman on each, because scale is ordinal (0 finest to 5
coarsest) and elongation is heavy-tailed:

| pair | question |
|---|---|
| scale vs elongation | do coarser fields come out longer? |
| wall distance vs elongation | are fields near a wall longer? |

Wall distance against the **angle to the wall** was a third pair until
25 September 2026. The angle is still measured, still written to `fields.csv`
(`angle_to_wall_deg`, `perpendicular`, `wall_normal_rad`) and still summarised
per scale in `descriptives.csv`; it is no longer correlated against distance
and no longer has a figure of its own.

Reported pooled, per arena, per arena × channel, and per arena × scale, the
last of which holds field size still. **Descriptive** — no null model, no
permutation, no resampling. `q` is Benjamini-Hochberg across a table's rows,
which is not a null model either: it is the correction for asking the same
question of every arena, channel and scale, so a table of 150 p-values does
not read as one.

## One thing to read before the numbers

Rule 7 fits a field's ellipse to the second moments of its **mask**, and the
mask is intersected with the floor. A field whose shape reaches past the wall
is cut, and a cut blob's moments are elongated **along** the wall — so both
wall correlations are partly the arena's outline rather than the fields.

So every correlation is reported twice:

- **all fields** — every admitted field.
- **clear of wall** — only those whose recorded ellipse does not reach the
  wall, that is `dist_to_wall_m < reach_to_wall_m`, where the reach is the
  ellipse's own support function toward the wall,
  `sqrt(a²cos²φ + b²sin²φ)` with φ the angle between the major axis and the
  wall's normal. Nothing cut these shapes.

Where the two agree the result stands on its own. Where they disagree, the
trend lives in the cut fields, and that is what the correlation found. On
synthetic test libraries with elongation planted against scale and **nothing**
planted against the wall, wall distance vs elongation came out rho −0.064 at
p = 3e-10 over all fields and rho −0.011 at p = 0.30 over the uncut ones: the
apparently significant wall effect was 4% of the fields, and exactly the ones
the wall had cut.

## Definitions worth pinning down

**Angle to wall** is the acute angle between the field's major axis and the
*inward normal* at the nearest wall point, 0–90°. **0 means the field points
straight at the wall** — perpendicular to it — and 90 means it lies along the
wall. So a positive rho against distance means fields grow more wall-parallel
as they move inward; a negative one means they point at the wall more.
`perpendicular` is the boolean `angle < 45°`.

**Wall distance** is normalised: 0 is as near a wall as the collection
keep-out lets a field's centre sit, 1 is the disc's centre or the rectangle's
midline. Within one arena this gives the same Spearman as metres — ranks do
not care — so the normalised form exists for the pooled rows, where 1 m from a
wall means very different things in an r = 6 disc and a 2 m corridor. Both
columns are in `fields.csv`.

A field at the exact centre of a disc has no nearest wall point, and one in a
rectangle's corner has two within a lattice spacing. Both are flagged
`wall_frame_ambiguous` and dropped from the angle correlations only — never
assigned a wall.

## Running

```
sbatch --mem=32G --time=2:00:00 slurm/field_geometry.sh
```

Run all eight arenas in one job: the figures put them side by side and the
pooled correlations need them together. Fan out one arena per job only if the
libraries have to be built, then re-run once over all eight. With the cache in
place there is no GPU work and the job takes minutes.

## Outputs

`data_cache/field_geometry/`

| file | contents |
|------|----------|
| `fields.csv` | one row per admitted field: scale, area, semi-axes, elongation, orientation, centroid, the nearest wall point and its inward normal, distance to wall in metres and normalised, angle to the wall, `perpendicular`, `reach_to_wall_m`, `crosses_wall`, `clear_of_wall`, `wall_frame_ambiguous` |
| `correlations.csv` | one row per grouping × pair × subset: n, rho, p, q |
| `descriptives.csv` | per arena × channel × scale: counts, the fraction whose ellipse crosses the wall, median area, median and p90 elongation, median wall distance, median angle, fraction perpendicular |

Figures — `figures/field_geometry/`

| figure | shows |
|--------|-------|
| `G1` | elongation by scale, one box per scale, one panel per arena |
| `G2` | elongation against wall distance: grey IQR band and black median for the whole library, plus the median within each scale |
| `G4` | both correlations as a grid — arenas × channels, one panel per pair, rho printed in each cell, `*` for q < 0.05, last column pools an arena's channels |

There is no `G3`: it plotted the angle against wall distance and went with
that correlation. `G4` keeps its number so a figure filed from an earlier run
still means the same plot.

**`G4` uses one colour scale across both panels**, symmetric about zero and
scaled to the data. That makes a red in one panel mean the same strength as a
red in the other. The cost, if one pair is strong and the other weak, is that
the weak panel renders close to white — which is why every cell prints its
exact rho.

No panel shares a y axis with another: the arenas differ by 5× in radius and
16× in area, so one scale across all eight compresses the small arenas into a
strip. Every panel carries its own ticks.

G2 and G3 show a median and an interquartile band rather than a point per
field — at ten thousand-odd fields the cloud buried its own median and the eye
read the densest region as the trend. The per-scale median lines keep the
stratification the band hides, so a trend can be seen to hold *inside* a scale
rather than being the scales sliding past each other.

The figures show the **all-fields** subset only. That is a presentation
choice, not a loss of evidence: both subsets stay in `correlations.csv`, and
the report prints how far apart they ever get (max and median |Δrho|, and
whether any sign differs). If a future run moves those numbers, the subset
column belongs back in the tables.



---

# Restoring the coverage filter, for a comparison run

The standard model has no coverage requirement: Rule 12 measures each scale's
coverage of the floor and reports it, and admits nothing. To put the filter
back — to reproduce the pre-review libraries, or to ask what it was deleting —
pass a threshold to either script:

```bash
sbatch slurm/scale_distribution.sh --tiling-frac-min 0.5
```

```bash
sbatch slurm/prune_audit.sh --tiling-frac-min 0.5
```

0.5 was the old default. Any value above 0 restores the filter; 0 is the
standard model.

## It cannot be confused with the standard model

| | standard (`TILING_FRAC_MIN` 0) | filter restored (`0.5`) |
|---|---|---|
| Experiment 2 cache and CSVs | `data_cache/scale_distribution/` | `data_cache/scale_distribution_cov0.5/` |
| Experiment 2 figures | `figures/scale_distribution/` | `figures/scale_distribution_cov0.5/` |
| bank filenames | `..._roff_cov0_bank.csv` | `..._roff_cov0.5_bank.csv` |
| prune audit | `data_cache/prune_audit/` | `data_cache/prune_audit_cov0.5/` |
| admission rules | three | four |
| email subject | as usual | prefixed `[Rule 12 coverage 0.5]` |

The coverage setting is in every **bank filename unconditionally**, not just
in the directory. When the default moved from 0.50 to 0 a name carrying no
coverage marker would otherwise have been claimed by the new default and read
back as a library it is not — so banks built before the change match no
current key and are inert. If you have pre-change banks in
`data_cache/scale_distribution/`, they are the `..._roff_bank.csv` files with
no `_cov` in the name, and they can be deleted.

Both scripts lead their report with a section saying the filter is on, so a
comparison run can never be read as the standard one.

## What restoring it does

Rule 12 policed **both** ends of the ladder — coarse scales for having too few
nodes that large, fine scales for needing more fields than the tree produces —
so restoring it narrows the size range from both ends. A smaller CV and a
narrower max/min are arithmetic, not findings.

It also deleted **wholesale**: a scale at 0.49 coverage went entirely while one
at 0.51 was kept entirely. That discontinuity, and the fact that the deleted
scales had passed every other rule, is why it is no longer an admission rule.

For the prune audit specifically, Rules 8/9, 1 and 11 all sit **upstream** of
Rule 12 and none reads the threshold, so their counts come out identical either
way, field for field. Only the coverage column and the scales it deletes
differ.

---

# Extent validation: where should a field's edge be drawn?

`run_extent_validation.py` — the calibration behind `EXTENT_PCTL`. It replaced
`run_field_recovery.py`, which chose 65 on 21 August 2026 in the r = 10 disc
and can no longer run (see `RETIRED.md`). Its report always judges whatever
value `rules.py` holds, and can come back "no".

## What it found (job 494145, 29 September 2026)

It came back "no" for 65. Pooled over eight arenas × six channels:

| criterion | best q (95% CI) | nearly as good | at 65 | at 80 |
|---|---|---|---|---|
| accuracy (IoU) | 80 (75–85) | 70–85 | 0.625 | 0.692 |
| calibration (drawn ÷ true area) | 82 (79–85) | 75–85 | ×0.67 | ×0.95 |
| discrimination (Youden's J) | 75 | 65–85 | 0.697 | 0.700 |

Only 75–85 passes all three. **`EXTENT_PCTL` moved to 80**: the accuracy
peak, nearest the right-size point, and near-best for scales 0–4 and all four
wall contours, where 65 was near-best for one scale and one contour.

- **Why 65 was too tight.** Below about q = 70 almost no look-alike floor
  enters, so the drawn field is just the inner q% of the group — at 65, a third
  too small in area. Past it, a ring of floor just outside the field's edge
  starts to enter, because it looks like the edge; the drawn field is the right
  size where that ring replaces the members left out.
- **The best q is a property of the arena.** It falls as a space gets more
  visually ambiguous: 85–90 in the r = 3 discs, 75–80 in the r = 6 discs and
  squares, 40–60 in the corridor without landmarks, 95 for lidar outside the
  corridors. The r = 10 disc, with small cues in a large room, is why 65 was
  chosen once.
- **The papers' results do not rest on it.** Rebuilt at 50, 65 and 80, field
  counts move about 20% per step; median radius moves about 1%, every library
  keeps six scales, and the wall correlations stay put (elongation −0.61 to
  −0.62, size ≈ 0.06).
- **Drawing a round field near a wall stretches it along the wall** (V4). The
  report now measures how much (V7): true against drawn elongation by wall
  distance, and the wall correlation for true discs, drawn discs and the
  libraries. The first run predates that measurement; `--stages recovery`
  adds it in minutes.

## What q is

A field starts as a group of positions whose views look alike. q sets where its
edge is drawn: the boundary encloses q% of the group's own members. Too low and
the field is drawn too small; too high and it lets in every position that looks
at least as much like the group as its least typical member — look-alike floor,
often far away. Somewhere between, the members left out and the look-alikes let
in balance. V1 shows it happening to one field.

## The test

The model is handed fields whose answer is known — a disc of floor — and its
own extent machinery draws each one at every q from 5 to 100. It is handed
nothing else: only the views from inside the disc, which is exactly what it
gets from a group the clustering built.

| | |
|---|---|
| sizes | one disc per scale, 0 to 5, at the geometric centre of the scale's radius range — so sizes follow each arena's own Rule 8/9 window, and with sample count held constant each scale holds about the same number of positions in every arena |
| places | 24 sites, six on each of four contours from the wall to the most open floor (5%, 30%, 55%, 80% of the way), spread by farthest-point sampling |
| non-fields | *scattered* (the field's member count, drawn from 16× its area), *shuffled* (drawn from anywhere), *oversized* (40% of the floor, twice the ceiling), *two lobes* and *ring* (the field's area, split or hollow) |
| q | 5, 10, …, 100 |
| arenas × channels | all eight × all six |

**The code path is the pipeline's own.** `evaluate` mirrors
`prepare_candidates` and `admit_fields` for a single group: the centroid over
all members, the spread from a subsample of at most `SIGMA_MAX_MEMBERS`, sigma
solved to put the cut at the q-th percentile, the readout's fallback for a
degenerate sigma, `mask_from_grid`, and Rules 8, 9 and 1. Checked against
`prepare_candidates` on real tree nodes at q = 35, 65 and 90: sigma agrees to
~1e-6 and the masks are identical bin for bin (one bin, once, at float32
rounding). Distance to the centroid does not depend on q, so each group's is
computed once and the whole sweep reads it — the reason 20 values of q cost
little more than one.

## Three criteria, fixed before the run

| criterion | measured as | "nearly as good" |
|---|---|---|
| accuracy | median IoU of the drawn field with the true one | within 5% of the best |
| calibration | median drawn area ÷ true area; best where it crosses 1 | within 25% of the true area |
| discrimination | true fields admitted − single-region non-fields admitted (Youden's J) | within 0.05 of the best |

Each gets a best q, a 95% interval from a bootstrap that resamples whole
(arena, channel) pairs — trials inside one pair share a feature space and are
not independent — and the run of q that is nearly as good. The report says, per
criterion, whether 65 is inside, and leads with the verdict. **It can come
back "no"**, and on synthetic features with no look-alike floor it does.

Two-lobed controls are reported and not scored: one centroid cannot represent
two regions at any q, so scoring q on them would choose it for a reason q
cannot affect. An (arena, channel) pair whose best IoU is under 0.25 recovers no
field at any q and is left out of the robustness count, and named.

**Downstream.** The pipeline stage rebuilds every real library at q = 50, 65
and 80 (one tree per channel serves all three) and reports field count, median
and smallest radius, scales occupied, and the Spearman correlations of
elongation and of radius with wall distance. The libraries at the value in use are
Experiment 2's.

## Running

One job per arena, then a report job that waits for them and sends the one
email:

```bash
bash slurm/extent_validation.sh --submit
```

Or everything in one job:

```bash
sbatch slurm/extent_validation.sh
```

Rebuild the report and figures from the cache, without recomputing:

```bash
sbatch --mem=16G --time=1:00:00 slurm/extent_validation.sh --report-only
```

Options: `--envs`, `--channels`, `--stages recovery,pipeline`, `--q`,
`--pipeline-q`, `--gallery-channel` (V1 and the mailed V4; default `all`),
`--example-env` (V1; default `circ_lm8_r6`), `--n-boot`, `--no-report`,
`--report-only`, `--no-email`, `--no-gpu`.

## Outputs

`data_cache/extent_validation/<env>/`, written after every channel so a job
that stops early keeps what it finished:

| file | contents |
|---|---|
| `recovery.csv` | one row per (channel, true field, q): site, contour, wall distance, scale, radius, member count, sigma, IoU, recall, precision, centre error, log2 area ratio, drawn area and shape, contiguity, admission |
| `controls.csv` | the same for every non-field, with `kind` |
| `pipeline.csv`, `pipeline_scales.csv` | per (channel, q) library summary, and fields per scale |
| `figdata.npz` | the grid, every true field and every field drawn at the value in use (bit-packed), and V1's example per channel |
| `meta.json` | geometry, radii, q grid, code revision, channels done, any pipeline failures |

`data_cache/extent_validation/` — written by the report pass and attached to
the email: `q_curves.csv` (the pooled criteria at every q with intervals),
`cell_optima.csv` (best q per arena × channel), `level_optima.csv` (best q per
scale and per wall contour), `downstream.csv`, `wall_shape.csv` (true and drawn
elongation and orientation by wall contour) and `wall_shape_pairs.csv` (the
wall correlation per pair).

## Figures

`figures/extent_validation/` — PNG at 300 dpi, PDF with editable text, both
at 174 mm double-column width, and all of them in
`extent_validation_figures.pdf`.

| figure | shows |
|---|---|
| V1 | one representative field (the trial nearest the median, not the best): the floor coloured by the q at which each spot joins the field, the area taken in as q grows split into true field and look-alike floor, and the field at q = 20, 65, 95 |
| V2 | the three criteria against q, pooled with 95% interval, each arena as a thin line, best q marked, the nearly-as-good range shaded — **the main result** |
| V3 | accuracy at the value in use as % of each arena × channel pair's own best, with that pair's best q; a histogram of those best q; accuracy against q per scale and per wall contour |
| V4 | every arena, true fields and the fields drawn at the value in use; one figure per channel, the `--gallery-channel` one mailed |
| V5 | the real libraries at q = 50, 65, 80, each library a thin line |
| V6 | each non-field against the real fields, with a sketch of what it is |
| V7 | true against drawn elongation by wall distance, the share of elongated fields lying along the wall, and the wall correlation for true discs, drawn discs and the model's own libraries |
