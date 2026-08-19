# Handoff prompt — resolve the HOG zero-field question

Paste everything below into a new session.

---

## Context

REALM-VPCE builds a library of hippocampal-style place fields from a simulated
agent's visual input. Code lives in `analysis/experiment_channel_isolation/`;
write-ups live in the Obsidian vault at `~/VPCE-Brain/VPCE-Brain/`
(`models/Current Model.md`, `results/Channel Isolation Experiment.md`,
`reports/2026-08-19-place-fields-from-vision.md`). Jobs run on GAIVI via
`slurm/*.sh` and email themselves a report through
`realm_tools/experiment_lib/reporting.py`.

Environment `circ_lm8_r0`: a 10 m circular arena (314 m²) sampled at 30,147
positions. Six feature configurations: `hog` (edges, 30,240-d), `color`
(4,096-d), `spatial` (24,576-d), `lidar` (range capped at 5 m, 720-d),
`visual` (the three camera channels), `all`.

The pipeline agglomerates positions by view similarity (Ward), summarises each
group by a centroid μ and a width σ (90th percentile of within-group pairwise
distance), projects it onto the floor as an RBF response, thresholds at 50% of
that group's own peak, and admits the result as a place field only if it passes
four rules: contiguity, a size window, same-scale competition, and per-scale
coverage.

## The question

**`hog` produces zero place fields.** 99.3% of its candidate groups exceed the
maximum allowed field size — they respond across almost the entire floor.
Two explanations were proposed:

- **A.** Edges carry no location information in this arena.
- **B.** The width statistic σ fails in 30,240 dimensions, because pairwise
  distances concentrate and a percentile of them stops discriminating tight
  groups from loose ones.

`run_locality_test.py` (job 447358, results in
`~/Downloads/[REALM-VPCE] circ_lm8_r0 locality-test .../`) was built to
separate them. It did not, and the reason is interesting.

## What the locality test established

Numbers below were read off figures L1–L3; **the exact CSV has not been parsed
yet — start by loading it.**

**Explanation A is dead.** Every channel decodes position almost perfectly.
Nearest neighbours in feature space are 0.02–0.03 × chance away on the floor,
and k-NN position decoding error is roughly 0.1–0.2 m against a chance level of
8.7 m. `hog` is among the best. Edges carry abundant location information.

**Explanation B is not confirmed, and the obvious version of it is refuted.**
The patch-width test (feature-space width of a disc of floor, over the global
width, at radius 0.35 m) gives:

| channel | patch ratio | relative contrast | fields produced |
|---|---|---|---|
| `color` | 0.33 | 0.605 | 139 |
| `visual` | 0.79 | 0.281 | 32 |
| `all` | 0.82 | 0.214 | 134 |
| `hog` | 0.86 | 0.183 | **0** |
| `lidar` | 0.90 | 0.298 | 200 |
| `spatial` | 0.93 | 0.317 | 59 |

**`lidar` and `spatial` have worse patch ratios than `hog` and produce 200 and
59 fields.** A high patch ratio is therefore compatible with producing plenty
of fields, so "the width statistic is blind" cannot on its own explain the
zero. Relative contrast orders `hog` lowest but does not separate it from
`lidar` or `visual` either.

## The lead to follow

From `data_cache/channel_isolation/circ_lm8_r0/metrics.csv` at λ = 0:

| channel | median candidate radius | % above size ceiling | candidates passing size | fields |
|---|---|---|---|---|
| `hog` | 7.69 m | **99.3%** | 18 | 0 |
| `lidar` | **8.08 m** | 73.2% | 1,664 | 200 |
| `visual` | 6.38 m | 85.3% | 376 | 32 |
| `spatial` | 4.40 m | 48.3% | 1,334 | 59 |
| `all` | 2.99 m | 41.9% | 1,493 | 134 |
| `color` | 1.80 m | 10.8% | 2,307 | 139 |

`lidar`'s **median** candidate is larger than `hog`'s, yet 27% of its
candidates fall under the ceiling against 0.7% of `hog`'s. The difference is
not the centre of the width distribution but its **lower tail**: `lidar` has a
substantial minority of tight groups and `hog` has essentially none.

Note also that the patch test measures spatially defined groups, while the tree
builds feature-defined ones. For `lidar`, feature-defined groups can be far
tighter than any floor patch. For `hog`, apparently they cannot. That gap is
where the answer probably lives.

## What to do

1. **Load the locality-test CSV** and confirm the figures. Ask the user to copy
   it into the repo — this shell cannot read `~/Downloads` (macOS denies
   directory access; `test -f` passes but reads fail).

2. **Measure the width distribution over actual tree nodes**, per channel — not
   spatial patches. For each candidate node record σ, the distance from μ to
   every sampled position, and the resulting field area. The question to answer:
   why does `hog` have no tight nodes when `lidar` does, given that `lidar`'s
   spatial patches are looser?

3. **Check the peak-relative threshold.** Field extent is where
   `d(x, μ) ≤ 1.177 σ`, but the mask is taken at 50% of each field's *own peak*.
   If `hog`'s response profile is flat across the arena, the peak is barely
   above the floor and the 50% cut admits nearly everything. Measure the
   response profile — activity against floor distance from μ — for matched
   nodes in `hog` and `lidar`. This is the most likely mechanism and is cheap to
   test.

4. **If the width statistic is implicated**, the candidate replacement is
   scale-free: a quantile of distance-to-centroid relative to the group's own
   distribution rather than an absolute percentile of pairwise distances.
   Sweep it the way `run_threshold_sweep.py` sweeps the response threshold —
   `rules.prepare_candidates` / `rules.admit_fields` are already split so one
   tree serves many settings.

5. **Report correction, do not skip.** The published report currently states
   *"edges carry no location information"* (Summary item 3, Part II result 5,
   and the Part V open question). **That is now false** — edges decode position
   as well as any channel. Fix in all three places:
   - `~/VPCE-Brain/VPCE-Brain/reports/2026-08-19-place-fields-from-vision.md`
   - `~/VPCE-Brain/VPCE-Brain/results/Channel Isolation Experiment.md`
   - the published artifact at
     https://claude.ai/code/artifact/c9104c91-029f-4ba6-8169-26b7625004b0
     (republish with `url=` to keep the link)

   The correct statement is that edges carry location information but do not
   yield place fields under the current width measure, for a reason still being
   determined.

## Useful commands

```bash
cd ~/REALM-VPCE && git pull origin main
```

```bash
sbatch slurm/locality_test.sh circ_lm8_r0
```

```bash
python analysis/experiment_channel_isolation/run_channel_isolation.py circ_lm8_r0 --figures-only
```

Local Python is `./realm_venv/bin/python`. GAIVI wants `--gres=gpu:A40:1`,
already in the job headers; GRES type strings are case-sensitive. Keep
`numpy<2` in the conda env — a torch reinstall has broken it twice.

## Locked parameters — do not change without a reason

bin 0.25 m · response threshold 0.50 of peak · width percentile 90 · λ = 0 ·
connected fraction 0.80 · field area 0.388–62.8 m² · competition separation
0.35 · band ratio 1.6 · coverage 0.50.

The last two are choices rather than measurements; a 5×5 sweep showed the
headline results (field size grows away from the wall, fields elongate along
it) hold in 96 of 97 settings, which is what makes them safe to report.
