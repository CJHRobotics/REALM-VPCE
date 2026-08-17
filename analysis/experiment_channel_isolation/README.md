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
