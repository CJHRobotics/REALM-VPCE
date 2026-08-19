# SLURM job scripts for GAIVI

Job scripts here submit the place-field analysis pipeline to GAIVI's
scheduler. Each script is a thin wrapper: activate the venv, `cd` into
the repo, and run one of the analysis entry points.

## Cluster-specific header

GAIVI's partition / account / QoS / GPU count are not encoded here —
fill in the `#SBATCH` header at the top of each script before the first
submission, then keep it consistent across jobs. Common lines:

```bash
#SBATCH --partition=<name>
#SBATCH --account=<if-required>
#SBATCH --time=HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=<int>
#SBATCH --mem=<size>G
#SBATCH --gres=gpu:1                 # only if the run actually uses GPU
#SBATCH --job-name=<label>
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
```

`compare_feature_sets.sh` is CPU-only: it benefits from many cores (numpy
BLAS parallelises well) and lots of RAM (~20 GB for the full 30 k-position
run).

`channel_isolation.sh` uses a GPU for the two dominant steps — the N × N
feature Gram matrix and the environment readout — and falls back to CPU
automatically if none is visible, at roughly 20× the wall time. Ward linkage
is sequential and stays on the CPU either way.

## Submit

```bash
sbatch slurm/compare_feature_sets.sh circ_lm8_r0
```

Logs go to `slurm/logs/` (git-ignored via the top-level rule).

## Scripts

| script | what it runs |
|---|---|
| `compare_feature_sets.sh` | three-way comparison (full / lidar / visual) via `analysis/experiment_feature_selection/compare_feature_sets.py` |
| `channel_isolation.sh` | per-channel isolation (hog / color / spatial / lidar / visual / all) × spatial-weighting sweep, under the agglomeration rules, via `analysis/experiment_channel_isolation/run_channel_isolation.py` |
| `threshold_sweep.sh` | response-threshold sweep (0.20 ephys convention vs our 0.50) via `analysis/experiment_channel_isolation/run_threshold_sweep.py` |
| `pruning_sweep.sh` | competition separation × coverage requirement grid via `analysis/experiment_channel_isolation/run_pruning_sweep.py` |
| `locality_test.sh` | does a channel carry location information, or does the width statistic fail? via `analysis/experiment_channel_isolation/run_locality_test.py` |

## Emailed reports

New experiments build their own report by subclassing `ExperimentReport` in
`realm_tools/experiment_lib/reporting.py` — the body is composed from the
experiment's results rather than scraped from the job log. `send_report.py`
keeps the older log-scraping CLI for job scripts that predate it, sharing the
same SMTP transport. Every send is a silent no-op without `EMAIL_TO`.

```bash
sbatch slurm/channel_isolation.sh circ_lm8_r0
```

```bash
sbatch slurm/threshold_sweep.sh circ_lm8_r0
```

```bash
sbatch slurm/pruning_sweep.sh circ_lm8_r0
```

```bash
sbatch slurm/locality_test.sh circ_lm8_r0
```

## GPU selection

`general` mixes card types. Exact GRES strings — **case-sensitive**:

| node | GRES | VRAM | arch | TF32 |
|---|---|---|---|---|
| GPU6 | `gpu:1080Ti:4` | 11 GB | sm_61 | no |
| GPU42 | `gpu:TitanRTX:4` | 24 GB | sm_75 | no |
| GPU43, GPU44 | `gpu:A40:4` | 48 GB | sm_86 | **yes** |

Both GPU job scripts request `--gres=gpu:A40:1` in their header: the widest
feature matrix is 7.2 GB, which is tight on an 11 GB 1080 Ti, and TF32 needs
Ampere. Take whatever is free instead with `--gres=gpu:1` at submit time.
Re-check the strings after any cluster change:

```bash
sinfo -p general -N -o "%N %G %t"
```
