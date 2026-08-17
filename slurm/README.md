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

```bash
sbatch slurm/channel_isolation.sh circ_lm8_r0
```
