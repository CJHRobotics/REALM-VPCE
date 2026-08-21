# SLURM job scripts for GAIVI

Job scripts here submit the place-field analysis pipeline to GAIVI's
scheduler. Each script is a thin wrapper: activate the venv, `cd` into
the repo, and run one of the analysis entry points.

## Quick reference

Every script carries a complete `#SBATCH` header, so
`sbatch slurm/<name>.sh <env>` is all that is needed. Arguments after
the environment name are forwarded to the Python entry point.

```bash
sbatch slurm/field_recovery.sh circ_lm8_r0
```

```bash
sbatch slurm/field_recovery.sh circ_lm8_r0 --tests 1,2 --channels color,hog
```

| script | job name | time | mem | GPU | what it runs |
|---|---|---|---|---|---|
| `channel_isolation.sh` | `chan-iso` | 6 h | 128 G | `gpu:1` | per-channel isolation (hog / color / spatial / lidar / visual / all) × spatial-weighting sweep, under the agglomeration rules, via `analysis/experiment_channel_isolation/run_channel_isolation.py` |
| `field_recovery.sh` | `fieldrec` | 6 h | 128 G | `gpu:1` | recovery of ideal place cells of known size, rejection of non-fields, and the `EXTENT_PCTL` sweep, via `analysis/experiment_channel_isolation/run_field_recovery.py` |
| `pruning_sweep.sh` | `prune-sweep` | 6 h | 128 G | `gpu:1` | competition separation × coverage requirement grid via `analysis/experiment_channel_isolation/run_pruning_sweep.py` |
| `locality_test.sh` | `locality` | 4 h | 128 G | `gpu:1` | does a channel carry location information, or does the width statistic fail? via `analysis/experiment_channel_isolation/run_locality_test.py` |
| `compare_feature_sets.sh` | `cmp-feat` | 4 h | 64 G | none | three-way comparison (full / lidar / visual) via `analysis/experiment_feature_selection/compare_feature_sets.py` |

All run on the **`general`** partition with 16 CPUs. Logs land in
`slurm/logs/<job-name>-<jobid>.out` and `.err`, git-ignored.

## GPU devices on `general`

The partition mixes card types. GRES strings are **case-sensitive** — `A40`,
not `a40`; a wrong string does not error, it just never matches.

| node | GRES string | VRAM | arch | TF32 |
|---|---|---|---|---|
| GPU6 | `gpu:1080Ti:4` | 11 GB | sm_61 | no |
| GPU42 | `gpu:TitanRTX:4` | 24 GB | sm_75 | no |
| GPU43, GPU44 | `gpu:A40:4` | 48 GB | sm_86 | **yes** |

Re-check after any cluster change:

```bash
sinfo -p general -N -o "%N %G %t"
```

### Why the scripts ask for `gpu:1` rather than a named type

Pinning the GRES to `gpu:A40:1` queues the job behind everything else waiting
on those four nodes, which in practice costs far more than the A40 saves.
Asking for any GPU schedules more or less immediately.

Nothing breaks on a smaller card. The widest feature matrices (`visual`
7.2 GB, `all` 7.3 GB) will not fit an 11 GB 1080 Ti alongside their working
set, but `rules.feature_sq_distances` and `rules.environment_readout` both
test free VRAM and fall back to the CPU for that stage — so a small GPU costs
time, not correctness. Ward linkage is sequential and stays on the CPU either
way.

Take an A40 for a run that needs the speed, without editing anything:

```bash
sbatch --gres=gpu:A40:1 slurm/channel_isolation.sh circ_lm8_r0
```

## Useful scheduler commands

```bash
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.10l %.20R"
```

```bash
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES%40,ExitCode
```

```bash
scancel <jobid>
```

## Emailed reports

New experiments build their own report by subclassing `ExperimentReport` in
`realm_tools/experiment_lib/reporting.py` — the body is composed from the
experiment's results rather than scraped from the job log. `send_report.py`
keeps the older log-scraping CLI for job scripts that predate it, sharing the
same SMTP transport. Every send is a silent no-op without `EMAIL_TO`.
