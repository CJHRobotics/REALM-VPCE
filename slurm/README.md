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

## GAIVI partitions

Snapshot from `sinfo`; re-run it rather than trusting this table for
availability. `general` is the default and the only one every job script
targets.

| partition | time limit | nodes |
|---|---|---|
| `general`* | 7 days | GPU6, GPU42, GPU43, GPU44 |
| `nopreempt` | 7 days | GPU2, GPU3, GPU4 |
| `Quick` | 1 day | GPU1, GPU45, GPU46, GPU47 |
| `Extended` | unlimited | GPU8, GPU9, GPU22 |
| `Contributors` | unlimited | GPU1–2, GPU42–55 |
| `YES` | unlimited | GPU56 |

```bash
sinfo -p general -N -o "%N %G %t"
```

## GPU devices on `general`

Four nodes, three card types. GRES strings are **case-sensitive** — `A40`, not
`a40`; a wrong string does not error, it just never matches.

| node | GRES string | cards | VRAM per card | node RAM | cores | TF32 |
|---|---|---|---|---|---|---|
| GPU6 | `gpu:1080Ti:4` | 4 × 1080 Ti | 11 GB | 128 GB | 32 | no |
| GPU42 | `gpu:TitanRTX:4` | 4 × Titan RTX | 24 GB | 192 GB | 64 | no |
| GPU43 | `gpu:A40:4` | 4 × A40 | 48 GB | 512 GB | 128 | **yes** |
| GPU44 | `gpu:A40:4` | 4 × A40 | 48 GB | 512 GB | 64 | **yes** |

### Why the scripts ask for `gpu:1` rather than a named type

Pinning the GRES to `gpu:A40:1` queues the job behind everything else waiting
on GPU43 and GPU44, which in practice costs far more than the A40 saves.
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

### Two things the node table implies

**`--mem=128G` excludes GPU6.** That is exactly GPU6's total RAM, and SLURM
only offers a node slightly less than its physical memory, so the request can
never fit there. "Any GPU" therefore means GPU42/43/44 in practice. The jobs
do not need 128 GB — the largest working set is a 7.3 GB feature matrix plus
its 3.6 GB pairwise block, and local runs peak around 12 GB — so lowering the
request to `--mem=64G` would genuinely widen where they can land.

**`Quick` holds the A100 nodes and allows 1-day jobs.** Every script here is
6 hours or less, so they fit comfortably. If `general` is congested, that
partition is worth trying:

```bash
sbatch --partition=Quick slurm/channel_isolation.sh circ_lm8_r0
```

## Full node inventory

| node | cores | processor | RAM | cards | GPU RAM |
|---|---|---|---|---|---|
| GPU1 | 96 | AMD EPYC 7352 24-Core | 256 GB | 3 × A100 | 240 GB |
| GPU2 | 96 | AMD EPYC 7352 24-Core | 256 GB | 3 × A100 | 240 GB |
| GPU3 | 32 | Dual Xeon E5-2620 v4 @ 2.10 GHz | 128 GB | 4 × 1080 Ti | 44 GB |
| GPU4 | 32 | Dual Xeon E5-2620 v4 @ 2.10 GHz | 128 GB | 4 × 1080 Ti | 44 GB |
| GPU6 | 32 | Dual Xeon E5-2620 v4 @ 2.10 GHz | 128 GB | 4 × 1080 Ti | 44 GB |
| GPU8 | 32 | Dual Xeon E5-2620 v4 @ 2.10 GHz | 128 GB | 4 × 1080 Ti | 44 GB |
| GPU9 | 32 | Dual Xeon E5-2620 v4 @ 2.10 GHz | 128 GB | 4 × 1080 Ti | 44 GB |
| GPU22 | 40 | Dual Xeon E5-2630 v4 @ 2.20 GHz | 1024 GB | 8 × 1080 Ti | 88 GB |
| GPU42 | 64 | Dual Xeon Gold 6226R @ 2.90 GHz | 192 GB | 4 × Titan RTX | 96 GB |
| GPU43 | 128 | AMD EPYC 7662 64-Core | 512 GB | 4 × A40 | 192 GB |
| GPU44 | 64 | AMD EPYC 7532 32-Core | 512 GB | 4 × A40 | 192 GB |
| GPU45 | 48 | AMD EPYC 7413 24-Core | 256 GB | 2 × A100 | 160 GB |
| GPU46 | 96 | AMD EPYC 7352 24-Core | 2 TB | 3 × A100 | 240 GB |
| GPU47 | 64 | AMD EPYC 7513 32-Core | 512 GB | 2 × A100 | 160 GB |
| GPU48 | 256 | AMD EPYC 9554 64-Core | 768 GB | 6 × H100 | 480 GB |
| GPU49 | 64 | Xeon Silver 4314 @ 2.40 GHz | 128 GB | 2 × A100 + 2 × L40S | 256 GB |
| GPU50 | 64 | AMD EPYC Genoa 9354 @ 3.3 GHz | 384 GB | 1 × RTX A6000 | 48 GB |
| GPU51 | 64 | AMD EPYC Genoa 9374F @ 3.85 GHz | 384 GB | 2 × RTX 6000 Ada | 96 GB |
| GPU52 | 64 | AMD EPYC 7543 32-Core @ 2.8 GHz | 524 GB | 4 × RTX A6000 | 192 GB |
| GPU53 | 192 | AMD EPYC Milan 7643 @ 2.3 GHz | 1024 GB | 8 × L40S | 384 GB |
| GPU54 | 256 | AMD EPYC Genoa 9554 64-Core @ 3.1 GHz | 768 GB | 8 × RTX 6000 Ada | 384 GB |
| GPU55 | 128 | AMD EPYC 9354 32-Core @ 3.25 GHz | 384 GB | 2 × RTX Pro 6000 BW | 192 GB |
| GPU56 | 128 | AMD EPYC 9354 32-Core | 384 GB | 3 × RTX Pro 6000 BW | 288 GB |
| GPU57 | 128 | AMD EPYC 9354 32-Core | 1128 GB | 8 × RTX Pro 6000 BW | 768 GB |

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
