# Webots on GAIVI — container setup and status

Status as of **2026-08-24**. Webots R2025a runs headless on GAIVI compute
nodes, verified end to end. Data collection and RL training on the cluster
are not yet wired up.

This exists to set up **reinforcement learning**, which needs many
simulator instances running for a long time. Batch data collection is the
near-term use and the convenient way to prove the setup works.

---

## The problem

Webots is not one binary — it is a binary plus a large dependency tree
(Qt 6, X11 client libs, an OpenGL implementation). Three things collide on
GAIVI:

1. **No root**, so its dependencies cannot be installed.
2. **No display** on compute nodes, and no Xvfb or Mesa there either.
3. **The workload renders anyway.** `capture_pov_images` drives a camera
   and a lidar, which Webots draws through OpenGL. No GL context, no
   observations. Physics alone would have been easy; rendering is the
   hard part.

## The solution

A container image carries a complete Ubuntu 22.04 userspace with Webots,
Python 3.11, Mesa, and Xvfb inside. Nothing is installed on GAIVI.

Building an image needs root, which we do not have on GAIVI, and GAIVI's
`singularity` is **3.6.3** (2020) which cannot build unprivileged. GAIVI
ships a `container-builder-client` for exactly this, but its build daemon
(`eregion.cse.usf.edu:80`) is **unreachable** — connection times out from
the login node. A helpdesk ticket was filed on 2026-08-24; no reply yet.
It is probably a retired service.

So the build happens on GitHub Actions instead — a throwaway x86_64 VM
where we do have root — and the result is shipped through a registry.
`singularity pull` needs no privileges.

```
container/Dockerfile
      │  GitHub Actions: x86_64 runner, root, runs the RUN steps
      ▼
ghcr.io/cjhrobotics/realm-webots:r2025a
      │  singularity pull   ← no privileges needed
      ▼
/home/c/chamilton4/REALM-VPCE/webots_r2025a.sif   (429 MB, read-only)
```

Building on the Mac is not an option: it is arm64, GAIVI is x86_64.

### Rendering: software, on purpose

`LIBGL_ALWAYS_SOFTWARE=1` and `GALLIUM_DRIVER=llvmpipe` force OpenGL onto
the CPU via Mesa. This is viable because the render load is tiny — a
224x224 camera, a 360x1 single-layer lidar, and a scene of one cylinder
plus a handful of flat panels.

The payoff is scheduling: these jobs need **no GPU**, so they do not queue
behind the cluster's GPU work. That matters much more for RL (dozens of
parallel envs) than for one collection run.

---

## Files

| file | what it is |
|---|---|
| `container/Dockerfile` | the image recipe: Ubuntu 22.04, Python 3.11 (deadsnakes), Xvfb, Mesa, Webots R2025a `.deb` |
| `.github/workflows/build-webots-image.yml` | builds it on Actions, pushes to GHCR; manual trigger only |
| `slurm/webots_smoke.sh` | staged headless smoke test, no GPU requested |

The REALM repo and its venv deliberately stay **outside** the image and are
bind-mounted at run time (singularity auto-mounts `$HOME`, `$PWD`, `/tmp`).
Editing code never means rebuilding the image.

---

## How to rebuild and redeploy

1. Edit `container/Dockerfile`, commit, push.
2. GitHub → Actions → `build-webots-image` → Run workflow. **Use a new tag**
   (`r2025a-2`, ...) rather than reusing one — see the stale-image trap
   below. ~10-15 min. Free: the repo is public.
3. First time only: repo → Packages → `realm-webots` → visibility **public**,
   or the pull needs credentials on GAIVI.
4. On GAIVI:

```bash
export SINGULARITY_CACHEDIR=/home/c/chamilton4/.sing-cache
singularity pull webots_r2025a.sif docker://ghcr.io/cjhrobotics/realm-webots:<tag>
```

5. Verify and smoke test:

```bash
sbatch slurm/webots_smoke.sh
```

### The stale-image trap

Reusing a tag means `singularity pull` may hand back the cached layers and
silently give you the **old** image. This cost us one confused debugging
round. Either use a fresh tag each build, or clear both artifacts:

```bash
rm -f webots_r2025a.sif && rm -rf /home/c/chamilton4/.sing-cache
```

Stage 1b of the smoke test prints the Qt xcb packages in the image
specifically so a stale image is visible at a glance in the log.

---

## What has been tested

Smoke test job `451104` on node GPU6, 16 CPUs, **no GPU**, all stages pass:

| check | result |
|---|---|
| image runs, Webots version | `R2025a` |
| Python | `3.11.15` |
| Webots install size | 433 MB in `/usr/local/webots` |
| Qt xcb deps | `libxcb-cursor0`, `libxcb-xinerama0`, `libxkbcommon-x11-0` present |
| offscreen GL | Mesa `llvmpipe`, **OpenGL 4.5** (Webots needs 3.3) |
| headless world load | sample camera world loads, controller starts, **camera frames render** ("found a blue blob" etc.) |
| singularity 3.6.3 vs GHCR manifest | pulls and converts fine |

`llvmpipe` reports **256 bits** SIMD on compute nodes vs 128 on the login
node — the compute nodes are meaningfully faster. Benchmark there, not on
a login node.

### Problems hit and fixed

- **Registry tag rejected.** `CJHRobotics` is uppercase; registries require
  lowercase. Workflow now lowercases `github.repository_owner`. (`899dff0`)
- **Webots aborted at startup.** Webots R2025a ships Qt 6.5+, which made
  `xcb-cursor` a hard requirement of the xcb platform plugin. Failed before
  ever reaching a display, so Xvfb was not the issue. (`9190d0d`)
- **Smoke test aborted at stage 1.** `webots --version` initialises Qt and
  needs a display; under `set -e` that killed the job before the
  informative stages ran. Now runs under `xvfb-run`. (`911dc3a`)

### Harmless noise in the logs

- `Cannot initialize the sound engine / OpenAL` — no audio device on a
  compute node. Nothing we use touches audio.
- `WARNING: System below the minimal requirements` — Webots noticing it is
  software rendering. Expected. But see the confound below.

---

## Open issue: render quality is machine-dependent

On detecting software rendering, Webots **silently downgraded** itself:

```
- Shadows have been deactivated.
- Anti-aliasing has been deactivated.
- Main 3D view global ambient occlusion has been de-activated.
- Texture quality has been reduced.
```

A GPU-equipped Mac does none of this. So the same pose in the same world
does **not** produce identical pixels on the two machines — and HOG and
colour-histogram features are computed straight from those pixels.

**Consequence:** datasets collected on the Mac are not strictly comparable
to datasets collected on GAIVI. For the landmark-count experiment
(4/6/8/10) that would put a rendering difference directly on top of the
independent variable.

**Current decision (2026-08-24):** all four datasets were collected on the
**Mac** and will be copied to GAIVI. Comparability is preserved because
every condition shares one renderer. GAIVI is used for analysis only, for
now.

**When collection moves to GAIVI, re-collect every condition there.** Do
not mix. The `.h5`-exists skip in `collect_data.py` makes a partial
re-collection easy to get wrong — delete deliberately.

Worth doing regardless: pin the render settings explicitly in Webots
preferences rather than letting a per-machine heuristic choose them. The
features were disabled by a *speed heuristic*, not a hardware limit —
llvmpipe implements OpenGL 4.5 and can do shadows and AA, just slowly.
Auto-selected fidelity is a bad property for reproducible data collection.

---

## Why not use the GPUs

GAIVI has plenty of GPUs, and `singularity --nv` would bind the driver in.
It does not help by itself: Webots renders through **GLX**, GLX
acceleration comes from the X server, and Xvfb is a software X server. The
GPU would be mounted and never asked for anything.

Real options, if throughput ever demands it:

- **VirtualGL** — `vglrun` interposes on GLX, renders on the GPU (3.x can
  use EGL with no X server), blits back into Xvfb. The standard solution.
  Installable at image build time. `libEGL_nvidia.so.0` is present on the
  nodes, so the pieces are there.
- **`QT_QPA_PLATFORM=eglfs`** — skips X entirely, but depends on Webots
  supporting it. Unverified and probably a rabbit hole.

**This is not a fidelity fix.** Hardware GL would still not match the Mac —
different vendor, driver, rasterization, texture filtering. Only collecting
every condition on one platform fixes comparability.

Order to try things in, if rendering turns out too slow:

1. Disable what is not needed. The robot `Display` is **800x800** and
   redraws on every teleport — 12x the camera's pixels, for a
   visualisation nobody watches on a cluster.
2. Scale out, not up: many CPU-only Webots instances. For RL this beats GPU
   rendering anyway, since CPU allocations schedule far faster.
3. Only then VirtualGL.

Drive this with a measured positions-per-second number from the real world
file, not from the fact that the GPUs exist. **That benchmark has not been
run yet.**

---

## Where things stand

**Done:** image builds and deploys; Webots runs headless on a compute node
and renders camera frames; no GPU needed.

**Not done:**

- **Run REALM's own controller.** Everything so far used a Webots *sample*
  world. Needs three things: a venv built **inside** the container (so
  packages link against container libs) on the host filesystem, a
  `runtime.ini` pointing at that interpreter instead of the hardcoded Mac
  path, and a job script launching the right `.wbt` headless. Note there is
  no obvious VPCE `.wbt` — worlds present are `break_room.wbt`,
  `empty_room.wbt`, `empty_room_tile.wbt`.
- **Verify Webots' Python bindings import under Python 3.11** in the
  container (`python3.11 -c "import controller"` with `PYTHONPATH` set).
  Untested.
- **Benchmark** llvmpipe on the real world.
- **RL: extern controllers.** The `WebotsEnv` skeleton in
  `realm_tools/simulation_lib/webots_torch_environment.py` is still empty.
  Build it **extern-controller-first** (`WEBOTS_CONTROLLER_URL` +
  `webots-controller`), so the training script is the parent process and
  Webots is a subprocess it manages. That inversion is what allows SB3
  vectorised envs, N parallel instances each with its own Xvfb display and
  port, and checkpointing untied to Webots' process lifetime. Retrofitting
  it later means rewriting the process/lifecycle layer.
- **Helpdesk ticket** on the dead build service — still open. A working
  institutional builder, or a modern Apptainer module, would be cleaner
  than routing images through a public registry.

## Reference

Cluster facts worth not rediscovering:

- `singularity` at `/apps/singularity/bin/singularity`, version **3.6.3**.
  No `apptainer`. No unprivileged build.
- Compute nodes have `libGL`, `libX11`, and `libEGL_nvidia` — but **no**
  Xvfb, Mesa, or `glxinfo`. Those come from the container.
- Outbound HTTPS to `ghcr.io` works from compute nodes (401 on `/v2/` is a
  pass — the registry is asking for auth, which proves the path).
- Webots R2025a release assets: `webots_2025a_amd64.deb` (used here) and
  `webots-R2025a-x86-64.tar.bz2` (the no-container fallback: the tarball
  needs no root and bundles its own Qt).
