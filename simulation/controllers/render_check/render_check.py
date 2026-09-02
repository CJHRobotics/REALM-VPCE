"""Smallest possible proof that Webots renders on GAIVI.

Loads one arena, teleports to a few poses, captures the camera image and the
lidar scan at each, and mails them. The point is to see an actual picture
before committing twenty hours to a collection run -- a headless renderer
that returns blank frames produces datasets of exactly the right shape whose
features are identical everywhere, and nothing downstream notices.

Looking at the image answers in one glance what no summary statistic does:
is the arena there, are the landmarks the right colours, is the floor
textured, is the geometry sane.

Writes to data_cache/render_check/ and emails the lot. Needs EMAIL_TO set;
without it the files are still written and the send is a silent no-op.
"""

import os
import sys

# Webots launches this from the controller's own directory, so the repo is
# neither the working directory nor on sys.path. chdir alone is not enough:
# sys.path[0] is the controller directory, and `realm_tools` lives at the
# repo root. Resolve from __file__ rather than a relative chdir so the two
# cannot disagree, and add the root explicitly -- the Mac venv gets this
# from a .pth that realm_install.py writes, which a conda environment on the
# cluster does not have.
REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
print(f"repo root: {REPO}", flush=True)

import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from realm_tools.robot_lib.my_robot import MyRobot
from realm_tools.experiment_lib.reporting import send_email

MAZE = os.environ.get('REALM_MAZE', 'circ_lm8_r0')
# Distinguishes repeated runs, so two configurations can be compared instead
# of overwriting each other's output.
TAG = os.environ.get('REALM_RUN_TAG', '')
SUF = f'_{TAG}' if TAG else ''
# Captures used for the timing estimate. Three poses is enough to see whether
# the renderer works and far too few to time it.
TIMED_N = int(os.environ.get('REALM_TIMED_N', '20'))
MAZE_FILE = f'simulation/worlds/environments/vpce/{MAZE}.xml'
OUT_DIR = 'data_cache/render_check'

# Centre, off-centre, and near the wall: enough to tell a real render from a
# constant one, since the three should look obviously different.
POSES = [(0.0, 0.0, 0.0, 'centre_facing_0'),
         (0.0, 0.0, 1.5708, 'centre_facing_90'),
         (5.0, 0.0, 0.0, 'offset_facing_wall')]

os.makedirs(OUT_DIR, exist_ok=True)

robot = MyRobot(enable_cnn_features=False)
robot.load_environment(MAZE_FILE, floor_texture='carpet')
print(f'loaded {MAZE_FILE}', flush=True)

attachments, lines = [], [f'render check | maze = {MAZE}'
                         + (f' | run = {TAG}' if TAG else ''), '']

for (x, y, theta, tag) in POSES:
    robot.teleport_robot(x=x, y=y, theta=theta)
    images, masks, azimuths, lidar = robot.capture_pov_images([theta])
    img = np.asarray(images[0])

    png = f'{OUT_DIR}/camera_{tag}{SUF}.png'
    Image.fromarray(img.astype(np.uint8)).save(png)
    attachments.append(png)

    # A constant image is the failure mode this whole check exists for, so
    # report the spread as well as shipping the picture.
    lines += [f'--- {tag}  (x={x}, y={y}, theta={theta:.4f})',
              f'  image   shape {img.shape} dtype {img.dtype}',
              f'  pixels  min {img.min()} max {img.max()} '
              f'mean {img.mean():.2f} std {img.std():.2f}'
              + ('   <-- CONSTANT IMAGE' if img.std() < 1e-6 else ''),
              f'  landmarks visible {int(np.nansum(masks[0]))} of {len(masks[0])}']

    if lidar is not None:
        lidar = np.asarray(lidar, dtype=float)
        finite = np.isfinite(lidar)
        txt = f'{OUT_DIR}/lidar_{tag}{SUF}.txt'
        with open(txt, 'w') as f:
            f.write(f'# lidar at x={x} y={y} theta={theta}\n')
            f.write('# index,angle_rad,range_m\n')
            ang = np.linspace(0, 2 * np.pi, len(lidar), endpoint=False)
            for i, (a, r) in enumerate(zip(ang, lidar)):
                f.write(f'{i},{a:.6f},{r:.6f}\n')
        attachments.append(txt)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='polar')
        ax.plot(ang[finite], lidar[finite], '.', ms=3)
        ax.set_title(f'lidar — {tag}')
        plot = f'{OUT_DIR}/lidar_{tag}{SUF}.png'
        fig.savefig(plot, dpi=130, bbox_inches='tight')
        plt.close(fig)
        attachments.append(plot)

        lines += [f'  lidar   {len(lidar)} beams, {100*finite.mean():.1f}% finite, '
                  f'range {np.nanmin(lidar):.3f}–{np.nanmax(lidar):.3f} m']
    else:
        lines.append('  lidar   none returned')
    lines.append('')
    print('\n'.join(lines[-6:]), flush=True)

# --- timing --------------------------------------------------------------
# The question this answers: does a Webots flag that skips the main 3D view
# also skip the offscreen camera render? If frames stay correct and this
# number drops, collection gets cheaper by the same factor. If frames go
# blank, the flag disables the sensor render too and is unusable here.
rng = np.random.default_rng(0)
# Time on the arena's own collection grid rather than invented coordinates,
# so this measures what a collection run would actually do and works for any
# geometry, corridor included.
import pandas as pd
grid = pd.read_csv(f'simulation/worlds/environments/vpce/positions/'
                   f'{MAZE}_positions.csv')
sample = grid.sample(n=min(TIMED_N, len(grid)), random_state=0)

robot.capture_pov_images([0.0])                      # warm up
t0 = time.perf_counter()
for _, pos in sample.iterrows():
    a = float(rng.uniform(0, 2 * np.pi))
    robot.teleport_robot(x=float(pos.x), y=float(pos.y), theta=a)
    robot.capture_pov_images([a])
dt = (time.perf_counter() - t0) / len(sample)

lines += ['--- timing ---',
          f'  {len(sample)} single-heading captures on the real grid, '
          f'{1000*dt:.1f} ms each',
          f'  implied per position (8 headings): {8*1000*dt:.0f} ms',
          f'  implied for this arena ({len(grid):,} positions): '
          f'{8*dt*len(grid)/3600:.2f} h',
          '']

body = '\n'.join(lines)
print(body, flush=True)
send_email(f'[REALM-VPCE] render check — {MAZE}' + (f' [{TAG}]' if TAG else ''),
           body, attachments=attachments)
print(f'wrote {len(attachments)} files to {OUT_DIR}', flush=True)

# Quit, not reset: reset restarts the controller and this would run forever.
robot.experiment_supervisor.simulationQuit(0)
