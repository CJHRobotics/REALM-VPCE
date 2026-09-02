"""Where does a Webots simulation step actually go?

A single-heading capture measured 15.5 s on GAIVI: two simulation steps, so
~7.75 s each. That is far too slow to be the 224x224 camera -- 50k pixels
should rasterise in milliseconds even in software -- so something else per
step dominates, and guessing which is a poor way to spend twenty hours of
collection.

Each sensor can be toggled at runtime, so this times bare steps with
different subsets enabled and reports the share attributable to each:

    camera       224x224 RGB
    recognition  an extra segmentation pass plus object detection, used only
                 to fill landmark_mask / landmark_azimuths, which the current
                 channel-isolation pipeline does not read
    lidar        360 degrees of field of view at maxRange 40, which Webots
                 covers with several depth-render passes

Reports CPU count too, since llvmpipe is threaded and the allocation size is
one of the levers being considered.
"""

import os
import sys

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
print(f"repo root: {REPO}", flush=True)

import time
import numpy as np

from realm_tools.robot_lib.my_robot import MyRobot
from realm_tools.experiment_lib.reporting import send_email

MAZE = os.environ.get('REALM_MAZE', 'circ_lm8_r0')
N = int(os.environ.get('REALM_PROBE_N', '10'))

robot = MyRobot(enable_cnn_features=False)
robot.load_environment(
    f'simulation/worlds/environments/vpce/{MAZE}.xml', floor_texture='carpet')
ts = robot.timestep
sup = robot.experiment_supervisor
print(f'loaded {MAZE}; basicTimeStep {ts} ms; cpus {os.cpu_count()}', flush=True)


def time_steps(label, n=N, read_camera=False, read_lidar=False):
    """Mean wall time of one sim step under the current device configuration."""
    sup.step(ts)                                        # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        sup.step(ts)
        if read_camera:
            robot.camera.getImage()
        if read_lidar:
            robot.lidar.getRangeImage()
    dt = (time.perf_counter() - t0) / n
    print(f'  {label:38s} {1000*dt:9.1f} ms/step', flush=True)
    return dt


rows = []
print('--- timing ---', flush=True)

# Everything on, as collection runs today.
rows.append(('all devices, as collected', time_steps(
    'all on (camera+recognition+lidar)', read_camera=True, read_lidar=True)))

robot.camera.recognitionDisable()
rows.append(('recognition off', time_steps(
    'camera + lidar, no recognition', read_camera=True, read_lidar=True)))

robot.lidar.disable()
rows.append(('recognition + lidar off', time_steps(
    'camera only', read_camera=True)))

robot.camera.disable()
rows.append(('all sensors off', time_steps('bare physics step')))

# Put them back and isolate the lidar on its own.
robot.camera.enable(ts)
robot.lidar.enable(ts)
robot.camera.recognitionDisable()
rows.append(('lidar re-enabled, camera not read', time_steps(
    'camera enabled but unread + lidar', read_lidar=True)))

base = rows[0][1]
lines = [f'perf probe | maze = {MAZE} | cpus = {os.cpu_count()} | '
         f'basicTimeStep = {ts} ms', '',
         f'{N} steps per configuration, mean wall time per step:', '']
for label, dt in rows:
    lines.append(f'  {label:38s} {1000*dt:9.1f} ms   '
                 f'({100*dt/base:5.1f}% of full)')
lines += ['',
          'Differences between consecutive rows give each device its share.',
          'A capture is two steps, so full-configuration cost per capture is '
          f'about {2*1000*base:.0f} ms.']

body = '\n'.join(lines)
print(body, flush=True)
send_email(f'[REALM-VPCE] perf probe — {MAZE}', body)

sup.simulationQuit(0)
