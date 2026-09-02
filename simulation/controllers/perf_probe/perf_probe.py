"""Where does the cost of a Webots capture actually go?

First probe result, 32 CPUs, circ_lm8_r0:

    all on (camera+recognition+lidar)   212.2 ms/step
    camera + lidar, no recognition      210.2 ms/step
    camera only                         206.4 ms/step
    bare physics step                     0.7 ms/step

So the camera render is essentially the entire per-step cost, and
recognition (~2 ms) and lidar (~4 ms) are noise beside it. 206 ms for 50k
pixels is roughly 200x what hardware GL would cost -- this is the price of
llvmpipe.

That leaves a second discrepancy this probe exists to settle. At 212 ms per
step a capture is two steps, so ~0.42 s; the render check measured 15.5 s.
Bare step() calls are not what collection does, so this times the real path
instead -- supervisor field writes, resetPhysics, and the per-heading loop
-- to find where the other 36x lives.
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
import traceback

import numpy as np

from realm_tools.robot_lib.my_robot import MyRobot
from realm_tools.experiment_lib.reporting import send_email

MAZE = os.environ.get('REALM_MAZE', 'circ_lm8_r0')
N = int(os.environ.get('REALM_PROBE_N', '10'))

rows = []
lines = []


def bench(label, fn, n=N):
    fn()                                                # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n
    rows.append((label, dt))
    print(f'  {label:44s} {1000*dt:9.1f} ms', flush=True)
    return dt


try:
    robot = MyRobot(enable_cnn_features=False)
    robot.load_environment(
        f'simulation/worlds/environments/vpce/{MAZE}.xml', floor_texture='carpet')
    ts, sup = robot.timestep, robot.experiment_supervisor
    hdr = (f'perf probe | {MAZE} | cpus {os.cpu_count()} | '
           f'basicTimeStep {ts} ms | {N} reps')
    print(hdr, flush=True)

    rng = np.random.default_rng(0)

    def rand_xy():
        a = rng.uniform(0, 2 * np.pi)
        r = 8.0 * np.sqrt(rng.uniform())
        return float(r * np.cos(a)), float(r * np.sin(a)), float(a)

    print('--- per-step primitives ---', flush=True)
    bench('bare step()', lambda: sup.step(ts))
    bench('step() + camera.getImage()',
          lambda: (sup.step(ts), robot.camera.getImage()))
    bench('step() + getImage + getRecognitionObjects',
          lambda: (sup.step(ts), robot.camera.getImage(),
                   robot.camera.getRecognitionObjects()))

    print('--- supervisor writes, no step ---', flush=True)
    bench('setSFRotation only',
          lambda: robot.robot_rotation_field.setSFRotation([0, 0, 1, 1.0]))

    def _tp_fields():
        x, y, th = rand_xy()
        robot.robot_translation_field.setSFVec3f([x, y, 0.09])
        robot.robot_rotation_field.setSFRotation([0, 0, 1, th])
    bench('setSFVec3f + setSFRotation', _tp_fields)
    bench('resetPhysics() only', lambda: robot.robot_node.resetPhysics())

    print('--- the real collection path ---', flush=True)

    def _teleport():
        x, y, th = rand_xy()
        robot.teleport_robot(x=x, y=y, theta=th)
    bench('teleport_robot()  [fields + resetPhysics + 1 step]', _teleport)

    def _one_heading():
        robot.capture_pov_images([0.0])
    bench('capture_pov_images(1 heading)', _one_heading)

    def _teleport_capture_1():
        x, y, th = rand_xy()
        robot.teleport_robot(x=x, y=y, theta=th)
        robot.capture_pov_images([th])
    bench('teleport + capture 1 heading  [render_check unit]',
          _teleport_capture_1)

    THETAS = [0.0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]

    def _collection_unit():
        x, y, th = rand_xy()
        robot.teleport_robot(x=x, y=y, theta=th)
        robot.capture_pov_images(THETAS)
    bench('teleport + capture 8 headings [collection unit]',
          _collection_unit, n=max(3, N // 3))

    unit = dict(rows)['teleport + capture 8 headings [collection unit]']
    lines = [hdr, '', 'mean wall time:', '']
    lines += [f'  {k:44s} {1000*v:9.1f} ms' for k, v in rows]
    lines += ['',
              f'One collection position costs {1000*unit:.0f} ms.',
              f'30,149 positions -> {unit*30149/3600:.1f} h per arena.']
except Exception:
    tb = traceback.format_exc()
    print(tb, flush=True)
    lines = (lines or ['perf probe failed before producing rows']) + ['', tb]
finally:
    body = '\n'.join(lines) if lines else 'perf probe produced no output'
    print(body, flush=True)
    # Sent from `finally` so a crash still reports: the previous version died
    # partway and mailed nothing, leaving only the job log.
    send_email(f'[REALM-VPCE] perf probe — {MAZE}', body)
    try:
        robot.experiment_supervisor.simulationQuit(0)
    except Exception:
        pass
