"""Circular arenas for the landmark-count sweep: circ_lm{4,6,8,10}_r0.

All four are the same disc -- radius 3, area 28.27 m^2 -- differing only in
how many 0.75 m panels sit on the wall, which sets interlandmark spacing from
9.4 m (lm2) to 1.6 m (lm12).

Radius 3, not 10. A 0.75 m panel seen from across a 20 m disc spans about
11 px of a 224 px image, so twelve of them cover roughly 3% of the frame and
a colour histogram is 97% floor and wall -- identical from everywhere. That
is why the colour channel collapsed to single-digit field counts at r = 10
regardless of landmark count or placement. The archived original arena, on
which the model was tuned, had a walkable radius of 2.4 m plus margin; this
restores that scale.

Counts are 2, 4, 8 and 12: each divides the clock face evenly from noon, so
every landmark sits on a whole hour and none falls on the seam between two of
the robot's eight camera views.

These were previously hand-written, which is how circ_lm8_r0 came to have its
landmarks on the camera's heading-sector boundaries. Generating them keeps
the placement rule in one place, shared with the area sweep.

    python simulation/worlds/environments/vpce/make_landmark_sweep_envs.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_area_sweep_envs import (COLORS, PANEL, WALL_H, WALL_THICK, SUBDIV,
                                  MARGIN, N_TARGET, fmt, landmark_angles)

RADIUS = 3.0
# 2, 4, 8, 12 rather than 4, 6, 8, 10: each divides the clock face evenly
# from noon, so every landmark sits at a whole clock hour and none lands on
# the seam between two camera views.
#
#   lm2   noon, 6                       ->  90, 270 deg
#   lm4   noon, 3, 6, 9                 ->  90, 0, 270, 180
#   lm8   every 1.5 hours               ->  every 45 deg, one per view
#   lm12  every hour                    ->  every 30 deg
#
# 6 and 10 are dropped because they do not divide 12, so their landmarks fall
# between clock hours and their spacing beats against the eight camera
# headings rather than aligning with them.
COUNTS = [2, 4, 8, 12]


def build_xml(n):
    a_deg = np.degrees(landmark_angles(n))
    lines = [
        '<?xml version="1.0" encoding="us-ascii"?>\n\n',
        f'<!-- Circular arena, radius {RADIUS:g} m, area '
        f'{np.pi*RADIUS**2:.2f} m^2, with {n} landmarks.\n'
        f'     XML `radius` is the inner walkable radius; the wall material\n'
        f'     extends outward by `thickness`. Panels sit on the inner face\n'
        f'     and are {PANEL} m square. Interlandmark spacing along the wall\n'
        f'     is {2*np.pi*RADIUS/n:.2f} m.\n\n'
        f'     Landmark bearings are anchored at 90 degrees:\n'
        f'     {np.round(a_deg, 1).tolist()}\n'
        f'     The robot captures at headings 0, 45 ... 315 with a 45-degree\n'
        f'     camera, so the views partition the circle at 22.5 + 45k. The\n'
        f'     earlier half-spacing convention put the 8 landmarks exactly on\n'
        f'     those boundaries. See make_area_sweep_envs.landmark_angles. -->\n\n',
        '<world>\n',
        f'\t<circular_wall radius="{RADIUS:.1f}" height="{WALL_H}" '
        f'thickness="{WALL_THICK}" subdivision="{SUBDIV}"/>\n\n',
    ]
    for k, a in enumerate(landmark_angles(n)):
        x, y, th = RADIUS * np.cos(a), RADIUS * np.sin(a), a - np.pi
        r, g, b = COLORS[k % len(COLORS)]
        lines.append(
            f'    <landmark type="panel" x="{fmt(x,4)}" y="{fmt(y,4)}" '
            f'theta="{fmt(th,3)}" height="{PANEL}" width="{PANEL}"\n'
            f'              texture="../protos/world_objects/textures/flags/flag_{k}.png"\n'
            f'              red="{r:.2f}" green="{g:.2f}" blue="{b:.2f}"/>\n')
    lines.append('''
    <train_start_positions>
        <pos x="0.0" y="0.0" theta="0.0"/>
    </train_start_positions>

    <test_start_positions>
        <pos x="0.0" y="0.0" theta="0.0"/>
    </test_start_positions>

    <goal id="0" x="0.0" y="0.0"/>
</world>
''')
    path = os.path.join(HERE, f'circ_lm{n}_r0.xml')
    with open(path, 'w') as f:
        f.writelines(lines)
    return path


def build_shared_grid():
    """The grid all four arenas use, via the fallback in collect_data.py.

    Named for circ_lm8_r0 because that is the name the fallback looks for.
    Spacing is set to hold the sample count near N_TARGET rather than fixed
    at 0.1 m: at radius 3 a 0.1 m lattice would give only ~2,500 positions,
    and field count scales with sample count whether or not the model does
    anything.
    """
    usable = RADIUS - MARGIN
    spacing = float(np.sqrt(np.pi * usable ** 2 / N_TARGET))
    n = int(np.floor(usable / spacing + 1e-9))
    ax = np.round(np.arange(-n, n + 1) * spacing, 5)
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    keep = np.hypot(X, Y) <= usable + 1e-9
    xs, ys = X[keep], Y[keep]
    out_dir = os.path.join(HERE, 'positions')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'circ_lm8_r0_positions.csv')
    with open(path, 'w') as f:
        f.write('x,y,theta\n')
        for x, y in zip(xs, ys):
            f.write(f'{x:g},{y:g},0.0\n')
    return path, len(xs), spacing


if __name__ == '__main__':
    for n in COUNTS:
        cov = n * PANEL / (2 * np.pi * RADIUS)
        print(f'circ_lm{n}_r0  spacing {2*np.pi*RADIUS/n:5.2f} m  '
              f'wall cover {100*cov:4.1f}%  -> {build_xml(n)}')
    p, cnt, sp = build_shared_grid()
    print(f'\nshared grid -> {p}\n  {cnt} positions, spacing {sp:.4f} m'
          f'  (radius {RADIUS}, margin {MARGIN})')
