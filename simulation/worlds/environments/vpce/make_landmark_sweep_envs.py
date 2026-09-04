"""Circular arenas for the landmark-count sweep: circ_lm{4,6,8,10}_r0.

All four are the same disc -- radius 10, area 314.16 m^2 -- differing only in
how many 0.75 m panels sit on the wall, which sets interlandmark spacing from
15.7 m (lm4) to 6.3 m (lm10).

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
                                  fmt, landmark_angles, build_r0_grid)

RADIUS = 10.0
COUNTS = [4, 6, 8, 10]


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


if __name__ == '__main__':
    for n in COUNTS:
        print(f'circ_lm{n}_r0  spacing {2*np.pi*RADIUS/n:5.2f} m  '
              f'-> {build_xml(n)}')
    p, cnt, sp = build_r0_grid()
    print(f'\nshared grid -> {p}  ({cnt} positions, spacing {sp} m)')
