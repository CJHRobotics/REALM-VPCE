"""Circular arenas for the environment-size sweep (experiment 3).

Replicates Harland et al. 2021's four-environment design -- number of fields,
summed field area, and fractional coverage against enclosure area -- with the
landmarks held at a FIXED physical size, as their room cues were.

    circ_lm8_rad1p25    r = 1.25     4.91 m2   12.5 robot lengths across
    circ_lm8_rad2p0     r = 2.0     12.57 m2     20 robot lengths across
    circ_lm8_rad3p5     r = 3.5     38.48 m2     35
    circ_lm8_rad6p0     r = 6.0    113.10 m2     60
    circ_lm8_r0         r = 10.0   314.16 m2    100     (already collected)

Two design constraints set the endpoints, and neither is negotiable.

**The small end is set by the landmarks and the robot together.** Eight 0.75 m
panels occupy 6 m of wall: 76% of the circumference at r = 1.25 and 48% at
r = 2.0, so the smallest arena is closer to a ring of flags than to a room
with landmarks in it. The robot binds too -- its circumscribing radius is
0.31 m and the 0.2 m keep-out leaves a walkable disc of 1.05 m at r = 1.25.

Note the cost at the other end. A 0.75 m panel seen from across the r = 10
arena spans roughly 11 px of a 224 px image, against 33 px at r = 3.5.
Landmark visibility therefore falls as the arena grows, and the sweep
confounds enclosure size with cue salience. That is a genuine property of
fixed-size cues rather than a defect -- distant cues do subtend less -- but it
belongs in the interpretation.

**Sampling density must NOT be held constant.** At the usual 0.1 m spacing the
four arenas would carry 1,018 / 3,421 / 10,568 / 30,172 positions -- a 30-fold
range that tracks area exactly. Since the agglomeration builds its candidate
fields out of the sampled positions, field count would then scale with area
whether or not the model does anything, manufacturing Harland's headline result
from the sampling design. Every arena is therefore sampled at ~30,000 positions
and the spacing scales with radius instead.

Collect dense and decimate in analysis: a constant-density design is recoverable
from these grids by subsampling, but a constant-N design is not recoverable from
constant-density grids.

    python simulation/worlds/environments/vpce/make_area_sweep_envs.py
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Twelve, enough for circ_lm12_r0. get_landmark_observations identifies a
# landmark by matching its recognition colour, so these must stay mutually
# distinguishable; the extras are appended rather than inserted so every
# arena with fewer landmarks keeps the colours it already had.
COLORS = [(1.00, 0.00, 0.00), (0.00, 1.00, 0.00), (0.00, 0.00, 1.00),
          (1.00, 1.00, 0.00), (0.00, 1.00, 1.00), (1.00, 0.50, 0.00),
          (0.50, 0.00, 0.50), (0.00, 0.50, 0.50), (0.80, 0.20, 0.60),
          (0.20, 0.60, 0.20), (1.00, 0.00, 1.00), (0.60, 0.30, 0.10)]

N_LANDMARKS = 8
PANEL = 0.75          # fixed physical size -- the point of this sweep
WALL_H = 0.5
WALL_THICK = 0.5
SUBDIV = 128
MARGIN = 0.2          # keep-out from the wall, absolute (a physical robot
                      # clearance does not shrink with the room)
N_TARGET = 30147      # matches circ_lm8_r0, so N is not a variable

# (name, radius). r = 10 is circ_lm8_r0, already built and collected.
# rad10p0 replaces circ_lm8_r0 as the largest member: that arena is now
# radius 3, belonging to the landmark sweep, so the area sweep carries its
# own r = 10 disc rather than borrowing one.
SPECS = [('circ_lm8_rad1p25', 1.25),
         ('circ_lm8_rad2p0', 2.0),
         ('circ_lm8_rad3p5', 3.5),
         ('circ_lm8_rad6p0', 6.0),
         ('circ_lm8_rad10p0', 10.0)]


def landmark_angles(n):
    """Landmark bearings, anchored at 90 degrees and evenly spaced.

    The robot captures at headings 0, 45, ... 315 with a 45-degree camera, so
    the eight views partition the circle at 22.5 + 45k degrees. The previous
    convention -- a half-spacing offset, pi/n + 2*pi*k/n -- put N=8 landmarks
    at exactly those boundaries, splitting every panel across two views, and
    left only 4 of the 8 headings containing a landmark at all. A heading
    whose view holds no landmark returns the same floor-and-wall colour
    histogram everywhere, carrying no position information.

    Measured effect on the colour channel, which depends on this most:

        env   sectors occupied   admitted colour fields
        lm4        4/8                    7
        lm6        6/8                   28
        lm8        4/8 (on edges)         4
        lm10       8/8                  373

    Anchoring at 90 degrees puts every landmark inside a view for all four
    counts, and for N=8 centres one in each of the eight. It cannot fix the
    deeper limit: 4 or 6 landmarks cannot occupy 8 sectors, so lm4 and lm6
    still leave headings empty by construction.
    """
    return np.pi / 2 + np.arange(n) * 2 * np.pi / n


def fmt(v, prec):
    s = f'{v:.{prec}f}'
    if s.startswith('-') and float(s) == 0.0:
        s = s[1:]
    return s if s.startswith('-') else ' ' + s


def build_xml(name, radius):
    """Same landmark convention as circ_lm8_r0: bearings anchored at 90
    degrees and evenly spaced, off the camera's view seams. theta points
    inward. See landmark_angles."""
    area = np.pi * radius ** 2
    cover = N_LANDMARKS * PANEL / (2 * np.pi * radius)
    lines = [
        '<?xml version="1.0" encoding="us-ascii"?>\n\n',
        f'<!-- Circular arena, radius {radius:g} m, area {area:.2f} m^2. Member of\n'
        f'     the environment-size sweep; landmarks are held at a fixed physical\n'
        f'     size ({PANEL} m) across the sweep, so they occupy {100*cover:.0f}% of this\n'
        f'     arena\'s wall against 9.5% at r = 10. Diameter is {2*radius/0.2:.0f} robot\n'
        f'     body lengths. -->\n\n',
        '<world>\n',
        f'\t<circular_wall radius="{radius:.1f}" height="{WALL_H}" '
        f'thickness="{WALL_THICK}" subdivision="{SUBDIV}"/>\n\n',
    ]
    for k, a in enumerate(landmark_angles(N_LANDMARKS)):
        x, y, th = radius * np.cos(a), radius * np.sin(a), a - np.pi
        r, g, b = COLORS[k]
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
    path = os.path.join(HERE, name + '.xml')
    with open(path, 'w') as f:
        f.writelines(lines)
    return path, area, cover


def build_grid(name, radius):
    """Lattice spaced to give ~N_TARGET positions, so sample count is held
    constant across the sweep rather than scaling with area."""
    usable = radius - MARGIN
    spacing = float(np.sqrt(np.pi * usable ** 2 / N_TARGET))
    n = int(np.floor(usable / spacing + 1e-9))
    ax = np.round(np.arange(-n, n + 1) * spacing, 5)
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    keep = np.hypot(X, Y) <= usable + 1e-9
    xs, ys = X[keep], Y[keep]

    out_dir = os.path.join(HERE, 'positions')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + '_positions.csv')
    with open(path, 'w') as f:
        f.write('x,y,theta\n')
        for x, y in zip(xs, ys):
            f.write(f'{x:g},{y:g},0.0\n')
    return path, len(xs), spacing


if __name__ == '__main__':
    print(f'{"env":18s} {"radius":>7} {"area":>9} {"wall cov":>9} '
          f'{"spacing":>8} {"N":>7} {"diam/body":>10}')
    for name, radius in SPECS:
        xml_path, area, cover = build_xml(name, radius)
        csv_path, n, spacing = build_grid(name, radius)
        print(f'{name:18s} {radius:7.2f} {area:8.2f}m2 {100*cover:8.1f}% '
              f'{spacing:8.4f} {n:7d} {2*radius/0.2:10.0f}')
