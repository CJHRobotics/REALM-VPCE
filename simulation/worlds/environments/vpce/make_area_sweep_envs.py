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

**The small end is set by the robot.** Eight 0.25 m panels occupy 2 m of wall,
which is 26% of the circumference at r = 1.25 and still only 42% at r = 0.75 --
so landmark crowding is not the binding constraint. What binds is the robot:
its circumscribing radius is 0.31 m, and the 0.2 m wall keep-out leaves a
walkable disc of only 1.05 m radius at r = 1.25. Below that the robot occupies
an appreciable fraction of the arena it is meant to be mapping.

Note the cost at the other end. A 0.25 m panel seen from across the r = 10
arena spans roughly 4 px of a 224 px image, against 11 px for the same panel
at r = 3.5. Landmark visibility therefore falls as the arena grows, and the
sweep confounds enclosure size with cue salience. That is a genuine property of
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

COLORS = [(1.00, 0.00, 0.00), (0.00, 1.00, 0.00), (0.00, 0.00, 1.00),
          (1.00, 1.00, 0.00), (0.00, 1.00, 1.00), (1.00, 0.50, 0.00),
          (0.50, 0.00, 0.50), (0.00, 0.50, 0.50)]

N_LANDMARKS = 8
PANEL = 0.50          # fixed physical size -- the point of this sweep
WALL_H = 0.5
WALL_THICK = 0.5
SUBDIV = 128
MARGIN = 0.2          # keep-out from the wall, absolute (a physical robot
                      # clearance does not shrink with the room)
N_TARGET = 30147      # matches circ_lm8_r0, so N is not a variable

# (name, radius). r = 10 is circ_lm8_r0, already built and collected.
SPECS = [('circ_lm8_rad1p25', 1.25),
         ('circ_lm8_rad2p0', 2.0),
         ('circ_lm8_rad3p5', 3.5),
         ('circ_lm8_rad6p0', 6.0)]


def fmt(v, prec):
    s = f'{v:.{prec}f}'
    if s.startswith('-') and float(s) == 0.0:
        s = s[1:]
    return s if s.startswith('-') else ' ' + s


def build_xml(name, radius):
    """Same landmark convention as circ_lm8_r0: angle = pi/n + k*2pi/n, so the
    first panel sits half a spacing off the +x axis; theta points inward."""
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
    for k in range(N_LANDMARKS):
        a = np.pi / N_LANDMARKS + k * 2 * np.pi / N_LANDMARKS
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


def build_r0_grid():
    """The r = 10 grid, shared by every circ_lm*_r0 arena.

    circ_lm8_r0 predates these generators: its world file is hand-written and
    its grid was produced by a one-off script that was never kept. Since the
    positions directory is gitignored in full, a fresh clone has no way to
    rebuild it -- and circ_lm4/6/8/10_r0 all resolve to this one file through
    the fallback in collect_data.py, so all four collections fail without it.

    Fixed 0.1 m spacing rather than the constant-N spacing used above: this
    grid is the established one, and at r = 10 the two agree anyway
    (~30,150 against a 30,147 target).
    """
    radius, spacing = 10.0, 0.1
    usable = radius - MARGIN
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
    print(f'{"env":18s} {"radius":>7} {"area":>9} {"wall cov":>9} '
          f'{"spacing":>8} {"N":>7} {"diam/body":>10}')
    for name, radius in SPECS:
        xml_path, area, cover = build_xml(name, radius)
        csv_path, n, spacing = build_grid(name, radius)
        print(f'{name:18s} {radius:7.2f} {area:8.2f}m2 {100*cover:8.1f}% '
              f'{spacing:8.4f} {n:7d} {2*radius/0.2:10.0f}')
    p, n, sp = build_r0_grid()
    print(f'{"circ_lm8_r0":18s} {10.0:7.2f} {314.16:8.2f}m2 '
          f'{100*8*PANEL/(2*np.pi*10):8.1f}% {sp:8.4f} {n:7d} {100:10.0f}')
    print(f'\n  circ_lm8_r0 grid -> {p}')
    print('  (shared by circ_lm4/6/8/10_r0 through the collect_data fallback)')
