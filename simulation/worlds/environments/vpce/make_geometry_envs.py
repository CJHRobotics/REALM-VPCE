"""Generate the rectangular and corridor arenas for the geometry sweep.

All three environments in the sweep hold **area** at ~314 m^2 (the circle's
pi * 10^2) and **landmark count** at 8, so geometry is the only thing that
varies:

    circ_lm8_r0    disc, r = 10          314.16 m^2   aspect 1.0
    rect_lm8_r0    20 x 15.708           314.16 m^2   aspect 1.27
    corr_lm8_r0    56 x 5.6              313.60 m^2   aspect 10.0

The rectangle also matches the circle's long dimension (20 m = the disc's
diameter), so it differs from the circle in shape alone.

Landmarks are placed at equal arc length around the perimeter, offset by half
a spacing, mirroring the circ_lm8_r0 convention (where landmark 0 sits at
22.5 deg = half of the 45 deg spacing). Perimeter is walked counter-clockwise
from the midpoint of the +x wall, which is the rectangular analogue of the
circle's theta = 0 start. Flags and colours are the same eight used by
circ_lm8_r0, so landmark identity is constant across the sweep.

theta is the direction from the landmark toward the interior, matching both
circ_lm8_r0 and corridor_lm10.

Position grids match circ_lm8_r0_positions.csv: 0.1 m lattice aligned to the
origin, 0.2 m margin from the wall, theta = 0.

    python simulation/worlds/environments/vpce/make_geometry_envs.py
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# The eight landmark identities from circ_lm8_r0, reused unchanged.
COLORS = [(1.00, 0.00, 0.00), (0.00, 1.00, 0.00), (0.00, 0.00, 1.00),
          (1.00, 1.00, 0.00), (0.00, 1.00, 1.00), (1.00, 0.50, 0.00),
          (0.50, 0.00, 0.50), (0.00, 0.50, 0.50)]

PANEL = 0.75          # landmark panel size, m -- same as circ_lm8_r0
WALL_H = 0.5
WALL_W = 0.012
GRID_STEP = 0.1       # position lattice, m
GRID_MARGIN = 0.2     # keep-out from the wall, m


def fmt(v, prec):
    """Match the leading-space alignment used by the existing world files."""
    s = f'{v:.{prec}f}'
    if s.startswith('-') and float(s) == 0.0:      # kill negative zero
        s = s[1:]
    return s if s.startswith('-') else ' ' + s


def perimeter_walk(hx, hy):
    """CCW segments of the rectangle boundary, starting at the +x wall midpoint.

    Returns [(x0, y0, x1, y1, length, theta_inward), ...]. Starting mid-wall
    rather than at a corner is what makes this the analogue of the circle's
    theta = 0, and it keeps landmarks off the corners for even counts.
    """
    return [
        (hx,  0.0,  hx,  hy,  hy,       np.pi),        # +x wall, lower half
        (hx,  hy,  -hx,  hy,  2 * hx,  -np.pi / 2),    # +y wall
        (-hx, hy,  -hx, -hy,  2 * hy,   0.0),          # -x wall
        (-hx, -hy,  hx, -hy,  2 * hx,   np.pi / 2),    # -y wall
        (hx, -hy,   hx,  0.0, hy,       np.pi),        # +x wall, upper half
    ]


def landmarks_on_perimeter(hx, hy, n):
    """n landmarks at equal arc length, offset half a spacing from the start."""
    segs = perimeter_walk(hx, hy)
    total = sum(s[4] for s in segs)
    spacing = total / n
    out = []
    for k in range(n):
        s = spacing * (k + 0.5)
        acc = 0.0
        for (x0, y0, x1, y1, ln, th) in segs:
            if s <= acc + ln or (x0, y0) == segs[-1][:2]:
                t = (s - acc) / ln
                out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), th))
                break
            acc += ln
    return out


def build_xml(name, hx, hy, n_landmarks, blurb):
    # XML forbids '--' inside a comment; the blurb goes into one.
    assert '--' not in blurb, blurb
    segs = perimeter_walk(hx, hy)
    area = 4 * hx * hy
    lines = [
        '<?xml version="1.0" encoding="us-ascii"?>\n\n',
        f'<!-- {blurb} -->\n',
        f'<!-- x in [{-hx:g}, {hx:g}] (long axis); '
        f'y in [{-hy:g}, {hy:g}] (short axis).  Area {area:.2f} m^2. -->\n',
        '<!-- Landmark theta = angle of the inward-facing surface normal. -->\n',
        f'<!-- {n_landmarks} landmarks at equal arc length around the '
        'perimeter, offset by half a spacing. -->\n\n',
        '<world>\n',
    ]
    # Four boundary walls, written as whole edges rather than the five
    # perimeter-walk segments (which split the +x wall at its midpoint).
    corners = [(hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        # 4 dp, not 2: the short-axis half-extent is 7.8540, and rounding
        # the wall to 7.85 would leave the landmarks 4 mm outside it.
        lines.append(f'    <wall x1="{fmt(x0,4)}" y1="{fmt(y0,4)}" '
                     f'x2="{fmt(x1,4)}" y2="{fmt(y1,4)}" type="boundary" '
                     f'height="{WALL_H}" width="{WALL_W}"/>\n')
    lines.append('\n')

    for k, (x, y, th) in enumerate(landmarks_on_perimeter(hx, hy, n_landmarks)):
        r, g, b = COLORS[k % len(COLORS)]
        lines.append(
            f'    <landmark type="panel" x="{fmt(x,4)}" y="{fmt(y,4)}" '
            f'theta="{fmt(th,6)}" height="{PANEL}" width="{PANEL}"\n'
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
    return path, area


def build_grid(name, hx, hy):
    """0.1 m lattice aligned to the origin, 0.2 m clear of every wall."""
    lim_x, lim_y = hx - GRID_MARGIN, hy - GRID_MARGIN
    # Round the limit down onto the lattice, with a tolerance so a point
    # landing exactly on it is not lost to floating point.
    nx = int(np.floor(lim_x / GRID_STEP + 1e-9))
    ny = int(np.floor(lim_y / GRID_STEP + 1e-9))
    xs = np.round(np.arange(-nx, nx + 1) * GRID_STEP, 4)
    ys = np.round(np.arange(-ny, ny + 1) * GRID_STEP, 4)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    path = os.path.join(HERE, 'positions', name + '_positions.csv')
    with open(path, 'w') as f:
        f.write('x,y,theta\n')
        for x, y in zip(X.ravel(), Y.ravel()):
            f.write(f'{x:g},{y:g},0.0\n')
    return path, X.size


SPECS = [
    ('rect_lm8_r0', 10.0, 15.70796 / 2, 8,
     'Rectangular arena 20 m x 15.708 m: same area as circ_lm8_r0 '
     '(314.16 m^2) and the same long dimension (20 m = the disc diameter), '
     'so shape is the only difference.'),
    ('corr_lm8_r0', 28.0, 2.8, 8,
     'Corridor 56 m x 5.6 m: same area as circ_lm8_r0 to within 0.2%, at '
     'aspect 10:1. The elongated extreme of the geometry sweep.'),
]

if __name__ == '__main__':
    for name, hx, hy, n, blurb in SPECS:
        xml_path, area = build_xml(name, hx, hy, n, blurb)
        csv_path, npts = build_grid(name, hx, hy)
        print(f'{name}: {2*hx:g} x {2*hy:g} m  area {area:.2f} m^2  '
              f'aspect {max(hx,hy)/min(hx,hy):.2f}:1')
        print(f'  {xml_path}')
        print(f'  {csv_path}  ({npts} points)')
