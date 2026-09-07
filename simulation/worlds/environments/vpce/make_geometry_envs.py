"""Generate the rectangular and corridor arenas for the geometry sweep.

All three environments in the sweep hold **area** at ~28.3 m^2 (the circle's
pi * 3^2) and **landmark count** at 8, so geometry is the only thing that
varies:

    circ_lm8_r0    disc, r = 3           28.27 m^2   aspect 1.0
    rect_lm8_r0    6 x 4.712             28.27 m^2   aspect 1.27
    corr_lm8_r0    16.8 x 1.68           28.22 m^2   aspect 10.0

The rectangle also matches the circle's long dimension (6 m = the disc's
diameter), so it differs from the circle in shape alone. Its aspect is
therefore exactly 4/pi, as it was at the previous scale.

Scale follows the disc. These were built at 314 m^2 to match circ_lm8_r0 when
that arena had r = 10; the landmark sweep then dropped to r = 3, the scale the
model was actually tuned at, and left the geometry arenas eleven times larger
than the disc they are supposed to be a shape control for. Every dimension
here is the old one times 0.3, which preserves both aspect ratios exactly and
keeps the corridor's rounding as it was (16.8 x 1.68 sits 0.18% under the
disc's area, as 56 x 5.6 did).

**Sampling density must NOT be held constant**, and this is the part the
rescale cannot skip. These grids used a fixed 0.1 m lattice, which was right
at 314 m^2 (30,142 points) and is ruinous at 28 m^2: the same lattice gives
about 2,700 positions, and the agglomeration builds its candidate fields out
of the sampled positions. Sample count is not a free parameter -- at 6,000
positions the hog channel produces zero fields, every candidate exceeding the
Rule 9 ceiling. The spacing is therefore solved to hold ~N_TARGET positions,
matching make_area_sweep_envs and make_landmark_sweep_envs, so N is not a
variable anywhere in the series:

    rect   spacing 0.0283 m   30,141 positions
    corr   spacing 0.0264 m   30,429 positions

Two properties of this scale that belong in the interpretation rather than in
the geometry:

**The corridor is now narrow in robot units.** 1.68 m wide, less 0.2 m of
keep-out at each wall, leaves 1.28 m of walkable width against the robot's
0.31 m circumscribing radius -- about two body widths. That is the honest
consequence of asking for aspect 10:1 at 28 m^2, and it is the same trade the
area sweep accepts at r = 1.25, but the corridor is no longer a room the robot
moves freely in.

**Landmark wall coverage is not constant across the sweep.** Eight 0.75 m
panels cover 31.8% of the disc's circumference, 28.0% of the rectangle's
perimeter and 16.2% of the corridor's, because perimeter grows with elongation
at fixed area. The 2x spread between disc and corridor is unchanged from the
314 m^2 build, but the absolute coverage is now high enough to matter. Holding
count, area and coverage at once is not possible; count is held, and coverage
is reported.

Landmarks are placed at equal arc length around the perimeter, offset by half
a spacing, mirroring the circ_lm8_r0 convention (where landmark 0 sits at
22.5 deg = half of the 45 deg spacing). Perimeter is walked counter-clockwise
from the midpoint of the +x wall, which is the rectangular analogue of the
circle's theta = 0 start. Flags and colours are the same eight used by
circ_lm8_r0, so landmark identity is constant across the sweep.

theta is the direction from the landmark toward the interior, matching both
circ_lm8_r0 and corridor_lm10.

    python simulation/worlds/environments/vpce/make_geometry_envs.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_area_sweep_envs import MARGIN, N_TARGET

# The eight landmark identities from circ_lm8_r0, reused unchanged.
COLORS = [(1.00, 0.00, 0.00), (0.00, 1.00, 0.00), (0.00, 0.00, 1.00),
          (1.00, 1.00, 0.00), (0.00, 1.00, 1.00), (1.00, 0.50, 0.00),
          (0.50, 0.00, 0.50), (0.00, 0.50, 0.50)]

PANEL = 0.75          # landmark panel size, m -- same as circ_lm8_r0
WALL_H = 0.5
WALL_W = 0.012

# The disc these two are the shape control for. Every dimension below is
# derived from it, so the sweep cannot drift out of scale again the way it did
# when the landmark sweep moved from r = 10 to r = 3.
RADIUS = 3.0
AREA = np.pi * RADIUS ** 2

# MARGIN (wall keep-out) and N_TARGET (positions per arena) come from
# make_area_sweep_envs so there is one definition of each across the series.
# There is deliberately no GRID_STEP: a fixed lattice is what broke these
# grids at this scale, and build_grid solves the spacing instead.


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


HALF_THICK = 0.015     # RectangularPanel is 0.03 m thick along its own X,
                       # which theta points inward.


def landmarks_on_perimeter(hx, hy, n):
    """n landmarks at equal arc length, offset half a spacing from the start.

    Each is inset from the wall by half the panel's thickness, so the box's
    outer face is flush with the wall rather than half buried in it. On a
    straight wall that is the whole correction; the curved arenas need more,
    since a flat chord's ends reach past the radius its centre sits at.
    """
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
                px = x0 + t * (x1 - x0) + HALF_THICK * np.cos(th)
                py = y0 + t * (y1 - y0) + HALF_THICK * np.sin(th)
                out.append((px, py, th))
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
    """Lattice spaced to give ~N_TARGET positions, MARGIN clear of every wall.

    Spacing is solved rather than fixed, for the reason in the module
    docstring: field count scales with sample count whether or not the model
    does anything, so holding N is what makes arenas of different area and
    shape comparable. Collect dense and decimate in analysis -- a
    constant-density design is recoverable from these grids by subsampling,
    but a constant-N design is not recoverable from constant-density ones.
    """
    ux, uy = hx - MARGIN, hy - MARGIN
    spacing = float(np.sqrt(4.0 * ux * uy / N_TARGET))
    # Round the limit down onto the lattice, with a tolerance so a point
    # landing exactly on it is not lost to floating point.
    nx = int(np.floor(ux / spacing + 1e-9))
    ny = int(np.floor(uy / spacing + 1e-9))
    xs = np.round(np.arange(-nx, nx + 1) * spacing, 5)
    ys = np.round(np.arange(-ny, ny + 1) * spacing, 5)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    # positions/ is tracked (515c167), but a grid for a newly added arena has
    # no file yet -- create the directory rather than assuming it.
    out_dir = os.path.join(HERE, 'positions')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + '_positions.csv')
    with open(path, 'w') as f:
        f.write('x,y,theta\n')
        for x, y in zip(X.ravel(), Y.ravel()):
            f.write(f'{x:g},{y:g},0.0\n')
    return path, X.size, spacing


# Derived from RADIUS rather than written out, so the two arenas follow the
# disc automatically if it moves again.
#
#   rect  long dimension = the disc's diameter, so shape is the only
#         difference from the circle; the short side then falls out of the
#         area constraint and the aspect is exactly 4/pi.
#   corr  aspect 10:1 at the same area, rounded to a whole centimetre the way
#         56 x 5.6 was, which lands 0.18% under the disc.
RECT_HX = RADIUS                        # half of the 6 m long dimension
RECT_HY = AREA / (4.0 * RECT_HX)        # 2.35619
CORR_HY = round(np.sqrt(AREA / 10.0), 2) / 2.0    # 1.68 m wide
CORR_HX = 10.0 * CORR_HY                          # aspect 10:1 -> 16.8 m long

SPECS = [
    ('rect_lm8_r0', RECT_HX, RECT_HY, 8,
     f'Rectangular arena {2*RECT_HX:g} m x {2*RECT_HY:.3f} m: same area as '
     f'circ_lm8_r0 ({AREA:.2f} m^2) and the same long dimension '
     f'({2*RECT_HX:g} m = the disc diameter), so shape is the only '
     f'difference.'),
    ('corr_lm8_r0', CORR_HX, CORR_HY, 8,
     f'Corridor {2*CORR_HX:g} m x {2*CORR_HY:g} m: same area as circ_lm8_r0 '
     f'to within 0.2%, at aspect 10:1. The elongated extreme of the geometry '
     f'sweep. Walkable width is {2*(CORR_HY-MARGIN):.2f} m, about two robot '
     f'body widths.'),
]

if __name__ == '__main__':
    for name, hx, hy, n, blurb in SPECS:
        xml_path, area = build_xml(name, hx, hy, n, blurb)
        csv_path, npts, spacing = build_grid(name, hx, hy)
        cover = 100 * n * PANEL / (4 * (hx + hy))
        print(f'{name}: {2*hx:g} x {2*hy:g} m  area {area:.2f} m^2  '
              f'aspect {max(hx,hy)/min(hx,hy):.2f}:1  wall cover {cover:.1f}%')
        print(f'  {xml_path}')
        print(f'  {csv_path}  ({npts} points, spacing {spacing:.4f} m)')
