"""Generate the rectangular and corridor arenas that sit beside the area sweep.

Two arenas, two different jobs. They are no longer both area controls.

    rect_lm8_r0   6 x 4.712 m    28.27 m^2   aspect 1.27   SHAPE control
    corr_lm8_r0   10 x 2 m       20.00 m^2   aspect 5.0    ELIAV comparison

**The rectangle is the shape control.** Same area as circ_lm8_r0, the small
end of the area sweep, and the same long dimension (6 m = the disc diameter),
so it differs from that disc in shape alone. Its aspect is exactly 4/pi. Pair
it against the disc and any difference is boundary geometry rather than
enclosure size.

**The corridor is the Eliav comparison, and is deliberately NOT area-matched.**
Eliav et al. 2021 report field sizes along a 200 m tunnel, and -- the control
that matters more here -- along a 6 m segment of the same tunnel, where mean
field size fell from 5.9 m to 1.5 m and the within-neuron size ratio from 4.4
to 1.6. Their claim is that multiscale coding is a property of a large space
rather than of the hippocampus. A 10 x 2 m corridor is the closest thing this
series can offer to that short segment: long enough to be read as
one-dimensional, short enough to be the small case. Matching it to a disc's
area would serve a different question and is the rectangle's job.

At 10 x 2 m the corridor is also a usable space for the robot, which the
previous 16.8 x 1.68 m build was not: 1.6 m of walkable width after the 0.2 m
keep-out at each wall, against a 0.31 m circumscribing radius -- about 2.6
body widths rather than 2.

Eight 0.75 m panels cover 25% of the 24 m perimeter, between the 31.8% of the
r = 3 disc and the 15.9% of r = 6, so cue availability is in the same range as
the sweep rather than an outlier.

**Sampling density is not held constant; sample COUNT is.** The spacing is
solved to put ~N_TARGET positions in every arena, matching
make_area_sweep_envs and make_landmark_sweep_envs, because field count scales
with sample count whether or not the model does anything. The corridor lands
at 0.0226 m spacing and 30,175 positions.

Landmarks are placed at equal arc length around the perimeter, offset by half
a spacing, mirroring the circ_lm8_r0 convention. The perimeter is walked
counter-clockwise from the midpoint of the +x wall, which is the rectangular
analogue of the circle's theta = 0. Flags and colours are the same eight used
by circ_lm8_r0, so landmark identity is constant across the series. theta is
the direction from the landmark toward the interior.

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


# The rectangle follows the disc; the corridor does not.
#
#   rect  long dimension = the disc's diameter, so shape is the only
#         difference from circ_lm8_r0; the short side falls out of the area
#         constraint and the aspect is exactly 4/pi.
#   corr  a fixed 10 x 2 m. Its job is the Eliav comparison, not an area
#         control, so it is written out rather than derived -- tying it to the
#         disc would move it every time the sweep moved and would answer the
#         rectangle's question instead of its own.
RECT_HX = RADIUS                        # half of the 6 m long dimension
RECT_HY = AREA / (4.0 * RECT_HX)        # 2.35619
CORR_L, CORR_W = 10.0, 2.0

SPECS = [
    ('rect_lm8_r0', RECT_HX, RECT_HY, 8,
     f'Rectangular arena {2*RECT_HX:g} m x {2*RECT_HY:.3f} m: same area as '
     f'circ_lm8_r0 ({AREA:.2f} m^2) and the same long dimension '
     f'({2*RECT_HX:g} m = the disc diameter), so shape is the only '
     f'difference. The shape control for the small end of the area sweep.'),
    ('corr_lm8_r0', CORR_L / 2.0, CORR_W / 2.0, 8,
     f'Corridor {CORR_L:g} m x {CORR_W:g} m, area {CORR_L*CORR_W:.2f} m^2, '
     f'aspect {CORR_L/CORR_W:g}:1. The Eliav comparison: long enough to read '
     f'as one-dimensional, short enough to stand for the 6 m tunnel segment '
     f'in which mean field size fell from 5.9 m to 1.5 m. Deliberately not '
     f'area-matched to any disc; that is the rectangle\'s job. Walkable '
     f'width {CORR_W - 2*MARGIN:.2f} m, about 2.6 robot body widths.'),
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
