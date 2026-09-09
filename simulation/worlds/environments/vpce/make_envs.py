"""Generate every environment this experiment uses. One file, four arenas.

    circ_lm_8_r3      disc, r = 3       28.27 m^2   small
    circ_lm_8_r6      disc, r = 6      113.10 m^2   medium
    circ_lm_8_r10     disc, r = 10     314.16 m^2   mega
    corr_lm_8_l10w2   10 x 2 m          20.00 m^2   Eliav comparison

Naming is `<shape>_lm_<landmarks>_<size>`, where size is `r<radius>` for a
disc and `l<length>w<width>` for a box. It replaces three overlapping
generators (area sweep, landmark sweep, geometry) that between them produced
ten arenas under three incompatible conventions, most of which no experiment
used any more.

**The three discs are the area sweep.** They span 11.1x in area against the
8.8x between Harland et al. 2021's smallest and largest enclosures, at fixed
shape and fixed landmark count, so enclosure size is the only thing that
varies between them.

**The corridor is the Eliav comparison, not an area control.** Eliav et al.
2021 report field sizes along a 200 m tunnel and, more usefully, along a 6 m
segment of it, where mean field size fell from 5.9 m to 1.5 m and the
within-neuron size ratio from 4.4 to 1.6 -- their evidence that a spread of
scales belongs to a large space rather than to the hippocampus. A 10 x 2 m
corridor is the nearest this series gets to that short segment: long enough to
read as one-dimensional, short enough to be the small case. Its dimensions are
written out rather than derived from a disc, because tying it to one would
answer a different question.

Two constraints govern every arena here.

**Landmarks are a FIXED physical size** (0.75 m), as Harland's room cues were.
That means they occupy a smaller share of the wall as the arena grows -- 32%
of the circumference at r = 3, 16% at r = 6, 10% at r = 10 -- so enclosure
size is unavoidably confounded with how prominent the cues are. It is a
genuine property of fixed-size cues rather than a defect, but it belongs in
the interpretation, and it is worst at r = 10: a 0.75 m panel spans roughly
11 px of a 224 px image from across that disc.

**Sampling COUNT is held constant, not density.** Every arena carries about
N_TARGET positions, and the lattice spacing is solved to achieve that. Field
count scales with sample count whether or not the model does anything, so a
constant-density design would manufacture a size-with-area result from the
sampling alone. Collect dense and decimate in analysis: a constant-density
design is recoverable from these grids by subsampling, a constant-N design is
not recoverable from constant-density ones.

Panels are mounted flush against the inner wall face. On a curve that needs an
inset of sqrt(R^2 - (w/2)^2) - HALF_THICK, because a flat panel centred at the
wall radius buries its own corners in the wall -- badly at small radii.

    python simulation/worlds/environments/vpce/make_envs.py
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Eight landmark identities. get_landmark_observations identifies a landmark
# by matching its recognition colour, so these must stay distinguishable.
COLORS = [(1.00, 0.00, 0.00), (0.00, 1.00, 0.00), (0.00, 0.00, 1.00),
          (1.00, 1.00, 0.00), (0.00, 1.00, 1.00), (1.00, 0.50, 0.00),
          (0.50, 0.00, 0.50), (0.00, 0.50, 0.50)]

N_LANDMARKS = 8
PANEL       = 0.75      # fixed physical cue size, m
WALL_H      = 0.5
WALL_THICK  = 0.5       # circular wall
WALL_W      = 0.012     # box wall
SUBDIV      = 128
MARGIN      = 0.2       # collection keep-out from the wall, m
N_TARGET    = 30147     # positions per arena, held constant
HALF_THICK  = 0.015     # half a panel's own thickness

RADII       = [3.0, 6.0, 10.0]
CORR_L, CORR_W = 10.0, 2.0


def fmt(v, prec):
    """Leading-space alignment, matching the existing world files."""
    s = f'{v:.{prec}f}'
    if s.startswith('-') and float(s) == 0.0:      # kill negative zero
        s = s[1:]
    return s if s.startswith('-') else ' ' + s


def landmark_angles(n):
    """Bearings anchored at 90 degrees, so no panel sits on a camera seam.

    The robot captures at headings 0, 45 ... 315 with a 45-degree camera, so
    the views partition the circle at 22.5 + 45k. Anchoring at 90 keeps every
    panel inside a view rather than across the boundary between two.
    """
    return (np.pi / 2.0) + np.arange(n) * (2.0 * np.pi / n)


def wall_mount_radius(radius, width=PANEL):
    """Centre radius that puts a panel's outer corners on the wall's face.

    A panel is a flat chord on a curved wall. Centred at the wall radius R,
    every point but the tangent point lies beyond R and its ends bury
    themselves in the wall -- half the panel at R = 1.25. Pulling the centre in
    to sqrt(R^2 - (w/2)^2) - HALF_THICK puts the outermost corner exactly on
    the inner face, so the panel is flush and fully visible at any radius.
    """
    return float(np.sqrt(radius ** 2 - (width / 2) ** 2) - HALF_THICK)


def lattice(usable_area, half_extents):
    """Spacing solved for ~N_TARGET positions, and the resulting axis counts."""
    spacing = float(np.sqrt(usable_area / N_TARGET))
    return spacing, [int(np.floor(h / spacing + 1e-9)) for h in half_extents]


# ------------------------------------------------------------------ circles

def build_disc(radius):
    name = f'circ_lm_{N_LANDMARKS}_r{radius:g}'
    a_deg = np.degrees(landmark_angles(N_LANDMARKS))
    area = np.pi * radius ** 2
    cover = 100 * N_LANDMARKS * PANEL / (2 * np.pi * radius)
    lines = [
        '<?xml version="1.0" encoding="us-ascii"?>\n\n',
        f'<!-- Circular arena, radius {radius:g} m, area {area:.2f} m^2, with '
        f'{N_LANDMARKS} landmarks of {PANEL} m.\n'
        f'     XML `radius` is the inner walkable radius; the wall material\n'
        f'     extends outward by `thickness`. Panels sit flush on the inner\n'
        f'     face. Interlandmark spacing along the wall is '
        f'{2*np.pi*radius/N_LANDMARKS:.2f} m; the panels cover {cover:.1f}% of\n'
        f'     the circumference. Landmark bearings are anchored at 90 deg,\n'
        f'     off the camera view seams: {np.round(a_deg, 1).tolist()} -->\n\n',
        '<world>\n',
        f'\t<circular_wall radius="{radius:g}" height="{WALL_H}" '
        f'thickness="{WALL_THICK}" subdivision="{SUBDIV}"/>\n\n',
    ]
    r_mount = wall_mount_radius(radius)
    for k, a in enumerate(landmark_angles(N_LANDMARKS)):
        x, y, th = r_mount * np.cos(a), r_mount * np.sin(a), a - np.pi
        r, g, b = COLORS[k % len(COLORS)]
        lines.append(
            f'    <landmark type="panel" x="{fmt(x,4)}" y="{fmt(y,4)}" '
            f'theta="{fmt(th,4)}" height="{PANEL}" width="{PANEL}"\n'
            f'              texture="../protos/world_objects/textures/flags/flag_{k}.png"\n'
            f'              red="{r:.2f}" green="{g:.2f}" blue="{b:.2f}"/>\n')
    lines.append(TAIL)
    path = os.path.join(HERE, name + '.xml')
    with open(path, 'w') as f:
        f.writelines(lines)

    usable = radius - MARGIN
    spacing, (n,) = lattice(np.pi * usable ** 2, [usable])
    ax = np.round(np.arange(-n, n + 1) * spacing, 5)
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    keep = np.hypot(X, Y) <= usable + 1e-9
    npts = write_grid(name, X[keep], Y[keep])
    return name, area, npts, spacing, cover


# ----------------------------------------------------------------- corridor

def perimeter_walk(hx, hy):
    """CCW boundary segments from the +x wall midpoint.

    Starting mid-wall rather than at a corner is the analogue of the circle's
    theta = 0, and it keeps landmarks off the corners for even counts.
    """
    return [
        (hx,  0.0,  hx,  hy,  hy,       np.pi),        # +x wall, lower half
        (hx,  hy,  -hx,  hy,  2 * hx,  -np.pi / 2),    # +y wall
        (-hx, hy,  -hx, -hy,  2 * hy,   0.0),          # -x wall
        (-hx, -hy,  hx, -hy,  2 * hx,   np.pi / 2),    # -y wall
        (hx, -hy,   hx,  0.0, hy,       np.pi),        # +x wall, upper half
    ]


def build_box(length, width):
    name = f'corr_lm_{N_LANDMARKS}_l{length:g}w{width:g}'
    hx, hy = length / 2.0, width / 2.0
    area = length * width
    per = 2 * (length + width)
    cover = 100 * N_LANDMARKS * PANEL / per
    corners = [(hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]
    lines = [
        '<?xml version="1.0" encoding="us-ascii"?>\n\n',
        f'<!-- Corridor {length:g} m x {width:g} m, area {area:.2f} m^2, '
        f'aspect {length/width:g}:1, with {N_LANDMARKS} landmarks of {PANEL} m\n'
        f'     covering {cover:.1f}% of the {per:g} m perimeter. The Eliav\n'
        f'     comparison: long enough to read as one-dimensional, short\n'
        f'     enough to stand for the 6 m tunnel segment. Walkable width\n'
        f'     {width - 2*MARGIN:.2f} m after the keep-out. Landmark theta is the\n'
        f'     inward surface normal; panels are at equal arc length around\n'
        f'     the perimeter, offset by half a spacing. -->\n\n',
        '<world>\n',
    ]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        lines.append(f'    <wall x1="{fmt(x0,4)}" y1="{fmt(y0,4)}" '
                     f'x2="{fmt(x1,4)}" y2="{fmt(y1,4)}" type="boundary" '
                     f'height="{WALL_H}" width="{WALL_W}"/>\n')
    lines.append('\n')

    segs = perimeter_walk(hx, hy)
    spacing_lm = sum(s[4] for s in segs) / N_LANDMARKS
    for k in range(N_LANDMARKS):
        s, acc = spacing_lm * (k + 0.5), 0.0
        for (x0, y0, x1, y1, ln, th) in segs:
            if s <= acc + ln or (x0, y0) == segs[-1][:2]:
                t = (s - acc) / ln
                px = x0 + t * (x1 - x0) + HALF_THICK * np.cos(th)
                py = y0 + t * (y1 - y0) + HALF_THICK * np.sin(th)
                break
            acc += ln
        r, g, b = COLORS[k % len(COLORS)]
        lines.append(
            f'    <landmark type="panel" x="{fmt(px,4)}" y="{fmt(py,4)}" '
            f'theta="{fmt(th,6)}" height="{PANEL}" width="{PANEL}"\n'
            f'              texture="../protos/world_objects/textures/flags/flag_{k}.png"\n'
            f'              red="{r:.2f}" green="{g:.2f}" blue="{b:.2f}"/>\n')
    lines.append(TAIL)
    with open(os.path.join(HERE, name + '.xml'), 'w') as f:
        f.writelines(lines)

    ux, uy = hx - MARGIN, hy - MARGIN
    spacing, (nx, ny) = lattice(4.0 * ux * uy, [ux, uy])
    xs = np.round(np.arange(-nx, nx + 1) * spacing, 5)
    ys = np.round(np.arange(-ny, ny + 1) * spacing, 5)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    npts = write_grid(name, X.ravel(), Y.ravel())
    return name, area, npts, spacing, cover


TAIL = '''
    <train_start_positions>
        <pos x="0.0" y="0.0" theta="0.0"/>
    </train_start_positions>

    <test_start_positions>
        <pos x="0.0" y="0.0" theta="0.0"/>
    </test_start_positions>

    <goal id="0" x="0.0" y="0.0"/>
</world>
'''


def write_grid(name, xs, ys):
    out_dir = os.path.join(HERE, 'positions')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + '_positions.csv'), 'w') as f:
        f.write('x,y,theta\n')
        for x, y in zip(xs, ys):
            f.write(f'{x:g},{y:g},0.0\n')
    return len(xs)


if __name__ == '__main__':
    built = [build_disc(r) for r in RADII] + [build_box(CORR_L, CORR_W)]
    print(f"{'arena':20s} {'area':>9s} {'positions':>10s} {'spacing':>9s} "
          f"{'cue cover':>10s}")
    for name, area, npts, spacing, cover in built:
        print(f'{name:20s} {area:8.2f}  {npts:10d} {spacing:9.4f} m '
              f'{cover:9.1f}%')
    areas = [b[1] for b in built[:len(RADII)]]
    print(f'\narea sweep spans {max(areas)/min(areas):.1f}x  '
          f'(Harland 8.8x); sample count held at ~{N_TARGET}')
