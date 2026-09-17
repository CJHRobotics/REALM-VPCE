"""Are place fields more elongated near a wall, and do they line up with it?

Experiment 4. Takes the field libraries Experiment 2 built -- the same eight
arenas, the same channels, the same rules, the same operating point -- and asks
two things of each field: how elongated it is, and which way it points relative
to the nearest wall. Both are read against wall distance, per channel, and the
four discs are compared with the four rectangles.

The confound that decides the whole design
------------------------------------------
Rule 7 fits a field's ellipse to the second moments of its MASK, and the mask
is intersected with the floor (`& G['in_env']`). A field whose extent reaches
past the wall is therefore truncated, and the second moments of a truncated
blob are elongated ALONG the wall. So both of this experiment's measures --
elongation rising toward the wall, the major axis lining up with the wall --
are produced by clipping alone, in a pipeline with no anisotropy in it
anywhere. A raw curve of elongation against wall distance is not evidence of
anything; it is a picture of the arena's outline.

A second confound rides along with the first. A large field cannot sit near a
wall without being cut, so "near the wall" also means "small", and a small
field's moments are estimated from few bins, where the axis ratio is noisier
and biased upward. Size has to be held still, not averaged over.

Everything below exists to separate a real effect from those two.

Two arms
--------
ARM A -- fields clear of the wall. A field whose recorded ellipse, grown by
one bin, lies wholly on the floor has its shape measured with no clipping at
all. Among those fields the measurement is clean and the question can be asked
directly: are the more elongated ones nearer the wall, and do they point along
it? The null holds each field's shape rigid and moves it to a random position
where it still fits whole -- Experiment 3's placement null, reused unchanged,
uniform and Rule 11 tiling. Because the shape never changes, a field keeps its
elongation in every draw, so correlating fixed elongations against null
positions carries the availability structure (a big field's accepted positions
are all away from the wall) without assuming anything about it. Alignment does
change with position, because the wall's tangent turns under the field, so
alignment also gets a per-field excess.

This arm is unbiased by construction and the report leads with it. Its cost is
the population it can speak for: a large field at a wall is always cut, so the
near-wall end of the sample is small fields only.

ARM B -- every field, against a clipping-matched null. Keeps the fields Arm A
has to drop and makes the null reproduce the clipping instead of avoiding it.
For each library a donor pool is built from its fields that are clear of the
wall: each donor's ellipse is re-laid at 12 orientations, placed at random
floor positions, CLIPPED to the floor, and re-measured through the same Rule 7
arithmetic that measured the real fields. Every observed field is then matched
to pool placements of the same VISIBLE area at the same wall distance, and its
excess is its value minus the mean of its matches. Matching on visible area
and distance is what holds size and position still; randomising the donor's
orientation is what makes the alignment null mean "orientation independent of
the wall".

A donor is ANCHORED at its recorded shape: laying a continuous ellipse on the
lattice and taking the moments of the bins inflates the axis ratio, and the
recorded value had already been through that once, so a template built from it
and re-measured carries the inflation twice. The pool therefore keeps only the
increment clipping adds to its own unclipped measurement and reports it on top
of the donor's recorded ratio. Without that anchor --synthetic round found an
elongation effect at p = 0.009 in data containing none.

Arm B's statistic is the NEAR minus FAR difference of that excess, not the
excess itself. The absolute level of the excess depends on the donor pool's
size composition, and the pool is drawn from fields that fit whole, which
under-represents large fields; the difference across distance is first-order
insensitive to that composition, the way Experiment 3's statistic is a LARGE
minus SMALL difference rather than a LARGE level. The levels are reported
beside it and are not the test.

What is measured
----------------
  elongation  semi-major / semi-minor, >= 1, from the bank. Averaged in logs
              and reported back as a ratio, so a mean is a geometric mean and
              a difference is a factor.
  alignment   cos(2 x angle between the major axis and the nearest wall's
              tangent): +1 the field lies ALONG the wall, -1 it points
              STRAIGHT AT it, 0 no relation. Doubling the angle is what makes
              it blind to which end of the axis is which, as an axis has no
              direction. In a disc the tangent turns with the field's bearing;
              in a rectangle it runs along whichever wall is nearest, and a
              field within one bin of being equidistant from two walls has no
              nearest wall and is left out rather than assigned one.
  near / far  wall distance on the range a field's centre can actually
              occupy: 0 is as near the wall as the collection keep-out
              (IN_ENV_MARGIN_M, 0.2 m) allows, 1 is the disc's centre or the
              rectangle's midline. NEAR is the inner 20% of that, FAR the
              outer 40%. Normalising by the half-width instead put the whole
              near-wall bin of the 2 m corridor inside the keep-out, where no
              field can be, and every corridor statistic came out empty.

  level       each measure also averaged over every field in the arm, against
              the same null. A preference that does not vary with distance is
              invisible to a near-minus-far statistic, and for alignment that
              is the likelier shape of a real effect.

Discs against rectangles
------------------------
Arena shape is declared, never inferred, as Experiment 2 declares its roles:
four discs (r = 3 and r = 6, with and without panels), two 10 x 10 m squares,
two 10 x 2 m corridors. The squares and corridors are both rectangles but are
kept apart as well as pooled, because the corridor is 2 m wide: every field
wide enough to span it is cut on both sides, its wall frame barely turns, and
half its wall distance range is a handful of bins. It belongs in this
experiment as the extreme case of "everything is near a wall", not as a place
to read a distance profile.

Inference
---------
Per library, p from a normal fitted to the null draws and a Benjamini-Hochberg
q across libraries. Per arena, the channels' excesses are averaged and divided
by the mean of their null standard deviations -- the value for perfectly
correlated channels -- because six channels in one arena share most of their
structure. That can only understate significance. The disc-against-rectangle
contrast is a Welch test on the arena-level values, four against four, and is
reported as the weak comparison it is at that n.

Calibration
-----------
--synthetic replaces every library with fields of known shape and re-runs the
whole pipeline, which is the only way to know what these statistics do:

  --synthetic round    circular fields, random orientation, areas resampled
                       from the real library, placed uniformly and clipped by
                       the wall. Every scrap of elongation and alignment in
                       this library is clipping. Both arms must find nothing.
  --synthetic planted  the same, but intrinsic elongation rises toward the
                       wall (axis ratio 1 + PLANT_STRENGTH x (1 - d)^2) with
                       the major axis on the wall tangent. Both arms must find
                       it.

Run the pair before trusting a number out of this script, and read the two
results together: `round` bounds the false-positive rate, `planted` bounds the
power, and a null result means little without both.

--calibration then takes the `round` run's summary.csv and subtracts its
measured bias from every statistic, re-deriving each p from the corrected
value against the same null standard deviation. The raw values stay in
summary.csv as *_excess_raw, because a correction the reader cannot see is not
a correction.

What development measured -- on synthetic banks at a deliberately coarsened
bin size, in circ_lm8_r3, corr_lm8_l10w2 and corr_lm8_l10w10, at 200 draws.
These are NOT measurements on the real data; run the calibration here and read
its own numbers:

  Arm A   clean on 6 of 6 round tests (p 0.27 to 0.81) and found the planted
          effect in 6 of 6 (p 0.044 to 2e-14). Needs no correction.
  Arm B   found the planted effect in 6 of 6 (p 3e-4 to 1e-81) but kept a
          residual false positive in two places: near-minus-far elongation in
          the disc (p 0.0008) and alignment in the 2 m corridor (p 0.03).
          Do not read Arm B without --calibration.

Two things were caught this way and are fixed rather than noted. Donors are
anchored at their recorded shape, without which the lattice's own inflation
of the axis ratio was counted twice and `round` reported elongation at
p = 0.009 in data with none. And donors are reweighted toward the size
distribution of the fields far from the wall, without which the pool offered
the wrong mixture of "large field heavily cut" against "small field barely
touched" for a near-wall field of a given visible area.

A third was not fixable and is a limitation: Arm A's elongation LEVEL has no
null at all. The null moves a rigid field, so its elongation is identical in
every draw, the null standard deviation is float noise, and a z of any size
can be manufactured from it. That statistic is not computed. Alignment does
change with position -- the wall's tangent turns under the field -- so it
keeps its level test.

Usage
    python run_wall_elongation.py [--envs a,b] [--channels ...] [--n-null 1000]
                                  [--synthetic round|planted] [--rebuild]
                                  [--seed 0]
"""

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch
import rules as R
import run_scale_distribution as SD
import run_wall_proximity as WP
from realm_tools.experiment_lib.reporting import ExperimentReport

# The eight arenas that were collected. Shape is declared per arena, not read
# off the geometry: aspect ratio cannot tell a square from a disc, and an lm0
# arena has exactly its lm8 twin's outline, so either would be mis-grouped.
# circ_lm8_r10 is in Experiment 2's env list but was never collected; a
# missing dataset is skipped with a message in any case.
ARENA_SHAPE = {'circ_lm8_r3': 'disc', 'circ_lm8_r6': 'disc',
               'circ_lm0_r3': 'disc', 'circ_lm0_r6': 'disc',
               'corr_lm8_l10w10': 'square', 'corr_lm0_l10w10': 'square',
               'corr_lm8_l10w2': 'corridor', 'corr_lm0_l10w2': 'corridor'}
# The contrast the experiment is for. Squares and corridors are both
# rectangles; they are pooled for the headline and kept apart underneath it.
SHAPE_GROUP = {'disc': 'circular', 'square': 'rectangular',
               'corridor': 'rectangular'}
# Reading order for every table and figure: discs smallest first, then the
# squares, then the corridors. Sorting on the shape's name would lead with the
# corridor, which is the one arena the report warns about reading first.
SHAPE_ORDER = {'disc': 0, 'square': 1, 'corridor': 2}
ENVS = list(ARENA_SHAPE)
CHANNELS = SD.CHANNELS
CHANNEL_COLORS = SD.CHANNEL_COLORS

# Place-field formation is Experiment 2's, exactly: its operating point, Rule 2
# off, LAMBDA 0, seed 0. Nothing here is a knob, because the point of reusing
# the libraries is that the experiments describe the same fields.
PCTL, THRESH = SD.SETTINGS[0]
IOU = None
BASE_C = R.resolve_cfg(dict(LAMBDA=0.0, RANDOM_SEED=0))

DATA_DIR = f'{REPO}/data/vpce/collect_data'
XML_DIR = f'{REPO}/simulation/worlds/environments/vpce'
# Experiment 2's cache, shared on purpose, with Experiment 2's cache key.
BANK_DIR = f'{REPO}/data_cache/scale_distribution'
OUT_DIR = f'{REPO}/data_cache/wall_elongation'
FIG_DIR = f'{HERE}/figures/wall_elongation'

MIN_FIELDS = SD.MIN_FIELDS     # below this a library says nothing
MIN_CLEAR = 40                 # fields clear of the wall Arm A needs
MIN_DONORS = 25                # donors Arm B's pool needs
MIN_BIN = 12                   # fields a distance bin needs to be reported
MIN_ACCEPT = 20                # null placements a field needs a baseline
MIN_MATCH = 30                 # matched pool rows a field needs a baseline
N_NULL = 1000                  # null realisations of each library
POOL_TARGET = 150_000          # donor x position placements measured per library
POOL_ORIENTATIONS = 12         # orientations each donor's ellipse is re-laid at
AREA_TOL_LOG = float(np.log(1.25))   # visible area matched within this factor
DIST_TOL = 0.06                # normalised wall distance matched within this
CHUNK_ELEMS = 2_000_000        # template x placement elements per clipping batch
VERIFY_N = 240                 # placements re-measured through rules.field_shape
VERIFY_TOL = 1e-8              # and the agreement demanded of them
PLANT_STRENGTH = 1.0           # --synthetic planted axis ratio at the wall
ALPHA = 0.05

# Wall distance as a fraction of the furthest possible. Finer near the wall,
# which is where the question is, and where the fields are.
DIST_EDGES = np.array([0.0, 0.10, 0.20, 0.35, 0.60, 1.0])
# The near/far test cut is a TERCILE of each arm's own eligible fields, not a
# fixed distance. A fixed cut cannot work: Arm A needs a field to fit whole,
# which in the 2 m corridor puts its nearest field 0.38 m from the wall, so a
# fixed near-wall bin there is empty by construction and every corridor
# statistic came out NaN. Terciles always have fields on both sides, and the
# cut each library used is reported in metres so "near" is never left to mean
# whatever the reader assumes.
NEAR_Q, FAR_Q = 1.0 / 3.0, 2.0 / 3.0
# E6's histogram splits at fixed distances instead, because it pools arenas
# and a per-library cut would put different fields in the same curve.
E6_NEAR, E6_FAR = 0.25, 0.75

NULLS = ('uniform', 'tiling')
MEASURES = ('elongation', 'alignment')
MEASURE_UNITS = {'elongation': 'log elongation (a/b); +0.10 = 10% longer',
                 'alignment': 'cos(2 x angle to the wall tangent); '
                              '+1 along the wall, -1 straight at it'}

# Colours. Channels are identities and keep Experiment 2's distinct hues.
# Wall distance is a magnitude, so it runs along a multi-hue ramp rather than
# one hue: plasma, which falls monotonically in lightness and stays separable
# under colour-blind simulation. The near end is the dark end, so the bin the
# question is about carries the heaviest ink.
#
# The ramp stops at 0.72 rather than running to plasma's yellow end. Yellow
# measures under 2:1 against this surface, and a histogram outline that faint
# is barely there -- the same reason Experiment 2 starts its size ramp a
# quarter of the way in. Orange holds the light end at about 2.8:1.
DIST_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'vpce_dist', matplotlib.colormaps['plasma'](np.linspace(0.05, 0.72, 256)))
DIST_COLORS = [mcolors.to_hex(DIST_CMAP(t))
               for t in np.linspace(0.0, 1.0, len(DIST_EDGES) - 1)]
# Arena shape is an identity with three values: two warm, one cool would read
# as two groups, so all three are drawn from far apart on the hue circle.
SHAPE_COLORS = {'disc': '#1b6ca8', 'square': '#c1440e', 'corridor': '#6a3d9a'}
INK, MUTED, RULE_GRAY, SURFACE = SD.INK, SD.MUTED, SD.RULE_GRAY, SD.SURFACE


def bank_path(env_name, cname):
    """Experiment 2's cache key at the operating point, Rule 2 off."""
    return WP.bank_path(env_name, cname)


def arena_shape(env_name):
    """'disc', 'square' or 'corridor'. 'other' keeps an arena out of the
    shape contrast without keeping it out of the per-library tables."""
    return ARENA_SHAPE.get(env_name, 'other')


def arena_group(env_name):
    """'circular', 'rectangular', or 'other'."""
    return SHAPE_GROUP.get(arena_shape(env_name), 'other')


# ----------------------------------------------------------------- geometry

def wall_frame(cx, cy, env, bin_m):
    """Angle of the nearest wall's tangent at each point, and where there is
    no nearest wall.

    In a disc the wall curves, so the tangent is perpendicular to the field's
    own bearing from the centre and turns with it; a point within half a bin of
    the exact centre has no bearing and is marked ambiguous. In a rectangle the
    tangent runs along whichever wall is nearest -- along y for the left and
    right walls, along x for the top and bottom -- and a point whose two
    nearest walls are within one bin of equal distance sits in a corner, where
    the frame would be a coin toss, so it is marked ambiguous instead.

    Returns (tangent_rad, ambiguous). NaN in, NaN out.
    """
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    if env['is_circular']:
        dx, dy = cx - env['env_cx'], cy - env['env_cy']
        r = np.hypot(dx, dy)
        tangent = np.arctan2(dy, dx) + 0.5 * np.pi
        amb = ~(r > 0.5 * bin_m)
    else:
        d = np.stack([cx - env['x_min'], env['x_max'] - cx,
                      cy - env['y_min'], env['y_max'] - cy], axis=0)
        order = np.sort(d, axis=0)
        nearest = np.argmin(d, axis=0)
        # Walls 0 and 1 are the x = const walls; their faces run along y.
        tangent = np.where(nearest <= 1, 0.5 * np.pi, 0.0)
        amb = (order[1] - order[0]) < bin_m
    amb = amb | ~np.isfinite(cx) | ~np.isfinite(cy)
    return np.where(amb, np.nan, tangent), amb


def wall_alignment(theta, cx, cy, env, bin_m):
    """cos(2 x angle between a field's major axis and the nearest wall).

    +1 the field lies along the wall, -1 it points straight at it, 0 no
    relation. The angle is doubled because an axis has no direction: theta and
    theta + pi describe the same field and must give the same number.
    """
    tangent, _ = wall_frame(cx, cy, env, bin_m)
    return np.cos(2.0 * (np.asarray(theta, dtype=float) - tangent))


def max_wall_distance(env):
    """The furthest any point can be from the boundary: a disc's centre, or a
    rectangle's midline."""
    return WP.max_wall_distance(env)


def wall_margin(env):
    """The nearest a field's centre can be to a wall: the collection keep-out.

    Rule-grid bins stop IN_ENV_MARGIN_M short of the wall, so no mask bin and
    therefore no centroid is ever closer than that. It is 0.2 m, which is 6.7%
    of the r = 3 disc's range and 20% of the 2 m corridor's.
    """
    return float(env.get('wall_margin_m', 0.0))


def norm_wall_distance(x, y, env):
    """Wall distance on the range a field can actually occupy: 0 as near the
    wall as the collection lattice allows, 1 as far as the arena allows.

    Dividing by the arena's half-width instead would make the scale mean
    something different in every arena, and in the 2 m corridor it put the
    whole near-wall bin below the keep-out, where no field can be: the bin
    came out empty and every corridor statistic was NaN.
    """
    lo, hi = wall_margin(env), max_wall_distance(env)
    d = R.wall_distance(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                        env)
    with np.errstate(invalid='ignore'):
        return np.clip((d - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def _cut_m(dn_cut, env):
    """A normalised cut read back as a wall distance in metres."""
    if not np.isfinite(dn_cut):
        return np.nan
    lo, hi = wall_margin(env), max_wall_distance(env)
    return float(lo + dn_cut * (hi - lo))


def dist_bin(dn):
    """Index into DIST_EDGES, or -1 outside it."""
    dn = np.asarray(dn, dtype=float)
    b = np.digitize(dn, DIST_EDGES[1:-1], right=False)
    return np.where(np.isfinite(dn), b, -1)


def dist_bin_labels():
    return [f'{DIST_EDGES[i]:.2f}-{DIST_EDGES[i + 1]:.2f}'
            for i in range(len(DIST_EDGES) - 1)]


def ellipse_offsets(a, b, theta, G, n_bins):
    """The n_bins bin offsets nearest the centre in elliptical radius.

    The same ranking rules._templates uses, but purely geometric: no floor is
    consulted, so the template is a shape rather than a shape-at-a-place and
    can be laid down anywhere. Used for Arm B's donors and for the synthetic
    libraries, both of which need a known shape placed at a chosen spot.
    """
    sx = float(G['xc'][1] - G['xc'][0])
    sy = float(G['yc'][1] - G['yc'][0])
    a = max(float(a), 1e-9)
    b = max(float(b), 1e-9)
    n_bins = max(1, int(n_bins))
    for grow in (1.6, 2.5, 4.0, 7.0):
        ri = int(np.ceil(grow * a / sx)) + 1
        rj = int(np.ceil(grow * a / sy)) + 1
        di, dj = np.meshgrid(np.arange(-ri, ri + 1), np.arange(-rj, rj + 1),
                             indexing='ij')
        di, dj = di.ravel(), dj.ravel()
        if len(di) >= n_bins:
            ox, oy = di * sx, dj * sy
            u = ox * np.cos(theta) + oy * np.sin(theta)
            v = -ox * np.sin(theta) + oy * np.cos(theta)
            order = np.argsort((u / a) ** 2 + (v / b) ** 2, kind='stable')
            return di[order][:n_bins], dj[order][:n_bins]
    return di, dj


# ------------------------------------------- Rule 7 on many placements at once

def clipped_shape(ci, cj, di, dj, G):
    """Rule 7's ellipse for one template placed at many bins and clipped to
    the floor.

    A vectorised twin of rules.field_shape and it has to stay exactly that:
    the same second moments at the same ddof, the same one-bin variance floor,
    the same rescale to pi*a*b = the visible area, the same major axis. If the
    two ever drifted apart, Arm B would be comparing observed values measured
    one way against null values measured another, and the difference between
    the two implementations would look like the effect. `verify_clipped_shape`
    checks them against each other at run time on every arena.

    Bins are offsets from the placement's centre bin, so the local coordinates
    di*sx and dj*sy do not depend on the placement -- only which of them land
    on the floor does. That turns the moment sums into one matrix product and
    keeps the centred sums away from the arena's absolute coordinates, where
    cancellation would cost precision.

    Returns (n_vis, cx, cy, a, b, theta) per placement; n_vis 0 where the
    template landed wholly off the floor.
    """
    in_env, xc, yc = G['in_env'], G['xc'], G['yc']
    gx, gy = in_env.shape
    sx, sy = float(xc[1] - xc[0]), float(yc[1] - yc[0])
    bin_area = float(G['bin_area'])
    ci = np.asarray(ci, dtype=np.int64)
    cj = np.asarray(cj, dtype=np.int64)
    T = len(di)
    ox, oy = di * sx, dj * sy
    W = np.stack([np.ones(T), ox, oy, ox * ox, oy * oy, ox * oy], axis=1)
    n = len(ci)
    M = np.empty((n, 6))
    step = max(1, CHUNK_ELEMS // max(T, 1))
    for s in range(0, n, step):
        e = min(s + step, n)
        I = (ci[s:e, None] + di[None, :]).astype(np.int32)
        J = (cj[s:e, None] + dj[None, :]).astype(np.int32)
        inb = (I >= 0) & (I < gx) & (J >= 0) & (J < gy)
        Ic = np.clip(I, 0, gx - 1)
        Jc = np.clip(J, 0, gy - 1)
        M[s:e] = (inb & in_env[Ic, Jc]).astype(np.float64) @ W
    n_vis, mx, my, a, b, theta = _ellipse_from_moments(M, bin_area)
    good = n_vis > 0
    cx = np.where(good, xc[np.clip(ci, 0, gx - 1)] + mx, np.nan)
    cy = np.where(good, yc[np.clip(cj, 0, gy - 1)] + my, np.nan)
    return n_vis, cx, cy, a, b, theta


def _ellipse_from_moments(M, bin_area):
    """Rule 7's ellipse from raw moment sums [n, Sx, Sy, Sxx, Syy, Sxy].

    The arithmetic itself, shared by `clipped_shape` and `template_shape`, so
    a template measured with no floor and a placement measured through one
    cannot drift apart. Coordinates are relative to the placement's centre
    bin, so the centre comes back as an offset.
    """
    n_vis = np.rint(M[:, 0]).astype(np.int64)
    good = n_vis > 0
    area = n_vis * bin_area
    r_eq = np.sqrt(np.maximum(area, 0.0) / np.pi)
    cnt = np.where(good, M[:, 0], np.nan)
    mx, my = M[:, 1] / cnt, M[:, 2] / cnt
    # Central second moments at ddof = 1, as np.cov gives rules.field_shape.
    with np.errstate(invalid='ignore', divide='ignore'):
        den = cnt - 1.0
        A = (M[:, 3] - cnt * mx * mx) / den
        B = (M[:, 4] - cnt * my * my) / den
        C = (M[:, 5] - cnt * mx * my) / den
    half, root = 0.5 * (A + B), np.sqrt(0.25 * (A - B) ** 2 + C * C)
    e0, e1 = half + root, half - root
    theta = np.where(np.abs(C) > 1e-18, np.arctan2(e0 - A, C),
                     np.where(A >= B, 0.0, 0.5 * np.pi))
    # rules.field_shape floors an axis at one bin via bin_side = sqrt(bin_area);
    # min(sx, sy) would differ the moment the grid is not square.
    floor = bin_area / 12.0
    a_raw = 2.0 * np.sqrt(np.maximum(e0, floor))
    b_raw = 2.0 * np.sqrt(np.maximum(e1, floor))
    with np.errstate(invalid='ignore', divide='ignore'):
        scale = np.sqrt(area / (np.pi * a_raw * b_raw))
    a, b = a_raw * scale, b_raw * scale
    # Fewer than three bins has no covariance: rules.field_shape returns a
    # disc of the equivalent radius and theta 0, so this does too.
    tiny = good & (n_vis < 3)
    a = np.where(tiny, r_eq, a)
    b = np.where(tiny, r_eq, b)
    theta = np.where(tiny, 0.0, theta)
    bad = ~good
    return (n_vis, np.where(bad, np.nan, mx), np.where(bad, np.nan, my),
            np.where(bad, np.nan, a), np.where(bad, np.nan, b),
            np.where(bad, np.nan, theta))


def template_shape(di, dj, G):
    """The ellipse a template measures with nothing clipping it.

    A template built from a field's recorded (a, b) does not measure back that
    same (a, b): laying a continuous ellipse on a lattice and taking the
    moments of the bins inflates the axis ratio, and the recorded value had
    already been through that once. Measuring the template unclipped is what
    lets `donor_pool` anchor a donor at its recorded shape and keep only the
    part clipping added -- without which the null carries the discretisation
    bias twice and Arm B reports an effect where there is none. This was not
    hypothetical: --synthetic round found it at p = 0.009.
    """
    sx = float(G['xc'][1] - G['xc'][0])
    sy = float(G['yc'][1] - G['yc'][0])
    ox, oy = di * sx, dj * sy
    M = np.array([[len(di), ox.sum(), oy.sum(), (ox * ox).sum(),
                   (oy * oy).sum(), (ox * oy).sum()]], dtype=float)
    n, _, _, a, b, th = _ellipse_from_moments(M, float(G['bin_area']))
    return float(a[0]), float(b[0]), float(th[0])


def verify_clipped_shape(G, rng, n=VERIFY_N):
    """Check `clipped_shape` against rules.field_shape itself.

    Builds real boolean masks for a sample of placements of a sample of
    ellipses -- deliberately including ones that hang over the wall -- runs
    both implementations, and returns the worst disagreement in area, the
    axes, and the axis direction. Raises if they differ by more than
    VERIFY_TOL, because past that point Arm B's excess is measuring the gap
    between two pieces of code.
    """
    in_env = G['in_env']
    gx, gy = in_env.shape
    env_i, env_j = np.nonzero(in_env)
    bin_side = float(np.sqrt(G['bin_area']))
    worst = dict(area=0.0, a=0.0, b=0.0, axis=0.0, n_checked=0, n_clipped=0,
                 n_axis=0)
    for _ in range(n):
        a = float(rng.uniform(1.0, 12.0)) * bin_side
        b = a / float(rng.uniform(1.0, 3.0))
        th = float(rng.uniform(0.0, np.pi))
        n_bins = max(1, int(round(np.pi * a * b / G['bin_area'])))
        di, dj = ellipse_offsets(a, b, th, G, n_bins)
        k = int(rng.integers(0, len(env_i)))
        ci, cj = int(env_i[k]), int(env_j[k])
        I, J = ci + di, cj + dj
        inb = (I >= 0) & (I < gx) & (J >= 0) & (J < gy)
        mask = np.zeros((gx, gy), dtype=bool)
        keep = np.zeros(len(di), dtype=bool)
        keep[inb] = in_env[I[inb], J[inb]]
        if not keep.any():
            continue
        mask[I[keep], J[keep]] = True
        ref = R.field_shape(mask, G)
        nv, cx, cy, aa, bb, tt = clipped_shape(np.array([ci]), np.array([cj]),
                                               di, dj, G)
        worst['n_checked'] += 1
        worst['n_clipped'] += int(keep.sum() < len(di))
        scale = max(ref['a'], bin_side)
        worst['area'] = max(worst['area'],
                            abs(ref['area'] - nv[0] * G['bin_area']) / scale ** 2)
        worst['a'] = max(worst['a'], abs(ref['a'] - aa[0]) / scale)
        worst['b'] = max(worst['b'], abs(ref['b'] - bb[0]) / scale)
        # Axis direction only, so theta and theta + pi agree. Skipped for a
        # near-circular result: there the eigenvectors are ill-conditioned and
        # the two implementations may legitimately disagree about a direction
        # the field does not have. The axis ratio is still checked, and that is
        # what such a field contributes to either measure.
        if ref['b'] > 0 and ref['a'] / ref['b'] > 1.05:
            d = abs(np.cos(2.0 * (ref['theta'] - tt[0])) - 1.0)
            worst['axis'] = max(worst['axis'], float(d))
            worst['n_axis'] = worst.get('n_axis', 0) + 1
    bad = {k: v for k, v in worst.items()
           if not k.startswith('n_') and v > VERIFY_TOL}
    if bad:
        raise AssertionError(
            f'clipped_shape disagrees with rules.field_shape: {bad} '
            f'(tolerance {VERIFY_TOL:g}, {worst["n_checked"]} placements, '
            f'{worst["n_clipped"]} of them clipped). Arm B is not safe to '
            f'run until the two agree.')
    return worst


# ------------------------------------------------------------ Arm B: the pool

def donor_weights(bank, clear, dn, n_bins_target=14):
    """How many placements to spend on each donor.

    A donor has to fit whole, and a large field rarely does, so the donors are
    biased small -- and that bias is not harmless. Matching is on VISIBLE
    area, so a near-wall field showing a small area could be a large field
    heavily cut or a small one barely touched, and which of those the pool
    offers decides how elongated the null looks. Getting the mixture wrong is
    what --synthetic round caught in the r = 3 disc at p = 0.0005, in data
    with no elongation in it at all.

    The target is the size distribution of the fields FAR from the wall, where
    little is cut, so a recorded area is close to a true area. Under the null
    being tested -- shape independent of the wall -- that is the distribution
    the near-wall fields' true sizes are drawn from too. Donors are then given
    placements in proportion to how much that target wants their size and how
    few donors of that size there are.

    Returns (weight per donor, share of the target's mass no donor can cover).
    The second number is the honest limit: where it is large, the pool cannot
    represent the sizes the near-wall fields were probably drawn from, and
    Arm B should not be believed for that library.
    """
    la = np.log(np.maximum(bank.area_env_m2.to_numpy(dtype=float), 1e-12))
    donors = np.flatnonzero(clear)
    tgt = np.isfinite(dn) & (dn >= FAR_Q)
    if not tgt.any() or not len(donors):
        return np.ones(len(donors)), 1.0
    edges = np.linspace(la.min(), la.max() + 1e-9, n_bins_target + 1)
    t_hist = np.histogram(la[tgt], bins=edges)[0].astype(float)
    d_hist = np.histogram(la[donors], bins=edges)[0].astype(float)
    t_hist /= max(t_hist.sum(), 1.0)
    unrepresented = float(t_hist[d_hist == 0].sum())
    which = np.clip(np.digitize(la[donors], edges[1:-1]), 0, n_bins_target - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(d_hist[which] > 0, t_hist[which] / d_hist[which], 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() <= 0:
        return np.ones(len(donors)), unrepresented
    # Capped so one rare large donor cannot become most of the pool: it would
    # be a single shape standing in for a whole size class.
    mean = w[w > 0].mean()
    return np.clip(w, 0.05 * mean, 20.0 * mean), unrepresented


def donor_pool(bank, clear, env, G, rng, dn=None, n_target=POOL_TARGET):
    """Placements of this library's own shapes, clipped by the wall.

    Donors are the fields clear of the wall, whose ellipses are honestly
    recorded. Each donor's ellipse is re-laid at POOL_ORIENTATIONS evenly
    spaced angles, offset by a random amount per donor so the set of
    orientations is not the same grid for every field, and each orientation is
    dropped at random floor positions and clipped.

    Randomising the orientation is what makes this a null for ALIGNMENT: a
    donor keeps its axis ratio and its area but forgets which way it pointed,
    so any alignment the pool shows is the wall cutting it, never the wall
    having oriented it in the first place. Keeping the axis ratio is what makes
    it a null for ELONGATION at the same time: the pool's elongation is this
    library's own elongation plus whatever clipping adds.

    Returns a DataFrame of placements: donor, visible area, normalised wall
    distance, log elongation, alignment.
    """
    donors = np.flatnonzero(clear)
    if len(donors) < MIN_DONORS:
        return pd.DataFrame(), 0, 1.0
    bin_m = float(np.sqrt(G['bin_area']))
    in_env = G['in_env']
    env_i, env_j = np.nonzero(in_env)
    a_all = bank.semi_major_m.to_numpy(dtype=float)
    b_all = bank.semi_minor_m.to_numpy(dtype=float)
    n_bins_all = WP._n_bins(bank, G)
    w, unrepresented = (donor_weights(bank, clear, dn) if dn is not None
                        else (np.ones(len(donors)), np.nan))
    budget = n_target / POOL_ORIENTATIONS
    per_k = np.maximum(2, np.round(budget * w / max(w.sum(), 1e-12))).astype(int)
    parts = []
    for kk, k in enumerate(donors):
        per = int(per_k[kk])
        phase = float(rng.uniform(0.0, np.pi))
        for t in range(POOL_ORIENTATIONS):
            th = phase + t * np.pi / POOL_ORIENTATIONS
            di, dj = ellipse_offsets(a_all[k], b_all[k], th, G, n_bins_all[k])
            # Anchor: what this template measures with nothing clipping it.
            # The pool keeps the increment clipping adds and reports it on top
            # of the donor's RECORDED shape, so the lattice's own inflation is
            # counted once, as it is in the observed value.
            ra, rb, _ = template_shape(di, dj, G)
            anchor = (np.log(max(ra, 1e-12) / max(rb, 1e-12))
                      - np.log(max(a_all[k], 1e-12) / max(b_all[k], 1e-12)))
            pick = rng.integers(0, len(env_i), size=per)
            nv, cx, cy, aa, bb, tt = clipped_shape(env_i[pick], env_j[pick],
                                                   di, dj, G)
            ok = np.isfinite(aa) & (bb > 0)
            if not ok.any():
                continue
            dn = norm_wall_distance(cx[ok], cy[ok], env)
            parts.append(pd.DataFrame(dict(
                donor=int(k),
                area=nv[ok] * G['bin_area'],
                wall_dist_norm=dn,
                log_elongation=np.log(aa[ok] / bb[ok]) - anchor,
                alignment=wall_alignment(tt[ok], cx[ok], cy[ok], env, bin_m))))
    if not parts:
        return pd.DataFrame(), len(donors), unrepresented
    pool = pd.concat(parts, ignore_index=True)
    pool['log_area'] = np.log(np.maximum(pool.area.to_numpy(dtype=float), 1e-12))
    return pool, len(donors), unrepresented


def matched_baseline(bank, pool, dn, rng, n_draws):
    """Each field's Arm B baseline, and n_draws null values drawn from the
    same matches.

    A field's matches are the pool placements within AREA_TOL_LOG of its
    visible area and DIST_TOL of its wall distance, excluding placements of
    the field's own shape -- without that exclusion a field would partly be
    its own null and every excess would be pulled toward zero. A field with
    fewer than MIN_MATCH matches has no baseline and drops out of every mean.

    Returns dict per measure of (baseline, null values (n_fields, n_draws)),
    plus the match count per field.
    """
    n = len(bank)
    out = {m: (np.full(n, np.nan), np.full((n, n_draws), np.nan))
           for m in MEASURES}
    n_match = np.zeros(n, dtype=int)
    if not len(pool):
        return out, n_match
    log_area = np.log(np.maximum(bank.area_env_m2.to_numpy(dtype=float), 1e-12))
    p_la = pool.log_area.to_numpy(dtype=float)
    p_dn = pool.wall_dist_norm.to_numpy(dtype=float)
    p_donor = pool.donor.to_numpy(dtype=int)
    vals = {'elongation': pool.log_elongation.to_numpy(dtype=float),
            'alignment': pool.alignment.to_numpy(dtype=float)}
    # Sorted by area so the area window is a slice rather than a full scan;
    # the distance window and the self-exclusion are then a mask over it.
    order = np.argsort(p_la, kind='stable')
    p_la_s, p_dn_s, p_donor_s = p_la[order], p_dn[order], p_donor[order]
    vals_s = {m: v[order] for m, v in vals.items()}
    lo = np.searchsorted(p_la_s, log_area - AREA_TOL_LOG, side='left')
    hi = np.searchsorted(p_la_s, log_area + AREA_TOL_LOG, side='right')
    for k in range(n):
        if not np.isfinite(dn[k]) or hi[k] <= lo[k]:
            continue
        sl = slice(lo[k], hi[k])
        ok = (np.abs(p_dn_s[sl] - dn[k]) <= DIST_TOL) & (p_donor_s[sl] != k)
        idx = np.flatnonzero(ok)
        n_match[k] = len(idx)
        if len(idx) < MIN_MATCH:
            continue
        for m in MEASURES:
            v = vals_s[m][sl][idx]
            v = v[np.isfinite(v)]
            if len(v) < MIN_MATCH:
                continue
            out[m][0][k] = float(v.mean())
            out[m][1][k] = v[rng.integers(0, len(v), size=n_draws)]
    return out, n_match


# -------------------------------------------------------------- the statistics

def _nanmean(x, axis=None):
    return WP._nanmean(x, axis=axis)


def _mean_where(values, sel):
    """Mean of `values` over the True entries of each column of `sel`.

    `values` is (n_fields,) or (n_fields, n_draws); `sel` is (n_fields,) or
    (n_fields, n_draws). NaN where a column selects nothing finite.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(sel, dtype=bool)
    if v.ndim == 1 and s.ndim == 2:
        v = np.broadcast_to(v[:, None], s.shape)
    elif v.ndim == 2 and s.ndim == 1:
        s = np.broadcast_to(s[:, None], v.shape)
    fin = s & np.isfinite(v)
    cnt = fin.sum(axis=0)
    tot = np.where(fin, v, 0.0).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return tot / np.where(cnt > 0, cnt, np.nan), cnt


def tercile_cuts(dn, eligible=None):
    """The wall-distance cuts that split an arm's own fields into thirds.

    Derived once from the observed positions and then held fixed, so the null
    is scored against the same thresholds the observation was. Returns
    (near_max, far_min), NaN where there are too few fields to split.
    """
    v = np.asarray(dn, dtype=float)
    ok = np.isfinite(v)
    if eligible is not None:
        ok &= np.asarray(eligible, dtype=bool)
    v = v[ok]
    if len(v) < 3 * MIN_BIN:
        return np.nan, np.nan
    return float(np.quantile(v, NEAR_Q)), float(np.quantile(v, FAR_Q))


def near_far(values, dn, near_max, far_min, eligible=None):
    """Mean of a measure over near-wall fields minus over far ones.

    `values` and `dn` may each be per field or (n_fields, n_draws): Arm A holds
    the shape still and moves the field, Arm B holds the field still and
    redraws the shape, and both come through here. The cuts are passed in
    rather than recomputed, so observed and null are scored on the same
    thresholds.
    """
    sel = np.isfinite(np.asarray(dn, dtype=float))
    if eligible is not None:
        e = np.asarray(eligible, dtype=bool)
        sel = sel & (e[:, None] if (sel.ndim == 2 and e.ndim == 1) else e)
    if not np.isfinite(near_max) or not np.isfinite(far_min):
        nan = np.full(sel.shape[1] if sel.ndim == 2 else (), np.nan)
        return nan, nan, nan, 0, 0
    near, n_near = _mean_where(values, sel & (np.asarray(dn) <= near_max))
    far, n_far = _mean_where(values, sel & (np.asarray(dn) >= far_min))
    return near - far, near, far, n_near, n_far


def spearman_rows(values, dn):
    """Spearman rho of a fixed per-field measure against each column of `dn`.

    Ranks the measure once, then ranks each column of positions; that is all
    Spearman is, and the loop over 1000 draws through scipy is not worth its
    cost here.
    """
    v = np.asarray(values, dtype=float)
    D = np.asarray(dn, dtype=float)
    D = D[:, None] if D.ndim == 1 else D
    out = np.full(D.shape[1], np.nan)
    for d in range(D.shape[1]):
        ok = np.isfinite(v) & np.isfinite(D[:, d])
        if ok.sum() >= MIN_BIN and np.ptp(v[ok]) > 0 and np.ptp(D[ok, d]) > 0:
            out[d] = float(stats.spearmanr(v[ok], D[ok, d]).statistic)
    return out


def profile(values, dn, bins):
    """Mean of a measure in each wall-distance bin, observed or per draw.

    Either argument may carry the draws -- Arm A redraws positions against a
    fixed measure, Arm B redraws the measure against fixed positions -- so the
    result is as wide as the wider of the two, and squeezed to one column only
    when neither has draws.
    """
    v = np.asarray(values, dtype=float)
    D = np.asarray(dn, dtype=float)
    flat = (v.ndim == 1) and (D.ndim == 1)
    cols = max(v.shape[1] if v.ndim == 2 else 1,
               D.shape[1] if D.ndim == 2 else 1)
    means = np.full((len(bins), cols), np.nan)
    counts = np.zeros(len(bins), dtype=int)
    B = dist_bin(D)
    for i, b in enumerate(bins):
        m, c = _mean_where(v, B == b)
        means[i] = m
        counts[i] = int(np.atleast_1d(c)[0])
    return (means[:, 0] if flat else means), counts


# ------------------------------------------------------------ one library

def _stat_block(row, pre, obs, null, n_near=None, n_far=None):
    """Write an observed statistic, its null, and its p into a summary row."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    # A null with no spread is not a null. Every measure here is of order
    # 0.01 to 1, so a standard deviation this small is float noise from a
    # statistic that cannot vary, and a p derived from it is meaningless at
    # any magnitude.
    if len(null) > 1 and float(null.std(ddof=1)) < 1e-12:
        null = np.array([])
    row[f'{pre}'] = obs
    row[f'{pre}_null_mean'] = float(null.mean()) if len(null) else np.nan
    row[f'{pre}_null_sd'] = float(null.std(ddof=1)) if len(null) > 1 else np.nan
    row[f'{pre}_null_lo'] = float(np.percentile(null, 2.5)) if len(null) else np.nan
    row[f'{pre}_null_hi'] = float(np.percentile(null, 97.5)) if len(null) else np.nan
    row[f'{pre}_excess'] = (obs - row[f'{pre}_null_mean']
                            if np.isfinite(obs) else np.nan)
    row[f'{pre}_p'], row[f'{pre}_p_gt'] = WP.null_p(obs, null)
    if n_near is not None:
        row[f'{pre}_n_near'] = int(np.atleast_1d(n_near)[0])
        row[f'{pre}_n_far'] = int(np.atleast_1d(n_far)[0])
    return row


def analyse_library(bank, env, G, n_null, rng, tag, verbose=True):
    """Both arms, both measures, for one arena x channel library."""
    wmax = max_wall_distance(env)
    bin_m = float(np.sqrt(G['bin_area']))
    a_ax = bank.semi_major_m.to_numpy(dtype=float)
    b_ax = bank.semi_minor_m.to_numpy(dtype=float)
    theta = bank.orientation_rad.to_numpy(dtype=float)
    area = bank.area_env_m2.to_numpy(dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_elong = np.log(np.where(b_ax > 0, a_ax / np.maximum(b_ax, 1e-12),
                                    np.nan))

    row = dict(tag, shape=arena_shape(tag['env']), group=arena_group(tag['env']),
               env_area_m2=float(env['env_area']), max_wall_dist_m=wmax,
               wall_margin_m=wall_margin(env), bin_m=bin_m,
               n_fields=len(bank), tested=False,
               arm_a_tested=False, arm_b_tested=False)
    fields = bank[['area_env_m2', 'radius_env_m', 'semi_major_m',
                   'semi_minor_m', 'elongation', 'orientation_rad',
                   'centroid_x', 'centroid_y', 'dist_to_wall_m',
                   'scale_band']].copy()
    fields.insert(0, 'channel', tag['channel'])
    fields.insert(0, 'env', tag['env'])
    if len(bank) < MIN_FIELDS:
        return row, [], fields

    templates = WP._templates(bank, G)
    clear = WP.clear_of_wall(bank, G, templates)
    # Observed centres as the null measures its own, following Experiment 3: a
    # field clear of the wall by its template's centre, which is where the null
    # puts it back; a cut field by the bank's centroid, which the null only
    # rotates or slides.
    sx = float(G['xc'][1] - G['xc'][0])
    sy = float(G['yc'][1] - G['yc'][0])
    n_bins = WP._n_bins(bank, G)
    xs = bank.centroid_x.to_numpy(dtype=float).copy()
    ys = bank.centroid_y.to_numpy(dtype=float).copy()
    for k, (di, dj, ci, cj) in enumerate(templates):
        if clear[k]:
            xs[k] = G['xc'][ci] + sx * float(di[:n_bins[k]].mean())
            ys[k] = G['yc'][cj] + sy * float(dj[:n_bins[k]].mean())
    dn = norm_wall_distance(xs, ys, env)
    align = wall_alignment(theta, xs, ys, env, bin_m)
    _, ambiguous = wall_frame(xs, ys, env, bin_m)
    obs = {'elongation': log_elong, 'alignment': align}

    fields['clear_of_wall'] = clear
    fields['wall_dist_norm'] = dn
    fields['dist_bin'] = dist_bin(dn) + 1
    fields['log_elongation'] = log_elong
    fields['wall_alignment'] = align
    fields['corner_ambiguous'] = ambiguous
    row.update(n_clear=int(clear.sum()), n_ambiguous=int(ambiguous.sum()),
               elongation_median=float(np.nanmedian(bank.elongation)),
               alignment_mean=_nanmean(align))
    prof_rows = []
    bins = list(range(len(DIST_EDGES) - 1))

    # ------------------------------------------------- Arm A: no clipping
    a_near, a_far = tercile_cuts(dn, clear)
    row.update(arm_a_near_cut=a_near, arm_a_far_cut=a_far,
               arm_a_near_cut_m=_cut_m(a_near, env),
               arm_a_far_cut_m=_cut_m(a_far, env))
    row['arm_a_tested'] = (int(clear.sum()) >= MIN_CLEAR
                           and np.isfinite(a_near) and np.isfinite(a_far))
    if row['arm_a_tested']:
        for nn, spacing in (('uniform', None),
                            ('tiling', BASE_C['SAME_SCALE_SEPARATION'])):
            X, Y = WP.place_library(bank, G, env, templates, clear, xs, ys,
                                    n_null, rng, spacing=spacing)
            row[f'arm_a_{nn}_unplaced_frac'] = float(
                np.mean(~np.isfinite(X[clear])))
            DN = np.where(np.isfinite(X), norm_wall_distance(X, Y, env),
                          np.nan)
            # Elongation: the shape is rigid, so the field keeps its value and
            # only its position is redrawn. Alignment: the value is redrawn
            # with it, because the wall's tangent turns under the field.
            ANULL = wall_alignment(theta[:, None], X, Y, env, bin_m)
            null_vals = {'elongation': log_elong, 'alignment': ANULL}
            for measure in MEASURES:
                pre = f'arm_a_{nn}_{measure}'
                nv = np.asarray(null_vals[measure], dtype=float)
                if nv.ndim == 1:
                    nv = np.broadcast_to(nv[:, None], DN.shape)
                # Only the fields clear of the wall are in this arm, on both
                # sides of the comparison.
                ov = np.where(clear, obs[measure], np.nan)
                nv = np.where(clear[:, None], nv, np.nan)
                o, o_near, o_far, n_near, n_far = near_far(
                    ov, dn, a_near, a_far)
                nd = near_far(nv, DN, a_near, a_far)[0]
                row[f'{pre}_near'] = float(o_near)
                row[f'{pre}_far'] = float(o_far)
                _stat_block(row, f'{pre}_nearfar', float(o), nd,
                            n_near, n_far)
                if measure != 'elongation':
                    # Arm A's null moves a rigid field, so its ELONGATION does
                    # not change in any draw and the level has the same value
                    # in every one of them. The null standard deviation is then
                    # float noise, not spread, and dividing by it manufactures
                    # a z of any size you like. Alignment does change with
                    # position -- the wall's tangent turns -- so it has a real
                    # null and keeps its level test.
                    _stat_block(row, f'{pre}_level', _nanmean(ov),
                                _nanmean(nv, axis=0))
                if measure == 'elongation':
                    _stat_block(row, f'{pre}_rho',
                                spearman_rows(ov, dn[:, None])[0],
                                spearman_rows(ov, DN))
                po, cnt = profile(ov, dn, bins)
                pn, _ = profile(nv, DN, bins)
                for i, b in enumerate(bins):
                    v = pn[i][np.isfinite(pn[i])]
                    prof_rows.append(dict(
                        tag, arm='A', null=nn, measure=measure, bin=b + 1,
                        dist_lo=DIST_EDGES[b], dist_hi=DIST_EDGES[b + 1],
                        n_fields=int(cnt[i]), observed=po[i],
                        raw_observed=po[i],
                        raw_null=float(v.mean()) if len(v) else np.nan,
                        null_mean=float(v.mean()) if len(v) else np.nan,
                        null_lo=float(np.percentile(v, 2.5)) if len(v) else np.nan,
                        null_hi=float(np.percentile(v, 97.5)) if len(v) else np.nan))

    # --------------------------------------- Arm B: clipping-matched null
    pool, n_donors, unrep = donor_pool(bank, clear, env, G, rng, dn=dn)
    row['n_donors'] = int(n_donors)
    row['n_pool'] = int(len(pool))
    row['pool_unrepresented_frac'] = float(unrep)
    if len(pool):
        pa = pool.log_area.to_numpy(dtype=float)
        row['pool_log_area_mean'] = float(pa.mean())
        row['pool_log_area_sd'] = float(pa.std(ddof=1)) if len(pa) > 1 else np.nan
        la = np.log(np.maximum(area, 1e-12))
        row['lib_log_area_mean'] = float(la.mean())
        row['lib_log_area_sd'] = float(la.std(ddof=1)) if len(la) > 1 else np.nan
    base, n_match = matched_baseline(bank, pool, dn, rng, n_null)
    fields['n_match'] = n_match
    row['n_matched'] = int((n_match >= MIN_MATCH).sum())
    b_near, b_far = tercile_cuts(dn, n_match >= MIN_MATCH)
    row.update(arm_b_near_cut=b_near, arm_b_far_cut=b_far,
               arm_b_near_cut_m=_cut_m(b_near, env),
               arm_b_far_cut_m=_cut_m(b_far, env))
    row['arm_b_tested'] = (row['n_matched'] >= 3 * MIN_BIN
                           and np.isfinite(b_near) and np.isfinite(b_far))
    if row['arm_b_tested']:
        for measure in MEASURES:
            mu, NV = base[measure]
            ex = obs[measure] - mu
            EX = NV - mu[:, None]
            fields[f'arm_b_{measure}_excess'] = ex
            pre = f'arm_b_{measure}'
            o, nr, fr, nn_, nf_ = near_far(ex, dn, b_near, b_far)
            nd, _, _, _, _ = near_far(EX, dn, b_near, b_far)
            row.update({f'{pre}_near': float(np.atleast_1d(nr)[0]),
                        f'{pre}_far': float(np.atleast_1d(fr)[0])})
            _stat_block(row, f'{pre}_nearfar', float(np.atleast_1d(o)[0]),
                        nd, nn_, nf_)
            _stat_block(row, f'{pre}_level', _nanmean(ex),
                        _nanmean(EX, axis=0))
            po, cnt = profile(ex, dn, bins)
            pn, _ = profile(EX, dn, bins)
            # The levels as well as the excess: E1 and E4 show the raw curve
            # beside what clipping alone predicts, which is the only way to
            # see how much of the raw curve was ever the arena's outline.
            praw, _ = profile(obs[measure], dn, bins)
            pbase, _ = profile(mu, dn, bins)
            for i, b in enumerate(bins):
                v = pn[i][np.isfinite(pn[i])]
                prof_rows.append(dict(
                    tag, arm='B', null='matched', measure=measure, bin=b + 1,
                    dist_lo=DIST_EDGES[b], dist_hi=DIST_EDGES[b + 1],
                    n_fields=int(cnt[i]), observed=po[i],
                    raw_observed=praw[i], raw_null=pbase[i],
                    null_mean=float(v.mean()) if len(v) else np.nan,
                    null_lo=float(np.percentile(v, 2.5)) if len(v) else np.nan,
                    null_hi=float(np.percentile(v, 97.5)) if len(v) else np.nan))

    row['tested'] = bool(row['arm_a_tested'] or row['arm_b_tested'])
    if verbose:
        parts = [f'  [{tag["channel"]}] n={len(bank)} ({int(clear.sum())} clear, '
                 f'{row["n_matched"]} matched)']
        for measure in MEASURES:
            if row['arm_a_tested']:
                parts.append(f'A {measure[:5]} {row[f"arm_a_tiling_{measure}_nearfar"]:+.3f}'
                             f' (null {row[f"arm_a_tiling_{measure}_nearfar_null_mean"]:+.3f},'
                             f' p {row[f"arm_a_tiling_{measure}_nearfar_p"]:.2g})')
            if row['arm_b_tested']:
                parts.append(f'B {measure[:5]} {row[f"arm_b_{measure}_nearfar"]:+.3f}'
                             f' (p {row[f"arm_b_{measure}_nearfar_p"]:.2g})')
        print('  '.join(parts), flush=True)
    return row, prof_rows, fields


def load_calibration(path):
    """A --synthetic round run's measured bias, per library and statistic.

    The round libraries are matched to the real ones: same arena, same
    channel, same field count, areas resampled from the same library. So the
    bias that run reports for a statistic is an estimate of this run's bias in
    that same statistic, and can be subtracted from it.
    """
    c = pd.read_csv(path)
    keep = ['env', 'channel'] + [f'{k}_excess' for k, *_ in STAT_KEYS
                                 if f'{k}_excess' in c.columns]
    c = c[[col for col in keep if col in c.columns]].copy()
    return c.rename(columns={f'{k}_excess': f'calib_{k}'
                             for k, *_ in STAT_KEYS})


def apply_calibration(summary, calib):
    """Subtract the measured bias and re-derive each p from it.

    The null standard deviation is unchanged -- the calibration moves where
    zero is, not how noisy the statistic is -- so each p is recomputed as the
    corrected excess over that same sd. The raw value is kept alongside under
    `_excess_raw`, because a correction the reader cannot see is not a
    correction.
    """
    s = summary.merge(calib, on=['env', 'channel'], how='left')
    n_applied = 0
    for key, *_ in STAT_KEYS:
        col, cal = f'{key}_excess', f'calib_{key}'
        if col not in s or cal not in s:
            continue
        s[f'{col}_raw'] = s[col]
        off = s[cal].to_numpy(dtype=float)
        s[col] = s[col].to_numpy(dtype=float) - np.where(np.isfinite(off),
                                                         off, 0.0)
        n_applied += int(np.isfinite(off).sum())
        sd = s[f'{key}_null_sd'].to_numpy(dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            z = s[col].to_numpy(dtype=float) / np.where(sd > 0, sd, np.nan)
        s[f'{key}_p'] = 2.0 * stats.norm.sf(np.abs(z))
        s[f'{key}_p_gt'] = stats.norm.sf(z)
    return s, n_applied


# ------------------------------------------------------------- aggregation

# (statistic key, prefix its near/far columns carry or None, arm, null,
# measure, kind). The prefix is declared rather than sliced off the key: the
# rho statistics have no near/far columns at all.
STAT_KEYS = ([(f'arm_a_{nn}_{m}_nearfar', f'arm_a_{nn}_{m}', 'A', nn, m,
               'near-far') for nn in NULLS for m in MEASURES] +
             [(f'arm_a_{nn}_alignment_level', None, 'A', nn, 'alignment',
               'level') for nn in NULLS] +
             [(f'arm_a_{nn}_elongation_rho', None, 'A', nn, 'elongation',
               'rho') for nn in NULLS] +
             [(f'arm_b_{m}_nearfar', f'arm_b_{m}', 'B', 'matched', m,
               'near-far') for m in MEASURES] +
             [(f'arm_b_{m}_level', None, 'B', 'matched', m, 'level')
              for m in MEASURES])


def arena_table(summary):
    """Per arena, channels averaged against the conservative null sd.

    Six channels in one arena share most of their structure, so their excesses
    are averaged and divided by the mean of their null standard deviations --
    what the standard error would be if the channels were perfectly
    correlated. The arena test can only understate significance.
    """
    rows = []
    for e, g in summary.groupby('env', sort=False):
        for key, base, arm, nn, measure, kind in STAT_KEYS:
            if f'{key}_excess' not in g:
                continue
            ex = g[f'{key}_excess'].to_numpy(dtype=float)
            sd = g[f'{key}_null_sd'].to_numpy(dtype=float)
            ok = np.isfinite(ex) & np.isfinite(sd) & (sd > 0)
            if not ok.any():
                continue
            sdc = float(sd[ok].mean())
            z = float(ex[ok].mean() / sdc)

            def col(name):
                if base is None or f'{base}_{name}' not in g:
                    return np.nan
                v = g[f'{base}_{name}'].to_numpy(dtype=float)[ok]
                return _nanmean(v)

            rows.append(dict(
                env=e, shape=arena_shape(e), group=arena_group(e),
                env_area_m2=float(g.env_area_m2.iloc[0]), arm=arm, null=nn,
                measure=measure, kind=kind, n_channels=int(ok.sum()),
                n_positive=int((ex[ok] > 0).sum()),
                observed=_nanmean(g[key].to_numpy(dtype=float)[ok]),
                near=col('near'), far=col('far'),
                excess=float(ex[ok].mean()), sd_conservative=sdc, z=z,
                p=float(2.0 * stats.norm.sf(abs(z)))))
    a = pd.DataFrame(rows)
    if len(a):
        a['q'] = np.nan
        for _, idx in a.groupby(['arm', 'null', 'measure', 'kind']).groups.items():
            a.loc[idx, 'q'] = WP.bh_q(a.loc[idx, 'p'])
    return a


def shape_contrast(arena):
    """Circular against rectangular, on the arena-level excesses.

    Four arenas against four, so this is a weak comparison and the report says
    so; it is here because it is the comparison the experiment was asked for.
    Welch rather than Student: the corridor's numbers have no reason to have
    the discs' variance. The square and the corridor are also given their own
    row each, because a 2 m corridor is a different question from a 10 m
    square with the same walls.
    """
    rows = []
    if not len(arena):
        return pd.DataFrame(rows)
    for (arm, nn, measure, kind), g in arena.groupby(
            ['arm', 'null', 'measure', 'kind']):
        by = {k: v.excess.to_numpy(dtype=float) for k, v in g.groupby('group')}
        circ = by.get('circular', np.array([]))
        rect = by.get('rectangular', np.array([]))
        circ, rect = circ[np.isfinite(circ)], rect[np.isfinite(rect)]
        t, p = ((np.nan, np.nan) if min(len(circ), len(rect)) < 2 else
                stats.ttest_ind(circ, rect, equal_var=False))
        r = dict(arm=arm, null=nn, measure=measure, kind=kind,
                 n_circular=len(circ), n_rectangular=len(rect),
                 circular=float(circ.mean()) if len(circ) else np.nan,
                 rectangular=float(rect.mean()) if len(rect) else np.nan,
                 difference=(float(circ.mean() - rect.mean())
                             if len(circ) and len(rect) else np.nan),
                 t=float(t), p=float(p))
        for shp in ('disc', 'square', 'corridor'):
            v = g[g['shape'] == shp].excess.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            r[shp] = float(v.mean()) if len(v) else np.nan
            r[f'n_{shp}'] = len(v)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------- calibration

def synthetic_bank(bank, env, G, rng, mode):
    """A library of fields whose true shape is known, recorded as a real one is.

    Areas are resampled from the real library, so the size distribution, the
    Rule 8 floor and the range of template sizes are the real ones. Each field
    is then laid down as an ellipse of known axis ratio and orientation at a
    uniformly random floor bin, CLIPPED to the floor, and measured through
    `clipped_shape` -- so its recorded shape is what the real pipeline would
    have recorded for it, cut by the wall and all.

      round    axis ratio 1, orientation uniform. Every scrap of elongation
               and alignment in the result is the wall cutting a circle, so
               both arms must find nothing. This is the false-positive check.
      planted  axis ratio 1 + PLANT_STRENGTH * (1 - d)^2 with the major axis
               on the wall's tangent, d being the normalised wall distance of
               the field's centre. A real, wall-locked elongation that rises
               toward the wall, so both arms must find it. This is the power
               check.

    A null result from this experiment means little without both.
    """
    bin_m = float(np.sqrt(G['bin_area']))
    in_env = G['in_env']
    env_i, env_j = np.nonzero(in_env)
    n = len(bank)
    areas = rng.choice(bank.area_env_m2.to_numpy(dtype=float), size=n,
                       replace=True)
    pick = rng.integers(0, len(env_i), size=n)
    ci, cj = env_i[pick], env_j[pick]
    d0 = norm_wall_distance(G['xc'][ci], G['yc'][cj], env)
    tangent, amb = wall_frame(G['xc'][ci], G['yc'][cj], env, bin_m)
    if mode == 'planted':
        # Both measures have to vary WITH distance, or a near-minus-far
        # statistic cannot see them. An orientation locked to the wall at
        # every distance is a uniform preference, which near-minus-far is
        # blind to by construction -- the first draft planted that and the
        # power check failed for the wrong reason.
        lean = (1.0 - np.clip(d0, 0.0, 1.0)) ** 2
        ratio = 1.0 + PLANT_STRENGTH * lean
        th = np.where(amb | (rng.random(n) > lean),
                      rng.uniform(0.0, np.pi, n), tangent)
    else:
        ratio = np.ones(n)
        th = rng.uniform(0.0, np.pi, n)
    rows = []
    for k in range(n):
        a = float(np.sqrt(max(areas[k], 1e-12) * ratio[k] / np.pi))
        b = float(np.sqrt(max(areas[k], 1e-12) / (np.pi * ratio[k])))
        n_bins = max(1, int(round(areas[k] / G['bin_area'])))
        di, dj = ellipse_offsets(a, b, float(th[k]), G, n_bins)
        nv, cx, cy, aa, bb, tt = clipped_shape(np.array([ci[k]]),
                                               np.array([cj[k]]), di, dj, G)
        if nv[0] < 1 or not np.isfinite(aa[0]) or bb[0] <= 0:
            continue
        rows.append(dict(
            node_id=k, depth=0, n_members=int(nv[0]), sigma_feature=np.nan,
            area_env_m2=float(nv[0] * G['bin_area']),
            radius_env_m=float(np.sqrt(nv[0] * G['bin_area'] / np.pi)),
            semi_major_m=float(aa[0]), semi_minor_m=float(bb[0]),
            elongation=float(aa[0] / bb[0]), orientation_rad=float(tt[0]),
            centroid_x=float(cx[0]), centroid_y=float(cy[0]),
            dist_to_wall_m=float(R.wall_distance(cx[0], cy[0], env)),
            cc_frac=1.0, n_components=1, split_half_iou=0.0,
            parent_id=-1, is_leaf=True,
            planted_ratio=float(ratio[k]), planted_theta=float(th[k])))
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    r_eq = out.radius_env_m.to_numpy(dtype=float)
    out['scale_band'] = R.assign_bands(r_eq, max(float(r_eq.min()), 1e-6),
                                       BASE_C['BAND_RATIO'])
    return out


# ------------------------------------------------------------------ figures

FIGURES_WRITTEN = []


def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    FIGURES_WRITTEN.append(p)
    print(f'  {p}', flush=True)


def prune_orphan_figures():
    """Delete E<digit>*.png this run did not write, as Experiment 2 does."""
    keep = {os.path.abspath(p) for p in FIGURES_WRITTEN}
    for p in sorted(glob.glob(os.path.join(FIG_DIR, 'E[0-9]*.png'))):
        if os.path.abspath(p) not in keep:
            os.remove(p)
            print(f'  pruned orphaned figure {os.path.basename(p)}', flush=True)


def _mid(row):
    return 0.5 * (row.dist_lo + row.dist_hi)


def _grid(envs, nrow=1, w=3.3, h=2.9, sharey=True):
    ncol = int(np.ceil(len(envs) / nrow))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharey=sharey,
                             figsize=(w * ncol, h * nrow))
    flat = [ax for r in axes for ax in r]
    for ax in flat[len(envs):]:
        ax.axis('off')
    return fig, flat[:len(envs)]


def _decorate(ax, e, first, xlabel, ylabel):
    ax.set_title(e, fontsize=9, color=SHAPE_COLORS.get(arena_shape(e), INK))
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color(RULE_GRAY)
    ax.set_xlabel(xlabel, fontsize=7.5, color=MUTED)
    if first:
        ax.set_ylabel(ylabel, fontsize=8, color=INK)


def fig_raw_and_null(prof, envs, chans, measure, name, title, ylabel):
    """E1 / E4: the raw curve against wall distance, and what clipping alone
    predicts for it.

    Solid is what the fields actually measure. Dashed is the matched null: the
    same library's shapes, orientation forgotten, dropped at the same distance
    and cut by the same wall. The gap between them is the only part that could
    be a place-field property; where they run together, the raw curve was the
    arena's outline all along.
    """
    d = prof[(prof.arm == 'B') & (prof.measure == measure)]
    if not len(d):
        return
    fig, axes = _grid(envs, nrow=2)
    leg = None
    for ax, e in zip(axes, envs):
        de = d[d.env == e]
        # Zero is "nothing to see" for both measures: log elongation 0 is a
        # circle, alignment 0 is no relation to the wall.
        ax.axhline(0.0, color=RULE_GRAY, lw=0.8, zorder=0)
        for c in chans:
            g = de[de.channel == c].sort_values('bin')
            if not len(g):
                continue
            x = _mid(g)
            col = CHANNEL_COLORS.get(c, '0.4')
            ax.plot(x, g.raw_observed, '-', lw=1.1, color=col, label=c)
            ax.plot(x, g.raw_null, '--', lw=0.9, color=col, alpha=0.75)
            leg = leg or ax
        _decorate(ax, e, ax is axes[0] or ax is axes[len(axes) // 2],
                  'wall distance: 0 = as near the wall as a field\ncan sit, 1 = centre or midline', ylabel)
    if leg is not None:
        leg.legend(fontsize=5.5, frameon=False, ncol=2, loc='best')
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, name)


def fig_excess_profile(prof, envs, chans, measure, name, title, ylabel):
    """E2 / E5: observed minus the clipping-matched null, bin by bin.

    Zero is what clipping alone would produce. Grey is the median across
    channels of each channel's null 95% range for that bin, so a point outside
    it is beyond what the null does by chance.
    """
    d = prof[(prof.arm == 'B') & (prof.measure == measure)]
    if not len(d):
        return
    v = d[['observed', 'null_lo', 'null_hi']].to_numpy(dtype=float)
    lim = 1.15 * float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 0.1
    fig, axes = _grid(envs, nrow=2)
    leg = None
    for ax, e in zip(axes, envs):
        de = d[d.env == e]
        ax.axhline(0, color=RULE_GRAY, lw=0.8, zorder=0)
        if len(de):
            b = de.groupby('bin')[['dist_lo', 'dist_hi', 'null_lo',
                                   'null_hi']].median()
            ax.fill_between(0.5 * (b.dist_lo + b.dist_hi), b.null_lo, b.null_hi,
                            color='0.72', alpha=0.45, lw=0, zorder=1)
            for c in chans:
                g = de[de.channel == c].sort_values('bin')
                if len(g):
                    ax.plot(_mid(g), g.observed, '-', lw=1.0, alpha=0.85,
                            color=CHANNEL_COLORS.get(c, '0.4'), label=c)
            m = de.groupby('bin').agg(lo=('dist_lo', 'first'),
                                      hi=('dist_hi', 'first'),
                                      y=('observed', 'mean')).sort_index()
            ax.plot(0.5 * (m.lo + m.hi), m.y, 'o-', color=INK, lw=1.6, ms=3,
                    label='channel mean', zorder=3)
            leg = leg or ax
        ax.set_ylim(-lim, lim)
        _decorate(ax, e, ax is axes[0] or ax is axes[len(axes) // 2],
                  'wall distance: 0 = as near the wall as a field\ncan sit, 1 = centre or midline', ylabel)
    if leg is not None:
        leg.legend(fontsize=5.5, frameon=False, ncol=2, loc='best')
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, name)


def fig_arm_a_forest(summary, name):
    """E3: Arm A, one row per library, near-wall minus far-from-wall.

    These fields are never clipped, so the measurement needs no correction at
    all; the bar is the null's 95% range for the same statistic with the same
    shapes moved to random positions they still fit whole. A point outside its
    bar is the clean result.
    """
    t = summary[summary.arm_a_tested.astype(bool)]
    if not len(t):
        return
    # Reading order, the same one every other figure uses: discs smallest
    # first, then squares, then corridors. Sorting on the shape name would put
    # the corridor first and nothing would line up.
    t = t.assign(_ord=t['shape'].map(SHAPE_ORDER).fillna(3))
    t = t.sort_values(['_ord', 'env_area_m2', 'env', 'channel'])
    fig, axes = plt.subplots(1, len(MEASURES), squeeze=False, sharey=True,
                             figsize=(4.6 * len(MEASURES),
                                      max(3.0, 0.20 * len(t) + 1.4)))
    y = np.arange(len(t))
    for ax, measure in zip(axes[0], MEASURES):
        pre = f'arm_a_tiling_{measure}_nearfar'
        if pre not in t:
            continue
        ax.axvline(0, color=RULE_GRAY, lw=0.8)
        lo = t[f'{pre}_null_lo'].to_numpy(dtype=float)
        hi = t[f'{pre}_null_hi'].to_numpy(dtype=float)
        ax.hlines(y, lo, hi, color='0.72', lw=3.0, alpha=0.6)
        for shp, col in SHAPE_COLORS.items():
            sel = (t['shape'] == shp).to_numpy()
            if sel.any():
                ax.plot(t[pre].to_numpy(dtype=float)[sel], y[sel], 'o', ms=4,
                        color=col, label=shp)
        ax.set_xlabel(f'near minus far\n{MEASURE_UNITS[measure]}', fontsize=7.5,
                      color=MUTED)
        ax.margins(x=0.12)          # so a point at the extreme is not on the spine
        ax.set_title(measure, fontsize=9, color=INK)
        ax.tick_params(labelsize=6, colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(RULE_GRAY)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels([f'{r.env}  {r.channel}' for r in t.itertuples()],
                               fontsize=5.5)
    axes[0][0].set_ylim(-0.8, len(t) - 0.2)
    axes[0][-1].legend(fontsize=6.5, frameon=False, loc='lower right',
                       title='arena', title_fontsize=6.5)
    fig.suptitle('E3  Arm A: fields clear of the wall, where no clipping can '
                 'reach the measurement\ngrey = the 95% range of the same '
                 'statistic with those shapes moved at random (tiling null)',
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, name)


def fig_angle_histograms(fields, summary, name):
    """E6: which way fields point, near the wall against far from it.

    The angle between a field's major axis and the nearest wall, 0 = along the
    wall, 90 = straight at it. Channels pooled, one panel per arena shape. If
    fields lined up with walls for a reason of their own, the near-wall
    histogram would lean left of the far one by more than the clipping null.
    """
    if 'corner_ambiguous' not in fields or 'wall_alignment' not in fields:
        return
    t = summary[summary.tested][['env', 'channel']]
    f = fields.merge(t, on=['env', 'channel'], how='inner')
    f = f[~f.corner_ambiguous.fillna(True).astype(bool)]
    if not len(f):
        return
    shapes = [s for s in ('disc', 'square', 'corridor')
              if (f.env.map(arena_shape) == s).any()]
    fig, axes = plt.subplots(1, len(shapes), squeeze=False, sharey=True,
                             figsize=(3.5 * len(shapes), 3.1))
    bins = np.linspace(0.0, 90.0, 10)
    for ax, shp in zip(axes[0], shapes):
        fs = f[f.env.map(arena_shape) == shp]
        # cos(2x) = alignment, so the angle back out is half its arccos.
        ang = np.degrees(0.5 * np.arccos(
            np.clip(fs.wall_alignment.to_numpy(dtype=float), -1.0, 1.0)))
        dnv = fs.wall_dist_norm.to_numpy(dtype=float)
        for lab, sel, col in (
                (f'near (d <= {E6_NEAR:g})', dnv <= E6_NEAR, DIST_COLORS[0]),
                (f'far (d >= {E6_FAR:g})', dnv >= E6_FAR, DIST_COLORS[-1])):
            v = ang[sel & np.isfinite(ang)]
            if len(v):
                ax.hist(v, bins=bins, density=True, histtype='step', lw=1.6,
                        color=col, label=f'{lab}  n={len(v)}')
        ax.axhline(1.0 / 90.0, color=RULE_GRAY, lw=0.9, ls=':')
        ax.set_title(shp, fontsize=9, color=SHAPE_COLORS.get(shp, INK))
        ax.set_xlim(0, 90)
        ax.set_xticks([0, 30, 60, 90])
        ax.set_xlabel('angle to the nearest wall (deg)\n0 = along it, '
                      '90 = straight at it', fontsize=7.5, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.legend(fontsize=6, frameon=False, loc='upper right')
        for sp in ax.spines.values():
            sp.set_color(RULE_GRAY)
    axes[0][0].set_ylabel('density', fontsize=8, color=INK)
    fig.suptitle('E6  which way fields point relative to the nearest wall, '
                 'channels pooled\ndotted = flat, what no relation to the wall '
                 'looks like', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, name)


def fig_shape_contrast(arena, name):
    """E7: the answer to the discs-against-rectangles question.

    One point per arena, the channel-averaged excess, with the four discs, two
    squares and two corridors kept visibly apart. Four arenas a side is a weak
    comparison and the figure shows the points rather than a bar, so the
    spread is as visible as the means.
    """
    if not len(arena):
        return
    keep = arena[arena.kind.isin(('near-far', 'level')) &
                 arena.null.isin(('tiling', 'matched'))]
    if not len(keep):
        return
    cols = [(arm, m) for arm in ('A', 'B') for m in MEASURES]
    kinds = [k for k in ('near-far', 'level') if (keep.kind == k).any()]
    fig, axes = plt.subplots(len(kinds), len(cols), squeeze=False,
                             figsize=(2.9 * len(cols), 3.1 * len(kinds)))
    order = sorted(SHAPE_ORDER, key=SHAPE_ORDER.get)
    for row, kind in enumerate(kinds):
      for ax, (arm, measure) in zip(axes[row], cols):
        g = keep[(keep.arm == arm) & (keep.measure == measure) &
                 (keep.kind == kind)]
        if not len(g):
            ax.text(0.5, 0.5, 'no null for this\nstatistic (see report)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=7, color=MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(RULE_GRAY)
            ax.set_title(f'Arm {arm} — {measure}', fontsize=9, color=INK)
            continue
        ax.axhline(0, color=RULE_GRAY, lw=0.8)
        for i, shp in enumerate(order):
            v = g[g['shape'] == shp].excess.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            x = i + np.linspace(-0.13, 0.13, len(v))
            ax.plot(x, v, 'o', ms=6, color=SHAPE_COLORS[shp], alpha=0.9)
            ax.hlines(v.mean(), i - 0.26, i + 0.26, color=SHAPE_COLORS[shp],
                      lw=2.2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=7.5, color=MUTED)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(f'Arm {arm} — {measure}', fontsize=9, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(RULE_GRAY)
    for row, kind in enumerate(kinds):
        axes[row][0].set_ylabel(f'{kind}, beyond the null\n'
                                '(channels averaged per arena)', fontsize=8,
                                color=INK)
    fig.suptitle('E7  does wall proximity act differently in a round arena '
                 'than in a straight-walled one\none point per arena, bar = '
                 'the mean of that shape. Top row: near minus far. Bottom '
                 'row: the level over every field.', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, name)


# -------------------------------------------------------------------- report

def _sign_test(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    k = int((x > 0).sum())
    return k, len(x), (float(stats.binom.sf(k - 1, len(x), 0.5))
                       if len(x) else np.nan)


def _fmt_ratio(log_v):
    """A log-elongation difference read back as the factor it is."""
    return '--' if not np.isfinite(log_v) else f'x{np.exp(log_v):.3f}'


class WallElongationReport(ExperimentReport):
    experiment = 'wall-elongation'

    def title(self):
        a = getattr(self, 'arena', None)
        if a is None or not len(a):
            return 'no testable libraries'
        bits = []
        for arm, nn in (('A', 'tiling'), ('B', 'matched')):
            g = a[(a.arm == arm) & (a.null == nn) & (a.kind == 'near-far') &
                  (a.measure == 'elongation')]
            if len(g):
                hit = int(((g.q < ALPHA) & (g.excess > 0)).sum())
                bits.append(f'arm {arm} elongation {g.excess.mean():+.3f} '
                            f'({hit}/{len(g)} arenas at q<{ALPHA})')
        mode = getattr(self, 'synthetic', None)
        pre = f'[synthetic {mode}] ' if mode else ''
        return pre + ('; '.join(bits) or 'nothing testable')

    def figures(self):
        return sorted(FIGURES_WRITTEN)

    def data_files(self):
        return [p for p in (f'{self.out_dir}/summary.csv',
                            f'{self.out_dir}/arena_summary.csv',
                            f'{self.out_dir}/shape_contrast.csv',
                            f'{self.out_dir}/profiles.csv')
                if os.path.exists(p)]

    def _verify_worst(self):
        v = getattr(self, 'verified', None) or {}
        return max([max(d['a'], d['b'], d['axis']) for d in v.values()] or [np.nan])

    def _arena_block(self, arm, nn, measure, kind='near-far'):
        a = self.arena
        g = a[(a.arm == arm) & (a.null == nn) & (a.kind == kind) &
              (a.measure == measure)]
        if not len(g):
            if arm == 'A' and kind == 'level' and measure == 'elongation':
                return ['  (no null exists: Arm A moves a rigid field, so its '
                        'elongation is the same in every draw and the level '
                        'cannot vary. Not a failure -- the statistic is not '
                        'defined. Read the near-far row above, or Arm B.)']
            return ['  (not tested in any arena)']
        g = g.assign(_ord=g['shape'].map(SHAPE_ORDER).fillna(3))
        g = g.sort_values(['_ord', 'env_area_m2'])
        L = [f'  {"arena":17s} {"shape":9s} {"near":>7s} {"far":>7s} '
             f'{"observed":>9s} {"beyond null":>11s} {"z":>6s} {"q":>8s}  ch>0']
        for r in g.itertuples():
            near = f'{r.near:+7.3f}' if np.isfinite(r.near) else f'{"--":>7s}'
            far = f'{r.far:+7.3f}' if np.isfinite(r.far) else f'{"--":>7s}'
            L.append(f'  {r.env:17s} {r.shape:9s} {near} {far} '
                     f'{r.observed:+9.3f} '
                     f'{r.excess:+11.3f} {r.z:+6.1f} {r.q:8.2g}  '
                     f'{r.n_positive}/{r.n_channels}')
        return L

    def _channel_block(self, arm, nn, measure):
        s = self.results
        key = (f'arm_a_{nn}_{measure}_nearfar' if arm == 'A'
               else f'arm_b_{measure}_nearfar')
        col = f'{key}_excess'
        t = s[s[f'arm_{arm.lower()}_tested'].astype(bool)] if \
            f'arm_{arm.lower()}_tested' in s else s
        if col not in t or not len(t):
            return ['  (not tested)']
        L = [f'  {"channel":9s} {"arenas":>7s} {"mean":>9s} {"positive":>9s}']
        present = list(dict.fromkeys(t.channel.tolist()))
        order = [c for c in CHANNELS if c in present] + \
                [c for c in present if c not in CHANNELS]
        for c in order:
            v = t[t.channel == c][col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            L.append(f'  {c:9s} {len(v):7d} {v.mean():+9.3f} '
                     f'{int((v > 0).sum()):>5d}/{len(v):<3d}')
        return L

    def body(self):
        s, a, sc = self.results, self.arena, self.contrast
        if s is None or not len(s):
            return 'No field libraries were analysed.'
        S = self.section
        out = []
        mode = getattr(self, 'synthetic', None)

        if mode:
            out.append(S('THIS IS A CALIBRATION RUN, NOT A RESULT', '\n'.join([
                f'Every library below is synthetic (--synthetic {mode}). The '
                f'real field libraries were not used except to borrow their '
                f'size distributions.', '',
                'round   : circular fields at random orientations. There is '
                'no elongation and no alignment in this data, so anything '
                'either arm reports is a false positive and the run measures '
                'the false-positive rate.',
                'planted : elongation rising toward the wall with the major '
                'axis on the wall tangent. The effect is real and known, so '
                'what the arms report measures the power.'])))

        out.append(S('THE QUESTION', '\n'.join([
            'Are place fields more elongated the closer they sit to a wall, '
            'and do they line up with it? Asked per channel, and asked '
            'separately of round arenas and straight-walled ones. Libraries '
            f'are Experiment 2\'s, unchanged: EXTENT_PCTL {PCTL}, ACT_THRESH '
            f'{THRESH:g}, Rule 2 off, LAMBDA 0.'])))

        out.append(S('WHY A RAW CURVE ANSWERS NOTHING', '\n'.join([
            'Rule 7 measures a field from the second moments of its mask, and '
            'the mask stops at the wall. A field that reaches past the wall is '
            'cut, and a cut blob\'s moments are elongated ALONG the wall. So '
            'both of this experiment\'s measures come out positive near a wall '
            'in a pipeline containing no anisotropy whatever -- they are a '
            'picture of the arena\'s outline.', '',
            'A second bias rides with the first: a large field cannot sit near '
            'a wall without being cut, so near-wall fields are small fields, '
            'and a small field\'s axis ratio is estimated from few bins, where '
            'it is noisier and biased upward.', '',
            'E1 and E4 show both curves -- what the fields measure, and what '
            'clipping alone predicts for them. The distance between the two is '
            'the whole result. Everything else in this report is machinery for '
            'measuring that distance honestly.'])))

        n_clear_total = (int(np.nansum(s.n_clear.to_numpy(dtype=float)))
                         if 'n_clear' in s else 0)
        out.append(S('THE TWO ARMS', '\n'.join([
            f'ARM A -- the {n_clear_total} fields whose recorded '
            'ellipse, grown by one bin, lies wholly on the floor. Nothing '
            'clipped them, so nothing needs correcting: the measurement is '
            'clean and the statistic is simply near-wall mean minus '
            'far-from-wall mean. The null keeps each shape rigid and moves it '
            'to a random position where it still fits whole, so a field keeps '
            'its elongation and only its position is redrawn (Experiment 3\'s '
            'placement null, uniform and Rule 11 tiling). Alignment is redrawn '
            'with the position, because the wall\'s tangent turns under the '
            'field. This arm is unbiased and is the one to believe. Its limit '
            'is who it can speak for: at the near-wall end, only small '
            'fields fit whole.', '',
            'ARM B -- every field, against a null that clips the way the data '
            'was clipped. Each library\'s own clear fields become donors: '
            f'their ellipses are re-laid at {POOL_ORIENTATIONS} orientations, '
            'dropped at random floor positions, cut by the floor, and '
            're-measured through the same arithmetic. Randomising a donor\'s '
            'orientation is what makes it a null for alignment -- it keeps its '
            'axis ratio and forgets which way it pointed. Each real field is '
            'then matched to placements of the same VISIBLE area within a '
            f'factor of {np.exp(AREA_TOL_LOG):.2f} at the same wall distance '
            f'within {DIST_TOL:g}, which is what holds size and position '
            'still.', '',
            'Arm B is tested on the NEAR minus FAR difference of its excess, '
            'not on the excess itself. The level of the excess depends on the '
            'donor pool\'s size mix, and donors have to fit whole, which '
            'under-represents large fields; the difference across distance is '
            'first-order insensitive to that. The levels are printed beside it '
            'and are not the test. This mirrors Experiment 3, which tests '
            'LARGE minus SMALL rather than LARGE.', '',
            f'`clipped_shape`, the code that measures a null placement, is '
            f'checked against rules.field_shape itself on {VERIFY_N} real '
            f'masks per arena, clipped ones included, to {VERIFY_TOL:g}. '
            'Without that check a difference between two implementations '
            'would read as the effect.'] +
            ([f'Worst disagreement measured this run: '
              f'{self._verify_worst():.1e}.'] if getattr(self, 'verified', None)
             else []))))

        calib = getattr(self, 'calibration', None)
        if calib:
            out.append(S('A MEASURED BIAS HAS BEEN SUBTRACTED', '\n'.join([
                f'Every statistic below has had the bias measured by a '
                f'--synthetic round run subtracted from it, and its p '
                f're-derived from the corrected value against the same null '
                f'standard deviation. Applied to '
                f'{getattr(self, "n_calib", 0)} library x statistic values '
                f'from:', f'  {calib}', '',
                'The raw, uncorrected values are in summary.csv under '
                '*_excess_raw. Arm B needs this correction in a curved arena; '
                'Arm A came out unbiased in development and the correction '
                'there should be small.'])))
        else:
            out.append(S('NO CALIBRATION WAS SUBTRACTED', '\n'.join([
                'This run had no --calibration, so Arm B\'s numbers are '
                'uncorrected. In development Arm B\'s near-minus-far '
                'elongation carried a residual positive bias in a curved '
                'arena, and its alignment one in the 2 m corridor, on data '
                'with no effect planted in it. Run --synthetic round over the '
                'same arenas and pass its summary.csv back with '
                '--calibration before reading Arm B as a result. Arm A needs '
                'no correction.'])))

        out.append(S('WHAT THE NUMBERS MEAN', '\n'.join([
            'elongation  semi-major / semi-minor, averaged in logs. +0.10 '
            'means 10% longer for its width; the tables print the factor too.',
            'alignment   cos(2 x angle to the nearest wall\'s tangent). +1 the '
            'field lies along the wall, -1 it points straight at it, 0 no '
            'relation. The angle is doubled because an axis has no direction.',
            f'near / far  wall distance on the range a field can actually '
            f'occupy: 0 is as near the wall as the collection keep-out allows '
            f'any field\'s centre to be, 1 is the disc\'s centre or the '
            'rectangle\'s midline. Dividing by the half-width instead put '
            'the whole near-wall bin of the 2 m corridor below the keep-out, '
            'where no field can be, and every corridor statistic came out '
            'empty.',
            'near / far  the cut is a TERCILE of each arm\'s own fields, not '
            'a fixed distance, and the tables print it in metres. A fixed cut '
            'cannot work for Arm A: a field has to fit whole to be in it, '
            'which in the 2 m corridor puts its nearest field 0.38 m from the '
            'wall, so a fixed near-wall bin there holds nothing. Read the cut '
            'before reading the number -- in a corridor "near the wall" is a '
            'third of a metre, not a hand\'s breadth.',
            'level       the same measure averaged over every field in the '
            'arm, against the same null. A preference that does not vary with '
            'distance is invisible to near-minus-far, and for alignment that '
            'is the likelier shape of a real effect, so both are tested.',
            'A positive near-far means the near-wall fields are longer, or '
            'more wall-aligned, than the far ones.', '',
            'Per library, p comes from a normal fitted to the null draws and q '
            'is Benjamini-Hochberg across libraries. Per arena the channels '
            'are averaged and divided by the mean of their null standard '
            'deviations -- the value for perfectly correlated channels -- '
            'because six channels in one arena share most of their structure. '
            'The arena test can only understate significance.'])))

        for measure in MEASURES:
            blocks = [MEASURE_UNITS[measure], '']
            for kind in ('near-far', 'level'):
                for arm, nn in (('A', 'uniform'), ('A', 'tiling'),
                                ('B', 'matched')):
                    blocks.append(f'Arm {arm}, {nn} null, {kind}, channels '
                                  f'averaged per arena:')
                    blocks += self._arena_block(arm, nn, measure, kind)
                    blocks.append('')
            for arm, nn in (('A', 'tiling'), ('B', 'matched')):
                key = (f'arm_a_{nn}_{measure}_nearfar' if arm == 'A'
                       else f'arm_b_{measure}_nearfar')
                if f'{key}_excess' not in s:
                    continue
                t = s[s[f'arm_{arm.lower()}_tested'].astype(bool)]
                ex = t[f'{key}_excess']
                q = t[f'{key}_q'] if f'{key}_q' in t else pd.Series(dtype=float)
                up = int(((q < ALPHA) & (ex > 0)).sum()) if len(q) else 0
                dn_ = int(((q < ALPHA) & (ex < 0)).sum()) if len(q) else 0
                k, n, p = _sign_test(ex)
                blocks += [f'Arm {arm} per library at q < {ALPHA}: {up}/{len(t)} '
                           f'positive, {dn_}/{len(t)} negative',
                           f'  sign test over libraries: {k}/{n} positive, '
                           f'p {p:.2g} (assumes channels independent, which '
                           f'they are not -- optimistic)']
                if measure == 'elongation':
                    blocks.append(f'  mean excess as a factor: '
                                  f'{_fmt_ratio(_nanmean(ex.to_numpy(dtype=float)))}')
                blocks += ['', f'Arm {arm} by channel (excess beyond the null, '
                               f'one value per library):']
                blocks += self._channel_block(arm, nn, measure)
                blocks.append('')
            out.append(S(measure.upper(), '\n'.join(blocks)))

        if sc is not None and len(sc):
            L = ['Four discs against two squares and two corridors. At four '
                 'arenas a side this is a weak comparison and a null result '
                 'here is close to uninformative; it is reported because it is '
                 'the comparison the experiment was asked for.', '',
                 f'  {"arm":4s} {"measure":11s} {"circular":>9s} '
                 f'{"rect":>9s} {"diff":>9s} {"p":>8s}   '
                 f'{"disc":>8s} {"square":>8s} {"corridor":>8s}']
            for r in sc[sc.kind.isin(('near-far', 'level'))].itertuples():
                if r.null not in ('tiling', 'matched'):
                    continue
                L.append(f'  {r.arm:2s}{r.kind[:1]:2s} {r.measure:11s} '
                         f'{r.circular:+9.3f} '
                         f'{r.rectangular:+9.3f} {r.difference:+9.3f} '
                         f'{r.p:8.2g}   {r.disc:+8.3f} {r.square:+8.3f} '
                         f'{r.corridor:+8.3f}')
            L += ['', 'The corridor is 2 m wide. Every field wide enough to '
                      'span it is cut on both sides, its wall frame barely '
                      'turns along its length, and its whole wall-distance '
                      'range is 1 m. It belongs here as the extreme case of '
                      '"everything is near a wall", not as a place to read a '
                      'distance profile, and it is reported on its own line '
                      'for that reason.']
            out.append(S('ROUND ARENAS AGAINST STRAIGHT-WALLED ONES', '\n'.join(L)))

        not_a = s[~s.arm_a_tested.astype(bool)]
        not_b = s[~s.arm_b_tested.astype(bool)]
        if len(not_a) or len(not_b):
            out.append(S('NOT TESTED', '\n'.join(
                [f'Arm A needs {MIN_CLEAR} fields clear of the wall; Arm B '
                 f'needs {MIN_DONORS} donors and {2 * MIN_BIN} fields with '
                 f'{MIN_MATCH} matches each.'] +
                [f'  {r.env} {r.channel}: {r.n_fields} fields, '
                 f'{getattr(r, "n_clear", float("nan")):.0f} clear -- no Arm A'
                 if np.isfinite(getattr(r, 'n_clear', np.nan)) else
                 f'  {r.env} {r.channel}: {r.n_fields} fields -- no Arm A'
                 for r in not_a.itertuples()] +
                [f'  {r.env} {r.channel}: {r.n_matched} matched -- no Arm B'
                 for r in not_b.itertuples()])))

        cols = ['env', 'channel', 'shape', 'n_fields', 'n_clear', 'n_matched',
                'arm_a_tiling_elongation_nearfar',
                'arm_a_tiling_elongation_nearfar_q',
                'arm_a_tiling_alignment_nearfar',
                'arm_a_tiling_alignment_nearfar_q',
                'arm_b_elongation_nearfar', 'arm_b_elongation_nearfar_q',
                'arm_b_alignment_nearfar', 'arm_b_alignment_nearfar_q']
        out.append(S('Per arena and channel',
                     self.table(s[[c for c in cols if c in s.columns]])))
        return '\n'.join(out)


# ----------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--envs', default=','.join(ENVS),
                   help='default: the eight collected arenas')
    p.add_argument('--channels', default=','.join(CHANNELS))
    p.add_argument('--n-null', type=int, default=N_NULL,
                   help='null realisations of each library, per null')
    p.add_argument('--seed', type=int, default=0,
                   help='seeds the nulls and the donor pool only; the '
                        'libraries keep Experiment 2\'s seed')
    p.add_argument('--calibration', default=None, metavar='SUMMARY_CSV',
                   help='summary.csv from a --synthetic round run over the '
                        'same arenas and channels. Its measured bias is '
                        'subtracted from every statistic and each p is '
                        're-derived from the corrected value; the raw values '
                        'are kept as *_excess_raw. Arm B needs this in a '
                        'curved arena -- see the Calibration section.')
    p.add_argument('--synthetic', choices=('round', 'planted'), default=None,
                   help='replace every library with fields of known shape and '
                        'rerun: "round" measures the false-positive rate, '
                        '"planted" the power. Run both before trusting a '
                        'result.')
    p.add_argument('--rebuild', action='store_true',
                   help='rebuild field libraries even where Experiment 2\'s '
                        'cache already holds them')
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--no-email', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    envs = [e.strip() for e in args.envs.split(',') if e.strip()]
    chans = [c.strip() for c in args.channels.split(',') if c.strip()]
    rng = np.random.default_rng(args.seed)
    base_C = dict(BASE_C, USE_GPU=not args.no_gpu)
    out_dir = OUT_DIR + (f'_synthetic_{args.synthetic}' if args.synthetic else '')
    fig_dir = FIG_DIR + (f'_synthetic_{args.synthetic}' if args.synthetic else '')
    # _save and prune_orphan_figures read FIG_DIR, and a synthetic run must
    # not overwrite the real run's figures.
    globals()['FIG_DIR'] = fig_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=' * 72)
    print('Wall elongation | are fields longer, and wall-aligned, near a wall?')
    print(f'  envs     : {envs}')
    print(f'  channels : {chans}')
    print(f'  fields   : Experiment 2 config, EXTENT_PCTL {PCTL}, '
          f'ACT_THRESH {THRESH:g}, Rule 2 off, LAMBDA 0')
    print(f'  libraries: {"rebuilt" if args.rebuild else "from cache where present"}'
          f' ({BANK_DIR})')
    if args.synthetic:
        print(f'  SYNTHETIC: {args.synthetic} -- fields of known shape, '
              f'real libraries used only for their size distribution')
    print('  calib    : ' + (args.calibration or
                             'NONE -- Arm B in a curved arena is uncorrected'))
    print(f'  measures : elongation (log a/b), alignment (cos 2x to the wall)')
    print(f'  near/far : terciles of each arm\'s own fields '
          f'(<= {NEAR_Q:.2f} against >= {FAR_Q:.2f} of its wall-distance '
          f'range)')
    print(f'  nulls    : Arm A {", ".join(NULLS)}; Arm B clipping-matched, '
          f'{args.n_null} draws each, seed {args.seed}')
    print('=' * 72, flush=True)

    sum_rows, prof_rows, field_frames = [], [], []
    env_geom, missing, verified = {}, [], {}
    device = None
    for e in envs:
        data_path = f'{DATA_DIR}/{e}.h5'
        if not os.path.exists(data_path):
            print(f'\n[{e}] no dataset at {data_path} -- skipping', flush=True)
            missing.append(e)
            continue
        print(f'\n===== {e} ({arena_shape(e)}) =====', flush=True)
        root = ET.parse(f'{XML_DIR}/{e}.xml').getroot()
        xy = WP.load_positions(data_path)
        env = R.build_env(xy, root)
        env_geom[e] = env
        # The analysis grid the libraries were measured on, so a null field is
        # clipped to exactly the floor the real ones were.
        C = R.resolve_grid_cfg(base_C, xy, env=env, verbose=True)
        G = R._grid_setup(env, C)
        # The keep-out is what the distance scale is measured on, so it has to
        # travel with the arena rather than be re-derived downstream.
        env['wall_margin_m'] = float(C.get('IN_ENV_MARGIN_M', 0.0) or 0.0)
        print(f'  area {env["env_area"]:.1f} m^2, {len(xy)} locations, '
              f'wall distance {wall_margin(env):.2f}-'
              f'{max_wall_distance(env):.2f} m (keep-out to midline), '
              f'bin {np.sqrt(G["bin_area"]):.4f} m', flush=True)
        verified[e] = verify_clipped_shape(G, np.random.default_rng(args.seed))
        print(f'  clipped_shape agrees with rules.field_shape to '
              f'{max(verified[e]["a"], verified[e]["b"], verified[e]["axis"]):.2e} '
              f'over {verified[e]["n_checked"]} placements '
              f'({verified[e]["n_clipped"]} clipped)', flush=True)

        need = args.rebuild or any(not os.path.exists(bank_path(e, c))
                                   for c in chans)
        blocks = None
        if need:
            print('  building libraries: loading feature blocks', flush=True)
            blocks, _ = ch.load_channel_blocks(data_path)
            if device is None:
                device = R.pick_device(use_gpu=not args.no_gpu)

        for c in chans:
            banks = SD.build_banks(e, c, blocks, xy, env, [(PCTL, THRESH)],
                                   [IOU], base_C, device, BANK_DIR,
                                   use_cache=not args.rebuild)
            bank = banks[(PCTL, THRESH, IOU)]
            if args.synthetic:
                if len(bank) < MIN_FIELDS:
                    print(f'  [{c}] real library has {len(bank)} fields -- '
                          f'nothing to resample', flush=True)
                else:
                    bank = synthetic_bank(bank, env, G, rng, args.synthetic)
            row, prows, fields = analyse_library(
                bank, env, G, args.n_null, rng, dict(env=e, channel=c))
            sum_rows.append(row)
            prof_rows.extend(prows)
            field_frames.append(fields)
        del blocks

    if not sum_rows:
        print('\nNo results. Datasets missing: ' + (', '.join(missing) or 'none'))
        return 1

    summary = pd.DataFrame(sum_rows)
    n_calib = 0
    if args.calibration:
        summary, n_calib = apply_calibration(
            summary, load_calibration(args.calibration))
        print(f'\ncalibration from {args.calibration}: applied to {n_calib} '
              f'library x statistic values', flush=True)
    for key, _base, _arm, _nn, _m, _k in STAT_KEYS:
        if f'{key}_p' in summary:
            summary[f'{key}_q'] = WP.bh_q(summary[f'{key}_p'])
    summary.to_csv(f'{out_dir}/summary.csv', index=False)
    prof = pd.DataFrame(prof_rows)
    if len(prof):
        prof.to_csv(f'{out_dir}/profiles.csv', index=False)
    arena = arena_table(summary[summary.tested]) if summary.tested.any() \
        else pd.DataFrame()
    if len(arena):
        arena.to_csv(f'{out_dir}/arena_summary.csv', index=False)
    contrast = shape_contrast(arena)
    if len(contrast):
        contrast.to_csv(f'{out_dir}/shape_contrast.csv', index=False)
    fields = pd.concat(field_frames, ignore_index=True)
    fields.to_csv(f'{out_dir}/fields.csv', index=False)

    # Discs smallest first, then squares, then corridors: the reading order of
    # every figure, and the order the shape contrast is argued in.
    envs_drawn = sorted(env_geom, key=lambda e: (
        ['disc', 'square', 'corridor', 'other'].index(arena_shape(e)),
        env_geom[e]['env_area']))
    print('\nfigures:', flush=True)
    if len(prof):
        fig_raw_and_null(prof, envs_drawn, chans, 'elongation',
                         'E1_elongation_raw_and_null.png',
                         'E1  elongation against wall distance: what the fields '
                         'measure (solid) and what\nclipping alone predicts for '
                         'them (dashed). The gap is the only candidate result.',
                         'mean log elongation (a/b)')
        fig_excess_profile(prof, envs_drawn, chans, 'elongation',
                           'E2_elongation_excess.png',
                           'E2  elongation beyond the clipping-matched null, '
                           'bin by bin\nzero = exactly what cutting a field '
                           'of this size at this distance produces',
                           'log elongation beyond null')
        fig_raw_and_null(prof, envs_drawn, chans, 'alignment',
                         'E4_alignment_raw_and_null.png',
                         'E4  alignment with the nearest wall: measured (solid) '
                         'against clipping alone (dashed)\n+1 = along the wall, '
                         '-1 = straight at it',
                         'mean cos(2 x angle to the wall)')
        fig_excess_profile(prof, envs_drawn, chans, 'alignment',
                           'E5_alignment_excess.png',
                           'E5  alignment beyond the clipping-matched null, '
                           'bin by bin\nzero = the alignment a cut field shows '
                           'with no orientation preference at all',
                           'alignment beyond null')
    fig_arm_a_forest(summary, 'E3_arm_a_near_minus_far.png')
    fig_angle_histograms(fields, summary, 'E6_angle_to_wall.png')
    fig_shape_contrast(arena, 'E7_round_against_straight.png')
    prune_orphan_figures()

    rep = WallElongationReport(env_name=','.join(envs), out_dir=out_dir,
                               fig_dir=fig_dir, results=summary,
                               log_path=os.environ.get('REALM_LOG_PATH'))
    rep.arena, rep.profiles, rep.contrast = arena, prof, contrast
    rep.synthetic, rep.verified = args.synthetic, verified
    rep.calibration, rep.n_calib = args.calibration, n_calib
    if missing:
        print(f'\n!! datasets not found, excluded: {", ".join(missing)}')
    print('\n' + rep.compose(), flush=True)
    if not args.no_email:
        rep.send()
    print(f'\nsummary  -> {out_dir}/summary.csv'
          f'\narenas   -> {out_dir}/arena_summary.csv'
          f'\ncontrast -> {out_dir}/shape_contrast.csv'
          f'\nfigures  -> {fig_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
