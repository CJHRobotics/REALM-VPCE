"""Field shape against scale and wall proximity.

Experiment 4. Takes every field Experiment 2 admitted, in all eight collected
arenas, and writes down four things about each one: how elongated it is, which
way it points, how far it sits from the nearest wall, and the angle between its
long axis and that wall. Then it correlates the pairs.

Descriptive, not inferential. Three questions:

    scale        vs elongation        do coarser fields come out longer?
    wall distance vs elongation       are fields near a wall longer?
    wall distance vs angle to wall    do fields near a wall point at it?

Spearman on each, per arena and channel, per scale within that, and pooled.
Ranks rather than Pearson because scale is ordinal (0 finest to 5 coarsest)
and elongation is heavy-tailed. No null model, no permutation, no resampling:
the question here is what the libraries look like, and a correlation with its
n and its q answers that.

The one thing the table has to carry
------------------------------------
Rule 7 fits a field's ellipse to the second moments of its MASK, and the mask
is intersected with the floor. A field whose shape reaches past the wall is
therefore cut, and the second moments of a cut blob are elongated ALONG the
wall. So both of the wall correlations are partly manufactured by the arena's
outline rather than by the fields, in a pipeline with no anisotropy in it.

That is not a reason to build a null. It is a reason for one column:

    reach_to_wall_m   how far the recorded ellipse extends toward the nearest
                      wall, from the ellipse's own support function
                      sqrt(a^2 cos^2 phi + b^2 sin^2 phi), phi being the angle
                      between the major axis and the wall's normal.
    crosses_wall      dist_to_wall_m < reach_to_wall_m -- the ellipse reaches
                      past the wall, so the mask behind it was cut and the
                      shape on record is a cut shape.

Every correlation is then run twice: over all fields, and over the fields
clear of the wall, whose shapes nothing cut. Where the two agree the result
stands on its own. Where they disagree, that disagreement is the finding, and
it is visible in the same table rather than argued about.

What is measured
----------------
  scale        `scale_band` from the bank: 0 finest to 5 coarsest, geometric
               in radius. Reported as "scale" throughout.
  elongation   semi-major / semi-minor, >= 1. Correlated in logs so the
               relationship is multiplicative, which is how field size
               behaves; Spearman is blind to the transform, and the log is
               there for the plots and the slopes.
  wall point   the nearest point on the boundary: along the radius for a disc,
               the projection onto whichever of the four walls is nearest for
               a rectangle.
  angle        the acute angle between the field's major axis and the INWARD
               NORMAL at that wall point, 0 to 90 degrees. 0 means the field
               points straight at the wall -- perpendicular to it, in the
               sense the question was asked. 90 means it lies along the wall.
               `perpendicular` is the boolean angle < 45.
  distance     `dist_to_wall_m` from the bank, and `wall_dist_norm`, which is
               0 as near the wall as the collection lattice lets a field's
               centre sit and 1 at the disc's centre or the rectangle's
               midline. Within one arena the two give the same Spearman --
               ranks do not care -- so the normalised one exists for the
               pooled rows, where 1 m from a wall means different things in an
               r = 6 disc and a 2 m corridor.

Ambiguous frames are flagged, never guessed: a field at the exact centre of a
disc has no nearest wall point, and one equidistant from two walls of a
rectangle has two. `wall_frame_ambiguous` marks both, and they are dropped
from the angle correlations only.

Arenas
------
All eight that were collected, with the shape declared rather than inferred --
aspect ratio cannot tell a square from a disc, and an lm0 arena has exactly
its lm8 twin's outline:

    circ_lm8_r3, circ_lm8_r6, circ_lm0_r3, circ_lm0_r6    disc
    corr_lm8_l10w10, corr_lm0_l10w10                      square
    corr_lm8_l10w2, corr_lm0_l10w2                        corridor

Libraries are Experiment 2's, unchanged and from its cache under its cache
key: EXTENT_PCTL 65, ACT_THRESH 0.5, Rule 2 off, LAMBDA 0, seed 0. Nothing
here is a knob.

Usage
    python run_field_geometry.py [--envs a,b] [--channels ...] [--rebuild]
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
from realm_tools.experiment_lib.reporting import ExperimentReport

ARENA_SHAPE = {'circ_lm8_r3': 'disc', 'circ_lm8_r6': 'disc',
               'circ_lm0_r3': 'disc', 'circ_lm0_r6': 'disc',
               'corr_lm8_l10w10': 'square', 'corr_lm0_l10w10': 'square',
               'corr_lm8_l10w2': 'corridor', 'corr_lm0_l10w2': 'corridor'}
SHAPE_ORDER = {'disc': 0, 'square': 1, 'corridor': 2}
ENVS = list(ARENA_SHAPE)
CHANNELS = SD.CHANNELS
CHANNEL_COLORS = SD.CHANNEL_COLORS
SCALES = SD.SCALES
SCALE_COLORS = SD.SCALE_COLORS
INK, MUTED, RULE_GRAY, SURFACE = SD.INK, SD.MUTED, SD.RULE_GRAY, SD.SURFACE
# The two subsets are identities, not magnitudes, so they get two hues rather
# than two shades: black for every admitted field, orange for the ones whose
# ellipse does not reach past the wall. Where the two curves coincide -- the
# good case -- one colour would simply hide the other.
CLEAR_COLOR = '#c1440e'
# Arena shape is an identity with three values, so three hues taken from far
# apart on the circle rather than three shades of one. Used for G4's row
# labels, so the disc / square / corridor grouping is readable off the grid.
SHAPE_COLORS = {'disc': '#1b6ca8', 'square': '#c1440e', 'corridor': '#6a3d9a'}

PCTL, THRESH = SD.SETTINGS[0]
IOU = None
BASE_C = R.resolve_cfg(dict(LAMBDA=0.0, RANDOM_SEED=0))

DATA_DIR = f'{REPO}/data/vpce/collect_data'
XML_DIR = f'{REPO}/simulation/worlds/environments/vpce'
BANK_DIR = f'{REPO}/data_cache/scale_distribution'
OUT_DIR = f'{REPO}/data_cache/field_geometry'
FIG_DIR = f'{HERE}/figures/field_geometry'

MIN_N = 20              # fields a correlation needs before it is reported
PERP_DEG = 45.0         # below this the field counts as pointing at the wall
ALPHA = 0.05

# (x variable, y variable, label). The three questions, in the order asked.
PAIRS = [('scale', 'log_elongation', 'scale vs elongation'),
         ('wall_dist_norm', 'log_elongation', 'wall distance vs elongation'),
         ('wall_dist_norm', 'angle_to_wall_deg',
          'wall distance vs angle to wall')]
# The two wall pairs on their own. Grouping BY scale holds scale constant, so
# the scale pair has no variance left in x there and would emit a table of
# undefined correlations.
WALL_PAIRS = PAIRS[1:]
SUBSETS = ('all fields', 'clear of wall')


def arena_shape(env_name):
    return ARENA_SHAPE.get(env_name, 'other')


def bank_path(env_name, cname):
    """Experiment 2's cache key at the operating point, Rule 2 off."""
    return f'{BANK_DIR}/{env_name}/{cname}_p{PCTL}_t{THRESH:g}_roff_bank.csv'


def load_positions(data_path, n_orientations=8):
    """Positions only, without the feature blocks.

    The same xy that channels.load_channel_blocks returns. Reading the blocks
    costs several GB and nothing here needs them: every field is already
    described in the bank.
    """
    import h5py
    with h5py.File(data_path, 'r') as f:
        xs = np.asarray(f['x'][:], dtype=np.float64)
        ys = np.asarray(f['y'][:], dtype=np.float64)
    n_loc = len(xs) // n_orientations
    return np.stack([xs.reshape(n_loc, n_orientations)[:, 0],
                     ys.reshape(n_loc, n_orientations)[:, 0]],
                    axis=1).astype(np.float32)


# ----------------------------------------------------------------- geometry

def nearest_wall(cx, cy, env, tie_m):
    """The nearest point on the boundary, and the inward normal there.

    A disc: straight out along the radius, so the normal points back at the
    centre. A rectangle: the projection onto whichever of the four walls is
    nearest, with the normal pointing into the room.

    `ambiguous` marks the places where there is no single answer -- the exact
    centre of a disc, where every direction is equally outward, and a corner
    of a rectangle, where two walls are within `tie_m` of equally near. Those
    are flagged rather than assigned one, and drop out of the angle
    correlations only.

    Returns (wall_x, wall_y, normal_rad, ambiguous).
    """
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    if env['is_circular']:
        dx, dy = cx - env['env_cx'], cy - env['env_cy']
        r = np.hypot(dx, dy)
        amb = ~(r > tie_m)
        with np.errstate(invalid='ignore', divide='ignore'):
            ux, uy = dx / r, dy / r
        wx = env['env_cx'] + env['env_R'] * ux
        wy = env['env_cy'] + env['env_R'] * uy
        normal = np.arctan2(-uy, -ux)            # inward
    else:
        d = np.stack([cx - env['x_min'], env['x_max'] - cx,
                      cy - env['y_min'], env['y_max'] - cy], axis=0)
        near = np.argmin(d, axis=0)
        srt = np.sort(d, axis=0)
        amb = (srt[1] - srt[0]) < tie_m
        wx = np.select([near == 0, near == 1], [env['x_min'], env['x_max']], cx)
        wy = np.select([near == 2, near == 3], [env['y_min'], env['y_max']], cy)
        normal = np.select([near == 0, near == 1, near == 2, near == 3],
                           [0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi], 0.0)
    amb = amb | ~np.isfinite(cx) | ~np.isfinite(cy)
    return (np.where(amb, np.nan, wx), np.where(amb, np.nan, wy),
            np.where(amb, np.nan, normal), amb)


def acute_angle_deg(theta, normal):
    """The acute angle between a field's major axis and the wall's normal.

    0 means the axis lies along the normal: the field points straight at the
    wall, perpendicular to it. 90 means it lies along the wall. Folded to
    0-90 because an axis has no direction -- theta and theta + pi are the
    same field and must give the same number.
    """
    d = np.asarray(theta, dtype=float) - np.asarray(normal, dtype=float)
    return np.degrees(np.arccos(np.clip(np.abs(np.cos(d)), 0.0, 1.0)))


def ellipse_reach(a, b, angle_deg):
    """How far an ellipse extends from its centre in a given direction.

    The support function of an ellipse with semi-axes a >= b, at an angle phi
    from its major axis: sqrt(a^2 cos^2 phi + b^2 sin^2 phi). Along the major
    axis it is a, across it b, and in between it interpolates the way the
    ellipse's own outline does.

    Used against the wall's normal to ask whether the recorded shape reaches
    past the wall -- which is what decides whether the mask behind it was cut,
    and so whether the recorded shape can be read as the field's own.
    """
    phi = np.radians(np.asarray(angle_deg, dtype=float))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt((a * np.cos(phi)) ** 2 + (b * np.sin(phi)) ** 2)


def field_table(bank, env, env_name, cname, margin_m, tie_m):
    """One row per admitted field: its shape, its wall, and the angle between."""
    a = bank.semi_major_m.to_numpy(dtype=float)
    b = bank.semi_minor_m.to_numpy(dtype=float)
    th = bank.orientation_rad.to_numpy(dtype=float)
    cx = bank.centroid_x.to_numpy(dtype=float)
    cy = bank.centroid_y.to_numpy(dtype=float)
    d = bank.dist_to_wall_m.to_numpy(dtype=float)

    wx, wy, normal, amb = nearest_wall(cx, cy, env, tie_m)
    angle = acute_angle_deg(th, normal)
    reach = ellipse_reach(a, b, angle)
    hi = max_wall_distance(env)

    t = pd.DataFrame(dict(
        env=env_name, shape=arena_shape(env_name), channel=cname,
        env_area_m2=float(env['env_area']),
        max_wall_dist_m=hi, wall_margin_m=margin_m,
        node_id=bank.node_id.to_numpy() if 'node_id' in bank else np.arange(len(bank)),
        scale=bank.scale_band.to_numpy(dtype=int),
        area_env_m2=bank.area_env_m2.to_numpy(dtype=float),
        radius_env_m=bank.radius_env_m.to_numpy(dtype=float),
        semi_major_m=a, semi_minor_m=b,
        elongation=bank.elongation.to_numpy(dtype=float),
        log_elongation=np.log(np.where(b > 0, a / np.maximum(b, 1e-12), np.nan)),
        orientation_rad=th, orientation_deg=np.degrees(th) % 180.0,
        centroid_x=cx, centroid_y=cy,
        dist_to_wall_m=d,
        wall_dist_norm=np.clip((d - margin_m) / max(hi - margin_m, 1e-9),
                               0.0, 1.0),
        wall_x=wx, wall_y=wy, wall_normal_rad=normal,
        angle_to_wall_deg=angle,
        perpendicular=angle < PERP_DEG,
        reach_to_wall_m=reach,
        crosses_wall=d < reach,
        wall_frame_ambiguous=amb))
    t['clear_of_wall'] = ~t.crosses_wall
    return t


def max_wall_distance(env):
    """The furthest any point can be from the boundary: a disc's centre, or a
    rectangle's midline."""
    if env['is_circular']:
        return float(env['env_R'])
    return 0.5 * float(min(env['x_max'] - env['x_min'],
                           env['y_max'] - env['y_min']))


# -------------------------------------------------------------- correlations

def spearman(x, y):
    """Spearman rho, its p, and n over the rows where both are finite."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < MIN_N or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan, np.nan, n
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue), n


def bh_q(p):
    """Benjamini-Hochberg q-values; NaN passes through.

    Not a null model -- just the correction for asking the same question of
    every arena, channel and scale. Without it a table of 150 p-values reads
    as if it were one.
    """
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return q
    order = ok[np.argsort(p[ok])]
    ranked = p[order] * len(ok) / np.arange(1, len(ok) + 1)
    q[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return q


def _subsets(t):
    """The two views every correlation is reported over."""
    yield 'all fields', t
    yield 'clear of wall', t[t.clear_of_wall.astype(bool)]


def correlate(fields, group_cols, group_label, pairs=None):
    """Every pair, over both subsets, within each group.

    `group_cols` of [] pools everything, which is why wall distance is
    correlated in its normalised form: 1 m from a wall is a different thing in
    an r = 6 disc and a 2 m corridor. `pairs` narrows the questions asked of a
    grouping -- grouping by scale leaves the scale pair with no variance in x.
    """
    pairs = PAIRS if pairs is None else pairs
    rows = []
    groups = ([((), fields)] if not group_cols
              else list(fields.groupby(group_cols, sort=False)))
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(group_cols, key))
        for subset, gs in _subsets(g):
            for xcol, ycol, label in pairs:
                gg = gs
                if ycol == 'angle_to_wall_deg':
                    # No nearest wall point, no angle. Dropped here only.
                    gg = gs[~gs.wall_frame_ambiguous.astype(bool)]
                rho, p, n = spearman(gg[xcol], gg[ycol])
                rows.append(dict(
                    grouping=group_label, **meta, subset=subset, pair=label,
                    x=xcol, y=ycol, n=n, rho=rho, p=p))
    out = pd.DataFrame(rows)
    if len(out):
        out['q'] = np.nan
        for _, idx in out.groupby(['grouping', 'pair', 'subset']).groups.items():
            out.loc[idx, 'q'] = bh_q(out.loc[idx, 'p'])
    return out


def descriptives(fields):
    """Per arena, channel and scale: how many fields, how long, how aligned."""
    rows = []
    for (e, c, s), g in fields.groupby(['env', 'channel', 'scale'], sort=True):
        ang = g.angle_to_wall_deg.to_numpy(dtype=float)
        ang = ang[np.isfinite(ang)]
        rows.append(dict(
            env=e, shape=arena_shape(e), channel=c, scale=int(s),
            n=len(g), n_crosses_wall=int(g.crosses_wall.sum()),
            frac_crosses_wall=float(g.crosses_wall.mean()),
            area_median_m2=float(g.area_env_m2.median()),
            elongation_median=float(g.elongation.median()),
            elongation_p90=float(g.elongation.quantile(0.9)),
            dist_to_wall_median_m=float(g.dist_to_wall_m.median()),
            angle_to_wall_median_deg=float(np.median(ang)) if len(ang) else np.nan,
            frac_perpendicular=float(g.perpendicular.mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

FIGURES_WRITTEN = []


def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    FIGURES_WRITTEN.append(p)
    print(f'  {p}', flush=True)


def prune_orphan_figures():
    """Delete G<digit>*.png this run did not write, as Experiment 2 does."""
    keep = {os.path.abspath(p) for p in FIGURES_WRITTEN}
    for p in sorted(glob.glob(os.path.join(FIG_DIR, 'G[0-9]*.png'))):
        if os.path.abspath(p) not in keep:
            os.remove(p)
            print(f'  pruned orphaned figure {os.path.basename(p)}', flush=True)


def _envs_in_order(fields):
    """Discs smallest first, then squares, then corridors, then by name.

    Sorted on the ARENA's area, not on the total area of its fields: the
    second is a proxy that happens to correlate and would reorder panels as
    libraries change size.
    """
    area = fields.groupby('env').env_area_m2.first()
    return sorted(fields.env.unique(),
                  key=lambda e: (SHAPE_ORDER.get(arena_shape(e), 3),
                                 float(area.get(e, 0.0)), e))


def _panels(envs, sharey=False):
    """A panel per arena, each free to scale its own y axis.

    Not shared. The arenas differ by 5x in radius and 16x in area, so one axis
    across all eight compresses the small arenas into a strip to leave headroom
    for the large ones, and the shape of the thing being plotted stops being
    visible in most of the panels. Every panel carries its own ticks, so no
    scale is hidden by this -- it just has to be read per panel.
    """
    ncol = int(np.ceil(len(envs) / 2)) if len(envs) > 4 else len(envs)
    nrow = int(np.ceil(len(envs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharey=sharey,
                             figsize=(3.2 * ncol, 2.9 * nrow))
    flat = [ax for r in axes for ax in r]
    for ax in flat[len(envs):]:
        ax.axis('off')
    for ax in flat:
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(RULE_GRAY)
    return fig, flat[:len(envs)], ncol


def _scale_label(s):
    return f'scale {s}' + (' (finest)' if s == SCALES[0] else
                           ' (coarsest)' if s == SCALES[-1] else '')


def fig_elongation_by_scale(fields, name):
    """G1: how elongated a field is, scale by scale.

    One box per scale, channels pooled, one panel per arena. The notch is the
    median; the box is the interquartile range. Scale runs along a colour
    gradient rather than one hue, so a scale wears the same colour here as in
    Experiment 2's figures.
    """
    envs = _envs_in_order(fields)
    fig, axes, ncol = _panels(envs)
    for ax, e in zip(axes, envs):
        fe = fields[fields.env == e]
        data, pos, cols = [], [], []
        for s in SCALES:
            v = fe[fe.scale == s].elongation.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) >= 5:
                data.append(v)
                pos.append(s)
                cols.append(SCALE_COLORS[s])
        if data:
            bp = ax.boxplot(data, positions=pos, widths=0.66, showfliers=False,
                            patch_artist=True, medianprops=dict(color=INK, lw=1.2))
            for patch, c in zip(bp['boxes'], cols):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
                patch.set_edgecolor(c)
        ax.axhline(1.0, color=RULE_GRAY, lw=0.8, ls=':')
        ax.set_title(e, fontsize=9, color=INK)
        ax.set_xticks(SCALES)
        ax.set_xlim(-0.6, SCALES[-1] + 0.6)
        ax.set_xlabel('scale (0 finest, 5 coarsest)', fontsize=7.5, color=MUTED)
    for i, ax in enumerate(axes):
        if i % ncol == 0:
            ax.set_ylabel('elongation (a/b)', fontsize=8, color=INK)
    fig.suptitle('G1  elongation by scale, channels pooled\n'
                 'dotted = 1.0, a circular field', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, name)


def _binned_quantiles(x, y, edges, min_n=8, qs=(25, 50, 75)):
    """Quantiles of y in bins of x, with the bin centres.

    A bin holding fewer than `min_n` fields is dropped rather than drawn: a
    median of four fields is not a trend, and in the outer bins of a disc --
    a thin annulus -- that is what would otherwise be plotted.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, out = [], [[] for _ in qs]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi) & np.isfinite(y)
        if sel.sum() >= min_n:
            cx.append(0.5 * (lo + hi))
            for k, q in enumerate(qs):
                out[k].append(float(np.percentile(y[sel], q)))
    return np.array(cx), [np.array(o) for o in out]

def _vs_distance(fields, ycol, name, title, ylabel, hline=None):
    """G2 / G3: a measure against wall distance, one panel per arena.

    The grey band is the interquartile range of the whole library in bins of
    wall distance and the black line its median, which is what the Spearman
    picks up. The coloured lines are the median within each scale, so the
    stratification the band hides stays readable and a trend can be seen to
    hold inside a scale rather than being the scales sliding past each other.

    This replaced one point per field coloured by scale. At ten thousand-odd
    fields the cloud was ink rather than information: it buried its own median,
    the scales overplotted each other in whatever order they were drawn, and
    the eye read the densest region as the trend.
    """
    envs = _envs_in_order(fields)
    fig, axes, ncol = _panels(envs)
    edges = np.linspace(0.0, 1.0, 11)
    leg = None
    for ax, e in zip(axes, envs):
        fe = fields[fields.env == e]
        x = fe.wall_dist_norm.to_numpy(dtype=float)
        y = fe[ycol].to_numpy(dtype=float)
        bx, (q25, q50, q75) = _binned_quantiles(x, y, edges)
        if len(bx):
            ax.fill_between(bx, q25, q75, color='0.74', alpha=0.6, lw=0,
                            zorder=1, label='IQR, all fields')
            ax.plot(bx, q50, '-', color=INK, lw=2.0, zorder=4,
                    label='median, all fields')
            leg = leg or ax
        for sc in SCALES:
            sel = (fe.scale.to_numpy() == sc)
            if sel.sum() < MIN_N:
                continue
            # A per-scale median wants more than the band does before it is
            # worth a line: it is one sixth of the library spread over the
            # same bins, and at ten fields a bin the angle measure in
            # particular -- which ranges over 90 degrees -- draws noise.
            sx, (sm,) = _binned_quantiles(x[sel], y[sel], edges, min_n=15,
                                          qs=(50,))
            if len(sx) >= 4:
                ax.plot(sx, sm, '-', lw=1.1, alpha=0.9, zorder=3,
                        color=SCALE_COLORS[sc], label=_scale_label(sc))
        if hline is not None:
            ax.axhline(hline, color=RULE_GRAY, lw=0.8, ls=':', zorder=0)
        ax.set_xlim(0, 1)
        ax.set_title(e, fontsize=9, color=INK)
        ax.set_xlabel('wall distance: 0 = as near as a field can sit,\n'
                      '1 = centre or midline', fontsize=7.5, color=MUTED)
    for i, ax in enumerate(axes):
        if i % ncol == 0:
            ax.set_ylabel(ylabel, fontsize=8, color=INK)
    if leg is not None:
        leg.legend(fontsize=5, frameon=False, ncol=2, loc='best')
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, name)


def fig_rho_summary(corr, name):
    """G4: every correlation as a grid, one panel per pair.

    Rows are arenas in reading order, columns are channels, and the last column
    is that arena with its channels pooled. The number in each cell is the
    Spearman rho and a star marks q < ALPHA; colour only makes the pattern
    visible at a glance, so nothing depends on reading it precisely.

    This replaced a 48-row forest plot of three panels with two markers per
    row. The information was all there and little of it was legible: row labels
    at 5.5 pt, arena and channel structure impossible to see because both were
    folded onto one axis, and a second marker per row -- the clear-of-wall
    subset -- doubling the ink to show a difference that turns out not to
    exist. Only the all-fields subset is drawn now. Both are still in
    correlations.csv, and the report quantifies how closely they agree, so
    dropping it from the figure is a presentation choice and not a loss of
    evidence.
    """
    c = corr[(corr.grouping == 'env x channel') & (corr.subset == 'all fields')]
    ce = corr[(corr.grouping == 'env') & (corr.subset == 'all fields')]
    if not len(c):
        return
    envs = sorted(c.env.unique(),
                  key=lambda e: (SHAPE_ORDER.get(arena_shape(e), 3), e))
    chans = [ch_ for ch_ in CHANNELS if (c.channel == ch_).any()]
    cols = chans + ['all']

    def panel_vmax(label):
        """The colour range for one panel: symmetric about zero, scaled to
        that panel's own numbers.

        PER PANEL, not shared. The three pairs are not on a common scale --
        one of them regularly reaches rho 0.85 while the others live inside
        +-0.15 -- so a single range washes the small ones to a uniform white
        and the pattern that is actually being looked for disappears. Colour
        here is a cue for reading a panel, and every cell prints its exact
        value, so a range that differs between panels costs nothing. Each
        panel states its own range in its title.
        """
        v = np.concatenate([
            c[c.pair == label].rho.to_numpy(dtype=float),
            ce[ce.pair == label].rho.to_numpy(dtype=float)])
        v = v[np.isfinite(v)]
        return (max(0.05, float(np.ceil(np.abs(v).max() * 20) / 20))
                if len(v) else 0.05)

    fig, axes = plt.subplots(1, len(PAIRS), squeeze=False,
                             figsize=(3.7 * len(PAIRS),
                                      0.34 * len(envs) + 2.6))
    for ax, (_, _, label) in zip(axes[0], PAIRS):
        vmax = panel_vmax(label)
        M = np.full((len(envs), len(cols)), np.nan)
        Q = np.full((len(envs), len(cols)), np.nan)
        for i, e in enumerate(envs):
            for j, col in enumerate(cols):
                src = ce if col == 'all' else c
                sel = (src.env == e) & (src.pair == label)
                if col != 'all':
                    sel &= (src.channel == col)
                row = src[sel]
                if len(row):
                    M[i, j] = float(row.rho.iloc[0])
                    Q[i, j] = float(row.q.iloc[0])
        ax.imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  aspect='auto', interpolation='nearest')
        for i in range(len(envs)):
            for j in range(len(cols)):
                if not np.isfinite(M[i, j]):
                    ax.text(j, i, '--', ha='center', va='center', fontsize=6,
                            color=MUTED)
                    continue
                star = '*' if np.isfinite(Q[i, j]) and Q[i, j] < ALPHA else ''
                shade = INK if abs(M[i, j]) < 0.62 * vmax else '#ffffff'
                ax.text(j, i, f'{M[i, j]:+.2f}{star}', ha='center',
                        va='center', fontsize=6.2, color=shade)
        # The pooled column is a different grouping, so it gets a rule rather
        # than sitting in the grid as though it were another channel.
        ax.axvline(len(chans) - 0.5, color=SURFACE, lw=3.0)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=6.5, rotation=45, ha='right',
                           color=MUTED)
        ax.set_yticks(range(len(envs)))
        ax.set_yticklabels(envs, fontsize=6.5)
        for i, e in enumerate(envs):
            ax.get_yticklabels()[i].set_color(
                SHAPE_COLORS.get(arena_shape(e), INK))
        ax.set_title(f'{label}\ncolour spans +-{vmax:g}', fontsize=8.5,
                     color=INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes[0][1:]:
        ax.set_yticklabels([])
    fig.suptitle('G4  every correlation, per arena and channel\n'
                 f'the number in each cell is the Spearman rho, * = q < '
                 f'{ALPHA:g}. Last column pools an arena\'s channels; blue '
                 f'negative, red positive; arena labels coloured by shape.',
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, name)


# -------------------------------------------------------------------- report

class FieldGeometryReport(ExperimentReport):
    experiment = 'field-geometry'

    def title(self):
        c = self.corr
        pooled = c[c.grouping == 'pooled']
        bits = []
        for _, _, label in PAIRS:
            r = pooled[(pooled.pair == label) & (pooled.subset == 'all fields')]
            if len(r) and np.isfinite(r.rho.iloc[0]):
                bits.append(f'{label} rho {r.rho.iloc[0]:+.2f}')
        return '; '.join(bits) or 'no correlation reportable'

    def figures(self):
        return sorted(FIGURES_WRITTEN)

    def data_files(self):
        return [p for p in (f'{self.out_dir}/correlations.csv',
                            f'{self.out_dir}/descriptives.csv')
                if os.path.exists(p)]

    #: |d rho| above this, or a sign flip at a rho this large, counts as the
    #: two subsets disagreeing -- at which point the figures showing only one
    #: of them is no longer a safe simplification.
    AGREE_TOL = 0.05

    def _subset_agreement(self):
        """How far apart the two subsets' correlations ever get, and whether
        that is small enough to justify plotting only one of them.

        The figures show the all-fields subset alone. That is defensible only
        while the clear-of-wall subset agrees with it, so this measures the
        agreement rather than asserting it, and the verdict below is derived
        from the measurement. Both subsets stay in correlations.csv either way.

        A sign flip only counts when one of the two correlations is at least
        AGREE_TOL in magnitude: where the true value is near zero the sign is
        the sign of noise, and counting those would make a null result look
        like a disagreement.

        Returns (lines, agree) -- agree is False if any pair exceeds the
        tolerance on either count.
        """
        c = self.corr
        L, agree = [], True
        for _, _, label in PAIRS:
            a = c[(c.pair == label) & (c.subset == 'all fields')]
            b = c[(c.pair == label) & (c.subset == 'clear of wall')]
            keys = [k for k in ('grouping', 'env', 'channel', 'scale')
                    if k in a.columns and k in b.columns]
            m = a.merge(b, on=keys, suffixes=('_all', '_clear'))
            d = (m.rho_all - m.rho_clear).abs()
            ok = np.isfinite(d)
            if not ok.any():
                L.append(f'  {label:32s} (nothing to compare)')
                continue
            big = np.maximum(m.rho_all.abs(), m.rho_clear.abs()) >= self.AGREE_TOL
            flip = int((((m.rho_all > 0) != (m.rho_clear > 0)) & big & ok).sum())
            worst = float(d[ok].max())
            bad = worst > self.AGREE_TOL or flip > 0
            agree &= not bad
            L.append(f'  {label:32s} max |d rho| {worst:.3f}, median '
                     f'{float(d[ok].median()):.3f} over {int(ok.sum())} '
                     f'groupings; material sign flips {flip}'
                     + ('   <-- DISAGREES' if bad else ''))
        return L, agree

    def _table(self, grouping, cols, pairs=None, max_rows=64,
               subset='all fields'):
        c = self.corr[(self.corr.grouping == grouping) &
                      (self.corr.subset == subset)]
        if not len(c):
            return ['  (nothing reportable)']
        L = []
        for _, _, label in (PAIRS if pairs is None else pairs):
            L.append(f'  {label}')
            head = '    ' + ' '.join(f'{k:>{w}s}' for k, w in cols)
            L.append(head + f' {"n":>7s} {"rho":>7s} '
                            f'{"p":>9s} {"q":>9s}')
            g = c[c.pair == label]
            if 'env' in g:
                g = g.assign(_o=g['shape'].map(SHAPE_ORDER).fillna(3))
                g = g.sort_values(['_o', 'env'] +
                                  (['channel'] if 'channel' in g else []) +
                                  (['scale'] if 'scale' in g else []))
            shown = 0
            for r in g.itertuples():
                if shown >= max_rows:
                    L.append(f'    ... {len(g) - shown} more rows in '
                             f'correlations.csv')
                    break
                vals = '    ' + ' '.join(
                    f'{getattr(r, k, ""):>{w}}' if not isinstance(
                        getattr(r, k, ''), float)
                    else f'{getattr(r, k):>{w}.0f}' for k, w in cols)
                rho = f'{r.rho:+7.3f}' if np.isfinite(r.rho) else f'{"--":>7s}'
                p = f'{r.p:9.2g}' if np.isfinite(r.p) else f'{"--":>9s}'
                q = f'{r.q:9.2g}' if np.isfinite(r.q) else f'{"--":>9s}'
                L.append(f'{vals} {r.n:7d} {rho} {p} {q}')
                shown += 1
            L.append('')
        return L

    def body(self):
        S = self.section
        c, d = self.corr, self.desc
        out = []

        out.append(S('THE QUESTION', '\n'.join([
            'For every field Experiment 2 admitted, in all eight collected '
            'arenas: how elongated is it, which way does it point, how far is '
            'it from the nearest wall, and what is the angle between its long '
            'axis and that wall. Then three correlations:', '',
            '  scale         vs elongation       do coarser fields come out '
            'longer?',
            '  wall distance vs elongation       are fields near a wall '
            'longer?',
            '  wall distance vs angle to wall    do fields near a wall point '
            'at it?', '',
            f'Spearman, because scale is ordinal and elongation is '
            f'heavy-tailed. Libraries are Experiment 2\'s, unchanged: '
            f'EXTENT_PCTL {PCTL}, ACT_THRESH {THRESH:g}, Rule 2 off, '
            f'LAMBDA 0. Descriptive: no null model and no resampling.'])))

        agree_lines, agree = self._subset_agreement()
        out.append(S('THE CLIPPING CHECK, AND WHY IT IS NOT IN THE TABLES',
                     '\n'.join([
            'Rule 7 fits a field\'s ellipse to the second moments of its '
            'MASK, and the mask is intersected with the floor. A field whose '
            'shape reaches past the wall is cut, and a cut blob\'s moments '
            'are elongated ALONG the wall. So both wall correlations could be '
            'partly the arena\'s outline rather than the fields, and every '
            'correlation is computed twice: over all admitted fields, and over '
            'the ones whose recorded ellipse does not reach the wall '
            '(dist_to_wall_m below reach_to_wall_m, the ellipse\'s own extent '
            'toward it), whose shapes nothing cut.', '',
            f'Fields whose ellipse reaches past the wall: '
            f'{100 * float(self.fields.crosses_wall.mean()):.1f}% overall, '
            f'{100 * float(d.frac_crosses_wall.max()):.1f}% in the worst '
            f'arena x channel x scale cell.', '',
            'How far apart the two subsets come out, per pair:'] +
            agree_lines + [
            '', ('THE TWO SUBSETS AGREE to within '
                 f'|d rho| {self.AGREE_TOL:g} with no material sign flip, so '
                 'the tables and figures below show the all-fields subset '
                 'alone and nothing is lost by it.') if agree else
            ('THE TWO SUBSETS DO NOT AGREE on every pair -- see the flagged '
             'rows above. The tables and figures below still show the '
             'all-fields subset alone, so for a flagged pair read the '
             'clear-of-wall rows out of correlations.csv before drawing any '
             'conclusion from it: a trend that lives only in the fields the '
             'wall cut is the arena\'s outline, not the fields.'),
            '', 'Both subsets are in correlations.csv regardless, which is '
            'what makes showing one of them a presentation choice rather than '
            'a claim taken on trust.'])))

        out.append(S('WHAT THE NUMBERS MEAN', '\n'.join([
            'scale             0 finest to 5 coarsest, geometric in radius.',
            'elongation        semi-major / semi-minor, >= 1. Correlated in '
            'logs; Spearman is blind to that, the log is for the plots.',
            'angle to wall     the acute angle between the field\'s major '
            'axis and the INWARD NORMAL at the nearest wall point, 0 to 90 '
            'degrees. 0 = the field points straight at the wall '
            '(perpendicular to it); 90 = it lies along the wall. So a '
            'POSITIVE rho against distance means fields get more '
            'wall-parallel as they move away from the wall, and a NEGATIVE '
            'one means they point at it more.',
            f'perpendicular     the boolean, angle < {PERP_DEG:g} degrees.',
            'wall distance     normalised: 0 is as near a wall as the '
            'collection lattice lets a field\'s centre sit, 1 is the disc\'s '
            'centre or the rectangle\'s midline. Within one arena this gives '
            'the same Spearman as metres -- ranks do not care -- so it exists '
            'for the pooled rows, where 1 m means different things in an '
            'r = 6 disc and a 2 m corridor.', '',
            'q is Benjamini-Hochberg within each pair and subset. Not a null '
            'model, just the correction for asking the same question of every '
            'arena, channel and scale.', '',
            'A field at the exact centre of a disc has no nearest wall point '
            'and one in a rectangle\'s corner has two. Those are flagged '
            '(wall_frame_ambiguous) and dropped from the angle correlations '
            'only: '
            f'{int(self.fields.wall_frame_ambiguous.sum())} fields overall.'])))

        out.append(S('POOLED OVER EVERYTHING',
                     '\n'.join(self._table('pooled', []))))
        out.append(S('PER ARENA', '\n'.join(self._table(
            'env', [('env', 17)]))))
        out.append(S('PER ARENA AND CHANNEL', '\n'.join(self._table(
            'env x channel', [('env', 17), ('channel', 8)], max_rows=48))))
        out.append(S('WITHIN SCALE, PER ARENA', '\n'.join(
            ['Size held still: if wall distance still predicts elongation '
             'inside a single scale, it is not just that the big fields sit '
             'elsewhere. The scale pair is absent here by construction -- '
             'grouping by scale leaves it no variance in x.', ''] +
            self._table('env x scale', [('env', 17), ('scale', 5)],
                        pairs=WALL_PAIRS, max_rows=48))))

        keep = ['env', 'shape', 'channel', 'scale', 'n', 'frac_crosses_wall',
                'area_median_m2', 'elongation_median', 'dist_to_wall_median_m',
                'angle_to_wall_median_deg', 'frac_perpendicular']
        out.append(S('Per arena, channel and scale',
                     self.table(d[[c_ for c_ in keep if c_ in d.columns]],
                                max_rows=300)))
        return '\n'.join(out)


# ----------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--envs', default=','.join(ENVS),
                   help='default: the eight collected arenas')
    p.add_argument('--channels', default=','.join(CHANNELS))
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
    base_C = dict(BASE_C, USE_GPU=not args.no_gpu)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print('=' * 72)
    print('Field geometry | elongation and orientation against scale and walls')
    print(f'  envs     : {envs}')
    print(f'  channels : {chans}')
    print(f'  fields   : Experiment 2 config, EXTENT_PCTL {PCTL}, '
          f'ACT_THRESH {THRESH:g}, Rule 2 off, LAMBDA 0')
    print(f'  libraries: {"rebuilt" if args.rebuild else "from cache where present"}'
          f' ({BANK_DIR})')
    print(f'  pairs    : ' + '; '.join(lab for _, _, lab in PAIRS))
    print(f'  subsets  : {", ".join(SUBSETS)} (descriptive, no null model)')
    print('=' * 72, flush=True)

    frames, missing = [], []
    device = None
    for e in envs:
        data_path = f'{DATA_DIR}/{e}.h5'
        if not os.path.exists(data_path):
            print(f'\n[{e}] no dataset at {data_path} -- skipping', flush=True)
            missing.append(e)
            continue
        print(f'\n===== {e} ({arena_shape(e)}) =====', flush=True)
        root = ET.parse(f'{XML_DIR}/{e}.xml').getroot()
        xy = load_positions(data_path)
        env = R.build_env(xy, root)
        # The collection keep-out, measured rather than assumed: no field's
        # centre can be nearer a wall than the nearest sampled position.
        margin = float(R.wall_distance(xy[:, 0], xy[:, 1], env).min())
        # A wall tie inside the lattice spacing is a corner, not a choice.
        tie = float(R.lattice_spacing(xy))
        print(f'  area {env["env_area"]:.1f} m^2, wall distance '
              f'{margin:.2f}-{max_wall_distance(env):.2f} m, '
              f'lattice {tie:.4f} m', flush=True)

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
            if not len(bank):
                print(f'  [{c}] no admitted fields', flush=True)
                continue
            t = field_table(bank, env, e, c, margin, tie)
            frames.append(t)
            print(f'  [{c}] {len(t)} fields, scales '
                  f'{sorted(t.scale.unique())}, elongation median '
                  f'{t.elongation.median():.2f}, '
                  f'{100 * t.crosses_wall.mean():.0f}% reach past the wall, '
                  f'{100 * t.perpendicular.mean():.0f}% point at it',
                  flush=True)
        del blocks

    if not frames:
        print('\nNo fields. Datasets missing: ' + (', '.join(missing) or 'none'))
        return 1

    fields = pd.concat(frames, ignore_index=True)
    fields.to_csv(f'{OUT_DIR}/fields.csv', index=False)
    desc = descriptives(fields)
    desc.to_csv(f'{OUT_DIR}/descriptives.csv', index=False)
    corr = pd.concat([
        correlate(fields, [], 'pooled'),
        correlate(fields, ['env', 'shape'], 'env'),
        correlate(fields, ['env', 'shape', 'channel'], 'env x channel'),
        correlate(fields, ['env', 'shape', 'scale'], 'env x scale',
                  pairs=WALL_PAIRS),
    ], ignore_index=True)
    corr.to_csv(f'{OUT_DIR}/correlations.csv', index=False)

    print('\nfigures:', flush=True)
    fig_elongation_by_scale(fields, 'G1_elongation_by_scale.png')
    _vs_distance(fields, 'elongation', 'G2_elongation_vs_wall.png',
                 'G2  elongation against wall distance\ngrey = IQR of the '
                 'library, black = its median, coloured = the median within '
                 'each scale; dotted = 1.0, a circular field',
                 'elongation (a/b)', hline=1.0)
    _vs_distance(fields, 'angle_to_wall_deg', 'G3_angle_vs_wall.png',
                 'G3  angle between a field\'s long axis and the nearest '
                 'wall, against wall distance\n0 = points straight at the '
                 'wall, 90 = lies along it; dotted = 45, no preference',
                 'angle to wall normal (deg)', hline=45.0)
    fig_rho_summary(corr, 'G4_correlation_summary.png')
    prune_orphan_figures()

    rep = FieldGeometryReport(env_name=','.join(envs), out_dir=OUT_DIR,
                              fig_dir=FIG_DIR, results=fields,
                              log_path=os.environ.get('REALM_LOG_PATH'))
    rep.fields, rep.corr, rep.desc = fields, corr, desc
    if missing:
        print(f'\n!! datasets not found, excluded: {", ".join(missing)}')
    print('\n' + rep.compose(), flush=True)
    if not args.no_email:
        rep.send()
    print(f'\nfields       -> {OUT_DIR}/fields.csv'
          f'\ncorrelations -> {OUT_DIR}/correlations.csv'
          f'\ndescriptives -> {OUT_DIR}/descriptives.csv'
          f'\nfigures      -> {FIG_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
