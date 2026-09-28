"""Is q = 65 the right field extent? A validation against fields of known shape.

What q is
---------
A place field starts as a group of positions whose views look alike. To draw
the field on the floor, the model has to decide how unlike the group's typical
view a position may be and still count as inside. q sets that line: the
boundary is placed so that it encloses q% of the group's own members
(`SIGMA_MODE = 'quantile'`, `EXTENT_PCTL = q`; the response threshold cancels,
see RETIRED.md). The model runs at q = 65.

Think of drawing a line around a flock of birds. Draw it around every bird,
stragglers included, and it takes in a great deal of empty sky. Draw it around
only the densest core and it leaves out much of the flock. q is how much of the
flock the line goes around. In feature space the "empty sky" is floor that
merely looks like the field -- every position closer to the group's typical
view than the boundary is let in, member or not.

q = 65 was chosen on 21 August 2026 by run_field_recovery.py, in the r = 10
disc. That arena is gone, and the lattice, the analysis bin and every arena the
papers report have changed since. This asks the question again in the eight
arenas in use now, and is written so that the answer can come back "no".

The test
--------
**Fields with a known answer.** A disc of floor is declared to be a place
field. The model gets only the views from inside it -- the same thing it gets
from a group the clustering produced -- and its own extent machinery,
unmodified, draws the field. The drawn field is scored against the disc. One
disc per scale (scale 0 finest to scale 5 coarsest, each at the geometric
centre of its scale's radius range, so sizes follow each arena's own Rule 8/9
window), at 24 sites on four contours between the wall and the most open floor,
in every arena and every channel, at every q from 5 to 100.

**Things that are not fields.** Groups with no single compact region:
positions scattered over 16x the field's area, positions drawn from anywhere,
a group twice the size ceiling, two separate lobes, and a ring. A good q
returns the real fields and refuses these.

Three criteria, fixed before the run
------------------------------------
  accuracy        overlap with the true field (IoU), median over trials
  calibration     recovered area / true area; 1 is exact
  discrimination  % of true fields admitted minus % of non-fields admitted
                  (Youden's J), over the single-region non-fields

Each gives a best q with a 95% interval, from a bootstrap that resamples
(arena, channel) pairs, and a range of q that is nearly as good: within 5% of
the best IoU, within 25% of the true area, within 0.05 of the best J.

Two-lobed groups (`split`, `ring`) are reported but do not enter
discrimination: a group summarised by one centroid cannot represent two
regions whatever q is -- the centroid falls between the lobes -- so scoring q
on them would choose it for a reason q cannot affect.

Robustness: the same curves per arena x channel, per scale and per wall
contour, with the q each prefers. Downstream: the real pipeline at q = 50, 65
and 80 -- field count, sizes, scales occupied and the two wall correlations --
to show how much of what the papers report rests on the exact value.

Outputs
-------
`data_cache/extent_validation/<env>/`   recovery.csv, controls.csv,
    pipeline.csv, pipeline_scales.csv, figdata.npz, meta.json -- per arena, so
    one job per arena can run in parallel and a report pass joins them
`data_cache/extent_validation/`         q_curves.csv, cell_optima.csv,
    level_optima.csv, downstream.csv -- written by the report pass
`figures/extent_validation/`            V1-V6, PNG at 300 dpi and PDF with
    editable text at double-column width, and all of them in one PDF

Figures
  V1  what q does, on one representative field
  V2  the three criteria against q -- the main result
  V3  robustness: arena x channel, scale, wall distance
  V4  what q = 65 returns in every arena (one figure per channel)
  V5  downstream: the field libraries at q = 50, 65, 80
  V6  discrimination, control by control

Usage
    python run_extent_validation.py                     # every arena, then report
    python run_extent_validation.py --envs circ_lm8_r3 --no-report
    python run_extent_validation.py --report-only       # figures + mail from cache
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import textwrap
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch                                                # noqa: E402
import rules as R                                                    # noqa: E402
import run_scale_distribution as SD                                  # noqa: E402
from realm_tools.experiment_lib.reporting import ExperimentReport    # noqa: E402

try:
    import torch
    _HAS_TORCH = True
except ImportError:                                                  # pragma: no cover
    _HAS_TORCH = False

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


# ------------------------------------------------------------------ design

# The value under test, read from the rules rather than written down again.
Q_OP = float(R.DEFAULT_CFG['EXTENT_PCTL'])
Q_GRID = list(range(5, 101, 5))
PIPELINE_Q = (50, 65, 80)

SCALES = list(range(6))
# Sites sit on contours of wall distance, placed as a fraction of the way from
# the wall-most sampled position (the collection keep-out) to the most open
# one. Fractions rather than metres, because the arenas run from a corridor
# 1.6 m wide to a disc 12 m across. The most open contour stops at 0.8: past
# it a disc's contour shrinks to a point and its sites would sit on top of
# each other.
CONTOUR_FRACS = (0.05, 0.30, 0.55, 0.80)
SITES_PER_CONTOUR = 6
CONTROL_SITES_PER_CONTOUR = 2
MIN_MEMBERS = 12                 # below this the member spread is unstable

CONTROL_KINDS = ('scattered', 'shuffled', 'oversized', 'split', 'ring')
# The controls q can be held to. `split` and `ring` are two regions summarised
# by one centroid, which no extent can represent.
SINGLE_REGION = ('scattered', 'shuffled', 'oversized')
SCATTER_AREA = 16.0              # scattered: members drawn from 16x the area
OVERSIZED_FRAC = 0.40            # oversized: 2x the Rule 9 ceiling

# How near to the best counts as "as good". Stated here, not tuned afterwards.
IOU_PLATEAU = 0.95               # within 5% of the best median IoU
CAL_TOL = float(np.log2(1.25))   # within 25% of the true area
J_TOL = 0.05                     # within 0.05 of the best J
# An (arena, channel) pair whose best IoU is below this recovers no field at
# any q -- the channel cannot place a field there -- so it says nothing about
# which q is right, and the robustness count leaves it out.
INFORMATIVE_IOU = 0.25

N_BOOT = 1000
EXAMPLE_ENV = 'circ_lm8_r6'
# Scale 4, not the middle scale: radii follow each arena's own window, so a
# field is the same fraction of every disc, and at scale 3 it is too small in
# a full-arena map to read at print size.
EXAMPLE_SCALE = 4
EXAMPLE_Q = (20, 65, 95)
GALLERY_CHANNEL = 'all'
GALLERY_SCALE = 2

# Twins side by side, small to large, so a landmark effect reads down the rows.
ENV_ORDER = ['circ_lm8_r3', 'circ_lm0_r3', 'circ_lm8_r6', 'circ_lm0_r6',
             'corr_lm8_l10w10', 'corr_lm0_l10w10',
             'corr_lm8_l10w2', 'corr_lm0_l10w2']
ENV_LABEL = {'circ_lm8_r3': 'Disc r = 3 m',
             'circ_lm0_r3': 'Disc r = 3 m, no landmarks',
             'circ_lm8_r6': 'Disc r = 6 m',
             'circ_lm0_r6': 'Disc r = 6 m, no landmarks',
             'corr_lm8_l10w10': 'Square 10 × 10 m',
             'corr_lm0_l10w10': 'Square 10 × 10 m, no landmarks',
             'corr_lm8_l10w2': 'Corridor 10 × 2 m',
             'corr_lm0_l10w2': 'Corridor 10 × 2 m, no landmarks'}
CHANNEL_LABEL = {'hog': 'HOG', 'color': 'Colour', 'spatial': 'Spatial',
                 'lidar': 'Lidar', 'visual': 'Visual', 'all': 'All'}
CONTROL_LABEL = {'scattered': 'Scattered', 'shuffled': 'Shuffled',
                 'oversized': 'Oversized', 'split': 'Two lobes',
                 'ring': 'Ring'}

# ------------------------------------------------------------------ style
#
# Springer double-column width (174 mm). Text is set at 6.5-7.5 pt so it reads
# at print size without scaling, and PDFs carry it as editable TrueType.
FIG_W = 6.85
INK, INK_2, MUTED = SD.INK, SD.INK_2, SD.MUTED
RULE_GRAY = SD.RULE_GRAY
GRID = '#ebeae5'
SHADE = '#f0efeb'
IDEAL_FILL = '#d6d4cc'
NEVER_FILL = '#eeede8'
# Identity colours: the first three slots of the validated categorical
# palette, which pass all-pairs (worst colour-blind pair dE 9.2, worst
# normal-vision pair 24.0). Blue is a real place field or what was drawn from
# one, orange is anything that is not a field, and ink is a difference or the
# pooled result.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
# Magnitudes run along multi-hue ramps with a colourbar or legend, never one
# hue in steps. Scales wear Experiment 2's colours so a scale looks the same
# in every figure of the series. Wall contours take a separate ramp (plasma,
# trimmed where yellow falls under 2:1 on white), so the two never read as the
# same quantity. Both pass the ordinal checks -- monotone lightness, visible
# steps, light end >= 2:1 -- other than being one hue, which is deliberate.
SCALE_COLORS = SD.SCALE_COLORS
CONTOUR_COLORS = ['#38049a', '#9511a1', '#d5546e', '#fb9f3a']
QJOIN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'qjoin', matplotlib.colormaps['magma'](np.linspace(0.03, 0.88, 256)))
# The heatmap's ramp is YlGnBu with its near-white end cut off, so the
# worst cell still reads against the page.
HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'heat', matplotlib.colormaps['YlGnBu'](np.linspace(0.14, 1.0, 256)))


def _style():
    matplotlib.rcParams.update({
        # Arial, or Liberation Sans -- its metric twin -- on the cluster.
        # Not Helvetica: on macOS it is a .ttc collection that embeds badly in
        # PDF and lacks the fraction and arrow glyphs the axes use.
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
        'axes.linewidth': 0.6, 'axes.edgecolor': RULE_GRAY,
        'axes.labelcolor': INK, 'axes.titlecolor': INK, 'text.color': INK,
        'xtick.color': INK_2, 'ytick.color': INK_2,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'lines.linewidth': 1.3,
        'lines.solid_capstyle': 'round',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })


def _panel_label(ax, letter, dx=-22, dy=6):
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction',
                xytext=(dx, dy), textcoords='offset points',
                fontsize=9, fontweight='bold', ha='left', va='bottom')


def _grid_y(ax):
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def _q_axis(ax, label=True):
    ax.set_xlim(0, 101)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    if label:
        ax.set_xlabel('q  (% of the group inside its field)')


def _mark_op(ax, text=False):
    """The value in use, as a vertical line. Labelled above the plot, like a
    tick, when asked -- inside it, a label collides with whatever the data
    happen to do near q = 65."""
    ax.axvline(Q_OP, color=INK, lw=0.8, zorder=1.6)
    if text:
        ax.annotate(f'q = {Q_OP:g}', xy=(Q_OP, 1.0),
                    xycoords=('data', 'axes fraction'), xytext=(0, 1.5),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=6.5, color=INK)


def _title(ax, title, sub=None):
    """Bold title on the left, and the panel's key number beneath it."""
    ax.set_title(title, loc='left', fontweight='bold', pad=12 if sub else 4)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=6.5,
                color=INK_2, ha='left', va='bottom')


def _figure_key(fig, handles, y, ncol=None):
    """One key for the marks every panel shares, above the panels."""
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, y),
               ncol=ncol or len(handles), handlelength=1.8, columnspacing=1.6,
               fontsize=6.5)


def _end_labels(ax, x, ys, labels, colors, gap):
    """Direct labels at the right-hand end of lines, pushed apart to `gap`.

    Each label is a short stroke in the series colour and the name in ink, so
    identity is never carried by coloured text.
    """
    ys = np.asarray(ys, float)
    order = np.argsort(ys)
    pos = ys[order].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    top = ax.get_ylim()[1]
    if len(pos) and pos[-1] > top:
        pos -= pos[-1] - top
    for p_, i in zip(pos, order):
        ax.plot([x + 1.5, x + 5.5], [p_, p_], color=colors[i], lw=1.4,
                clip_on=False, solid_capstyle='butt')
        ax.text(x + 7, p_, labels[i], va='center', ha='left', fontsize=6.5,
                color=INK, clip_on=False)


# ---------------------------------------------------------------- the groups

def scale_radii(env, C):
    """One ideal radius per scale: the geometric centre of the scale's range.

    Scales are the geometric groups Rule 11 uses, ratio BAND_RATIO from the
    Rule 8 floor; the top one is cut at the Rule 9 ceiling. Taking the sizes
    from the arena's own window is what lets one design serve a 20 m^2
    corridor and a 113 m^2 disc: every arena is asked about the same six
    rungs of its own ladder, and with sample count held constant each rung
    holds about the same number of positions everywhere.
    """
    a_min = C['RULE8_AREA_FRAC'] * env['env_area']
    a_max = C['RULE9_AREA_FRAC'] * env['env_area']
    r_min, r_max = np.sqrt(a_min / np.pi), np.sqrt(a_max / np.pi)
    radii = {}
    for k in SCALES:
        lo = r_min * C['BAND_RATIO'] ** k
        hi = min(r_min * C['BAND_RATIO'] ** (k + 1), r_max)
        if lo < r_max:
            radii[k] = float(np.sqrt(lo * hi))
    return radii, float(r_min), float(r_max), float(a_min), float(a_max)


def plant_sites(xy, env, rng):
    """Sites on four contours of wall distance, spread by farthest-point sampling."""
    d = R.wall_distance(xy[:, 0], xy[:, 1], env)
    lo, hi = float(d.min()), float(d.max())
    targets = [lo + f * (hi - lo) for f in CONTOUR_FRACS]
    tol = 0.25 * (hi - lo) * (CONTOUR_FRACS[1] - CONTOUR_FRACS[0])
    sites = []
    for c, w in enumerate(targets):
        got = R.plant_sites_by_wall(xy, env, [w], SITES_PER_CONTOUR, rng, tol=tol)
        for k, (x, y, _, dw) in enumerate(got):
            sites.append(dict(site_id=len(sites), contour=c,
                              wall_frac=CONTOUR_FRACS[c], wall_dist_m=dw,
                              cx=x, cy=y, rank_on_contour=k))
    return sites


def _disc(xy, cx, cy, r):
    return np.flatnonzero(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) <= r)


def control_members(kind, xy, d_wall, site, r, n_ref, rng, bin_m):
    """A group that is not a place field, matched to a real one where it can be.

    Every control but `oversized` holds the member count or the area of the
    real field it stands beside, so the size rules cannot dispose of it on
    size alone and what separates it from a field is spatial structure.
    """
    cx, cy = site['cx'], site['cy']
    dist = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    if kind == 'scattered':
        # The right number of positions, from a neighbourhood 16x the area:
        # no compact core, but not spread so wide that size alone rejects it.
        near = np.argsort(dist)[:min(len(xy), int(SCATTER_AREA * n_ref))]
        return np.sort(rng.choice(near, min(n_ref, len(near)), replace=False))
    if kind == 'shuffled':
        return np.sort(rng.choice(len(xy), min(n_ref, len(xy)), replace=False))
    if kind == 'oversized':
        return np.sort(np.argsort(dist)[:int(OVERSIZED_FRAC * len(xy))])
    if kind == 'split':
        # Two lobes, each half the field's area, centres 3r apart. The second
        # lobe goes to the most open spot at that distance, so it is whole.
        tol = max(0.05 * 3 * r, 1.5 * bin_m)
        ring = np.flatnonzero(np.abs(dist - 3 * r) <= tol)
        if not len(ring):
            return np.array([], int)
        j = ring[np.argmax(d_wall[ring])]
        rr = r / np.sqrt(2.0)
        return np.union1d(_disc(xy, cx, cy, rr), _disc(xy, xy[j, 0], xy[j, 1], rr))
    if kind == 'ring':
        # The same area as the field, as an annulus around the same place.
        return np.flatnonzero((dist >= r) & (dist <= r * np.sqrt(2.0)))
    raise ValueError(kind)


def build_groups(xy, env, radii, bin_m, rng):
    """Every group the run scores, the same for every channel of an arena."""
    sites = plant_sites(xy, env, rng)
    d_wall = R.wall_distance(xy[:, 0], xy[:, 1], env)
    groups = []

    def add(kind, site, scale, r, mem):
        if len(mem) < MIN_MEMBERS:
            return
        groups.append(dict(group_id=len(groups), kind=kind,
                           site_id=site['site_id'], contour=site['contour'],
                           wall_frac=site['wall_frac'],
                           wall_dist_m=site['wall_dist_m'],
                           cx=site['cx'], cy=site['cy'], scale=scale,
                           nominal_r_m=r, n_members=int(len(mem)), members=mem))

    for s in sites:
        for k, r in radii.items():
            add('disc', s, k, r, _disc(xy, s['cx'], s['cy'], r))
    ctl_sites = [s for s in sites if s['rank_on_contour'] < CONTROL_SITES_PER_CONTOUR]
    for s in ctl_sites:
        for k, r in radii.items():
            n_ref = len(_disc(xy, s['cx'], s['cy'], r))
            for kind in ('scattered', 'shuffled', 'split', 'ring'):
                add(kind, s, k, r, control_members(kind, xy, d_wall, s, r, n_ref,
                                                   rng, bin_m))
    for s in [s for s in sites if s['rank_on_contour'] == 0]:
        add('oversized', s, -1, np.nan,
            control_members('oversized', xy, d_wall, s, 0.0, 0, rng, bin_m))
    return sites, groups


# ------------------------------------------------------------ the extent rule

class FeatureSpace:
    """Centroids, distances and member spreads for one channel, on one device.

    The feature matrix goes to the GPU once per channel when it fits, as in
    rules.environment_readout; otherwise everything runs in float32 BLAS on the
    CPU. Either way the quantities are the ones prepare_candidates computes.
    """

    def __init__(self, X, device):
        self.gpu = (_HAS_TORCH and device is not None and device.type == 'cuda'
                    and R._fits_on_gpu(X, device))
        if self.gpu:
            self.X = torch.from_numpy(np.ascontiguousarray(X)).to(device)
            self.Xsq = R._row_sqnorm(self.X)
        else:
            self.X = X
            self.Xsq = R._row_sqnorm(X)

    def centroids(self, groups):
        if self.gpu:
            out = torch.empty((len(groups), self.X.shape[1]), device=self.X.device)
            for i, g in enumerate(groups):
                idx = torch.from_numpy(g['members']).to(self.X.device)
                out[i] = self.X.index_select(0, idx).mean(0)
            return out
        return np.stack([self.X[g['members']].mean(axis=0) for g in groups]
                        ).astype(np.float32)

    def sq_dists(self, M):
        """(N, n_groups) squared distance from every position to each centroid."""
        if self.gpu:
            d2 = self.Xsq[:, None] - 2.0 * (self.X @ M.T) + R._row_sqnorm(M)[None, :]
            return torch.clamp(d2, min=0).cpu().numpy()
        d2 = self.Xsq[:, None] - 2.0 * (self.X @ M.T) + R._row_sqnorm(M)[None, :]
        return np.maximum(d2, 0, out=d2)

    def spread(self, sub):
        """Squared distance of each subsampled member to the subsample's centroid.

        prepare_candidates gets the same numbers from the pairwise block,
        ||x_i - mu||^2 = mean_j d2_ij - mean_jk d2_jk / 2, which is an identity;
        computing them directly avoids an n x n block per group.
        """
        if self.gpu:
            P = self.X.index_select(0, torch.from_numpy(sub).to(self.X.device))
            return ((P - P.mean(0)) ** 2).sum(1).double().cpu().numpy()
        P = self.X[sub]
        diff = P - P.mean(axis=0)
        return (diff * diff).sum(axis=1, dtype=np.float64)

    def close(self):
        if self.gpu:
            del self.X, self.Xsq
            torch.cuda.empty_cache()


def score(mask, imask, ts, G, C, lim):
    """Everything measured about one drawn field, against the true one."""
    sh = R.field_shape(mask, G)
    cc, ncomp = R.largest_component_fraction(mask)
    inter = float((mask & imask).sum())
    n_rec, n_true = float(mask.sum()), float(imask.sum())
    union = n_rec + n_true - inter
    ba = G['bin_area']
    return dict(
        iou=inter / union if union else 0.0,
        recall=inter / n_true if n_true else 0.0,
        precision=inter / n_rec if n_rec else 0.0,
        center_err_m=(float(np.hypot(sh['cx'] - ts['cx'], sh['cy'] - ts['cy']))
                      if n_rec else np.nan),
        log2_area_ratio=float(np.log2(max(sh['area'], ba) / max(ts['area'], ba))),
        rec_area_m2=float(sh['area']), rec_r_eq_m=float(sh['r_eq']),
        elongation=float(sh['elongation']),
        cc_frac=float(cc), n_components=int(ncomp),
        pass_size=bool(lim[0] <= sh['area'] <= lim[1]),
        pass_contiguity=bool(cc >= C['CC_FRAC_MIN']))


def evaluate(fs, groups, G, occupied, flat, C, lim, q_grid, rng,
             keep_grid_ids=(), chunk=128):
    """The model's extent rule on every group at every q, scored.

    Mirrors prepare_candidates and admit_fields for one group: centroid over
    all members; member spread from a subsample of at most SIGMA_MAX_MEMBERS,
    measured to the subsample's own centroid; sigma solved so the ACT_THRESH
    cut lands at the q-th percentile of that spread, with the readout's own
    fallback to 1 when it is degenerate; the response binned by its maximum;
    the mask from rules.mask_from_grid; admission by Rules 8, 9 and 1. Rule 11
    needs a population and has none to act on here.

    Distance to the centroid does not depend on q, only sigma does, so each
    group's distances are computed once and the whole q sweep reads them.

    Returns (rows, kept): one row per (group, q), plus for each group in
    `keep_grid_ids` the per-bin distance grid and member spread V1 draws from,
    and for every group the mask drawn at Q_OP, bit-packed, for V4.
    """
    n_bins = G['gx'] * G['gy']
    two_ln = 2.0 * np.log(1.0 / C['ACT_THRESH'])
    q_arr = np.asarray(q_grid, float)
    keep = set(keep_grid_ids)
    rows, grids, masks_op = [], {}, {}
    i_op = int(np.flatnonzero(np.isclose(q_arr, Q_OP))[0])
    for a in range(0, len(groups), chunk):
        part = groups[a:a + chunk]
        D2 = fs.sq_dists(fs.centroids(part))
        for j, g in enumerate(part):
            bin_d2 = np.full(n_bins, np.inf)
            np.minimum.at(bin_d2, flat, D2[:, j].astype(np.float64))
            mem = g['members']
            sub = mem if len(mem) <= C['SIGMA_MAX_MEMBERS'] else \
                rng.choice(mem, C['SIGMA_MAX_MEMBERS'], replace=False)
            dc2 = np.sort(np.maximum(fs.spread(sub), 0.0))
            q2 = np.percentile(dc2, q_arr)
            imask = np.zeros(n_bins, bool)
            imask[flat[mem]] = True
            imask = imask.reshape(G['gx'], G['gy']) & G['in_env']
            ts = R.field_shape(imask, G)
            static = {k: v for k, v in g.items() if k != 'members'}
            static.update(ideal_area_m2=float(ts['area']),
                          ideal_r_eq_m=float(ts['r_eq']))
            for i, q in enumerate(q_arr):
                sigma = np.sqrt((q2[i] - dc2[0]) / two_ln) if q2[i] > dc2[0] else 0.0
                s = sigma if sigma > 0 else 1.0
                grid = np.exp(-bin_d2 / (2.0 * s * s)).astype(np.float32)
                mask, _ = R.mask_from_grid(grid, G, occupied, C)
                rows.append(dict(static, q=float(q), sigma=float(sigma),
                                 **score(mask, imask, ts, G, C, lim)))
                if i == i_op and g['kind'] == 'disc':
                    masks_op[g['group_id']] = np.packbits(mask.ravel())
            if g['group_id'] in keep:
                grids[g['group_id']] = dict(bin_d2=bin_d2.astype(np.float32),
                                            dc2=dc2, imask=imask)
        del D2
    return rows, dict(grids=grids, masks_op=masks_op)


# ------------------------------------------------------------ the pipeline

def pipeline_libraries(X, xy, env, base_C, device, tag, q_list, verbose=True):
    """The real tree and rules at each q. One tree serves every q.

    Configured exactly as Experiment 2 builds its libraries, so the q = 65
    library here is Experiment 2's library.
    """
    D2 = R.feature_sq_distances(X, device=device, verbose=verbose)
    rng = np.random.default_rng(base_C['RANDOM_SEED'])
    feat_med = R._median_offdiag(D2, rng)
    d2xy = ((xy[:3000, None, :] - xy[None, :3000, :]) ** 2).sum(-1)
    xy_med = float(np.median(d2xy[np.triu_indices(len(d2xy), 1)]))
    tree = R.build_tree(D2, xy, feat_med, xy_med, cfg=base_C, verbose=verbose)
    rows, scale_rows = [], []
    for q in q_list:
        C = R.resolve_cfg(dict(base_C, EXTENT_PCTL=q))
        ctx = R.prepare_candidates(X, xy, env, D2, feat_med, xy_med, cfg=C,
                                   device=device, tree=tree, tag=f'{tag}/q{q:g}',
                                   verbose=verbose)
        bank, _, rep = R.admit_fields(ctx, cfg=C, verbose=verbose)
        del ctx
        row = dict(q=float(q), n_fields=int(len(bank)),
                   n_candidates=int(rep['n_candidates']))
        if len(bank):
            row.update(median_radius_m=float(bank.radius_env_m.median()),
                       min_radius_m=float(bank.radius_env_m.min()),
                       max_radius_m=float(bank.radius_env_m.max()),
                       n_scales=int(bank.scale_band.nunique()),
                       median_elongation=float(bank.elongation.median()))
        if len(bank) >= 10:
            row.update(
                rho_elong_wall=float(spearmanr(bank.elongation,
                                               bank.dist_to_wall_m)[0]),
                rho_radius_wall=float(spearmanr(bank.radius_env_m,
                                                bank.dist_to_wall_m)[0]))
        rows.append(row)
        for sc, n in bank.scale_band.value_counts().sort_index().items():
            scale_rows.append(dict(q=float(q), scale=int(sc), n_fields=int(n)))
        print(f'  [{tag}] q = {q:g}: {len(bank)} fields', flush=True)
    del D2
    return rows, scale_rows


# ------------------------------------------------------------ one arena

def _git_rev():
    try:
        return subprocess.check_output(['git', '-C', REPO, 'rev-parse', '--short',
                                        'HEAD'], text=True).strip()
    except Exception:                                             # noqa: BLE001
        return 'unknown'


def run_arena(env_name, chans, stages, args, base_C, device, cache_dir, q_grid):
    data_path = f'{args.data_dir}/{env_name}.h5'
    xml_path = f'{REPO}/simulation/worlds/environments/vpce/{env_name}.xml'
    if not os.path.exists(data_path):
        print(f'\n[{env_name}] no dataset at {data_path} -- skipping', flush=True)
        return False
    print(f'\n{"=" * 72}\n{env_name}\n{"=" * 72}', flush=True)
    t_env = time.time()
    rng = np.random.default_rng(args.seed)
    blocks, xy = ch.load_channel_blocks(data_path)
    if args.subsample and args.subsample < len(xy):
        sel = np.sort(rng.choice(len(xy), args.subsample, replace=False))
        blocks, xy = {k: v[sel] for k, v in blocks.items()}, xy[sel]
        print(f'  subsampled to {len(xy)} -- NOT comparable to a full run')
    root = ET.parse(xml_path).getroot()
    env = R.build_env(xy, root)
    C = R.resolve_grid_cfg(base_C, xy, env=env)
    G = R._grid_setup(env, C)
    flat = R._bin_indices(xy, G)
    occupied = np.zeros(G['gx'] * G['gy'], bool)
    occupied[flat] = True
    occupied = occupied.reshape(G['gx'], G['gy'])
    radii, r_min, r_max, a_min, a_max = scale_radii(env, C)
    sites, groups = build_groups(xy, env, radii, G['bin_m'], rng)
    rec_groups = [g for g in groups if g['kind'] == 'disc']
    n_sites = len(sites)
    print(f'  {len(xy)} positions, bin {G["bin_m"]:.4f} m, '
          f'Rule 8/9 radius {r_min:.3f}-{r_max:.3f} m')
    print(f'  ideal radii by scale: ' +
          ', '.join(f'{k}: {r:.3f} m' for k, r in radii.items()))
    print(f'  {n_sites} sites, {len(rec_groups)} true fields, '
          f'{len(groups) - len(rec_groups)} non-fields, {len(q_grid)} values of q',
          flush=True)

    out = f'{cache_dir}/{env_name}'
    os.makedirs(out, exist_ok=True)
    landmarks = [(float(l.get('x')), float(l.get('y')))
                 for l in root.findall('landmark')]
    geom = {k: (float(v) if isinstance(v, (int, float, np.floating)) and
                not isinstance(v, bool) else v) for k, v in env.items()}
    meta = dict(env=env_name, git=_git_rev(), started=time.strftime('%Y-%m-%d %H:%M'),
                q_grid=list(map(float, q_grid)), pipeline_q=list(map(float, args.pipeline_q)),
                q_op=Q_OP, seed=args.seed, subsample=args.subsample,
                n_positions=int(len(xy)), bin_m=float(G['bin_m']),
                bin_area=float(G['bin_area']), gx=int(G['gx']), gy=int(G['gy']),
                r_min=r_min, r_max=r_max, area_min=a_min, area_max=a_max,
                radii={str(k): v for k, v in radii.items()},
                geom=dict(geom, landmarks=landmarks,
                          n_landmarks=len(landmarks)),
                stages=sorted(stages), channels_done=[], pipeline_failed={})
    fig = dict(x_edges=G['x_edges'], y_edges=G['y_edges'], in_env=G['in_env'],
               rec_group_ids=np.array([g['group_id'] for g in rec_groups]))
    if 'recovery' in stages:
        ideal = np.stack([_pack_members(g['members'], flat, G) for g in rec_groups])
        fig['ideal_packed'] = ideal

    rec_rows, ctl_rows, pipe_rows, pscale_rows = [], [], [], []
    ex_ids = [g['group_id'] for g in rec_groups if g['scale'] == EXAMPLE_SCALE]
    lim = (a_min, a_max)
    for cname in chans:
        print(f'\n--- {env_name} / {cname} ---', flush=True)
        t0 = time.time()
        X = ch.assemble(blocks, ch.CHANNEL_SETS[cname], normalize=True)
        if 'recovery' in stages:
            fs = FeatureSpace(X, device)
            rows, kept = evaluate(fs, groups, G, occupied, flat, C, lim, q_grid,
                                  np.random.default_rng(args.seed + 1),
                                  keep_grid_ids=ex_ids)
            fs.close()
            for r in rows:
                r.update(env=env_name, channel=cname)
            rec_rows += [r for r in rows if r['kind'] == 'disc']
            ctl_rows += [r for r in rows if r['kind'] != 'disc']
            fig[f'rec_op_{cname}'] = np.stack(
                [kept['masks_op'][g['group_id']] for g in rec_groups])
            # V1's example: the trial at EXAMPLE_SCALE whose IoU at Q_OP is
            # nearest that scale's median -- representative, not the best.
            at = pd.DataFrame([r for r in rows if r['kind'] == 'disc'
                               and r['scale'] == EXAMPLE_SCALE
                               and np.isclose(r['q'], Q_OP)])
            if len(at):
                gid = int(at.iloc[(at.iou - at.iou.median()).abs().argmin()].group_id)
                k = kept['grids'][gid]
                fig[f'ex_{cname}_bin_d2'] = k['bin_d2']
                fig[f'ex_{cname}_dc2'] = k['dc2']
                fig[f'ex_{cname}_gid'] = np.array([gid])
            med = pd.DataFrame(rows).query('kind == "disc"').groupby('q').iou.median()
            print(f'  recovery: median IoU {med.max():.3f} at best q = '
                  f'{med.idxmax():g}, {med.get(Q_OP, np.nan):.3f} at q = {Q_OP:g}  '
                  f'({time.time() - t0:.0f}s)', flush=True)
        if 'pipeline' in stages:
            try:
                p_rows, s_rows = pipeline_libraries(
                    X, xy, env, base_C, device, f'{env_name}/{cname}',
                    args.pipeline_q, verbose=args.verbose)
                pipe_rows += [dict(r, env=env_name, channel=cname) for r in p_rows]
                pscale_rows += [dict(r, env=env_name, channel=cname) for r in s_rows]
            except Exception as ex:                               # noqa: BLE001
                # One library's failure must not cost the recovery results
                # already in hand, nor the other channels.
                meta['pipeline_failed'][cname] = f'{type(ex).__name__}: {ex}'
                print(f'  !! pipeline failed for {cname}, continuing:\n'
                      f'{traceback.format_exc()}', flush=True)
        del X
        meta['channels_done'].append(cname)
        meta['elapsed_s'] = round(time.time() - t_env)
        # Written after every channel, so a job that stops early leaves what
        # it finished.
        _write_arena(out, meta, fig, rec_rows, ctl_rows, pipe_rows, pscale_rows)
    del blocks
    print(f'\n{env_name} done in {time.time() - t_env:.0f}s', flush=True)
    return True


def _pack_members(mem, flat, G):
    m = np.zeros(G['gx'] * G['gy'], bool)
    m[flat[mem]] = True
    return np.packbits(m & G['in_env'].ravel())


def _unpack(packed, gx, gy):
    return np.unpackbits(packed)[:gx * gy].reshape(gx, gy).astype(bool)


def _write_arena(out, meta, fig, rec_rows, ctl_rows, pipe_rows, pscale_rows):
    for name, rows in (('recovery', rec_rows), ('controls', ctl_rows),
                       ('pipeline', pipe_rows), ('pipeline_scales', pscale_rows)):
        if rows:
            pd.DataFrame(rows).to_csv(f'{out}/{name}.csv', index=False)
    np.savez_compressed(f'{out}/figdata.npz', **fig)
    with open(f'{out}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=float)


# ------------------------------------------------------------ the analysis

def load_cache(cache_dir, envs):
    """Every arena with results in the cache, in ENV_ORDER."""
    have = [e for e in os.listdir(cache_dir)
            if os.path.exists(f'{cache_dir}/{e}/meta.json')] \
        if os.path.isdir(cache_dir) else []
    if envs:
        have = [e for e in have if e in envs]
    order = [e for e in ENV_ORDER if e in have] + sorted(set(have) - set(ENV_ORDER))
    data = dict(envs=order, meta={}, fig={}, rec=[], ctl=[], pipe=[], pscale=[])
    for e in order:
        d = f'{cache_dir}/{e}'
        data['meta'][e] = json.load(open(f'{d}/meta.json'))
        data['fig'][e] = dict(np.load(f'{d}/figdata.npz'))
        for key, name in (('rec', 'recovery'), ('ctl', 'controls'),
                          ('pipe', 'pipeline'), ('pscale', 'pipeline_scales')):
            if os.path.exists(f'{d}/{name}.csv'):
                data[key].append(pd.read_csv(f'{d}/{name}.csv'))
    for key in ('rec', 'ctl', 'pipe', 'pscale'):
        data[key] = (pd.concat(data[key], ignore_index=True) if data[key]
                     else pd.DataFrame())
    for key in ('rec', 'ctl'):
        df = data[key]
        if len(df):
            df['admitted'] = df.pass_size & df.pass_contiguity
    return data


def _mats(df, value, q):
    """{(env, channel): (n_groups, n_q) array}, one row per group."""
    out = {}
    for key, g in df.groupby(['env', 'channel'], sort=True):
        piv = g.pivot_table(index='group_id', columns='q', values=value,
                            aggfunc='first', dropna=False)
        out[key] = piv.reindex(columns=q).to_numpy(dtype=float)
    return out


def crossing(q, y):
    """First q at which y rises through 0, by linear interpolation."""
    for i in range(len(y) - 1):
        if not (np.isfinite(y[i]) and np.isfinite(y[i + 1])):
            continue
        if y[i] == 0:
            return float(q[i])
        if y[i] < 0 < y[i + 1]:
            return float(q[i] + (q[i + 1] - q[i]) * (-y[i]) / (y[i + 1] - y[i]))
    return np.nan


def plateau(q, y, ok):
    """The run of q around the best value over which ok(y) holds."""
    i = int(np.nanargmax(y))
    lo = hi = i
    while lo > 0 and ok(y[lo - 1]):
        lo -= 1
    while hi < len(y) - 1 and ok(y[hi + 1]):
        hi += 1
    return float(q[lo]), float(q[hi])


def _run_around(q, good, near):
    """The contiguous run of `good` containing index `near`."""
    if near is None or not good[near]:
        return np.nan, np.nan
    lo = hi = near
    while lo > 0 and good[lo - 1]:
        lo -= 1
    while hi < len(good) - 1 and good[hi + 1]:
        hi += 1
    return float(q[lo]), float(q[hi])


def pooled_curves(rec, ctl, q, n_boot, seed):
    """The three criteria against q, pooled, with a bootstrap over (arena, channel).

    Trials within one arena and channel share a feature space, so they are not
    independent; resampling whole (arena, channel) pairs is what makes the
    intervals honest about that.
    """
    M = dict(iou=_mats(rec, 'iou', q), lr=_mats(rec, 'log2_area_ratio', q),
             adm=_mats(rec, 'admitted', q),
             cadm=_mats(ctl[ctl.kind.isin(SINGLE_REGION)], 'admitted', q)
             if len(ctl) else {})
    keys = sorted(M['iou'])

    def curves(pick):
        st = lambda name: np.vstack([M[name][keys[i]] for i in pick
                                     if keys[i] in M[name]])
        out = dict(iou=np.nanmedian(st('iou'), 0), lr=np.nanmedian(st('lr'), 0),
                   tpr=np.nanmean(st('adm'), 0))
        c = [i for i in pick if keys[i] in M['cadm']]
        out['fpr'] = np.nanmean(st('cadm'), 0) if c else np.zeros(len(q))
        out['J'] = out['tpr'] - out['fpr']
        return out

    point = curves(np.arange(len(keys)))
    rng = np.random.default_rng(seed)
    boots = [curves(rng.integers(0, len(keys), len(keys))) for _ in range(n_boot)]
    B = {k: np.array([b[k] for b in boots]) for k in point}
    band = {k: (np.nanpercentile(B[k], 2.5, 0), np.nanpercentile(B[k], 97.5, 0))
            for k in point}
    qa = np.asarray(q, float)
    best = dict(
        iou=np.array([qa[np.nanargmax(b)] for b in B['iou']]),
        cal=np.array([crossing(qa, b) for b in B['lr']]),
        J=np.array([qa[np.nanargmax(b)] for b in B['J']]))
    return dict(q=qa, point=point, band=band, best_boot=best, n_cells=len(keys))


def criteria(cur):
    """Best q, its interval, the nearly-as-good range, and whether Q_OP is in it."""
    q, pt, bb = cur['q'], cur['point'], cur['best_boot']
    i_op = int(np.flatnonzero(np.isclose(q, Q_OP))[0])
    def ci(a, method='linear'):
        # Best-q draws live on the q grid, so their interval is read on it
        # too ('nearest'); the calibration crossing is interpolated and is not.
        if not np.isfinite(a).any():
            return (np.nan, np.nan)
        return (float(np.nanpercentile(a, 2.5, method=method)),
                float(np.nanpercentile(a, 97.5, method=method)))
    out = {}
    i = int(np.nanargmax(pt['iou']))
    peak = pt['iou'][i]
    lo, hi = plateau(q, pt['iou'], lambda v: v >= IOU_PLATEAU * peak)
    out['accuracy'] = dict(best=float(q[i]), ci=ci(bb['iou'], 'nearest'),
                           range=(lo, hi),
                           value_best=float(peak), value_op=float(pt['iou'][i_op]),
                           inside=lo <= Q_OP <= hi)
    x = crossing(q, pt['lr'])
    good = np.abs(pt['lr']) <= CAL_TOL
    near = int(np.nanargmin(np.abs(q - x))) if np.isfinite(x) else \
        (int(np.nanargmin(np.abs(pt['lr']))) if np.isfinite(pt['lr']).any() else None)
    lo, hi = _run_around(q, good, near)
    out['calibration'] = dict(best=x, ci=ci(bb['cal']), range=(lo, hi),
                              value_op=float(2 ** pt['lr'][i_op]),
                              inside=bool(good[i_op]))
    j = int(np.nanargmax(pt['J']))
    lo, hi = plateau(q, pt['J'], lambda v: v >= pt['J'][j] - J_TOL)
    out['discrimination'] = dict(best=float(q[j]), ci=ci(bb['J'], 'nearest'),
                                 range=(lo, hi),
                                 value_best=float(pt['J'][j]),
                                 value_op=float(pt['J'][i_op]),
                                 tpr_op=float(pt['tpr'][i_op]),
                                 fpr_op=float(pt['fpr'][i_op]),
                                 inside=lo <= Q_OP <= hi)
    return out


def cell_optima(rec):
    rows = []
    for (e, c), g in rec.groupby(['env', 'channel']):
        m = g.groupby('q').iou.median()
        adm = g.groupby('q').admitted.mean()
        lr = g.groupby('q').log2_area_ratio.median()
        best = float(m.max())
        at = float(m.get(Q_OP, np.nan))
        rows.append(dict(env=e, channel=c, best_q=float(m.idxmax()), best_iou=best,
                         iou_at_op=at,
                         pct_of_best=100.0 * at / best if best > 0 else np.nan,
                         area_ratio_at_op=float(2 ** lr.get(Q_OP, np.nan)),
                         admitted_at_op=float(adm.get(Q_OP, np.nan)),
                         informative=best >= INFORMATIVE_IOU))
    return pd.DataFrame(rows)


def level_optima(rec, key, q, n_boot, seed):
    """Best q per level of `key` (scale, or wall contour), bootstrapped."""
    rows, curves = [], {}
    rng = np.random.default_rng(seed)
    qa = np.asarray(q, float)
    i_op = int(np.flatnonzero(np.isclose(qa, Q_OP))[0])
    for lev, g in rec.groupby(key):
        arrs = list(_mats(g, 'iou', q).values())
        pt = np.nanmedian(np.vstack(arrs), 0)
        bq = [qa[np.nanargmax(np.nanmedian(np.vstack(
            [arrs[i] for i in rng.integers(0, len(arrs), len(arrs))]), 0))]
            for _ in range(n_boot)]
        i = int(np.nanargmax(pt))
        lo, hi = plateau(qa, pt, lambda v: v >= IOU_PLATEAU * pt[i])
        curves[lev] = pt
        rows.append(dict(by=key, level=lev, best_q=float(qa[i]),
                         ci_lo=float(np.percentile(bq, 2.5, method='nearest')),
                         ci_hi=float(np.percentile(bq, 97.5, method='nearest')),
                         range_lo=lo, range_hi=hi, iou_best=float(pt[i]),
                         iou_at_op=float(pt[i_op]),
                         pct_of_best=100.0 * pt[i_op] / pt[i] if pt[i] > 0 else np.nan,
                         op_inside=lo <= Q_OP <= hi, n_trials=int(len(g) / len(q))))
    return pd.DataFrame(rows), curves


def downstream_table(pipe):
    """Each library's quantities at every q, and relative to q = Q_OP."""
    if pipe is None or not len(pipe):
        return pd.DataFrame()
    base = pipe[np.isclose(pipe.q, Q_OP)].set_index(['env', 'channel'])
    rows = []
    for _, r in pipe.iterrows():
        b = base.loc[(r.env, r.channel)] if (r.env, r.channel) in base.index else None
        d = r.to_dict()
        for col in ('n_fields', 'median_radius_m', 'min_radius_m'):
            ok = b is not None and pd.notna(b.get(col)) and b.get(col, 0) > 0 \
                and pd.notna(r.get(col))
            d[f'{col}_pct_of_op'] = 100.0 * r[col] / b[col] if ok else np.nan
        rows.append(d)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

class Figures:
    """Saves every figure as PNG and PDF, and into one multi-page PDF."""

    def __init__(self, fig_dir):
        self.dir = fig_dir
        os.makedirs(fig_dir, exist_ok=True)
        self.written, self.mailed = [], []
        self.book_path = f'{fig_dir}/extent_validation_figures.pdf'
        self.book = PdfPages(self.book_path)

    def save(self, fig, name, mail=True):
        for ext, kw in (('png', dict(dpi=300)), ('pdf', {})):
            p = f'{self.dir}/{name}.{ext}'
            fig.savefig(p, bbox_inches='tight', pad_inches=0.03, **kw)
            self.written.append(p)
        if mail:
            self.book.savefig(fig, bbox_inches='tight', pad_inches=0.03)
            self.mailed.append(f'{self.dir}/{name}.png')
        plt.close(fig)
        print(f'  {self.dir}/{name}.png', flush=True)

    def close(self):
        self.book.close()
        keep = {os.path.abspath(p) for p in self.written + [self.book_path]}
        for f in sorted(os.listdir(self.dir)):
            p = os.path.abspath(f'{self.dir}/{f}')
            if f.startswith('V') and f[1:2].isdigit() and p not in keep:
                os.remove(p)


def _arena_axes(ax, geom, pad=0.04):
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if geom.get('is_circular'):
        R_ = geom['env_R']
        cx, cy = geom.get('env_cx', 0.0), geom.get('env_cy', 0.0)
        ax.add_patch(Circle((cx, cy), R_, fill=False, color=INK_2, lw=0.8, zorder=10))
        ax.set_xlim(cx - R_ * (1 + pad), cx + R_ * (1 + pad))
        ax.set_ylim(cy - R_ * (1 + pad), cy + R_ * (1 + pad))
    else:
        w, h = geom['x_max'] - geom['x_min'], geom['y_max'] - geom['y_min']
        ax.add_patch(plt.Rectangle((geom['x_min'], geom['y_min']), w, h,
                                   fill=False, color=INK_2, lw=0.8, zorder=10))
        m = pad * max(w, h)
        ax.set_xlim(geom['x_min'] - m, geom['x_max'] + m)
        ax.set_ylim(geom['y_min'] - m, geom['y_max'] + m)
    for lx, ly in geom.get('landmarks', []):
        ax.plot(lx, ly, 's', ms=2.2, color=INK, zorder=11)


def _scale_bar(ax, geom):
    """A round-length bar below the arena, in the same place for every shape,
    so panels of one row line up."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = x1 - x0
    length = max(v for v in (0.1, 0.2, 0.5, 1, 2, 5) if v <= 0.25 * span)
    by = y0 - 0.06 * span
    bx = x0 + 0.02 * span
    ax.set_ylim(by - 0.03 * span, y1)
    ax.plot([bx, bx + length], [by, by], color=INK, lw=1.4,
            solid_capstyle='butt', clip_on=False, zorder=12)
    ax.text(bx + length / 2, by + 0.015 * span, f'{length:g} m', ha='center',
            va='bottom', fontsize=6, color=INK_2)


def _centres(e):
    return 0.5 * (e[:-1] + e[1:])


def _example(data, env, cname):
    """V1's example: the stored grid, member spread and true field."""
    f, meta = data['fig'][env], data['meta'][env]
    if f'ex_{cname}_bin_d2' not in f:
        return None
    gx, gy = meta['gx'], meta['gy']
    gid = int(f[f'ex_{cname}_gid'][0])
    row = int(np.flatnonzero(f['rec_group_ids'] == gid)[0])
    rec = data['rec']
    g = rec[(rec.env == env) & (rec.channel == cname) & (rec.group_id == gid)]
    return dict(env=env, channel=cname, gid=gid, gx=gx, gy=gy,
                bin_d2=f[f'ex_{cname}_bin_d2'].astype(float).reshape(gx, gy),
                dc2=np.sort(f[f'ex_{cname}_dc2']),
                imask=_unpack(f['ideal_packed'][row], gx, gy),
                in_env=f['in_env'], x_edges=f['x_edges'], y_edges=f['y_edges'],
                bin_area=meta['bin_area'], geom=meta['geom'],
                rows=g.sort_values('q'))


def fig_mechanism(ex, figs):
    """V1 -- what q does, on one field.

    Every spot on the floor has a q at which it would join the field: the
    percentile of the group's own spread that its distance to the group's
    typical view corresponds to. Map that and the whole family of fields is
    on one panel -- the field at any q is the region below q.
    """
    bin_d2, dc2, imask, in_env = ex['bin_d2'], ex['dc2'], ex['imask'], ex['in_env']
    xe, ye = ex['x_edges'], ex['y_edges']
    xc, yc = _centres(xe), _centres(ye)
    ba = ex['bin_area']
    finite = np.isfinite(bin_d2) & in_env
    d2min = float(bin_d2[finite].min())
    v = bin_d2 - d2min + dc2[0]
    qj = 100.0 * np.interp(v, dc2, np.linspace(0, 1, len(dc2)))
    never = finite & (v > dc2[-1])
    qj = np.where(finite & ~never, qj, np.nan)

    qs = np.arange(0, 101)
    bounds = d2min + (np.percentile(dc2, qs) - dc2[0])
    true_in = np.array([((bin_d2 <= b) & finite & imask).sum() for b in bounds]) * ba
    let_in = np.array([((bin_d2 <= b) & finite & ~imask).sum() for b in bounds]) * ba
    whole = true_in + let_in
    true_area = imask.sum() * ba
    q_cal = crossing(qs.astype(float), whole - true_area)

    fig = plt.figure(figsize=(FIG_W, 4.55))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.0], hspace=0.42)
    top = gs[0].subgridspec(1, 4, width_ratios=[1.0, 0.045, 0.34, 1.3], wspace=0.05)
    bot = gs[1].subgridspec(1, 3, wspace=0.12)

    # (a) the join map
    ax = fig.add_subplot(top[0])
    _arena_axes(ax, ex['geom'])
    ax.pcolormesh(xe, ye, np.ma.masked_where(~never, never).T.astype(float),
                  cmap=mcolors.ListedColormap([NEVER_FILL]), vmin=0, vmax=1,
                  rasterized=True, zorder=1)
    pm = ax.pcolormesh(xe, ye, np.ma.masked_invalid(qj).T, cmap=QJOIN_CMAP,
                       vmin=0, vmax=100, rasterized=True, zorder=2)
    ax.contour(xc, yc, imask.T.astype(float), [0.5], colors='white',
               linewidths=2.0, zorder=5)
    ax.contour(xc, yc, imask.T.astype(float), [0.5], colors=INK,
               linewidths=0.9, zorder=6)
    _panel_label(ax, 'a', dx=-6)
    cax = fig.add_subplot(top[1])
    cb = fig.colorbar(pm, cax=cax)
    cb.set_ticks([0, 20, 40, 60, 80, 100])
    cb.outline.set_linewidth(0)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label('q at which the spot joins the field', fontsize=6.5)
    ax.text(0.0, -0.01, 'black outline: the true field\n'
            'grey: never joins, even at q = 100', transform=ax.transAxes,
            fontsize=6, color=INK_2, ha='left', va='top', linespacing=1.3)

    # (b) what the boundary takes in as q grows
    ax = fig.add_subplot(top[3])
    ax.fill_between(qs, 0, true_in, color=BLUE, alpha=0.18, lw=0)
    ax.plot(qs, true_in, color=BLUE, lw=1.3, label='true field inside the boundary')
    ax.plot(qs, let_in, color=ORANGE, lw=1.3, label='other floor let in (look-alikes)')
    ax.plot(qs, whole, color=INK, lw=1.5, label='whole field drawn')
    ax.axhline(true_area, color=MUTED, lw=0.8, zorder=1)
    ax.annotate('size of the true field', xy=(2, true_area), xytext=(0, 2),
                textcoords='offset points', fontsize=6, color=INK_2, va='bottom')
    if np.isfinite(q_cal):
        ax.plot([q_cal], [true_area], 'o', ms=4.5, mfc='white', mec=INK, mew=1.0,
                zorder=5)
    _mark_op(ax, text=True)
    _q_axis(ax)
    top_y = max(2.2 * true_area, float(np.nanmax(whole[qs <= 95])) * 1.05)
    ax.set_ylim(0, top_y)
    ax.set_ylabel('floor area (m$^2$)')
    ax.legend(loc='upper left', handlelength=1.6)
    _grid_y(ax)
    _panel_label(ax, 'b')

    # (c-e) the field itself at three settings
    rows = ex['rows'].set_index('q')
    for k, q in enumerate(EXAMPLE_Q):
        ax = fig.add_subplot(bot[k])
        _arena_axes(ax, ex['geom'])
        b = d2min + (np.percentile(dc2, q) - dc2[0])
        m = finite & (bin_d2 <= b)
        ax.pcolormesh(xe, ye, np.ma.masked_where(~m, m).T.astype(float),
                      cmap=mcolors.ListedColormap([BLUE]), vmin=0, vmax=1,
                      alpha=0.45, rasterized=True, zorder=2)
        ax.contour(xc, yc, m.T.astype(float), [0.5], colors=BLUE,
                   linewidths=0.8, zorder=4)
        ax.contour(xc, yc, imask.T.astype(float), [0.5], colors=INK,
                   linewidths=0.9, zorder=5)
        inter = float((m & imask).sum())
        iou = inter / float((m | imask).sum())
        ratio = m.sum() / max(imask.sum(), 1)
        verdict = ('too small' if ratio < 0.8 else 'too large' if ratio > 1.25
                   else 'about right')
        ax.set_title(f'q = {q}: {verdict}\nIoU {iou:.2f}, area ×{ratio:.2f}',
                     fontsize=7, pad=3)
        if k == 0:
            _scale_bar(ax, ex['geom'])
        _panel_label(ax, 'cde'[k], dx=-6)
    figs.save(fig, 'V1_what_q_does')
    return dict(q_cal=q_cal, true_area=true_area)


def _ci(v, digits=0):
    return f'{_q(v[0], digits)}-{_q(v[1], digits)}'


def fig_criteria(cur, crit, rec, figs):
    """V2 -- the three criteria against q. The main result.

    Each panel's key number sits under its title rather than beside its
    marker, and the marks every panel shares are explained once, in a key
    above the three. Annotations placed among the curves collide with
    whatever the data happen to do, and where the data go is what this figure
    exists to find out.
    """
    q, pt, band = cur['q'], cur['point'], cur['band']
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.35),
                             gridspec_kw=dict(wspace=0.5))
    per_env = {e: g for e, g in rec.groupby('env')}

    def best_mark(ax, x, y):
        ax.plot([x], [y], 'o', ms=4.8, mfc='white', mec=INK, mew=1.1, zorder=5)

    # (a) accuracy
    ax, a = axes[0], crit['accuracy']
    ax.axvspan(*a['range'], color=SHADE, lw=0, zorder=0)
    for e, g in per_env.items():
        m = g.groupby('q').iou.median().reindex(q)
        ax.plot(q, m.values, color=RULE_GRAY, lw=0.6, zorder=1)
    ax.fill_between(q, *band['iou'], color=INK, alpha=0.13, lw=0, zorder=2)
    ax.plot(q, pt['iou'], color=INK, lw=1.6, zorder=3)
    best_mark(ax, a['best'], a['value_best'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('overlap with the true field (IoU)')
    _title(ax, 'Accuracy', f'best at q = {_q(a["best"])}  (95% CI {_ci(a["ci"])})')

    # (b) calibration
    ax, c = axes[1], crit['calibration']
    if np.isfinite(c['range'][0]):
        ax.axvspan(*c['range'], color=SHADE, lw=0, zorder=0)
    for e, g in per_env.items():
        m = g.groupby('q').log2_area_ratio.median().reindex(q)
        ax.plot(q, m.values, color=RULE_GRAY, lw=0.6, zorder=1)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.fill_between(q, *band['lr'], color=INK, alpha=0.13, lw=0, zorder=2)
    ax.plot(q, pt['lr'], color=INK, lw=1.6, zorder=3)
    if np.isfinite(c['best']):
        best_mark(ax, c['best'], 0.0)
    lo = max(float(np.floor(min(-1, np.nanmin(pt['lr'])))), -3)
    hi = min(float(np.ceil(max(1, np.nanmax(pt['lr'])))), 3)
    ticks = np.arange(lo, hi + 1)
    ax.set_ylim(lo - 0.15, hi + 0.15)
    ax.set_yticks(ticks)
    ax.set_yticklabels([_ratio_label(t) for t in ticks])
    ax.set_ylabel('drawn area ÷ true area')
    _title(ax, 'Calibration',
           f'right size at q = {_q(c["best"])}  (95% CI {_ci(c["ci"])})'
           if np.isfinite(c['best']) else 'never the right size in this range')

    # (c) discrimination, labelled at the line ends
    ax, d = axes[2], crit['discrimination']
    ax.axvspan(*d['range'], color=SHADE, lw=0, zorder=0)
    series = (('tpr', BLUE, 'place fields'), ('fpr', ORANGE, 'non-fields'),
              ('J', INK, 'difference, J'))
    for key, col, _ in series:
        ax.fill_between(q, 100 * band[key][0], 100 * band[key][1], color=col,
                        alpha=0.15, lw=0, zorder=2)
        ax.plot(q, 100 * pt[key], color=col, lw=1.6 if key == 'J' else 1.3,
                zorder=3)
    best_mark(ax, d['best'], 100 * d['value_best'])
    ax.set_ylim(0, 105)
    ax.set_ylabel('% of groups admitted')
    _end_labels(ax, q[-1], [100 * pt[k][-1] for k, _, _ in series],
                [l for _, _, l in series], [c_ for _, c_, _ in series], gap=11)
    _title(ax, 'Discrimination', f'J best at q = {_q(d["best"])}  (95% CI {_ci(d["ci"])})')

    for k, ax in enumerate(axes):
        _q_axis(ax)
        _mark_op(ax)
        _grid_y(ax)
        _panel_label(ax, 'abc'[k], dx=-26, dy=14)
    _figure_key(fig, [
        Line2D([], [], color=INK, lw=1.6, label='all arenas and channels'),
        Patch(facecolor=INK, alpha=0.13, label='95% interval'),
        Line2D([], [], color=RULE_GRAY, lw=0.8, label='one arena'),
        Line2D([], [], color='none', marker='o', ms=4.8, mfc='white', mec=INK,
               mew=1.1, label='best q'),
        Patch(facecolor=SHADE, label='nearly as good as the best'),
        Line2D([], [], color=INK, lw=0.8, label=f'q = {Q_OP:g}, the value in use')],
        y=1.04, ncol=6)
    figs.save(fig, 'V2_three_criteria')


def _ratio_label(t):
    t = int(round(t))
    if t == 0:
        return '×1'
    if t > 0:
        return f'×{2 ** t}'
    return {-1: '×½', -2: '×¼', -3: '×⅛'}.get(t, f'×1/{2 ** -t}')


def fig_robustness(cells, lev_scale, curves_scale, lev_cont, curves_cont, q,
                   envs, chans, crit, figs):
    """V3 -- does q = 65 hold across arena x channel, scale and wall distance?"""
    fig = plt.figure(figsize=(FIG_W, 5.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.62)
    top = outer[0].subgridspec(1, 4, width_ratios=[1.6, 0.05, 0.42, 1.0],
                               wspace=0.05)
    bot = outer[1].subgridspec(1, 2, wspace=0.3)

    # (a) arena x channel
    ax = fig.add_subplot(top[0])
    H = np.full((len(envs), len(chans)), np.nan)
    B = np.full_like(H, np.nan)
    inf = np.zeros_like(H, bool)
    for _, r in cells.iterrows():
        if r.env in envs and r.channel in chans:
            i, j = envs.index(r.env), chans.index(r.channel)
            H[i, j], B[i, j], inf[i, j] = r.pct_of_best, r.best_q, r.informative
    vmin = 75.0
    norm = mcolors.Normalize(vmin, 100)
    Hs = np.where(inf, np.clip(H, vmin, 100), np.nan)
    im = ax.imshow(Hs, cmap=HEAT_CMAP, norm=norm, aspect='auto')
    ax.imshow(np.where(~inf & np.isfinite(H), 1.0, np.nan),
              cmap=mcolors.ListedColormap([NEVER_FILL]), aspect='auto')
    for i in range(len(envs)):
        for j in range(len(chans)):
            if not np.isfinite(H[i, j]):
                continue
            if inf[i, j]:
                r_, g_, b_, _ = HEAT_CMAP(norm(Hs[i, j]))
                dark = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_ < 0.5
                ax.text(j, i, f'{B[i, j]:g}', ha='center', va='center',
                        fontsize=6, color='white' if dark else INK)
            else:
                ax.text(j, i, '–', ha='center', va='center', fontsize=6.5,
                        color=MUTED)
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels([CHANNEL_LABEL.get(c, c) for c in chans])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([ENV_LABEL.get(e, e) for e in envs])
    ax.tick_params(length=0, labelsize=6.3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(chans)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(envs)), minor=True)
    ax.grid(which='minor', color='white', lw=1.5)
    ax.tick_params(which='minor', length=0)
    ax.text(0, -0.04, 'number: that pair\'s own best q    grey: no field '
            'recovered at any q', transform=ax.transAxes, fontsize=6,
            color=INK_2, ha='left', va='top')
    _title(ax, f'Accuracy at q = {Q_OP:g}, arena by channel')
    ax.title.set_position((0, 1.09))
    cax = fig.add_subplot(top[1])
    cb = fig.colorbar(im, cax=cax, extend='min')
    cb.outline.set_linewidth(0)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label(f'IoU at q = {Q_OP:g}, % of the pair\'s best', fontsize=6.5)
    _panel_label(ax, 'a', dx=-8, dy=16)

    # (b) each pair's own best q
    ax = fig.add_subplot(top[3])
    inf_cells = cells[cells.informative]
    step = float(np.min(np.diff(q))) if len(q) > 1 else 5.0
    edges = np.arange(q[0] - step / 2, q[-1] + step, step)
    a = crit['accuracy']
    ax.axvspan(*a['range'], color=SHADE, lw=0, zorder=0)
    ax.hist(inf_cells.best_q, bins=edges, color=BLUE, rwidth=0.82, zorder=2)
    _mark_op(ax)
    _q_axis(ax)
    ax.set_xlabel("that pair's best q")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel('arena–channel pairs')
    _grid_y(ax)
    _title(ax, 'Where each pair peaks',
           f'median {inf_cells.best_q.median():g}; shaded: pooled range '
           f'within 5% of best' if len(inf_cells) else None)
    _panel_label(ax, 'b', dx=-26, dy=14)

    # (c), (d) curves per scale and per wall contour
    for k, (lev, curves, colors, lab, title, key_title) in enumerate((
            (lev_scale, curves_scale, SCALE_COLORS,
             lambda v: f'scale {int(v)}', 'By field size',
             'scale 0 finest → 5 coarsest'),
            (lev_cont, curves_cont, CONTOUR_COLORS,
             lambda v: f'{CONTOUR_FRACS[int(v)]:.2f}',
             'By distance from the wall', 'wall (0) → open floor (1)'))):
        ax = fig.add_subplot(bot[k])
        for _, r in lev.iterrows():
            lv = int(r.level)
            col = colors[lv % len(colors)]
            ax.plot(q, curves[r.level], color=col, lw=1.3, label=lab(lv))
            ax.plot([r.best_q], [r.iou_best], 'o', ms=4, mfc='white', mec=col,
                    mew=1.1, zorder=4)
        _q_axis(ax)
        _mark_op(ax)
        _grid_y(ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel('overlap with the true field (IoU)')
        inside = int(lev.op_inside.sum())
        _title(ax, title, f'q = {Q_OP:g} within 5% of the best for {inside} of '
                          f'{len(lev)}')
        leg = ax.legend(loc='best', fontsize=6, handlelength=1.4,
                        labelspacing=0.3, ncol=2, columnspacing=1.0,
                        title=key_title, title_fontsize=6, alignment='left')
        _panel_label(ax, 'cd'[k], dx=-26, dy=14)
    figs.save(fig, 'V3_robustness')


def fig_gallery(data, cname, scale, figs, mail):
    """V4 -- what q = 65 draws, every arena, one channel."""
    envs = [e for e in data['envs'] if f'rec_op_{cname}' in data['fig'][e]]
    if not envs:
        return
    geo = {e: data['meta'][e]['geom'] for e in envs}

    def aspect(e):
        g = geo[e]
        if g.get('is_circular'):
            return 1.0
        return (g['x_max'] - g['x_min']) / (g['y_max'] - g['y_min'])

    # Shape along a row, landmarks down the columns: each arena sits directly
    # above its no-landmark twin.
    rank = lambda e: ('_lm0_' in e, ENV_ORDER.index(e) if e in ENV_ORDER else 99)
    compact = sorted([e for e in envs if aspect(e) < 2.5], key=rank)
    wide = sorted([e for e in envs if aspect(e) >= 2.5], key=rank)
    rows = [(compact[i:i + 3], 3) for i in range(0, len(compact), 3)] + \
           [(wide[i:i + 2], 2) for i in range(0, len(wide), 2)]
    heights = [(FIG_W / n) / max(aspect(e) for e in row) * 1.12 + 0.3
               for row, n in rows]
    fig = plt.figure(figsize=(FIG_W, sum(heights) + 0.25))
    gs = fig.add_gridspec(len(rows), 1, height_ratios=heights, hspace=0.12)
    rec = data['rec']
    letter = iter('abcdefghijklmnop')
    for r, (row, n) in enumerate(rows):
        sub = gs[r].subgridspec(1, n, wspace=0.08)
        for k, e in enumerate(row):
            ax = fig.add_subplot(sub[k])
            f, meta = data['fig'][e], data['meta'][e]
            gx, gy = meta['gx'], meta['gy']
            xe, ye = f['x_edges'], f['y_edges']
            xc, yc = _centres(xe), _centres(ye)
            _arena_axes(ax, geo[e])
            ids = f['rec_group_ids']
            sel = rec[(rec.env == e) & (rec.channel == cname) &
                      (rec.scale == scale) & np.isclose(rec.q, Q_OP)]
            want = set(sel.group_id)
            count = np.zeros((gx, gy))
            outlines = []
            for i, gid in enumerate(ids):
                if gid not in want:
                    continue
                count += _unpack(f['ideal_packed'][i], gx, gy)
                outlines.append(_unpack(f[f'rec_op_{cname}'][i], gx, gy))
            ax.pcolormesh(xe, ye, np.ma.masked_where(count == 0, count).T,
                          cmap=mcolors.ListedColormap([IDEAL_FILL, '#bdbbb2',
                                                       '#a7a59b']),
                          vmin=0.5, vmax=3.5, rasterized=True, zorder=1)
            for m in outlines:
                if m.any():
                    ax.contour(xc, yc, m.T.astype(float), [0.5], colors=BLUE,
                               linewidths=0.75, zorder=4)
            _scale_bar(ax, geo[e])
            med = sel.iou.median() if len(sel) else np.nan
            ax.set_title(f'{ENV_LABEL.get(e, e)}\nmedian IoU {med:.2f}',
                         fontsize=6.5, pad=2)
            _panel_label(ax, next(letter), dx=-4, dy=2)
    handles = [Patch(facecolor=IDEAL_FILL, edgecolor='none',
                     label=f'true field, scale {scale}'),
               Line2D([], [], color=BLUE, lw=1.0,
                      label=f'field drawn at q = {Q_OP:g}')]
    # Above the first row's two-line titles, which sit ~0.3 in over its axes.
    y_top = max(a.get_position().y1 for a in fig.axes)
    fig.legend(handles=handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, y_top + 0.36 / fig.get_figheight()),
               title=f'{CHANNEL_LABEL.get(cname, cname)} channel',
               title_fontsize=6.8, fontsize=6.5)
    figs.save(fig, f'V4_gallery_{cname}', mail=mail)


DOWNSTREAM = [('n_fields_pct_of_op', 'Fields in the library', True),
              ('median_radius_m_pct_of_op', 'Median field radius', True),
              ('min_radius_m_pct_of_op', 'Smallest field radius', True),
              ('n_scales', 'Scales occupied', False),
              ('rho_elong_wall', 'Elongation vs wall distance', False),
              ('rho_radius_wall', 'Field size vs wall distance', False)]


def fig_downstream(down, figs):
    """V5 -- the field libraries at q = 50, 65, 80."""
    if not len(down):
        return
    qs = sorted(down.q.unique())
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, 3.9),
                             gridspec_kw=dict(wspace=0.45, hspace=0.55))
    for k, (col, lab, rel) in enumerate(DOWNSTREAM):
        ax = axes.flat[k]
        if col not in down.columns:
            ax.set_visible(False)
            continue
        for _, g in down.groupby(['env', 'channel']):
            g = g.set_index('q').reindex(qs)
            ax.plot(qs, g[col].values, color=RULE_GRAY, lw=0.6, zorder=1)
        med = down.groupby('q')[col].median().reindex(qs)
        ax.plot(qs, med.values, '-o', color=INK, lw=1.6, ms=3.8, mfc='white',
                mew=1.0, zorder=3)
        if rel:
            ax.axhline(100, color=MUTED, lw=0.8, zorder=0)
            ax.set_ylabel(f'% of its value at q = {Q_OP:g}')
        elif col.startswith('rho'):
            ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
            ax.set_ylim(-1, 1)
            ax.set_ylabel('Spearman ρ')
        else:
            ax.set_ylabel('scales')
            ax.set_ylim(0, 6.5)
        ax.set_xticks(qs)
        ax.set_xlim(min(qs) - 5, max(qs) + 5)
        ax.set_xlabel('q')
        ax.axvline(Q_OP, color=INK, lw=0.8, zorder=0.5)
        ax.set_title(lab, loc='left', fontweight='bold')
        _grid_y(ax)
        _panel_label(ax, 'abcdef'[k], dx=-26)
    figs.save(fig, 'V5_downstream')


def fig_controls(rec, ctl, q, figs):
    """V6 -- each kind of non-field, against the real fields it stands beside."""
    if not len(ctl):
        return
    kinds = [k for k in CONTROL_KINDS if (ctl.kind == k).any()]
    fig = plt.figure(figsize=(FIG_W, 2.45))
    gs = fig.add_gridspec(2, len(kinds), height_ratios=[0.5, 1.0], hspace=0.12,
                          wspace=0.16)
    tpr = rec.groupby('q').admitted.mean().reindex(q) * 100
    rng = np.random.default_rng(3)
    for k, kind in enumerate(kinds):
        icon = fig.add_subplot(gs[0, k])
        _control_icon(icon, kind, rng)
        scored = kind in SINGLE_REGION
        icon.set_title(CONTROL_LABEL[kind] + ('' if scored else ' (not scored)'),
                       loc='left', fontsize=6.5, fontweight='bold', pad=3,
                       color=INK if scored else MUTED)
        _panel_label(icon, 'abcde'[k], dx=-4, dy=10)
        ax = fig.add_subplot(gs[1, k])
        f = ctl[ctl.kind == kind].groupby('q').admitted.mean().reindex(q) * 100
        ax.plot(q, tpr.values, color=BLUE, lw=1.2)
        ax.plot(q, f.values, color=ORANGE, lw=1.4)
        ax.set_xlim(0, 101)
        ax.set_xticks([0, 50, 100])
        _mark_op(ax)
        _grid_y(ax)
        ax.set_ylim(0, 105)
        if k == 0:
            ax.set_ylabel('% admitted')
        else:
            ax.set_yticklabels([])
    fig.text(0.5, 0.0, 'q  (% of the group inside its field)', ha='center',
             va='top', fontsize=7)
    _figure_key(fig, [
        Line2D([], [], color=BLUE, lw=1.2, label='real place fields admitted'),
        Line2D([], [], color=ORANGE, lw=1.4, label='this non-field admitted'),
        Line2D([], [], color=INK, lw=0.8, label=f'q = {Q_OP:g}'),
        Line2D([], [], color='none', marker='o', ms=4, mfc='none', mec=INK,
               mew=0.7, label='size of the real field beside it')],
        y=0.99)
    figs.save(fig, 'V6_controls')


def _control_icon(ax, kind, rng):
    """A sketch of the group: the true field's size in outline, members as dots."""
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.05, 1.05)
    ax.add_patch(plt.Rectangle((-2.1, -1.0), 4.2, 2.0, fill=False, color=RULE_GRAY,
                               lw=0.6))
    r = 0.3

    def dots(px, py):
        ax.scatter(px, py, s=1.6, color=ORANGE, lw=0, zorder=3)

    def disc(n, cx, cy, rad):
        t, u = rng.uniform(0, 2 * np.pi, n), np.sqrt(rng.uniform(0, 1, n))
        px, py = cx + rad * u * np.cos(t), cy + rad * u * np.sin(t)
        keep = np.abs(py) < 0.95
        return px[keep], py[keep]

    if kind == 'scattered':
        dots(*disc(60, 0, 0, 4 * r))
    elif kind == 'shuffled':
        dots(rng.uniform(-2.05, 2.05, 60), rng.uniform(-0.95, 0.95, 60))
    elif kind == 'oversized':
        px, py = rng.uniform(-2.05, 2.05, 900), rng.uniform(-0.95, 0.95, 900)
        near = np.argsort(np.hypot(px, py))[:int(OVERSIZED_FRAC * 900)]
        dots(px[near], py[near])
    elif kind == 'split':
        rr = r / np.sqrt(2)
        a, b = disc(30, -1.5 * r, 0, rr), disc(30, 1.5 * r, 0, rr)
        dots(np.r_[a[0], b[0]], np.r_[a[1], b[1]])
    elif kind == 'ring':
        t = rng.uniform(0, 2 * np.pi, 110)
        rad = np.sqrt(rng.uniform(r ** 2, 2 * r ** 2, 110))
        dots(rad * np.cos(t), rad * np.sin(t))
    ax.add_patch(Circle((0, 0), r, fill=False, color=INK, lw=0.6, zorder=4))


# ------------------------------------------------------------------ report

def _wrap(text, width=72):
    return '\n\n'.join(textwrap.fill(p.strip(), width) for p in text.split('\n\n'))


def _and(items):
    items = list(items)
    return items[0] if len(items) == 1 else \
        ', '.join(items[:-1]) + ' and ' + items[-1]


def _q(v, digits=0):
    return '–' if v is None or not np.isfinite(v) else f'{v:.{digits}f}'


class ExtentValidationReport(ExperimentReport):
    """Emailed summary: is q = 65 where the evidence puts it?"""

    experiment = 'extent-validation'

    def verdict(self):
        c = self.crit
        ok = [c[k]['inside'] for k in ('accuracy', 'calibration', 'discrimination')]
        return all(ok), ok

    def title(self):
        c = self.crit
        good, _ = self.verdict()
        return (f'q = {Q_OP:g} {"supported" if good else "NOT where the evidence is"}'
                f' -- best q: accuracy {_q(c["accuracy"]["best"])}, calibration '
                f'{_q(c["calibration"]["best"])}, discrimination '
                f'{_q(c["discrimination"]["best"])}')

    def body(self):
        c, cur, out = self.crit, self.cur, []
        a, k, d = c['accuracy'], c['calibration'], c['discrimination']
        good, ok = self.verdict()
        names = ['accuracy', 'size calibration', 'discrimination']
        if good:
            head = (f'q = {Q_OP:g} is supported. On all three criteria fixed '
                    f'before the run it sits inside the range of q that does '
                    f'nearly as well as the best: accuracy {_q(a["range"][0])}-'
                    f'{_q(a["range"][1])}, calibration {_q(k["range"][0])}-'
                    f'{_q(k["range"][1])}, discrimination {_q(d["range"][0])}-'
                    f'{_q(d["range"][1])}.')
        else:
            miss = [n for n, o in zip(names, ok) if not o]
            head = (f'q = {Q_OP:g} is NOT where this evidence puts it. It falls '
                    f'outside the nearly-as-good range on {_and(miss)}. '
                    f'Read the sections below before the value is changed: the '
                    f'best q for each criterion and its interval are there, and '
                    f'the downstream section shows what a change would do to '
                    f'the libraries.')
        out.append(self.section('In one paragraph', _wrap(head)))

        out.append(self.section('What q is', _wrap(
            f'A place field starts as a group of positions whose views look '
            f'alike. To draw it on the floor, the model decides how unlike the '
            f'group\'s typical view a position may be and still count as inside. '
            f'q sets that line: the field boundary encloses q% of the group\'s '
            f'own positions. At q = {Q_OP:g} the most typical {Q_OP:g}% are '
            f'inside and the least typical {100 - Q_OP:g}% outside.\n\n'
            f'Think of drawing a line around a flock of birds. Around every '
            f'bird, stragglers included, and the line takes in a lot of empty '
            f'sky. Around only the densest core and it misses much of the '
            f'flock. In our case the "sky" is floor that merely looks like the '
            f'field: any position closer to the group\'s typical view than the '
            f'boundary is let in, member or not. So a low q draws fields too '
            f'small, a high q draws them too big and lets in look-alike floor, '
            f'and somewhere between the two the misses and the look-alikes '
            f'balance. V1 shows this happening to one field.')))

        n_env = rec_n = len(self.data['envs'])
        rec = self.data['rec']
        n_ch = rec.channel.nunique()
        n_trials = len(rec) // max(len(cur['q']), 1)
        n_ctl = len(self.data['ctl']) // max(len(cur['q']), 1)
        out.append(self.section('How it was tested', _wrap(
            f'We cannot know where a real place field ought to be, so we made '
            f'fields whose answer we do know. A disc of floor is declared to be '
            f'a place field. The model is handed only the views from inside it '
            f'-- exactly what it gets from a group the clustering produced -- '
            f'and its own extent machinery, unchanged, draws a field. We compare '
            f'what it draws with the disc.\n\n'
            f'Discs come in six sizes, one per scale (0 finest to 5 coarsest, '
            f'taken from each arena\'s own size window), at 24 sites from '
            f'against the wall to the most open floor. That is {n_trials:,} '
            f'true fields across {n_env} arenas and {n_ch} channels, each drawn '
            f'at every q from {cur["q"][0]:g} to {cur["q"][-1]:g}. Beside them, '
            f'{n_ctl:,} groups that are NOT fields (scattered, shuffled, '
            f'oversized, two-lobed, ring-shaped), which a good q must refuse.\n\n'
            f'Intervals are 95% bootstrap intervals over the {cur["n_cells"]} '
            f'arena-channel pairs, because trials within one pair share a '
            f'feature space and are not independent.')))

        tab = pd.DataFrame([
            dict(criterion='accuracy (IoU)', best_q=_q(a['best']),
                 ci=f'{_q(a["ci"][0])}-{_q(a["ci"][1])}',
                 nearly_as_good=f'{_q(a["range"][0])}-{_q(a["range"][1])}',
                 at_65=f'{a["value_op"]:.3f} (best {a["value_best"]:.3f})',
                 inside='yes' if a['inside'] else 'NO'),
            dict(criterion='calibration (area)', best_q=_q(k['best']),
                 ci=f'{_q(k["ci"][0])}-{_q(k["ci"][1])}',
                 nearly_as_good=f'{_q(k["range"][0])}-{_q(k["range"][1])}',
                 at_65=f'area x{k["value_op"]:.2f}',
                 inside='yes' if k['inside'] else 'NO'),
            dict(criterion='discrimination (J)', best_q=_q(d['best']),
                 ci=f'{_q(d["ci"][0])}-{_q(d["ci"][1])}',
                 nearly_as_good=f'{_q(d["range"][0])}-{_q(d["range"][1])}',
                 at_65=f'J {d["value_op"]:.2f} (best {d["value_best"]:.2f})',
                 inside='yes' if d['inside'] else 'NO')])
        tab.columns = ['criterion', 'best q', '95% CI', 'nearly as good',
                       f'at q = {Q_OP:g}', f'{Q_OP:g} inside?']
        out.append(self.section(
            'Where q = 65 sits', self.table(tab) + '\n\n' + _wrap(
                f'"Nearly as good" is fixed in the code, not chosen after '
                f'looking: within 5% of the best median IoU; drawn area within '
                f'25% of the true area; J within 0.05 of its best.\n\n'
                f'1. Accuracy. The median overlap between the drawn field and '
                f'the true one peaks at q = {_q(a["best"])} (95% CI '
                f'{_q(a["ci"][0])}-{_q(a["ci"][1])}). At q = {Q_OP:g} it is '
                f'{a["value_op"]:.3f}, against {a["value_best"]:.3f} at the '
                f'peak.\n\n'
                f'2. Calibration. Drawn fields come out the same size as the '
                f'true ones at q = {_q(k["best"])} (CI {_q(k["ci"][0])}-'
                f'{_q(k["ci"][1])}). At q = {Q_OP:g} the median field is '
                f'x{k["value_op"]:.2f} its true area.\n\n'
                f'3. Discrimination. At q = {Q_OP:g}, {100 * d["tpr_op"]:.0f}% '
                f'of true fields are admitted and {100 * d["fpr_op"]:.0f}% of '
                f'single-region non-fields; the difference J is best at q = '
                f'{_q(d["best"])} (CI {_q(d["ci"][0])}-{_q(d["ci"][1])}).')))

        cells = self.cells
        inf = cells[cells.informative]
        n_ok = int((inf.pct_of_best >= 100 * IOU_PLATEAU).sum())
        worst = inf.sort_values('pct_of_best').head(3)
        lines = [f'{n_ok} of {len(inf)} arena-channel pairs reach at least '
                 f'{100 * IOU_PLATEAU:.0f}% of their own best accuracy at q = '
                 f'{Q_OP:g}. Their own best q has median '
                 f'{inf.best_q.median():g} (middle half '
                 f'{inf.best_q.quantile(0.25):g}-{inf.best_q.quantile(0.75):g}).']
        if len(worst):
            lines.append('Furthest from their own best: ' + '; '.join(
                f'{ENV_LABEL.get(r.env, r.env)}, {CHANNEL_LABEL.get(r.channel, r.channel)} '
                f'-- {r.pct_of_best:.0f}% (its best q = {r.best_q:g})'
                for r in worst.itertuples()) + '.')
        unin = cells[~cells.informative]
        if len(unin):
            lines.append(
                f'{len(unin)} pairs recover no field at any q (best IoU under '
                f'{INFORMATIVE_IOU}) and are left out of that count, because a '
                f'channel that cannot place a field there says nothing about '
                f'which q is right: ' + '; '.join(
                    f'{ENV_LABEL.get(r.env, r.env)}, {CHANNEL_LABEL.get(r.channel, r.channel)}'
                    for r in unin.itertuples()) + '.')
        body = _wrap('\n\n'.join(lines))
        for name, lev, lab in (('scale', self.lev_scale, 'scale'),
                               ('wall contour', self.lev_cont, 'contour')):
            t = lev[['level', 'best_q', 'ci_lo', 'ci_hi', 'range_lo', 'range_hi',
                     'iou_best', 'iou_at_op', 'op_inside']].copy()
            if lab == 'contour':
                t['level'] = [f'{CONTOUR_FRACS[int(v)]:.2f}' for v in t.level]
            else:
                t['level'] = t.level.astype(int)
            for col in ('best_q', 'ci_lo', 'ci_hi', 'range_lo', 'range_hi'):
                t[col] = [_q(v) for v in t[col]]
            t['op_inside'] = np.where(t.op_inside, 'yes', 'NO')
            t.columns = [lab, 'best q', 'CI lo', 'CI hi', 'good from', 'to',
                         'IoU best', f'IoU {Q_OP:g}', f'{Q_OP:g} inside']
            body += f'\n\nBy {name}' + (' (0 = at the wall, 1 = most open floor)'
                                        if lab == 'contour' else '') + ':\n\n' \
                + self.table(t, float_format='%.3f')
        out.append(self.section('Does it hold everywhere?', body))

        down = self.down
        if len(down):
            pooled = down.groupby('q')[[c_ for c_, _, _ in DOWNSTREAM
                                        if c_ in down.columns]].median()
            pooled.columns = [l for c_, l, _ in DOWNSTREAM if c_ in down.columns]
            pooled.index = [f'q = {v:g}' for v in pooled.index]
            txt = _wrap(
                f'The real pipeline -- tree, rules, everything -- rebuilt at q = '
                + ', '.join(f'{v:g}' for v in sorted(down.q.unique())) +
                f'. The q = {Q_OP:g} libraries are Experiment 2\'s libraries. '
                f'Medians over every arena-channel library; the first three rows '
                f'are percent of the same library at q = {Q_OP:g}.')
            out.append(self.section('What changes downstream',
                                    txt + '\n\n' + self.table(
                                        pooled.T.reset_index().rename(
                                            columns={'index': 'quantity'}),
                                        float_format='%.2f')))
        failed = {e: m['pipeline_failed'] for e, m in self.data['meta'].items()
                  if m.get('pipeline_failed')}
        if failed:
            out.append(self.section('Pipeline libraries that failed',
                                    '\n'.join(f'{e} / {c_}: {msg}'
                                              for e, f_ in failed.items()
                                              for c_, msg in f_.items())))

        ctl = self.data['ctl']
        if len(ctl):
            at = ctl[np.isclose(ctl.q, Q_OP)].groupby('kind').admitted.mean() * 100
            out.append(self.section('Two-lobed groups', _wrap(
                f'At q = {Q_OP:g} the two-lobed control is admitted '
                f'{at.get("split", np.nan):.0f}% of the time and the ring '
                f'{at.get("ring", np.nan):.0f}%. These do not enter the choice of '
                f'q. A group described by one centroid cannot represent two '
                f'regions: the centroid falls between the lobes, so the field '
                f'fills the gap, at any q (V6 shows it flat or nearly so). That '
                f'is a limit of one field per group, and the route past it is '
                f'multi-field cells, not a different q.')))

        out.append(self.section('Limits of the test', _wrap(
            'The true fields are discs because we said so. The test shows the '
            'model returns the region it was given; it does not show that real '
            'place fields are discs, or that the groups the clustering builds '
            'are shaped like these.\n\n'
            'Admission here is Rules 8, 9 and 1 on a single field. Rule 11 '
            '(competition) needs a population and is exercised only in the '
            'downstream section.\n\n'
            'Every criterion but one is defined against a truth we constructed. '
            'The downstream section is the part that does not depend on it.')))

        ex = self.example
        cap = [
            ('V1  What q does',
             f'One true field (black outline), {ENV_LABEL.get(ex["env"], ex["env"])}, '
             f'{CHANNEL_LABEL.get(ex["channel"], ex["channel"])} channel, scale '
             f'{EXAMPLE_SCALE} -- the trial whose overlap at q = {Q_OP:g} is '
             f'closest to the median for that scale, so it is typical rather '
             f'than flattering. (a) Every spot on the floor coloured by the q at '
             f'which it would join the field; the field at any q is everything '
             f'darker than q. Look-alike floor far from the field shows up as '
             f'dark patches elsewhere. (b) As q grows, the boundary takes in '
             f'more of the true field (blue) and more look-alike floor '
             f'(orange). The drawn field (black) matches the true size (grey '
             f'line) where the circle sits. (c-e) The field itself at three '
             f'settings.'),
            ('V2  The three criteria -- the main result',
             'Every panel: q along the bottom, the black vertical line at '
             f'q = {Q_OP:g}, grey shading the range that does nearly as well as '
             'the best. Bold line: all arenas and channels pooled, with its 95% '
             'interval; thin grey lines: each arena alone. (a) Overlap between '
             'drawn and true field. (b) Drawn area over true area -- x1 is exact, '
             'below is too small. (c) True fields admitted (blue), non-fields '
             'admitted (orange) and their difference (black).'),
            ('V3  Does it hold everywhere?',
             f'(a) Each cell: the overlap at q = {Q_OP:g} as a percentage of the '
             'best overlap any q achieves for that arena and channel; the number '
             'is that pair\'s own best q. Grey cells recover no field at any q. '
             '(b) Overlap against q for each field size, with each curve\'s '
             'best marked. (c) The same by distance from the wall.'),
            ('V4  What q = 65 draws',
             f'Every arena, {CHANNEL_LABEL.get(self.gallery_channel, self.gallery_channel)} '
             f'channel, scale {GALLERY_SCALE}. Grey: the true fields. Blue: the '
             f'fields drawn at q = {Q_OP:g}. One figure per channel is in the '
             'figure directory; this one is mailed.'),
            ('V5  What changes downstream',
             'Each thin line is one arena-channel library rebuilt by the full '
             'pipeline at each q; bold is the median. (a-c) relative to the same '
             'library at q = 65; (d) number of scales with fields; (e, f) the '
             'wall correlations the geometry experiments report.'),
            ('V6  Each kind of non-field',
             'Top: a sketch of each control, members as orange dots, the size of '
             'the real field it stands beside in black. Below: how often it is '
             'wrongly admitted (orange) against real fields admitted (blue). '
             'Two lobes and ring are shown but not scored, for the reason above.'),
        ]
        out.append(self.section('Figures -- how to read them', '\n\n'.join(
            f'{h}\n' + textwrap.indent(_wrap(t, 68), '    ') for h, t in cap)))

        notes = []
        subs = {e: m['subsample'] for e, m in self.data['meta'].items()
                if m.get('subsample')}
        if subs:
            notes.append('SUBSAMPLED arenas, not comparable to a full run: '
                         + ', '.join(f'{e} ({n})' for e, n in subs.items()))
        revs = sorted({m['git'] for m in self.data['meta'].values()})
        notes.append(f'code revision(s): {", ".join(revs)}')
        grids = {tuple(m['q_grid']) for m in self.data['meta'].values()}
        if len(grids) > 1:
            notes.append('!! arenas were run with different q grids')
        notes.append('arenas: ' + ', '.join(
            f'{e} ({",".join(m["channels_done"])})'
            for e, m in self.data['meta'].items()))
        notes.append(f'PDFs with editable text, and every figure in one PDF, '
                     f'are in {self.fig_dir}')
        out.append(self.section('Run', '\n'.join(notes)))
        return '\n'.join(out)

    def figures(self):
        return list(self.figs.mailed) + [self.figs.book_path]

    def data_files(self):
        return [f'{self.out_dir}/{n}' for n in
                ('q_curves.csv', 'cell_optima.csv', 'level_optima.csv',
                 'downstream.csv') if os.path.exists(f'{self.out_dir}/{n}')]


def report(args, cache_dir, fig_dir, envs):
    data = load_cache(cache_dir, envs)
    if not data['envs'] or not len(data['rec']):
        print(f'\nNothing to report: no recovery results in {cache_dir}')
        return 1
    _style()
    rec, ctl = data['rec'], data['ctl']
    q = sorted(rec.q.unique())
    if not np.isclose(q, Q_OP).any():
        print(f'!! q = {Q_OP:g} is not in the cached grid; cannot report')
        return 1
    print(f'\nreport over {len(data["envs"])} arena(s): {", ".join(data["envs"])}',
          flush=True)

    cur = pooled_curves(rec, ctl, q, args.n_boot, args.seed)
    crit = criteria(cur)
    cells = cell_optima(rec)
    lev_scale, curves_scale = level_optima(rec, 'scale', q, args.n_boot, args.seed)
    lev_cont, curves_cont = level_optima(rec, 'contour', q, args.n_boot, args.seed)
    down = downstream_table(data['pipe'])

    qc = pd.DataFrame(dict(q=cur['q'], **{k: v for k, v in cur['point'].items()},
                           **{f'{k}_lo': v[0] for k, v in cur['band'].items()},
                           **{f'{k}_hi': v[1] for k, v in cur['band'].items()}))
    qc.rename(columns={'lr': 'log2_area_ratio', 'lr_lo': 'log2_area_ratio_lo',
                       'lr_hi': 'log2_area_ratio_hi'}).to_csv(
        f'{cache_dir}/q_curves.csv', index=False)
    cells.to_csv(f'{cache_dir}/cell_optima.csv', index=False)
    pd.concat([lev_scale, lev_cont]).to_csv(f'{cache_dir}/level_optima.csv',
                                            index=False)
    if len(down):
        down.to_csv(f'{cache_dir}/downstream.csv', index=False)

    print('\n  criterion        best q   95% CI      nearly as good   65 inside')
    for k_, v in crit.items():
        print(f'  {k_:15s}  {_q(v["best"]):>6s}   {_q(v["ci"][0]):>3s}-{_q(v["ci"][1]):<5s}'
              f'   {_q(v["range"][0]):>3s}-{_q(v["range"][1]):<10s}   {v["inside"]}')

    print('\nfigures:', flush=True)
    figs = Figures(fig_dir)
    chans = [c for c in SD.CHANNELS if c in set(rec.channel)] + \
        sorted(set(rec.channel) - set(SD.CHANNELS))
    ex_env = args.example_env if args.example_env in data['envs'] else data['envs'][0]
    ex_ch = args.gallery_channel if args.gallery_channel in chans else chans[0]
    ex = _example(data, ex_env, ex_ch)
    if ex is not None:
        fig_mechanism(ex, figs)
    fig_criteria(cur, crit, rec, figs)
    fig_robustness(cells, lev_scale, curves_scale, lev_cont, curves_cont,
                   cur['q'], data['envs'], chans, crit, figs)
    for c in chans:
        fig_gallery(data, c, GALLERY_SCALE, figs, mail=(c == ex_ch))
    fig_downstream(down, figs)
    fig_controls(rec, ctl, q, figs)
    figs.close()

    rep = ExtentValidationReport(env_name=f'{len(data["envs"])} arenas',
                                 out_dir=cache_dir, fig_dir=fig_dir,
                                 log_path=os.environ.get('REALM_LOG_PATH'))
    rep.data, rep.cur, rep.crit, rep.cells = data, cur, crit, cells
    rep.lev_scale, rep.lev_cont, rep.down, rep.figs = lev_scale, lev_cont, down, figs
    rep.example = dict(env=ex_env, channel=ex_ch)
    rep.gallery_channel = ex_ch
    print('\n' + rep.compose(), flush=True)
    if not args.no_email:
        rep.send()
    return 0


# --------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--envs', default=','.join(SD.ALL_ENVS))
    p.add_argument('--channels', default=','.join(SD.CHANNELS))
    p.add_argument('--stages', default='recovery,pipeline',
                   help='recovery (the q sweep against known fields) and/or '
                        'pipeline (the real libraries at --pipeline-q). The '
                        'report needs recovery; pipeline adds V5.')
    p.add_argument('--q', default=','.join(map(str, Q_GRID)),
                   help=f'the q sweep. {Q_OP:g} and every value V1 shows are '
                        f'always added.')
    p.add_argument('--pipeline-q', default=','.join(map(str, PIPELINE_Q)))
    p.add_argument('--gallery-channel', default=GALLERY_CHANNEL,
                   help='the channel V1 and the mailed V4 show')
    p.add_argument('--example-env', default=EXAMPLE_ENV,
                   help='the arena V1 shows')
    p.add_argument('--n-boot', type=int, default=N_BOOT)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lam', type=float, default=0.0)
    p.add_argument('--subsample', type=int, default=0,
                   help='changes results, not only runtime -- for testing only')
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--no-email', action='store_true')
    p.add_argument('--no-report', action='store_true',
                   help='compute and cache only -- for one-job-per-arena runs, '
                        'which a --report-only job then joins')
    p.add_argument('--report-only', action='store_true',
                   help='figures and mail from the cache; no dataset is read')
    p.add_argument('--verbose', action='store_true',
                   help='the rules engine\'s own progress lines')
    p.add_argument('--data-dir', default=f'{REPO}/data/vpce/collect_data')
    p.add_argument('--cache-dir', default=f'{REPO}/data_cache/extent_validation')
    p.add_argument('--fig-dir', default=f'{HERE}/figures/extent_validation')
    return p.parse_args()


def main():
    args = parse_args()
    envs = [e.strip() for e in args.envs.split(',') if e.strip()]
    chans = [c.strip() for c in args.channels.split(',') if c.strip()]
    stages = {s.strip() for s in args.stages.split(',') if s.strip()}
    q_grid = sorted({float(v) for v in args.q.split(',') if v.strip()}
                    | {Q_OP} | {float(v) for v in EXAMPLE_Q})
    args.pipeline_q = sorted({float(v) for v in args.pipeline_q.split(',')
                              if v.strip()} | {Q_OP})
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.report_only:
        return report(args, args.cache_dir, args.fig_dir, envs)

    base_C = R.resolve_cfg(dict(LAMBDA=args.lam, RANDOM_SEED=args.seed,
                                USE_GPU=not args.no_gpu))
    device = R.pick_device(use_gpu=not args.no_gpu)
    print('=' * 72)
    print(f'Extent validation | is q = {Q_OP:g} where the evidence puts it?')
    print(f'  arenas   : {envs}')
    print(f'  channels : {chans}')
    print(f'  stages   : {sorted(stages)}')
    print(f'  q sweep  : {q_grid[0]:g}-{q_grid[-1]:g}, {len(q_grid)} values')
    if 'pipeline' in stages:
        print(f'  pipeline : q = {args.pipeline_q}')
    print('=' * 72, flush=True)

    ran = [e for e in envs if run_arena(e, chans, stages, args, base_C, device,
                                        args.cache_dir, q_grid)]
    if not ran:
        print('\nNo arena had a dataset; nothing was computed.')
        return 1
    if args.no_report:
        return 0
    # Every arena in the cache, not only this run's: a single-arena re-run
    # still reports the whole set.
    return report(args, args.cache_dir, args.fig_dir, None)


if __name__ == '__main__':
    sys.exit(main())
