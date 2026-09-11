"""Do large place fields sit further from the walls and landmarks?

Experiment 3. Takes the field libraries Experiment 2 built -- same arenas, same
channels, same rules, same operating point -- and asks where fields of a given
size are located.

Why the question is asked this way round
----------------------------------------
The first version correlated field area with wall distance over the whole
library, against a geometric null. It found nothing: pooled rho +0.06 against a
null of +0.05, no library of 23 beyond the null, and nothing within scale band
or among unclipped fields. That test could not find the effect it was after.
About 60% of every library is finest-band fields and they sit at every
distance from the wall, so a correlation over all fields mostly measures them.
The hints were all in the large fields -- band 3 rho of +0.51 to +0.72 in four
libraries, the centre of the r = 10 disc at 1.1-2.2x its null in five channels
-- and there are too few of those to move a library-wide statistic.

The claim is one-directional: IF a field is large THEN it sits away from the
walls. Small fields being everywhere is no evidence against it. So this
version fixes size and measures position.

Method
------
  size classes  Fields are ranked by area within each library and cut into
                deciles. The bottom 50% are SMALL, the top 10% LARGE.
                Percentiles rather than scale bands, so every library gives
                about 50 large fields and a class means the same thing in
                every arena -- Experiment 2 found the size distribution
                scales with the arena.
  excess        Per field: wall distance as a fraction of the furthest any
                point can be from a wall (the radius of a disc, half the width
                of the corridor), minus the mean of the same quantity over
                that field's null placements. Clipping and centroid push are
                removed field by field; 0 is where the field's shape would sit
                by chance.
  statistic     Mean excess of LARGE minus mean excess of SMALL, against the
                same difference on every null draw. Prediction: SMALL ~ 0,
                LARGE > 0, rising across the deciles.

Two nulls
---------
  uniform   A field clear of the wall goes to a uniformly random floor
            position, kept only where its recorded shape, grown by one bin,
            lies wholly on the floor -- the same test that made it clear. A
            field the wall cuts is moved only along the wall: a random bearing
            about a disc's centre, a random shift along the corridor's length.
            That leaves its cut, area and wall distance exactly as observed,
            so no shape beyond the wall is ever guessed; a corridor field at
            an end wall cannot slide without changing its cut and stays put.
            No placement changes a field's area, so no field changes class.
  tiling    The same, but each library is placed field by field in Rule 11's
            order, largest first, and a placement is also rejected if it
            breaks Rule 11's spacing against fields of its band already
            placed in that draw. The real library had to satisfy that
            spacing, and Rule 12 deletes any band that fails to cover half the
            floor, so a band of large fields packed into the centre could
            never have been observed at all. A uniform null carries neither
            constraint and can work against the hypothesis. Rule 12's coverage
            test is not re-applied. This is the like-for-like null and the
            one the report leads with.

Only fields clear of the wall enter the wall test
-------------------------------------------------
A library records each field as an ellipse, not a mask, so a field the wall
cut has no recorded shape beyond the wall and the null has to guess one to
move it. On synthetic libraries built from circles clipped by the wall, with
no effect planted, the guess biased the test toward the hypothesis: an
ellipse grown until its in-floor part matched the field left SMALL fields
0.02-0.05 nearer the wall than their null, and put the corridor beyond it at
p <= 0.001 in 3 of 3 libraries, while the same null built from the true
circles was unbiased (p 0.18-0.91). A field clear of the wall has its whole
shape on record, so restricted to those, each placed only where it fits
whole, the test needs no guess. It costs power, and the corridor's large
fields, which fill its width, are rarely clear of it.

"Clear" means the recorded ellipse, grown by one bin, lies wholly on the
floor. The margin is there because the ellipse stands in for an irregular
mask: a field whose ellipse only just fits may have touched the wall.

Landmarks
---------
Distance to a panel is mostly distance to the wall, because panels are mounted
on it. Rather than partial that out, position is measured along the wall, as a
phase from 0 (in front of a panel) to 1 (midway between two): bearing from the
nearest panel in a disc, position along the length in the corridor. In a disc
only fields within half a radius of the wall enter -- near the centre every
panel is about equally far away and bearing says nothing. The test is the same
LARGE minus SMALL excess against the same two nulls, over every field in that
region, clipped or not -- a clipped field's null moves it only along the wall,
which is exactly the direction phase measures. The first version found
landmark distance with wall distance partialled out positive beyond its null
in 8/23 libraries; this is the direct test of that lead.

Inference
---------
Per library: p from a normal fitted to the null draws, and a Benjamini-Hochberg
q across libraries. Six channels in one arena are not independent -- their
libraries share much of their structure -- so the per-arena test averages the
channels' excess and divides by the average of their null standard deviations,
the value for perfectly correlated channels. It can only understate
significance.

Calibration
-----------
On synthetic libraries -- circles clipped by the wall, built largest first
under Rule 11 spacing, in circ_lm8_r3, circ_lm8_r10 and corr_lm8_l10w2 -- at
200 null draws:

  no effect         wall p < 0.05 in 0/13 disc libraries under either null,
                    landmark p < 0.05 in 0/21 across all three arenas. Both
                    lean conservative, median p about 0.6. SMALL fields sat
                    0.006 nearer the wall than their null on average, in 11
                    of 13 libraries, too little to move a p-value.
  planted wall      large fields accepting a position with probability
                    (wall distance / max)^2: found at p < 0.05 in 2/5 disc
                    libraries. About 25 large fields per library are clear of
                    the wall, so this test is underpowered and a result short
                    of significance is weak evidence of no effect.
  planted landmark  large fields accepting a position with probability
                    phase^2: found in 7/8.

An earlier draft moved cut fields with a guessed shape and classed a field as
clear without the one-bin margin. With no effect it put the wall test below
p = 0.05 in 3 of 26 null runs, and LARGE excess averaged +0.03 where it should
be 0; this version averages 0.00. Which of the two changes did that was not
separated -- the uniform null improved too, and there a cut field never meets
a clear one, so the margin may be the part that mattered.

The corridor is reported but not counted toward the wall question. It is 2 m
wide, so a large field cannot avoid spanning its midline; its landmark phase
runs along the length and is unaffected.

Usage
    python run_wall_proximity.py [--envs a,b] [--channels ...] [--n-null 1000]
                                 [--rebuild] [--seed 0]
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

# The four Experiment 2 arenas: three discs spanning 11.1x in area, and the
# 10 x 2 m corridor.
ENVS = SD.AREA_ENVS + SD.ELIAV_ENVS
CHANNELS = SD.CHANNELS
CHANNEL_COLORS = SD.CHANNEL_COLORS

# Place-field formation is Experiment 2's, exactly: its operating point, Rule 2
# off, LAMBDA 0, seed 0. Nothing here is a knob, because the point of reusing
# the libraries is that the two experiments describe the same fields.
PCTL, THRESH = SD.SETTINGS[0]
IOU = None
BASE_C = R.resolve_cfg(dict(LAMBDA=0.0, RANDOM_SEED=0))

DATA_DIR = f'{REPO}/data/vpce/collect_data'
XML_DIR = f'{REPO}/simulation/worlds/environments/vpce'
# Experiment 2's cache, shared on purpose: a library already built there is
# read back rather than rebuilt, and one built here is available to it.
BANK_DIR = f'{REPO}/data_cache/scale_distribution'
OUT_DIR = f'{REPO}/data_cache/wall_proximity'
FIG_DIR = f'{HERE}/figures/wall_proximity'

MIN_FIELDS = SD.MIN_FIELDS
SMALL_MAX_DECILE = 4           # deciles 0-4: the bottom 50% by area
LARGE_MIN_DECILE = 9           # decile 9: the top 10%
MIN_CLASS = 15                 # fields each class needs for a measure to be tested
MIN_ACCEPT = 20                # null placements a field needs to have a baseline
N_NULL = 1000                  # null realisations of each library
MAX_ROUNDS = 60                # redraws before a field is left unplaced
OUTER_FRAC = 0.5               # disc fields this close to the wall enter the landmark test
ALPHA = 0.05
NULLS = ('uniform', 'tiling')
MEASURES = ('wall', 'landmark')
CHUNK_ELEMS = 4_000_000        # template x placement elements per clipping batch
NULL_SAMPLE = 4000             # null values kept per library and class for W2 / W3
SMALL_COLOR, LARGE_COLOR = '0.45', '#b2182b'


def bank_path(env_name, cname):
    """Experiment 2's cache key at the operating point, Rule 2 off."""
    return f'{BANK_DIR}/{env_name}/{cname}_p{PCTL}_t{THRESH:g}_roff_bank.csv'


# ----------------------------------------------------------------- geometry

def load_positions(data_path, n_orientations=8):
    """Positions only, without the feature blocks.

    Identical to the xy that channels.load_channel_blocks returns. Reading the
    blocks costs several GB and is pointless when every library is cached.
    """
    import h5py
    with h5py.File(data_path, 'r') as f:
        xs = np.asarray(f['x'][:], dtype=np.float64)
        ys = np.asarray(f['y'][:], dtype=np.float64)
    n_loc = len(xs) // n_orientations
    return np.stack([xs.reshape(n_loc, n_orientations)[:, 0],
                     ys.reshape(n_loc, n_orientations)[:, 0]],
                    axis=1).astype(np.float32)


def read_landmarks(root):
    """Panels as (x, y, theta, width). theta is the inward surface normal."""
    return [(float(l.get('x')), float(l.get('y')), float(l.get('theta', 0.0)),
             float(l.get('width', 0.0))) for l in root.findall('landmark')]


def landmark_distance(cx, cy, landmarks):
    """Distance from a point to the nearest point on any panel's face.

    Written to fields.csv for reference; the test uses landmark_phase.
    """
    cx, cy = np.asarray(cx, dtype=float), np.asarray(cy, dtype=float)
    if not landmarks:
        return np.full(np.broadcast(cx, cy).shape, np.nan)
    best = np.full(np.broadcast(cx, cy).shape, np.inf)
    for x, y, th, w in landmarks:
        tx, ty = -np.sin(th), np.cos(th)          # along the face
        t = np.clip((cx - x) * tx + (cy - y) * ty, -w / 2.0, w / 2.0)
        best = np.minimum(best, np.hypot(cx - (x + t * tx), cy - (y + t * ty)))
    return best


def landmark_phase(cx, cy, env, landmarks):
    """Position along the wall relative to the panels: 0 in front of a panel,
    1 midway between two.

    In a disc, bearing from the arena centre measured from the nearest panel's
    bearing; in a rectangle, position along the long axis measured from the
    nearest panel's position on it. Both are normalised by half the spacing
    between panels, which assumes the panels are evenly spaced, as they are in
    every arena here. NaN in, NaN out, so unplaced null fields pass through.
    """
    cx, cy = np.asarray(cx, dtype=float), np.asarray(cy, dtype=float)
    if not landmarks:
        return np.full(np.broadcast(cx, cy).shape, np.nan)
    if env['is_circular']:
        x0, y0 = env['env_cx'], env['env_cy']
        th = np.arctan2(cy - y0, cx - x0)
        best = np.full(th.shape, np.inf)
        for x, y, _, _ in landmarks:
            d = np.abs((th - np.arctan2(y - y0, x - x0) + np.pi)
                       % (2.0 * np.pi) - np.pi)
            best = np.minimum(best, d)
        half = np.pi / len(landmarks)
    else:
        long_x = (env['x_max'] - env['x_min']) >= (env['y_max'] - env['y_min'])
        u = cx if long_x else cy
        pos = np.unique(np.round([l[0] if long_x else l[1] for l in landmarks], 6))
        best = np.min(np.abs(u[..., None] - pos), axis=-1)
        half = 0.5 * float(np.median(np.diff(pos))) if len(pos) > 1 else 1.0
    return np.clip(best / half, 0.0, 1.0)


def landmark_eligible(dn, env):
    """Fields that enter the landmark test: within OUTER_FRAC of a radius of
    the wall in a disc, everything in a corridor, which is all near-wall."""
    dn = np.asarray(dn, dtype=float)
    return dn <= OUTER_FRAC if env['is_circular'] else np.isfinite(dn)


def max_wall_distance(env):
    """The furthest any point can be from the boundary: the disc's centre, or
    a rectangle's midline."""
    if env['is_circular']:
        return float(env['env_R'])
    return 0.5 * float(min(env['x_max'] - env['x_min'],
                           env['y_max'] - env['y_min']))


def size_deciles(area):
    """Decile of each field's area within its library, and the nine cuts.

    Ties at a cut go to the upper decile, so decile sizes are unequal where
    many fields share one area -- mostly at the Rule 8 floor -- and class
    sizes are reported rather than assumed. No null placement changes a
    field's area, so a field keeps its decile in every draw.
    """
    cuts = np.percentile(area, np.arange(10, 100, 10))
    return np.searchsorted(cuts, area, side='right'), cuts


# --------------------------------------------------------------- statistics

def _nanmean(x, axis=None):
    """Mean over finite entries; NaN where there are none, without warnings."""
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    cnt = fin.sum(axis=axis)
    tot = np.where(fin, x, 0.0).sum(axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = tot / np.where(cnt > 0, cnt, np.nan)
    return float(out) if axis is None else out


def null_p(obs, null):
    """Two-sided p of the observed value against the null draws, and the
    one-sided p in the predicted (positive) direction.

    From a normal fitted to the draws, not from counting them. An empirical p
    cannot fall below 1 / (n_draws + 1), and Benjamini-Hochberg over 24
    libraries needs 0.05 / 24 = 0.002 for a lone effect to survive, so a
    counted p would make an isolated real effect unpassable by construction.
    """
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) < 10:
        return np.nan, np.nan
    sd = float(null.std(ddof=1))
    if sd == 0:
        return np.nan, np.nan
    z = (obs - float(null.mean())) / sd
    return float(2.0 * stats.norm.sf(abs(z))), float(stats.norm.sf(z))


def bh_q(p):
    """Benjamini-Hochberg q-values; NaN passes through."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return q
    order = ok[np.argsort(p[ok])]
    ranked = p[order] * len(ok) / np.arange(1, len(ok) + 1)
    q[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return q


# -------------------------------------------------------------- the nulls

def _templates(bank, G):
    """Each field's own shape, as bin offsets ranked nearest first in
    elliptical radius about the field's own centre bin.

    The first n offsets, n being the field's measured bin count, are its
    recorded shape at its recorded size. Every bin inside the ellipse would be
    a few bins off instead, which at the Rule 8 floor is a template that can
    never be placed. For a field the wall cut, ranking continues past the wall
    until the in-floor part reaches that count, approximating the field's mask
    where it was observed. That approximation is used only to find how far a
    corridor field reaches along the corridor -- never to move a field off
    the wall.

    Returns (di, dj, ci, cj) per field: ranked offsets, and the centre bin.
    """
    sx = float(G['xc'][1] - G['xc'][0])
    sy = float(G['yc'][1] - G['yc'][0])
    in_env = G['in_env']
    gx, gy = in_env.shape
    out = []
    for r in bank.itertuples(index=False):
        n_bins = max(1, int(round(float(r.area_env_m2) / G['bin_area'])))
        a = max(float(r.semi_major_m), 1e-9)
        b = max(float(r.semi_minor_m), 1e-9)
        th = float(r.orientation_rad)
        ci = int(np.clip(np.round((float(r.centroid_x) - G['xc'][0]) / sx), 0, gx - 1))
        cj = int(np.clip(np.round((float(r.centroid_y) - G['yc'][0]) / sy), 0, gy - 1))
        for grow in (2, 3, 5, 8):
            ri, rj = int(np.ceil(grow * a / sx)) + 1, int(np.ceil(grow * a / sy)) + 1
            di, dj = np.meshgrid(np.arange(-ri, ri + 1), np.arange(-rj, rj + 1),
                                 indexing='ij')
            di, dj = di.ravel(), dj.ravel()
            ox, oy = di * sx, dj * sy
            u = ox * np.cos(th) + oy * np.sin(th)
            v = -ox * np.sin(th) + oy * np.cos(th)
            order = np.argsort((u / a) ** 2 + (v / b) ** 2, kind='stable')
            di, dj = di[order], dj[order]
            I, J = ci + di, cj + dj
            inb = (I >= 0) & (I < gx) & (J >= 0) & (J < gy)
            floor = np.zeros(len(di), dtype=bool)
            floor[inb] = in_env[I[inb], J[inb]]
            cum = np.cumsum(floor)
            if cum[-1] >= n_bins:
                k = int(np.searchsorted(cum, n_bins)) + 1
                break
        else:
            k = len(di)
        out.append((di[:k], dj[:k], ci, cj))
    return out


def _dilate(di, dj):
    """Template offsets grown by one bin in every direction."""
    ai = (di[:, None] + np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])).ravel()
    aj = (dj[:, None] + np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])).ravel()
    u = np.unique(np.stack([ai, aj], axis=1), axis=0)
    return u[:, 0], u[:, 1]


def _fits(ci, cj, di, dj, G):
    """Does a template placed at each bin (ci, cj) lie wholly on the floor?"""
    in_env = G['in_env']
    gx, gy = in_env.shape
    n = len(ci)
    out = np.empty(n, dtype=bool)
    step = max(1, CHUNK_ELEMS // max(len(di), 1))
    for s in range(0, n, step):
        e = min(s + step, n)
        I = ci[s:e, None] + di[None, :]
        J = cj[s:e, None] + dj[None, :]
        Ic, Jc = np.clip(I, 0, gx - 1), np.clip(J, 0, gy - 1)
        out[s:e] = ((I == Ic) & (J == Jc) & in_env[Ic, Jc]).all(axis=1)
    return out


def _n_bins(bank, G):
    return np.maximum(1, np.round(bank.area_env_m2.to_numpy(dtype=float) /
                                  G['bin_area']).astype(int))


def clear_of_wall(bank, G, templates):
    """Fields whose recorded shape, grown by one bin, lies wholly on the floor."""
    n_bins = _n_bins(bank, G)
    out = np.zeros(len(bank), dtype=bool)
    for k, (di, dj, ci, cj) in enumerate(templates):
        ddi, ddj = _dilate(di[:n_bins[k]], dj[:n_bins[k]])
        out[k] = _fits(np.array([ci]), np.array([cj]), ddi, ddj, G)[0]
    return out


def place_library(bank, G, env, templates, clear, obs_x, obs_y, n_draws, rng,
                  spacing=None):
    """n_draws null realisations of a whole library.

    A field clear of the wall goes to a uniformly random floor bin and is kept
    only where its shape, grown by one bin, still lies wholly on the floor --
    the test that made it clear. A field the wall cuts moves only along the
    wall: a random bearing about a disc's centre, or a random shift along a
    corridor's length that keeps it a bin clear of the end walls. Either leaves
    its cut, area and wall distance as observed, so no shape beyond the wall is
    needed. A corridor field within a bin of an end wall stays where it is.

    With `spacing`, a placement is also rejected if it breaks Rule 11's
    distance -- spacing * (r_a + r_b) between centres -- from any field of its
    band already placed in that realisation. Fields go in Rule 11's order,
    largest first, so a band fills the way admit_fields fills it. A rejected
    proposal is redrawn up to MAX_ROUNDS times; a field still unplaced is NaN
    and blocks nothing.

    Returns (n_fields, n_draws) arrays of null centre x and y.
    """
    in_env, xc, yc = G['in_env'], G['xc'], G['yc']
    gx, gy = in_env.shape
    sx, sy = float(xc[1] - xc[0]), float(yc[1] - yc[0])
    env_i, env_j = np.nonzero(in_env)
    n = len(bank)
    X = np.full((n, n_draws), np.nan)
    Y = np.full((n, n_draws), np.nan)
    r_eq = bank.radius_env_m.to_numpy(dtype=float)
    iou = (bank.split_half_iou.to_numpy(dtype=float)
           if 'split_half_iou' in bank else np.zeros(n))
    band = bank.scale_band.to_numpy(dtype=int)
    r_mean = 0.5 * (bank.semi_major_m.to_numpy(dtype=float) +
                    bank.semi_minor_m.to_numpy(dtype=float))
    n_bins = _n_bins(bank, G)
    long_x = (env['x_max'] - env['x_min']) >= (env['y_max'] - env['y_min'])
    step = sx if long_x else sy
    floor_u = xc[env_i] if long_x else yc[env_j]
    u_lo, u_hi = float(floor_u.min()), float(floor_u.max())
    placed = {}
    for k in np.lexsort((-iou, -r_eq)):
        di, dj, ci, cj = templates[k]
        slide = None
        if clear[k]:
            ddi, ddj = _dilate(di[:n_bins[k]], dj[:n_bins[k]])
            mx = sx * float(di[:n_bins[k]].mean())
            my = sy * float(dj[:n_bins[k]].mean())
        elif not env['is_circular']:
            # How far this field reaches along the corridor, from its template's
            # in-floor bins where it was observed.
            I, J = ci + di, cj + dj
            inb = (I >= 0) & (I < gx) & (J >= 0) & (J < gy)
            onf = np.zeros(len(di), dtype=bool)
            onf[inb] = in_env[I[inb], J[inb]]
            u = xc[I[onf]] if long_x else yc[J[onf]]
            lo = u_lo - float(u.min()) + step
            hi = u_hi - float(u.max()) - step
            slide = (lo, hi) if lo <= 0.0 <= hi else None
        peers = np.asarray(placed.get(band[k], []), dtype=int)
        todo = np.arange(n_draws)
        for _ in range(MAX_ROUNDS):
            if not len(todo):
                break
            m = len(todo)
            if clear[k]:
                pick = rng.integers(0, len(env_i), size=m)
                pi, pj = env_i[pick], env_j[pick]
                ok = _fits(pi, pj, ddi, ddj, G)
                cx, cy = xc[pi] + mx, yc[pj] + my
            elif env['is_circular']:
                t = rng.uniform(0.0, 2.0 * np.pi, m)
                dx, dy = obs_x[k] - env['env_cx'], obs_y[k] - env['env_cy']
                cx = env['env_cx'] + dx * np.cos(t) - dy * np.sin(t)
                cy = env['env_cy'] + dx * np.sin(t) + dy * np.cos(t)
                ok = np.ones(m, dtype=bool)
            else:
                s = (rng.uniform(slide[0], slide[1], m) if slide is not None
                     else np.zeros(m))
                cx = np.full(m, obs_x[k]) + (s if long_x else 0.0)
                cy = np.full(m, obs_y[k]) + (0.0 if long_x else s)
                ok = np.ones(m, dtype=bool)
            if spacing is not None and len(peers):
                d = np.hypot(X[np.ix_(peers, todo)] - cx,
                             Y[np.ix_(peers, todo)] - cy)
                ok &= ~(d < spacing * (r_mean[peers, None] + r_mean[k])).any(axis=0)
            X[k, todo[ok]] = cx[ok]
            Y[k, todo[ok]] = cy[ok]
            todo = todo[~ok]
        if spacing is not None:
            placed.setdefault(band[k], []).append(k)
    return X, Y


def class_contrast(obs, null, decile, elig_obs=None, elig_null=None):
    """Excess per field, per decile and per class, observed and per draw.

    A field's baseline is the mean of its accepted null placements, and only
    placements that are themselves eligible count toward it, so a landmark
    baseline is conditioned on the field sitting where the landmark test
    looks. Fields with fewer than MIN_ACCEPT placements have no baseline and
    drop out of every mean.
    """
    obs = np.asarray(obs, dtype=float).copy()
    null = np.asarray(null, dtype=float)
    if elig_obs is not None:
        obs[~elig_obs] = np.nan
        null = np.where(elig_null, null, np.nan)
    mu = _nanmean(null, axis=1)
    mu[np.isfinite(null).sum(axis=1) < MIN_ACCEPT] = np.nan
    ex = obs - mu
    exn = null - mu[:, None]
    small = decile <= SMALL_MAX_DECILE
    large = decile >= LARGE_MIN_DECILE
    out = dict(excess=ex)
    for name, sel in (('small', small), ('large', large)):
        out[f'{name}_n'] = int(np.isfinite(ex[sel]).sum())
        out[f'{name}_obs'] = _nanmean(ex[sel])
        out[f'{name}_null'] = _nanmean(exn[sel], axis=0)
    out['diff_obs'] = out['large_obs'] - out['small_obs']
    out['diff_null'] = out['large_null'] - out['small_null']
    out['dec_obs'] = np.array([_nanmean(ex[decile == d]) for d in range(10)])
    out['dec_null'] = np.stack([_nanmean(exn[decile == d], axis=0)
                                for d in range(10)])
    return out


# ----------------------------------------------------------------- analysis

def analyse_library(bank, env, G, landmarks, n_null, rng, tag):
    """Both measures against both nulls for one arena x channel library."""
    area = bank.area_env_m2.to_numpy(dtype=float)
    cx = bank.centroid_x.to_numpy(dtype=float)
    cy = bank.centroid_y.to_numpy(dtype=float)
    wmax = max_wall_distance(env)
    dn = bank.dist_to_wall_m.to_numpy(dtype=float) / wmax
    ph = landmark_phase(cx, cy, env, landmarks)
    elig = landmark_eligible(dn, env)

    fields = bank[['area_env_m2', 'radius_env_m', 'semi_major_m',
                   'semi_minor_m', 'centroid_x', 'centroid_y',
                   'dist_to_wall_m', 'scale_band']].copy()
    fields['dist_to_landmark_m'] = landmark_distance(cx, cy, landmarks)
    fields['wall_dist_norm'] = dn
    fields['landmark_phase'] = ph
    fields['landmark_eligible'] = elig
    fields.insert(0, 'channel', tag['channel'])
    fields.insert(0, 'env', tag['env'])

    row = dict(tag, env_area_m2=float(env['env_area']),
               aspect=float(env.get('aspect', 1.0)),
               max_wall_dist_m=wmax, n_fields=len(bank), tested=False,
               wall_tested=False, landmark_tested=False)
    if len(bank) < MIN_FIELDS:
        return row, [], fields, {}
    decile, cuts = size_deciles(area)
    small = decile <= SMALL_MAX_DECILE
    large = decile >= LARGE_MIN_DECILE
    fields['size_decile'] = decile + 1
    fields['size_class'] = np.where(large, 'large',
                                    np.where(small, 'small', 'middle'))
    row.update(n_small=int(small.sum()), n_large=int(large.sum()),
               area_cut_small_m2=float(cuts[SMALL_MAX_DECILE]),
               area_cut_large_m2=float(cuts[LARGE_MIN_DECILE - 1]),
               wall_dist_norm_median_small=float(np.median(dn[small]))
               if small.any() else np.nan,
               wall_dist_norm_median_large=float(np.median(dn[large]))
               if large.any() else np.nan)
    templates = _templates(bank, G)
    clear = clear_of_wall(bank, G, templates)
    # Observed centres, measured as the null measures its own: a field clear
    # of the wall by its template's centre, which is where the null puts it;
    # a field the wall cuts by the bank's centroid, which the null only ever
    # rotates or slides. Written over the bank-derived columns so the figures
    # show what was tested; dist_to_wall_m keeps the bank's value.
    sx = float(G['xc'][1] - G['xc'][0])
    sy = float(G['yc'][1] - G['yc'][0])
    n_bins = _n_bins(bank, G)
    xs, ys = cx.copy(), cy.copy()
    for k, (di, dj, ci, cj) in enumerate(templates):
        if clear[k]:
            xs[k] = G['xc'][ci] + sx * float(di[:n_bins[k]].mean())
            ys[k] = G['yc'][cj] + sy * float(dj[:n_bins[k]].mean())
    dn = R.wall_distance(xs, ys, env) / wmax
    ph = landmark_phase(xs, ys, env, landmarks)
    elig = landmark_eligible(dn, env)
    fields['wall_dist_norm'] = dn
    fields['landmark_phase'] = ph
    fields['landmark_eligible'] = elig
    fields['clear_of_wall'] = clear
    row.update(n_clear_small=int((small & clear).sum()),
               n_clear_large=int((large & clear).sum()),
               n_eligible_small=int((small & elig).sum()),
               n_eligible_large=int((large & elig).sum()))
    row['wall_tested'] = min(row['n_clear_small'],
                             row['n_clear_large']) >= MIN_CLASS
    row['landmark_tested'] = min(row['n_eligible_small'],
                                 row['n_eligible_large']) >= MIN_CLASS
    row['tested'] = row['wall_tested'] or row['landmark_tested']
    if not row['tested']:
        return row, [], fields, {}
    measures = [m for m in MEASURES if row[f'{m}_tested']]
    dec_rows, prof = [], {}
    for nn, spacing in (('uniform', None),
                        ('tiling', BASE_C['SAME_SCALE_SEPARATION'])):
        X, Y = place_library(bank, G, env, templates, clear, xs, ys, n_null,
                             rng, spacing=spacing)
        row[f'{nn}_unplaced_frac'] = float(np.mean(~np.isfinite(X)))
        DN = R.wall_distance(X, Y, env) / wmax
        PH = landmark_phase(X, Y, env, landmarks)
        EL = landmark_eligible(DN, env)
        inputs = {'wall': (dn, DN, clear,
                           np.broadcast_to(clear[:, None], DN.shape)),
                  'landmark': (ph, PH, elig, EL)}
        for measure in measures:
            o, nl, eo, en = inputs[measure]
            c = class_contrast(o, nl, decile, eo, en)
            pre = f'{measure}_{nn}'
            nd = c['diff_null'][np.isfinite(c['diff_null'])]
            ln = c['large_null'][np.isfinite(c['large_null'])]
            row.update({
                f'{pre}_n_small': c['small_n'], f'{pre}_n_large': c['large_n'],
                f'{pre}_small': c['small_obs'], f'{pre}_large': c['large_obs'],
                f'{pre}_diff': c['diff_obs'],
                f'{pre}_diff_null_mean': float(nd.mean()) if len(nd) else np.nan,
                f'{pre}_diff_null_sd': (float(nd.std(ddof=1))
                                        if len(nd) > 1 else np.nan),
                f'{pre}_diff_null_lo': (float(np.percentile(nd, 2.5))
                                        if len(nd) else np.nan),
                f'{pre}_diff_null_hi': (float(np.percentile(nd, 97.5))
                                        if len(nd) else np.nan),
                f'{pre}_large_p': null_p(c['large_obs'], ln)[0],
            })
            row[f'{pre}_p'], row[f'{pre}_p_gt'] = null_p(c['diff_obs'], nd)
            fields[f'{pre}_excess'] = c['excess']
            for d in range(10):
                dv = c['dec_null'][d]
                dv = dv[np.isfinite(dv)]
                dec_rows.append(dict(
                    tag, null=nn, measure=measure, decile=d + 1,
                    n_fields=int((decile == d).sum()), excess=c['dec_obs'][d],
                    null_lo=float(np.percentile(dv, 2.5)) if len(dv) else np.nan,
                    null_hi=float(np.percentile(dv, 97.5)) if len(dv) else np.nan))
            if nn == 'tiling':
                # A sample of null values per class, for W2 and W3.
                for cls, sel in (('small', small), ('large', large)):
                    v = nl[sel] if en is None else np.where(en[sel], nl[sel], np.nan)
                    v = v[np.isfinite(v)]
                    if len(v) > NULL_SAMPLE:
                        v = rng.choice(v, NULL_SAMPLE, replace=False)
                    prof[(measure, cls)] = v
    return row, dec_rows, fields, prof


def arena_table(summary, deciles):
    """Per arena, channels averaged, with the conservative null sd.

    Also the trend across deciles: Spearman of decile against the channel-mean
    excess, ten points. The prediction is a rise, not only a gap at the top.
    """
    t = summary[summary.tested]
    rows = []
    for e, g in t.groupby('env', sort=False):
        for measure in MEASURES:
            for nn in NULLS:
                pre = f'{measure}_{nn}'
                if f'{pre}_diff' not in g:
                    continue
                d = g[f'{pre}_diff'].to_numpy(dtype=float)
                m0 = g[f'{pre}_diff_null_mean'].to_numpy(dtype=float)
                sd = g[f'{pre}_diff_null_sd'].to_numpy(dtype=float)
                ok = np.isfinite(d) & np.isfinite(m0) & np.isfinite(sd)
                if not ok.any():
                    continue
                ex = d[ok] - m0[ok]
                s = float(sd[ok].mean())
                z = float(ex.mean() / s) if s > 0 else np.nan
                dd = deciles[(deciles.env == e) & (deciles.measure == measure) &
                             (deciles.null == nn)]
                m = dd.groupby('decile').excess.mean().dropna()
                trend = (stats.spearmanr(m.index, m.values)
                         if len(m) >= 5 and np.ptp(m.values) > 0
                         else (np.nan, np.nan))
                rows.append(dict(
                    env=e, env_area_m2=float(g.env_area_m2.iloc[0]),
                    aspect=float(g.aspect.iloc[0]), measure=measure, null=nn,
                    n_channels=int(ok.sum()), n_positive=int((ex > 0).sum()),
                    small=float(g[f'{pre}_small'].to_numpy(dtype=float)[ok].mean()),
                    large=float(g[f'{pre}_large'].to_numpy(dtype=float)[ok].mean()),
                    excess=float(ex.mean()), sd_conservative=s, z=z,
                    p=float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                    decile_trend_rho=float(trend[0]),
                    decile_trend_p=float(trend[1])))
    a = pd.DataFrame(rows)
    if len(a):
        a['q'] = np.nan
        for _, idx in a.groupby(['measure', 'null']).groups.items():
            a.loc[idx, 'q'] = bh_q(a.loc[idx, 'p'])
    return a


# ------------------------------------------------------------------ figures

FIGURES_WRITTEN = []


def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    FIGURES_WRITTEN.append(p)
    print(f'  {p}', flush=True)


def prune_orphan_figures():
    """Delete W<digit>*.png this run did not write, as Experiment 2 does."""
    keep = {os.path.abspath(p) for p in FIGURES_WRITTEN}
    for p in sorted(glob.glob(os.path.join(FIG_DIR, 'W[0-9]*.png'))):
        if os.path.abspath(p) not in keep:
            os.remove(p)
            print(f'  pruned orphaned figure {os.path.basename(p)}', flush=True)


def fig_excess_by_size(deciles, envs, chans):
    """W1: excess wall distance across size deciles, for both nulls.

    The prediction is flat near zero through the shaded small deciles and
    rising into the large one. Grey is the median, across channels, of each
    channel's null 95% range for that decile's mean.
    """
    dw = deciles[deciles.measure == 'wall']
    if not len(dw):
        return
    fig, axes = plt.subplots(len(NULLS), len(envs), squeeze=False,
                             figsize=(3.4 * len(envs), 2.9 * len(NULLS)))
    legend_ax = None
    for i, nn in enumerate(NULLS):
        dn = dw[dw.null == nn]
        v = dn[['excess', 'null_lo', 'null_hi']].to_numpy(dtype=float)
        lim = 1.1 * float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 0.1
        for j, e in enumerate(envs):
            ax = axes[i][j]
            ax.axvspan(0.5, SMALL_MAX_DECILE + 1.5, color='0.94', lw=0, zorder=0)
            ax.axvspan(LARGE_MIN_DECILE + 0.5, 10.5, color='#f7e1e1', lw=0,
                       zorder=0)
            ax.axhline(0, color='0.6', lw=0.8)
            de = dn[dn.env == e]
            if len(de):
                bnd = de.groupby('decile')[['null_lo', 'null_hi']].median()
                ax.fill_between(bnd.index, bnd.null_lo, bnd.null_hi,
                                color='0.65', alpha=0.4, lw=0)
                for c in chans:
                    g = de[de.channel == c].sort_values('decile')
                    if len(g):
                        ax.plot(g.decile, g.excess, '-', lw=0.9, alpha=0.8,
                                color=CHANNEL_COLORS.get(c, '0.4'), label=c)
                m = de.groupby('decile').excess.mean()
                ax.plot(m.index, m.values, 'o-', color='k', lw=1.6, ms=3,
                        label='channel mean')
                legend_ax = legend_ax or ax
            else:
                ax.text(0.5, 0.5, 'wall test not run\n(too few large fields '
                        'clear of the wall)', transform=ax.transAxes,
                        ha='center', va='center', fontsize=7, color='#b03030')
            ax.set_xlim(0.5, 10.5)
            ax.set_ylim(-lim, lim)
            ax.set_xticks(range(1, 11))
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(e, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{nn} null\nexcess wall distance', fontsize=8)
            if i == len(NULLS) - 1:
                ax.set_xlabel('size decile within library\n'
                              'grey = small, red = large', fontsize=8)
    if legend_ax is not None:
        legend_ax.legend(fontsize=5.5, frameon=False, ncol=2, loc='upper left')
    fig.suptitle('W1  how much further from the wall than chance, by field size\n'
                 'excess = wall distance minus its null, as a fraction of the '
                 'furthest possible (radius of a disc, half-width of the '
                 'corridor)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, 'W1_excess_by_size.png')


def _tested_fields(fields, summary, col):
    t = summary[summary[col].astype(bool)][['env', 'channel']]
    return fields.merge(t, on=['env', 'channel'], how='inner')


def _pooled_null(profiles, env, key):
    parts = [p[key] for (e, _), p in profiles.items() if e == env and key in p]
    return np.concatenate(parts) if parts else np.array([])


def fig_distance_by_class(fields, summary, profiles, envs):
    """W2: where small and large fields sit, channels pooled, observed against
    the tiling null, over the fields clear of the wall that the test uses. A
    large-field curve to the right of its dashed null is further from the
    wall than chance; the small-field pair should overlap."""
    f = _tested_fields(fields, summary, 'wall_tested')
    if not len(f):
        return
    f = f[f.clear_of_wall.astype(bool)]
    fig, axes = plt.subplots(1, len(envs), squeeze=False,
                             figsize=(3.4 * len(envs), 3.2))
    legend_ax = None
    for ax, e in zip(axes[0], envs):
        fe = f[f.env == e]
        if not len(fe):
            ax.text(0.5, 0.5, 'wall test not run\n(too few large fields '
                    'clear of the wall)', transform=ax.transAxes, ha='center',
                    va='center', fontsize=7, color='#b03030')
        else:
            legend_ax = legend_ax or ax
        for cls, col in (('small', SMALL_COLOR), ('large', LARGE_COLOR)):
            obs = fe[fe.size_class == cls].wall_dist_norm.to_numpy(dtype=float)
            nul = _pooled_null(profiles, e, ('wall', cls))
            for vals, ls, lab in ((obs, '-', f'{cls} (n={len(obs)})'),
                                  (nul, '--', f'{cls} null')):
                if len(vals):
                    xs = np.sort(vals)
                    ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), ls,
                            color=col, lw=1.3, label=lab)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(e, fontsize=9)
        ax.set_xlabel('wall distance / furthest possible\n0 = wall, '
                      '1 = centre or midline', fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel('fraction of fields at least this near the wall',
                          fontsize=8)
    if legend_ax is not None:
        legend_ax.legend(fontsize=6, frameon=False, loc='lower right')
    fig.suptitle('W2  where small and large fields clear of the wall sit, '
                 'channels pooled\nsolid = observed, dashed = tiling null',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, 'W2_distance_by_class.png')


def fig_landmark_phase(fields, summary, profiles, envs):
    """W3: position along the wall relative to the panels, small against
    large, observed against the tiling null. Large fields massed toward 1
    sit between panels rather than in front of them."""
    f = _tested_fields(fields, summary, 'landmark_tested')
    if not len(f):
        return
    f = f[f.landmark_eligible.astype(bool)]
    bins = np.linspace(0.0, 1.0, 6)
    fig, axes = plt.subplots(1, len(envs), squeeze=False, sharey=True,
                             figsize=(3.4 * len(envs), 3.2))
    for ax, e in zip(axes[0], envs):
        fe = f[f.env == e]
        for cls, col in (('small', SMALL_COLOR), ('large', LARGE_COLOR)):
            obs = fe[fe.size_class == cls].landmark_phase.to_numpy(dtype=float)
            nul = _pooled_null(profiles, e, ('landmark', cls))
            if len(obs):
                ax.hist(obs, bins=bins, density=True, histtype='step', lw=1.5,
                        color=col, label=f'{cls} (n={len(obs)})')
            if len(nul):
                ax.hist(nul, bins=bins, density=True, histtype='step', lw=1.0,
                        ls='--', color=col, label=f'{cls} null')
        where = ('within half a radius of the wall'
                 if e in summary.env.values and
                 float(summary[summary.env == e].aspect.iloc[0]) == 1.0
                 else 'whole corridor')
        ax.set_title(f'{e}\n({where})', fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel('phase along the wall\n0 = in front of a panel, '
                      '1 = midway between two', fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel('density', fontsize=8)
    axes[0][0].legend(fontsize=6, frameon=False, loc='lower left')
    fig.suptitle('W3  where small and large fields sit relative to the panels, '
                 'channels pooled\nsolid = observed, dashed = tiling null',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, 'W3_landmark_phase.png')


# -------------------------------------------------------------------- report

def _sign_test(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    k = int((x > 0).sum())
    return k, len(x), (float(stats.binom.sf(k - 1, len(x), 0.5))
                       if len(x) else np.nan)


class WallProximityReport(ExperimentReport):
    experiment = 'wall-proximity'

    def title(self):
        a = self.arena
        if a is None or not len(a):
            return 'no testable libraries'
        w = a[(a.measure == 'wall') & (a.null == 'tiling') & (a.aspect == 1.0)]
        lm = a[(a.measure == 'landmark') & (a.null == 'tiling')]
        k = lambda x: int(((x.q < ALPHA) & (x.excess > 0)).sum())
        wall = (f'large fields {w.excess.mean():+.3f} further out than small, '
                f'{k(w)}/{len(w)} discs' if len(w) else 'no disc tested')
        return f'{wall}; landmark {k(lm)}/{len(lm)} arenas (tiling null)'

    def figures(self):
        return sorted(FIGURES_WRITTEN)

    def data_files(self):
        return [p for p in (f'{self.out_dir}/summary.csv',
                            f'{self.out_dir}/arena_summary.csv',
                            f'{self.out_dir}/decile_summary.csv')
                if os.path.exists(p)]

    def _measure_section(self, measure, unit_note):
        s, a = self.results, self.arena
        t = s[s[f'{measure}_tested'].astype(bool)]
        L = [unit_note, '']
        if not len(t) or f'{measure}_tiling_q' not in t:
            return L + ['Not tested in any library.']
        for nn in NULLS:
            am = a[(a.measure == measure) & (a.null == nn)].sort_values('env_area_m2')
            if not len(am):
                continue
            L.append(f'{nn} null, channels averaged per arena:')
            L.append(f'  {"arena":16s} {"small":>7s} {"large":>7s} '
                     f'{"large-small":>11s} {"z":>5s} {"q":>7s}  channels>0  '
                     f'decile trend')
            for r in am.itertuples():
                L.append(f'  {r.env:16s} {r.small:+7.3f} {r.large:+7.3f} '
                         f'{r.excess:+11.3f} {r.z:+5.1f} {r.q:7.2g}  '
                         f'{r.n_positive}/{r.n_channels:<8d} '
                         f'rho {r.decile_trend_rho:+.2f}')
            q = t[f'{measure}_{nn}_q']
            ex = t[f'{measure}_{nn}_diff'] - t[f'{measure}_{nn}_diff_null_mean']
            up = int(((q < ALPHA) & (ex > 0)).sum())
            down = int(((q < ALPHA) & (ex < 0)).sum())
            pool = t if measure == 'landmark' else t[t.aspect == 1.0]
            k, n, p = _sign_test(pool[f'{measure}_{nn}_diff'] -
                                 pool[f'{measure}_{nn}_diff_null_mean'])
            L += [f'  per library at q < {ALPHA}: {up}/{len(t)} large further '
                  f'out, {down}/{len(t)} large nearer in',
                  f'  sign test over {"all" if measure == "landmark" else "disc"} '
                  f'libraries: {k}/{n} positive, p {p:.2g} (assumes channels '
                  f'independent, which they are not -- optimistic)', '']
        return L

    def body(self):
        s, a = self.results, self.arena
        if s is None or not len(s):
            return 'No field libraries were analysed.'
        S = self.section
        t = s[s.tested]
        out = []

        out.append(S('THE QUESTION', '\n'.join([
            'Do large place fields sit further from the walls and the '
            'landmarks than fields of their size would by chance? Libraries '
            f'are Experiment 2\'s, unchanged: EXTENT_PCTL {PCTL}, ACT_THRESH '
            f'{THRESH:g}, Rule 2 off, LAMBDA 0.', '',
            'This replaces a correlation of field area with wall distance over '
            'the whole library, which found nothing: pooled rho +0.06 against '
            'a null of +0.05, 0/23 libraries. About 60% of a library is '
            'finest-band fields and they sit at every distance, so a '
            'correlation over every field mostly measures them. The claim is '
            'one-directional -- if a field is large, it sits away from the '
            'wall -- so this version fixes size and measures position.'])))

        if not len(t) or a is None or not len(a):
            out.append(S('NO TESTABLE LIBRARIES',
                         f'No library has {MIN_CLASS} fields in both size '
                         f'classes.'))
            return '\n'.join(out)

        dt = t[t.aspect == 1.0] if (t.aspect == 1.0).any() else t
        out.append(S('HOW IT IS MEASURED', '\n'.join([
            f'SMALL is the bottom 50% of a library by area, LARGE the top 10% '
            f'(median {int(t.n_small.median())} and {int(t.n_large.median())} '
            f'fields per library).', '',
            'A field\'s EXCESS is its position minus the mean position of its '
            'own shape over null placements, so 0 is where it would sit by '
            'chance. The statistic is mean LARGE excess minus mean SMALL '
            'excess. The prediction is SMALL near 0 and LARGE above it, rising '
            'across the deciles (W1). Read the two columns and not only their '
            'difference: small fields sitting nearer the wall than chance '
            'would also make the difference positive, and that is a different '
            'finding.', '',
            'The WALL test uses only fields clear of the wall, each placed only '
            'where it fits whole. A library records a field as an ellipse, so '
            'for a field the wall cut the part beyond the wall is unknown, and '
            'on synthetic libraries guessing it biased the test toward the '
            'hypothesis. Clear of the wall per disc library: median '
            f'{int(dt.n_clear_small.median())} small and '
            f'{int(dt.n_clear_large.median())} large. The LANDMARK test uses '
            'every field in its region, clipped or not; a field the wall cut '
            'is only ever moved along the wall, which leaves its cut, area and '
            'wall distance unchanged.', '',
            'UNIFORM places fields independently. TILING places them largest '
            'first and rejects any placement that breaks Rule 11\'s spacing '
            'within its band, as the real library had to; it is the '
            'like-for-like null, and where the two disagree it is the one to '
            'believe.',
            f'Unplaced null fields: median {100*t.uniform_unplaced_frac.median():.2f}% '
            f'(uniform), {100*t.tiling_unplaced_frac.median():.2f}% (tiling).', '',
            'Per library, q is Benjamini-Hochberg across libraries. Per arena, '
            'the six channels are averaged and tested against the average of '
            'their null standard deviations, as if they were perfectly '
            'correlated, so the arena test can only understate '
            'significance.', '',
            'On synthetic libraries with no effect the wall test fell below '
            'p = 0.05 in 0 of 13 disc libraries and the landmark test in 0 of '
            '21. A planted landmark effect was found in 7 of 8, a planted wall '
            'effect in only 2 of 5: with about 25 large fields clear of the '
            'wall per library the wall test is underpowered, so a wall result '
            'short of significance is weak evidence of no effect.'])))

        out.append(S('WALLS', '\n'.join(self._measure_section(
            'wall', 'Units: fraction of the furthest any point can be from a '
                    'wall -- the radius of a disc, half the width of the '
                    'corridor. Positive = further from the wall than chance.'))))

        out.append(S('LANDMARKS', '\n'.join(self._measure_section(
            'landmark', 'Units: phase along the wall, 0 in front of a panel '
                        'and 1 midway between two. Positive = further from '
                        'the panels than chance. In a disc only fields within '
                        'half a radius of the wall enter (the outer 75% of '
                        'the floor); in the corridor every field does.') + [
            'The correlation version found landmark distance, with wall '
            'distance partialled out, positive beyond its null in 8/23 '
            'libraries. Measured along the wall, landmark position cannot be '
            'a stand-in for wall distance, so this is the direct test of that '
            'lead.'])))

        if (t.aspect != 1.0).any():
            out.append(S('THE CORRIDOR', '\n'.join([
                'Not counted toward the wall question. It is 2 m wide, so a '
                'large field cannot avoid spanning its midline, and its large '
                'fields fill its width so few are clear of the wall at all. '
                'The landmark phase runs along its length and is '
                'unaffected.'])))

        skipped = s[~s.tested]
        wall_skip = s[s.tested & ~s.wall_tested.astype(bool)]
        if len(skipped) or len(wall_skip):
            out.append(S('NOT TESTED', '\n'.join(
                [f'A measure needs {MIN_CLASS} small and {MIN_CLASS} large '
                 f'fields: clear of the wall for the wall test, in its region '
                 f'for the landmark test.'] +
                [f'  {r.env} {r.channel}: {r.n_fields} fields, neither test'
                 for r in skipped.itertuples()] +
                [f'  {r.env} {r.channel}: {int(r.n_clear_large)} large fields '
                 f'clear of the wall, landmark test only'
                 for r in wall_skip.itertuples()] +
                ['', 'Experiment 2 found the colour channel collapses at '
                 'r = 10; that library is expected here.'])))

        cols = ['env', 'channel', 'n_fields', 'n_large', 'n_clear_large',
                'wall_tiling_small', 'wall_tiling_large', 'wall_tiling_diff',
                'wall_tiling_q', 'wall_uniform_diff', 'wall_uniform_q',
                'landmark_tiling_diff', 'landmark_tiling_q']
        out.append(S('Per arena and channel',
                     self.table(t[[c for c in cols if c in t.columns]])))
        return '\n'.join(out)


# ----------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--envs', default=','.join(ENVS),
                   help='default: the four Experiment 2 arenas')
    p.add_argument('--channels', default=','.join(CHANNELS))
    p.add_argument('--n-null', type=int, default=N_NULL,
                   help='null realisations of each library, per null')
    p.add_argument('--seed', type=int, default=0,
                   help='seeds the null draws only; the libraries keep '
                        'Experiment 2\'s seed')
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
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print('=' * 72)
    print('Wall proximity | do large fields sit further from walls and landmarks?')
    print(f'  envs     : {envs}')
    print(f'  channels : {chans}')
    print(f'  fields   : Experiment 2 config, EXTENT_PCTL {PCTL}, '
          f'ACT_THRESH {THRESH:g}, Rule 2 off, LAMBDA 0')
    print(f'  libraries: {"rebuilt" if args.rebuild else "from cache where present"}'
          f' ({BANK_DIR})')
    print(f'  classes  : small = bottom 50% by area, large = top 10%')
    print(f'  nulls    : {", ".join(NULLS)}, {args.n_null} realisations each, '
          f'seed {args.seed}')
    print('=' * 72, flush=True)

    sum_rows, dec_rows, field_frames = [], [], []
    profiles, env_geom, missing = {}, {}, []
    device = None
    for e in envs:
        data_path = f'{DATA_DIR}/{e}.h5'
        if not os.path.exists(data_path):
            print(f'\n[{e}] no dataset at {data_path} -- skipping', flush=True)
            missing.append(e)
            continue
        print(f'\n===== {e} =====', flush=True)
        root = ET.parse(f'{XML_DIR}/{e}.xml').getroot()
        xy = load_positions(data_path)
        env = R.build_env(xy, root)
        env['aspect'] = (1.0 if env.get('is_circular') else
                         (env['x_max'] - env['x_min']) /
                         (env['y_max'] - env['y_min']))
        landmarks = read_landmarks(root)
        env_geom[e] = env
        # The analysis grid the libraries were measured on, so null fields
        # are clipped to exactly the floor the real ones were.
        C = R.resolve_grid_cfg(base_C, xy, env=env, verbose=True)
        G = R._grid_setup(env, C)
        print(f'  area {env["env_area"]:.1f} m^2, {len(landmarks)} landmarks, '
              f'{len(xy)} locations, furthest from a wall '
              f'{max_wall_distance(env):.2f} m', flush=True)

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
            row, drows, fields, prof = analyse_library(
                bank, env, G, landmarks, args.n_null, rng,
                dict(env=e, channel=c))
            sum_rows.append(row)
            dec_rows.extend(drows)
            field_frames.append(fields)
            profiles[(e, c)] = prof
            if row['tested']:
                parts = [f'  [{c}] n={row["n_fields"]} ({row["n_large"]} '
                         f'large, {row["n_clear_large"]} clear of the wall)']
                for m in MEASURES:
                    parts.append(
                        f'{m} large-small {row[f"{m}_tiling_diff"]:+.3f} '
                        f'(null {row[f"{m}_tiling_diff_null_mean"]:+.3f}, '
                        f'p {row[f"{m}_tiling_p"]:.2g})'
                        if row[f'{m}_tested'] else f'{m} not tested')
                print('  '.join(parts) + f'  unplaced '
                      f'{100*row["tiling_unplaced_frac"]:.2f}%', flush=True)
            else:
                print(f'  [{c}] n={row["n_fields"]} -- not tested', flush=True)
        del blocks

    if not sum_rows:
        print('\nNo results. Datasets missing: ' + (', '.join(missing) or 'none'))
        return 1

    summary = pd.DataFrame(sum_rows)
    for measure in MEASURES:
        for nn in NULLS:
            pc = f'{measure}_{nn}_p'
            summary[f'{measure}_{nn}_q'] = (bh_q(summary[pc]) if pc in summary
                                            else np.nan)
    summary.to_csv(f'{OUT_DIR}/summary.csv', index=False)
    deciles = pd.DataFrame(dec_rows)
    if len(deciles):
        deciles.to_csv(f'{OUT_DIR}/decile_summary.csv', index=False)
    arena = arena_table(summary, deciles) if len(deciles) else pd.DataFrame()
    if len(arena):
        arena.to_csv(f'{OUT_DIR}/arena_summary.csv', index=False)
    fields = pd.concat(field_frames, ignore_index=True)
    fields.to_csv(f'{OUT_DIR}/fields.csv', index=False)

    envs_by_area = sorted(env_geom, key=lambda e: env_geom[e]['env_area'])
    print('\nfigures:', flush=True)
    if len(deciles):
        fig_excess_by_size(deciles, envs_by_area, chans)
        fig_distance_by_class(fields, summary, profiles, envs_by_area)
        fig_landmark_phase(fields, summary, profiles, envs_by_area)
    prune_orphan_figures()

    rep = WallProximityReport(env_name=','.join(envs), out_dir=OUT_DIR,
                              fig_dir=FIG_DIR, results=summary,
                              log_path=os.environ.get('REALM_LOG_PATH'))
    rep.arena, rep.deciles = arena, deciles
    if missing:
        print(f'\n!! datasets not found, excluded: {", ".join(missing)}')
    print('\n' + rep.compose(), flush=True)
    if not args.no_email:
        rep.send()
    print(f'\nsummary -> {OUT_DIR}/summary.csv'
          f'\narenas  -> {OUT_DIR}/arena_summary.csv'
          f'\nfigures -> {FIG_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
