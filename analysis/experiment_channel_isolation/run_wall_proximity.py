"""Are larger place fields found further from the walls and landmarks?

Experiment 3. Takes the field libraries Experiment 2 built -- same arenas, same
channels, same rules, same operating point -- and asks whether a field's size
depends on how close it sits to the arena boundary or to the nearest landmark
panel.

The prediction
--------------
Larger fields further from the walls. Near a wall the view changes quickly as
the agent moves (the wall and any panel on it loom and recede), so appearance
should localise tightly there and fields should be small. In open floor the
view changes slowly and the same appearance spans more ground, so fields
should be large. Distance to the wall is never an input to any rule
(rules.py), so any such relationship has to come from the features.

The confound, and why a raw correlation is not the answer
---------------------------------------------------------
A positive size-distance correlation is guaranteed by geometry alone, whatever
the model does. Two mechanisms:

  * CLIPPING. A field centred near a wall loses the part of itself that would
    lie beyond the wall, so near-wall fields are smaller by construction.
  * CENTROID PUSH. A field's centre is the mean of its in-arena bins. A large
    field cannot have its centroid near the wall, because most of its area
    would have to lie outside the arena.

Both scale with field size, so both manufacture exactly the predicted result.
The raw Spearman rho is therefore compared against a GEOMETRIC NULL: every
admitted field, with its own measured shape and orientation, is dropped at a
uniformly random position in the arena, clipped to the analysed floor, and
re-measured -- area from the surviving bins, centre as their mean, distances
from that centre. Draws whose clipped area leaves the Rule 8/9 window are
discarded, as the rules would discard them. The null rho is what clipping and
centroid push produce with no model at all; the claim is supported only by
the EXCESS of the observed rho over it.

The null is approximate in one direction worth knowing. Its templates are the
observed shapes, and the near-wall ones were already clipped once, so the null
fields are slightly smaller than an unclipped population would be. That
understates clipping in the null, which makes the test mildly liberal for a
positive excess.

Controls read beside the pooled test
------------------------------------
  unclipped     Only fields whose ellipse stops short of the analysed floor's
                edge (wall distance >= semi-major + keep-out), in the
                observation and in every null draw alike. Clipping is then
                absent from both sides and only centroid push remains, which
                the null carries exactly. This is the control for the null's
                template bias above, and it matters most in the corridor,
                where almost every field touches a wall: on synthetic
                libraries with no planted effect the pooled test flagged the
                corridor at q = 0.04 and this one did not.
  within band   Experiment 2 found the library is a tiling at every scale, and
                the coarse bands are large by definition. A pooled rho can be
                driven by WHERE whole bands sit rather than by size changing
                with position. Rho is therefore also computed inside each
                scale band and combined (field-weighted mean), against the
                same null.
  landmarks     Panels are mounted flush on the wall, so landmark distance and
                wall distance are strongly collinear. Landmark distance is
                reported raw and as a partial Spearman controlling for wall
                distance. The partial also removes the geometric confound,
                which is a function of wall distance alone.
  multiple      4 arenas x 6 channels = 24 tests per question, so every null
  tests         p-value carries a Benjamini-Hochberg q.

Distance to "the closest wall or landmark" as one number would be wall
distance under another name: a panel sits 0.015 m off the wall, so the
nearest boundary feature is the wall everywhere but on a panel's own face.
The two are reported separately instead.

Usage
    python run_wall_proximity.py [--envs a,b] [--channels ...] [--n-null 200]
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

MIN_FIELDS = SD.MIN_FIELDS     # below this a library is reported, not tested
MIN_BAND_FIELDS = 15           # a band enters the within-band rho at this n
N_NULL = 1000                  # random placements per field
ALPHA = 0.05
N_PROFILE_BINS = 8             # wall-distance bins for the W1 profiles
N_RATIO_BINS = 5               # normalised-distance bins for W3


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

    To the face rather than the centre: a 0.75 m panel is wide against the
    finest fields, and a field centred in front of a panel's edge is not
    0.375 m further from it than one in front of its middle.
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


def max_wall_distance(env):
    """The furthest any point can be from the boundary: the disc's centre, or
    a rectangle's midline."""
    if env['is_circular']:
        return float(env['env_R'])
    return 0.5 * float(min(env['x_max'] - env['x_min'],
                           env['y_max'] - env['y_min']))


# --------------------------------------------------------------- statistics

def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < MIN_FIELDS or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return np.nan, np.nan
    r, p = stats.spearmanr(x[m], y[m])
    return float(r), float(p)


def partial_spearman(x, y, z):
    """Spearman of x and y with z partialled out of both, on ranks."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n = int(m.sum())
    if n < MIN_FIELDS:
        return np.nan, np.nan
    rx, ry, rz = (stats.rankdata(v[m]) for v in (x, y, z))
    Z = np.column_stack([np.ones(n), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(ex, ey)[0, 1])
    df = n - 3
    t = r * np.sqrt(df / max(1.0 - r * r, 1e-12))
    return r, float(2.0 * stats.t.sf(abs(t), df))


def within_band_rho(area, dist, band):
    """Field-weighted mean of the per-band Spearman rho.

    Bands under MIN_BAND_FIELDS are left out: a rho over a handful of coarse
    fields is noise and would carry its full weight in an unweighted mean.
    """
    rs, ns = [], []
    for b in np.unique(band):
        sel = (band == b) & np.isfinite(area) & np.isfinite(dist)
        if sel.sum() < MIN_BAND_FIELDS:
            continue
        r, _ = spearman(area[sel], dist[sel])
        if np.isfinite(r):
            rs.append(r)
            ns.append(sel.sum())
    return float(np.average(rs, weights=ns)) if rs else np.nan


def far_near_ratio(area, dist):
    """Median area in the furthest third of fields over the nearest third."""
    m = np.isfinite(area) & np.isfinite(dist)
    if m.sum() < MIN_FIELDS:
        return np.nan
    a, d = area[m], dist[m]
    lo, hi = np.percentile(d, [100 / 3, 200 / 3])
    near, far = a[d <= lo], a[d >= hi]
    return (float(np.median(far) / np.median(near))
            if len(near) and len(far) else np.nan)


def null_p(obs, null):
    """Two-sided p of the observed value against the null draws, and the
    one-sided p in the predicted (positive) direction.

    From a normal fitted to the draws, not from counting them. An empirical p
    cannot fall below 1 / (n_draws + 1), and Benjamini-Hochberg over 24
    libraries needs 0.05 / 24 = 0.002 for a lone effect to survive, so a
    counted p would make an isolated real effect unpassable by construction.
    A rho over hundreds of fields is close to normal under the null, and the
    empirical tail is still what the grey bars in W2 show.
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


# -------------------------------------------------------------- the null

def geometric_null(bank, env, G, landmarks, area_min, area_max, n_draws, rng):
    """Each field's shape at random positions, clipped and re-measured.

    Measured exactly as rules.field_shape measures a real field -- area from
    the in-arena bin count, centre as the mean of those bins -- so the null
    and the observation differ only in where the field was put.

    Returns four (n_fields, n_draws) arrays: area, wall distance, landmark
    distance and scale band. NaN (band -1) where a draw's clipped area left
    the Rule 8/9 window.

    The band is re-assigned from the CLIPPED area, as admit_fields assigns a
    real field's band from its measured one. Keeping the template's band
    instead lets a coarse field clipped to a sliver stay in the coarse band,
    which floods every band with clipping-driven size variation and pushes
    the within-band null rho far above anything a real library can show --
    measured on synthetic libraries, a null of +0.26 against a true +0.0.
    """
    r_min = np.sqrt(area_min / np.pi)
    in_env, xc, yc = G['in_env'], G['xc'], G['yc']
    gx, gy = in_env.shape
    sx, sy = float(xc[1] - xc[0]), float(yc[1] - yc[0])
    env_i, env_j = np.nonzero(in_env)
    n = len(bank)
    A = np.full((n, n_draws), np.nan)
    DW = np.full((n, n_draws), np.nan)
    DL = np.full((n, n_draws), np.nan)
    B = np.full((n, n_draws), -1, dtype=int)
    for k, r in enumerate(bank.itertuples(index=False)):
        a = max(float(r.semi_major_m), 1e-9)
        b = max(float(r.semi_minor_m), 1e-9)
        th = float(r.orientation_rad)
        ri, rj = int(np.ceil(a / sx)), int(np.ceil(a / sy))
        di, dj = np.meshgrid(np.arange(-ri, ri + 1), np.arange(-rj, rj + 1),
                             indexing='ij')
        ox, oy = di * sx, dj * sy
        u = ox * np.cos(th) + oy * np.sin(th)
        v = -ox * np.sin(th) + oy * np.cos(th)
        inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
        inside[ri, rj] = True                      # never an empty template
        di, dj = di[inside], dj[inside]

        pick = rng.integers(0, len(env_i), size=n_draws)
        I = env_i[pick, None] + di[None, :]
        J = env_j[pick, None] + dj[None, :]
        Ic, Jc = np.clip(I, 0, gx - 1), np.clip(J, 0, gy - 1)
        ok = (I == Ic) & (J == Jc) & in_env[Ic, Jc]
        cnt = ok.sum(axis=1)                       # >= 1: the centre is in
        cx = (xc[Ic] * ok).sum(axis=1) / cnt
        cy = (yc[Jc] * ok).sum(axis=1) / cnt
        area = cnt * G['bin_area']
        keep = (area >= area_min) & (area <= area_max)
        A[k, keep] = area[keep]
        DW[k, keep] = R.wall_distance(cx[keep], cy[keep], env)
        DL[k, keep] = landmark_distance(cx[keep], cy[keep], landmarks)
        B[k, keep] = R.assign_bands(np.sqrt(area[keep] / np.pi), r_min,
                                    BASE_C['BAND_RATIO'])
    return A, DW, DL, B


# ----------------------------------------------------------------- analysis

def analyse_library(bank, env, G, landmarks, area_min, area_max, margin,
                    n_null, rng, tag):
    """Every statistic for one arena x channel library, with its null.

    `margin` is the collection keep-out the floor was shrunk by, so a field
    counts as unclipped only if it stops short of the floor actually analysed.
    """
    area = bank.area_env_m2.to_numpy(dtype=float)
    dw = bank.dist_to_wall_m.to_numpy(dtype=float)
    reach = bank.semi_major_m.to_numpy(dtype=float) + margin
    dl = landmark_distance(bank.centroid_x.to_numpy(dtype=float),
                           bank.centroid_y.to_numpy(dtype=float), landmarks)
    band = bank.scale_band.to_numpy(dtype=int)
    wmax = max_wall_distance(env)

    fields = bank[['area_env_m2', 'radius_env_m', 'semi_major_m',
                   'semi_minor_m', 'centroid_x', 'centroid_y',
                   'dist_to_wall_m', 'scale_band']].copy()
    fields['dist_to_landmark_m'] = dl
    fields['wall_dist_norm'] = dw / wmax
    fields['coverage'] = area / float(env['env_area'])
    for k, v in tag.items():
        fields.insert(0, k, v)

    row = dict(tag, env_area_m2=float(env['env_area']),
               aspect=float(env.get('aspect', 1.0)),
               max_wall_dist_m=wmax, n_fields=len(bank),
               n_bands=int(len(np.unique(band))) if len(band) else 0)
    if len(bank) < MIN_FIELDS:
        row['tested'] = False
        return row, [], fields, None
    row['tested'] = True

    A, DW, DL, B = geometric_null(bank, env, G, landmarks, area_min, area_max,
                                  n_null, rng)
    row['null_frac_discarded'] = float(np.mean(~np.isfinite(A)))

    # Each statistic, observed and over every null draw.
    interior = dw >= reach
    row['n_unclipped'] = int(interior.sum())
    obs = dict(
        wall=spearman(area, dw)[0],
        wall_unclipped=spearman(area[interior], dw[interior])[0],
        lm=spearman(area, dl)[0],
        lm_given_wall=partial_spearman(area, dl, dw)[0],
        wall_given_lm=partial_spearman(area, dw, dl)[0],
        wall_band=within_band_rho(area, dw, band),
        far_near=far_near_ratio(area, dw),
    )
    nul = {k: np.full(n_null, np.nan) for k in obs}
    for j in range(n_null):
        a, w, l = A[:, j], DW[:, j], DL[:, j]
        nul['wall'][j] = spearman(a, w)[0]
        # Same criterion as the observation; an unclipped null field's centre
        # is exactly where it was placed.
        m = w >= reach
        nul['wall_unclipped'][j] = spearman(a[m], w[m])[0]
        nul['lm'][j] = spearman(a, l)[0]
        nul['lm_given_wall'][j] = partial_spearman(a, l, w)[0]
        nul['wall_given_lm'][j] = partial_spearman(a, w, l)[0]
        nul['wall_band'][j] = within_band_rho(a, w, B[:, j])
        nul['far_near'][j] = far_near_ratio(a, w)

    row['rho_wall'], row['p_wall'] = spearman(area, dw)
    row['rho_lm'], row['p_lm'] = spearman(area, dl)
    row['prho_lm_given_wall'], row['p_prho_lm_given_wall'] = \
        partial_spearman(area, dl, dw)
    row['prho_wall_given_lm'], row['p_prho_wall_given_lm'] = \
        partial_spearman(area, dw, dl)
    # How collinear the two distances are in this arena: near 1 and the
    # landmark numbers are the wall numbers again.
    row['rho_dist_wall_lm'] = spearman(dw, dl)[0]
    for k, v in obs.items():
        name = {'wall': 'rho_wall', 'wall_unclipped': 'rho_wall_unclipped',
                'lm': 'rho_lm',
                'lm_given_wall': 'prho_lm_given_wall',
                'wall_given_lm': 'prho_wall_given_lm',
                'wall_band': 'rho_wall_within_band',
                'far_near': 'far_near_ratio'}[k]
        nv = nul[k][np.isfinite(nul[k])]
        row[name] = v
        row[f'{name}_null_mean'] = float(nv.mean()) if len(nv) else np.nan
        row[f'{name}_null_lo'] = (float(np.percentile(nv, 2.5))
                                  if len(nv) else np.nan)
        row[f'{name}_null_hi'] = (float(np.percentile(nv, 97.5))
                                  if len(nv) else np.nan)
        row[f'{name}_excess'] = (v - row[f'{name}_null_mean']
                                 if np.isfinite(v) else np.nan)
        row[f'{name}_p_null'], row[f'{name}_p_null_gt'] = null_p(v, nv)

    # Per band, for the band table and W3's reading.
    band_rows = []
    for b in np.unique(band):
        sel = band == b
        if sel.sum() < MIN_BAND_FIELDS:
            continue
        r, p = spearman(area[sel], dw[sel])
        nr = np.array([spearman(A[B[:, j] == b, j], DW[B[:, j] == b, j])[0]
                       for j in range(n_null)])
        nr = nr[np.isfinite(nr)]
        pn, _ = null_p(r, nr)
        band_rows.append(dict(
            tag, scale_band=int(b), n_fields=int(sel.sum()),
            area_median_m2=float(np.median(area[sel])),
            wall_dist_median_m=float(np.median(dw[sel])),
            rho_wall=r, p_wall=p,
            null_mean=float(nr.mean()) if len(nr) else np.nan,
            excess=r - float(nr.mean()) if len(nr) and np.isfinite(r) else np.nan,
            p_null=pn))

    # Profiles for the figures: observed and null median area by distance.
    edges = np.linspace(0.0, wmax, N_PROFILE_BINS + 1)
    redges = np.linspace(0.0, 1.0, N_RATIO_BINS + 1)
    fa, fw = A[np.isfinite(A)], DW[np.isfinite(A)]
    prof = dict(edges=edges, obs=_binned(dw, area, edges),
                null_med=_binned(fw, fa, edges),
                null_q25=_binned(fw, fa, edges, 25),
                null_q75=_binned(fw, fa, edges, 75),
                redges=redges,
                ratio=_binned(dw / wmax, area, redges, min_n=5) /
                      _binned(fw / wmax, fa, redges, min_n=5))
    return row, band_rows, fields, prof


def _binned(x, y, edges, q=50, min_n=3):
    out = np.full(len(edges) - 1, np.nan)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    for i in range(len(out)):
        s = y[idx == i]
        if len(s) >= min_n:
            out[i] = np.percentile(s, q)
    return out


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


def fig_scatter(fields, summary, profiles, envs, chans, env_geom):
    """W1: field area against wall distance, with the geometric null behind.

    The grey band is the null's interquartile range and the dashed line its
    median: what clipping and centroid push alone give at each distance. The
    black line is the observed median. The prediction is the black line
    rising ABOVE the grey one away from the wall, not merely rising.
    """
    fig, axes = plt.subplots(len(envs), len(chans), squeeze=False,
                             figsize=(2.9 * len(chans), 2.5 * len(envs)))
    for i, e in enumerate(envs):
        fe = fields[fields.env == e]
        lo = fe.area_env_m2.min() if len(fe) else 1e-3
        hi = fe.area_env_m2.max() if len(fe) else 1.0
        wmax = max_wall_distance(env_geom[e])
        area = env_geom[e]['env_area']
        for j, c in enumerate(chans):
            ax = axes[i][j]
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(c, fontsize=10)
            if j == 0:
                ax.set_ylabel(f'{e}\n{area:.0f} m$^2$\n\nfield area (m$^2$)',
                              fontsize=8)
            if i == len(envs) - 1:
                ax.set_xlabel('distance to wall (m)', fontsize=8)
            f = fe[fe.channel == c]
            if len(f) < MIN_FIELDS:
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, f'{len(f)} field' + ('' if len(f) == 1 else 's')
                        + '\ntoo few to test', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='#b03030')
                continue
            ax.scatter(f.dist_to_wall_m, f.area_env_m2, s=3, alpha=0.35,
                       color=CHANNEL_COLORS.get(c, '0.4'), lw=0)
            pr = profiles.get((e, c))
            if pr is not None:
                xm = 0.5 * (pr['edges'][:-1] + pr['edges'][1:])
                ax.fill_between(xm, pr['null_q25'], pr['null_q75'],
                                color='0.6', alpha=0.25, lw=0)
                ax.plot(xm, pr['null_med'], '--', color='0.45', lw=1.0)
                ax.plot(xm, pr['obs'], 'o-', color='k', ms=2.5, lw=1.1)
            ax.set_yscale('log')
            ax.set_ylim(lo * 0.8, hi * 1.25)
            ax.set_xlim(0, wmax)
            s = summary[(summary.env == e) & (summary.channel == c)]
            if len(s) and s.tested.iloc[0]:
                s = s.iloc[0]
                ax.text(0.03, 0.97,
                        f'rho {s.rho_wall:+.2f}  null {s.rho_wall_null_mean:+.2f}'
                        f'\nq {s.rho_wall_q:.2g}', transform=ax.transAxes,
                        va='top', fontsize=6, color='0.2')
    fig.suptitle('W1  field area against distance to the wall\n'
                 'black = observed median; dashed and grey = the geometric '
                 'null (same shapes, random positions, clipped to the arena)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, 'W1_size_vs_wall.png')


def fig_forest(summary, envs, chans):
    """W2: observed rho against its null interval, one row per library.

    Filled markers clear the null at BH q < 0.05; open ones do not. A marker
    right of its grey bar is a field size that grows away from the boundary
    by more than geometry accounts for.
    """
    s = summary[summary.tested].copy()
    if not len(s):
        return
    s['_e'] = s.env.map({e: i for i, e in enumerate(envs)})
    s['_c'] = s.channel.map({c: i for i, c in enumerate(chans)})
    s = s.sort_values(['_e', '_c']).reset_index(drop=True)
    panels = [('rho_wall', 'area ~ wall distance'),
              ('rho_wall_unclipped', 'area ~ wall distance, unclipped fields'),
              ('rho_wall_within_band', 'area ~ wall distance, within band'),
              ('prho_lm_given_wall', 'area ~ landmark distance | wall')]
    fig, axes = plt.subplots(1, len(panels), sharey=True,
                             figsize=(4.2 * len(panels), 0.26 * len(s) + 1.6))
    y = np.arange(len(s))[::-1]
    for ax, (col, title) in zip(axes, panels):
        ax.axvline(0, color='0.8', lw=0.8)
        for yi, (_, r) in zip(y, s.iterrows()):
            ax.plot([r[f'{col}_null_lo'], r[f'{col}_null_hi']], [yi, yi],
                    color='0.75', lw=3, solid_capstyle='butt')
            ax.plot(r[f'{col}_null_mean'], yi, '|', color='0.4', ms=7)
            sig = r[f'{col}_q'] < ALPHA if np.isfinite(r[f'{col}_q']) else False
            colr = CHANNEL_COLORS.get(r.channel, '0.3')
            ax.plot(r[col], yi, 'o', ms=5, color=colr,
                    mfc=colr if sig else 'white', mew=1.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Spearman rho')
        ax.tick_params(labelsize=7)
        # Separators between arenas.
        for k in np.flatnonzero(np.diff(s._e.to_numpy()))[::1]:
            ax.axhline(y[k] - 0.5, color='0.85', lw=0.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f'{r.env}  {r.channel}' for _, r in s.iterrows()],
                            fontsize=7)
    fig.suptitle('W2  observed rho (dot) against the geometric null (grey: 95% '
                 'of draws, tick: mean)\nfilled = beyond the null at BH q < 0.05',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, 'W2_rho_vs_null.png')


def fig_excess_profile(profiles, envs, chans):
    """W3: observed over null median area, across the floor.

    1.0 everywhere is geometry alone. The prediction is a line below 1 at the
    wall rising above 1 toward the centre (disc) or midline (corridor). The x
    axis is normalised so the corridor's 1 m and the r = 10 disc's 10 m read
    on one scale.
    """
    envs = [e for e in envs if any((e, c) in profiles for c in chans)]
    if not envs:
        return
    fig, axes = plt.subplots(1, len(envs), squeeze=False, sharey=True,
                             figsize=(3.6 * len(envs), 3.2))
    for ax, e in zip(axes[0], envs):
        ax.axhline(1.0, color='0.6', lw=0.8, ls='--')
        for c in chans:
            pr = profiles.get((e, c))
            if pr is None:
                continue
            xm = 0.5 * (pr['redges'][:-1] + pr['redges'][1:])
            ax.plot(xm, pr['ratio'], 'o-', ms=3, lw=1.2,
                    color=CHANNEL_COLORS.get(c, '0.4'), label=c)
        ax.set_title(e, fontsize=9)
        ax.set_xlabel('wall distance / furthest possible\n0 = wall, '
                      '1 = centre or midline', fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel('median area, observed / null')
    axes[0][0].legend(fontsize=6, frameon=False)
    fig.suptitle('W3  field size beyond geometry, across the floor', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, 'W3_excess_profile.png')


# -------------------------------------------------------------------- report

class WallProximityReport(ExperimentReport):
    experiment = 'wall-proximity'

    def title(self):
        s = self.results
        t = s[s.tested] if s is not None and len(s) else None
        if t is None or not len(t):
            return 'no testable libraries'
        up = int(((t.rho_wall_q < ALPHA) & (t.rho_wall_excess > 0)).sum())
        return (f'rho {t.rho_wall.median():+.2f} vs null '
                f'{t.rho_wall_null_mean.median():+.2f}, '
                f'{up}/{len(t)} larger-away beyond geometry')

    def figures(self):
        return sorted(FIGURES_WRITTEN)

    def data_files(self):
        return [p for p in (f'{self.out_dir}/summary.csv',
                            f'{self.out_dir}/band_summary.csv')
                if os.path.exists(p)]

    @staticmethod
    def _count(t, col):
        sig = t[f'{col}_q'] < ALPHA
        return (int((sig & (t[f'{col}_excess'] > 0)).sum()),
                int((sig & (t[f'{col}_excess'] < 0)).sum()), len(t))

    def body(self):
        s = self.results
        if s is None or not len(s):
            return 'No field libraries were analysed.'
        S = self.section
        t = s[s.tested]
        out = []

        out.append(S('THE QUESTION', '\n'.join([
            'Are larger place fields found further from the walls and the '
            'landmarks? Libraries are Experiment 2\'s, unchanged: '
            f'EXTENT_PCTL {PCTL}, ACT_THRESH {THRESH:g}, Rule 2 off, LAMBDA 0.',
            '',
            'A raw positive correlation is guaranteed by geometry: a field '
            'near a wall is clipped by it, and a large field cannot have its '
            'centre near one. Every rho is therefore read against a geometric '
            'null -- each field\'s own shape dropped at random positions, '
            f'clipped and re-measured, {self.n_null} draws per field -- and '
            'only the EXCESS over that null speaks to the model. Null p-values '
            'carry a Benjamini-Hochberg q over the libraries in this run.'])))

        if not len(t):
            out.append(S('NO TESTABLE LIBRARIES',
                         f'Every library has fewer than {MIN_FIELDS} fields.'))
            return '\n'.join(out)

        def block(col, label):
            up, down, n = self._count(t, col)
            L = [f'{label}',
                 f'  observed rho   median {t[col].median():+.3f} '
                 f'(range {t[col].min():+.2f} to {t[col].max():+.2f})',
                 f'  geometric null median {t[f"{col}_null_mean"].median():+.3f}',
                 f'  excess         median {t[f"{col}_excess"].median():+.3f}',
                 f'  beyond the null at q < {ALPHA}: {up}/{n} larger away, '
                 f'{down}/{n} SMALLER away']
            for e, g in t.groupby('env', sort=False):
                u, d, k = self._count(g, col)
                L.append(f'    {e:16s} excess median '
                         f'{g[f"{col}_excess"].median():+.3f}   '
                         f'{u}/{k} up, {d}/{k} down')
            return L

        L = block('rho_wall', 'Field area against distance to the wall, '
                  'pooled over the library:')
        L += ['',
              f'Far/near ratio (median area of the furthest third of fields '
              f'over the nearest third): observed median '
              f'{t.far_near_ratio.median():.2f}x, null median '
              f'{t.far_near_ratio_null_mean.median():.2f}x. The null value is '
              f'how much of that ratio geometry produces on its own.', '']
        L += block('rho_wall_unclipped', 'The same, over fields that stop '
                   'short of the wall (median '
                   f'{int(t.n_unclipped.median())} per library):')
        L += ['',
              'The null\'s templates are the observed shapes, and a near-wall '
              'field was already clipped once, so the pooled null understates '
              'clipping and leans toward a false positive excess -- most in '
              'the corridor, where nearly every field touches a wall. Among '
              'unclipped fields there is no clipping on either side. Where '
              'the two disagree, believe this one.']
        out.append(S('WALL DISTANCE', '\n'.join(L)))

        L = block('rho_wall_within_band', 'The same, inside each scale band '
                  f'(bands with >= {MIN_BAND_FIELDS} fields, field-weighted):')
        L += ['',
              'Read this beside the pooled result. The coarse bands are large '
              'by definition, so if they simply sit further from the wall the '
              'pooled rho rises without any field changing size with its '
              'position. A pooled excess that vanishes here is band placement, '
              'not a size gradient. Per-band numbers are in band_summary.csv.']
        out.append(S('WITHIN A SCALE BAND', '\n'.join(L)))

        L = block('rho_lm', 'Field area against distance to the nearest '
                  'landmark face:')
        L += [''] + block('prho_lm_given_wall',
                          'Landmark distance with wall distance partialled out:')
        L += ['',
              f'The two distances are collinear: median Spearman between them '
              f'{t.rho_dist_wall_lm.median():+.2f}. Panels are flush on the '
              f'wall, so the raw landmark rho is mostly the wall rho again. '
              f'The partial is the landmark\'s own contribution, and since the '
              f'geometric confound is a function of wall distance it is also '
              f'largely free of it -- but only linearly, in ranks. A strong, '
              f'curved wall effect leaks through: on synthetic libraries with '
              f'a wall effect and no landmark effect, one of three arenas '
              f'still flagged the partial at q = 0.03. Treat a partial '
              f'result as the landmark\'s only where the wall result beside '
              f'it is weak. The corridor\'s panels are on its long walls '
              f'only, none on the end walls.']
        out.append(S('LANDMARKS', '\n'.join(L)))

        cor = t[t.aspect != 1.0]
        if len(cor):
            out.append(S('THE CORRIDOR', '\n'.join([
                f'No point in the corridor is more than '
                f'{cor.max_wall_dist_m.iloc[0]:.1f} m from a wall, and the '
                f'collection keep-out takes 0.2 m of that, so its wall-distance '
                f'axis is a tenth of the r = 10 disc\'s. Experiment 2 found it '
                f'had already lost its coarse bands to that width. A weak rho '
                f'here is a short axis as much as an absent effect; W3 puts '
                f'every arena on one normalised axis for that reason.'])))

        skipped = s[~s.tested]
        if len(skipped):
            out.append(S('NOT TESTED', '\n'.join(
                [f'Fewer than {MIN_FIELDS} fields, reported but not tested:'] +
                [f'  {r.env} {r.channel}: {r.n_fields} fields'
                 for r in skipped.itertuples()] +
                ['', 'Experiment 2 found the colour channel collapses at '
                 'r = 10; that library is expected here.'])))

        cols = ['env', 'channel', 'n_fields', 'rho_wall', 'rho_wall_null_mean',
                'rho_wall_q', 'rho_wall_unclipped', 'rho_wall_unclipped_q',
                'rho_wall_within_band',
                'rho_wall_within_band_q', 'prho_lm_given_wall',
                'prho_lm_given_wall_q', 'far_near_ratio']
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
                   help='random placements per field for the geometric null')
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
    print('Wall proximity | are larger fields further from walls and landmarks?')
    print(f'  envs     : {envs}')
    print(f'  channels : {chans}')
    print(f'  fields   : Experiment 2 config, EXTENT_PCTL {PCTL}, '
          f'ACT_THRESH {THRESH:g}, Rule 2 off, LAMBDA 0')
    print(f'  libraries: {"rebuilt" if args.rebuild else "from cache where present"}'
          f' ({BANK_DIR})')
    print(f'  null     : {args.n_null} random placements per field, seed {args.seed}')
    print('=' * 72, flush=True)

    sum_rows, band_rows, field_frames = [], [], []
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
        area_min = C['RULE8_AREA_FRAC'] * env['env_area']
        area_max = C['RULE9_AREA_FRAC'] * env['env_area']
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
            tag = dict(env=e, channel=c)
            row, brows, fields, prof = analyse_library(
                bank, env, G, landmarks, area_min, area_max,
                C['IN_ENV_MARGIN_M'], args.n_null, rng, tag)
            sum_rows.append(row)
            band_rows.extend(brows)
            field_frames.append(fields)
            if prof is not None:
                profiles[(e, c)] = prof
            if row['tested']:
                print(f'  [{c}] n={row["n_fields"]}  rho_wall '
                      f'{row["rho_wall"]:+.3f} (null '
                      f'{row["rho_wall_null_mean"]:+.3f})  unclipped '
                      f'{row["rho_wall_unclipped"]:+.3f} (null '
                      f'{row["rho_wall_unclipped_null_mean"]:+.3f})  within-band '
                      f'{row["rho_wall_within_band"]:+.3f}  lm|wall '
                      f'{row["prho_lm_given_wall"]:+.3f}', flush=True)
            else:
                print(f'  [{c}] n={row["n_fields"]} -- too few to test',
                      flush=True)
        del blocks

    if not sum_rows:
        print('\nNo results. Datasets missing: ' + (', '.join(missing) or 'none'))
        return 1

    summary = pd.DataFrame(sum_rows)
    # BH across libraries, separately for each question.
    for col in ('rho_wall', 'rho_wall_unclipped', 'rho_wall_within_band',
                'rho_lm',
                'prho_lm_given_wall', 'prho_wall_given_lm', 'far_near_ratio'):
        pc = f'{col}_p_null'
        summary[f'{col}_q'] = bh_q(summary[pc]) if pc in summary else np.nan
    summary.to_csv(f'{OUT_DIR}/summary.csv', index=False)
    bands = pd.DataFrame(band_rows)
    if len(bands):
        bands['q_null'] = bh_q(bands.p_null)
        bands.to_csv(f'{OUT_DIR}/band_summary.csv', index=False)
    fields = pd.concat(field_frames, ignore_index=True)
    fields.to_csv(f'{OUT_DIR}/fields.csv', index=False)

    envs_by_area = sorted(env_geom, key=lambda e: env_geom[e]['env_area'])
    print('\nfigures:', flush=True)
    fig_scatter(fields, summary, profiles, envs_by_area, chans, env_geom)
    fig_forest(summary, envs_by_area, chans)
    fig_excess_profile(profiles, envs_by_area, chans)
    prune_orphan_figures()

    rep = WallProximityReport(env_name=','.join(envs), out_dir=OUT_DIR,
                              fig_dir=FIG_DIR, results=summary,
                              log_path=os.environ.get('REALM_LOG_PATH'))
    rep.n_null = args.n_null
    if missing:
        print(f'\n!! datasets not found, excluded: {", ".join(missing)}')
    print('\n' + rep.compose(), flush=True)
    if not args.no_email:
        rep.send()
    print(f'\nsummary -> {OUT_DIR}/summary.csv'
          f'\nfields  -> {OUT_DIR}/fields.csv'
          f'\nfigures -> {FIG_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
