"""Does the extent rule recover a place field, and does it reject a non-field?

The pipeline turns a group of positions into a field by placing an RBF at the
group's centroid and thresholding at a fraction of its own peak. The width of
that RBF is the 90th percentile of within-group pairwise distance. Handed a
*known* field — a disc of floor — that rule returns 7 to 10 m for a 1 m truth,
in every channel. The mask sits at

    d <= sqrt(d_min^2 + 2 sigma^2 ln(1/T))

and d_min, the closest any position gets to the centroid, is large in high
dimensions because distances concentrate. Every member is at least d_min from
the centroid, so no percentile of any within-group distance is small enough to
matter beside it. The statistic cannot reach the value it needs.

SIGMA_MODE='quantile' solves for the sigma that puts the cut at a chosen
quantile of the group's own centroid distances, which makes the cut land where
it was asked to. This experiment asks whether that is actually better.

Three tests
-----------
**Test 1 - recovery.** Construct an ideal place cell — a disc of floor of
radius r — and compare the field the rule returns against it. Swept over
radius, wall distance, channel, rule, and EXTENT_PCTL.

**Test 2 - discrimination.** The same machinery on inputs that are *not* place
fields: two half-size discs a gap apart, a ring, a scattered group, positions
drawn at random, and a disc above the size ceiling. A rule that returns a tidy
compact field from any of these is not recovering structure, it is imposing
it. Test 1 means nothing without this.

**Test 3 - pipeline.** The real tree and the real rules under both settings,
to see whether the product improves and whether the key findings survive.

Judged on: recovery IoU and the recovered-vs-true radius transfer; the margin
between test 1 and test 2; split-half reliability, which presupposes no field
size and is the one non-circular metric here; and whether corr(size, wall) and
elongation hold. A rule that improves test 1 while also making fields out of
shuffled positions is worse, not better.

Usage
    python run_field_recovery.py [env_name] [--channels ...] [--tests 1,2,3]
                                 [--pctls 20,35,50,65,80,90] [--no-gpu]
                                 [--no-email] [--subsample N]
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.ticker import ScalarFormatter, NullFormatter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch
import rules as R
from realm_tools.experiment_lib.reporting import ExperimentReport

CHANNEL_COLORS = {'hog': '#1f77b4', 'color': '#d62728', 'spatial': '#2ca02c',
                  'lidar': '#9467bd', 'visual': '#ff7f0e', 'all': '#17becf'}
MODE_STYLE = {'pairwise': dict(color='#888888', ls='--', marker='s'),
              'quantile': dict(color='#d62728', ls='-', marker='o')}

IDEAL_RADII = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
WALL_RINGS = [0.5, 1.0, 2.0, 4.0, 8.0]          # distance from the wall
N_ANGLES = 4                                     # ideal place cells per ring


# ----------------------------------------------------------------- planting

def plant_sites(env, wall_rings, n_angles, rng):
    """Centers for the ideal place cells, on rings at fixed distance from the wall."""
    R_arena = np.sqrt(env['env_area'] / np.pi)
    sites = []
    for w in wall_rings:
        rad = R_arena - w
        if rad < 0:
            continue
        if rad < 1e-6:                            # the center is a single site
            sites.append((0.0, 0.0, w))
            continue
        for k in range(n_angles):
            th = 2 * np.pi * (k + 0.5 * (w in wall_rings[1::2])) / n_angles
            sites.append((rad * np.cos(th), rad * np.sin(th), w))
    return sites


def members_disc(xy, cx, cy, r):
    return np.flatnonzero(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) <= r)


def members_control(kind, xy, env, cx, cy, r, rng):
    """Groups that are not place fields. Each should be rejected.

    Every control except `oversized` is deliberately *size-matched* to a real
    field of radius r. An earlier version used widely separated discs and a
    full-arena ring; those were rejected on size alone, so contiguity was
    never tested and the test proved nothing. A control is only informative if
    the size rule cannot dispose of it, leaving spatial structure as the one
    thing that separates it from a genuine field.
    """
    R_arena = np.sqrt(env['env_area'] / np.pi)
    n = len(members_disc(xy, cx, cy, r))
    if kind == 'split':
        # Two half-size discs a short gap apart. Same area as one field of
        # radius r; only contiguity can tell it from the real thing.
        rr = r / np.sqrt(2.0)
        gap = 3.0 * r
        ux, uy = (-cx, -cy) / max(np.hypot(cx, cy), 1e-9)
        a = members_disc(xy, cx + ux * gap / 2, cy + uy * gap / 2, rr)
        b = members_disc(xy, cx - ux * gap / 2, cy - uy * gap / 2, rr)
        return np.unique(np.concatenate([a, b]))
    if kind == 'ring':
        # An annulus of the same area, centered on the same place. Spatially
        # coherent but not a disc; the shape should not be filled in.
        d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
        return np.flatnonzero((d >= r) & (d <= r * np.sqrt(2.0)))
    if kind == 'scattered':
        # Same member count, drawn at random from a neighbourhood several
        # times the field's own size: no fine spatial structure, but not
        # spread so widely that the size rule alone disposes of it.
        near = members_disc(xy, cx, cy, min(4.0 * r, R_arena))
        return rng.choice(near, size=min(n, len(near)), replace=False)
    if kind == 'shuffled':
        # No spatial structure at all — the easy case, kept as a floor.
        return rng.choice(len(xy), size=min(n, len(xy)), replace=False)
    if kind == 'oversized':
        # Past the Rule 9 ceiling. The size rule should, and must, catch this.
        return members_disc(xy, 0.0, 0.0, min(6.0, 0.9 * R_arena))
    raise ValueError(kind)


# ------------------------------------------------------------ the extent rule

def sigma_for(D2sub, mode, C):
    """Width for one group, from its within-group squared-distance block."""
    iu = np.triu_indices(len(D2sub), k=1)
    if mode == 'pairwise':
        return float(np.sqrt(np.percentile(D2sub[iu], C['SIGMA_PCTL'])))
    dc2 = np.maximum(D2sub.mean(axis=1) - 0.5 * D2sub.mean(), 0.0)
    q2 = float(np.percentile(dc2, C['EXTENT_PCTL']))
    d_min2 = float(dc2.min())
    if q2 <= d_min2:
        return 0.0
    return float(np.sqrt((q2 - d_min2) / (2.0 * np.log(1.0 / C['ACT_THRESH']))))


def field_from_members(X, xy, mem, G, occupied, C, mode, rng, chunk=4096):
    """Run the pipeline's own extent machinery on an arbitrary group."""
    mu = X[mem].mean(axis=0)
    s = mem if len(mem) <= C['SIGMA_MAX_MEMBERS'] else \
        rng.choice(mem, C['SIGMA_MAX_MEMBERS'], replace=False)
    P = X[s].astype(np.float32)
    D2sub = np.maximum(
        (P ** 2).sum(1)[:, None] - 2.0 * (P @ P.T) + (P ** 2).sum(1)[None, :], 0.0)
    sigma = sigma_for(D2sub, mode, C)

    d2 = np.empty(len(X), np.float64)
    for a in range(0, len(X), chunk):
        d2[a:a + chunk] = ((X[a:a + chunk] - mu) ** 2).sum(1)
    resp = np.exp(-d2 / (2.0 * sigma ** 2)) if sigma > 0 else (d2 <= 1e-12).astype(float)

    flat = R._bin_indices(xy, G)
    grid = np.zeros(G['gx'] * G['gy'], np.float32)
    np.maximum.at(grid, flat, resp.astype(np.float32))
    mask, _ = R.mask_from_grid(grid, G, occupied, C)
    return mask, sigma


def ideal_mask(xy, mem, G, occupied, C):
    """The ideal place cell rendered on the same grid, for a fair comparison."""
    flat = R._bin_indices(xy, G)
    grid = np.zeros(G['gx'] * G['gy'], np.float32)
    np.maximum.at(grid, flat[mem], np.ones(len(mem), np.float32))
    m = grid.reshape(G['gx'], G['gy']) > 0
    return m & G['in_env']


def score(mask, imask, G, C, env, area_min, area_max):
    """Everything measured about one recovered field."""
    sh = R.field_shape(mask, G)
    ts = R.field_shape(imask, G)
    cc, ncomp = R.largest_component_fraction(mask)
    inter = float((mask & imask).sum())
    union = float((mask | imask).sum())
    bin_area = C['BIN_M'] ** 2
    rec_area, ideal_area = sh['area'], ts['area']
    return dict(
        rec_area_m2=rec_area, ideal_area_m2=ideal_area,
        rec_r_eq_m=sh['r_eq'], ideal_r_eq_m=ts['r_eq'],
        rec_elongation=sh['elongation'], rec_cx=sh['cx'], rec_cy=sh['cy'],
        iou=inter / union if union else 0.0,
        center_err_m=float(np.hypot(sh['cx'] - ts['cx'], sh['cy'] - ts['cy'])),
        log2_area_ratio=float(np.log2(max(rec_area, bin_area) /
                                      max(ideal_area, bin_area))),
        cc_frac=cc, n_components=int(ncomp),
        pass_size=bool(area_min <= rec_area <= area_max),
        pass_contiguity=bool(cc >= C['CC_FRAC_MIN']),
    )


# ----------------------------------------------------------------- the tests

def run_recovery(X, xy, env, G, occupied, C, cname, modes, pctls, rng,
                 verbose=True, collect=None, collect_pctl=None):
    area_min = C['RULE8_AREA_FRAC'] * env['env_area']
    area_max = C['RULE9_AREA_FRAC'] * env['env_area']
    sites = plant_sites(env, WALL_RINGS, N_ANGLES, rng)
    rows = []
    for mode in modes:
        for pctl in (pctls if mode == 'quantile' else [np.nan]):
            cfg = dict(C)
            if mode == 'quantile':
                cfg['EXTENT_PCTL'] = pctl
            for (cx, cy, w) in sites:
                for r in IDEAL_RADII:
                    mem = members_disc(xy, cx, cy, r)
                    if len(mem) < 12:
                        continue
                    mask, sigma = field_from_members(X, xy, mem, G, occupied,
                                                     cfg, mode, rng)
                    im = ideal_mask(xy, mem, G, occupied, cfg)
                    if collect is not None and r == 1.0 and (
                            mode == 'pairwise' or pctl == collect_pctl):
                        sh = R.field_shape(mask, G)
                        collect.setdefault(mode, {'truth': [], 'fields': []})
                        collect[mode]['truth'].append((cx, cy, r))
                        collect[mode]['fields'].append(sh)
                    rows.append(dict(test='recovery', channel=cname, mode=mode,
                                     extent_pctl=pctl, kind='disc',
                                     ideal_r_m=r, wall_dist_m=w, cx=cx, cy=cy,
                                     n_members=len(mem), sigma=sigma,
                                     **score(mask, im, G, cfg, env,
                                             area_min, area_max)))
            if verbose:
                sub = [x for x in rows if x['mode'] == mode
                       and (np.isnan(pctl) or x['extent_pctl'] == pctl)]
                med = np.median([x['iou'] for x in sub]) if sub else float('nan')
                lab = mode if mode == 'pairwise' else f'{mode} p{pctl:g}'
                print(f'    [{cname}] recovery {lab:16s} median IoU {med:.3f}',
                      flush=True)
    return rows


def run_controls(X, xy, env, G, occupied, C, cname, modes, pctls, rng, verbose=True):
    area_min = C['RULE8_AREA_FRAC'] * env['env_area']
    area_max = C['RULE9_AREA_FRAC'] * env['env_area']
    R_arena = np.sqrt(env['env_area'] / np.pi)
    rows = []
    for mode in modes:
        for pctl in (pctls if mode == 'quantile' else [np.nan]):
            cfg = dict(C)
            if mode == 'quantile':
                cfg['EXTENT_PCTL'] = pctl
            for kind in ['split', 'ring', 'scattered', 'shuffled', 'oversized']:
                for r in [1.0, 2.0]:
                    for (cx, cy) in [(R_arena - 1.0, 0.0), (R_arena - 4.0, 0.0)]:
                        mem = members_control(kind, xy, env, cx, cy, r, rng)
                        if len(mem) < 12:
                            continue
                        mask, sigma = field_from_members(X, xy, mem, G, occupied,
                                                         cfg, mode, rng)
                        im = ideal_mask(xy, mem, G, occupied, cfg)
                        rows.append(dict(test='control', channel=cname, mode=mode,
                                         extent_pctl=pctl, kind=kind,
                                         ideal_r_m=r, wall_dist_m=np.nan,
                                         cx=cx, cy=cy, n_members=len(mem),
                                         sigma=sigma,
                                         **score(mask, im, G, cfg, env,
                                                 area_min, area_max)))
                        if kind == 'oversized':
                            break
                    if kind == 'oversized':
                        break
            if verbose:
                sub = [x for x in rows if x['mode'] == mode
                       and (np.isnan(pctl) or x['extent_pctl'] == pctl)]
                adm = np.mean([x['pass_size'] and x['pass_contiguity']
                               for x in sub]) if sub else float('nan')
                lab = mode if mode == 'pairwise' else f'{mode} p{pctl:g}'
                print(f'    [{cname}] controls {lab:16s} wrongly admitted '
                      f'{100*adm:.0f}%', flush=True)
    return rows


def run_pipeline(X, xy, env, D2, feat_med, xy_med, C, cname, modes, pctls,
                 device, out_dir, verbose=True):
    """Test 3: the real tree and the real rules. One tree serves every setting."""
    rows = []
    tree = R.build_tree(D2, xy, feat_med, xy_med, cfg=C, verbose=verbose)
    for mode in modes:
        for pctl in (pctls if mode == 'quantile' else [np.nan]):
            cfg = dict(C, SIGMA_MODE=mode)
            if mode == 'quantile':
                cfg['EXTENT_PCTL'] = pctl
            tag = f'{cname}/{mode}' + (f'_p{pctl:g}' if mode == 'quantile' else '')
            ctx = R.prepare_candidates(X, xy, env, D2, feat_med, xy_med, cfg=cfg,
                                       device=device, tag=tag, verbose=verbose,
                                       tree=tree)
            bank, kept_mu, rep = R.admit_fields(ctx, cfg=cfg, verbose=verbose)
            d = f'{out_dir}/pipeline/{cname}/{mode}' + \
                (f'_p{pctl:g}' if mode == 'quantile' else '')
            os.makedirs(d, exist_ok=True)
            bank.to_csv(f'{d}/bank.csv', index=False)
            with open(f'{d}/report.json', 'w') as f:
                json.dump({k: v for k, v in rep.items()
                           if not isinstance(v, np.ndarray)}, f, indent=2,
                          default=float)
            row = dict(test='pipeline', channel=cname, mode=mode,
                       extent_pctl=pctl, n_fields=len(bank),
                       band_lo=rep.get('band_lo'), band_hi=rep.get('band_hi'),
                       split_half_iou=rep.get('median_split_half_iou'),
                       frac_cand_above_cap=rep.get('frac_cand_above_cap'))
            if len(bank):
                row.update(
                    median_area_m2=float(bank.area_env_m2.median()),
                    min_radius_m=float(bank.radius_env_m.min()),
                    median_radius_m=float(bank.radius_env_m.median()),
                    median_elongation=float(bank.elongation.median()),
                    corr_size_wall=float(np.corrcoef(bank.dist_to_wall_m,
                                                     bank.area_env_m2)[0, 1])
                    if len(bank) > 3 else np.nan,
                    corr_elong_wall=float(np.corrcoef(bank.dist_to_wall_m,
                                                      bank.elongation)[0, 1])
                    if len(bank) > 3 else np.nan)
            rows.append(row)
            if verbose:
                print(f'    [{tag}] {len(bank)} fields  '
                      f'split-half {row["split_half_iou"]}', flush=True)
    return rows


# ------------------------------------------------------------------ figures

def _arena(ax, env):
    R_a = np.sqrt(env['env_area'] / np.pi)
    ax.add_patch(Circle((0, 0), R_a, fill=False, ec='0.4', lw=1.0))
    ax.set_xlim(-R_a * 1.05, R_a * 1.05); ax.set_ylim(-R_a * 1.05, R_a * 1.05)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


def fig_transfer(rec, fig_dir, env, best_pctl):
    """R1 — recovered radius against true radius. The identity line is the aim."""
    chans = list(dict.fromkeys(rec.channel))
    n = len(chans)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    R_a = np.sqrt(env['env_area'] / np.pi)
    for ax, cn in zip(axes, chans):
        g = rec[rec.channel == cn]
        lo, hi = 0.3, max(IDEAL_RADII) * 1.15
        ax.plot([lo, hi], [lo, hi], color='0.3', lw=1.0, zorder=1,
                label='exact recovery')
        ax.axhline(R_a, color='0.75', lw=0.8, ls=':', zorder=0)
        ax.text(0.98, 0.985, 'arena radius', fontsize=6.5, color='0.55',
                ha='right', va='top', transform=ax.transAxes)
        for mode, st in MODE_STYLE.items():
            sub = g[g['mode'] == mode]
            if mode == 'quantile':
                sub = sub[sub.extent_pctl == best_pctl]
            if not len(sub):
                continue
            m = sub.groupby('ideal_r_m').rec_r_eq_m.median()
            q1 = sub.groupby('ideal_r_m').rec_r_eq_m.quantile(0.25)
            q3 = sub.groupby('ideal_r_m').rec_r_eq_m.quantile(0.75)
            ax.fill_between(m.index, q1, q3, color=st['color'], alpha=0.15, lw=0)
            ax.plot(m.index, m.values, **st, ms=4,
                    label=mode if mode == 'pairwise' else f'quantile p{best_pctl:g}')
        ax.set_xscale('log'); ax.set_yscale('log')
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_formatter(ScalarFormatter())
            a.set_minor_formatter(NullFormatter())
        ax.set_xticks([0.5, 1, 2, 3]); ax.set_yticks([0.5, 1, 2, 5, 10])
        ax.set_xlim(0.38, 4.0); ax.set_ylim(0.38, 13.0)
        ax.set_title(cn, fontsize=10, color=CHANNEL_COLORS.get(cn, 'k'))
        ax.set_xlabel('ideal field radius (m)')
    axes[0].set_ylabel('recovered field radius (m)')
    axes[0].legend(fontsize=7, loc='lower right', frameon=False)
    fig.suptitle('R1  Give the model a place field of known size; measure what it '
                 'hands back.\nThe diagonal is a perfect answer. Grey saturates at '
                 'the arena whatever it is given.', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/R1_radius_transfer.png', dpi=140,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_ideal_maps(X_maps, fig_dir, env, best_pctl):
    """R2 — what each rule returns for a grid of ideal place cells."""
    if not X_maps:
        return
    modes = list(X_maps.keys())
    fig, axes = plt.subplots(1, len(modes) + 1,
                             figsize=(3.1 * (len(modes) + 1), 3.4))
    _arena(axes[0], env)
    for (cx, cy, r) in X_maps[modes[0]]['truth']:
        axes[0].add_patch(Circle((cx, cy), r, fc='0.75', ec='0.4', lw=0.6, alpha=0.8))
    axes[0].set_title('ideal place cells', fontsize=10)
    for ax, mode in zip(axes[1:], modes):
        _arena(ax, env)
        for sh in X_maps[mode]['fields']:
            if not np.isfinite(sh['a']) or sh['a'] <= 0:
                continue
            ax.add_patch(Ellipse((sh['cx'], sh['cy']), 2 * sh['a'], 2 * sh['b'],
                                 angle=np.degrees(sh['theta']), fc='none',
                                 ec=MODE_STYLE[mode]['color'], lw=0.9, alpha=0.85))
        lab = mode if mode == 'pairwise' else f'quantile p{best_pctl:g}'
        ax.set_title(f'recovered — {lab}', fontsize=10)
    fig.suptitle('R2  Ideal place cells across the arena, '
                 'and what each rule returns', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/R2_ideal_maps.png', dpi=140,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig_iou(rec, ctl, fig_dir, best_pctl):
    """R3 — recovery quality, with the control band behind it."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, key, xlab in ((axes[0], 'ideal_r_m', 'ideal field radius (m)'),
                          (axes[1], 'wall_dist_m', 'distance to wall (m)')):
        for mode, st in MODE_STYLE.items():
            sub = rec[rec['mode'] == mode]
            if mode == 'quantile':
                sub = sub[sub.extent_pctl == best_pctl]
            if not len(sub):
                continue
            m = sub.groupby(key).iou.median()
            ax.plot(m.index, m.values, **st, ms=4,
                    label=mode if mode == 'pairwise' else f'quantile p{best_pctl:g}')
            c = ctl[ctl['mode'] == mode]
            if mode == 'quantile':
                c = c[c.extent_pctl == best_pctl]
            if len(c):
                ax.axhspan(0, float(c.iou.quantile(0.9)), color=st['color'],
                           alpha=0.08, lw=0)
        ax.set_xlabel(xlab); ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel('IoU with ideal place cell')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('R3  Recovery of the ideal place cell. Shaded bands are the '
                 '90th percentile of the non-field controls.', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/R3_iou.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def fig_pctl_sweep(rec, ctl, pipe, fig_dir, act_thresh):
    """R4 — calibrating EXTENT_PCTL. 100(1-T) is the value the Gaussian
    mass identity predicts: for a 2D Gaussian rate map the mass inside the
    T-of-peak contour is exactly 1-T."""
    q = rec[rec['mode'] == 'quantile']
    if not len(q):
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    pred = 100.0 * (1.0 - act_thresh)

    m = q.groupby('extent_pctl').iou.median()
    axes[0].plot(m.index, m.values, '-o', color='#d62728', ms=4, label='ideal place cells')
    cq = ctl[ctl['mode'] == 'quantile']
    if len(cq):
        mc = cq.groupby('extent_pctl').iou.median()
        axes[0].plot(mc.index, mc.values, '-s', color='0.6', ms=4, label='controls')
    axes[0].set_ylabel('median IoU'); axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title('recovery and discrimination', fontsize=10)

    margin = m - (mc.reindex(m.index) if len(cq) else 0.0)
    axes[1].plot(margin.index, margin.values, '-o', color='#1f77b4', ms=4)
    axes[1].set_ylabel('IoU margin (field − control)')
    axes[1].set_title('discrimination margin', fontsize=10)

    if pipe is not None and len(pipe):
        pq = pipe[pipe['mode'] == 'quantile'].dropna(subset=['split_half_iou'])
        if len(pq):
            for cn, g in pq.groupby('channel'):
                g = g.sort_values('extent_pctl')
                axes[2].plot(g.extent_pctl, g.split_half_iou, '-o', ms=3,
                             color=CHANNEL_COLORS.get(cn, 'k'), label=cn)
            axes[2].legend(fontsize=7, frameon=False, ncol=2)
    axes[2].set_ylabel('split-half IoU')
    axes[2].set_title('reliability (presupposes no field size)', fontsize=10)

    for ax in axes:
        ax.axvline(pred, color='0.3', ls='--', lw=1.0)
        ax.annotate(f'100(1−T) = {pred:g}', xy=(pred, ax.get_ylim()[1]),
                    xytext=(3, -10), textcoords='offset points',
                    fontsize=7, color='0.3', rotation=90, va='top')
        ax.set_xlabel('EXTENT_PCTL'); ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle('R4  Calibrating EXTENT_PCTL', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/R4_pctl_sweep.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def fig_controls(rec, ctl, fig_dir, best_pctl):
    """R5 — admission rates for real fields and for non-fields, together.

    Separately, neither number means anything: a rule that rejects every
    input scores a perfect 0% on the controls. Only the pair does.
    """
    if not len(ctl):
        return
    kinds = ['split', 'ring', 'scattered', 'shuffled', 'oversized']
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5),
                             gridspec_kw=dict(width_ratios=[1, 2.1]))

    def admitted(df):
        return (df.pass_size & df.pass_contiguity)

    w = 0.38
    for i, (mode, st) in enumerate(MODE_STYLE.items()):
        sr = rec[rec['mode'] == mode]
        sc = ctl[ctl['mode'] == mode]
        if mode == 'quantile':
            sr = sr[sr.extent_pctl == best_pctl]
            sc = sc[sc.extent_pctl == best_pctl]
        lab = mode if mode == 'pairwise' else f'quantile p{best_pctl:g}'
        tpr = 100.0 * admitted(sr).mean() if len(sr) else 0.0
        fpr = 100.0 * admitted(sc).mean() if len(sc) else 0.0
        axes[0].bar([0 + (i - 0.5) * w, 1 + (i - 0.5) * w], [tpr, fpr],
                    width=w, color=st['color'], alpha=0.85, label=lab)
        vals = [100.0 * admitted(sc[sc.kind == k]).mean()
                if len(sc[sc.kind == k]) else 0.0 for k in kinds]
        axes[1].bar(np.arange(len(kinds)) + (i - 0.5) * w, vals, width=w,
                    color=st['color'], alpha=0.85, label=lab)

    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['ideal place cells\nadmitted (want high)',
                             'non-fields\nadmitted (want low)'], fontsize=8)
    axes[0].set_ylabel('% admitted'); axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title('the pair is the result', fontsize=10)
    axes[1].set_xticks(range(len(kinds)))
    axes[1].set_xticklabels(kinds, rotation=20, ha='right')
    axes[1].set_ylabel('% wrongly admitted'); axes[1].set_ylim(0, 105)
    axes[1].set_title('by control type (all but `oversized` are size-matched)',
                      fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5, axis='y')
    fig.suptitle('R5  Discrimination: a rule that rejects everything scores '
                 '0% on the right-hand panel', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/R5_controls.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def fig_ladder(out_dir, fig_dir, best_pctl, channels):
    """R6 — what the field library looks like under each rule.

    The case for the new width rule in the two terms that matter for a place
    cell population: how big the fields are, and how many there are at each
    scale. Kjelstrup found a full population at every dorsoventral level and
    Jung found the coarse levels thinner than the fine ones, so a healthy
    ladder spans a wide range of sizes and thins as it climbs.
    """
    import glob
    def bank(cn, st):
        f = f'{out_dir}/pipeline/{cn}/{st}/bank.csv'
        return pd.read_csv(f) if os.path.exists(f) else None

    qdir = f'quantile_p{best_pctl:g}'
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7))

    # (a) size distribution, pooled
    for lab, st, col in (('current rule', 'pairwise', '#888888'),
                         (f'new rule (p{best_pctl:g})', qdir, '#d62728')):
        r = np.concatenate([b.radius_env_m.values for cn in channels
                            if (b := bank(cn, st)) is not None and len(b)] or [np.array([])])
        if len(r):
            # as a fraction of each library, since the two differ ~10x in count
            # and the comparison here is of shape and range, not of size
            axes[0].hist(r, bins=np.logspace(np.log10(0.3), np.log10(5), 26),
                         weights=np.full(len(r), 1.0 / len(r)), color=col,
                         alpha=0.55, label=f'{lab}  (n={len(r)})')
    axes[0].set_xscale('log')
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xticks([0.5, 1, 2, 4])
    axes[0].set_xlabel('field radius (m)')
    axes[0].set_ylabel('fraction of library')
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[0].set_title('(a) field size distribution, all channels pooled',
                      fontsize=9.5)

    # (b) the scale ladder for one representative channel
    cn = 'color' if 'color' in channels else channels[0]
    w = 0.38
    for i, (lab, st, col) in enumerate((('current rule', 'pairwise', '#888888'),
                                        (f'new rule (p{best_pctl:g})', qdir, '#d62728'))):
        b = bank(cn, st)
        if b is None or not len(b):
            continue
        vc = b.scale_band.value_counts().sort_index()
        axes[1].bar(vc.index + (i - 0.5) * w, vc.values, width=w, color=col,
                    alpha=0.85, label=lab)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('scale band  (coarser to the right)')
    axes[1].set_ylabel('fields in band')
    axes[1].legend(fontsize=7.5, frameon=False)
    axes[1].set_title(f'(b) the scale ladder — {cn}', fontsize=9.5)

    # (c) how many fields each channel yields
    xs = np.arange(len(channels))
    for i, (lab, st, col) in enumerate((('current rule', 'pairwise', '#888888'),
                                        (f'new rule (p{best_pctl:g})', qdir, '#d62728'))):
        vals = [len(b) if (b := bank(cn_, st)) is not None else 0 for cn_ in channels]
        axes[2].bar(xs + (i - 0.5) * w, vals, width=w, color=col, alpha=0.85, label=lab)
    axes[2].set_xticks(xs); axes[2].set_xticklabels(channels, rotation=30, ha='right')
    axes[2].set_ylabel('fields in library')
    axes[2].set_title('(c) library size by channel', fontsize=9.5)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5, axis='y')
    fig.suptitle('R6  The field library under each rule: smaller, more numerous '
                 'fields spanning a wider range of scales', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/R6_library.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def fig_pipeline(pipe, fig_dir, best_pctl):
    """R6 — do the published results survive the rule change?"""
    if pipe is None or not len(pipe):
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    chans = list(dict.fromkeys(pipe.channel))
    xs = np.arange(len(chans)); w = 0.38
    panels = [('n_fields', 'fields', axes[0]),
              ('corr_size_wall', 'corr(size, wall distance)', axes[1]),
              ('median_elongation', 'median elongation', axes[2])]
    for col, lab, ax in panels:
        for i, (mode, st) in enumerate(MODE_STYLE.items()):
            sub = pipe[pipe['mode'] == mode]
            if mode == 'quantile':
                sub = sub[sub.extent_pctl == best_pctl]
            vals = [float(sub[sub.channel == c][col].iloc[0])
                    if len(sub[sub.channel == c]) and
                    pd.notna(sub[sub.channel == c][col].iloc[0]) else np.nan
                    for c in chans]
            ax.bar(xs + (i - 0.5) * w, vals, width=w, color=st['color'],
                   alpha=0.85,
                   label=mode if mode == 'pairwise' else f'quantile p{best_pctl:g}')
        ax.set_xticks(xs); ax.set_xticklabels(chans, rotation=30, ha='right')
        ax.set_ylabel(lab); ax.grid(alpha=0.25, lw=0.5, axis='y')
        if col == 'corr_size_wall':
            ax.axhline(0, color='0.3', lw=0.8)
        if col == 'n_fields':
            ax.set_yscale('symlog')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('R6  The published results under each rule', fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/R6_pipeline.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


# ------------------------------------------------------------------- report

class FieldRecoveryReport(ExperimentReport):
    """Emailed summary for the field-recovery experiment."""

    experiment = 'field-recovery'

    def title(self):
        if self.metrics is None or not len(self.metrics):
            return 'no results'
        rec = self.metrics[self.metrics.test == 'recovery']
        if not len(rec):
            return f'{len(self.metrics)} rows'
        best = self.best_pctl
        a = rec[rec['mode'] == 'pairwise'].iou.median()
        b = rec[(rec['mode'] == 'quantile') &
                (rec.extent_pctl == best)].iou.median()
        return (f'recovery IoU {a:.2f} -> {b:.2f} at p{best:g}'
                if np.isfinite(a) and np.isfinite(b) else f'{len(rec)} ideal place cells')

    def body(self):
        m, out = self.metrics, []
        rec = m[m.test == 'recovery']
        ctl = m[m.test == 'control']
        pipe = m[m.test == 'pipeline']
        best = self.best_pctl

        out.append(self.section(
            'Question',
            "A group becomes a field by placing an RBF at its centroid and\n"
            "cutting at a fraction of its own peak. The width of that RBF is\n"
            "the 90th percentile of within-group pairwise distance. Handed a\n"
            "known field -- a disc of floor -- that rule returns 7-10 m for a\n"
            "1 m truth, in every channel, because the cut sits at\n"
            "sqrt(d_min^2 + 2 sigma^2 ln(1/T)) and d_min is large when\n"
            "distances concentrate. No percentile of a within-group distance\n"
            "can be small beside it.\n\n"
            "SIGMA_MODE='quantile' solves for the sigma that puts the cut at a\n"
            "chosen quantile of the group's own centroid distances. Is that\n"
            "better -- and does it still reject things that are not fields?"))

        if len(rec):
            g = (rec.assign(setting=np.where(rec['mode'] == 'pairwise', 'pairwise',
                                             'quantile p' + rec.extent_pctl.astype(str)))
                 .groupby(['channel', 'setting'])
                 .agg(median_iou=('iou', 'median'),
                      median_log2_ratio=('log2_area_ratio', 'median'),
                      median_center_err=('center_err_m', 'median'))
                 .reset_index())
            out.append(self.section('Test 1 - recovery of an ideal place cell', self.table(g)))

        if len(ctl):
            c = (ctl.assign(setting=np.where(ctl['mode'] == 'pairwise', 'pairwise',
                                             'quantile p' + ctl.extent_pctl.astype(str)))
                 .assign(admitted=lambda d: d.pass_size & d.pass_contiguity)
                 .groupby(['setting', 'kind'])
                 .agg(pct_admitted=('admitted', lambda s: 100.0 * s.mean()),
                      median_iou=('iou', 'median'))
                 .reset_index())
            out.append(self.section(
                'Test 2 - non-fields that were wrongly admitted',
                'Each of these should be rejected. A rule that admits them is\n'
                'imposing structure, not recovering it.\n\n' + self.table(c)))

        if len(ctl):
            sp = ctl[(ctl.kind == 'split') & (ctl['mode'] == 'quantile') &
                     (ctl.extent_pctl == best)]
            if len(sp) and sp.pass_contiguity.mean() > 0.5:
                out.append(self.section(
                    'Rule 1 does not catch a split field',
                    f'The `split` control is two half-size discs a short gap\n'
                    f'apart -- the case contiguity exists to reject. It passes\n'
                    f'contiguity {100*sp.pass_contiguity.mean():.0f}% of the time,\n'
                    f'as a single connected component. The centroid of a\n'
                    f'two-lobed group sits between the lobes in feature space,\n'
                    f'so positions in the gap can be closer to it than the\n'
                    f'members are, and the mask fills in. This is a property of\n'
                    f'summarising a group by its mean, not of the width rule,\n'
                    f'and it holds under both settings.'))

        if len(pipe):
            cols = [x for x in ('channel', 'mode', 'extent_pctl', 'n_fields',
                                'median_radius_m', 'median_elongation',
                                'corr_size_wall', 'split_half_iou',
                                'band_lo', 'band_hi') if x in pipe.columns]
            out.append(self.section('Test 3 - the real pipeline', self.table(pipe, cols)))

        out.append(self.section(
            'Calibration',
            f'For a 2D Gaussian rate map the mass inside the T-of-peak contour\n'
            f'is exactly 1-T, so the value of EXTENT_PCTL matching the locked\n'
            f'ACT_THRESH is 100(1-T) = {100*(1-self.act_thresh):g}. The sweep\n'
            f'selected p{best:g} on the discrimination margin.'))
        return ''.join(out)

    def figures(self):
        import glob
        return sorted(glob.glob(f'{self.fig_dir}/R*.png'))

    def data_files(self):
        return [os.path.join(self.out_dir, 'metrics.csv')]


# --------------------------------------------------------------------- main


def select_pctl(rec, ctl, convex, tol=0.05):
    """Choose EXTENT_PCTL: best discrimination, ties broken on reconstruction.

    Youden's J -- ideal place cells admitted minus non-fields admitted -- is
    the primary criterion, but it saturates: several settings sit within noise
    of the best. Taking a bare argmax lets a 0.02 difference in J outweigh a
    large difference in how accurately the field is reconstructed. So among
    settings whose J is within `tol` of the maximum, take the one with the
    highest recovery IoU.
    """
    adm = lambda d: (d.pass_size & d.pass_contiguity)
    rq = rec[rec['mode'] == 'quantile']
    cq = ctl[(ctl['mode'] == 'quantile') & ctl.kind.isin(convex)]
    if not len(rq):
        return None, None
    tpr = adm(rq).groupby(rq.extent_pctl).mean()
    fpr = (adm(cq).groupby(cq.extent_pctl).mean().reindex(tpr.index).fillna(0.0)
           if len(cq) else 0.0)
    J = tpr - fpr
    iou = rq.groupby('extent_pctl').iou.median()
    near = J[J >= J.max() - tol].index
    best = float(iou.reindex(near).idxmax())
    return best, pd.DataFrame({'admitted': tpr, 'false_pos': fpr, 'J': J,
                               'iou': iou.reindex(J.index)})


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('env', nargs='?', default='circ_lm8_r0')
    ap.add_argument('--channels', default='hog,color,spatial,lidar,visual,all')
    ap.add_argument('--tests', default='1,2,3')
    ap.add_argument('--pctls', default='20,35,50,65,80,90')
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--subsample', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-gpu', action='store_true')
    ap.add_argument('--no-email', action='store_true')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--figures-only', action='store_true',
                    help='replot from a saved metrics.csv; runs no tests')
    return ap.parse_args()


def main():
    args = parse_args()
    env_name = args.env
    channel_names = [c.strip() for c in args.channels.split(',') if c.strip()]
    tests = {int(a) for a in args.tests.split(',') if a.strip()}
    pctls = [float(p) for p in args.pctls.split(',') if p.strip()]
    rng = np.random.default_rng(args.seed)

    data_path = f'{REPO}/data/vpce/collect_data/{env_name}.h5'
    xml_path = f'{REPO}/simulation/worlds/environments/vpce/{env_name}.xml'
    out_dir = f'{REPO}/data_cache/field_recovery/{env_name}'
    fig_dir = f'{HERE}/figures/field_recovery/{env_name}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    if args.figures_only:
        metrics = pd.read_csv(f'{out_dir}/metrics.csv').rename(columns={
            'arm': 'test', 'true_r_m': 'ideal_r_m', 'true_area_m2': 'ideal_area_m2',
            'true_r_eq_m': 'ideal_r_eq_m', 'centre_err_m': 'center_err_m'})
        cfg = json.load(open(f'{out_dir}/config.json'))
        rec = metrics[metrics.test == 'recovery']
        ctl = metrics[metrics.test == 'control']
        pipe = metrics[metrics.test == 'pipeline']
        CONVEX = ('scattered', 'shuffled', 'oversized')
        best_pctl, table = select_pctl(rec, ctl, CONVEX)
        print(table.round(3).to_string())
        act = cfg.get('act_thresh', 0.5)
        print(f'replotting from {out_dir}/metrics.csv  '
              f'({len(metrics)} rows), EXTENT_PCTL = {best_pctl:g}')
        fig_transfer(rec, fig_dir, R.build_env(
            np.zeros((2, 2)), None) if False else dict(
            env_area=float(cfg.get('env_area', 314.159))), best_pctl)
        fig_iou(rec, ctl, fig_dir, best_pctl)
        fig_controls(rec, ctl, fig_dir, best_pctl)
        fig_pctl_sweep(rec, ctl, pipe, fig_dir, act)
        fig_ladder(out_dir, fig_dir, best_pctl,
                   [c for c in channel_names
                    if os.path.exists(f'{out_dir}/pipeline/{c}')])
        print(f'Figures -> {fig_dir}')
        return

    print('=' * 72)
    print(f'Field recovery | env={env_name}')
    print(f'  channels : {channel_names}')
    print(f'  tests    : {sorted(tests)}   EXTENT_PCTL: {pctls}')
    print('=' * 72, flush=True)

    blocks, xy = ch.load_channel_blocks(data_path)
    if args.subsample and args.subsample < len(xy):
        keep = np.sort(rng.choice(len(xy), args.subsample, replace=False))
        xy = xy[keep]
        blocks = {k: v[keep] for k, v in blocks.items()}
        print(f'subsampled to {len(xy)} positions')
    root = ET.parse(xml_path).getroot()
    env = R.build_env(xy, root)
    device = R.pick_device(use_gpu=not args.no_gpu)

    C = R.resolve_cfg(dict(LAMBDA=args.lam, RANDOM_SEED=args.seed,
                           USE_GPU=not args.no_gpu))
    G = R._grid_setup(env, C)
    occupied = np.zeros(G['gx'] * G['gy'], dtype=bool)
    occupied[R._bin_indices(xy, G)] = True
    occupied = occupied.reshape(G['gx'], G['gy'])
    modes = ['pairwise', 'quantile']

    rows, ideal_maps = [], {}
    d2xy = ((xy[:3000, None, :] - xy[None, :3000, :]) ** 2).sum(-1)
    xy_med = float(np.median(d2xy[np.triu_indices(len(d2xy), 1)]))
    del d2xy

    for cname in channel_names:
        print(f'\n===== {cname} =====', flush=True)
        t0 = time.time()
        X = ch.assemble(blocks, ch.CHANNEL_SETS[cname], normalize=True)
        if 1 in tests:
            grab = ideal_maps if cname == channel_names[0] else None
            rows += run_recovery(X, xy, env, G, occupied, C, cname, modes,
                                 pctls, rng, collect=grab,
                                 collect_pctl=100.0 * (1.0 - C['ACT_THRESH']))
        if 2 in tests:
            rows += run_controls(X, xy, env, G, occupied, C, cname, modes,
                                 pctls, rng)
        if 3 in tests:
            D2 = R.feature_sq_distances(X, device=device)
            feat_med = R._median_offdiag(D2, rng)
            rows += run_pipeline(X, xy, env, D2, feat_med, xy_med, C, cname,
                                 modes, pctls, device, out_dir)
            del D2
        del X
        print(f'  {cname} done in {time.time()-t0:.0f}s', flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(f'{out_dir}/metrics.csv', index=False)
    print(f'\nMetrics -> {out_dir}/metrics.csv  ({len(metrics)} rows)')

    # EXTENT_PCTL is selected on the admission trade-off, not on recovery
    # alone. Recovery IoU on its own prefers a tighter cut, but a cut that
    # recovers ideal place cells beautifully while also admitting size-matched
    # non-fields has bought nothing. Youden's J -- the rate at which real
    # fields are admitted minus the rate at which non-fields are -- is the
    # standard way to score exactly that trade-off, and it is what decides.
    rec = metrics[metrics.test == 'recovery'] if len(metrics) else metrics
    ctl = metrics[metrics.test == 'control'] if len(metrics) else metrics

    # `split` and `ring` are non-convex. A cluster summarised by one centroid
    # cannot represent either -- the centroid of a two-lobed cluster sits
    # between the lobes, so the field fills the gap -- and no choice of
    # EXTENT_PCTL repairs that: they leak at every setting, 48% even at p90.
    # They measure the single-centroid representation, not the width rule, so
    # scoring the width rule on them would pick a value for the wrong reason.
    # Selection therefore uses the convex controls, which the width rule is
    # genuinely responsible for.
    CONVEX_CONTROLS = ('scattered', 'shuffled', 'oversized')

    def admit_rate(df):
        return (df.pass_size & df.pass_contiguity).groupby(df.extent_pctl).mean()

    best_pctl, jj = pctls[0], None
    if len(rec) and (rec['mode'] == 'quantile').any():
        best_pctl, _tbl = select_pctl(rec, ctl, CONVEX_CONTROLS)
        tpr = admit_rate(rec[rec['mode'] == 'quantile'])
        cq = ctl[(ctl['mode'] == 'quantile') & ctl.kind.isin(CONVEX_CONTROLS)]
        fpr = admit_rate(cq) if len(cq) else 0.0
        jj = tpr - (fpr.reindex(tpr.index).fillna(0.0)
                    if hasattr(fpr, 'reindex') else fpr)
        iou_pick = float(rec[rec['mode'] == 'quantile']
                         .groupby('extent_pctl').iou.median().idxmax())
        print(f'\n  EXTENT_PCTL   admitted: fields / non-fields    J')
        for pv in tpr.index:
            f_ = float(fpr[pv]) if hasattr(fpr, '__getitem__') else 0.0
            print(f'  {pv:11g}   {100*tpr[pv]:8.0f}% / {100*f_:9.0f}%   {jj[pv]:+6.2f}')
        print(f'\nselected EXTENT_PCTL = {best_pctl:g} on Youden J '
              f'(recovery IoU alone would pick {iou_pick:g}; '
              f'the 1-T identity predicts {100*(1-C["ACT_THRESH"]):g})')

    with open(f'{out_dir}/config.json', 'w') as f:
        json.dump(dict(env=env_name, channels=channel_names, tests=sorted(tests),
                       pctls=pctls, lam=args.lam, seed=args.seed,
                       act_thresh=C['ACT_THRESH'], sigma_pctl=C['SIGMA_PCTL'],
                       ideal_radii=IDEAL_RADII, wall_rings=WALL_RINGS,
                       best_pctl=best_pctl), f, indent=2)

    if not args.no_plots and len(metrics):
        pipe = metrics[metrics.test == 'pipeline']
        if len(rec):
            fig_transfer(rec, fig_dir, env, best_pctl)
            fig_iou(rec, ctl, fig_dir, best_pctl)
        if len(ctl):
            fig_controls(rec, ctl, fig_dir, best_pctl)
        fig_pctl_sweep(rec, ctl, pipe, fig_dir, C['ACT_THRESH'])
        fig_pipeline(pipe, fig_dir, best_pctl)
        fig_ladder(out_dir, fig_dir, best_pctl,
                   [c for c in channel_names
                    if os.path.exists(f'{out_dir}/pipeline/{c}')])
        if ideal_maps:
            fig_ideal_maps(ideal_maps, fig_dir, env, best_pctl)
        print(f'Figures -> {fig_dir}')

    if not args.no_email:
        rep = FieldRecoveryReport(env_name=env_name, fig_dir=fig_dir,
                                  out_dir=out_dir, metrics=metrics)
        rep.best_pctl = best_pctl
        rep.act_thresh = C['ACT_THRESH']
        rep.send()


if __name__ == '__main__':
    main()
