"""Which channel reconstructs a known place field, and does that survive geometry?

The recovery test (see run_field_recovery.py, and the 2026-08-26 report)
establishes that a channel can return a place field it was handed. It has only
ever been run in the circular arena. This experiment asks the same question in
three environments that hold **area** at ~314 m^2 and **landmark count** at 8,
so that shape is the only thing that varies:

    circ_lm8_r0   disc, r = 10         aspect  1.00
    rect_lm8_r0   20 x 15.708          aspect  1.27
    corr_lm8_r0   56 x 5.6             aspect 10.00

Ideal place cells vary in **size** (six radii, 0.5-3.0 m) and in **location**
(contours at fixed distance from the nearest wall, eight sites spread along
each). Only the new sigma definition is run -- SIGMA_MODE='quantile' at
EXTENT_PCTL=65, the setting selected in the 2026-08-21 report. The old
'pairwise' rule is not re-litigated here; it returns the whole arena.

What comes out
--------------
`metrics.csv`  one row per (env, channel, site, radius) trial, carrying IoU,
               centre error, area ratio, admission, sigma and the recovered
               shape. This is the file to plot from.
`masks.npz`    the reconstructed and ideal masks for every trial, so field
               maps can be drawn later without re-running anything.
`config.json`  settings and per-environment geometry.

Two geometry fixes this depends on, both in rules.py: `build_env` now takes a
non-circular arena's extent from its boundary walls rather than from the
sampled positions (they differ by the 0.2 m collection margin, which
understated area by 5-8% and put every wall distance 0.2 m short), and
`plant_sites_by_wall` places ideal cells by true wall distance instead of on
rings of a nominal arena radius. Ring planting is circle-only: in a corridor it
puts cells outside the floor, and the old code then skipped them silently.

Usage
    python run_geometry_recovery.py [--envs a,b,c] [--channels ...]
                                    [--pctl 65] [--sites 8] [--subsample N]
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch
import rules as R
import run_field_recovery as rf

ENVS = ['circ_lm8_r0', 'rect_lm8_r0', 'corr_lm8_r0']
CHANNELS = ['hog', 'color', 'spatial', 'lidar', 'visual', 'all']
IDEAL_RADII = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
WALL_DISTS = [0.5, 1.0, 2.0, 4.0, 8.0]
MIN_MEMBERS = 12


# ----------------------------------------------------------------- one trial

def trial(X, Xsq, xy, mem, G, occupied, C, rng):
    """One reconstruction. Mirrors rf.field_from_members in 'quantile' mode.

    The only departure is how the response is evaluated. rf computes
    ||x - mu||^2 by materialising the difference in chunks; here it comes from
    the expansion ||x||^2 - 2 x.mu + ||mu||^2, which is one BLAS matvec against
    a precomputed norm. Same quantity, and `--verify` checks that against rf
    directly -- but several thousand trials over a 60k-dimensional matrix is
    the difference between hours and most of a day.
    """
    mu = X[mem].mean(axis=0)

    s = mem if len(mem) <= C['SIGMA_MAX_MEMBERS'] else \
        rng.choice(mem, C['SIGMA_MAX_MEMBERS'], replace=False)
    P = X[s].astype(np.float32)
    D2sub = np.maximum(
        (P ** 2).sum(1)[:, None] - 2.0 * (P @ P.T) + (P ** 2).sum(1)[None, :], 0.0)
    sigma = rf.sigma_for(D2sub, 'quantile', C)

    d2 = np.maximum(Xsq - 2.0 * (X @ mu.astype(np.float64)) +
                    float((mu.astype(np.float64) ** 2).sum()), 0.0)
    resp = np.exp(-d2 / (2.0 * sigma ** 2)) if sigma > 0 else \
        (d2 <= 1e-12).astype(float)

    flat = R._bin_indices(xy, G)
    grid = np.zeros(G['gx'] * G['gy'], np.float32)
    np.maximum.at(grid, flat, resp.astype(np.float32))
    mask, _ = R.mask_from_grid(grid, G, occupied, C)
    return mask, sigma


def verify_against_rf(X, Xsq, xy, env, G, occupied, C, rng, n=3):
    """The fast response path must agree with rf.field_from_members."""
    sites = R.plant_sites_by_wall(xy, env, WALL_DISTS, 2,
                                  np.random.default_rng(0))[:n]
    worst = 0.0
    for (cx, cy, _, _) in sites:
        mem = rf.members_disc(xy, cx, cy, 1.0)
        if len(mem) < MIN_MEMBERS:
            continue
        a, sa = trial(X, Xsq, xy, mem, G, occupied, C,
                      np.random.default_rng(1))
        b, sb = rf.field_from_members(X, xy, mem, G, occupied, C,
                                      'quantile', np.random.default_rng(1))
        worst = max(worst, float((a ^ b).sum()) / max(float(b.sum()), 1.0))
        assert abs(sa - sb) < 1e-6 * max(abs(sb), 1.0), (sa, sb)
    return worst


# ------------------------------------------------------------------ one env

def run_env(env_name, channel_names, C, args, rng):
    data_path = f'{REPO}/data/vpce/collect_data/{env_name}.h5'
    xml_path = f'{REPO}/simulation/worlds/environments/vpce/{env_name}.xml'

    print(f'\n{"=" * 72}\n{env_name}\n{"=" * 72}', flush=True)
    blocks, xy = ch.load_channel_blocks(data_path)
    if args.subsample and args.subsample < len(xy):
        keep = np.sort(rng.choice(len(xy), args.subsample, replace=False))
        xy, blocks = xy[keep], {k: v[keep] for k, v in blocks.items()}
        print(f'  subsampled to {len(xy)} positions')

    env = R.build_env(xy, ET.parse(xml_path).getroot())
    G = R._grid_setup(env, C)
    occupied = np.zeros(G['gx'] * G['gy'], dtype=bool)
    occupied[R._bin_indices(xy, G)] = True
    occupied = occupied.reshape(G['gx'], G['gy'])

    area_min = C['RULE8_AREA_FRAC'] * env['env_area']
    area_max = C['RULE9_AREA_FRAC'] * env['env_area']
    sites = R.plant_sites_by_wall(xy, env, WALL_DISTS, args.sites, rng)
    contours = sorted({s[2] for s in sites})
    print(f'  area {env["env_area"]:.2f} m2 | {len(sites)} sites on '
          f'wall contours {contours}')
    if not sites:
        print('  no plantable sites -- skipping')
        return [], {}, {}

    rows, masks = [], {}
    for cname in channel_names:
        t0 = time.time()
        X = ch.assemble(blocks, ch.CHANNEL_SETS[cname], normalize=True)
        Xsq = (X.astype(np.float64) ** 2).sum(1)

        if args.verify:
            w = verify_against_rf(X, Xsq, xy, env, G, occupied, C, rng)
            print(f'  [{cname}] fast path vs rf: worst mask disagreement '
                  f'{100 * w:.4f}%')

        n_done = 0
        for si, (cx, cy, w_target, w_actual) in enumerate(sites):
            for r in IDEAL_RADII:
                mem = rf.members_disc(xy, cx, cy, r)
                if len(mem) < MIN_MEMBERS:
                    continue
                mask, sigma = trial(X, Xsq, xy, mem, G, occupied, C, rng)
                im = rf.ideal_mask(xy, mem, G, occupied, C)
                rows.append(dict(
                    env=env_name, channel=cname, site=si, cx=cx, cy=cy,
                    wall_target_m=w_target, wall_actual_m=w_actual,
                    ideal_r_m=r, n_members=len(mem), sigma=sigma,
                    extent_pctl=C['EXTENT_PCTL'],
                    **rf.score(mask, im, G, C, env, area_min, area_max)))
                masks[f'{env_name}|{cname}|{si}|{r}'] = mask
                masks[f'{env_name}|ideal|{si}|{r}'] = im
                n_done += 1

        sub = [x for x in rows if x['channel'] == cname]
        iou = np.median([x['iou'] for x in sub])
        adm = np.mean([x['pass_size'] and x['pass_contiguity'] for x in sub])
        print(f'  [{cname:8s}] {n_done:4d} trials  median IoU {iou:.3f}  '
              f'admitted {100 * adm:3.0f}%  ({time.time() - t0:.0f}s)',
              flush=True)
        del X, Xsq

    geom = dict(env_area=env['env_area'], is_circular=env['is_circular'],
                long_dim=env['long_dim'], n_positions=int(len(xy)),
                n_sites=len(sites), wall_contours=contours,
                grid=[int(G['gx']), int(G['gy'])])
    return rows, masks, geom


# ------------------------------------------------------------ initial results

def summarise(df):
    """The tables worth reading before anyone opens a plotting notebook."""
    out = []

    def block(title, tbl):
        out.append(f'\n{title}\n' + '-' * len(title) + '\n' + tbl.to_string())

    g = df.groupby(['env', 'channel'])
    per = g.agg(trials=('iou', 'size'),
                median_iou=('iou', 'median'),
                center_err_m=('center_err_m', 'median'),
                area_ratio=('log2_area_ratio', lambda s: 2 ** s.median()),
                admitted=('pass_size', 'mean')).round(3)
    per['admitted'] = (100 * df.groupby(['env', 'channel']).apply(
        lambda d: (d.pass_size & d.pass_contiguity).mean(),
        include_groups=False)).round(0)
    block('Per environment and channel (median over all sizes and locations)', per)

    piv = df.pivot_table(index='channel', columns='env', values='iou',
                         aggfunc='median').round(3)
    piv['mean'] = piv.mean(axis=1).round(3)
    block('Median IoU by channel and environment (higher is better)',
          piv.sort_values('mean', ascending=False))

    one = df[np.isclose(df.ideal_r_m, 1.0)]
    if len(one):
        block('A 1 m ideal place cell comes back as (equivalent-disc radius, m)',
              one.pivot_table(index='channel', columns='env',
                              values='rec_r_eq_m', aggfunc='median').round(2))

    block('Median IoU by ideal-cell radius (m), pooled over channels',
          df.pivot_table(index='ideal_r_m', columns='env', values='iou',
                         aggfunc='median').round(3))

    block('Median IoU by distance from the wall (m), pooled over channels',
          df.pivot_table(index='wall_target_m', columns='env', values='iou',
                         aggfunc='median').round(3))
    return '\n'.join(out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--envs', default=','.join(ENVS))
    p.add_argument('--channels', default=','.join(CHANNELS))
    p.add_argument('--pctl', type=float, default=65.0)
    p.add_argument('--sites', type=int, default=8,
                   help='ideal cells per wall-distance contour')
    p.add_argument('--subsample', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--verify', action='store_true',
                   help='check the fast response path against run_field_recovery')
    p.add_argument('--out', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    envs = [e.strip() for e in args.envs.split(',') if e.strip()]
    chans = [c.strip() for c in args.channels.split(',') if c.strip()]
    rng = np.random.default_rng(args.seed)
    out_dir = args.out or f'{REPO}/data_cache/geometry_recovery'
    os.makedirs(out_dir, exist_ok=True)

    C = R.resolve_cfg(dict(SIGMA_MODE='quantile', EXTENT_PCTL=args.pctl,
                           RANDOM_SEED=args.seed))

    print('=' * 72)
    print('Geometry x channel recovery')
    print(f'  envs      : {envs}')
    print(f'  channels  : {chans}')
    print(f'  sigma     : quantile, EXTENT_PCTL = {C["EXTENT_PCTL"]:g}')
    print(f'  radii     : {IDEAL_RADII}')
    print(f'  wall dists: {WALL_DISTS}  ({args.sites} sites each)')
    print(f'  out       : {out_dir}')
    print('=' * 72, flush=True)

    all_rows, all_masks, geom = [], {}, {}
    for env_name in envs:
        rows, masks, g = run_env(env_name, chans, C, args, rng)
        all_rows += rows
        all_masks.update(masks)
        geom[env_name] = g
        # Written per environment so a run that dies late still leaves usable
        # results for the environments that finished.
        if all_rows:
            pd.DataFrame(all_rows).to_csv(f'{out_dir}/metrics.csv', index=False)

    if not all_rows:
        print('\nNo trials ran.')
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(f'{out_dir}/metrics.csv', index=False)
    np.savez_compressed(f'{out_dir}/masks.npz', **all_masks)
    with open(f'{out_dir}/config.json', 'w') as f:
        json.dump(dict(envs=envs, channels=chans, extent_pctl=C['EXTENT_PCTL'],
                       sigma_mode='quantile', ideal_radii=IDEAL_RADII,
                       wall_dists=WALL_DISTS, sites_per_contour=args.sites,
                       act_thresh=C['ACT_THRESH'], bin_m=C['BIN_M'],
                       cc_frac_min=C['CC_FRAC_MIN'],
                       rule8_area_frac=C['RULE8_AREA_FRAC'],
                       rule9_area_frac=C['RULE9_AREA_FRAC'],
                       subsample=args.subsample, seed=args.seed,
                       geometry=geom), f, indent=2, default=float)

    text = summarise(df)
    print(text)
    with open(f'{out_dir}/initial_results.txt', 'w') as f:
        f.write(text + '\n')

    print(f'\n{len(df)} trials -> {out_dir}/metrics.csv')
    print(f'masks           -> {out_dir}/masks.npz')
    print(f'initial results -> {out_dir}/initial_results.txt')


if __name__ == '__main__':
    main()
