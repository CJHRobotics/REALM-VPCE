"""Which pruning rule is removing the small fields?

The channel-isolation run left every bank with a floor on field size, and
that floor is set by the coverage rule: the smallest surviving field sits at
the bottom edge of the finest scale band that managed to cover the arena.
The response threshold turned out not to be the lever — lowering it to the
ephys convention made small fields *worse*, because it enlarges every field
rather than admitting more of them.

Two rules remain, and they interact:

  * **competition** suppresses same-band fields whose centres are closer than
    SAME_SCALE_SEPARATION x (r_a + r_b). It removes 90-96% of candidates, by
    far the largest cut in the pipeline. At the current 0.5, two equal fields
    must be a full radius apart — stricter than real place cells, which
    overlap heavily within one dorsoventral level.
  * **coverage** discards any band whose surviving fields cover less than
    TILING_FRAC_MIN of the arena. Fine bands need many more fields to tile
    than coarse ones (about 41 fields at 1.1 m radius against 15 at 1.8 m),
    so a scale-independent 90% cut by competition hits them hardest, and
    coverage then deletes the band outright.

This sweeps both together. Neither value was derived from a measurement, so
the point is to see which one is actually holding the floor up, and whether
either recovers the fine bands at a setting that remains defensible.

Both parameters act only in `admit_fields`, so the tree, the candidates and
their centres and widths are computed once per channel and reused across the
whole grid — every cell is scored against identical groups.

Usage
    python run_pruning_sweep.py [env_name]
        [--separations 0.15,0.25,0.35,0.5,0.7]
        [--coverages 0.3,0.4,0.5,0.6,0.7]
        [--channels color,lidar,spatial,all] [--act-thresh 0.5]
        [--subsample N] [--no-gpu] [--no-email]
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
from matplotlib.patches import Ellipse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch
import rules as R
from realm_tools.experiment_lib.reporting import ExperimentReport

LIDAR_MAX_RANGE, LIDAR_SENTINEL = 5.0, -1.0
DEFAULT_SEP = R.DEFAULT_CFG['SAME_SCALE_SEPARATION']    # 0.50
DEFAULT_COV = R.DEFAULT_CFG['TILING_FRAC_MIN']          # 0.50


# ------------------------------------------------------------------ report

class PruningSweepReport(ExperimentReport):
    """Emailed summary for the competition x coverage sweep."""

    experiment = 'pruning-sweep'

    def _baseline(self, cn):
        m = self.metrics
        b = m[(m.channel == cn) & np.isclose(m.separation, DEFAULT_SEP)
              & np.isclose(m.coverage, DEFAULT_COV)]
        return b.iloc[0] if len(b) else None

    def _best(self, cn):
        """Setting reaching the finest band; ties broken by most fields."""
        g = self.metrics[(self.metrics.channel == cn) & (self.metrics.band_lo >= 0)]
        if not len(g):
            return None
        g = g.sort_values(['band_lo', 'n_fields'], ascending=[True, False])
        return g.iloc[0]

    def title(self):
        if self.metrics is None or not len(self.metrics):
            return 'no results'
        gains = []
        for cn in self.metrics.channel.unique():
            base, best = self._baseline(cn), self._best(cn)
            if base is not None and best is not None and best.band_lo < base.band_lo:
                gains.append(cn)
        if not gains:
            return 'no setting reached a finer band than the current default'
        return f'finer bands reachable in {len(gains)}/{self.metrics.channel.nunique()} channels'

    def body(self):
        m = self.metrics
        out = [self.section(
            'Question',
            'The smallest field in every bank sits at the bottom edge of the\n'
            'finest scale band that passed the coverage rule. Competition\n'
            f'(separation, default {DEFAULT_SEP}) removes 90-96% of candidates and\n'
            f'coverage (default {DEFAULT_COV}) then deletes bands left too thin to\n'
            'tile. Neither number came from a measurement. Which one is\n'
            'holding the floor up?')]

        rows = []
        for cn in m.channel.unique():
            base, best = self._baseline(cn), self._best(cn)
            if base is None or best is None:
                continue
            rows.append(dict(
                channel=cn,
                base_band=int(base.band_lo), base_fields=int(base.n_fields),
                base_min_r=base.min_radius_m,
                best_sep=best.separation, best_cov=best.coverage,
                best_band=int(best.band_lo), best_fields=int(best.n_fields),
                best_min_r=best.min_radius_m))
        if rows:
            out.append(self.section(
                f'Current default (sep={DEFAULT_SEP}, cov={DEFAULT_COV}) vs the best setting found',
                self.table(pd.DataFrame(rows))))

        # which lever moves the floor: vary one, hold the other at default
        lines = []
        for cn in m.channel.unique():
            g = m[m.channel == cn]
            sep_only = g[np.isclose(g.coverage, DEFAULT_COV)].sort_values('separation')
            cov_only = g[np.isclose(g.separation, DEFAULT_SEP)].sort_values('coverage')
            def span(d):
                v = d[d.band_lo >= 0]['band_lo']
                return f'{int(v.max())} -> {int(v.min())}' if len(v) else 'n/a'
            lines.append(f'{cn:8s} varying separation alone: band {span(sep_only)}   '
                         f'varying coverage alone: band {span(cov_only)}')
        out.append(self.section('Which lever moves the floor?', self.bullets(lines)))

        cols = [c for c in ('channel', 'separation', 'coverage', 'n_fields',
                            'band_lo', 'band_hi', 'min_radius_m', 'median_radius_m')
                if c in m.columns]
        out.append(self.section('Full grid', self.table(m, cols, max_rows=120)))

        out.append(self.section(
            'Reading these',
            'A lower band_lo means a finer scale survived, which is the\n'
            'outcome we want. Relaxing competition is the more defensible of\n'
            'the two: real place fields at one dorsoventral level overlap\n'
            'heavily, so requiring a full radius between centres is stricter\n'
            'than the biology. Lowering the coverage requirement is the less\n'
            'principled lever - 0.5 was already a choice, and so is 0.4.\n'
            'If only coverage moves the floor, that is worth knowing before\n'
            'either number is defended in print.'))

        out.append(self.section('Attached', self.bullets([
            'S1 finest band, smallest field and field count across the grid',
            'S2 field maps at the grid corners against the current default',
            'S3 how many fields survive in each band',
            'metrics.csv, every cell of the grid'])))
        return '\n'.join(out)

    def figures(self):
        return [os.path.join(self.fig_dir, f) for f in
                ('S1_grid_heatmaps.png', 'S2_field_maps.png', 'S3_fields_per_band.png')]

    def data_files(self):
        return [os.path.join(self.out_dir, 'metrics.csv')]


# ------------------------------------------------------------------ figures

def _heat(ax, M, seps, covs, title, cmap, fmt='{:.0f}'):
    im = ax.imshow(M, cmap=cmap, origin='lower', aspect='auto')
    ax.set_xticks(range(len(covs))); ax.set_xticklabels([f'{c:g}' for c in covs])
    ax.set_yticks(range(len(seps))); ax.set_yticklabels([f'{s:g}' for s in seps])
    ax.set_xlabel('coverage required'); ax.set_ylabel('competition separation')
    ax.set_title(title, fontsize=10)
    finite = M[np.isfinite(M)]
    vmin, vmax = (finite.min(), finite.max()) if len(finite) else (0.0, 1.0)
    cm = plt.get_cmap(cmap)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='gray')
                continue
            # Pick the label colour from the cell's actual luminance rather
            # than guessing from the value: reversed colormaps put dark at the
            # high end and the guess comes out backwards.
            t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            r, g, b = cm(t)[:3]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8,
                    color='white' if lum < 0.5 else 'black')
    # ring the current default
    if DEFAULT_COV in list(covs) and DEFAULT_SEP in list(seps):
        j, i = list(covs).index(DEFAULT_COV), list(seps).index(DEFAULT_SEP)
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   edgecolor='red', lw=2.5))
    return im


def make_figures(banks, metrics, env, xml_root, fig_dir, env_name,
                 channels, seps, covs, map_channel):
    os.makedirs(fig_dir, exist_ok=True)
    S, C = len(seps), len(covs)

    def grid_of(cn, col):
        M = np.full((S, C), np.nan)
        for i, s in enumerate(seps):
            for j, c in enumerate(covs):
                r = metrics[(metrics.channel == cn) & np.isclose(metrics.separation, s)
                            & np.isclose(metrics.coverage, c)]
                if len(r):
                    v = r.iloc[0][col]
                    M[i, j] = np.nan if (col == 'band_lo' and v < 0) else v
        return M

    # ---- S1 heatmaps -----------------------------------------------------
    fig, axes = plt.subplots(len(channels), 3, figsize=(15, 4.1 * len(channels)),
                             squeeze=False)
    for r, cn in enumerate(channels):
        _heat(axes[r][0], grid_of(cn, 'band_lo'), seps, covs,
              f'{cn} — finest band admitted (lower = finer)', 'viridis_r')
        _heat(axes[r][1], grid_of(cn, 'min_radius_m'), seps, covs,
              f'{cn} — smallest field (m)', 'magma_r', fmt='{:.2f}')
        _heat(axes[r][2], grid_of(cn, 'n_fields'), seps, covs,
              f'{cn} — fields admitted', 'cividis')
    fig.suptitle(f'S1  {env_name} | competition separation x coverage requirement\n'
                 f'red box = current default (sep={DEFAULT_SEP:g}, cov={DEFAULT_COV:g})',
                 y=1.003, fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/S1_grid_heatmaps.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

    # ---- S2 field maps at the corners ------------------------------------
    picks = [(seps[0], covs[0]), (seps[0], covs[-1]), (DEFAULT_SEP, DEFAULT_COV),
             (seps[-1], covs[0]), (seps[-1], covs[-1])]
    picks = [p for p in picks if (map_channel, p[0], p[1]) in banks]
    if picks:
        allr = np.concatenate([banks[(map_channel, s, c)]['radius_env_m'].to_numpy()
                               for s, c in picks
                               if len(banks[(map_channel, s, c)])] or [np.array([1.0])])
        cmap, norm = plt.get_cmap('plasma'), plt.Normalize(allr.min(), allr.max())
        fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 4.0),
                                 squeeze=False)
        for k, (s, c) in enumerate(picks):
            ax = axes[0][k]
            th = np.linspace(0, 2 * np.pi, 240)
            if env['is_circular']:
                ax.plot(env['env_R'] * np.cos(th), env['env_R'] * np.sin(th),
                        color='black', lw=1.5)
            b = banks[(map_channel, s, c)]
            for _, row in b.iterrows():
                col = cmap(norm(row['radius_env_m']))
                ax.add_patch(Ellipse((row['centroid_x'], row['centroid_y']),
                                     2 * row['semi_major_m'], 2 * row['semi_minor_m'],
                                     angle=np.degrees(row['orientation_rad']),
                                     facecolor=col, edgecolor=col, alpha=0.20, lw=0.8))
            tag = ' (current)' if (s == DEFAULT_SEP and c == DEFAULT_COV) else ''
            ax.set_title(f'sep={s:g}, cov={c:g}{tag}\n{len(b)} fields', fontsize=10)
            ax.set_xlim(env['x_min'] - .5, env['x_max'] + .5)
            ax.set_ylim(env['y_min'] - .5, env['y_max'] + .5)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02).set_label('field radius (m)')
        fig.suptitle(f'S2  {env_name} | {map_channel} fields at the grid corners',
                     y=1.02, fontsize=13)
        fig.savefig(f'{fig_dir}/S2_field_maps.png', dpi=140, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)

    # ---- S3 fields per band ---------------------------------------------
    fig, axes = plt.subplots(1, len(channels), figsize=(4.4 * len(channels), 4.2),
                             squeeze=False)
    for k, cn in enumerate(channels):
        ax = axes[0][k]
        for s in seps:
            key = (cn, s, DEFAULT_COV)
            b = banks.get(key)
            if b is None or not len(b):
                continue
            cnt = b['scale_band'].value_counts().sort_index()
            ax.plot(cnt.index, cnt.values, marker='o', ms=5, label=f'sep={s:g}')
        ax.set_title(f'{cn}  (coverage held at {DEFAULT_COV:g})', fontsize=10)
        ax.set_xlabel('scale band'); ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel('fields surviving')
        ax.legend(fontsize=8)
    fig.suptitle(f'S3  {env_name} | where the fields go as competition is relaxed',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/S3_fields_per_band.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


# ------------------------------------------------------------------ main

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('env', nargs='?', default='circ_lm8_r0')
    ap.add_argument('--channels', default='color,lidar,spatial,all')
    ap.add_argument('--separations', default='0.15,0.25,0.35,0.5,0.7')
    ap.add_argument('--coverages', default='0.3,0.4,0.5,0.6,0.7')
    ap.add_argument('--act-thresh', type=float, default=R.DEFAULT_CFG['ACT_THRESH'])
    ap.add_argument('--map-channel', default='color')
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--bin-m', type=float, default=R.DEFAULT_CFG['BIN_M'])
    ap.add_argument('--subsample', type=int, default=0)
    ap.add_argument('--no-gpu', action='store_true')
    ap.add_argument('--no-email', action='store_true')
    ap.add_argument('--exit-status', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    env_name = args.env
    channel_names = [c.strip() for c in args.channels.split(',') if c.strip()]
    seps = sorted(float(x) for x in args.separations.split(',') if x.strip())
    covs = sorted(float(x) for x in args.coverages.split(',') if x.strip())

    data_path = f'{REPO}/data/vpce/collect_data/{env_name}.h5'
    xml_path = f'{REPO}/simulation/worlds/environments/vpce/{env_name}.xml'
    out_dir = f'{REPO}/data_cache/pruning_sweep/{env_name}'
    fig_dir = f'{HERE}/figures/pruning_sweep/{env_name}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=' * 72)
    print(f'Pruning sweep | env={env_name}')
    print(f'  channels    : {channel_names}')
    print(f'  separations : {seps}   (current default {DEFAULT_SEP})')
    print(f'  coverages   : {covs}   (current default {DEFAULT_COV})')
    print(f'  threshold   : {args.act_thresh}   lambda: {args.lam}')
    print(f'  grid        : {len(seps)*len(covs)} settings x {len(channel_names)} channels')
    print('=' * 72)

    blocks, xy = ch.load_channel_blocks(
        data_path, lidar_max_range=LIDAR_MAX_RANGE,
        lidar_sentinel=LIDAR_SENTINEL, lidar_mask_channel=True)
    xml_root = ET.parse(xml_path).getroot() if os.path.exists(xml_path) else None
    env = R.build_env(xy, xml_root)

    if args.subsample and args.subsample < len(xy):
        sel = np.random.default_rng(args.seed).choice(len(xy), args.subsample,
                                                      replace=False)
        sel.sort()
        blocks = {k: v[sel] for k, v in blocks.items()}
        xy = xy[sel]
        print(f'  subsampled to {len(xy)} locations')

    N = len(xy)
    device = R.pick_device(use_gpu=not args.no_gpu)
    rng = np.random.default_rng(args.seed)
    ij = rng.integers(0, N, size=(2, 2_000_000))
    keep = ij[0] != ij[1]
    xy_med = float(np.median(((xy[ij[0][keep]] - xy[ij[1][keep]]) ** 2).sum(1))) or 1.0

    banks, metrics = {}, []

    for cname in channel_names:
        keys = ch.CHANNEL_SETS.get(cname)
        if not keys or any(k not in blocks for k in keys):
            print(f'  !! skipping {cname}')
            continue
        print(f'\n----- channel: {cname} -----')
        X = ch.assemble(blocks, keys, normalize=True)
        D2 = R.feature_sq_distances(X, device=device)
        feat_med = R._median_offdiag(D2, rng)

        base_cfg = dict(LAMBDA=args.lam, BIN_M=args.bin_m,
                        ACT_THRESH=args.act_thresh, RANDOM_SEED=args.seed)
        # One tree and one readout serve the whole grid: both swept
        # parameters act only inside admit_fields.
        ctx = R.prepare_candidates(X, xy, env, D2, feat_med, xy_med,
                                   cfg=base_cfg, device=device, tag=cname)

        t_grid = time.time()
        for s in seps:
            for c in covs:
                ctx['tag'] = f'{cname}/s{s:g}/c{c:g}'
                cfg = dict(base_cfg, SAME_SCALE_SEPARATION=s, TILING_FRAC_MIN=c)
                bank, _, rep = R.admit_fields(ctx, cfg=cfg, verbose=False)
                banks[(cname, s, c)] = bank
                cov_map = rep['coverage']
                row = dict(channel=cname, separation=s, coverage=c,
                           n_fields=len(bank), band_lo=rep['band_lo'],
                           band_hi=rep['band_hi'],
                           lowest_band_coverage=(cov_map.get(rep['band_lo'], np.nan)
                                                 if rep['band_lo'] >= 0 else np.nan))
                for name, cnt in rep['funnel']:
                    row[f'funnel_{name}'] = cnt
                if len(bank):
                    row.update(
                        min_radius_m=float(bank['radius_env_m'].min()),
                        median_radius_m=float(bank['radius_env_m'].median()),
                        max_radius_m=float(bank['radius_env_m'].max()),
                        median_elongation=float(bank['elongation'].median()),
                        corr_radius_wall=(float(np.corrcoef(
                            bank['dist_to_wall_m'], bank['radius_env_m'])[0, 1])
                            if len(bank) >= 5 else np.nan))
                metrics.append(row)
                print(f'  sep={s:<5g} cov={c:<5g} -> {len(bank):4d} fields, '
                      f'band_lo={rep["band_lo"]}, '
                      f'min_r={row.get("min_radius_m", float("nan")):.2f}')
        print(f'  grid for {cname}: {time.time()-t_grid:.1f}s')
        del X, D2, ctx

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{out_dir}/metrics.csv', index=False)
    print(f'\nMetrics -> {out_dir}/metrics.csv')

    # persist only the banks at the default and at the finest-band setting
    for cn in metrics_df.channel.unique():
        g = metrics_df[(metrics_df.channel == cn) & (metrics_df.band_lo >= 0)]
        picks = {('default', DEFAULT_SEP, DEFAULT_COV)}
        if len(g):
            b = g.sort_values(['band_lo', 'n_fields'], ascending=[True, False]).iloc[0]
            picks.add(('finest', b.separation, b.coverage))
        for label, s, c in picks:
            bank = banks.get((cn, s, c))
            if bank is None:
                continue
            d = f'{out_dir}/{cn}/{label}_s{s:g}_c{c:g}'
            os.makedirs(d, exist_ok=True)
            bank.to_csv(f'{d}/bank.csv', index=False)

    if len(metrics_df):
        cols = [c for c in ('channel', 'separation', 'coverage', 'n_fields',
                            'band_lo', 'min_radius_m') if c in metrics_df.columns]
        for line in metrics_df[cols].to_string(index=False).splitlines():
            print(f'[summary] {line}')

    present = [c for c in channel_names if any(k[0] == c for k in banks)]
    map_ch = args.map_channel if args.map_channel in present else (present[0] if present else None)
    if map_ch:
        make_figures(banks, metrics_df, env, xml_root, fig_dir, env_name,
                     present, seps, covs, map_ch)
        print(f'Figures -> {fig_dir}')

    if not args.no_email:
        PruningSweepReport(env_name=env_name, exit_status=args.exit_status,
                           fig_dir=fig_dir, out_dir=out_dir,
                           metrics=metrics_df).send()


if __name__ == '__main__':
    main()
