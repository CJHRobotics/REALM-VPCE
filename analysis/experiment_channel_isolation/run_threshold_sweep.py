"""Does the response threshold decide the smallest field we can find?

Fields are defined as the part of the arena where a group's response reaches
some fraction of its own peak. We have used 0.50. The electrophysiology
convention is 0.20 (Kjelstrup et al. 2008; O'Keefe & Burgess 1996), so our
fields are measured more conservatively than every field size we compare
against.

Two questions:

1. **Comparability.** How much do reported field sizes change between our
   0.50 and the literature's 0.20? Any comparison against Harland or Eliav
   is only meaningful at a matched threshold.

2. **The smallest field.** In the channel-isolation run the minimum field
   size was set by the coverage rule, which drops scale bands whose fields
   cannot tile the arena. A lower threshold makes every field larger, which
   should make bands tile more easily — but it also pushes fields up into
   coarser bands, and enlarges the exclusion radius in the competition rule.
   Which effect wins is not something to reason out; it is measurable.

The tree, the candidate groups, and their centres and widths are computed
once per channel and reused for every threshold, so all settings are scored
against identical groups and only the threshold differs.

Usage
    python run_threshold_sweep.py [env_name] [--thresholds 0.2,0.3,0.4,0.5]
                                  [--channels color,lidar,spatial,all]
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
import matplotlib.colors as mcolors
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
CHANNEL_COLORS = {'hog': '#1f77b4', 'color': '#d62728', 'spatial': '#2ca02c',
                  'lidar': '#9467bd', 'visual': '#ff7f0e', 'all': '#17becf'}


# ------------------------------------------------------------------ report

class ThresholdSweepReport(ExperimentReport):
    """Emailed summary for the response-threshold sweep."""

    experiment = 'threshold-sweep'

    def title(self):
        if self.metrics is None or not len(self.metrics):
            return 'no results'
        moved = self._bands_moved()
        n = len(self.metrics)
        return (f'{n} runs — finest band '
                + ('moved down at lower threshold' if moved else 'unchanged'))

    def _bands_moved(self):
        """Did a lower threshold admit a finer scale band anywhere?"""
        m = self.metrics
        if m is None or not len(m):
            return False
        for _, g in m.groupby('channel'):
            g = g.sort_values('act_thresh')
            if g['band_lo'].iloc[0] < g['band_lo'].iloc[-1]:
                return True
        return False

    def body(self):
        m = self.metrics
        out = []

        out.append(self.section(
            'Question',
            'Our field boundary is 50% of a group\'s own peak response. The\n'
            'ephys convention is 20%, so our fields are measured more\n'
            'conservatively than the literature we compare against. Does\n'
            'matching the convention change which fields survive, and does it\n'
            'lower the smallest field we can find?'))

        cols = [c for c in ('channel', 'act_thresh', 'n_fields', 'min_radius_m',
                            'median_radius_m', 'max_radius_m', 'median_elongation',
                            'band_lo', 'band_hi', 'lowest_band_coverage')
                if c in m.columns]
        out.append(self.section('Results', self.table(m, cols)))

        # size inflation relative to our 0.50 baseline
        lines = []
        for cn, g in m.groupby('channel'):
            g = g.sort_values('act_thresh')
            base = g[np.isclose(g['act_thresh'], 0.50)]
            low = g[np.isclose(g['act_thresh'], 0.20)]
            if len(base) and len(low) and base['median_radius_m'].iloc[0] > 0:
                ratio = low['median_radius_m'].iloc[0] / base['median_radius_m'].iloc[0]
                lines.append(f'{cn:8s} median radius x{ratio:.2f} '
                             f'({base["median_radius_m"].iloc[0]:.2f} m -> '
                             f'{low["median_radius_m"].iloc[0]:.2f} m)')
        if lines:
            out.append(self.section(
                'Size change, 0.50 -> 0.20 (comparability)',
                self.bullets(lines) +
                '\n\n  Field sizes quoted against Harland or Eliav are only\n'
                '  comparable at a matched threshold. These factors are the\n'
                '  correction our previously reported numbers would need.'))

        # the finest-band question
        lines = []
        for cn, g in m.groupby('channel'):
            g = g.sort_values('act_thresh')
            lo_at_low, lo_at_high = int(g['band_lo'].iloc[0]), int(g['band_lo'].iloc[-1])
            rmin_low, rmin_high = g['min_radius_m'].iloc[0], g['min_radius_m'].iloc[-1]
            verdict = ('finer band admitted' if lo_at_low < lo_at_high
                       else 'no change' if lo_at_low == lo_at_high
                       else 'coarser band only')
            lines.append(f'{cn:8s} band_lo {lo_at_high} -> {lo_at_low}, '
                         f'min radius {rmin_high:.2f} -> {rmin_low:.2f} m  [{verdict}]')
        out.append(self.section(
            'Does a lower threshold reach smaller fields?',
            self.bullets(lines) +
            '\n\n  Note the two effects pull opposite ways: a lower threshold\n'
            '  enlarges every field (raising the minimum size in metres) while\n'
            '  also making bands easier to tile (which can admit a finer band).'))

        out.append(self.section(
            'Attached',
            self.bullets([
                'T1 field maps, channel x threshold',
                'T2 how each measured quantity moves with threshold',
                'T3 band coverage against the 50% tiling requirement',
                'T4 field size distributions',
                'metrics.csv, every column',
            ])))
        return '\n'.join(out)

    def figures(self):
        return [os.path.join(self.fig_dir, f) for f in
                ('T1_field_maps.png', 'T2_threshold_effect.png',
                 'T3_band_coverage.png', 'T4_size_distributions.png')]

    def data_files(self):
        return [os.path.join(self.out_dir, 'metrics.csv')]


# ------------------------------------------------------------------ figures

def _draw_env(ax, env, xml_root):
    th = np.linspace(0, 2 * np.pi, 240)
    if env['is_circular']:
        ax.plot(env['env_R'] * np.cos(th), env['env_R'] * np.sin(th),
                color='black', lw=1.5, zorder=6)
    if xml_root is not None:
        for lm in xml_root.findall('landmark'):
            x, y = float(lm.get('x')), float(lm.get('y'))
            c = (float(lm.get('red', 0)), float(lm.get('green', 0)),
                 float(lm.get('blue', 0)))
            ax.plot([x], [y], marker='s', ms=5, color=c, zorder=7)
    ax.set_xlim(env['x_min'] - 0.5, env['x_max'] + 0.5)
    ax.set_ylim(env['y_min'] - 0.5, env['y_max'] + 0.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


def make_figures(banks, reports, metrics, env, xml_root, fig_dir,
                 env_name, channels, thresholds):
    os.makedirs(fig_dir, exist_ok=True)

    # ---- T1 field maps ---------------------------------------------------
    nz = [b for b in banks.values() if len(b)]
    if nz:
        a_all = np.concatenate([b['area_env_m2'].to_numpy() for b in nz])
        cmap = plt.get_cmap('plasma')
        norm = mcolors.LogNorm(max(a_all.min(), 1e-3), a_all.max())
        fig, axes = plt.subplots(len(channels), len(thresholds),
                                 figsize=(3.4 * len(thresholds), 3.4 * len(channels)),
                                 squeeze=False)
        for i, cn in enumerate(channels):
            for j, t in enumerate(thresholds):
                ax = axes[i][j]
                _draw_env(ax, env, xml_root)
                b = banks.get((cn, t))
                if b is not None and len(b):
                    for _, r in b.iterrows():
                        c = cmap(norm(r['area_env_m2']))
                        ax.add_patch(Ellipse(
                            (r['centroid_x'], r['centroid_y']),
                            2 * r['semi_major_m'], 2 * r['semi_minor_m'],
                            angle=np.degrees(r['orientation_rad']),
                            facecolor=c, edgecolor=c, alpha=0.20, lw=0.8))
                ax.set_title(f'{cn} | thresh={t:g}\n{0 if b is None else len(b)} fields',
                             fontsize=9)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02).set_label('field area (m$^2$)')
        fig.suptitle(f'T1  {env_name} | fields by channel and response threshold\n'
                     f'0.20 is the ephys convention; 0.50 is what we have used',
                     y=1.004, fontsize=13)
        fig.savefig(f'{fig_dir}/T1_field_maps.png', dpi=140, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)

    # ---- T2 effect curves ------------------------------------------------
    panels = [('n_fields', 'fields admitted'),
              ('min_radius_m', 'smallest field (m)'),
              ('median_radius_m', 'median field radius (m)'),
              ('band_lo', 'finest band admitted'),
              ('lowest_band_coverage', 'coverage of finest admitted band'),
              ('median_elongation', 'median elongation')]
    panels = [(c, l) for c, l in panels if c in metrics.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4), squeeze=False)
    for k, (col, label) in enumerate(panels):
        ax = axes[k // 3][k % 3]
        for cn, g in metrics.groupby('channel'):
            g = g.sort_values('act_thresh')
            ax.plot(g['act_thresh'], g[col], marker='o', ms=6,
                    color=CHANNEL_COLORS.get(cn), label=cn)
        ax.axvline(0.20, color='green', ls=':', lw=1.4)
        ax.axvline(0.50, color='gray', ls=':', lw=1.4)
        ax.set_xlabel('response threshold (fraction of peak)')
        ax.set_ylabel(label); ax.grid(alpha=0.3)
    for k in range(len(panels), 6):
        axes[k // 3][k % 3].axis('off')
    axes[0][0].legend(fontsize=8)
    axes[0][0].text(0.20, axes[0][0].get_ylim()[1], ' ephys', fontsize=8,
                    color='green', va='top')
    axes[0][0].text(0.50, axes[0][0].get_ylim()[1], ' ours', fontsize=8,
                    color='gray', va='top')
    fig.suptitle(f'T2  {env_name} | effect of the response threshold', y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/T2_threshold_effect.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

    # ---- T3 band coverage ------------------------------------------------
    fig, axes = plt.subplots(1, len(channels), figsize=(4.4 * len(channels), 4.2),
                             squeeze=False)
    for i, cn in enumerate(channels):
        ax = axes[0][i]
        for t in thresholds:
            rep = reports.get((cn, t))
            if not rep or not rep.get('coverage'):
                continue
            bs = sorted(rep['coverage'])
            ax.plot(bs, [100 * rep['coverage'][b] for b in bs], marker='o', ms=5,
                    label=f'{t:g}')
        ax.axhline(50, color='red', ls='--', lw=1.4)
        ax.set_title(cn); ax.set_xlabel('scale band'); ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel('% of arena covered')
        ax.legend(fontsize=8, title='threshold')
    fig.suptitle(f'T3  {env_name} | can each band tile the arena? '
                 f'(red = the 50% requirement)', y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/T3_band_coverage.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

    # ---- T4 size distributions -------------------------------------------
    fig, axes = plt.subplots(1, len(channels), figsize=(4.4 * len(channels), 4.2),
                             squeeze=False)
    for i, cn in enumerate(channels):
        ax = axes[0][i]
        data, labels = [], []
        for t in thresholds:
            b = banks.get((cn, t))
            if b is not None and len(b):
                data.append(b['radius_env_m'].to_numpy()); labels.append(f'{t:g}')
        if data:
            try:
                bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                                showfliers=False)
            except TypeError:
                bp = ax.boxplot(data, labels=labels, patch_artist=True,
                                showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor(CHANNEL_COLORS.get(cn, 'gray')); patch.set_alpha(0.5)
        ax.set_title(cn); ax.set_xlabel('threshold'); ax.grid(alpha=0.3, axis='y')
        if i == 0:
            ax.set_ylabel('field radius (m)')
    fig.suptitle(f'T4  {env_name} | field size distribution by threshold',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/T4_size_distributions.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


# ------------------------------------------------------------------ main

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('env', nargs='?', default='circ_lm8_r0')
    ap.add_argument('--channels', default='color,lidar,spatial,all')
    ap.add_argument('--thresholds', default='0.2,0.3,0.4,0.5')
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
    thresholds = sorted(float(t) for t in args.thresholds.split(',') if t.strip())

    data_path = f'{REPO}/data/vpce/collect_data/{env_name}.h5'
    xml_path = f'{REPO}/simulation/worlds/environments/vpce/{env_name}.xml'
    out_dir = f'{REPO}/data_cache/threshold_sweep/{env_name}'
    fig_dir = f'{HERE}/figures/threshold_sweep/{env_name}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=' * 72)
    print(f'Response-threshold sweep | env={env_name}')
    print(f'  channels   : {channel_names}')
    print(f'  thresholds : {thresholds}   (0.20 = ephys convention, 0.50 = ours)')
    print(f'  lambda     : {args.lam}')
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

    banks, reports, metrics = {}, {}, []

    for cname in channel_names:
        keys = ch.CHANNEL_SETS.get(cname)
        if not keys or any(k not in blocks for k in keys):
            print(f'  !! skipping {cname}')
            continue
        print(f'\n----- channel: {cname} -----')
        X = ch.assemble(blocks, keys, normalize=True)
        D2 = R.feature_sq_distances(X, device=device)
        feat_med = R._median_offdiag(D2, rng)

        # One tree, one set of candidates, one readout — reused for every
        # threshold, so the comparison isolates the threshold alone.
        base_cfg = dict(LAMBDA=args.lam, BIN_M=args.bin_m, RANDOM_SEED=args.seed)
        ctx = R.prepare_candidates(X, xy, env, D2, feat_med, xy_med,
                                   cfg=base_cfg, device=device, tag=cname)

        for t in thresholds:
            tag = f'{cname}/t{t:g}'
            print(f'\n  === {tag} ===')
            t0 = time.time()
            cfg = dict(base_cfg, ACT_THRESH=t)
            ctx['tag'] = tag
            bank, kept_mu, rep = R.admit_fields(ctx, cfg=cfg)

            d = f'{out_dir}/{cname}/t{t:g}'
            os.makedirs(d, exist_ok=True)
            bank.to_csv(f'{d}/bank.csv', index=False)
            with open(f'{d}/report.json', 'w') as f:
                json.dump({k: v for k, v in rep.items()
                           if not isinstance(v, np.ndarray)}, f, indent=2,
                          default=float)

            banks[(cname, t)] = bank
            reports[(cname, t)] = rep
            cov = rep['coverage']
            row = dict(channel=cname, act_thresh=t, n_fields=len(bank),
                       band_lo=rep['band_lo'], band_hi=rep['band_hi'],
                       lowest_band_coverage=(cov.get(rep['band_lo'], np.nan)
                                             if rep['band_lo'] >= 0 else np.nan),
                       runtime_s=round(time.time() - t0, 1))
            for name, cnt in rep['funnel']:
                row[f'funnel_{name}'] = cnt
            if len(bank):
                row.update(min_radius_m=float(bank['radius_env_m'].min()),
                           median_radius_m=float(bank['radius_env_m'].median()),
                           max_radius_m=float(bank['radius_env_m'].max()),
                           median_elongation=float(bank['elongation'].median()),
                           corr_radius_wall=float(np.corrcoef(
                               bank['dist_to_wall_m'], bank['radius_env_m'])[0, 1])
                           if len(bank) >= 5 else np.nan)
            metrics.append(row)
            print(f'  -> {len(bank)} fields, band_lo={rep["band_lo"]}, '
                  f'{row["runtime_s"]}s')

        del X, D2, ctx

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{out_dir}/metrics.csv', index=False)
    print(f'\nMetrics -> {out_dir}/metrics.csv')
    if len(metrics_df):
        cols = [c for c in ('channel', 'act_thresh', 'n_fields', 'min_radius_m',
                            'median_radius_m', 'band_lo', 'lowest_band_coverage')
                if c in metrics_df.columns]
        for line in metrics_df[cols].to_string(index=False).splitlines():
            print(f'[summary] {line}')

    present = [c for c in channel_names if any(k[0] == c for k in banks)]
    make_figures(banks, reports, metrics_df, env, xml_root, fig_dir,
                 env_name, present, thresholds)
    print(f'Figures -> {fig_dir}')

    if not args.no_email:
        ThresholdSweepReport(
            env_name=env_name, exit_status=args.exit_status,
            fig_dir=fig_dir, out_dir=out_dir, metrics=metrics_df,
        ).send()


if __name__ == '__main__':
    main()
