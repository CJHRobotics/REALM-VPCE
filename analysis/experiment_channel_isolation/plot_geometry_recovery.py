"""Figures for the geometry x channel recovery experiment.

Answers three questions from the saved results of run_geometry_recovery.py:

1. **Which channel reconstructs a reference field most closely**, and does the
   ranking survive a change of enclosure shape?
2. **Is elongation channel-dependent**, and does it vary with distance to the
   wall?
3. **Does reconstructed scale vary with distance to the wall**, and by channel?

Two controls matter for 2 and 3 and are applied throughout.

A reference disc planted near a wall is *clipped by that wall*, so it is itself
elongated and itself smaller than a free disc. Comparing a reconstruction's raw
elongation across wall distances therefore measures the clipping, not the model.
Every elongation figure here plots the reconstruction against the elongation of
the reference field it came from, recovered from the saved masks; every size
figure uses the ratio of reconstructed to reference area rather than raw area.

Reference elongation is computed from the second moments of the mask, which is
scale-free -- it needs neither bin size nor grid origin, so the masks alone are
sufficient and nothing has to be re-run.

Usage
    python plot_geometry_recovery.py [--in DIR] [--out DIR]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))

CHANNEL_COLORS = {'hog': '#1f77b4', 'color': '#d62728', 'spatial': '#2ca02c',
                  'lidar': '#9467bd', 'visual': '#ff7f0e', 'all': '#17becf'}
ENV_LABEL = {'circ_lm8_r0': 'disc\n(aspect 1.0)',
             'rect_lm8_r0': 'rectangle\n(aspect 1.3)',
             'corr_lm8_r0': 'corridor\n(aspect 10)'}
ENV_ORDER = ['circ_lm8_r0', 'rect_lm8_r0', 'corr_lm8_r0']


# ------------------------------------------------------------------ helpers

def mask_elongation(mask):
    """Major/minor axis ratio of the mask's second-moment ellipse.

    A ratio of eigenvalues, so it is invariant to bin size and origin: the
    masks alone determine it. Returns NaN for a mask too small or too
    degenerate for the moments to be meaningful.
    """
    ij = np.argwhere(mask)
    if len(ij) < 3:
        return np.nan
    c = np.cov((ij - ij.mean(0)).T.astype(float))
    if not np.all(np.isfinite(c)):
        return np.nan
    w = np.linalg.eigvalsh(c)
    w = np.clip(w, 0, None)
    if w[0] <= 1e-12:
        return np.nan
    return float(np.sqrt(w[1] / w[0]))


def add_reference_shape(df, masks_path):
    """Attach the reference field's own elongation to every row."""
    if not os.path.exists(masks_path):
        print(f'  no masks at {masks_path}; reference elongation unavailable')
        df['ideal_elongation'] = np.nan
        df['elongation_excess'] = np.nan
        return df
    z = np.load(masks_path)
    ideal = {}
    for k in z.files:
        env, who, si, r = k.split('|')
        if who == 'ideal':
            ideal[(env, int(si), float(r))] = mask_elongation(z[k])
    df['ideal_elongation'] = [
        ideal.get((e, int(s), float(r)), np.nan)
        for e, s, r in zip(df.env, df.site, df.ideal_r_m)]
    # Excess over the reference: >0 means the reconstruction is more
    # elongated than the (possibly wall-clipped) field it was built from.
    df['elongation_excess'] = df.rec_elongation - df.ideal_elongation
    print(f'  reference elongation for {df.ideal_elongation.notna().sum()} '
          f'of {len(df)} rows')
    return df


def spearman(x, y):
    """Rank correlation, with a t-based p-value. No scipy dependency."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, n
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan, n
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0, n
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    # Normal approximation to the t tail; adequate at these sample sizes and
    # keeps this script free of scipy.
    p = float(1 + math.erf(-abs(t) / math.sqrt(2)))
    return rho, p, n


def _channels(df):
    order = ['hog', 'color', 'spatial', 'lidar', 'visual', 'all']
    return [c for c in order if c in set(df.channel)]


def _envs(df):
    return [e for e in ENV_ORDER if e in set(df.env)]


def _wall_axis(ax, values):
    """Wall distances are geometric (0.5, 1, 2, 4, 8); space them that way."""
    v = sorted({float(x) for x in values})
    ax.set_xscale('log')
    ax.set_xticks(v)
    ax.set_xticklabels([f'{x:g}' for x in v])
    ax.minorticks_off()


def _save(fig, out_dir, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


# ------------------------------------------------------------------ figures

def fig_accuracy(df, out_dir):
    """Q1: which channel reconstructs the reference field most closely."""
    envs, chans = _envs(df), _channels(df)
    panels = [('iou', 'Median IoU\n(1 = identical)', False),
              ('center_err_m', 'Median centre error (m)\n(lower is better)', True),
              ('admitted', 'Admitted (%)', False)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout='constrained')
    w = 0.8 / max(len(envs), 1)
    for ax, (col, label, logy) in zip(axes, panels):
        for j, env in enumerate(envs):
            vals = []
            for c in chans:
                d = df[(df.env == env) & (df.channel == c)]
                if col == 'admitted':
                    vals.append(100 * (d.pass_size & d.pass_contiguity).mean()
                                if len(d) else np.nan)
                else:
                    vals.append(d[col].median() if len(d) else np.nan)
            ax.bar(np.arange(len(chans)) + j * w - 0.4 + w / 2, vals, w,
                   label=ENV_LABEL.get(env, env).replace('\n', ' '),
                   edgecolor='k', linewidth=0.4)
        ax.set_xticks(range(len(chans)))
        ax.set_xticklabels(chans, rotation=30, ha='right')
        ax.set_ylabel(label)
        if logy:
            ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='outside lower center', ncol=len(envs),
               fontsize=9, frameon=False)
    fig.suptitle('Reconstruction of a reference place field, by channel and '
                 'enclosure shape')
    _save(fig, out_dir, 'G1_accuracy_by_channel.png')


def fig_transfer(df, out_dir):
    """Q1: is reconstructed size calibrated against true size?"""
    envs, chans = _envs(df), _channels(df)
    fig, axes = plt.subplots(1, len(envs), figsize=(5.4 * len(envs), 4.6),
                             squeeze=False, layout='constrained')
    for ax, env in zip(axes[0], envs):
        d0 = df[df.env == env]
        lo = hi = None
        for c in chans:
            d = d0[d0.channel == c]
            if not len(d):
                continue
            g = d.groupby('ideal_r_eq_m').rec_r_eq_m.median()
            ax.plot(g.index, g.values, 'o-', ms=4, lw=1.4,
                    color=CHANNEL_COLORS.get(c), label=c)
            lo = min(lo, g.index.min()) if lo else g.index.min()
            hi = max(hi, g.index.max()) if hi else g.index.max()
        if lo is not None:
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='exact')
        ax.set_xlabel('reference field radius (m)')
        ax.set_ylabel('reconstructed radius (m)')
        ax.set_title(ENV_LABEL.get(env, env).replace('\n', ' '))
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle('Size transfer: recovered against true field size '
                 '(dashed = exact)')
    _save(fig, out_dir, 'G2_size_transfer.png')


def fig_error_modes(df, out_dir):
    """Q1: separate 'wrong size' from 'wrong place'.

    IoU confounds the two. Area ratio on one axis and centre error on the
    other separates them: the lower-centre region is a correct
    reconstruction, the upper-centre is correctly sized but mislocated.
    """
    envs, chans = _envs(df), _channels(df)
    fig, axes = plt.subplots(1, len(envs), figsize=(5.4 * len(envs), 4.6),
                             squeeze=False, layout='constrained')
    for ax, env in zip(axes[0], envs):
        d0 = df[df.env == env]
        for c in chans:
            d = d0[d0.channel == c]
            if not len(d):
                continue
            ax.scatter(2 ** d.log2_area_ratio, d.center_err_m, s=9, alpha=0.35,
                       color=CHANNEL_COLORS.get(c), edgecolors='none')
            ax.scatter([2 ** d.log2_area_ratio.median()], [d.center_err_m.median()],
                       s=150, marker='X', color=CHANNEL_COLORS.get(c),
                       edgecolors='k', linewidths=0.8, zorder=5, label=c)
        ax.axvline(1.0, color='k', ls='--', lw=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('area ratio (reconstructed / reference)')
        ax.set_ylabel('centre error (m)')
        ax.set_title(ENV_LABEL.get(env, env).replace('\n', ' '))
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, frameon=False, title='median (X)')
    fig.suptitle('Error decomposition: size calibration against localisation. '
                 'Dashed line is exact area; bottom-left is correct.')
    _save(fig, out_dir, 'G3_error_modes.png')


def fig_elongation(df, out_dir):
    """Q2: elongation by channel, against wall distance and its reference."""
    envs, chans = _envs(df), _channels(df)
    fig, axes = plt.subplots(2, len(envs), figsize=(5.4 * len(envs), 8.4),
                             squeeze=False, layout='constrained')
    for j, env in enumerate(envs):
        d0 = df[df.env == env]
        ax = axes[0][j]
        for c in chans:
            d = d0[d0.channel == c]
            if not len(d):
                continue
            g = d.groupby('wall_target_m').rec_elongation.median()
            ax.plot(g.index, g.values, 'o-', ms=4, color=CHANNEL_COLORS.get(c),
                    label=c)
        gi = d0.groupby('wall_target_m').ideal_elongation.median()
        if gi.notna().any():
            ax.plot(gi.index, gi.values, 'k--', lw=1.6, label='reference field')
        ax.axhline(1.0, color='grey', lw=0.8, ls=':')
        _wall_axis(ax, d0.wall_target_m)
        ax.set_xlabel('distance to nearest wall (m)')
        ax.set_ylabel('elongation (major / minor)')
        ax.set_title(ENV_LABEL.get(env, env).replace('\n', ' '))
        ax.grid(alpha=0.3)

        ax = axes[1][j]
        for c in chans:
            d = d0[d0.channel == c]
            if not len(d) or not d.elongation_excess.notna().any():
                continue
            g = d.groupby('wall_target_m').elongation_excess.median()
            ax.plot(g.index, g.values, 'o-', ms=4, color=CHANNEL_COLORS.get(c),
                    label=c)
        ax.axhline(0.0, color='k', lw=1)
        _wall_axis(ax, d0.wall_target_m)
        ax.set_xlabel('distance to nearest wall (m)')
        ax.set_ylabel('elongation in excess of reference')
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle('Elongation by channel. Top: raw, with the reference field\'s '
                 'own elongation (dashed) — a disc clipped by a wall is '
                 'elongated on its own.\nBottom: excess over that reference, '
                 'which is the part attributable to the model.')
    _save(fig, out_dir, 'G4_elongation.png')


def fig_scale_vs_wall(df, out_dir):
    """Q3: does reconstructed scale grow with distance from the wall?"""
    envs, chans = _envs(df), _channels(df)
    fig, axes = plt.subplots(1, len(envs), figsize=(5.4 * len(envs), 4.6),
                             squeeze=False, layout='constrained')
    for ax, env in zip(axes[0], envs):
        d0 = df[df.env == env]
        for c in chans:
            d = d0[d0.channel == c]
            if not len(d):
                continue
            g = d.groupby('wall_target_m').log2_area_ratio.median()
            ax.plot(g.index, 2 ** g.values, 'o-', ms=4,
                    color=CHANNEL_COLORS.get(c), label=c)
        ax.axhline(1.0, color='k', ls='--', lw=1)
        ax.set_yscale('log')
        _wall_axis(ax, d0.wall_target_m)
        ax.set_xlabel('distance to nearest wall (m)')
        ax.set_ylabel('area ratio (reconstructed / reference)')
        ax.set_title(ENV_LABEL.get(env, env).replace('\n', ' '))
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle('Reconstructed scale against wall distance, as a ratio to the '
                 'reference field.\nA ratio held at 1 means the model neither '
                 'inflates nor shrinks with proximity to a wall.')
    _save(fig, out_dir, 'G5_scale_vs_wall.png')


# -------------------------------------------------------------- correlations

def correlations(df, out_dir):
    """Q2/Q3 stated as numbers rather than read off a plot."""
    rows = []
    for env in _envs(df):
        for c in _channels(df):
            d = df[(df.env == env) & (df.channel == c)]
            if len(d) < 5:
                continue
            for label, col in [('elongation ~ wall', 'rec_elongation'),
                               ('elongation excess ~ wall', 'elongation_excess'),
                               ('log2 area ratio ~ wall', 'log2_area_ratio'),
                               ('recovered radius ~ wall', 'rec_r_eq_m')]:
                rho, p, n = spearman(d.wall_actual_m.to_numpy(),
                                     d[col].to_numpy())
                rows.append(dict(env=env, channel=c, relation=label,
                                 spearman_rho=rho, p_value=p, n=n))
    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'correlations.csv')
    out.to_csv(path, index=False)
    print(f'  {path}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_dir',
                    default=f'{REPO}/data_cache/geometry_recovery')
    ap.add_argument('--out', dest='out_dir',
                    default=f'{HERE}/figures/geometry_recovery')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = pd.read_csv(f'{a.in_dir}/metrics.csv')
    print(f'{len(df)} trials | envs {sorted(set(df.env))} | '
          f'channels {sorted(set(df.channel))}')
    df = add_reference_shape(df, f'{a.in_dir}/masks.npz')

    print('figures:')
    fig_accuracy(df, a.out_dir)
    fig_transfer(df, a.out_dir)
    fig_error_modes(df, a.out_dir)
    fig_elongation(df, a.out_dir)
    fig_scale_vs_wall(df, a.out_dir)
    corr = correlations(df, a.out_dir)

    # ---- what the figures say, in text ----
    lines = []

    def block(t, s):
        lines.append(f'\n{t}\n' + '-' * len(t) + '\n' + s)

    piv = df.pivot_table(index='channel', columns='env', values='iou',
                         aggfunc='median').round(3)
    piv['mean'] = piv.mean(axis=1).round(3)
    block('Median IoU by channel (higher is better)',
          piv.sort_values('mean', ascending=False).to_string())

    block('Ranking of channels within each enclosure (best first)',
          '\n'.join(
              f'  {e:14s} ' + ' > '.join(
                  df[df.env == e].groupby('channel').iou.median()
                  .sort_values(ascending=False).index)
              for e in _envs(df)))

    block('Elongation: median reconstructed, and excess over the reference',
          df.pivot_table(index='channel', columns='env',
                         values=['rec_elongation', 'elongation_excess'],
                         aggfunc='median').round(2).to_string())

    sig = corr[(corr.p_value < 0.05) & corr.spearman_rho.notna()]
    block('Wall-distance relationships significant at p < 0.05',
          sig.round(3).to_string(index=False) if len(sig)
          else '  none')

    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(a.out_dir, 'figure_notes.txt'), 'w') as f:
        f.write(text + '\n')
    print(f'\nfigures -> {a.out_dir}')


if __name__ == '__main__':
    main()
