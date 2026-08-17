"""Figures for the channel-isolation experiment.

Every figure is tied to a rule so the rules are visible in the output
rather than only in the code:

    F1  channel x lambda field maps        Rules 4 and 7
    F2  lambda effect curves               Rule 4
    F3  anisotropy vs wall distance        Rule 7
    F4  fragmentation diagnostics          Rule 1
    F5  split-half reliability             Rule 2
    F6  funnel and band coverage           Rules 8, 9, 11, 12
    F7  field size vs wall distance        the Paper 1 measurement

Importable so figures can be regenerated from cached banks without
rerunning the agglomeration.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

CHANNEL_COLORS = {
    'hog': '#1f77b4', 'color': '#d62728', 'spatial': '#2ca02c',
    'lidar': '#9467bd', 'visual': '#ff7f0e', 'all': '#17becf',
}


def _save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    {os.path.basename(path)}')


def _draw_env(ax, env, xml_root):
    if env['is_circular']:
        th = np.linspace(0, 2 * np.pi, 240)
        ax.plot(env['env_cx'] + env['env_R'] * np.cos(th),
                env['env_cy'] + env['env_R'] * np.sin(th),
                color='black', lw=1.6, zorder=6)
    if xml_root is not None:
        for w in xml_root.findall('wall'):
            ax.plot([float(w.get('x1')), float(w.get('x2'))],
                    [float(w.get('y1')), float(w.get('y2'))],
                    color='black', lw=1.6, zorder=6)
        for lm in xml_root.findall('landmark'):
            x, y = float(lm.get('x')), float(lm.get('y'))
            t = float(lm.get('theta', 0)); hw = float(lm.get('width', 0.75)) / 2
            tx, ty = -np.sin(t), np.cos(t)
            c = (float(lm.get('red', 0)), float(lm.get('green', 0)), float(lm.get('blue', 0)))
            ax.plot([x - hw * tx, x + hw * tx], [y - hw * ty, y + hw * ty],
                    color=c, lw=4, solid_capstyle='butt', zorder=7)
    ax.set_xlim(env['x_min'] - 0.5, env['x_max'] + 0.5)
    ax.set_ylim(env['y_min'] - 0.5, env['y_max'] + 0.5)
    ax.set_aspect('equal')


def _draw_fields(ax, bank, cmap, norm):
    """Rule 7 — fields drawn as ellipses, not discs."""
    for _, r in bank.iterrows():
        c = cmap(norm(r['radius_env_m']))
        ax.add_patch(Ellipse((r['centroid_x'], r['centroid_y']),
                             width=2 * r['semi_major_m'], height=2 * r['semi_minor_m'],
                             angle=np.degrees(r['orientation_rad']),
                             facecolor=c, edgecolor=c, alpha=0.18, lw=0.9, zorder=3))
        ax.add_patch(Ellipse((r['centroid_x'], r['centroid_y']),
                             width=2 * r['semi_major_m'], height=2 * r['semi_minor_m'],
                             angle=np.degrees(r['orientation_rad']),
                             facecolor='none', edgecolor=c, alpha=0.8, lw=0.9, zorder=4))


# ------------------------------------------------------------------ F1

def f1_field_maps(banks, env, xml_root, fig_dir, env_name, channels, lambdas):
    """Rules 4 and 7 — the headline grid: one field map per (channel, lambda)."""
    nz = [b for b in banks.values() if len(b)]
    if not nz:
        return
    r_all = np.concatenate([b['radius_env_m'].to_numpy() for b in nz])
    cmap, norm = plt.get_cmap('plasma'), plt.Normalize(r_all.min(), r_all.max())

    nr, nc = len(channels), len(lambdas)
    fig, axes = plt.subplots(nr, nc, figsize=(3.5 * nc, 3.5 * nr), squeeze=False)
    for i, cn in enumerate(channels):
        for j, lam in enumerate(lambdas):
            ax = axes[i][j]
            bank = banks.get((cn, lam))
            _draw_env(ax, env, xml_root)
            if bank is not None and len(bank):
                _draw_fields(ax, bank, cmap, norm)
            n = 0 if bank is None else len(bank)
            ax.set_title(f'{cn} | $\\lambda$={lam:g}\n{n} fields', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02).set_label('field radius (m)')
    fig.suptitle(f'F1  {env_name} | place fields by channel (rows) and '
                 f'spatial weighting $\\lambda$ (columns)\n'
                 f'Rule 7: fields drawn as ellipses, not discs', y=1.005, fontsize=13)
    _save(fig, f'{fig_dir}/F1_field_maps.png')


# ------------------------------------------------------------------ F2

def f2_lambda_curves(metrics, fig_dir, env_name):
    """Rule 4 — what the positional term does, per channel."""
    panels = [('n_fields', 'fields admitted'),
              ('median_radius_m', 'median field radius (m)'),
              ('median_elongation', 'median elongation (a/b)'),
              ('frag_rate', 'Rule 1 rejection rate (fragmented)'),
              ('median_split_half_iou', 'median split-half IoU'),
              ('corr_radius_wall', 'corr(radius, wall distance)')]
    panels = [(c, l) for c, l in panels if c in metrics.columns]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    for k, (col, label) in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        for cn, grp in metrics.groupby('channel'):
            g = grp.sort_values('lam')
            ax.plot(g['lam'], g[col], marker='o', ms=5,
                    color=CHANNEL_COLORS.get(cn, None), label=cn)
        ax.set_xlabel('$\\lambda$  (positional weight)')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if col == 'corr_radius_wall':
            ax.axhline(0, color='black', lw=0.8, ls=':')
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    axes[0][0].legend(fontsize=8, ncol=2)
    fig.suptitle(f'F2  {env_name} | Rule 4: effect of the positional term on '
                 f'the fields that form', y=1.01, fontsize=13)
    fig.tight_layout()
    _save(fig, f'{fig_dir}/F2_lambda_effect.png')


# ------------------------------------------------------------------ F3

def f3_anisotropy(banks, env, fig_dir, env_name, channels, lam_ref):
    """Rule 7 — is shape heterogeneity there, and does it track the wall?

    Prediction: near a wall the view changes much faster moving toward or
    away from it than moving along it, so fields there should stretch
    tangentially. Fields in the open centre should be rounder.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for cn in channels:
        b = banks.get((cn, lam_ref))
        if b is None or not len(b):
            continue
        ax.scatter(b['dist_to_wall_m'], b['elongation'], s=14, alpha=0.45,
                   color=CHANNEL_COLORS.get(cn), label=cn)
        if len(b) >= 8:
            bins = np.linspace(0, b['dist_to_wall_m'].max(), 7)
            idx = np.digitize(b['dist_to_wall_m'], bins)
            xs = [b['dist_to_wall_m'][idx == i].mean() for i in range(1, len(bins))
                  if (idx == i).sum() >= 3]
            ys = [b['elongation'][idx == i].median() for i in range(1, len(bins))
                  if (idx == i).sum() >= 3]
            ax.plot(xs, ys, lw=2, color=CHANNEL_COLORS.get(cn))
    ax.axhline(1.0, color='black', ls=':', lw=1)
    ax.set_xlabel('distance to wall (m)'); ax.set_ylabel('elongation (a / b)')
    ax.set_title(f'(a) shape vs wall distance  ($\\lambda$={lam_ref:g})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for cn in channels:
        b = banks.get((cn, lam_ref))
        if b is None or len(b) < 3 or not env['is_circular']:
            continue
        phi = np.arctan2(b['centroid_y'] - env['env_cy'], b['centroid_x'] - env['env_cx'])
        align = np.abs(np.cos(b['orientation_rad'] - (phi + np.pi / 2)))
        near = b['dist_to_wall_m'] < 0.25 * env['env_R']
        if near.sum() >= 3:
            ax.hist(align[near], bins=np.linspace(0, 1, 11), histtype='step', lw=2,
                    density=True, color=CHANNEL_COLORS.get(cn), label=f'{cn} (near wall)')
    ax.axhline(1.0, color='black', ls=':', lw=1, label='uniform (no alignment)')
    ax.set_xlabel('|cos(major axis $-$ wall tangent)|   1 = along the wall')
    ax.set_ylabel('density')
    ax.set_title('(b) do near-wall fields follow the wall?')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    data, labels, colors = [], [], []
    for cn in channels:
        b = banks.get((cn, lam_ref))
        if b is not None and len(b):
            data.append(b['elongation'].to_numpy()); labels.append(cn)
            colors.append(CHANNEL_COLORS.get(cn, 'gray'))
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.axhline(1.0, color='black', ls=':', lw=1)
    ax.set_ylabel('elongation (a / b)')
    ax.set_title('(c) shape heterogeneity by channel')
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(f'F3  {env_name} | Rule 7: fields have shape, not just size',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    _save(fig, f'{fig_dir}/F3_anisotropy.png')


# ------------------------------------------------------------------ F4 / F5

def f4_contiguity(reports, metrics, fig_dir, env_name, channels, lam_ref):
    """Rule 1 — how much fragmentation is there, and does lambda cure it?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    for cn in channels:
        rep = reports.get((cn, lam_ref))
        if rep is None:
            continue
        v = rep['cand_cc_frac'][rep['cand_pass_size']]
        if len(v):
            ax.hist(v, bins=np.linspace(0, 1, 26), histtype='step', lw=2,
                    density=True, color=CHANNEL_COLORS.get(cn), label=cn)
    ax.axvline(0.80, color='red', ls='--', lw=1.4, label='Rule 1 threshold')
    ax.set_xlabel('largest connected component / mask area')
    ax.set_ylabel('density')
    ax.set_title(f'(a) field fragmentation ($\\lambda$={lam_ref:g})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for cn, grp in metrics.groupby('channel'):
        g = grp.sort_values('lam')
        ax.plot(g['lam'], 100 * g['frag_rate'], marker='o', ms=5,
                color=CHANNEL_COLORS.get(cn), label=cn)
    ax.set_xlabel('$\\lambda$'); ax.set_ylabel('% of size-passing candidates rejected')
    ax.set_title('(b) fragmentation vs positional weight')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'F4  {env_name} | Rule 1: a field is one connected patch of floor',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    _save(fig, f'{fig_dir}/F4_contiguity.png')


def f5_reliability(reports, metrics, fig_dir, env_name, channels, lam_ref):
    """Rule 2 — do the fields reproduce across independent halves?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    for cn in channels:
        rep = reports.get((cn, lam_ref))
        if rep is None:
            continue
        v = rep['cand_split_half_iou'][rep['cand_pass_size']]
        if len(v):
            ax.hist(v, bins=np.linspace(0, 1, 26), histtype='step', lw=2,
                    density=True, color=CHANNEL_COLORS.get(cn), label=cn)
    ax.axvline(0.40, color='red', ls='--', lw=1.4, label='Rule 2 threshold')
    ax.set_xlabel('split-half mask IoU'); ax.set_ylabel('density')
    ax.set_title(f'(a) reliability ($\\lambda$={lam_ref:g})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for cn, grp in metrics.groupby('channel'):
        g = grp.sort_values('lam')
        ax.plot(g['lam'], g['median_split_half_iou'], marker='o', ms=5,
                color=CHANNEL_COLORS.get(cn), label=cn)
    ax.axhline(0.40, color='red', ls='--', lw=1.2)
    ax.set_xlabel('$\\lambda$'); ax.set_ylabel('median split-half IoU')
    ax.set_title('(b) reliability vs positional weight')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'F5  {env_name} | Rule 2: a field has to show up twice',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    _save(fig, f'{fig_dir}/F5_reliability.png')


# ------------------------------------------------------------------ F6

def f6_funnel_coverage(reports, metrics, fig_dir, env_name, channels, lam_ref):
    """Rules 8, 9, 11, 12 — what each admission rule removed, and where the
    scale ladder ran out of floor to cover."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]
    stages = ['candidates', 'rule_8_9_size', 'rule_1_contiguity',
              'rule_2_reliability', 'rule_11_competition', 'rule_12_tiling']
    present = [s for s in stages if f'funnel_{s}' in metrics.columns]
    w, x = 0.8 / max(len(channels), 1), np.arange(len(present))
    for i, cn in enumerate(channels):
        row = metrics[(metrics['channel'] == cn) & (metrics['lam'] == lam_ref)]
        if not len(row):
            continue
        vals = [row.iloc[0][f'funnel_{s}'] for s in present]
        ax.bar(x + i * w, vals, width=w, color=CHANNEL_COLORS.get(cn), label=cn)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([s.replace('rule_', 'R').replace('_', ' ') for s in present],
                       rotation=30, ha='right', fontsize=8)
    ax.set_yscale('log'); ax.set_ylabel('candidates surviving')
    ax.set_title(f'(a) admission funnel ($\\lambda$={lam_ref:g})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    for cn in channels:
        rep = reports.get((cn, lam_ref))
        if rep is None or not rep['coverage']:
            continue
        bs = sorted(rep['coverage'])
        ax.plot(bs, [100 * rep['coverage'][b] for b in bs], marker='o', ms=5,
                color=CHANNEL_COLORS.get(cn), label=cn)
    ax.axhline(50, color='red', ls='--', lw=1.4, label='Rule 12 threshold')
    ax.set_xlabel('scale band (geometric, ratio 1.6)')
    ax.set_ylabel('% of environment covered')
    ax.set_title('(b) can each scale still tile the floor?')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for cn in channels:
        rep = reports.get((cn, lam_ref))
        if rep is None:
            continue
        v = rep['cand_r_eq'][rep['cand_pass_size']]
        if len(v):
            ax.hist(v, bins=np.logspace(np.log10(max(v.min(), 1e-3)),
                                        np.log10(v.max() * 1.02), 30),
                    histtype='step', lw=2, color=CHANNEL_COLORS.get(cn), label=cn)
    rep0 = next((r for r in reports.values() if r), None)
    if rep0:
        ax.axvline(rep0['r_min'], color='black', ls='--', lw=1.2, label='Rule 8 floor')
        ax.axvline(rep0['r_max'], color='black', ls=':', lw=1.2, label='Rule 9 ceiling')
    ax.set_xscale('log'); ax.set_xlabel('candidate field radius (m)')
    ax.set_ylabel('count')
    ax.set_title('(c) size window (Rules 8 / 9)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which='both')

    fig.suptitle(f'F6  {env_name} | admission rules and the scale ladder',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    _save(fig, f'{fig_dir}/F6_funnel_coverage.png')


# ------------------------------------------------------------------ F7

def f7_size_vs_wall(banks, env, fig_dir, env_name, channels, lam_ref):
    """The Paper 1 measurement: do fields get larger away from the walls?

    Wall distance is never an input to any rule, so whatever appears here
    came out of the features.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    for cn in channels:
        b = banks.get((cn, lam_ref))
        if b is None or not len(b):
            continue
        ax.scatter(b['dist_to_wall_m'], b['radius_env_m'], s=14, alpha=0.4,
                   color=CHANNEL_COLORS.get(cn), label=cn)
        if len(b) >= 8:
            bins = np.linspace(0, b['dist_to_wall_m'].max(), 7)
            idx = np.digitize(b['dist_to_wall_m'], bins)
            ok = [i for i in range(1, len(bins)) if (idx == i).sum() >= 3]
            ax.plot([b['dist_to_wall_m'][idx == i].mean() for i in ok],
                    [b['radius_env_m'][idx == i].median() for i in ok],
                    lw=2.2, color=CHANNEL_COLORS.get(cn))
    ax.axvspan(1.2, 1.5, color='gray', alpha=0.15, label='1.2-1.5 m (predicted onset)')
    ax.set_xlabel('distance to wall (m)'); ax.set_ylabel('field radius (m)')
    ax.set_title(f'(a) field size vs wall distance ($\\lambda$={lam_ref:g})')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    data, labels, colors = [], [], []
    for cn in channels:
        b = banks.get((cn, lam_ref))
        if b is not None and len(b):
            data.append(b['radius_env_m'].to_numpy()); labels.append(cn)
            colors.append(CHANNEL_COLORS.get(cn, 'gray'))
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.5)
    ax.set_ylabel('field radius (m)')
    ax.set_title('(b) size spectrum by channel')
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(f'F7  {env_name} | measured, never enforced: '
                 f'wall distance is not an input to any rule', y=1.02, fontsize=13)
    fig.tight_layout()
    _save(fig, f'{fig_dir}/F7_size_vs_wall.png')


# ------------------------------------------------------------------ driver

def make_all(banks, reports, metrics, env, xml_root, fig_dir, env_name,
             channels, lambdas):
    os.makedirs(fig_dir, exist_ok=True)
    channels = [c for c in channels if any(k[0] == c for k in banks)]
    lam_ref = 0.0 if 0.0 in lambdas else lambdas[0]
    print('  figures:')
    f1_field_maps(banks, env, xml_root, fig_dir, env_name, channels, lambdas)
    if len(metrics):
        f2_lambda_curves(metrics, fig_dir, env_name)
        f4_contiguity(reports, metrics, fig_dir, env_name, channels, lam_ref)
        f5_reliability(reports, metrics, fig_dir, env_name, channels, lam_ref)
        f6_funnel_coverage(reports, metrics, fig_dir, env_name, channels, lam_ref)
    f3_anisotropy(banks, env, fig_dir, env_name, channels, lam_ref)
    f7_size_vs_wall(banks, env, fig_dir, env_name, channels, lam_ref)
