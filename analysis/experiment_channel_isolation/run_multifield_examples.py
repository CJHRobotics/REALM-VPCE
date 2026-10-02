"""Multi-field responses: the clusters contiguity rejects, drawn on the floor.

The model admits a cluster as a place field only if its field is one connected
patch of floor (Rule 1: the largest connected patch must hold at least
CC_FRAC_MIN of the field). A cluster that fails is thrown away -- but what it
fails for is often interesting in its own right: its response is high in
several separate places. That is a multi-field place cell, which the
single-field model produces and then discards.

This draws those responses. For one arena and channel it rebuilds the field
library exactly as Experiment 2 does, keeps every candidate that passed the
size rule and failed contiguity, and for each one measures its **subfields**:
the connected patches of its field that are each at least as large as the
smallest admissible field (the Rule 8 floor). A patch smaller than that is
speckle, not a field, so it does not count.

Not every contiguity reject is a multi-field cell. Some are one main patch
with fragments around it. The report says how many of the rejects have two or
more subfields and how many do not, so the figures cannot be read as "every
reject looks like this".

Why it has to rebuild the library: a rejected cluster's response map is kept
nowhere. The bank holds the survivors, and the prune audit holds counts.

What comes out
--------------
`data_cache/multifield/<env>_<channel>_rejected.csv`   every contiguity reject:
    node, scale, members, field area, patches, subfields and their areas
`figures/multifield/<env>_<channel>/`
    M00_overview        the selected clusters on one sheet
    M01.. one figure per selected cluster: its response on the floor, as a
          fraction of its own peak, with the field's edge outlined
    (PNG at 300 dpi and PDF with editable text)

Which clusters are drawn: those with the most evenly sized subfields, taken
in turn from each scale so the figures are not all the finest fields.

Usage
    python run_multifield_examples.py                       # corr_lm8_l10w2, hog
    python run_multifield_examples.py --env circ_lm8_r6 --channel color --n 8
"""

import argparse
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
from matplotlib.patches import Circle
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch                                                # noqa: E402
import rules as R                                                    # noqa: E402
import run_scale_distribution as SD                                  # noqa: E402
from realm_tools.experiment_lib.reporting import ExperimentReport    # noqa: E402

PCTL, THRESH = SD.SETTINGS[0]
INK, INK_2, MUTED = SD.INK, SD.INK_2, SD.MUTED
# The response as a fraction of its own peak. A multi-hue ramp, pale where the
# response is low so the floor recedes, with its near-white end cut off so the
# lowest step still shows against the page.
RESP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'response', matplotlib.colormaps['YlGnBu'](np.linspace(0.06, 1.0, 256)))
SINGLE_W, DOUBLE_W = 3.35, 6.85          # journal column widths, inches


# ------------------------------------------------------------ the library

def build(env_name, cname, args):
    """Candidates and their responses, at Experiment 2's operating point.

    The same calls, seeds and configuration as SD.build_banks, so the
    candidates are the ones that experiment's library was admitted from.
    """
    data_path = f'{args.data_dir}/{env_name}.h5'
    if not os.path.exists(data_path):
        raise SystemExit(f'no dataset at {data_path}')
    root = ET.parse(f'{REPO}/simulation/worlds/environments/vpce/{env_name}.xml').getroot()
    blocks, xy = ch.load_channel_blocks(data_path)
    env = R.build_env(xy, root)
    base_C = R.resolve_cfg(dict(LAMBDA=0.0, RANDOM_SEED=args.seed,
                                USE_GPU=not args.no_gpu,
                                TILING_FRAC_MIN=SD.DEFAULT_TILING))
    C = R.resolve_cfg(dict(base_C, EXTENT_PCTL=PCTL, ACT_THRESH=THRESH))
    device = R.pick_device(use_gpu=not args.no_gpu)
    X = ch.assemble(blocks, ch.CHANNEL_SETS[cname], normalize=True)
    del blocks
    D2 = R.feature_sq_distances(X, device=device)
    feat_med = R._median_offdiag(D2, np.random.default_rng(base_C['RANDOM_SEED']))
    d2xy = ((xy[:3000, None, :] - xy[None, :3000, :]) ** 2).sum(-1)
    xy_med = float(np.median(d2xy[np.triu_indices(len(d2xy), 1)]))
    tree = R.build_tree(D2, xy, feat_med, xy_med, cfg=base_C)
    ctx = R.prepare_candidates(X, xy, env, D2, feat_med, xy_med, cfg=C,
                               device=device, tree=tree,
                               tag=f'{env_name}/{cname}')
    del X, D2
    bank, _, rep = R.admit_fields(ctx, cfg=C)
    geom = dict(env, landmarks=[(float(l.get('x')), float(l.get('y')))
                                for l in root.findall('landmark')])
    return ctx, rep, C, geom, len(bank)


def field_of(ctx, k, C):
    """One candidate's response on the floor and its field, as admission saw them."""
    G, occ = ctx['G'], ctx['occupied']
    grid = ctx['resp_all'][k]
    mask, peak = R.mask_from_grid(grid, G, occ, C)
    shown = R._fill_empty_bins(grid.reshape(G['gx'], G['gy']), occ, G['in_env'])
    frac = np.where(G['in_env'], shown / peak if peak > 0 else 0.0, np.nan)
    return frac, mask


def patches(mask, bin_area):
    """Connected patches of a field, largest first, with Rule 1's connectivity."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return lab, np.array([]), np.array([], dtype=int)
    sizes = np.bincount(lab.ravel())[1:] * bin_area
    order = np.argsort(sizes)[::-1]
    return lab, sizes[order], order + 1


def rejected_table(ctx, rep, C, env_name, cname):
    """Every candidate that passed size and failed contiguity, with its patches."""
    G = ctx['G']
    pass_size = np.asarray(rep['cand_pass_size'], bool)
    pass_cc = np.asarray(rep['cand_pass_contiguity'], bool)
    floor = float(rep['area_min'])
    rows = []
    for k in np.flatnonzero(pass_size & ~pass_cc):
        _, mask = field_of(ctx, k, C)
        _, sizes, _ = patches(mask, G['bin_area'])
        sub = sizes[sizes >= floor]
        nid = int(ctx['cand'][k])
        rows.append(dict(
            env=env_name, channel=cname, cand_index=int(k), node_id=nid,
            scale=int(rep['cand_band'][k]), n_members=int(ctx['count'][nid]),
            field_area_m2=float(rep['cand_area'][k]),
            largest_patch_share=float(rep['cand_cc_frac'][k]),
            n_patches=int(len(sizes)), n_subfields=int(len(sub)),
            subfield_1_m2=float(sub[0]) if len(sub) > 0 else np.nan,
            subfield_2_m2=float(sub[1]) if len(sub) > 1 else np.nan,
            subfield_3_m2=float(sub[2]) if len(sub) > 2 else np.nan,
            # 1 = two equal subfields; near 0 = one field and a scrap.
            balance=float(sub[1] / sub[0]) if len(sub) > 1 else 0.0,
            size_floor_m2=floor))
    return pd.DataFrame(rows)


def select(table, n, min_subfields):
    """The clusters to draw: most evenly split first, a scale at a time.

    Round-robin over scales, so the figures are not all scale 0 -- the finest
    scale holds most of any library and would otherwise fill every slot.
    """
    multi = table[table.n_subfields >= min_subfields]
    by_scale = {s: g.sort_values('balance', ascending=False).cand_index.tolist()
                for s, g in multi.groupby('scale')}
    picked = []
    while len(picked) < n and any(by_scale.values()):
        for s in sorted(by_scale):
            if by_scale[s] and len(picked) < n:
                picked.append(by_scale[s].pop(0))
    return table.set_index('cand_index').loc[picked].reset_index()


# ------------------------------------------------------------------ figures

def _arena(ax, geom):
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if geom.get('is_circular'):
        r, cx, cy = geom['env_R'], geom.get('env_cx', 0.0), geom.get('env_cy', 0.0)
        ax.add_patch(Circle((cx, cy), r, fill=False, color=INK_2, lw=0.8, zorder=10))
        x0, x1, y0, y1 = cx - r, cx + r, cy - r, cy + r
    else:
        x0, x1, y0, y1 = geom['x_min'], geom['x_max'], geom['y_min'], geom['y_max']
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   color=INK_2, lw=0.8, zorder=10))
    pad = 0.03 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    for lx, ly in geom.get('landmarks', []):
        ax.plot(lx, ly, 's', ms=2.2, color=INK, zorder=11)


def _scale_bar(ax, draw=True):
    """A scale bar under the arena. With draw=False it only reserves the room,
    so every panel of a sheet is the same size whether or not it carries one."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = x1 - x0
    length = max(v for v in (0.1, 0.2, 0.5, 1, 2, 5) if v <= 0.25 * span)
    by = y0 - 0.05 * span
    ax.set_ylim(by - 0.04 * span, y1)
    if not draw:
        return
    ax.plot([x0 + 0.02 * span, x0 + 0.02 * span + length], [by, by], color=INK,
            lw=1.4, solid_capstyle='butt', clip_on=False, zorder=12)
    ax.text(x0 + 0.02 * span + length / 2, by + 0.012 * span, f'{length:g} m',
            ha='center', va='bottom', fontsize=6, color=INK_2)


def _draw(ax, frac, mask, G, geom):
    """A response map with its field outlined."""
    _arena(ax, geom)
    pm = ax.pcolormesh(G['x_edges'], G['y_edges'], np.ma.masked_invalid(frac).T,
                       cmap=RESP_CMAP, vmin=0, vmax=1, rasterized=True, zorder=1)
    ax.contour(G['xc'], G['yc'], mask.T.astype(float), [0.5], colors=INK,
               linewidths=0.7, zorder=5)
    return pm


def _aspect(geom):
    if geom.get('is_circular'):
        return 1.0
    return (geom['x_max'] - geom['x_min']) / (geom['y_max'] - geom['y_min'])


def _save(fig, path, written):
    for ext, kw in (('png', dict(dpi=300)), ('pdf', {})):
        fig.savefig(f'{path}.{ext}', bbox_inches='tight', pad_inches=0.03, **kw)
    written.append(f'{path}.png')
    plt.close(fig)
    print(f'  {path}.png', flush=True)


def _label(row):
    n = int(row.n_subfields)
    return f'{n} subfield' + ('' if n == 1 else 's')


def fig_cluster(ctx, C, geom, row, path, written):
    """One rejected cluster: its response on the floor and its field's edge."""
    frac, mask = field_of(ctx, int(row.cand_index), C)
    asp = _aspect(geom)
    w = DOUBLE_W if asp >= 2.5 else SINGLE_W
    fig, ax = plt.subplots(figsize=(w, (w - 0.75) / asp * 1.1 + 0.45))
    pm = _draw(ax, frac, mask, ctx['G'], geom)
    _scale_bar(ax)
    ax.set_title(f'Cluster {int(row.node_id)}: {_label(row)}', loc='left',
                 fontweight='bold')
    cb = fig.colorbar(pm, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
    cb.set_label('response (fraction of peak)', fontsize=6.5)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0)
    _save(fig, path, written)


def fig_overview(ctx, C, geom, chosen, title, path, written):
    """The selected clusters on one sheet."""
    asp = _aspect(geom)
    ncol = 2 if asp >= 2.5 else min(4, max(1, len(chosen)))
    nrow = int(np.ceil(len(chosen) / ncol))
    pw = DOUBLE_W / ncol
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(DOUBLE_W, nrow * (pw / asp * 1.1 + 0.3) + 0.55))
    pm = None
    for ax in axes.ravel()[len(chosen):]:
        ax.axis('off')
    for ax, row in zip(axes.ravel(), chosen.itertuples()):
        frac, mask = field_of(ctx, int(row.cand_index), C)
        pm = _draw(ax, frac, mask, ctx['G'], geom)
        _scale_bar(ax, draw=ax is axes[0][0])
        ax.set_title(f'{int(row.node_id)} · scale {int(row.scale)} · {_label(row)}',
                     fontsize=6.3, pad=2)
    fig.suptitle(title, fontsize=8, fontweight='bold', y=1.0)
    fig.tight_layout(rect=(0, 0, 0.93, 0.97))
    if pm is not None:
        fig.canvas.draw()
        top, bot = axes[0][-1].get_position(), axes[-1][-1].get_position()
        mid = 0.5 * (top.y1 + bot.y0)
        cax = fig.add_axes([0.945, mid - 0.14, 0.012, 0.28])
        cb = fig.colorbar(pm, cax=cax)
        cb.set_label('response (fraction of peak)', fontsize=6.5)
        cb.set_ticks([0, 0.5, 1])
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_linewidth(0)
    _save(fig, path, written)


# ------------------------------------------------------------------- report

class MultifieldReport(ExperimentReport):
    """Emailed summary: how many contiguity rejects are multi-field, and the maps."""

    experiment = 'multifield-examples'

    def title(self):
        t = self.rejects
        return (f'{int((t.n_subfields >= 2).sum())} of {len(t)} contiguity '
                f'rejects have 2+ subfields')

    def body(self):
        t, out = self.rejects, []
        arena, chan = SD.arena_label(self.env), SD.channel_label(self.channel)
        n = len(t)
        multi = int((t.n_subfields >= 2).sum())
        out.append(self.section('What this is', (
            f'{arena}, {chan} channel, at the operating point (q = {PCTL}).\n\n'
            f'The model admits a cluster only if its field is one connected\n'
            f'patch of floor. {n} of the {self.n_sized} candidates that passed the\n'
            f'size rule failed that test ({self.n_admitted} fields were admitted).\n'
            f'These are their responses.\n\n'
            f'A SUBFIELD is a connected patch of the field at least as large as\n'
            f'the smallest admissible field ({t.size_floor_m2.iloc[0]:.3g} m^2\n'
            f'here). Smaller patches are speckle and are not counted.')
            if n else 'No candidate failed contiguity in this library.'))
        if not n:
            return '\n'.join(out)
        counts = (t.n_subfields.clip(upper=4).value_counts().sort_index()
                  .rename(index={4: '4 or more'}))
        tab = pd.DataFrame({'subfields': counts.index.astype(str),
                            'clusters': counts.values,
                            '% of rejects': 100.0 * counts.values / n})
        out.append(self.section(
            'How many are multi-field',
            self.table(tab, float_format='%.0f') + '\n\n'
            f'{multi} of {n} ({100.0 * multi / n:.0f}%) have two or more\n'
            f'subfields: separate places where the same cluster responds, each\n'
            f'large enough to be a field. The rest have one subfield or none --\n'
            f'a single patch with fragments, or only fragments -- and are NOT\n'
            f'multi-field cells. The figures show the first kind only.'))
        by = (t.assign(multi=t.n_subfields >= 2).groupby('scale')
              .agg(rejects=('multi', 'size'), multi_field=('multi', 'sum'))
              .reset_index())
        out.append(self.section('By scale (0 finest)', self.table(by)))
        c = self.chosen
        if len(c):
            show = c[['node_id', 'scale', 'n_members', 'n_subfields',
                      'subfield_1_m2', 'subfield_2_m2', 'field_area_m2',
                      'largest_patch_share']].copy()
            show.columns = ['cluster', 'scale', 'members', 'subfields',
                            'largest m^2', 'second m^2', 'field m^2',
                            'largest share']
            out.append(self.section(
                'The clusters drawn', self.table(show) + '\n\n'
                'Chosen as the most evenly split, a scale at a time. Each map is\n'
                'the cluster\'s response at every position, as a fraction of its\n'
                'own peak; the black line is the field\'s edge, at half the peak.'))
        out.append(self.section('What to keep in mind', (
            'These clusters are ones the model REJECTS. They show that the\n'
            'representation produces multi-field responses; the single-field\n'
            'admission rule then discards them. Admitting them would need a\n'
            'model in which one cell may own several fields.\n\n'
            'A cluster is summarized by one centroid in feature space. Two\n'
            'places give one cluster when their views are alike, so in an arena\n'
            'with symmetry -- and most of all without landmarks -- some\n'
            'subfields are the arena\'s symmetry rather than anything learned.')))
        return '\n'.join(out)

    def figures(self):
        return list(self.figure_paths)

    def data_files(self):
        return [self.csv] if os.path.exists(self.csv) else []


# --------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default='corr_lm8_l10w2')
    p.add_argument('--channel', default='hog')
    p.add_argument('--n', type=int, default=12,
                   help='how many clusters to draw, one figure each')
    p.add_argument('--min-subfields', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--no-email', action='store_true')
    p.add_argument('--data-dir', default=f'{REPO}/data/vpce/collect_data')
    p.add_argument('--cache-dir', default=f'{REPO}/data_cache/multifield')
    p.add_argument('--fig-dir', default=f'{HERE}/figures/multifield')
    return p.parse_args()


def main():
    args = parse_args()
    tag = f'{args.env}_{args.channel}'
    fig_dir = f'{args.fig_dir}/{tag}'
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    print('=' * 72)
    print(f'Multi-field examples | {args.env} / {args.channel}, '
          f'EXTENT_PCTL {PCTL}, ACT_THRESH {THRESH:g}')
    print('=' * 72, flush=True)

    t0 = time.time()
    ctx, rep, C, geom, n_admitted = build(args.env, args.channel, args)
    table = rejected_table(ctx, rep, C, args.env, args.channel)
    csv = f'{args.cache_dir}/{tag}_rejected.csv'
    table.to_csv(csv, index=False)
    n_sized = int(np.asarray(rep['cand_pass_size'], bool).sum())
    print(f'\n{len(table)} of {n_sized} sized candidates failed contiguity; '
          f'{int((table.n_subfields >= 2).sum()) if len(table) else 0} have 2+ '
          f'subfields  ({time.time() - t0:.0f}s)', flush=True)

    written = []
    chosen = select(table, args.n, args.min_subfields) if len(table) else table
    if len(chosen):
        print('\nfigures:', flush=True)
        with plt.rc_context(SD.PAPER_RC):
            fig_overview(ctx, C, geom, chosen,
                         f'Clusters rejected for contiguity: '
                         f'{SD.arena_label(args.env)}, '
                         f'{SD.channel_label(args.channel, long=False)}',
                         f'{fig_dir}/M00_overview', written)
            for i, row in enumerate(chosen.itertuples(), 1):
                fig_cluster(ctx, C, geom, row,
                            f'{fig_dir}/M{i:02d}_cluster_{int(row.node_id)}',
                            written)
    else:
        print(f'\nno cluster with {args.min_subfields}+ subfields to draw')

    r = MultifieldReport(env_name=args.env, out_dir=args.cache_dir,
                         fig_dir=fig_dir,
                         log_path=os.environ.get('REALM_LOG_PATH'))
    r.rejects, r.chosen, r.env, r.channel = table, chosen, args.env, args.channel
    r.n_sized, r.n_admitted, r.figure_paths, r.csv = n_sized, n_admitted, written, csv
    print('\n' + r.compose(), flush=True)
    if not args.no_email:
        r.send()
    return 0


if __name__ == '__main__':
    sys.exit(main())
