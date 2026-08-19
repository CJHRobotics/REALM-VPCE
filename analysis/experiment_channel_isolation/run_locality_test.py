"""Does a channel carry location information, or does our width measure fail?

The `hog` channel produced zero place fields: 99.3% of its candidate groups
responded across almost the whole floor. Two explanations fit that observation
and the pipeline cannot separate them.

  A. **The channel carries no location information.** Edge statistics are
     similar everywhere in this arena, so no grouping of them could ever be
     spatially localised.

  B. **Our width measure fails in high dimensions.** A group's width is a high
     percentile of within-group distances. In high dimensions pairwise
     distances concentrate, so that statistic drifts toward the same value for
     every group whether or not the group is spatially tight. `hog` has 30,240
     numbers per position, by far the most, and field count falls almost
     perfectly as channel dimensionality rises — which is what makes B a live
     possibility rather than a quibble.

Neither test below uses the clustering, the rules, or any threshold, so the
answer does not depend on the model whose behaviour is in question.

Test 1 — spatially defined groups (decisive)
--------------------------------------------
Take a disc of floor of radius r and measure how wide that group of positions
is in feature space, relative to the spread of two random positions. A patch of
floor is the tightest spatially coherent group that exists, so this is an upper
bound on how localised any appearance-based grouping could be.

    ratio near 0  -> compact floor patches are compact in feature space;
                     the channel carries location information, and a width
                     statistic that fails to see it is at fault (explanation B)
    ratio near 1  -> even a compact floor patch is as spread out as the whole
                     dataset; no clustering could localise, and the zero-field
                     result stands (explanation A)

Test 2 — nearest neighbours in feature space
--------------------------------------------
For each position, find the positions whose views are most similar and measure
how far away they are on the floor, against the distance between two random
positions. Also decodes position as the mean of those neighbours' positions and
reports the error, which puts the same quantity in metres.

Test 3 — distance concentration
--------------------------------
The relative contrast (std / mean) of pairwise distances, which is the standard
measure of the concentration effect and should fall as dimensionality rises if
B is operating.

Usage
    python run_locality_test.py [env_name] [--channels ...] [--radii 0.5,1,2]
                                [--k 16] [--subsample N] [--no-gpu] [--no-email]
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
DIMS = {'hog': 30240, 'color': 4096, 'spatial': 24576, 'lidar': 720,
        'visual': 58912, 'all': 59632}

# A patch ratio at or below this counts as "the channel localises"; at or above
# the upper bound it counts as "it does not". Between them the test is
# inconclusive and says so rather than guessing.
LOCALISES_BELOW = 0.60
FAILS_ABOVE = 0.90


# ------------------------------------------------------------------ report

class LocalityReport(ExperimentReport):
    """Emailed summary for the channel-locality test."""

    experiment = 'locality-test'

    @staticmethod
    def verdict(row):
        """Diagnose from decoding and patch width together.

        Decoding answers whether the location information is present at all.
        The patch ratio answers whether our width statistic can see it. The
        informative case is both at once: a channel that decodes position
        accurately while a compact patch of floor still spans most of the
        global distance range carries the information in a form the width
        statistic cannot express.
        """
        pr, dec, chance = row.patch_ratio_ref, row.decode_error_m, row.chance_dist_m
        if not np.isfinite(dec) or not np.isfinite(chance):
            return 'no data'
        decodes = dec < 0.25 * chance
        if not decodes:
            return 'no location information'
        if not np.isfinite(pr) or pr <= LOCALISES_BELOW:
            return 'localises; width statistic OK'
        if pr >= FAILS_ABOVE:
            return 'localises; width statistic BLIND'
        return 'localises; width statistic marginal'

    def title(self):
        m = self.metrics
        if m is None or not len(m):
            return 'no results'
        h = m[m.channel == 'hog']
        if not len(h):
            return f'{len(m)} channels tested'
        return f'hog {self.verdict(h.iloc[0])}'

    def body(self):
        m = self.metrics.copy()
        m['verdict'] = m.apply(self.verdict, axis=1)
        out = [self.section(
            'Question',
            'The hog channel produced zero place fields. Either it carries no\n'
            'location information, or our width statistic fails in 30,240\n'
            'dimensions. Neither test here uses the clustering or the rules.')]

        out.append(self.section(
            'Test 1 - spatially defined groups (decisive)',
            'Width of a disc of floor in feature space, over the width of the\n'
            'whole dataset. A floor patch is the tightest spatially coherent\n'
            'group there is, so this bounds how localised any appearance-based\n'
            f'grouping could be. <={LOCALISES_BELOW} means the channel localises and a\n'
            f'width statistic that misses it is at fault; >={FAILS_ABOVE} means no\n'
            'clustering could localise and the zero-field result stands.\n\n'
            + self.table(m, [c for c in ('channel', 'dim', 'patch_ratio_ref',
                                         'sigma_ratio_global', 'verdict')
                             if c in m.columns])))

        out.append(self.section(
            'Test 2 - nearest neighbours in feature space',
            'Floor distance to the k most similar views, over the distance\n'
            'between two random positions, plus the decoding error in metres.\n\n'
            + self.table(m, [c for c in ('channel', 'knn_floor_ratio',
                                         'knn_floor_median_m', 'decode_error_m',
                                         'chance_dist_m') if c in m.columns])))

        out.append(self.section(
            'Test 3 - distance concentration',
            'Relative contrast (std / mean) of pairwise distances. Falls as\n'
            'dimensionality rises when the concentration effect is operating.\n\n'
            + self.table(m, [c for c in ('channel', 'dim', 'rel_contrast')
                             if c in m.columns])))

        h = m[m.channel == 'hog']
        if len(h):
            r = h.iloc[0]
            v = self.verdict(r)
            reading = {
                'no location information':
                    'hog carries no usable location information: it cannot decode\n'
                    'position better than chance. The zero-field result stands as\n'
                    'reported, and the ordering by dimensionality is a property of\n'
                    'this arena rather than a measurement artifact.',
                'localises; width statistic OK':
                    'hog both carries location information and expresses it at a\n'
                    'usable scale, so neither explanation holds and the zero-field\n'
                    'result needs a third one. Check the candidate prefilter and\n'
                    'the size bounds.',
                'localises; width statistic BLIND':
                    f'hog decodes position to {r.decode_error_m:.2f} m against a chance\n'
                    f'level of {r.chance_dist_m:.2f} m, so the location information is\n'
                    'present. But a compact patch of floor is already\n'
                    f'{100*r.patch_ratio_ref:.0f}% as wide in feature space as the whole\n'
                    'dataset, so a percentile of within-group distances cannot\n'
                    'separate a tight group from a loose one.\n\n'
                    'The zero-field result is then an artifact of the width\n'
                    'statistic, not a property of the channel, and it biases every\n'
                    'high-dimensional channel in the same direction. The width\n'
                    'measure needs replacing before the channel comparison in the\n'
                    'report can stand. A scale-free alternative — a quantile of\n'
                    'distance-to-centroid relative to the group\'s own\n'
                    'distribution rather than an absolute percentile — is the\n'
                    'obvious candidate.',
                'localises; width statistic marginal':
                    f'hog decodes position to {r.decode_error_m:.2f} m, so the\n'
                    'information is present, and the patch ratio sits between the\n'
                    'thresholds. Widen the radius range and check whether the ratio\n'
                    'is scale-dependent before concluding.',
            }.get(v, v)
            out.append(self.section('Reading', reading))

        out.append(self.section('Attached', self.bullets([
            'L1 patch width against radius, per channel',
            'L2 neighbour distance and decoding error',
            'L3 concentration against dimensionality',
            'metrics.csv'])))
        return '\n'.join(out)

    def figures(self):
        return [os.path.join(self.fig_dir, f) for f in
                ('L1_patch_width.png', 'L2_neighbours.png', 'L3_concentration.png')]

    def data_files(self):
        return [os.path.join(self.out_dir, 'metrics.csv')]


# ------------------------------------------------------------------ tests

def patch_widths(D2, xy, radii, pctl, n_patches, rng, global_scale):
    """Test 1. Feature width of a disc of floor, over the global width.

    Discs are centred on randomly chosen sampled positions so every patch is
    genuinely occupied. Patches with too few members to support a percentile
    are skipped.
    """
    out = {}
    N = len(xy)
    for r in radii:
        vals = []
        for _ in range(n_patches):
            c = xy[rng.integers(0, N)]
            idx = np.flatnonzero(((xy - c) ** 2).sum(1) <= r * r)
            if len(idx) < 12:
                continue
            if len(idx) > 512:
                idx = rng.choice(idx, 512, replace=False)
            sub = D2[np.ix_(idx, idx)]
            iu = np.triu_indices(len(idx), k=1)
            vals.append(np.sqrt(np.percentile(sub[iu], pctl)))
        out[r] = float(np.median(vals) / global_scale) if vals else np.nan
    return out


def neighbour_locality(D2, xy, k, rng, n_query=4000):
    """Test 2. Floor distance to the k most similar views, and decoding error."""
    N = len(xy)
    q = rng.choice(N, min(n_query, N), replace=False)
    floor_d, decode_err = [], []
    for i in q:
        d = D2[i].copy()
        d[i] = np.inf
        nn = np.argpartition(d, k)[:k]
        floor_d.append(np.median(np.hypot(*(xy[nn] - xy[i]).T)))
        decode_err.append(float(np.hypot(*(xy[nn].mean(0) - xy[i]))))
    a, b = rng.integers(0, N, 200_000), rng.integers(0, N, 200_000)
    chance = float(np.median(np.hypot(*(xy[a] - xy[b]).T)))
    return (float(np.median(floor_d)), float(np.median(decode_err)), chance,
            np.asarray(floor_d))


def relative_contrast(D2, rng, n=2_000_000):
    """Test 3. std / mean of pairwise distances — the concentration measure."""
    N = D2.shape[0]
    i, j = rng.integers(0, N, n), rng.integers(0, N, n)
    keep = i != j
    d = np.sqrt(D2[i[keep], j[keep]])
    return float(d.std() / d.mean())


# ------------------------------------------------------------------ figures

def make_figures(per_radius, metrics, fig_dir, env_name, radii, nn_hist):
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 5))
    for cn, vals in per_radius.items():
        ax.plot(radii, [vals[r] for r in radii], marker='o', ms=6,
                color=CHANNEL_COLORS.get(cn), label=f'{cn} ({DIMS.get(cn,"?")}-d)')
    ax.axhspan(0, LOCALISES_BELOW, color='#10b981', alpha=.09)
    ax.axhspan(FAILS_ABOVE, 1.25, color='#ef4444', alpha=.09)
    ax.axhline(LOCALISES_BELOW, color='#10b981', ls='--', lw=1.2)
    ax.axhline(FAILS_ABOVE, color='#ef4444', ls='--', lw=1.2)
    ax.text(radii[-1], LOCALISES_BELOW - .04, 'localises', ha='right', va='top',
            fontsize=9, color='#047857')
    ax.text(radii[-1], FAILS_ABOVE + .03, 'no location information', ha='right',
            va='bottom', fontsize=9, color='#b91c1c')
    ax.set_xlabel('radius of the floor patch (m)')
    ax.set_ylabel('feature width of patch / global width')
    ax.set_ylim(0, 1.25)
    ax.set_title(f'L1  {env_name} | how tight is a patch of floor in feature space?')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/L1_patch_width.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for _, r in metrics.iterrows():
        ax.bar(r.channel, r.knn_floor_ratio, color=CHANNEL_COLORS.get(r.channel))
    ax.axhline(1.0, color='black', ls='--', lw=1.2, label='chance')
    ax.set_ylabel('floor distance to nearest views / chance')
    ax.set_title('(a) are similar views nearby?'); ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis='y')

    ax = axes[1]
    for _, r in metrics.iterrows():
        ax.bar(r.channel, r.decode_error_m, color=CHANNEL_COLORS.get(r.channel))
    if len(metrics):
        ax.axhline(metrics.chance_dist_m.iloc[0], color='black', ls='--', lw=1.2,
                   label='chance')
    ax.set_ylabel('position decoding error (m)')
    ax.set_title('(b) how well does the channel locate the agent?')
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis='y')

    ax = axes[2]
    for cn, v in nn_hist.items():
        ax.hist(v, bins=40, histtype='step', lw=2, density=True,
                color=CHANNEL_COLORS.get(cn), label=cn)
    ax.set_xlabel('floor distance to the most similar views (m)')
    ax.set_ylabel('density'); ax.set_title('(c) distribution')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle(f'L2  {env_name} | nearest neighbours in feature space',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/L2_neighbours.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5))
    for _, r in metrics.iterrows():
        ax.scatter(r.dim, r.rel_contrast, s=110, color=CHANNEL_COLORS.get(r.channel),
                   zorder=3)
        ax.annotate(r.channel, (r.dim, r.rel_contrast), textcoords='offset points',
                    xytext=(8, 4), fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('channel dimensionality')
    ax.set_ylabel('relative contrast  (std / mean of pairwise distance)')
    ax.set_title(f'L3  {env_name} | distance concentration against dimensionality')
    ax.grid(alpha=.3, which='both')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/L3_concentration.png', dpi=140, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


# ------------------------------------------------------------------ main

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('env', nargs='?', default='circ_lm8_r0')
    ap.add_argument('--channels', default='hog,color,spatial,lidar,visual,all')
    ap.add_argument('--radii', default='0.35,0.5,0.75,1.0,1.5,2.0')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--n-patches', type=int, default=400)
    ap.add_argument('--pctl', type=float, default=R.DEFAULT_CFG['SIGMA_PCTL'])
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
    radii = sorted(float(r) for r in args.radii.split(',') if r.strip())

    data_path = f'{REPO}/data/vpce/collect_data/{env_name}.h5'
    xml_path = f'{REPO}/simulation/worlds/environments/vpce/{env_name}.xml'
    out_dir = f'{REPO}/data_cache/locality_test/{env_name}'
    fig_dir = f'{HERE}/figures/locality_test/{env_name}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=' * 72)
    print(f'Channel locality test | env={env_name}')
    print(f'  channels : {channel_names}')
    print(f'  radii    : {radii} m   patches per radius: {args.n_patches}')
    print(f'  k        : {args.k}   width percentile: {args.pctl}')
    print('=' * 72)

    blocks, xy = ch.load_channel_blocks(
        data_path, lidar_max_range=LIDAR_MAX_RANGE,
        lidar_sentinel=LIDAR_SENTINEL, lidar_mask_channel=True)
    xml_root = ET.parse(xml_path).getroot() if os.path.exists(xml_path) else None

    if args.subsample and args.subsample < len(xy):
        sel = np.random.default_rng(args.seed).choice(len(xy), args.subsample,
                                                      replace=False)
        sel.sort()
        blocks = {k: v[sel] for k, v in blocks.items()}
        xy = xy[sel]
        print(f'  subsampled to {len(xy)} locations')

    device = R.pick_device(use_gpu=not args.no_gpu)
    rows, per_radius, nn_hist = [], {}, {}

    for cname in channel_names:
        keys = ch.CHANNEL_SETS.get(cname)
        if not keys or any(k not in blocks for k in keys):
            print(f'  !! skipping {cname}')
            continue
        print(f'\n----- channel: {cname} -----')
        rng = np.random.default_rng(args.seed)
        X = ch.assemble(blocks, keys, normalize=True)
        D2 = R.feature_sq_distances(X, device=device)

        t0 = time.time()
        global_scale = float(np.sqrt(R._median_offdiag(D2, rng)))
        pw = patch_widths(D2, xy, radii, args.pctl, args.n_patches, rng, global_scale)
        per_radius[cname] = pw
        print(f'  patch widths: ' + '  '.join(f'r={r:g}:{pw[r]:.3f}' for r in radii)
              + f'   [{time.time()-t0:.1f}s]')

        knn_m, dec_m, chance_m, hist = neighbour_locality(D2, xy, args.k, rng)
        nn_hist[cname] = hist
        rc = relative_contrast(D2, rng)
        print(f'  neighbours: median floor dist {knn_m:.2f} m '
              f'(chance {chance_m:.2f} m), decode error {dec_m:.2f} m')
        print(f'  relative contrast: {rc:.4f}')

        # The reference radius is the smallest one that still holds enough
        # positions to be measurable; it is the tightest bound available.
        ref = next((r for r in radii if np.isfinite(pw[r])), radii[0])
        row = dict(channel=cname, dim=X.shape[1], patch_radius_ref_m=ref,
                   patch_ratio_ref=pw[ref],
                   sigma_ratio_global=1.0,
                   knn_floor_median_m=knn_m, chance_dist_m=chance_m,
                   knn_floor_ratio=knn_m / chance_m if chance_m else np.nan,
                   decode_error_m=dec_m, rel_contrast=rc)
        row.update({f'patch_ratio_r{r:g}': pw[r] for r in radii})
        rows.append(row)
        del X, D2

    metrics = pd.DataFrame(rows)
    metrics.to_csv(f'{out_dir}/metrics.csv', index=False)
    with open(f'{out_dir}/config.json', 'w') as f:
        json.dump(dict(env=env_name, radii=radii, k=args.k, pctl=args.pctl,
                       n_patches=args.n_patches,
                       localises_below=LOCALISES_BELOW, fails_above=FAILS_ABOVE), f,
                  indent=2)
    print(f'\nMetrics -> {out_dir}/metrics.csv')
    if len(metrics):
        cols = ['channel', 'dim', 'patch_ratio_ref', 'knn_floor_ratio',
                'decode_error_m', 'rel_contrast']
        for line in metrics[cols].to_string(index=False).splitlines():
            print(f'[summary] {line}')

    if len(metrics):
        make_figures(per_radius, metrics, fig_dir, env_name, radii, nn_hist)
        print(f'Figures -> {fig_dir}')

    if not args.no_email:
        LocalityReport(env_name=env_name, exit_status=args.exit_status,
                       fig_dir=fig_dir, out_dir=out_dir, metrics=metrics).send()


if __name__ == '__main__':
    main()
