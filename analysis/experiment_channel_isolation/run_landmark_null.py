"""Do landmarks organise where our fields sit, or how large they are?

Eliav et al. 2021 tested this five ways in a 200 m tunnel and found nothing:
field peaks uniformly distributed; distance from peak to nearest landmark
indistinguishable from shuffled (P >= 0.18); gaps between fields exponentially
distributed; field size no different near versus far from a landmark
(P = 0.80, 0.25); and no relation between interlandmark distance and field
size. It is the strongest negative result in that literature.

Our model is driven purely by view appearance, so it has an obvious route to
over-predicting landmark dependence: nothing in the architecture prevents
fields from forming where the view changes fastest, which is near a landmark.
This is the sharpest available test of whether the mechanism produces a
spatial code or a landmark-proximity detector.

Run over circ_lm4/6/8/10_r0, which vary interlandmark spacing from 15.7 m to
6.3 m by construction. `lidar` is the control: it cannot see the panels, so
any landmark relationship it shows is geometry rather than appearance.

Three departures from a literal reading of Eliav, each because the test does
not transfer unchanged from a 1D tunnel to a 2D disc:

**Two shuffle nulls, not one.** Landmarks sit on the wall, so a model with any
wall-proximity bias -- which Harland reports and which we have measured --
would register as landmark structure under a null that redraws centres
uniformly over the floor. `uniform` asks whether fields are spatially
structured at all; `rotation` keeps each field's radius and randomises its
angle, isolating alignment to landmarks specifically. The second is the one
that answers the question.

**Angular gaps, not 2D ones.** Nearest-neighbour distances under a 2D Poisson
process are Rayleigh, not exponential, so Eliav's exponential test applies to
the angular coordinate -- which is also the coordinate the landmarks are
arranged along. 2D nearest-neighbour distance is reported too, against the
shuffle rather than against a distribution.

**Two near/far cuts.** Half the interlandmark spacing is 7.85 m for lm4, most
of a 10 m arena, against 3.14 m for lm10; the per-environment cut is
principled but not comparable across environments, so a fixed cut is reported
alongside it.

LAMBDA is 0 throughout: a non-zero spatial weight injects spatial structure
into the clustering and would confound the entire experiment.

Usage
    python run_landmark_null.py [--envs a,b] [--channels ...] [--use-cache]
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
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch
import rules as R
from realm_tools.experiment_lib.reporting import ExperimentReport

ENVS = ['circ_lm4_r0', 'circ_lm6_r0', 'circ_lm8_r0', 'circ_lm10_r0']
CHANNELS = ['hog', 'color', 'spatial', 'lidar', 'visual', 'all']
CHANNEL_COLORS = {'hog': '#1f77b4', 'color': '#d62728', 'spatial': '#2ca02c',
                  'lidar': '#9467bd', 'visual': '#ff7f0e', 'all': '#17becf'}
N_SHUFFLE = 2000
FIXED_CUT_M = 2.0          # comparable near/far cut across environments


# ------------------------------------------------------------------ geometry

def landmarks_from_xml(root):
    """(n, 2) landmark positions."""
    return np.array([[float(l.get('x')), float(l.get('y'))]
                     for l in root.findall('landmark')], dtype=float)


def nearest_landmark_dist(xy, lms):
    d = np.hypot(xy[:, 0][:, None] - lms[None, :, 0],
                 xy[:, 1][:, None] - lms[None, :, 1])
    return d.min(axis=1)


def interlandmark_spacing(lms, env):
    """Arc length between neighbouring landmarks on the wall."""
    a = np.sort(np.arctan2(lms[:, 1], lms[:, 0]) % (2 * np.pi))
    gaps = np.diff(np.concatenate([a, a[:1] + 2 * np.pi]))
    return float(np.median(gaps) * env['env_R'])


# --------------------------------------------------------------------- nulls

def shuffle_centres(cx, cy, env, kind, rng):
    """Null field centres.

    uniform  -- redrawn over the walkable disc. Asks whether the fields are
                spatially structured at all, and will flag a wall-proximity
                bias as landmark structure because landmarks sit on the wall.
    rotation -- each field keeps its radius and is given a new angle. Removes
                any radial structure from the comparison, so what remains is
                alignment to landmark *angles*.
    """
    n = len(cx)
    if kind == 'uniform':
        a = rng.uniform(0, 2 * np.pi, n)
        r = env['env_R'] * np.sqrt(rng.uniform(size=n))
        return r * np.cos(a), r * np.sin(a)
    r = np.hypot(cx - env['env_cx'], cy - env['env_cy'])
    a = rng.uniform(0, 2 * np.pi, n)
    return env['env_cx'] + r * np.cos(a), env['env_cy'] + r * np.sin(a)


def perm_p(obs, null):
    """Two-sided permutation p from a null sample of a scalar statistic."""
    null = np.asarray(null)
    k = min((null <= obs).mean(), (null >= obs).mean())
    return float(min(1.0, 2 * k))


# --------------------------------------------------------------------- tests

def run_tests(bank, lms, env, rng):
    """Eliav's five tests on one (environment, channel) field library."""
    out = {}
    cx, cy = bank.centroid_x.to_numpy(), bank.centroid_y.to_numpy()
    size = bank.radius_env_m.to_numpy()
    n = len(bank)
    out['n_fields'] = n
    if n < 20:
        out['note'] = 'too few fields for the tests'
        return out

    theta = np.arctan2(cy - env['env_cy'], cx - env['env_cx']) % (2 * np.pi)
    d_lm = nearest_landmark_dist(np.column_stack([cx, cy]), lms)
    spacing = interlandmark_spacing(lms, env)
    out['interlandmark_m'] = spacing
    out['median_dist_lm_m'] = float(np.median(d_lm))

    # --- 1. are field centres uniform in angle? --------------------------
    # Rayleigh tests for a single preferred direction; KS tests the whole
    # distribution, which is what catches clustering at all n landmarks at
    # once -- a Rayleigh test is blind to that, since n evenly spaced clusters
    # have no net resultant.
    Cbar, Sbar = np.cos(theta).mean(), np.sin(theta).mean()
    Rbar = float(np.hypot(Cbar, Sbar))
    out['rayleigh_R'] = Rbar
    out['rayleigh_p'] = float(np.exp(np.sqrt(1 + 4 * n + 4 * n ** 2 *
                                             (1 - Rbar ** 2)) - (1 + 2 * n)))
    out['ks_angle_uniform_p'] = float(
        stats.kstest(theta / (2 * np.pi), 'uniform').pvalue)
    # Folded onto one landmark sector: clustering at every landmark shows here.
    k = len(lms)
    folded = (theta * k / (2 * np.pi)) % 1.0
    out['ks_folded_uniform_p'] = float(stats.kstest(folded, 'uniform').pvalue)

    # --- 2. distance to nearest landmark against the nulls ----------------
    for kind in ('uniform', 'rotation'):
        stat_null = np.empty(N_SHUFFLE)
        ks_null = np.empty(N_SHUFFLE)
        for i in range(N_SHUFFLE):
            sx, sy = shuffle_centres(cx, cy, env, kind, rng)
            sd = nearest_landmark_dist(np.column_stack([sx, sy]), lms)
            stat_null[i] = sd.mean()
            ks_null[i] = stats.ks_2samp(sd, d_lm).statistic
        out[f'mean_dist_lm_{kind}_null'] = float(stat_null.mean())
        out[f'p_dist_lm_{kind}'] = perm_p(float(d_lm.mean()), stat_null)
        # Effect size in units of the null's own spread, so a p-value driven
        # purely by a large field count is visible as a small effect.
        out[f'z_dist_lm_{kind}'] = float(
            (d_lm.mean() - stat_null.mean()) / (stat_null.std() + 1e-12))

    # --- 3. gaps between neighbouring fields -----------------------------
    a = np.sort(theta)
    gaps = np.diff(np.concatenate([a, a[:1] + 2 * np.pi]))
    gaps = gaps[gaps > 0]
    if len(gaps) > 5:
        # Fitted rate, location fixed at 0: an exponential gap distribution is
        # the signature of no spatial structure.
        loc, scale = stats.expon.fit(gaps, floc=0)
        out['ks_gaps_expon_p'] = float(
            stats.kstest(gaps, 'expon', args=(loc, scale)).pvalue)
        out['gap_cv'] = float(gaps.std() / (gaps.mean() + 1e-12))
    nn = _nn_dist(cx, cy)
    nn_null = np.empty(N_SHUFFLE // 10)
    for i in range(len(nn_null)):
        sx, sy = shuffle_centres(cx, cy, env, 'rotation', rng)
        nn_null[i] = _nn_dist(sx, sy).mean()
    out['mean_nn_m'] = float(nn.mean())
    out['p_nn_rotation'] = perm_p(float(nn.mean()), nn_null)

    # --- 4. field size near versus far from a landmark -------------------
    for label, cut in (('half_spacing', 0.5 * spacing), ('fixed', FIXED_CUT_M)):
        near, far = size[d_lm < cut], size[d_lm >= cut]
        out[f'cut_{label}_m'] = float(cut)
        out[f'n_near_{label}'] = int(len(near))
        if len(near) >= 5 and len(far) >= 5:
            out[f'size_near_{label}'] = float(np.median(near))
            out[f'size_far_{label}'] = float(np.median(far))
            out[f'p_size_{label}'] = float(
                stats.mannwhitneyu(near, far, alternative='two-sided').pvalue)

    # --- 5. field size against distance to the nearest landmark ----------
    rho, p = stats.spearmanr(d_lm, size)
    out['rho_size_dist'] = float(rho)
    out['p_size_dist'] = float(p)
    return out


def _nn_dist(cx, cy):
    d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


# ------------------------------------------------------------ field libraries

def build_bank(env_name, cname, blocks, xy, env, C, device, out_dir, use_cache,
               verbose=True):
    """The admitted field library for one channel, built or loaded."""
    cache = f'{out_dir}/{env_name}/{cname}_bank.csv'
    if use_cache and os.path.exists(cache):
        print(f'  [{cname}] cached bank', flush=True)
        return pd.read_csv(cache)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    t0 = time.time()
    X = ch.assemble(blocks, ch.CHANNEL_SETS[cname], normalize=True)
    D2 = R.feature_sq_distances(X, device=device, verbose=verbose)
    rng = np.random.default_rng(C['RANDOM_SEED'])
    feat_med = R._median_offdiag(D2, rng)
    d2xy = ((xy[:3000, None, :] - xy[None, :3000, :]) ** 2).sum(-1)
    xy_med = float(np.median(d2xy[np.triu_indices(len(d2xy), 1)]))
    ctx = R.prepare_candidates(X, xy, env, D2, feat_med, xy_med, cfg=C,
                               device=device, tag=f'{env_name}/{cname}',
                               verbose=verbose)
    bank, _, rep = R.admit_fields(ctx, cfg=C, verbose=verbose)
    bank.to_csv(cache, index=False)
    with open(f'{out_dir}/{env_name}/{cname}_report.json', 'w') as f:
        json.dump({k: v for k, v in rep.items()
                   if not isinstance(v, np.ndarray)}, f, indent=2, default=float)
    print(f'  [{cname}] {len(bank)} fields in {time.time()-t0:.0f}s', flush=True)
    del X, D2
    return bank


# ------------------------------------------------------------------- figures

def _save(fig, fig_dir, name):
    p = os.path.join(fig_dir, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}', flush=True)


def fig_field_maps(banks, lms_by_env, envs, chans, fig_dir):
    """Where the fields are, with the landmarks drawn on. The picture that
    makes a positive result obvious before any test is read."""
    fig, axes = plt.subplots(len(chans), len(envs),
                             figsize=(3.1 * len(envs), 3.1 * len(chans)),
                             squeeze=False, layout='constrained')
    for j, e in enumerate(envs):
        lms, R_ = lms_by_env[e]
        for i, c in enumerate(chans):
            ax = axes[i][j]
            b = banks.get((e, c))
            ax.add_patch(plt.Circle((0, 0), R_, fill=False, color='k', lw=1))
            if b is not None and len(b):
                ax.scatter(b.centroid_x, b.centroid_y, s=4, alpha=0.35,
                           color=CHANNEL_COLORS.get(c), edgecolors='none')
            ax.scatter(lms[:, 0], lms[:, 1], marker='s', s=44, color='k',
                       zorder=5, label='landmark' if i == j == 0 else None)
            ax.set_aspect('equal')
            ax.set_xlim(-R_ * 1.1, R_ * 1.1)
            ax.set_ylim(-R_ * 1.1, R_ * 1.1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(e.replace('circ_', '').replace('_r0', ''))
            if j == 0:
                ax.set_ylabel(c, fontsize=10)
    fig.suptitle('Field centres and landmark positions')
    _save(fig, fig_dir, 'L1_field_maps.png')


def fig_angle_cdf(banks, lms_by_env, envs, chans, fig_dir):
    """Test 1. Cumulative distribution of field angle; the diagonal is
    uniform. Landmark angles marked."""
    fig, axes = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 3.8),
                             squeeze=False, layout='constrained')
    for ax, e in zip(axes[0], envs):
        lms, _ = lms_by_env[e]
        for c in chans:
            b = banks.get((e, c))
            if b is None or not len(b):
                continue
            th = np.sort(np.arctan2(b.centroid_y, b.centroid_x) % (2 * np.pi))
            ax.plot(th, np.arange(1, len(th) + 1) / len(th), lw=1.3,
                    color=CHANNEL_COLORS.get(c), label=c)
        ax.plot([0, 2 * np.pi], [0, 1], 'k--', lw=1, label='uniform')
        for a in np.arctan2(lms[:, 1], lms[:, 0]) % (2 * np.pi):
            ax.axvline(a, color='grey', lw=0.6, alpha=0.6)
        ax.set_xlabel('field angle (rad)')
        ax.set_ylabel('cumulative fraction')
        ax.set_title(e.replace('circ_', '').replace('_r0', ''))
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle('Test 1 — angular position of field centres '
                 '(grey lines: landmarks)')
    _save(fig, fig_dir, 'L2_angle_cdf.png')


def fig_dist_null(banks, lms_by_env, envs, chans, fig_dir, rng):
    """Test 2. Observed distance-to-nearest-landmark against both nulls."""
    fig, axes = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 3.8),
                             squeeze=False, layout='constrained')
    for ax, e in zip(axes[0], envs):
        lms, R_ = lms_by_env[e]
        env = dict(env_R=R_, env_cx=0.0, env_cy=0.0)
        for c in chans:
            b = banks.get((e, c))
            if b is None or not len(b):
                continue
            d = nearest_landmark_dist(
                np.column_stack([b.centroid_x, b.centroid_y]), lms)
            ax.hist(d, bins=30, density=True, histtype='step', lw=1.3,
                    color=CHANNEL_COLORS.get(c), label=c)
        n = max((len(banks[(e, c)]) for c in chans if (e, c) in banks), default=0)
        if n:
            for kind, ls in (('uniform', ':'), ('rotation', '--')):
                sx, sy = shuffle_centres(np.zeros(n * 20), np.zeros(n * 20),
                                         env, 'uniform', rng) if kind == 'uniform' \
                    else shuffle_centres(
                        np.concatenate([banks[(e, c)].centroid_x.to_numpy()
                                        for c in chans if (e, c) in banks]),
                        np.concatenate([banks[(e, c)].centroid_y.to_numpy()
                                        for c in chans if (e, c) in banks]),
                        env, 'rotation', rng)
                sd = nearest_landmark_dist(np.column_stack([sx, sy]), lms)
                ax.hist(sd, bins=30, density=True, histtype='step', lw=1.6,
                        color='k', ls=ls, label=f'{kind} null')
        ax.set_xlabel('distance to nearest landmark (m)')
        ax.set_ylabel('density')
        ax.set_title(e.replace('circ_', '').replace('_r0', ''))
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle('Test 2 — distance to the nearest landmark against the nulls')
    _save(fig, fig_dir, 'L3_dist_to_landmark.png')


def fig_size_vs_dist(banks, lms_by_env, envs, chans, fig_dir):
    """Tests 4 and 5. Field size against landmark distance."""
    fig, axes = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 3.8),
                             squeeze=False, layout='constrained')
    for ax, e in zip(axes[0], envs):
        lms, _ = lms_by_env[e]
        for c in chans:
            b = banks.get((e, c))
            if b is None or not len(b):
                continue
            d = nearest_landmark_dist(
                np.column_stack([b.centroid_x, b.centroid_y]), lms)
            q = pd.qcut(d, min(8, max(2, len(d) // 40)), duplicates='drop')
            g = pd.DataFrame({'d': d, 's': b.radius_env_m.to_numpy()}).groupby(q,
                observed=True)
            ax.plot(g.d.median(), g.s.median(), 'o-', ms=4,
                    color=CHANNEL_COLORS.get(c), label=c)
        ax.set_xlabel('distance to nearest landmark (m)')
        ax.set_ylabel('median field radius (m)')
        ax.set_title(e.replace('circ_', '').replace('_r0', ''))
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle('Tests 4-5 — field size against distance to the nearest '
                 'landmark')
    _save(fig, fig_dir, 'L4_size_vs_distance.png')


def fig_spacing(results, fig_dir):
    """Test 5 across environments: does field size track landmark spacing?"""
    fig, ax = plt.subplots(figsize=(5.6, 4.2), layout='constrained')
    for c in sorted(set(results.channel)):
        d = results[results.channel == c].sort_values('interlandmark_m')
        if 'median_size_m' in d and d.median_size_m.notna().any():
            ax.plot(d.interlandmark_m, d.median_size_m, 'o-', ms=5,
                    color=CHANNEL_COLORS.get(c), label=c)
    ax.set_xlabel('interlandmark spacing (m)')
    ax.set_ylabel('median field radius (m)')
    ax.set_title('Test 5 — field size against interlandmark spacing\n'
                 'lm10 (6.3 m) to lm4 (15.7 m)')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, fig_dir, 'L5_size_vs_spacing.png')


# -------------------------------------------------------------------- report

ALPHA = 0.05


class LandmarkNullReport(ExperimentReport):
    """Emailed summary. Leads with the verdict, because the whole experiment
    is a yes/no about whether the model is a landmark-proximity detector."""

    experiment = 'landmark-null'

    def title(self):
        if self.results is None or not len(self.results):
            return 'no results'
        n_sig = int(self.results.n_landmark_effects.sum())
        n_tot = int(self.results.n_tests_run.sum())
        if n_sig == 0:
            return f'NULL — no landmark structure in {len(self.results)} runs'
        return f'{n_sig}/{n_tot} landmark tests significant'

    def body(self):
        r = self.results
        if r is None or not len(r):
            return 'No field libraries were produced.'
        S = self.section
        out = []

        n_sig = int(r.n_landmark_effects.sum())
        lidar = r[r.channel == 'lidar']
        vis = r[r.channel != 'lidar']
        verdict = [
            f'{n_sig} of {int(r.n_tests_run.sum())} landmark tests reached '
            f'p < {ALPHA}, across {len(r)} environment x channel runs.', '']
        if n_sig == 0:
            verdict += [
                'NULL RESULT. Fields are not organised by landmarks. This '
                'matches Eliav and is the strong outcome: appearance-driven '
                'fields that do not concentrate on the appearance sources.']
        else:
            verdict += [
                'POSITIVE RESULT. Report this as a divergence from the one '
                'direct measurement available, not as a finding. Eliav\'s own '
                'caveat applies in reverse: their landmarks may have had low '
                'salience for bats and a 200 m 1D tunnel is not a 10 m arena, '
                'so a positive result here is informative but not fatal.',
                '',
                f'Visual channels: {int(vis.n_landmark_effects.sum())} '
                f'significant of {int(vis.n_tests_run.sum())}. '
                f'lidar (control, cannot see the panels): '
                f'{int(lidar.n_landmark_effects.sum())} of '
                f'{int(lidar.n_tests_run.sum())}.',
                'If lidar shows as much structure as the visual channels, the '
                'effect is arena geometry rather than landmark appearance.']
        out.append(S('VERDICT', '\n'.join(verdict)))

        cols = ['env', 'channel', 'n_fields', 'interlandmark_m',
                'ks_folded_uniform_p', 'p_dist_lm_rotation', 'z_dist_lm_rotation',
                'ks_gaps_expon_p', 'p_size_half_spacing', 'p_size_dist',
                'n_landmark_effects']
        cols = [c for c in cols if c in r.columns]
        out.append(S('Per environment and channel', self.table(r[cols])))

        out.append(S('The five tests', '\n'.join([
            '1. angular uniformity   ks_folded_uniform_p -- folded onto one',
            '   landmark sector, so clustering at every landmark shows up;',
            '   a Rayleigh test alone is blind to n evenly spaced clusters.',
            '2. distance to landmark p_dist_lm_rotation -- against a null that',
            '   keeps each field radius and randomises its angle, isolating',
            '   landmark alignment from any wall-proximity bias. The uniform',
            '   null (p_dist_lm_uniform) is reported too but conflates them.',
            '3. gaps                 ks_gaps_expon_p -- angular gaps against an',
            '   exponential, the no-structure signature. 2D nearest-neighbour',
            '   distance is p_nn_rotation.',
            '4. size near vs far     p_size_half_spacing (per-environment cut)',
            f'   and p_size_fixed (a common {FIXED_CUT_M:g} m cut).',
            '5. size vs distance     p_size_dist, Spearman rho_size_dist;',
            '   across environments, L5 plots size against spacing.',
            '',
            'z_dist_lm_rotation is the effect size in units of the null\'s own',
            'spread. With ~1000 fields a tiny effect reaches significance, so',
            'read it beside the p-value: |z| below about 2 is negligible',
            'however small p is.'])))
        return '\n'.join(out)


# ----------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--envs', default=','.join(ENVS))
    p.add_argument('--channels', default=','.join(CHANNELS))
    p.add_argument('--lam', type=float, default=0.0)
    p.add_argument('--subsample', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--use-cache', action='store_true',
                   help='reuse field libraries already built by this script')
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--no-email', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    envs = [e.strip() for e in args.envs.split(',') if e.strip()]
    chans = [c.strip() for c in args.channels.split(',') if c.strip()]
    rng = np.random.default_rng(args.seed)

    out_dir = f'{REPO}/data_cache/landmark_null'
    fig_dir = f'{HERE}/figures/landmark_null'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    C = R.resolve_cfg(dict(LAMBDA=args.lam, RANDOM_SEED=args.seed,
                           USE_GPU=not args.no_gpu))
    device = R.pick_device(use_gpu=not args.no_gpu)

    print('=' * 72)
    print('Landmark null | do landmarks organise where fields sit or how big')
    print(f'  envs     : {envs}')
    print(f'  channels : {chans}')
    print(f'  LAMBDA   : {C["LAMBDA"]}  (0 = feature only; a spatial weight '
          f'would confound this test)')
    print('=' * 72, flush=True)

    banks, lms_by_env, rows = {}, {}, []
    for e in envs:
        data_path = f'{REPO}/data/vpce/collect_data/{e}.h5'
        xml_path = f'{REPO}/simulation/worlds/environments/vpce/{e}.xml'
        if not os.path.exists(data_path):
            print(f'\n[{e}] no dataset -- skipping', flush=True)
            continue
        print(f'\n===== {e} =====', flush=True)
        blocks, xy = ch.load_channel_blocks(data_path)
        if args.subsample and args.subsample < len(xy):
            sel = np.sort(rng.choice(len(xy), args.subsample, replace=False))
            blocks, xy = {k: v[sel] for k, v in blocks.items()}, xy[sel]
            print(f'  subsampled to {len(xy)}')
        root = ET.parse(xml_path).getroot()
        env = R.build_env(xy, root)
        lms = landmarks_from_xml(root)
        lms_by_env[e] = (lms, env['env_R'])
        print(f'  {len(lms)} landmarks, spacing '
              f'{interlandmark_spacing(lms, env):.2f} m', flush=True)

        for c in chans:
            bank = build_bank(e, c, blocks, xy, env, C, device, out_dir,
                              args.use_cache)
            banks[(e, c)] = bank
            res = run_tests(bank, lms, env, rng)
            res.update(env=e, channel=c,
                       median_size_m=float(bank.radius_env_m.median())
                       if len(bank) else np.nan)
            # Count only the landmark-specific tests, and require an effect
            # size as well as a p-value: with ~1000 fields, significance alone
            # is nearly guaranteed and says little.
            checks = [('ks_folded_uniform_p', None),
                      ('p_dist_lm_rotation', 'z_dist_lm_rotation'),
                      ('ks_gaps_expon_p', None),
                      ('p_size_half_spacing', None),
                      ('p_size_dist', 'rho_size_dist')]
            n_run = sum(1 for k, _ in checks if k in res)
            n_sig = 0
            for k, eff in checks:
                if k not in res or not (res[k] < ALPHA):
                    continue
                if eff == 'z_dist_lm_rotation' and abs(res.get(eff, 0)) < 2:
                    continue
                if eff == 'rho_size_dist' and abs(res.get(eff, 0)) < 0.1:
                    continue
                n_sig += 1
            res['n_tests_run'], res['n_landmark_effects'] = n_run, n_sig
            rows.append(res)
            print(f'    -> {n_sig}/{n_run} landmark tests significant',
                  flush=True)
        del blocks

    if not rows:
        print('\nNo results.')
        return 1

    results = pd.DataFrame(rows)
    front = ['env', 'channel', 'n_fields', 'interlandmark_m',
             'n_landmark_effects', 'n_tests_run']
    results = results[front + [c for c in results.columns if c not in front]]
    results.to_csv(f'{out_dir}/results.csv', index=False)

    print('\nfigures:', flush=True)
    fig_field_maps(banks, lms_by_env, envs, chans, fig_dir)
    fig_angle_cdf(banks, lms_by_env, envs, chans, fig_dir)
    fig_dist_null(banks, lms_by_env, envs, chans, fig_dir, rng)
    fig_size_vs_dist(banks, lms_by_env, envs, chans, fig_dir)
    fig_spacing(results, fig_dir)

    rep = LandmarkNullReport(env_name=','.join(envs), out_dir=out_dir,
                             fig_dir=fig_dir, results=results,
                             log_path=os.environ.get('REALM_LOG_PATH'))
    print('\n' + rep.compose(), flush=True)
    if not args.no_email:
        rep.send()
    print(f'\nresults -> {out_dir}/results.csv\nfigures -> {fig_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
