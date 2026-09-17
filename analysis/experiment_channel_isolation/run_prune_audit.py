"""Where do a channel's fields die? A per-scale audit of the admission rules.

Experiment 2 reports what survived. This reports what did not, and at which
rule. It exists because several libraries collapse in ways a field count
cannot explain: the 10 x 2 m corridor with no panels admits nothing at all on
hog, spatial and visual, while the same corridor with panels gives 370, 266
and 432 fields; the square admits 7 colour fields with panels and 291 without,
and 18 spatial fields without panels against 485 with them.

A field count says a library is empty. It does not say whether the tree never
built a candidate of that size, whether the candidates were built and came out
in pieces, or whether they were whole and lost to a later rule. Those have
different causes and different fixes, and this separates them.

Every candidate is followed through the four rules, and the audit reports
which one took it:

  size          the field came out smaller than the floor or larger than the
                ceiling. These are the only candidates with no scale of their
                own, so they sit at the two ends of the scale axis.
  contiguity    the field came out in pieces -- the largest connected patch
                held under CC_FRAC_MIN of it. This is what an incoherent
                response looks like: a channel that cannot tell two distant
                places apart responds in both, and the field fragments.
  competition   a larger field of the same scale had already claimed the
                ground. Routine, and the main reason counts fall with scale.
  coverage      the scale cleared competition, but its survivors covered less
                than TILING_FRAC_MIN of the floor, so the whole scale went.
                This is how a scale disappears wholesale rather than thinning.

One outcome is not a rule at all: a scale can have no candidates to begin
with, because the tree never produced a field of that size. Nothing
downstream could have saved it, and no threshold would bring it back.

The counts come from the rules engine itself, which records the survivors of
each stage, rather than from a second implementation here that could drift
from it.

The configuration is Experiment 2's operating point exactly -- EXTENT_PCTL 65,
ACT_THRESH 0.5, Rule 2 off, LAMBDA 0, seed 0 -- so these are the same
libraries that experiment reports, and the audit describes those fields rather
than a re-tuned copy of them.

Usage
    python run_prune_audit.py                          # the collapses, plus controls
    python run_prune_audit.py --pairs corr_lm0_l10w2:hog
    python run_prune_audit.py --envs corr_lm0_l10w2    # every channel of one arena
"""

import argparse
import os
import sys
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

import channels as ch                       # noqa: E402
import rules as R                           # noqa: E402
import run_scale_distribution as SD         # noqa: E402
from realm_tools.experiment_lib.reporting import ExperimentReport   # noqa: E402

PCTL, THRESH = SD.SETTINGS[0]

# The libraries that motivated the audit, each paired with the same channel
# where it works, so an empty funnel can be read against a healthy one.
DEFAULT_PAIRS = [
    ('corr_lm0_l10w2', 'hog'),        ('corr_lm8_l10w2', 'hog'),
    ('corr_lm0_l10w2', 'spatial'),    ('corr_lm8_l10w2', 'spatial'),
    ('corr_lm0_l10w2', 'visual'),     ('corr_lm0_l10w2', 'color'),
    ('corr_lm8_l10w10', 'color'),     ('corr_lm0_l10w10', 'color'),
    ('corr_lm0_l10w10', 'spatial'),   ('corr_lm8_l10w10', 'spatial'),
    ('corr_lm0_l10w10', 'visual'),    ('circ_lm8_r6', 'color'),
]

INK, INK_2, MUTED = SD.INK, SD.INK_2, SD.MUTED
RULE_GRAY, SURFACE = SD.RULE_GRAY, SD.SURFACE

# Four hues, one per rule, rather than four steps of one hue. What a reader
# asks here is *which* rule took a library, and that is identity, not
# magnitude; one hue in four steps makes neighbouring rules hard to tell apart
# at a glance. Checked with a palette validator at its strictest setting,
# since any two bars in a group can be compared: worst colour-blind pair
# dE 9.2, worst normal-vision pair 16.3. Aqua sits just under 3:1 against the
# surface, which the printed counts relieve. Candidates are grey -- a starting
# count, not a rule.
RULE_NAMES = ['size', 'contiguity', 'competition', 'coverage']
RULE_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']
COUNT_COLUMNS = ['n_candidates', 'n_pass_size', 'n_pass_contiguity',
                 'n_pass_competition', 'n_pass_coverage']
BAR_COLORS = [RULE_GRAY] + RULE_COLORS
BAR_LABELS = ['candidates'] + [f'passed {n}' for n in RULE_NAMES]


def parse_pairs(args):
    """--pairs env:channel,... and --envs env,... into one ordered pair list."""
    pairs = []
    if args.pairs:
        for tok in args.pairs.split(','):
            if not tok.strip():
                continue
            env_name, _, cname = tok.partition(':')
            if not cname:
                raise SystemExit(f'--pairs wants env:channel, got {tok!r}')
            pairs.append((env_name.strip(), cname.strip()))
    if args.envs:
        chans = [c.strip() for c in args.channels.split(',') if c.strip()]
        for e in args.envs.split(','):
            if e.strip():
                pairs += [(e.strip(), c) for c in chans]
    return pairs or list(DEFAULT_PAIRS)


def build_library(env_name, cname, blocks, xy, env, base_C, device, verbose=True):
    """One field library, with the report that says where each candidate died."""
    X = ch.assemble(blocks, ch.CHANNEL_SETS[cname], normalize=True)
    D2 = R.feature_sq_distances(X, device=device, verbose=verbose)
    rng = np.random.default_rng(base_C['RANDOM_SEED'])
    feat_med = R._median_offdiag(D2, rng)
    d2xy = ((xy[:3000, None, :] - xy[None, :3000, :]) ** 2).sum(-1)
    xy_med = float(np.median(d2xy[np.triu_indices(len(d2xy), 1)]))
    tree = R.build_tree(D2, xy, feat_med, xy_med, cfg=base_C, verbose=verbose)
    C = R.resolve_cfg(dict(base_C, EXTENT_PCTL=PCTL, ACT_THRESH=THRESH))
    ctx = R.prepare_candidates(X, xy, env, D2, feat_med, xy_med, cfg=C,
                               device=device, tree=tree,
                               tag=f'{env_name}/{cname}', verbose=verbose)
    bank, _, rep = R.admit_fields(ctx, cfg=C, verbose=verbose)
    del X, D2, ctx
    return bank, rep, C


def scale_rows(rep, C, tag):
    """One row per scale: how many candidates reached it, and which rule took them.

    The two end rows are the size rule's own casualties. A field below the
    floor or above the ceiling has no scale of its own, so it cannot appear in
    a numbered row; within a scale, size has nothing left to reject, and the
    other three rules do the work.
    """
    area = np.asarray(rep['cand_area'], dtype=float)
    band = np.asarray(rep['cand_band'], dtype=int)
    r_eq = np.asarray(rep['cand_r_eq'], dtype=float)
    pass_size = np.asarray(rep['cand_pass_size'], dtype=bool)
    pass_cc = np.asarray(rep['cand_pass_contiguity'], dtype=bool)
    kept11 = np.asarray(rep['cand_kept_rule11'], dtype=bool)
    kept12 = np.asarray(rep['cand_kept_rule12'], dtype=bool)
    cc_frac = np.asarray(rep['cand_cc_frac'], dtype=float)
    sigma_ratio = np.asarray(rep['cand_sigma_ratio'], dtype=float)
    coverage, thr = rep['coverage'], C['TILING_FRAC_MIN']
    lo, hi = rep['band_lo'], rep['band_hi']

    def med(x, sel):
        return float(np.median(x[sel])) if sel.any() else np.nan

    rows = []
    for label, sel, why in (
            ('< floor', ~pass_size & (area < rep['area_min']),
             'size: smaller than the floor'),
            ('> ceiling', ~pass_size & (area > rep['area_max']),
             'size: larger than the ceiling')):
        n = int(sel.sum())
        rows.append(dict(tag, scale=label, n_candidates=n, n_pass_size=0,
                         n_pass_contiguity=0, n_pass_competition=0,
                         n_pass_coverage=0, median_radius_m=med(r_eq, sel),
                         median_cc_frac=med(cc_frac, sel),
                         median_sigma_ratio=med(sigma_ratio, sel),
                         coverage_reached=np.nan, coverage_needed=thr,
                         scale_kept=False, verdict=why if n else 'none'))
    for sc in SD.SCALES:
        sel = pass_size & (band == sc)
        whole = sel & pass_cc
        won = sel & kept11
        adm = sel & kept12
        cov = float(coverage.get(sc, np.nan)) if isinstance(coverage, dict) else np.nan
        n, n_wh, n_won, n_adm = map(int, (sel.sum(), whole.sum(), won.sum(), adm.sum()))
        if n == 0:
            v = 'no candidate reached this scale'
        elif n_wh == 0:
            v = f'contiguity took all {n}: every field came out in pieces'
        elif n_won == 0:
            v = f'competition took all {n_wh}: every whole field lost its ground'
        elif n_adm == 0:
            v = (f'coverage dropped the scale: {100*cov:.0f}% of the floor '
                 f'against the {100*thr:.0f}% it needs' if np.isfinite(cov)
                 else 'coverage dropped the scale')
        else:
            v = f'{n_adm} admitted'
        rows.append(dict(tag, scale=str(sc), n_candidates=n, n_pass_size=n,
                         n_pass_contiguity=n_wh, n_pass_competition=n_won,
                         n_pass_coverage=n_adm, median_radius_m=med(r_eq, sel),
                         median_cc_frac=med(cc_frac, sel),
                         median_sigma_ratio=med(sigma_ratio, sel),
                         coverage_reached=cov, coverage_needed=thr,
                         scale_kept=bool(lo <= sc <= hi), verdict=v))
    return rows


def pair_row(rep, C, env, tag, n_admitted):
    """One row per (arena, channel): the funnel totals and a one-line diagnosis."""
    funnel = dict(rep['funnel'])
    r_eq = np.asarray(rep['cand_r_eq'], dtype=float)
    area = np.asarray(rep['cand_area'], dtype=float)
    pass_size = np.asarray(rep['cand_pass_size'], dtype=bool)
    return dict(
        tag,
        env_area_m2=float(env['env_area']),
        n_candidates=int(rep['n_candidates']),
        # The engine's own funnel keys, reported under the four names the
        # papers use: size, contiguity, competition, coverage.
        pass_size=int(funnel.get('rule_8_9_size', 0)),
        pass_contiguity=int(funnel.get('rule_1_contiguity', 0)),
        pass_competition=int(funnel.get('rule_11_competition', 0)),
        pass_coverage=int(n_admitted),
        frac_below_floor=float(np.mean(~pass_size & (area < rep['area_min']))),
        frac_above_ceiling=float(np.mean(~pass_size & (area > rep['area_max']))),
        cand_radius_min_m=float(r_eq.min()) if len(r_eq) else np.nan,
        cand_radius_median_m=float(np.median(r_eq)) if len(r_eq) else np.nan,
        cand_radius_max_m=float(r_eq.max()) if len(r_eq) else np.nan,
        floor_radius_m=float(rep['r_min']), ceiling_radius_m=float(rep['r_max']),
        median_cc_frac=float(rep['median_cc_frac']),
        frag_rate=float(rep['frag_rate']),
        median_sigma_ratio=float(rep['median_sigma_ratio']),
        scales_kept=f"{rep['band_lo']}..{rep['band_hi']}",
        grid_bin_m=float(np.sqrt(rep['grid']['bin_area'])),
    )


def diagnose(pair, scales):
    """The sentence a reader wants: which rule emptied this library, if any."""
    env_name, cname = pair['env'], pair['channel']
    numbered = [r for r in scales if r['scale'].isdigit()]
    if pair['pass_coverage'] >= SD.MIN_FIELDS and all(
            r['n_pass_coverage'] for r in numbered if r['n_candidates']):
        return f'{env_name} {cname}: healthy, {pair["pass_coverage"]} fields'
    if pair['n_candidates'] == 0:
        return f'{env_name} {cname}: the tree produced no candidate at all'
    if pair['pass_size'] == 0:
        return (f'{env_name} {cname}: size took the library -- every candidate '
                f'fell outside the window ({100*pair["frac_below_floor"]:.0f}% '
                f'under the floor, {100*pair["frac_above_ceiling"]:.0f}% over '
                f'the ceiling). The channel localises to a size the rules do '
                f'not admit')
    if pair['pass_contiguity'] == 0:
        return (f'{env_name} {cname}: contiguity took the library -- '
                f'{pair["pass_size"]} candidates were the right size but none '
                f'was whole (median largest piece '
                f'{100*pair["median_cc_frac"]:.0f}% of the field)')
    empty = [r for r in numbered if r['n_candidates'] and not r['n_pass_coverage']]
    if empty:
        which = ', '.join(r['scale'] for r in empty)
        return (f'{env_name} {cname}: scale(s) {which} were built and emptied -- '
                + '; '.join(f'scale {r["scale"]}: {r["verdict"]}' for r in empty[:3]))
    return f'{env_name} {cname}: {pair["pass_coverage"]} fields, nothing emptied'


def fig_funnels(scales_df, pairs_df, fig_dir):
    """One panel per library: how many candidates each rule lets through.

    Five bars per scale -- the candidates built at that scale, then what
    survives size, contiguity, competition and coverage in turn. Counts on a
    linear axis, because the question is whether anything came through at all,
    and a bar of zero beside a bar of hundreds is the answer. The number that
    survives all four is printed, since it is often too small to see.

    The two end categories belong to the size rule alone: a field below the
    floor or above the ceiling has no scale of its own to be drawn at.
    """
    pairs = list(pairs_df.itertuples())
    if not pairs:
        return None
    n_c = min(3, len(pairs))
    n_r = int(np.ceil(len(pairs) / n_c))
    fig, axes = plt.subplots(n_r, n_c, squeeze=False,
                             figsize=(5.0 * n_c, 3.1 * n_r + 1.0))
    fig.patch.set_facecolor(SURFACE)
    cats = ['< floor'] + [str(s) for s in SD.SCALES] + ['> ceiling']
    w = 0.17
    for ax_i, pr in enumerate(pairs):
        ax = axes[ax_i // n_c][ax_i % n_c]
        ax.set_facecolor(SURFACE)
        d = scales_df[(scales_df.env == pr.env) & (scales_df.channel == pr.channel)]
        d = d.set_index('scale').reindex(cats)
        x = np.arange(len(cats))
        for k, (col, colour) in enumerate(zip(COUNT_COLUMNS, BAR_COLORS)):
            ax.bar(x + (k - 2) * w, d[col].fillna(0).to_numpy(), width=w * 0.9,
                   color=colour, edgecolor=SURFACE, linewidth=0.4, zorder=2)
        for xi, v in zip(x, d.n_pass_coverage.fillna(0).to_numpy()):
            ax.text(xi + 2 * w, max(v, 0), f'{int(v)}', ha='center', va='bottom',
                    fontsize=6, color=INK_2)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(RULE_GRAY)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=7)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_xlabel('scale (0 finest)', fontsize=8, color=INK_2)
        ax.set_ylabel('candidates', fontsize=8, color=INK_2)
        ax.set_title(f'{pr.env} · {pr.channel} — {pr.pass_coverage} admitted',
                     fontsize=9, color=INK)
    for ax_i in range(len(pairs), n_r * n_c):
        axes[ax_i // n_c][ax_i % n_c].set_visible(False)
    height = fig.get_figheight()
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in BAR_COLORS],
               BAR_LABELS, loc='upper center', ncol=len(BAR_LABELS), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1 - 0.48 / height))
    fig.suptitle("P1  which rule takes each scale's candidates\n"
                 'bars left to right: built at that scale, then what survives '
                 'size, contiguity, competition and coverage',
                 fontsize=10, color=INK, y=1 - 0.06 / height)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / height))
    path = os.path.join(fig_dir, 'P1_prune_funnels.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}', flush=True)
    return path


class PruneAuditReport(ExperimentReport):
    """Emailed summary: one diagnosis per library, then the per-scale tables."""

    experiment = 'prune-audit'

    def title(self):
        p = self.results
        if p is None or not len(p):
            return 'no libraries audited'
        empty = int((p.pass_coverage < SD.MIN_FIELDS).sum())
        return f'{len(p)} libraries, {empty} empty or near-empty'

    def figures(self):
        return [f for f in getattr(self, 'figure_paths', []) if f and os.path.exists(f)]

    def data_files(self):
        return [p for p in (f'{self.out_dir}/prune_audit_scales.csv',
                            f'{self.out_dir}/prune_audit_pairs.csv')
                if os.path.exists(p)]

    def body(self):
        p, sc = self.results, self.scales
        if p is None or not len(p):
            return 'No libraries were audited.'
        S = self.section
        out = [S('WHAT EMPTIED EACH LIBRARY', '\n'.join(
            ['One line per library, in the order they were audited.', '']
            + [f'  {d}' for d in self.diagnoses]))]

        cols = ['env', 'channel', 'n_candidates', 'pass_size', 'pass_contiguity',
                'pass_competition', 'pass_coverage', 'median_cc_frac',
                'median_sigma_ratio', 'scales_kept']
        out.append(S('THE FOUR RULES, PER LIBRARY', self.table(p[cols])))

        for r in p.itertuples():
            d = sc[(sc.env == r.env) & (sc.channel == r.channel)]
            cols2 = ['scale', 'n_candidates', 'n_pass_size', 'n_pass_contiguity',
                     'n_pass_competition', 'n_pass_coverage', 'median_radius_m',
                     'coverage_reached', 'verdict']
            out.append(S(f'{r.env} · {r.channel}', self.table(d[cols2])))

        out.append(S('HOW TO READ IT', '\n'.join([
            'Each row follows one scale through the four rules, in order.', '',
            'candidates    fields the tree built at that scale. A zero here is '
            'not a rule at work: the channel never localised to that size, and '
            'nothing downstream could have changed it.',
            'size          the floor and the ceiling. A field outside them has '
            'no scale of its own, so size only ever shows at the two ends of '
            'the axis -- and a library whose candidates all land there is one '
            'localising to a size the rules do not admit.',
            'contiguity    a field must be one connected patch. Fields that '
            'fragment are the signature of a channel that cannot separate two '
            'distant places: it responds in both, and the field breaks apart.',
            'competition   same-scale neighbours suppress each other. Counts '
            'falling here is normal, and is how the ladder thins with scale.',
            'coverage      a scale whose survivors cover too little floor is '
            'dropped whole. A scale with candidates but nothing admitted was '
            'emptied wholesale rather than thinned.', '',
            'median_sigma_ratio near 1 means a node\'s members are as far '
            'apart in feature space as two random locations are: the response '
            'is flat, and a flat response makes a field that either fills the '
            'arena or breaks into pieces. It is the input-side number to read '
            'when contiguity is taking everything.'])))

        return '\n'.join(out)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pairs', default='',
                   help='env:channel pairs, comma separated. Default is the '
                        'collapsed libraries plus controls (DEFAULT_PAIRS).')
    p.add_argument('--envs', default='',
                   help='audit every channel of these arenas, comma separated')
    p.add_argument('--channels', default=','.join(SD.CHANNELS),
                   help='channels used with --envs')
    p.add_argument('--subsample', type=int, default=0,
                   help='positions to subsample -- a quick look, not comparable '
                        'to a full run')
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--no-email', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    pairs = parse_pairs(args)
    out_dir = f'{REPO}/data_cache/prune_audit'
    fig_dir = f'{HERE}/figures/prune_audit'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    base_C = R.resolve_cfg(dict(LAMBDA=0.0, RANDOM_SEED=0, USE_GPU=not args.no_gpu))
    device = R.pick_device(use_gpu=not args.no_gpu)
    rng = np.random.default_rng(0)

    print('=' * 72)
    print('Prune audit | which rule empties a library, scale by scale')
    print(f'  pairs    : {len(pairs)}')
    print(f'  operating point: EXTENT_PCTL {PCTL}, ACT_THRESH {THRESH}, Rule 2 off')
    print('=' * 72, flush=True)

    scale_rows_all, pair_rows, diagnoses, missing = [], [], [], []
    loaded_env, blocks, xy, env = None, None, None, None
    for env_name, cname in pairs:
        data_path = f'{REPO}/data/vpce/collect_data/{env_name}.h5'
        xml_path = f'{REPO}/simulation/worlds/environments/vpce/{env_name}.xml'
        if not os.path.exists(data_path):
            print(f'\n[{env_name}] no dataset at {data_path} -- skipping', flush=True)
            missing.append(env_name)
            continue
        if env_name != loaded_env:            # one load serves every channel
            print(f'\n===== {env_name} =====', flush=True)
            blocks, xy = ch.load_channel_blocks(data_path)
            if args.subsample and args.subsample < len(xy):
                sel = np.sort(rng.choice(len(xy), args.subsample, replace=False))
                blocks, xy = {k: v[sel] for k, v in blocks.items()}, xy[sel]
                print(f'  subsampled to {len(xy)} -- not comparable to a full run')
            root = ET.parse(xml_path).getroot()
            env = R.build_env(xy, root)
            env['role'] = SD.env_role(env_name)
            env['n_landmarks'] = len(root.findall('landmark'))
            loaded_env = env_name
        print(f'\n--- {env_name} / {cname} ---', flush=True)
        bank, rep, C = build_library(env_name, cname, blocks, xy, env, base_C, device)
        tag = dict(env=env_name, channel=cname)
        rows = scale_rows(rep, C, tag)
        prow = pair_row(rep, C, env, tag, len(bank))
        scale_rows_all += rows
        pair_rows.append(prow)
        diagnoses.append(diagnose(prow, rows))
        print(f'  {diagnoses[-1]}', flush=True)

    if not pair_rows:
        print('\nNothing audited. Datasets missing: ' + (', '.join(missing) or 'none'))
        return 1

    scales_df = pd.DataFrame(scale_rows_all)
    pairs_df = pd.DataFrame(pair_rows)
    scales_df.to_csv(f'{out_dir}/prune_audit_scales.csv', index=False)
    pairs_df.to_csv(f'{out_dir}/prune_audit_pairs.csv', index=False)

    print('\nfigures:', flush=True)
    fig_path = fig_funnels(scales_df, pairs_df, fig_dir)

    rep_obj = PruneAuditReport(env_name=','.join(sorted({e for e, _ in pairs})),
                               out_dir=out_dir, fig_dir=fig_dir, results=pairs_df,
                               log_path=os.environ.get('REALM_LOG_PATH'))
    rep_obj.scales, rep_obj.diagnoses = scales_df, diagnoses
    rep_obj.figure_paths = [fig_path]
    if missing:
        print(f'\n!! datasets not found, skipped: {", ".join(sorted(set(missing)))}')
    print('\n' + rep_obj.compose(), flush=True)
    if not args.no_email:
        rep_obj.send()
    print(f'\nscales -> {out_dir}/prune_audit_scales.csv'
          f'\npairs  -> {out_dir}/prune_audit_pairs.csv'
          f'\nfigure -> {fig_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
