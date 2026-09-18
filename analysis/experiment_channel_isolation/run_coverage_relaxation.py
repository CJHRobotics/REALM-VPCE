"""What happens to the scale distribution when Rule 12's coverage test is relaxed.

A side analysis on Experiment 2. Same arenas, same channels, same operating
point, same code -- one knob moved.

Which rule this is
------------------
`TILING_FRAC_MIN`, the coverage threshold in Rule 12 (tiling stop). A scale
whose admitted fields, unioned, cover less than that fraction of the floor is
not a population code for space at that resolution, so Rule 12 drops the whole
scale; the surviving ladder is the contiguous run of qualifying scales around
the best-covered one, so it never has a hole in the middle. The default is
0.50.

Rule 4 in this codebase is spatial weighting -- merge cost = feature distance
+ LAMBDA * space -- and it is already off at the operating point (LAMBDA 0), so
toggling that would change nothing. Coverage is Rule 12. This script sweeps
Rule 12.

Why relaxing it is a clean manipulation
---------------------------------------
Rule 12 runs LAST, after Rule 11's competition, and only deletes whole scales
from a ladder that is already settled. Nothing upstream of it depends on
`TILING_FRAC_MIN`: not the tree, not the candidate set, not the size window,
not which field beats which in competition. So lowering the threshold can only
ADD scales back, and the library at a lower threshold is a strict superset of
the library at a higher one. That is asserted per library rather than assumed
(`nested` in the summary); if it ever fails, something upstream is reading the
knob and the comparison is void.

It also means the sweep is nearly free. `prepare_candidates` -- the tree, the
Gram matrix, the candidate measurement and the environment readout, which is
all of the cost -- is built once per arena and channel and reused for every
threshold. Only `admit_fields` repeats, and that is the cheap stage.

The sweep
---------
    0.50   the operating point. Experiment 2's libraries exactly.
    0.35
    0.20
    0.00   Rule 12 off: every scale that survived competition is kept.

The run checks itself against Experiment 2 at 0.50: the bank it builds there
must be identical, field for field, to the one in Experiment 2's cache, since
it is the same code at the same settings with the same seed. `matches_exp2` in
the summary says whether it was. A mismatch means this script is not measuring
what it claims to be comparing against, and nothing below it can be read.

What is reported
----------------
Per arena x channel x threshold, through Experiment 2's own functions so the
numbers are the same numbers:

  * field count, and the surviving scale ladder (`band_lo` to `band_hi`);
  * per scale: how many fields, share of the library, median area and
    coverage, tiling multiple, CV -- `scale_table`;
  * CV of field size, min / median / max, max/min ratio, fraction of the arena
    per field, truncation diagnostics -- `describe`;
  * the three-form fit -- log-normal, negative exponential, Gaussian, each
    truncated to the Rule 8/9 window -- on area and on equivalent diameter,
    so the question "does the shape of the distribution change" is answered
    on the same footing Experiment 2 answered it.

And the measured coverage of every scale, including the ones Rule 12 dropped,
which is the number that says how close each scale was to the cut. A scale at
0.49 and a scale at 0.02 are both dropped at the operating point and are not
the same finding.

Writes to its own cache (`data_cache/coverage_relaxation`) under its own key,
which carries the threshold. Experiment 2's cache is read for the comparison
and never written.

Cost
----
This cannot reuse Experiment 2's banks: they hold the survivors, and the whole
question is about the scales it dropped. So the pipeline runs in full -- feature
blocks, Gram matrix, Ward tree -- at Experiment 2's cost and with its GPU and
memory needs, once per arena and channel. Fan out one arena per job, as
Experiment 2 does, then re-run over all eight from the cache for the combined
figures.

Usage
    python run_coverage_relaxation.py [--envs a,b] [--channels ...]
                                      [--tiling 0.5,0.35,0.2,0] [--rebuild]
"""

import argparse
import glob
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch
import rules as R
import run_scale_distribution as SD
from realm_tools.experiment_lib.reporting import ExperimentReport

# The eight collected arenas. circ_lm8_r10 is in Experiment 2's list but was
# never collected.
ENVS = [e for e in SD.ALL_ENVS if e != 'circ_lm8_r10']
CHANNELS = SD.CHANNELS
CHANNEL_COLORS = SD.CHANNEL_COLORS
SCALES = SD.SCALES
SCALE_COLORS = SD.SCALE_COLORS
INK, MUTED, RULE_GRAY, SURFACE = SD.INK, SD.MUTED, SD.RULE_GRAY, SD.SURFACE

# Experiment 2's operating point, unchanged. The only knob this script moves
# is TILING_FRAC_MIN.
PCTL, THRESH = SD.SETTINGS[0]
IOU = None
BASE_C = R.resolve_cfg(dict(LAMBDA=0.0, RANDOM_SEED=0))
OPERATING_TILING = float(BASE_C['TILING_FRAC_MIN'])      # 0.50
TILING_SWEEP = (0.50, 0.35, 0.20, 0.0)

DATA_DIR = f'{REPO}/data/vpce/collect_data'
XML_DIR = f'{REPO}/simulation/worlds/environments/vpce'
EXP2_BANK_DIR = f'{REPO}/data_cache/scale_distribution'   # read only
BANK_DIR = f'{REPO}/data_cache/coverage_relaxation'
OUT_DIR = BANK_DIR
FIG_DIR = f'{HERE}/figures/coverage_relaxation'

MIN_FIELDS = SD.MIN_FIELDS
N_BOOT = SD.N_BOOT

# The threshold is an ordered sweep, so it runs along a multi-hue ramp rather
# than one hue. Stops short of plasma's yellow end, which measures under 2:1
# against this surface.
TILING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'vpce_tiling', matplotlib.colormaps['plasma'](np.linspace(0.05, 0.72, 256)))


def tiling_colors(thrs):
    """One colour per threshold, darkest at the operating point."""
    n = max(len(thrs) - 1, 1)
    return {t: mcolors.to_hex(TILING_CMAP(i / n))
            for i, t in enumerate(sorted(thrs, reverse=True))}


def tiling_label(thr):
    if abs(thr - OPERATING_TILING) < 1e-12:
        return f'{thr:.2f} (operating point)'
    if thr <= 0:
        return f'{thr:.2f} (Rule 12 off)'
    return f'{thr:.2f}'


def bank_path(env_name, cname, thr):
    """This script's own cache key, carrying the threshold.

    Deliberately not Experiment 2's key: a bank built at a relaxed threshold
    written under Experiment 2's name would silently replace its library for
    every experiment downstream.
    """
    return (f'{BANK_DIR}/{env_name}/{cname}_p{PCTL}_t{THRESH:g}_roff'
            f'_tf{thr:g}_bank.csv')


def exp2_bank_path(env_name, cname):
    """Experiment 2's bank at the operating point, for the self-check."""
    return (f'{EXP2_BANK_DIR}/{env_name}/{cname}_p{PCTL}_t{THRESH:g}'
            f'_roff_bank.csv')


# -------------------------------------------------------------- the sweep

def build_sweep(env_name, cname, blocks, xy, env, thrs, base_C, device,
                use_cache, verbose=True):
    """One bank per threshold, from one shared candidate context.

    The expensive stages -- the feature Gram matrix, the Ward tree, the
    candidate measurement and the environment readout -- depend on none of the
    admission parameters, so they are built once and every threshold is scored
    against an identical tree. That is what makes the sweep cheap and what
    makes it strictly like-for-like: a difference between two thresholds
    cannot be a difference in the tree.

    Returns {thr: (bank, report)}.
    """
    want = {t: bank_path(env_name, cname, t) for t in thrs}
    rep_of = lambda p: p.replace('_bank.csv', '_report.json')
    if use_cache and all(os.path.exists(v) and os.path.exists(rep_of(v))
                         for v in want.values()):
        if verbose:
            print(f'  [{cname}] cached sweep', flush=True)
        return {t: (pd.read_csv(v), json.load(open(rep_of(v))))
                for t, v in want.items()}
    os.makedirs(f'{BANK_DIR}/{env_name}', exist_ok=True)

    t0 = time.time()
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
                               tag=f'{env_name}/{cname}/p{PCTL}t{THRESH:g}',
                               verbose=verbose)
    out = {}
    for thr in thrs:
        Ct = R.resolve_cfg(dict(C, SPLIT_HALF_IOU_MIN=IOU,
                                TILING_FRAC_MIN=float(thr)))
        bank, _, rep = R.admit_fields(ctx, cfg=Ct, verbose=verbose)
        bank.to_csv(want[thr], index=False)
        rep_clean = {k: v for k, v in rep.items()
                     if not isinstance(v, np.ndarray)}
        with open(rep_of(want[thr]), 'w') as f:
            json.dump(rep_clean, f, indent=2, default=float)
        out[thr] = (bank, rep_clean)
        if verbose:
            print(f'  [{cname}] TILING_FRAC_MIN {thr:g}: {len(bank)} fields, '
                  f'scales {rep_clean.get("band_lo")}-'
                  f'{rep_clean.get("band_hi")}', flush=True)
    if verbose:
        print(f'  [{cname}] {len(thrs)} banks in {time.time()-t0:.0f}s',
              flush=True)
    del X, D2
    return out


# ------------------------------------------------------------- validation

def _ids(bank):
    """A library as a set of node ids, for the nesting and equality checks."""
    if 'node_id' not in bank or not len(bank):
        return set()
    return set(int(v) for v in bank.node_id.to_numpy())


def check_nested(sweep):
    """Is each library a superset of the one at the next threshold up?

    Rule 12 runs after competition and only deletes whole scales, so lowering
    the threshold can only add fields back. If this fails, something upstream
    of Rule 12 is reading TILING_FRAC_MIN and the sweep is not a clean
    manipulation of one rule.
    """
    thrs = sorted(sweep, reverse=True)
    rows, ok = [], True
    for hi, lo in zip(thrs[:-1], thrs[1:]):
        a, b = _ids(sweep[hi][0]), _ids(sweep[lo][0])
        nested = a <= b
        ok &= nested
        rows.append(dict(thr_high=hi, thr_low=lo, n_high=len(a), n_low=len(b),
                         added=len(b - a), lost=len(a - b), nested=bool(nested)))
    return ok, rows


def check_against_exp2(env_name, cname, bank):
    """Does this script's operating-point bank equal Experiment 2's?

    Same code, same settings, same seed, so it must. Returns (verdict, note)
    where the verdict is True, False, or None when there is nothing to compare
    against.
    """
    p = exp2_bank_path(env_name, cname)
    if not os.path.exists(p):
        return None, 'no Experiment 2 bank cached to compare against'
    ref = pd.read_csv(p)
    if len(ref) != len(bank):
        return False, f'{len(bank)} fields here against {len(ref)} there'
    if _ids(ref) != _ids(bank):
        return False, 'same count, different node ids'
    d = float(np.abs(np.sort(ref.area_env_m2.to_numpy(dtype=float)) -
                     np.sort(bank.area_env_m2.to_numpy(dtype=float))).max())
    if d > 1e-9:
        return False, f'same ids, areas differ by up to {d:.3g} m^2'
    return True, f'identical, {len(ref)} fields'


# -------------------------------------------------------------- statistics

def library_rows(bank, rep, env, env_name, cname, thr, n_boot, seed):
    """Experiment 2's own descriptors, plus what Rule 12 saw.

    `describe`, `scale_table` and `fit_forms` are imported rather than
    reimplemented: the point of the comparison is that the numbers either side
    of it are the same numbers.
    """
    tag = dict(env=env_name, channel=cname, tiling_frac_min=float(thr),
               env_area_m2=float(env['env_area']))
    cov = {int(k): float(v) for k, v in (rep.get('coverage') or {}).items()}
    row = dict(tag, n_fields=len(bank),
               band_lo=int(rep.get('band_lo', -1)),
               band_hi=int(rep.get('band_hi', -1)),
               n_scales_kept=(0 if rep.get('band_lo', -1) < 0
                              else int(rep['band_hi']) - int(rep['band_lo']) + 1),
               n_scales_measured=len(cov),
               coverage_json=json.dumps(cov))
    for s in SCALES:
        row[f'coverage_scale_{s}'] = cov.get(s, np.nan)
    if len(bank):
        row.update(SD.describe(bank, env, BASE_C))
    scale_rows = SD.scale_table(bank, env, BASE_C, tag) if len(bank) else []
    fit_rows = []
    if len(bank) >= MIN_FIELDS:
        a_lo = BASE_C['RULE8_AREA_FRAC'] * env['env_area']
        a_hi = BASE_C['RULE9_AREA_FRAC'] * env['env_area']
        for var, x, lo, hi in (
                ('area', bank.area_env_m2, a_lo, a_hi),
                ('diameter', 2.0 * bank.radius_env_m,
                 2.0 * np.sqrt(a_lo / np.pi), 2.0 * np.sqrt(a_hi / np.pi))):
            for d in SD.fit_forms(x, lo, hi, n_boot=n_boot, seed=seed):
                d['params'] = json.dumps(d['params'])
                fit_rows.append(dict(tag, variable=var, **d))
    return row, scale_rows, fit_rows


def paired_table(summary, fits, thr_off, thr_on=OPERATING_TILING):
    """Rule 12 off against the operating point, one row per library.

    The comparison the analysis exists for. Everything is a difference or a
    ratio against the same library's operating-point value, so a channel that
    simply has more fields everywhere cannot look like an effect.
    """
    on = summary[np.isclose(summary.tiling_frac_min, thr_on)].set_index(
        ['env', 'channel'])
    off = summary[np.isclose(summary.tiling_frac_min, thr_off)].set_index(
        ['env', 'channel'])
    best = {}
    if fits is not None and len(fits):
        f = fits[fits.variable == 'area']
        for key, g in f.groupby(['env', 'channel', 'tiling_frac_min']):
            if 'aic' in g and g.aic.notna().any():
                best[key] = str(g.loc[g.aic.idxmin(), 'form'])
    rows = []
    for key in on.index.intersection(off.index):
        a, b = on.loc[key], off.loc[key]
        e, c = key
        rows.append(dict(
            env=e, channel=c, env_area_m2=float(a.env_area_m2),
            n_on=int(a.n_fields), n_off=int(b.n_fields),
            n_added=int(b.n_fields) - int(a.n_fields),
            fold_more=(float(b.n_fields) / a.n_fields
                       if a.n_fields else np.nan),
            scales_on=f'{int(a.band_lo)}-{int(a.band_hi)}',
            scales_off=f'{int(b.band_lo)}-{int(b.band_hi)}',
            n_scales_on=int(a.n_scales_kept), n_scales_off=int(b.n_scales_kept),
            cv_on=float(a.get('cv_area_pct', np.nan)),
            cv_off=float(b.get('cv_area_pct', np.nan)),
            area_median_on_m2=float(a.get('area_median_m2', np.nan)),
            area_median_off_m2=float(b.get('area_median_m2', np.nan)),
            ratio_on=float(a.get('area_max_min_ratio', np.nan)),
            ratio_off=float(b.get('area_max_min_ratio', np.nan)),
            best_form_on=best.get((e, c, thr_on), '--'),
            best_form_off=best.get((e, c, thr_off), '--'),
            form_changed=(best.get((e, c, thr_on), '--') !=
                          best.get((e, c, thr_off), '--'))))
    out = pd.DataFrame(rows)
    return out.sort_values(['env_area_m2', 'env', 'channel']) if len(out) else out


# ------------------------------------------------------------------ figures

FIGURES_WRITTEN = []


def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    FIGURES_WRITTEN.append(p)
    print(f'  {p}', flush=True)


def prune_orphan_figures():
    """Delete C<digit>*.png this run did not write, as Experiment 2 does."""
    keep = {os.path.abspath(p) for p in FIGURES_WRITTEN}
    for p in sorted(glob.glob(os.path.join(FIG_DIR, 'C[0-9]*.png'))):
        if os.path.abspath(p) not in keep:
            os.remove(p)
            print(f'  pruned orphaned figure {os.path.basename(p)}', flush=True)


def _panels(envs, sharey=False):
    ncol = int(np.ceil(len(envs) / 2)) if len(envs) > 4 else max(len(envs), 1)
    nrow = int(np.ceil(len(envs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharey=sharey,
                             figsize=(3.3 * ncol, 2.9 * nrow))
    flat = [ax for r in axes for ax in r]
    for ax in flat[len(envs):]:
        ax.axis('off')
    for ax in flat:
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(RULE_GRAY)
    return fig, flat[:len(envs)], ncol


def _envs_in_order(summary):
    return list(summary.sort_values('env_area_m2').env.drop_duplicates())


def fig_scales_recovered(scales, envs, thrs, name):
    """C1: how many fields at each scale, at each threshold.

    The direct answer to what relaxing Rule 12 does: a bar that exists only at
    the lower thresholds is a scale the coverage test was deleting. Channels
    pooled, because the question is about the rule and not about a channel.
    """
    if not len(scales):
        return
    cols = tiling_colors(thrs)
    fig, axes, ncol = _panels(envs)
    width = 0.8 / max(len(thrs), 1)
    for ax, e in zip(axes, envs):
        se = scales[scales.env == e]
        for i, thr in enumerate(sorted(thrs, reverse=True)):
            g = se[np.isclose(se.tiling_frac_min, thr)]
            n = [int(g[g.scale == s].n_fields.sum()) for s in SCALES]
            ax.bar(np.array(SCALES) + (i - (len(thrs) - 1) / 2) * width, n,
                   width=width, color=cols[thr], label=tiling_label(thr),
                   edgecolor='none')
        ax.set_title(e, fontsize=9, color=INK)
        ax.set_xticks(SCALES)
        ax.set_xlabel('scale (0 finest, 5 coarsest)', fontsize=7.5, color=MUTED)
    for i, ax in enumerate(axes):
        if i % ncol == 0:
            ax.set_ylabel('fields, channels pooled', fontsize=8, color=INK)
    axes[0].legend(fontsize=5.5, frameon=False, title='TILING_FRAC_MIN',
                   title_fontsize=5.5)
    fig.suptitle('C1  fields per scale as Rule 12\'s coverage test is relaxed\n'
                 'a bar present only at the lower thresholds is a scale the '
                 'rule was deleting', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, name)


def fig_coverage_per_scale(summary, envs, thrs, name):
    """C2: the coverage each scale actually reached, against the thresholds.

    This is the number Rule 12 tests, and it says how close a call each
    deletion was. A scale sitting at 0.49 and one sitting at 0.02 are both
    dropped at the operating point and are not the same finding. Measured on
    the fields that survived competition, so it is the same for every
    threshold -- the threshold only decides where the line falls.
    """
    cov_cols = [f'coverage_scale_{s}' for s in SCALES]
    if not any(c in summary for c in cov_cols):
        return
    s0 = summary[np.isclose(summary.tiling_frac_min, min(thrs))]
    if not len(s0):
        return
    cols = tiling_colors(thrs)
    fig, axes, ncol = _panels(envs, sharey=True)
    for ax, e in zip(axes, envs):
        se = s0[s0.env == e]
        for thr in sorted(thrs, reverse=True):
            ax.axhline(thr, color=cols[thr], lw=1.0, ls='--', zorder=1,
                       label=tiling_label(thr))
        for c in CHANNELS:
            g = se[se.channel == c]
            if not len(g):
                continue
            v = [float(g[f'coverage_scale_{s}'].iloc[0]) for s in SCALES]
            ax.plot(SCALES, v, 'o-', ms=3, lw=1.1, zorder=3,
                    color=CHANNEL_COLORS.get(c, '0.4'), label=c)
        ax.set_title(e, fontsize=9, color=INK)
        ax.set_xticks(SCALES)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('scale (0 finest, 5 coarsest)', fontsize=7.5, color=MUTED)
    for i, ax in enumerate(axes):
        if i % ncol == 0:
            ax.set_ylabel('floor covered by that scale', fontsize=8, color=INK)
    axes[0].legend(fontsize=5, frameon=False, ncol=2)
    fig.suptitle('C2  the coverage each scale reached, and where each threshold '
                 'cuts\ndashed lines are the swept thresholds; a scale below a '
                 'line is deleted at it', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, name)


def fig_size_distributions(banks, envs, thrs, name):
    """C3: the field-size distribution at each threshold.

    Experiment 2's question, asked of each relaxed library. Log area, because
    the distribution spans orders of magnitude and Experiment 2 fits it that
    way. Channels pooled. If relaxing Rule 12 only adds fields at one end, the
    curve grows a shoulder there; if it rescales the whole thing, the shape
    moves.
    """
    cols = tiling_colors(thrs)
    fig, axes, ncol = _panels(envs)
    for ax, e in zip(axes, envs):
        for thr in sorted(thrs, reverse=True):
            parts = [b for (ee, _, t), b in banks.items()
                     if ee == e and np.isclose(t, thr) and len(b)]
            if not parts:
                continue
            a = np.concatenate([b.area_env_m2.to_numpy(dtype=float)
                                for b in parts])
            a = a[np.isfinite(a) & (a > 0)]
            if len(a) < 5:
                continue
            ax.hist(np.log10(a), bins=34, histtype='step', lw=1.4,
                    color=cols[thr], label=f'{tiling_label(thr)}  n={len(a)}')
        ax.set_title(e, fontsize=9, color=INK)
        ax.set_xlabel('log10 field area (m^2)', fontsize=7.5, color=MUTED)
    for i, ax in enumerate(axes):
        if i % ncol == 0:
            ax.set_ylabel('fields, channels pooled', fontsize=8, color=INK)
    axes[0].legend(fontsize=5, frameon=False)
    fig.suptitle('C3  field-size distribution at each coverage threshold\n'
                 'channels pooled, log area', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, name)


def fig_library_growth(summary, envs, thrs, name):
    """C4: how much bigger each library gets, and how many scales it keeps.

    One line per channel. The left panel is the field count against the
    threshold; the right is the number of surviving scales. Read together they
    separate "the rule was deleting a whole scale" from "the rule was trimming
    a few fields".
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    order = sorted(thrs, reverse=True)
    x = np.arange(len(order))
    for ax, col, ylab in ((axes[0], 'n_fields', 'fields in the library'),
                          (axes[1], 'n_scales_kept', 'scales surviving')):
        for e in envs:
            for c in CHANNELS:
                g = summary[(summary.env == e) & (summary.channel == c)]
                if not len(g):
                    continue
                y = [float(g[np.isclose(g.tiling_frac_min, t)][col].iloc[0])
                     if len(g[np.isclose(g.tiling_frac_min, t)]) else np.nan
                     for t in order]
                ax.plot(x, y, '-', lw=0.9, alpha=0.75,
                        color=CHANNEL_COLORS.get(c, '0.4'))
        ax.set_xticks(x)
        ax.set_xticklabels([f'{t:g}' for t in order], fontsize=7)
        ax.set_xlabel('TILING_FRAC_MIN (relaxing to the right)', fontsize=8,
                      color=MUTED)
        ax.set_ylabel(ylab, fontsize=8, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(RULE_GRAY)
    axes[0].set_yscale('log')
    handles = [plt.Line2D([], [], color=CHANNEL_COLORS.get(c, '0.4'), lw=1.4,
                          label=c) for c in CHANNELS]
    axes[1].legend(handles=handles, fontsize=6.5, frameon=False, ncol=2)
    fig.suptitle('C4  library size and surviving scales as Rule 12 is relaxed\n'
                 'one line per arena x channel; left axis is logarithmic',
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, name)


# -------------------------------------------------------------------- report

class CoverageRelaxationReport(ExperimentReport):
    experiment = 'coverage-relaxation'

    def title(self):
        p = self.paired
        if p is None or not len(p):
            return 'no libraries compared'
        bad = int((~self.summary.matches_exp2.fillna(True).astype(bool)).sum())
        warn = f'; {bad} DISAGREE WITH EXPERIMENT 2' if bad else ''
        return (f'Rule 12 off: {p.n_added.sum():+d} fields over '
                f'{len(p)} libraries, median {p.fold_more.median():.2f}x, '
                f'{int((p.n_scales_off > p.n_scales_on).sum())}/{len(p)} gain '
                f'a scale{warn}')

    def figures(self):
        return sorted(FIGURES_WRITTEN)

    def data_files(self):
        return [p for p in (f'{self.out_dir}/summary.csv',
                            f'{self.out_dir}/paired.csv',
                            f'{self.out_dir}/scale_summary.csv',
                            f'{self.out_dir}/fits.csv')
                if os.path.exists(p)]

    def body(self):
        S = self.section
        s, p = self.summary, self.paired
        out = []

        out.append(S('WHAT WAS CHANGED', '\n'.join([
            'One knob: TILING_FRAC_MIN, the coverage threshold in Rule 12 '
            '(tiling stop). A scale whose admitted fields, unioned, cover less '
            'than that fraction of the floor is dropped whole; the surviving '
            'ladder is the contiguous run of qualifying scales around the '
            f'best-covered one. Default {OPERATING_TILING:.2f}.', '',
            f'Swept: {", ".join(tiling_label(t) for t in sorted(TILING_SWEEP, reverse=True))}.',
            '',
            'Everything else is Experiment 2\'s operating point exactly: '
            f'EXTENT_PCTL {PCTL}, ACT_THRESH {THRESH:g}, Rule 2 off, LAMBDA 0, '
            'seed 0.', '',
            'NOTE ON THE RULE NUMBER. Rule 4 in this codebase is spatial '
            'weighting -- merge cost = feature distance + LAMBDA * space -- '
            'and it is already off at the operating point (LAMBDA 0), so '
            'toggling it would change nothing. The coverage rule is Rule 12, '
            'and Rule 12 is what this swept.'])))

        out.append(S('WHY THIS IS A CLEAN ONE-RULE CHANGE', '\n'.join([
            'Rule 12 runs LAST, after Rule 11\'s competition, and only deletes '
            'whole scales from a ladder that is already settled. Nothing '
            'upstream reads the threshold: not the tree, not the candidate '
            'set, not the size window, not which field beats which. So '
            'lowering it can only ADD scales back, and each library is a '
            'strict superset of the one above it.', '',
            'That is checked, not assumed. `nested` in the summary is the '
            'per-library verdict:',
            f'  nested at every step: '
            f'{int(s.nested.fillna(False).astype(bool).sum())}/{len(s)} rows.',
            'A failure means something upstream is reading the knob and this '
            'comparison is void.', '',
            'The same fact makes the sweep cheap: prepare_candidates -- the '
            'Gram matrix, the Ward tree, the candidate measurement, the '
            'environment readout, which is all of the cost -- is built once '
            'per arena and channel and every threshold is scored against an '
            'identical tree. A difference between thresholds cannot be a '
            'difference in the tree.'])))

        ok = s.matches_exp2.fillna(False).astype(bool).sum()
        nocmp = int(s.matches_exp2.isna().sum())
        bad = s[s.matches_exp2 == False]
        out.append(S('SELF-CHECK AGAINST EXPERIMENT 2', '\n'.join(
            [f'At the operating point this script must reproduce Experiment '
             f'2\'s library exactly -- same code, same settings, same seed. '
             f'Verdict: {int(ok)} identical, {len(bad)} different, '
             f'{nocmp} with nothing cached to compare against.'] +
            ([''] + [f'  !! {r.env} {r.channel}: {r.matches_exp2_note}'
                     for r in bad.itertuples()] +
             ['', 'A mismatch means the operating-point column below is not '
              'Experiment 2\'s library, so nothing here can be read as a '
              'comparison against it.'] if len(bad) else
             ['', 'So the operating-point column is Experiment 2\'s library, '
              'and every difference below is Rule 12 and nothing else.']))))

        if p is None or not len(p):
            out.append(S('NO COMPARISON', 'No library had both thresholds.'))
            return '\n'.join(out)

        L = ['Rule 12 off against the operating point, one row per library.', '',
             f'  {"arena":17s} {"chan":8s} {"n on":>6s} {"n off":>7s} '
             f'{"added":>7s} {"x":>6s} {"scales on":>10s} {"scales off":>11s} '
             f'{"CV on":>7s} {"CV off":>7s} {"form on":>11s} {"form off":>11s}']
        for r in p.itertuples():
            L.append(f'  {r.env:17s} {r.channel:8s} {r.n_on:6d} {r.n_off:7d} '
                     f'{r.n_added:+7d} {r.fold_more:6.2f} {r.scales_on:>10s} '
                     f'{r.scales_off:>11s} {r.cv_on:7.1f} {r.cv_off:7.1f} '
                     f'{r.best_form_on:>11s} {r.best_form_off:>11s}'
                     + ('  *' if r.form_changed else ''))
        L += ['', f'Libraries gaining at least one scale: '
                  f'{int((p.n_scales_off > p.n_scales_on).sum())}/{len(p)}.',
              f'Median growth: {p.fold_more.median():.2f}x '
              f'(range {p.fold_more.min():.2f}-{p.fold_more.max():.2f}).',
              f'Best-fitting form changes in {int(p.form_changed.sum())}/{len(p)} '
              f'libraries (* above). The form is the headline of Experiment 2, '
              f'so this is the row that says whether its conclusion depends on '
              f'Rule 12.']
        out.append(S('RULE 12 OFF AGAINST THE OPERATING POINT', '\n'.join(L)))

        cov = [f'coverage_scale_{x}' for x in SCALES]
        if all(c in s for c in cov):
            s0 = s[np.isclose(s.tiling_frac_min, min(TILING_SWEEP))]
            L = ['The coverage each scale actually reached, measured on the '
                 'fields that survived competition. This is the quantity Rule '
                 '12 tests, and it says how close each deletion was: a scale '
                 'at 0.49 and a scale at 0.02 are both dropped at 0.50 and are '
                 'not the same finding.', '',
                 f'  {"arena":17s} {"chan":8s} ' +
                 ' '.join(f'{"s" + str(x):>7s}' for x in SCALES)]
            for r in s0.sort_values(['env_area_m2', 'env', 'channel']).itertuples():
                vals = []
                for x in SCALES:
                    v = getattr(r, f'coverage_scale_{x}')
                    vals.append(f'{v:7.2f}' if np.isfinite(v) else f'{"--":>7s}')
                L.append(f'  {r.env:17s} {r.channel:8s} ' + ' '.join(vals))
            out.append(S('HOW CLOSE EACH SCALE WAS TO THE CUT', '\n'.join(L)))

        keep = ['env', 'channel', 'tiling_frac_min', 'n_fields', 'band_lo',
                'band_hi', 'n_scales_kept', 'cv_area_pct', 'area_median_m2',
                'area_max_min_ratio', 'frac_at_floor', 'nested',
                'matches_exp2']
        out.append(S('Per arena, channel and threshold',
                     self.table(s[[c for c in keep if c in s.columns]],
                                max_rows=220)))
        return '\n'.join(out)


# ----------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--envs', default=','.join(ENVS),
                   help='default: the eight collected arenas')
    p.add_argument('--channels', default=','.join(CHANNELS))
    p.add_argument('--tiling', default=','.join(f'{t:g}' for t in TILING_SWEEP),
                   help='TILING_FRAC_MIN values to sweep. The operating point '
                        f'({OPERATING_TILING:g}) must be among them or there '
                        'is nothing to compare against.')
    p.add_argument('--n-boot', type=int, default=N_BOOT,
                   help='parametric-bootstrap draws for the KS p-value')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rebuild', action='store_true',
                   help='rebuild the sweep even where this script\'s own '
                        'cache already holds it')
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--no-email', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    envs = [e.strip() for e in args.envs.split(',') if e.strip()]
    chans = [c.strip() for c in args.channels.split(',') if c.strip()]
    thrs = sorted({float(t) for t in args.tiling.split(',') if t.strip()},
                  reverse=True)
    base_C = dict(BASE_C, USE_GPU=not args.no_gpu)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    has_op = any(np.isclose(t, OPERATING_TILING) for t in thrs)

    print('=' * 72)
    print('Coverage relaxation | what Rule 12 was deleting')
    print(f'  envs     : {envs}')
    print(f'  channels : {chans}')
    print(f'  sweeping : TILING_FRAC_MIN {thrs}')
    print(f'  fixed    : EXTENT_PCTL {PCTL}, ACT_THRESH {THRESH:g}, '
          f'Rule 2 off, LAMBDA 0, seed {base_C["RANDOM_SEED"]}')
    print(f'  own cache: {BANK_DIR} (Experiment 2\'s is read, never written)')
    if not has_op:
        print(f'  !! the operating point {OPERATING_TILING:g} is not in the '
              f'sweep: no paired comparison and no self-check')
    print('=' * 72, flush=True)

    sum_rows, scale_rows, fit_rows, nest_rows = [], [], [], []
    banks_all, env_geom, missing = {}, {}, []
    device = None
    for e in envs:
        data_path = f'{DATA_DIR}/{e}.h5'
        if not os.path.exists(data_path):
            print(f'\n[{e}] no dataset at {data_path} -- skipping', flush=True)
            missing.append(e)
            continue
        print(f'\n===== {e} =====', flush=True)
        root = ET.parse(f'{XML_DIR}/{e}.xml').getroot()
        xy, _ = None, None
        blocks = None
        # The banks this script needs are its own, so the cache test is its
        # own key -- Experiment 2's presence says nothing about whether this
        # sweep has been built.
        need = args.rebuild or any(
            not os.path.exists(bank_path(e, c, t)) for c in chans for t in thrs)
        if need:
            print('  loading feature blocks (the sweep has to be built)',
                  flush=True)
            blocks, xy = ch.load_channel_blocks(data_path)
            if device is None:
                device = R.pick_device(use_gpu=not args.no_gpu)
        else:
            xy = SD_load_positions(data_path)
        env = R.build_env(xy, root)
        env['role'] = SD.env_role(e)
        env['n_landmarks'] = len(root.findall('landmark'))
        env['aspect'] = (1.0 if env.get('is_circular') else
                         (env['x_max'] - env['x_min']) /
                         (env['y_max'] - env['y_min']))
        env_geom[e] = env
        print(f'  area {env["env_area"]:.1f} m^2, role {env["role"]}, '
              f'{env["n_landmarks"]} landmarks', flush=True)

        for c in chans:
            sweep = build_sweep(e, c, blocks, xy, env, thrs, base_C, device,
                                use_cache=not args.rebuild)
            nested_ok, nrows = check_nested(sweep)
            for r in nrows:
                nest_rows.append(dict(env=e, channel=c, **r))
            verdict, note = (check_against_exp2(e, c, sweep[OPERATING_TILING][0])
                             if has_op else (None, 'operating point not swept'))
            for thr in thrs:
                bank, rep = sweep[thr]
                banks_all[(e, c, thr)] = bank
                row, srows, frows = library_rows(
                    bank, rep, env, e, c, thr, args.n_boot, args.seed)
                row.update(nested=bool(nested_ok), matches_exp2=verdict,
                           matches_exp2_note=note)
                sum_rows.append(row)
                scale_rows.extend(srows)
                fit_rows.extend(frows)
            n_op = len(sweep[OPERATING_TILING][0]) if has_op else 0
            n_off = len(sweep[min(thrs)][0])
            print(f'  [{c}] {n_op} -> {n_off} fields as TILING_FRAC_MIN goes '
                  f'{OPERATING_TILING:g} -> {min(thrs):g}   nested '
                  f'{nested_ok}   matches Exp 2: {verdict} ({note})',
                  flush=True)
        del blocks

    if not sum_rows:
        print('\nNo results. Datasets missing: ' + (', '.join(missing) or 'none'))
        return 1

    summary = pd.DataFrame(sum_rows)
    summary.to_csv(f'{OUT_DIR}/summary.csv', index=False)
    scales = pd.DataFrame(scale_rows)
    if len(scales):
        scales.to_csv(f'{OUT_DIR}/scale_summary.csv', index=False)
    fits = pd.DataFrame(fit_rows)
    if len(fits):
        fits.to_csv(f'{OUT_DIR}/fits.csv', index=False)
    nest = pd.DataFrame(nest_rows)
    if len(nest):
        nest.to_csv(f'{OUT_DIR}/nesting.csv', index=False)
    paired = (paired_table(summary, fits, min(thrs)) if has_op
              else pd.DataFrame())
    if len(paired):
        paired.to_csv(f'{OUT_DIR}/paired.csv', index=False)

    envs_drawn = _envs_in_order(summary)
    print('\nfigures:', flush=True)
    fig_scales_recovered(scales, envs_drawn, thrs, 'C1_scales_recovered.png')
    fig_coverage_per_scale(summary, envs_drawn, thrs,
                           'C2_coverage_per_scale.png')
    fig_size_distributions(banks_all, envs_drawn, thrs,
                           'C3_size_distributions.png')
    fig_library_growth(summary, envs_drawn, thrs, 'C4_library_growth.png')
    prune_orphan_figures()

    rep = CoverageRelaxationReport(env_name=','.join(envs), out_dir=OUT_DIR,
                                   fig_dir=FIG_DIR, results=summary,
                                   log_path=os.environ.get('REALM_LOG_PATH'))
    rep.summary, rep.paired, rep.scales, rep.fits = summary, paired, scales, fits
    if missing:
        print(f'\n!! datasets not found, excluded: {", ".join(missing)}')
    print('\n' + rep.compose(), flush=True)
    if not args.no_email:
        rep.send()
    print(f'\nsummary -> {OUT_DIR}/summary.csv'
          f'\npaired  -> {OUT_DIR}/paired.csv'
          f'\nscales  -> {OUT_DIR}/scale_summary.csv'
          f'\nfigures -> {FIG_DIR}')
    return 0


def SD_load_positions(data_path, n_orientations=8):
    """Positions only, for a cached run that needs no feature blocks."""
    import h5py
    with h5py.File(data_path, 'r') as f:
        xs = np.asarray(f['x'][:], dtype=np.float64)
        ys = np.asarray(f['y'][:], dtype=np.float64)
    n_loc = len(xs) // n_orientations
    return np.stack([xs.reshape(n_loc, n_orientations)[:, 0],
                     ys.reshape(n_loc, n_orientations)[:, 0]],
                    axis=1).astype(np.float32)


if __name__ == '__main__':
    sys.exit(main())
