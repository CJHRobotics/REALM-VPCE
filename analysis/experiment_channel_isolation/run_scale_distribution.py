"""Where our field-size distribution sits against the three published forms.

Experiment 2. Descriptive rather than decisive: it establishes our baseline in
the units Eliav and Harland report, and is a prerequisite for reading
Experiment 3.

The targets
-----------
  Eliav et al. 2021    field size log-normal.
  Harland et al. 2021  Fig 3F-G: negative exponential in the megaspace
                       (r = 0.995; 78% of fields <= 1 m^2), Gaussian in the
                       small environments (r = 0.985).
  Harland Fig 6E       coefficient of variation of field size rising with
                       environment area, ~70 -> 85 -> 101 (per cent).
  Harland              a single field covers ~9-13% of the arena.

The two papers disagree about the functional form, so all three are fitted for
every library and all three goodness-of-fit numbers are reported side by side.
Picking a favourite and reporting only its fit would be the one result this
experiment cannot support.

Goodness of fit is reported three ways, because Harland's statistic and a
model-selection statistic answer different questions:

  r_hist   Pearson r between the binned density and the fitted pdf at the bin
           centres. This is *their* statistic, and the only one comparable to
           the published 0.995 and 0.985. It is also a weak discriminator: on
           a heavy-fine-end histogram all three forms clear r = 0.9, which is
           why it must not be read alone.
  ks       Kolmogorov-Smirnov distance, with a parametric-bootstrap p. The
           analytic p is anticonservative when the parameters were estimated
           from the same sample, so it is not used.
  aic      2k - 2 logL. This is the discriminator; `winner` is its argmin.

The threshold caveat, and why it no longer sweeps
------------------------------------------------
Harland show their exponential fit becomes quasi-linear at a lower
field-detection threshold, so distribution shape is not threshold-independent
and the comparison is meaningless without checking ours.

It has been checked. `EXTENT_PCTL` saturates at 65 (run_field_recovery,
against ideal place cells of known size), and the first full run of this
experiment swept 50 / 65 / 80 and found log-normal winning on AIC at every
setting in all 24 environment x channel libraries. The caveat is answered, so
the default is now the single operating point. `--settings 50:0.5,65:0.5,80:0.5`
re-opens the sweep if anything upstream changes.

`ACT_THRESH` was never the knob and cannot be. Under `SIGMA_MODE = 'quantile'`
sigma is solved so the threshold contour lands at `Q`, the `EXTENT_PCTL`
percentile of a cluster's centroid-distance distribution:

    sigma = sqrt( (Q^2 - d_min^2) / (2 ln(1/T)) )

and substituting back into the mask boundary gives `Q` for any `T`. The
threshold cancels exactly; sweeping it varies sigma and leaves every mask,
field and admission decision untouched. That is why `run_threshold_sweep.py`
was retired (see RETIRED.md). Measured at 24 paired runs: identical field
counts, maximum area difference exactly 0. `--settings 65:0.5,65:0.2`
re-checks it.

Scale
-----
Two of the three targets above are claims about ENVIRONMENT SCALE. Fig 3F-G is
a scale-dependent shape claim -- exponential in the megaspace, Gaussian in the
small environments -- and Fig 6E is CV against enclosure area. Neither can be
read from datasets that hold area constant.

So the primary env list is `AREA_ENVS`, the area sweep: six discs from 4.91 to
314.16 m^2, a 64x range at fixed shape and fixed landmark count, each sampled
at ~N_TARGET positions so sample count is not a covariate. `ENVS`, the six
same-area datasets, is the control that says whether anything other than scale
moves the distribution.

Truncation
----------
Rules 8 and 9 bound field size by construction -- floor at Harland's smallest
measured field (0.023 m^2 in 18.6 m^2), ceiling at 20% of arena area. Every
fit here is therefore to a doubly-truncated sample, and the fraction of fields
resting on each bound is reported beside it. A distribution whose mass piles
against a bound is being described by the rule, not by the model.

Usage
    python run_scale_distribution.py [--envs a,b] [--channels ...]
                                     [--settings 50:0.5,65:0.5,80:0.5,65:0.2]
                                     [--use-cache]
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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch
import rules as R
from realm_tools.experiment_lib.reporting import ExperimentReport

# The six collected datasets: four landmark counts on one disc, plus the two
# geometry arenas. The landmark counts each divide the clock face evenly from
# noon, so no panel sits on a camera view seam.
# Same area (~28.3 m^2), varying cue density and shape. A control: it says
# whether the distribution moves with anything other than scale.
ENVS = ['circ_lm2_r0', 'circ_lm4_r0', 'circ_lm8_r0', 'circ_lm12_r0',
        'rect_lm8_r0', 'corr_lm8_r0']

# Varying area at fixed shape and landmark count: small, medium, mega, in
# Harland's proportions rather than in ours.
#
#   circ_lm8_rad2p0    r = 2    12.57 m^2   small     wall cover 48%
#   circ_lm8_r0        r = 3    28.27 m^2   medium    wall cover 32%
#   circ_lm8_rad6p0    r = 6   113.10 m^2   mega      wall cover 16%
#
# Three, not six, and spanning 9.0x rather than 64x. Harland's megaspace is
# 8.8x their small environment, so a 64x sweep is seven times wider than the
# design it is being compared to, and the extra arenas cost collection without
# buying a comparison. 9.0x is the closest span the built arenas offer, and
# circ_lm8_r0 is already collected, so this needs two new datasets.
#
# We match Harland's RATIO, not their absolute areas, and cannot do otherwise.
# Their megaspace is 18.6 m^2 -- smaller than our medium arena -- and matching
# it in absolute terms would put the small environment near 2.1 m^2, a disc of
# radius 0.82 m. The robot's circumscribing radius is 0.31 m and the
# collection keep-out 0.2 m, so it would barely fit, and eight 0.75 m panels
# need 6 m of wall against a 5.2 m circumference: they would overlap. The
# agent is simply larger relative to its arena than a rat is to a room, so
# ordering and ratio are what transfer.
#
# The remaining built arenas (circ_lm8_rad1p25, rad3p5, rad10p0) still work
# with --envs if a wider span is ever wanted; rad1p25 is the weakest of them,
# with eight panels covering 76% of its circumference.
AREA_ENVS = ['circ_lm8_rad2p0', 'circ_lm8_r0', 'circ_lm8_rad6p0']

CHANNELS = ['hog', 'color', 'spatial', 'lidar', 'visual', 'all']
CHANNEL_COLORS = {'hog': '#1f77b4', 'color': '#d62728', 'spatial': '#2ca02c',
                  'lidar': '#9467bd', 'visual': '#ff7f0e', 'all': '#17becf'}

# (EXTENT_PCTL, ACT_THRESH). One setting: the operating point.
#
# The sweep this used to run has served its purpose and is retired from the
# default. EXTENT_PCTL saturates at 65 -- established by run_field_recovery
# against ideal place cells of known size -- and the first full run of this
# experiment found log-normal winning on AIC at 50, 65 and 80 alike, in all
# 24 environment x channel libraries. The threshold caveat is therefore
# answered rather than open, and re-running the sweep every time buys nothing.
#
# Pass --settings to sweep again if something upstream changes: e.g.
#   --settings 50:0.5,65:0.5,80:0.5      re-open the EXTENT_PCTL sweep
#   --settings 65:0.5,65:0.2             re-check the ACT_THRESH invariance
# Two settings sharing an EXTENT_PCTL still trigger the invariance check.
SETTINGS = [(65, 0.5)]
DEFAULT_PCTL, DEFAULT_T = 65, 0.5

N_BOOT = 200               # parametric-bootstrap draws for the KS p-value
MIN_FIELDS = 10            # below this a three-way fit comparison is noise
BANDS = list(range(6))     # scale-band occupancy reported over bands 0-5

# Published reference values, for the report and the figures.
HARLAND_CV = {'CA1 small': 70.0, 'CA1 medium': 85.0, 'CA1 megaspace': 101.0}
HARLAND_COVERAGE = (0.09, 0.13)
# Harland's megaspace is 8.8x their small environment. Ours is 9.0x, so the
# two spans are directly comparable and their per-quantity changes across that
# span are numbers to hit rather than directions.
HARLAND_AREA_RATIO = 8.8
# Coverage per cell SATURATES: only ~2 percentage points higher in the
# megaspace than in the small environment despite 8.8x the area.
HARLAND_COVERAGE_RISE_PP = 2.0
# Their reported fit quality in the r_hist statistic, and the headline
# shape number from the megaspace.
HARLAND_R_EXPON, HARLAND_R_GAUSS = 0.995, 0.985
HARLAND_FRAC_UNDER_1M2 = 0.78

# Area counts as varied only when it spans at least this ratio. Not
# `nunique() > 1`: the corridor is 28.224 m^2 against the discs' 28.274, a
# 0.2% rounding difference that would otherwise be read as an area axis and
# produce a Fig 6E plot out of points that all share one scale.
AREA_SPAN_MIN = 2.0


def area_varies(s):
    """Does this set of runs span enough area to speak to Harland Fig 6E?"""
    a = s.env_area_m2.dropna()
    return len(a) > 0 and float(a.max()) / float(a.min()) >= AREA_SPAN_MIN

# The quantities tracked across scale: (column, label, expected direction,
# source, what that source says). Expected direction is +1 rises with area,
# 0 flat, or None where the source gives a value rather than a direction and
# a directional verdict would be meaningless. `source` matters: only two of
# these are Harland's, two are Eliav's read backwards from her 6 m control,
# and one is ours alone. Scoring an Eliav-derived expectation as agreement
# with Harland would misattribute the comparison.
SCALE_QUANTITIES = [
    ('cv_area_pct',        'CV of field area (%)',          +1, 'Harland',
     'rises with area, 70 -> 85 -> 101'),
    ('coverage_median',    'arena covered per field',        0,  'Harland',
     'SATURATES: ~2 pp higher in megaspace despite 8.8x the area'),
    ('area_median_m2',     'median field area (m^2)',       +1, 'Eliav',
     'mean field size fell 5.9 -> 1.5 m in a 6 m segment of the same tunnel, '
     'so size should fall in a smaller space'),
    ('area_max_min_ratio', 'max/min field area',            +1, 'Eliav',
     'within-neuron size ratio fell 4.4 -> 1.6 in the 6 m segment. Ours is a '
     'POPULATION spread, not a within-neuron one'),
    ('n_bands_occupied',   'scale bands occupied',          +1, 'ours',
     'neither paper measures this; our own index of scale diversity'),
    ('area_median_over_floor', 'median field / Rule 8 floor', None, 'ours',
     'a CONTROL, not a claim. The Rule 8 floor is a fixed fraction of arena '
     'area, so it grows with the arena. If this is flat, the rule is setting '
     'field size and any rise in absolute size is the floor moving, not the '
     'model responding to scale'),
    ('frac_under_1m2',     'fraction of fields <= 1 m^2', None, 'Harland',
     '78% in the megaspace -- a value to compare at the mega end, not a '
     'direction'),
]

# ------------------------------------------------------------------ fitting

# floc = 0 for the two positive-support forms: field area cannot be negative
# and a free location parameter would let each fit slide its own origin, which
# makes the AIC comparison between forms meaningless.
FORMS = {
    'lognormal':   dict(dist=stats.lognorm, kw=dict(floc=0), k=2),
    'exponential': dict(dist=stats.expon,   kw=dict(floc=0), k=1),
    'gaussian':    dict(dist=stats.norm,    kw=dict(),       k=2),
}


def _hist(x):
    """Binned density and bin centres, Freedman-Diaconis with a floor.

    Harland's r is computed against a histogram, so one has to be built. The
    bin count is the one free choice in that statistic and it is not
    innocuous -- too few bins and every form fits -- so it is derived from
    the data rather than fixed, and reported.
    """
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    w = 2.0 * iqr / np.cbrt(len(x)) if iqr > 0 else 0.0
    nb = int(np.clip(np.ceil((x.max() - x.min()) / w) if w > 0 else 0, 8, 60))
    dens, edges = np.histogram(x, bins=nb, density=True)
    return dens, 0.5 * (edges[:-1] + edges[1:]), nb


def fit_forms(x, n_boot=N_BOOT, seed=0):
    """Fit all three forms to one sample and score each three ways."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < MIN_FIELDS:
        return []
    dens, centres, nb = _hist(x)
    rng = np.random.default_rng(seed)
    out = []
    for name, spec in FORMS.items():
        dist, k = spec['dist'], spec['k']
        try:
            par = dist.fit(x, **spec['kw'])
        except Exception:
            continue
        logpdf = dist.logpdf(x, *par)
        if not np.all(np.isfinite(logpdf)):
            continue
        ll = float(logpdf.sum())

        # Harland's statistic: fitted pdf against the binned density.
        pdf_c = dist.pdf(centres, *par)
        r_hist = (float(np.corrcoef(dens, pdf_c)[0, 1])
                  if np.std(pdf_c) > 0 and np.std(dens) > 0 else np.nan)

        # KS with a parametric-bootstrap p. Refitting on each synthetic
        # sample is the point: it reproduces the same parameter-estimation
        # optimism the observed statistic carries, which the analytic p does
        # not.
        ks = float(stats.kstest(x, dist.cdf, args=par).statistic)
        n_ge = 0
        for _ in range(n_boot):
            xs = dist.rvs(*par, size=len(x), random_state=rng)
            if name != 'gaussian':
                xs = xs[xs > 0]
                if len(xs) < MIN_FIELDS:
                    continue
            try:
                ps = dist.fit(xs, **spec['kw'])
                n_ge += stats.kstest(xs, dist.cdf, args=ps).statistic >= ks
            except Exception:
                pass
        out.append(dict(form=name, n=len(x), n_bins=nb,
                        params=[float(v) for v in par],
                        loglik=ll, k=k, aic=2 * k - 2 * ll,
                        r_hist=r_hist, ks=ks,
                        ks_p_boot=(n_ge + 1) / (n_boot + 1)))
    if out:
        best = min(out, key=lambda d: d['aic'])['form']
        for d in out:
            d['winner'] = best
            d['d_aic'] = d['aic'] - min(o['aic'] for o in out)
    return out


# --------------------------------------------------------------- statistics

def describe(bank, env, C):
    """The non-fit descriptors: CV, extremes, bands, coverage, truncation."""
    a = bank.area_env_m2.to_numpy(dtype=float)
    r = bank.radius_env_m.to_numpy(dtype=float)
    area = float(env['env_area'])
    cov = a / area

    # Rules 8/9 in the same units, to say how much of the distribution is the
    # rule rather than the model.
    a_floor = C['RULE8_AREA_FRAC'] * area
    a_ceil = C['RULE9_AREA_FRAC'] * area

    d = dict(
        n_fields=len(a), env_area_m2=area,
        # The two axes that vary across the six datasets. Area does not:
        # it is held at ~28.3 m^2 so the geometry arenas are a shape control.
        aspect=float(env.get('aspect', 1.0)),
        n_landmarks=float(env.get('n_landmarks', np.nan)),
        # Harland report the CV as a percentage; 70/85/101 are per cent.
        cv_area_pct=100.0 * a.std(ddof=1) / a.mean() if len(a) > 1 else np.nan,
        cv_radius_pct=100.0 * r.std(ddof=1) / r.mean() if len(r) > 1 else np.nan,
        area_min_m2=float(a.min()), area_median_m2=float(np.median(a)),
        area_max_m2=float(a.max()), area_max_min_ratio=float(a.max() / a.min()),
        radius_min_m=float(r.min()), radius_median_m=float(np.median(r)),
        radius_max_m=float(r.max()), radius_max_min_ratio=float(r.max() / r.min()),
        coverage_median=float(np.median(cov)),
        coverage_q25=float(np.percentile(cov, 25)),
        coverage_q75=float(np.percentile(cov, 75)),
        frac_in_harland_coverage=float(np.mean((cov >= HARLAND_COVERAGE[0]) &
                                               (cov <= HARLAND_COVERAGE[1]))),
        # Harland's headline shape number, quoted directly. Only comparable
        # in an arena of comparable size, so env_area is carried beside it.
        frac_under_1m2=float(np.mean(a <= 1.0)),
        rule8_floor_m2=float(a_floor), rule9_ceiling_m2=float(a_ceil),
        frac_at_floor=float(np.mean(a <= a_floor * 1.05)),
        frac_at_ceiling=float(np.mean(a >= a_ceil * 0.95)),
        # Field size in units of the Rule 8 floor. The floor is a FRACTION of
        # arena area, so it grows with the arena: if fields merely sit on it,
        # their absolute size rises with area for reasons that have nothing to
        # do with the model, and "field size grows with the space" is an
        # artifact of the rule. This ratio is the control for that -- flat
        # means the floor is setting the scale, rising means the model is.
        area_median_over_floor=float(np.median(a) / a_floor),
        frac_within_2x_floor=float(np.mean(a <= 2.0 * a_floor)),
        # Harland's typical field covers 9-13%; this is how far below that a
        # median field sits, at whatever arena size.
        coverage_shortfall_x=float(
            np.mean(HARLAND_COVERAGE) / (float(np.median(a)) / area)),
    )
    band = bank.scale_band.to_numpy(dtype=int)
    for b in BANDS:
        d[f'band{b}_frac'] = float(np.mean(band == b))
    d['band6plus_frac'] = float(np.mean(band > max(BANDS)))
    d['n_bands_occupied'] = int(len(np.unique(band)))
    return d


def band_table(bank, env, C, tag):
    """One row per scale band: the library is a tiling at every scale.

    A tiling at scale s needs about arena/s tiles, so the finest band always
    holds most of the library and always sets any pooled median, mean or CV.
    Measured on the first full sweep: band 0 is 61-65% of every channel's
    library, and the pooled median coverage (0.26%) is simply band 0's.

    Per band the picture is different -- bands 4 and 5 sit at 7-9% and 15-17%
    of the arena, bracketing Harland's 9-13% -- so the model does reach their
    scale and the pooled statistic hides it. Report both; the per-band numbers
    are the ones comparable to a recorded sample of cells.
    """
    if not len(bank):
        return []
    area = float(env['env_area'])
    rows = []
    for band, g in bank.groupby('scale_band'):
        a = g.area_env_m2.to_numpy(dtype=float)
        rows.append(dict(
            tag, scale_band=int(band), n_fields=len(a),
            share_of_library=float(len(a) / len(bank)),
            area_median_m2=float(np.median(a)),
            coverage_median=float(np.median(a) / area),
            # Total floor this band lays down, as a multiple of the arena.
            # Around 1.0 means the band tiles it once.
            tiling_multiple=float(a.sum() / area),
            cv_area_pct=(100.0 * a.std(ddof=1) / a.mean()
                         if len(a) > 1 else np.nan),
            split_half_iou_median=(float(g.split_half_iou.median())
                                   if 'split_half_iou' in g else np.nan),
        ))
    return rows


def scale_trends(summary):
    """How each tracked quantity moves across arena area.

    Three arenas is too few for a meaningful per-channel correlation -- a
    Spearman rho over three points takes one of four values -- so the trend is
    read two ways: the mega/small ratio per channel, which is the effect size,
    and a Spearman pooled over every (channel, arena) point, which is the only
    place there are enough points to test a direction.
    """
    rows = []
    if summary.env_area_m2.nunique() < 2:
        return pd.DataFrame(rows)
    a_lo, a_hi = summary.env_area_m2.min(), summary.env_area_m2.max()
    for col, label, expect_dir, source, says in SCALE_QUANTITIES:
        if col not in summary.columns:
            continue
        d = summary[['channel', 'env_area_m2', col]].dropna()
        if not len(d):
            continue
        rho, pval = (stats.spearmanr(d.env_area_m2, d[col])
                     if d.env_area_m2.nunique() > 1 else (np.nan, np.nan))
        lo = d[d.env_area_m2 == a_lo][col]
        hi = d[d.env_area_m2 == a_hi][col]
        lo_m = float(lo.median()) if len(lo) else np.nan
        hi_m = float(hi.median()) if len(hi) else np.nan
        rows.append(dict(
            quantity=col, label=label, expected_direction=expect_dir,
            source=source, source_says=says,
            area_small=float(a_lo), area_mega=float(a_hi),
            value_small=lo_m, value_mega=hi_m,
            ratio_mega_small=hi_m / lo_m if lo_m else np.nan,
            delta_mega_small=hi_m - lo_m,
            spearman_rho=float(rho), spearman_p=float(pval),
            # Our direction, with a deadband: a change under 10% across a 9x
            # area span is flat, whatever its sign or its p-value.
            direction=(0 if not np.isfinite(hi_m / lo_m if lo_m else np.nan)
                       or abs(hi_m / lo_m - 1) < 0.10
                       else (1 if hi_m > lo_m else -1)),
        ))
    out = pd.DataFrame(rows)
    if len(out):
        # NaN, not False, where the source gives no direction to agree with.
        # pd.isna, not `is not None`: a column mixing ints and None becomes
        # float with NaN, and `direction == NaN` is False rather than unknown,
        # which would score a value-only claim as a divergence.
        out['agrees'] = [
            None if pd.isna(r.expected_direction)
            else bool(r.direction == r.expected_direction)
            for r in out.itertuples()]
    return out


# ------------------------------------------------------------ bank building

def build_banks(env_name, cname, blocks, xy, env, settings, base_C, device,
                out_dir, use_cache, verbose=True):
    """Field libraries for one channel at every setting in the sweep.

    The Gram matrix and the Ward tree depend on neither EXTENT_PCTL nor
    ACT_THRESH, so both are computed once and reused across the sweep. That
    is not only cheaper: it makes the settings strictly like-for-like, every
    one scored against an identical tree.
    """
    want = {(p, t): f'{out_dir}/{env_name}/{cname}_p{p}_t{t:g}_bank.csv'
            for p, t in settings}
    if use_cache and all(os.path.exists(v) for v in want.values()):
        print(f'  [{cname}] cached banks', flush=True)
        return {k: pd.read_csv(v) for k, v in want.items()}
    os.makedirs(f'{out_dir}/{env_name}', exist_ok=True)

    t0 = time.time()
    X = ch.assemble(blocks, ch.CHANNEL_SETS[cname], normalize=True)
    D2 = R.feature_sq_distances(X, device=device, verbose=verbose)
    rng = np.random.default_rng(base_C['RANDOM_SEED'])
    feat_med = R._median_offdiag(D2, rng)
    d2xy = ((xy[:3000, None, :] - xy[None, :3000, :]) ** 2).sum(-1)
    xy_med = float(np.median(d2xy[np.triu_indices(len(d2xy), 1)]))
    tree = R.build_tree(D2, xy, feat_med, xy_med, cfg=base_C, verbose=verbose)

    banks = {}
    for p, t in settings:
        C = R.resolve_cfg(dict(base_C, EXTENT_PCTL=p, ACT_THRESH=t))
        ctx = R.prepare_candidates(X, xy, env, D2, feat_med, xy_med, cfg=C,
                                   device=device, tree=tree,
                                   tag=f'{env_name}/{cname}/p{p}t{t:g}',
                                   verbose=verbose)
        bank, _, rep = R.admit_fields(ctx, cfg=C, verbose=verbose)
        bank.to_csv(want[(p, t)], index=False)
        with open(f'{out_dir}/{env_name}/{cname}_p{p}_t{t:g}_report.json', 'w') as f:
            json.dump({k: v for k, v in rep.items()
                       if not isinstance(v, np.ndarray)}, f, indent=2,
                      default=float)
        banks[(p, t)] = bank
        print(f'  [{cname}] pctl {p} T {t:g}: {len(bank)} fields', flush=True)
    print(f'  [{cname}] {len(settings)} settings in {time.time()-t0:.0f}s',
          flush=True)
    del X, D2
    return banks


def threshold_invariance(banks):
    """Do the two ACT_THRESH points at one EXTENT_PCTL agree exactly?

    They must, by the algebra in the module docstring. Reported rather than
    assumed: a non-zero difference means the identity has been broken and
    every threshold claim in this experiment is void.
    """
    pairs = {}
    for (p, t) in banks:
        pairs.setdefault(p, []).append(t)
    rows = []
    for p, ts in pairs.items():
        if len(ts) < 2:
            continue
        ref = banks[(p, max(ts))]
        for t in sorted(ts)[:-1]:
            b = banks[(p, t)]
            same_n = len(b) == len(ref)
            if same_n and len(b):
                d = float(np.abs(np.sort(b.area_env_m2.to_numpy()) -
                                 np.sort(ref.area_env_m2.to_numpy())).max())
            else:
                d = np.nan
            rows.append(dict(extent_pctl=p, act_thresh_a=t,
                             act_thresh_b=max(ts), n_a=len(b), n_b=len(ref),
                             same_n_fields=bool(same_n), max_abs_area_diff=d))
    return rows


# ------------------------------------------------------------------ figures

# Figures written by this run. The report attaches these rather than globbing
# the directory: figures come and go as the experiment changes, and a glob
# picks up orphans from earlier runs -- which is how a retired threshold sweep
# and a single-arena coverage plot ended up attached to a three-arena report.
FIGURES_WRITTEN = []


def _save(fig, fig_dir, name):
    p = os.path.join(fig_dir, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    FIGURES_WRITTEN.append(p)
    print(f'  {p}', flush=True)


def prune_orphan_figures(fig_dir):
    """Delete figures this experiment used to produce and no longer does.

    Scoped to `S<digit>*.png` in this experiment's own figure directory, all
    of which are regenerated on every run, so nothing unrecoverable is at
    risk. Without it the directory accumulates plots from retired figures and
    from earlier single-arena runs, and they are indistinguishable from
    current output when someone opens the folder later.
    """
    keep = {os.path.abspath(p) for p in FIGURES_WRITTEN}
    removed = []
    for p in sorted(glob.glob(os.path.join(fig_dir, 'S[0-9]*.png'))):
        if os.path.abspath(p) not in keep:
            os.remove(p)
            removed.append(os.path.basename(p))
    if removed:
        print(f'  pruned {len(removed)} orphaned figure(s): '
              f'{", ".join(removed)}', flush=True)
    return removed


def fig_distributions(banks_all, fits, envs, chans, fig_dir):
    """S1: the size histogram per env x channel with all three fits drawn."""
    envs = [e for e in envs if any((e, c, DEFAULT_PCTL, DEFAULT_T) in banks_all
                                   for c in chans)]
    if not envs:
        return
    fig, axes = plt.subplots(len(envs), len(chans), squeeze=False,
                             figsize=(3.0 * len(chans), 2.4 * len(envs)))
    for i, e in enumerate(envs):
        for j, c in enumerate(chans):
            ax = axes[i][j]
            b = banks_all.get((e, c, DEFAULT_PCTL, DEFAULT_T))
            if b is None or len(b) < MIN_FIELDS:
                ax.set_axis_off()
                continue
            x = b.area_env_m2.to_numpy(dtype=float)
            dens, centres, nb = _hist(x)
            ax.bar(centres, dens, width=(centres[1] - centres[0]) * 0.9,
                   color='0.82', edgecolor='none')
            gx = np.linspace(x.min(), x.max(), 300)
            sub = fits[(fits.env == e) & (fits.channel == c) &
                       (fits.extent_pctl == DEFAULT_PCTL) &
                       (fits.act_thresh == DEFAULT_T) & (fits.variable == 'area')]
            for _, row in sub.iterrows():
                par = json.loads(row.params)
                ax.plot(gx, FORMS[row.form]['dist'].pdf(gx, *par), lw=1.4,
                        label=f"{row.form[:4]} r={row.r_hist:.3f}")
            ax.set_title(f'{e}\n{c}' if i == 0 else c, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5, frameon=False)
            if j == 0:
                ax.set_ylabel('density', fontsize=7)
            if i == len(envs) - 1:
                ax.set_xlabel('field area (m$^2$)', fontsize=7)
    fig.suptitle('S1  field-size distribution with all three published forms '
                 f'(EXTENT_PCTL {DEFAULT_PCTL})', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, fig_dir, 'S1_size_distributions.png')


def _scale_panel(ax, s, col, ylabel, pct=False):
    """One quantity against arena area, a line per channel."""
    for c in sorted(s.channel.unique()):
        g = s[s.channel == c].sort_values('env_area_m2')
        if len(g):
            ax.plot(g.env_area_m2, 100 * g[col] if pct else g[col], 'o-',
                    ms=5, lw=1.2, color=CHANNEL_COLORS.get(c, '0.4'), label=c)
    ax.set_xlabel('arena area (m$^2$)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def fig_cv(summary, fig_dir):
    """S2: CV of field size against arena area — Harland Fig 6E.

    Their claim is directional: CV rises with enclosure area, 70 -> 85 -> 101
    across a span of 8.8x. Ours spans 9.0x, so the comparison is like for
    like. Read the direction first; landing on their absolute values is not
    expected from a different agent in a different arena.

    Falls back to cue density and shape when area is not varied, so a control
    run is not mislabelled as a reading of 6E.
    """
    s = summary[(summary.extent_pctl == DEFAULT_PCTL) &
                (summary.act_thresh == DEFAULT_T)]
    if not len(s):
        return
    if not area_varies(s):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        for ax, (xcol, xlabel, d) in zip(axes, [
                ('n_landmarks', 'landmark count (disc)', s[s.aspect == 1.0]),
                ('aspect', 'aspect ratio (8 landmarks)', s[s.n_landmarks == 8])]):
            for c in sorted(d.channel.unique()):
                g = d[d.channel == c].sort_values(xcol)
                if len(g):
                    ax.plot(g[xcol], g.cv_area_pct, 'o-', ms=5, lw=1.2,
                            color=CHANNEL_COLORS.get(c, '0.4'), label=c)
            ax.set_xlabel(xlabel); ax.set_ylabel('CV of field area (%)')
            ax.set_ylim(bottom=0)
            for k, v in HARLAND_CV.items():
                ax.axhline(v, color='k', ls=':', lw=0.9)
                ax.text(ax.get_xlim()[1], v, f'  {k} {v:g}', fontsize=6, va='center')
            ax.legend(fontsize=7, frameon=False, ncol=2)
        fig.suptitle('S2  CV of field size. Area is held constant in this run, '
                     'so the dotted Harland Fig 6E\nvalues are a reference '
                     'scale, not a trend to fit.', fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        _save(fig, fig_dir, 'S2_cv.png')
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _scale_panel(ax, s, 'cv_area_pct', 'CV of field area (%)')
    for k, v in HARLAND_CV.items():
        ax.axhline(v, color='k', ls=':', lw=0.9)
        ax.text(ax.get_xlim()[1], v, f'  {k} {v:g}', fontsize=7, va='center')
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title('S2  CV of field size against arena area — Harland Fig 6E\n'
                 'their claim is that this RISES; dotted lines are their '
                 '70 / 85 / 101', fontsize=9)
    _save(fig, fig_dir, 'S2_cv.png')


def fig_field_maps(banks_all, envs_by_area, chans, env_geom, fig_dir):
    """S3: the admitted fields drawn on the arena, coloured by scale band.

    The picture behind the per-band table. Each field is its Rule 7 ellipse --
    two axes and an orientation -- at its measured position, so the multiscale
    tiling is visible directly: a dense carpet of band-0 fields with
    progressively fewer, larger ones above it.

    Every panel is drawn to its own arena, with a 1 m bar for scale, because
    the arenas span 9x in area and a common frame would render the smallest
    one unreadable. Field size RELATIVE to the arena is what the eye should
    compare across a row.
    """
    if not banks_all:
        return
    n_r, n_c = len(envs_by_area), len(chans)
    fig, axes = plt.subplots(n_r, n_c, squeeze=False,
                             figsize=(2.7 * n_c, 2.9 * n_r))
    cmap = plt.cm.viridis
    nb = max(BANDS) + 1
    for i, e in enumerate(envs_by_area):
        geom = env_geom.get(e, {})
        for j, c in enumerate(chans):
            ax = axes[i][j]
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            b = banks_all.get((e, c, DEFAULT_PCTL, DEFAULT_T))
            R_ = geom.get('env_R')
            if R_ is not None:
                ax.add_patch(plt.Circle((geom.get('env_cx', 0.0),
                                         geom.get('env_cy', 0.0)), R_,
                                        fill=False, color='0.2', lw=1.2))
                lim = R_ * 1.08
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            else:
                x0, x1 = geom.get('x_min', -1), geom.get('x_max', 1)
                y0, y1 = geom.get('y_min', -1), geom.get('y_max', 1)
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           fill=False, color='0.2', lw=1.2))
                mx = 0.04 * max(x1 - x0, y1 - y0)
                ax.set_xlim(x0 - mx, x1 + mx); ax.set_ylim(y0 - mx, y1 + mx)
            for lm in geom.get('landmarks', []):
                ax.plot(lm[0], lm[1], 's', ms=3, color='#d64545', zorder=5)
            if b is not None and len(b):
                # Coarsest first, so the fine carpet is drawn on top of the
                # large fields rather than hidden beneath them.
                for _, r in b.sort_values('radius_env_m', ascending=False).iterrows():
                    ax.add_patch(Ellipse(
                        (r.centroid_x, r.centroid_y),
                        2 * r.semi_major_m, 2 * r.semi_minor_m,
                        angle=np.degrees(r.orientation_rad),
                        facecolor=cmap(min(int(r.scale_band), nb - 1) / max(nb - 1, 1)),
                        edgecolor='none', alpha=0.30, lw=0))
                ax.text(0.02, 0.02, f'{len(b)}', transform=ax.transAxes,
                        fontsize=6, color='0.35')
            # 1 m scale bar, so the arenas stay comparable despite the framing
            if R_ is not None or 'x_min' in geom:
                xl = ax.get_xlim(); yl = ax.get_ylim()
                x_s = xl[0] + 0.06 * (xl[1] - xl[0])
                y_s = yl[0] + 0.06 * (yl[1] - yl[0])
                ax.plot([x_s, x_s + 1.0], [y_s, y_s], '-', color='k', lw=1.6)
                ax.text(x_s, y_s + 0.02 * (yl[1] - yl[0]), '1 m', fontsize=5)
            if i == 0:
                ax.set_title(c, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{e.replace('circ_lm8_', '')}\n"
                              f"{geom.get('env_area', float('nan')):.0f} m$^2$",
                              fontsize=8)
    handles = [plt.Line2D([], [], marker='o', ls='', color=cmap(k / max(nb - 1, 1)),
                          label=f'band {k}') for k in range(nb)]
    fig.legend(handles=handles, loc='lower center', ncol=nb, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('S3  admitted fields drawn on the arena, coloured by scale '
                 'band\neach panel to its own arena with a 1 m bar; the count '
                 'is bottom-left', fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    _save(fig, fig_dir, 'S3_field_maps.png')


def fig_form_vs_scale(fits, fig_dir):
    """S4: does the winning form change with scale? — Harland Fig 3F-G.

    Their 3F-G is not one claim but a contrast: a negative exponential in the
    megaspace against a Gaussian in the small environments. That is a
    scale-DEPENDENT form, and it is the only figure here that can test it. A
    single form winning at every area is a divergence from Harland and an
    agreement with Eliav, who fit one form throughout.
    """
    f = fits[(fits.variable == 'area') & (fits.extent_pctl == DEFAULT_PCTL) &
             (fits.act_thresh == DEFAULT_T)]
    if not len(f) or f.env_area_m2.nunique() < 2:
        return
    areas = sorted(f.env_area_m2.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    for name in FORMS:
        d = f[f.form == name].groupby('env_area_m2')
        axes[0].plot(areas, [d.get_group(a).r_hist.median() if a in d.groups
                             else np.nan for a in areas], 'o-', ms=5, label=name)
        axes[1].plot(areas, [d.get_group(a).d_aic.median() if a in d.groups
                             else np.nan for a in areas], 'o-', ms=5, label=name)
    axes[0].axhline(HARLAND_R_EXPON, color='k', ls=':', lw=0.9)
    axes[0].axhline(HARLAND_R_GAUSS, color='k', ls='--', lw=0.9)
    axes[0].set_ylabel("median $r_{hist}$")
    axes[0].set_title("fit quality in Harland's own statistic\n"
                      'dotted 0.995 their exponential, dashed 0.985 their '
                      'Gaussian', fontsize=8)
    axes[1].set_ylabel('median $\\Delta$AIC from the best form')
    axes[1].set_title('model selection (0 = winner at that area)', fontsize=8)
    for ax in axes:
        ax.set_xlabel('arena area (m$^2$)')
        ax.legend(fontsize=7, frameon=False)
    axes[1].set_ylim(bottom=0)
    won = f[f.d_aic == 0].groupby('env_area_m2').form.agg(
        lambda v: v.value_counts().idxmax())
    fig.suptitle('S4  which distribution form wins, against area — Harland '
                 'Fig 3F-G predicts this CHANGES\n(exponential in the '
                 'megaspace, Gaussian in the small environments).   winner: '
                 + ',  '.join(f'{a:.0f} m$^2$ {w}' for a, w in won.items()),
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, fig_dir, 'S4_form_vs_scale.png')


def fig_size_vs_scale(summary, fig_dir):
    """S5: the size ladder against area.

    Median field size, the max/min spread and the number of occupied scale
    bands. Together these say whether a larger space buys a wider range of
    scales or merely a uniformly coarser one -- which is the question
    Experiment 3 is built on, and the reason this experiment is its
    prerequisite.
    """
    s = summary[(summary.extent_pctl == DEFAULT_PCTL) &
                (summary.act_thresh == DEFAULT_T)]
    if not len(s) or not area_varies(s):
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    # Median alone, no min-max band. On a linear axis the largest field is two
    # orders of magnitude above the median, so a shaded range flattens the
    # line it is meant to annotate. The spread is panel 2's job.
    _scale_panel(axes[0], s, 'area_median_m2', 'median field area (m$^2$)')
    _scale_panel(axes[1], s, 'area_max_min_ratio', 'max / min field area')
    _scale_panel(axes[2], s, 'n_bands_occupied', 'scale bands occupied')
    axes[0].set_title('typical field size', fontsize=8)
    axes[1].set_title('size spread', fontsize=8)
    axes[2].set_title('scale diversity', fontsize=8)
    for ax in axes:
        ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle('S5  the size ladder against arena area — does a larger space '
                 'buy a WIDER range of scales, or a uniformly coarser one?',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, fig_dir, 'S5_size_vs_scale.png')


# -------------------------------------------------------------------- report

class ScaleDistributionReport(ExperimentReport):
    """Emailed summary. Leads with the shape, not with a winning form: the
    two source papers disagree about the form, so 'which of the three wins'
    is the weaker question and is reported second."""

    experiment = 'scale-distribution'

    def title(self):
        s = self.results
        if s is None or not len(s):
            return 'no results'
        base = s[(s.extent_pctl == DEFAULT_PCTL) & (s.act_thresh == DEFAULT_T)]
        if not len(base):
            return f'{len(s)} runs'
        w = getattr(self, 'winners', None)
        top = (f'{w.idxmax()} wins {int(w.max())}/{int(w.sum())}'
               if w is not None and len(w) and w.sum() else 'no fit')
        return (f'CV {base.cv_area_pct.median():.0f}%, '
                f'max/min {base.area_max_min_ratio.median():.1f}x, {top}')

    def figures(self):
        # What this run wrote, not what is lying in the directory.
        return sorted(FIGURES_WRITTEN)

    def data_files(self):
        return [p for p in (f'{self.out_dir}/summary.csv',
                            f'{self.out_dir}/fits.csv',
                            f'{self.out_dir}/scale_trends.csv',
                            f'{self.out_dir}/band_summary.csv',
                            f'{self.out_dir}/threshold_invariance.csv')
                if os.path.exists(p)]

    def body(self):
        s = self.results
        if s is None or not len(s):
            return 'No field libraries were produced.'
        S = self.section
        base = s[(s.extent_pctl == DEFAULT_PCTL) & (s.act_thresh == DEFAULT_T)]
        f = self.fits[(self.fits.variable == 'area') &
                      (self.fits.act_thresh == DEFAULT_T)]
        fb = f[f.extent_pctl == DEFAULT_PCTL]
        out = []

        # --- the shape, which is the actual question ----------------------
        w = self.winners
        shape = [
            f'{len(base)} environment x channel libraries at the operating '
            f'point (EXTENT_PCTL {DEFAULT_PCTL}), '
            f'{int(base.n_fields.sum())} fields in total.', '',
            f'CV of field area   median {base.cv_area_pct.median():.0f}% '
            f'(range {base.cv_area_pct.min():.0f}-{base.cv_area_pct.max():.0f}%). '
            f'Harland: {", ".join(f"{k} {v:g}" for k, v in HARLAND_CV.items())}.',
            f'max/min ratio      median {base.area_max_min_ratio.median():.1f}x '
            f'(range {base.area_max_min_ratio.min():.1f}-'
            f'{base.area_max_min_ratio.max():.1f}x).',
            f'fields <= 1 m^2    median {100*base.frac_under_1m2.median():.0f}% '
            f'against Harland\'s {100*HARLAND_FRAC_UNDER_1M2:.0f}% in the '
            f'megaspace. Comparable only at comparable arena size; our arenas '
            f'span {base.env_area_m2.min():.0f}-{base.env_area_m2.max():.0f} m^2.',
            f'coverage per field median {100*base.coverage_median.median():.1f}% '
            f'of the arena, against Harland\'s '
            f'{100*HARLAND_COVERAGE[0]:g}-{100*HARLAND_COVERAGE[1]:g}%; '
            f'{100*base.frac_in_harland_coverage.median():.0f}% of fields fall '
            f'inside that band.', '',
            'The shared signature in both papers is a heavy fine end with a '
            'thin coarse tail. Matching that matters more than matching a '
            'functional form, which is fortunate, because the two papers '
            'report different forms for the same quantity.']
        out.append(S('SHAPE', '\n'.join(shape)))

        # --- the three forms ----------------------------------------------
        forms = []
        if w is not None and w.sum():
            forms.append('Runs won on AIC at the operating point: ' +
                         ', '.join(f'{k} {int(v)}' for k, v in w.items()) + '.')
        for name in FORMS:
            d = fb[fb.form == name]
            if not len(d):
                continue
            forms.append(
                f'{name:12s} median r_hist {d.r_hist.median():.3f}   '
                f'median KS {d.ks.median():.3f}   '
                f'KS bootstrap p >= 0.05 in {int((d.ks_p_boot >= 0.05).sum())}'
                f'/{len(d)} runs   median dAIC {d.d_aic.median():.0f}')
        forms += ['',
                  f'Harland report r = {HARLAND_R_EXPON} for the exponential '
                  f'in the megaspace and r = {HARLAND_R_GAUSS} for the '
                  f'Gaussian in the small environments. r_hist above is the '
                  f'same statistic, so those numbers are the ones to compare '
                  f'against. Read it beside dAIC: r_hist is computed on a '
                  f'binned histogram and is a weak discriminator — all three '
                  f'forms routinely clear 0.9 on a heavy-fine-end sample.',
                  '',
                  'Expect the KS bootstrap to reject every form once a library '
                  'runs to ~1000 fields: at that n a KS distance of 0.05 is '
                  'already significant, and no two-parameter form describes a '
                  'real distribution that closely. "All three rejected" is the '
                  'normal outcome and is not a failure — it is why dAIC, which '
                  'ranks the forms rather than testing them, is the statistic '
                  'the winner is taken from.']
        out.append(S('THE THREE FORMS', '\n'.join(forms)))

        # --- the threshold caveat, now settled ----------------------------
        inv = self.invariance
        cav = [
            'Harland show their exponential fit becomes quasi-linear at a '
            'lower detection threshold, so distribution shape is not '
            'threshold-independent and the comparison is meaningless without '
            'checking ours. It has been checked, and it is settled:', '',
            '  EXTENT_PCTL saturates at 65, established by run_field_recovery '
            'against ideal place cells of known size.',
            '  The first full run of this experiment swept 50 / 65 / 80 and '
            'found log-normal winning on AIC at every setting, in all 24 '
            'environment x channel libraries.', '',
            'So the sweep no longer runs by default. Re-open it with '
            '--settings 50:0.5,65:0.5,80:0.5 if anything upstream changes.', '',
            'ACT_THRESH is not the knob and cannot be. Under SIGMA_MODE = '
            '"quantile" sigma is solved so the ACT_THRESH contour lands at '
            'the EXTENT_PCTL quantile, and the threshold cancels exactly from '
            'the mask boundary — sweeping it varies sigma and changes nothing '
            'else, which is why run_threshold_sweep.py was retired. Measured '
            'once at 24 paired runs: identical field counts, maximum area '
            'difference exactly 0.']
        if inv is not None and len(inv):
            bad = inv[~inv.same_n_fields | (inv.max_abs_area_diff.fillna(1) > 1e-9)]
            cav += ['',
                    f'Re-checked in this run, {len(inv)} paired runs: ' +
                    ('every pair identical.' if not len(bad) else
                     f'{len(bad)} PAIRS DIVERGED. The identity has been broken '
                     f'somewhere; treat every threshold statement here as void '
                     f'until that is found.')]
        if f.extent_pctl.nunique() > 1:
            sweep = []
            for p in sorted(f.extent_pctl.unique()):
                d = f[(f.extent_pctl == p) & (f.d_aic == 0)]
                n = d.form.value_counts()
                sweep.append(f'  EXTENT_PCTL {p:>3}: ' +
                             (', '.join(f'{k} {v}' for k, v in n.items())
                              or 'no fit'))
            cav += ['', 'Winning form per setting in THIS run:'] + sweep
        out.append(S('THE THRESHOLD CAVEAT — settled', '\n'.join(cav)))

        # --- what bounds the distribution ---------------------------------
        out.append(S('TRUNCATION — read before the fits', '\n'.join([
            'Rules 8 and 9 bound field size by construction: floor at '
            'Harland\'s smallest measured field (0.023 m^2 in 18.6 m^2), '
            'ceiling at 20% of arena area. Every fit is to a doubly-truncated '
            'sample.',
            f'  fields resting on the Rule 8 floor    median '
            f'{100*base.frac_at_floor.median():.1f}%',
            f'  fields resting on the Rule 9 ceiling  median '
            f'{100*base.frac_at_ceiling.median():.1f}%',
            '',
            'Mass piling against a bound is the rule speaking, not the model. '
            'A log-normal or exponential tail cut by Rule 9 will fit worse '
            'than it should, and the fine end is floored at exactly the value '
            'Harland measured — so agreement at the fine end is partly '
            'assumed rather than found.'])))

        # --- the library is a tiling at every scale -----------------------
        bd = getattr(self, 'bands', None)
        if bd is not None and len(bd):
            # Column names deliberately not `cov`/`n`: those collide with
            # pandas Series methods and attribute access silently returns the
            # method instead of the value.
            g = bd.groupby('scale_band').agg(
                n_f=('n_fields', 'sum'), share_lib=('share_of_library', 'median'),
                cov_med=('coverage_median', 'median'),
                tile_mult=('tiling_multiple', 'median'),
                iou_med=('split_half_iou_median', 'median'))
            L = ['A field library is a TILING AT EVERY SCALE, not a sample of '
                 'cells. A tiling at scale s needs about arena/s tiles, so the '
                 'finest band necessarily holds most of the library and '
                 'necessarily sets any pooled median, mean or CV. Read the '
                 'bands, not the pool.', '',
                 f'  {"band":>4s} {"fields":>7s} {"share":>7s} '
                 f'{"median cov":>11s} {"tiling":>7s} {"IoU":>5s}']
            for b_, r in g.iterrows():
                L.append(f'  {int(b_):>4d} {int(r.n_f):>7d} {100*r.share_lib:>6.1f}% '
                         f'{100*r.cov_med:>10.2f}% {r.tile_mult:>7.2f} {r.iou_med:>5.2f}')
            hi = g[(g.cov_med >= HARLAND_COVERAGE[0]) &
                   (g.cov_med <= HARLAND_COVERAGE[1])]
            near = g[(g.cov_med >= 0.5 * HARLAND_COVERAGE[0]) &
                     (g.cov_med <= 2.0 * HARLAND_COVERAGE[1])]
            L += ['',
                  f'"tiling" is the floor a band lays down as a multiple of '
                  f'the arena; ~1 means the band covers it once. IoU is '
                  f'split-half reliability, which rises with band.']
            if len(near):
                L += ['',
                      f'THE MODEL DOES REACH HARLAND\'S SCALE. Band(s) '
                      f'{", ".join(str(int(i)) for i in near.index)} sit at '
                      + ', '.join(f'{100*v:.1f}%' for v in near.cov_med) +
                      f' of the arena, against their 9-13% per cell'
                      + ('' if len(hi) else ' (bracketing it rather than '
                         'landing inside)') + '. The pooled median is far '
                      f'below that only because band '
                      f'{int(g.share_lib.idxmax())} is '
                      f'{100*g.share_lib.max():.0f}% of the library.']
            else:
                L += ['',
                      'No band reaches Harland\'s 9-13% per cell. This is a '
                      'divergence in the coarse tail, not an artifact of '
                      'pooling, and should be reported as such.']
            L += ['',
                  'Rule 2 (split-half reliability) is the principled way to '
                  'thin the fine end if you want to: reliability rises '
                  'monotonically with band, so a threshold removes fine '
                  'fields for being unreproducible rather than for being '
                  'small, and approximates an experimenter\'s detection '
                  'criterion. --split-half-iou-min 0.5 keeps roughly 57%. '
                  'Raising the Rule 8 floor instead would not work: the '
                  'median lands at about 2x the floor wherever the floor is '
                  'put, so choosing it chooses the answer.']
            out.append(S('PER SCALE BAND — read this before any pooled number',
                         '\n'.join(L)))

        # --- what changes as scale changes -------------------------------
        areas = sorted(base.env_area_m2.unique())
        tr = getattr(self, 'trends', None)
        if area_varies(base) and tr is not None and len(tr):
            lo, hi = min(areas), max(areas)
            L = [f'{len(areas)} arenas, {lo:.1f} to {hi:.0f} m^2 ({hi/lo:.1f}x). '
                 f'Harland span {HARLAND_AREA_RATIO}x, so the two are directly '
                 f'comparable and their per-quantity changes are numbers to '
                 f'hit rather than directions.', '']
            for _, r in tr.iterrows():
                arrow = {1: 'RISES', -1: 'FALLS', 0: 'flat'}[r.direction]
                if r.quantity == 'coverage_median':
                    L.append(f'  {r.label:26s} {100*r.value_small:6.2f}% -> '
                             f'{100*r.value_mega:6.2f}%   '
                             f'{100*r.delta_mega_small:+.2f} pp   {arrow}')
                else:
                    L.append(f'  {r.label:26s} {r.value_small:8.3g} -> '
                             f'{r.value_mega:8.3g}   '
                             f'x{r.ratio_mega_small:.2f}   {arrow}')
                L.append(f'      {r.source}: {r.source_says}')
                if r.agrees is None or pd.isna(r.agrees):
                    L.append('      -> no directional claim to match; compare '
                             'the value above')
                else:
                    L.append(f'      -> {"matches" if r.agrees else "DIVERGES"}'
                             f'  (pooled Spearman rho {r.spearman_rho:+.2f}, '
                             f'p {r.spearman_p:.3g})')
            scored = tr[tr.agrees.notna()]
            n_ok = int(scored.agrees.sum()) if len(scored) else 0
            by_src = ', '.join(
                f'{src} {int(g.agrees.sum())}/{len(g)}'
                for src, g in scored.groupby('source'))
            L += ['',
                  f'{n_ok} of {len(scored)} quantities with a directional claim '
                  f'move the predicted way ({by_src}). Read the direction first '
                  f'and the absolute values second: a different agent in a '
                  f'different arena is not expected to land on 70/85/101.',
                  '',
                  'Three arenas is few. The per-channel mega/small ratio is the '
                  'effect size and the pooled Spearman is the only test with '
                  'enough points to run, so neither is strong on its own; a '
                  'quantity is called flat when it moves under 10% across a '
                  f'{hi/lo:.0f}x area span, whatever the sign or the p-value.']
            # The size trend is only the model's if it survives division by
            # the Rule 8 floor, which grows with the arena on its own.
            def _row(q):
                r = tr[tr.quantity == q]
                return r.iloc[0] if len(r) else None
            med, ratio = _row('area_median_m2'), _row('area_median_over_floor')
            if med is not None and ratio is not None:
                if med.direction == 1 and ratio.direction == 0:
                    L += ['',
                          '!! THE SIZE TREND IS THE RULE, NOT THE MODEL. '
                          f'Median field area rises x{med.ratio_mega_small:.2f} '
                          f'across the span, but measured against the Rule 8 '
                          f'floor it is flat '
                          f'(x{ratio.ratio_mega_small:.2f}, '
                          f'{ratio.value_small:.2f} -> {ratio.value_mega:.2f} '
                          f'floors). The floor is a fixed fraction of arena '
                          f'area, so it grows with the arena by construction '
                          f'and the fields are riding it. Do not report field '
                          f'size as growing with the space, and do not score '
                          f'that against Eliav.']
                elif med is not None and ratio is not None and ratio.direction == 1:
                    L += ['',
                          f'The size trend survives the floor: median/floor '
                          f'itself rises x{ratio.ratio_mega_small:.2f}, so the '
                          f'model is responding to scale rather than tracking '
                          f'the rule.']
            base_med = base.area_median_over_floor.median()
            base_w2 = base.frac_within_2x_floor.median()
            base_short = base.coverage_shortfall_x.median()
            if base_med < 4.0 or base_short > 5.0:
                L += ['',
                      f'!! FIELDS SIT AT THE BOTTOM OF THE ADMISSIBLE WINDOW. '
                      f'The median field is {base_med:.1f}x the Rule 8 floor'
                      + ('' if pd.isna(base_w2) else
                         f' and {100*base_w2:.0f}% of fields are within 2x of it')
                      + f', while the smallest admitted field IS the floor. A '
                      f'median field covers '
                      f'{100*base.coverage_median.median():.2f}% of the arena '
                      f'against Harland\'s '
                      f'{100*HARLAND_COVERAGE[0]:g}-{100*HARLAND_COVERAGE[1]:g}%, '
                      f'a shortfall of about {base_short:.0f}x.',
                      '',
                      'Rule 8 is Harland\'s SMALLEST measured field, roughly '
                      '0.12% of their arena, while their TYPICAL field is '
                      '9-13% -- so the admissible window is about two orders '
                      'of magnitude wide and our population occupies its very '
                      'bottom. The model can reach the right scale (the '
                      'largest fields here approach the Rule 9 ceiling); it '
                      'simply produces far more small fields than large ones.',
                      '',
                      'Worth considering before reading any of the above as a '
                      'result: a field library enumerates every admissible '
                      'node of a hierarchy, and a Ward tree has far more fine '
                      'nodes than coarse ones. Harland and Eliav report fields '
                      'from a RECORDED SAMPLE of neurons, which is not an '
                      'exhaustive enumeration of a hierarchy. Comparing the '
                      'two populations assumes a selection step this model '
                      'does not have.']
            out.append(S('WHAT CHANGES WITH SCALE', '\n'.join(L)))

            out.append(S('TWO OF HARLAND\'S FOUR SCALE MEASURES ARE OUT OF REACH',
                         '\n'.join([
                'Harland give four quantities against area. Two are WITHIN-CELL '
                'and this model cannot produce them: subfields per cell (linear '
                'in area, R^2 = 0.9776) and summed subfield area per cell '
                '(exponential, r = 0.996). A single-centroid cluster owns '
                'exactly one field, so it has no subfield count and no sum over '
                'subfields. Those wait on multi-field place cells.', '',
                'The two reported above are the two that transfer. Even there, '
                'coverage is not quite like for like: Harland measure it per '
                'CELL, summed over that cell\'s subfields, and we measure it '
                'per FIELD. The two coincide only for single-field cells. '
                'Comparing a population spread against a within-cell one would '
                'overstate the match, and the same caution applies to the '
                'max/min ratio, where Eliav\'s 4.4 -> 1.6 is within-neuron and '
                'ours is across the population.'])))
        else:
            out.append(S('SCALE IS NOT VARIED IN THIS RUN', '\n'.join([
                f'Every dataset here is within a factor '
                f'{max(areas)/min(areas):.2f} of {areas[0]:.1f} m^2, so Harland '
                'Fig 6E (CV against enclosure area) and the 3F/3G contrast (a '
                'negative exponential in the megaspace against a Gaussian in '
                'the small environments) CANNOT be read at all. Both are claims '
                'about scale, and scale is held constant.', '',
                'What this run does establish is the shape at one scale, and '
                'whether cue density or arena shape move it — a control, and a '
                'prerequisite for reading the area sweep, but not a test of '
                'either published claim.', '',
                'Run over AREA_ENVS (circ_lm8_rad2p0, circ_lm8_r0, '
                'circ_lm8_rad6p0 — 12.6 to 113.1 m^2, 9.0x against Harland\'s '
                '8.8x) for the comparison this experiment is named after.'])))

        base = base.sort_values(['env_area_m2', 'channel'])
        cols = ['env', 'env_area_m2', 'channel', 'n_fields', 'cv_area_pct',
                'area_min_m2', 'area_median_m2', 'area_max_m2',
                'area_max_min_ratio', 'coverage_median', 'frac_under_1m2',
                'n_bands_occupied', 'frac_at_floor', 'frac_at_ceiling']
        out.append(S(f'Per environment and channel, in scale order '
                     f'(EXTENT_PCTL {DEFAULT_PCTL})',
                     self.table(base[[c for c in cols if c in base.columns]])))
        return '\n'.join(out)


# ----------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--envs', default=','.join(AREA_ENVS),
                   help='default is the area sweep (AREA_ENVS, 4.91-314 m^2), '
                        'the axis Harland vary and the only one on which Figs '
                        '3F-G and 6E can be read. Pass the six same-area '
                        'datasets (ENVS) for the cue-density/shape control.')
    p.add_argument('--channels', default=','.join(CHANNELS))
    p.add_argument('--settings',
                   default=','.join(f'{p}:{t:g}' for p, t in SETTINGS),
                   help='EXTENT_PCTL:ACT_THRESH pairs, comma separated')
    p.add_argument('--lam', type=float, default=0.0,
                   help='LAMBDA. 0 = feature only, as everywhere else.')
    p.add_argument('--split-half-iou-min', type=float, default=None,
                   metavar='IOU',
                   help='Rule 2: reject a field whose split-half IoU is below '
                        'this. Off by default, as in every other experiment. '
                        'Reliability rises monotonically with scale band '
                        '(median IoU 0.45 at band 0 to 0.69 at band 5), so a '
                        'threshold removes fine fields for being unreliable '
                        'rather than for being small, and approximates an '
                        'experimenter\'s detection criterion. 0.5 keeps ~57%.')
    p.add_argument('--subsample', type=int, default=0)
    p.add_argument('--n-boot', type=int, default=N_BOOT)
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
    settings = []
    for tok in args.settings.split(','):
        if not tok.strip():
            continue
        p_s, _, t_s = tok.partition(':')
        settings.append((int(p_s), float(t_s or DEFAULT_T)))
    rng = np.random.default_rng(args.seed)

    out_dir = f'{REPO}/data_cache/scale_distribution'
    fig_dir = f'{HERE}/figures/scale_distribution'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    base_C = R.resolve_cfg(dict(LAMBDA=args.lam, RANDOM_SEED=args.seed,
                                USE_GPU=not args.no_gpu,
                                SPLIT_HALF_IOU_MIN=args.split_half_iou_min))
    device = R.pick_device(use_gpu=not args.no_gpu)

    print('=' * 72)
    print('Scale distribution | what shape is our field-size distribution')
    print(f'  envs     : {envs}')
    print(f'  channels : {chans}')
    print(f'  settings : {[(p, t) for p, t in settings]}  (EXTENT_PCTL, ACT_THRESH)')
    print(f'  LAMBDA   : {base_C["LAMBDA"]}')
    print(f'  Rule 2   : ' + ('off (measured, not enforced)'
                              if base_C['SPLIT_HALF_IOU_MIN'] is None else
                              f'split-half IoU >= {base_C["SPLIT_HALF_IOU_MIN"]}'))
    print(f'  areas    : {"varies — Fig 6E readable" if len(envs) > 1 else "one"}'
          '  (a single area cannot speak to Harland 3F-G or 6E)')
    print('  note     : EXTENT_PCTL saturates at 65 and the sweep is settled;')
    print('             pass --settings to re-open it. ACT_THRESH cancels')
    print('             under SIGMA_MODE=quantile and is not a knob.')
    print('=' * 72, flush=True)

    banks_all, sum_rows, fit_rows, inv_rows = {}, [], [], []
    band_rows, env_geom = [], {}
    missing = []
    for e in envs:
        data_path = f'{REPO}/data/vpce/collect_data/{e}.h5'
        xml_path = f'{REPO}/simulation/worlds/environments/vpce/{e}.xml'
        if not os.path.exists(data_path):
            print(f'\n[{e}] no dataset at {data_path} -- skipping', flush=True)
            missing.append(e)
            continue
        print(f'\n===== {e} =====', flush=True)
        blocks, xy = ch.load_channel_blocks(data_path)
        if args.subsample and args.subsample < len(xy):
            sel = np.sort(rng.choice(len(xy), args.subsample, replace=False))
            blocks, xy = {k: v[sel] for k, v in blocks.items()}, xy[sel]
            print(f'  subsampled to {len(xy)} -- results not comparable to a '
                  f'full run; field size scales with sample count')
        root = ET.parse(xml_path).getroot()
        env = R.build_env(xy, root)
        # Area is held constant across the six; shape and landmark count are
        # what vary, so both are carried into the summary for S2.
        env['n_landmarks'] = len(root.findall('landmark'))
        env['aspect'] = (1.0 if env.get('is_circular') else
                         (env['x_max'] - env['x_min']) /
                         (env['y_max'] - env['y_min']))
        env_geom[e] = dict(env, landmarks=[
            (float(l.get('x')), float(l.get('y')))
            for l in root.findall('landmark')])
        print(f'  arena area {env["env_area"]:.1f} m^2, aspect '
              f'{env["aspect"]:.2f}, {env["n_landmarks"]} landmarks, '
              f'{len(xy)} locations', flush=True)

        for c in chans:
            banks = build_banks(e, c, blocks, xy, env, settings, base_C,
                                device, out_dir, args.use_cache)
            for row in threshold_invariance(banks):
                inv_rows.append(dict(row, env=e, channel=c))
            for (p, t), bank in banks.items():
                banks_all[(e, c, p, t)] = bank
                if not len(bank):
                    continue
                tag = dict(env=e, channel=c, extent_pctl=p,
                           act_thresh=t,
                           env_area_m2=float(env['env_area']))
                d = describe(bank, env, base_C)
                sum_rows.append({**tag, **d})
                band_rows.extend(band_table(bank, env, base_C, tag))
                # Area is Harland's unit; equivalent diameter is the closest
                # thing we have to Eliav's 1D field width, so both are fitted.
                for var, x in (('area', bank.area_env_m2),
                               ('diameter', 2.0 * bank.radius_env_m)):
                    for d in fit_forms(x, n_boot=args.n_boot, seed=args.seed):
                        d['params'] = json.dumps(d['params'])
                        fit_rows.append(dict(tag, variable=var, **d))
        del blocks

    if not sum_rows:
        print('\nNo results. Datasets missing: ' + (', '.join(missing) or 'none'))
        return 1

    summary = pd.DataFrame(sum_rows)
    fits = pd.DataFrame(fit_rows)
    inv = pd.DataFrame(inv_rows) if inv_rows else None
    summary.to_csv(f'{out_dir}/summary.csv', index=False)
    bands = pd.DataFrame(band_rows)
    if len(bands):
        bands = bands.sort_values(['env_area_m2', 'channel', 'scale_band'])
        bands.to_csv(f'{out_dir}/band_summary.csv', index=False)
    fits.to_csv(f'{out_dir}/fits.csv', index=False)
    if inv is not None:
        inv.to_csv(f'{out_dir}/threshold_invariance.csv', index=False)

    trends = scale_trends(summary[(summary.extent_pctl == DEFAULT_PCTL) &
                                  (summary.act_thresh == DEFAULT_T)])
    if len(trends):
        trends.to_csv(f'{out_dir}/scale_trends.csv', index=False)

    winners = (fits[(fits.variable == 'area') & (fits.d_aic == 0) &
                    (fits.extent_pctl == DEFAULT_PCTL) &
                    (fits.act_thresh == DEFAULT_T)].form.value_counts())

    print('\nfigures:', flush=True)
    # Scale order, so S1 reads small -> mega down the page.
    envs_by_area = list(summary.sort_values('env_area_m2')
                        .drop_duplicates('env').env)
    fig_distributions(banks_all, fits, envs_by_area, chans, fig_dir)
    fig_cv(summary, fig_dir)
    fig_field_maps(banks_all, envs_by_area, chans, env_geom, fig_dir)
    fig_form_vs_scale(fits, fig_dir)
    fig_size_vs_scale(summary, fig_dir)
    prune_orphan_figures(fig_dir)

    rep = ScaleDistributionReport(env_name=','.join(envs), out_dir=out_dir,
                                  fig_dir=fig_dir, results=summary,
                                  log_path=os.environ.get('REALM_LOG_PATH'))
    rep.fits, rep.invariance, rep.winners, rep.trends = fits, inv, winners, trends
    rep.bands = bands
    if missing:
        print(f'\n!! datasets not found, excluded: {", ".join(missing)}')
    print('\n' + rep.compose(), flush=True)
    if not args.no_email:
        rep.send()
    print(f'\nsummary -> {out_dir}/summary.csv'
          f'\nfits    -> {out_dir}/fits.csv'
          f'\nfigures -> {fig_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
