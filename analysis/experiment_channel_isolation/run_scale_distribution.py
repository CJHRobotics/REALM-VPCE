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
import matplotlib.colors as mcolors
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
# Varying area at fixed shape and landmark count: small, medium, mega.
#
#   circ_lm8_r3        r = 3     28.27 m^2   small     wall cover 32%
#   circ_lm8_r6    r = 6    113.10 m^2   medium    wall cover 16%
#   circ_lm8_r10   r = 10   314.16 m^2   mega      wall cover 10%
#
# An 11.1x span against Harland's 8.8x. Each arena is sampled at ~N_TARGET
# positions, so sample count is not a covariate.
#
# Note the cue-salience confound grows across this sweep and is worst at the
# top: a fixed 0.75 m panel spans roughly 11 px of a 224 px image from across
# the r = 10 disc, and the colour channel has previously collapsed to
# single-digit field counts there. It did not at r = 6 (511 fields), so the
# collapse may have been an artifact of the older configuration -- but r = 10
# is 2.8x that area again, and colour is the channel to check first.
AREA_ENVS = ['circ_lm8_r3', 'circ_lm8_r6', 'circ_lm8_r10']

# The Eliav comparison: a 10 x 2 m corridor, 20 m^2, aspect 5:1. Long enough
# to read as one-dimensional, short enough to stand for the 6 m tunnel segment
# in which Eliav found mean field size fall from 5.9 m to 1.5 m and the
# within-neuron size ratio from 4.4 to 1.6 -- their evidence that multiscale
# coding is a property of a large space rather than of the hippocampus.
#
# Deliberately not area-matched to any disc, and excluded from the area trend
# for the same reason the rectangle is: its aspect is not 1.
ELIAV_ENVS = ['corr_lm8_l10w2']

# Eliav's numbers, as field LENGTH along the tunnel -- a one-dimensional
# width, so ours has to be measured the same way (the field's extent along the
# long axis) rather than as an area.
ELIAV_MEAN_LEN_M = {'200 m tunnel': 5.9, '6 m segment': 1.5}
ELIAV_SIZE_RATIO = {'200 m tunnel': 4.4, '6 m segment': 1.6}

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

# The Rule 2 setting the figures and the report are drawn at. Set from the
# first --split-half-iou-min entry, so a sweep still has one primary view and
# the rest lands in the CSVs. None = Rule 2 off, the series default.
PRIMARY_IOU = None


def at_operating_point(df):
    """Rows at the primary EXTENT_PCTL, ACT_THRESH and Rule 2 setting."""
    if df is None or not len(df):
        return df
    m = (df.extent_pctl == DEFAULT_PCTL) & (df.act_thresh == DEFAULT_T)
    if 'split_half_iou_min' in df.columns:
        col = df.split_half_iou_min
        m &= col.isna() if PRIMARY_IOU is None else (col == PRIMARY_IOU)
    return df[m]
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


def eliav_lengths(bank, env, tag):
    """Field LENGTH along the long axis, for comparison with Eliav.

    Eliav report field size as a one-dimensional width in metres along a
    tunnel, not as an area, so an area cannot be compared to it. In an
    elongated arena the matching quantity is how far a field extends along the
    long axis: the projection of its Rule 7 ellipse onto that axis,

        length = 2 * sqrt( (a cos t)^2 + (b sin t)^2 )

    for semi-axes a, b at orientation t. In a disc there is no long axis and
    the quantity is meaningless, so this returns nothing.
    """
    if not len(bank) or env.get('is_circular'):
        return []
    long_is_x = (env['x_max'] - env['x_min']) >= (env['y_max'] - env['y_min'])
    a = bank.semi_major_m.to_numpy(dtype=float)
    b = bank.semi_minor_m.to_numpy(dtype=float)
    t = bank.orientation_rad.to_numpy(dtype=float)
    if not long_is_x:
        t = t + np.pi / 2.0
    length = 2.0 * np.sqrt((a * np.cos(t)) ** 2 + (b * np.sin(t)) ** 2)
    span = float(max(env['x_max'] - env['x_min'], env['y_max'] - env['y_min']))
    return [dict(tag, arena_length_m=span, n_fields=len(length),
                 len_mean_m=float(length.mean()),
                 len_median_m=float(np.median(length)),
                 len_min_m=float(length.min()), len_max_m=float(length.max()),
                 len_max_min_ratio=float(length.max() / length.min()),
                 # Eliav's within-neuron ratio has no equivalent while a
                 # cluster owns one field; this is the population spread, and
                 # the two are not the same quantity.
                 len_p90_p10_ratio=float(np.percentile(length, 90) /
                                         np.percentile(length, 10)),
                 frac_of_arena_length=float(np.median(length) / span))]


def scale_trends(summary):
    """How each tracked quantity moves across arena area.

    Three arenas is too few for a meaningful per-channel correlation -- a
    Spearman rho over three points takes one of four values -- so the trend is
    read two ways: the mega/small ratio per channel, which is the effect size,
    and a Spearman pooled over every (channel, arena) point, which is the only
    place there are enough points to test a direction.
    """
    rows = []
    # Circles only. The corridor's aspect is 5, and letting an elongated
    # arena into a slope about area would confound shape with scale.
    if 'aspect' in summary.columns:
        summary = summary[summary.aspect == 1.0]
    if not len(summary) or summary.env_area_m2.nunique() < 2:
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

def build_banks(env_name, cname, blocks, xy, env, settings, ious, base_C,
                device, out_dir, use_cache, verbose=True):
    """Field libraries for one channel at every (setting, Rule 2) combination.

    Three stages, split by what each depends on, so nothing is recomputed
    that did not change:

      Gram + Ward tree   depend on the channel and LAMBDA only -- once.
      prepare_candidates depends on EXTENT_PCTL -- once per setting.
      admit_fields       depends on ACT_THRESH and Rule 2 -- once per
                         (setting, IoU) pair, and it is the cheap stage.

    So a Rule 2 sweep costs one extra admit_fields per threshold, not a
    rebuild. That matters because Rule 2 CANNOT be applied by filtering a
    finished bank: it sits upstream of Rule 11, whose competition ordering is
    already tie-broken on reliability, and upstream of Rule 12, which decides
    which bands survive on the coverage the survivors reach. Remove a field
    before competition and a different one claims that territory; remove
    enough and a whole band stops tiling. Post-hoc filtering answers a
    different question.
    """
    def _key(p, t, iou):
        r = 'off' if iou is None else f'{iou:g}'
        return f'{out_dir}/{env_name}/{cname}_p{p}_t{t:g}_r{r}_bank.csv'

    want = {(p, t, iou): _key(p, t, iou)
            for p, t in settings for iou in ious}
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
        for iou in ious:
            Ci = R.resolve_cfg(dict(C, SPLIT_HALF_IOU_MIN=iou))
            bank, _, rep = R.admit_fields(ctx, cfg=Ci, verbose=verbose)
            bank.to_csv(want[(p, t, iou)], index=False)
            with open(want[(p, t, iou)].replace('_bank.csv', '_report.json'),
                      'w') as f:
                json.dump({k: v for k, v in rep.items()
                           if not isinstance(v, np.ndarray)}, f, indent=2,
                          default=float)
            banks[(p, t, iou)] = bank
            lbl = 'off' if iou is None else f'>= {iou:g}'
            print(f'  [{cname}] pctl {p} T {t:g} Rule2 {lbl}: '
                  f'{len(bank)} fields', flush=True)
    print(f'  [{cname}] {len(want)} bank(s) in {time.time()-t0:.0f}s',
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


def fig_distributions(banks_all, fits, envs, chans, env_geom, fig_dir):
    """S1: the size histogram per arena x channel, with all three fits drawn.

    Arena on the ROW and channel on the COLUMN, each named once. The previous
    layout put the arena in the title of the top row only and repeated the
    channel on every row, so rows 2 and below carried no arena label at all
    and every panel appeared to come from the first environment.

    A channel that collapsed still gets a panel, labelled with its field
    count. Leaving it blank makes a real failure look like a plotting gap --
    at r = 10 the colour channel returns single-digit field counts, which is
    a result worth seeing rather than an absence.
    """
    envs = [e for e in envs
            if any((e, c, DEFAULT_PCTL, DEFAULT_T, PRIMARY_IOU) in banks_all
                   for c in chans)]
    if not envs:
        return
    fig, axes = plt.subplots(len(envs), len(chans), squeeze=False,
                             figsize=(3.0 * len(chans), 2.5 * len(envs)))
    fo = at_operating_point(fits)
    for i, e in enumerate(envs):
        area = env_geom.get(e, {}).get('env_area')
        for j, c in enumerate(chans):
            ax = axes[i][j]
            if i == 0:
                ax.set_title(c, fontsize=10)
            if j == 0:
                label = e if area is None else f'{e}\n{area:.0f} m$^2$'
                ax.set_ylabel(f'{label}\n\n% of fields', fontsize=8)
            else:
                ax.set_ylabel('% of fields', fontsize=7)
            ax.tick_params(labelsize=6)
            if i == len(envs) - 1:
                ax.set_xlabel('field area (m$^2$)', fontsize=8)

            b = banks_all.get((e, c, DEFAULT_PCTL, DEFAULT_T, PRIMARY_IOU))
            n = 0 if b is None else len(b)
            if n < MIN_FIELDS:
                # Keep the frame and say why it is empty.
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, f'{n} field' + ('' if n == 1 else 's') +
                        '\ntoo few to fit', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='#b03030')
                continue
            x = b.area_env_m2.to_numpy(dtype=float)
            dens, centres, _ = _hist(x)
            # Plot the percentage of fields in each bin, not a probability
            # density. np.histogram(density=True) returns counts / (N * width),
            # which carries units of 1/m^2 and reads as an arbitrary number;
            # multiplying by the bin width recovers the fraction of fields, so
            # the bars sum to 100 and each one is directly readable. The
            # fitted curves are converted the same way, pdf * width * 100,
            # which is the percentage the form predicts for that bin. r_hist
            # is unaffected: it is a correlation, and both series are scaled
            # by the same constant.
            bw = float(centres[1] - centres[0])
            ax.bar(centres, 100.0 * dens * bw, width=bw * 0.9,
                   color='0.82', edgecolor='none')
            gx = np.linspace(x.min(), x.max(), 300)
            sub = fo[(fo.env == e) & (fo.channel == c) & (fo.variable == 'area')]
            for _, row in sub.iterrows():
                pdf = FORMS[row.form]['dist'].pdf(gx, *json.loads(row.params))
                ax.plot(gx, 100.0 * bw * pdf, lw=1.4,
                        label=f"{row.form[:4]} r={row.r_hist:.3f}")
            ax.legend(fontsize=5, frameon=False)
            ax.text(0.97, 0.55, f'n={n}', transform=ax.transAxes, ha='right',
                    fontsize=6, color='0.35')
    fig.suptitle('S1  field-size distribution per arena and channel, with all '
                 f'three published forms\narenas in scale order '
                 f'(EXTENT_PCTL {DEFAULT_PCTL})', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, fig_dir, 'S1_size_distributions.png')


def fig_field_maps(banks_all, envs_by_area, chans, env_geom, fig_dir):
    """S2: the admitted fields drawn on the arena, coloured by field area.

    Two things are held fixed so the panels can be read against each other.

    **One physical scale for every panel.** All axes share the extent of the
    largest arena, so a 28 m^2 disc is drawn a third the width of a 314 m^2
    one instead of being blown up to fill its own panel. Drawing each arena to
    its own frame hides the very thing this experiment measures.

    **One colour scale for every panel.** Colour is field area in m^2 under a
    single normalisation pooled over every arena and channel, so a given
    colour means the same size everywhere and the growth of fields with the
    enclosure is visible as the panels warming from top to bottom.

    The normalisation is square-root, not linear. Field areas span about
    2400-fold, so a linear ramp puts the median field at under 1% of the range
    and every panel comes out one flat colour. Under a square root the colour
    tracks field WIDTH rather than area, which spreads the bulk of the
    population across the ramp while the ticks stay labelled in m^2. The top
    of the range is the 99th percentile rather than the maximum, so that a
    handful of ceiling-sized fields do not consume the whole scale; anything
    above it takes the end colour and the bar says so.
    """
    if not banks_all:
        return
    key = lambda e, c: (e, c, DEFAULT_PCTL, DEFAULT_T, PRIMARY_IOU)
    pooled = np.concatenate([
        banks_all[key(e, c)].area_env_m2.to_numpy(dtype=float)
        for e in envs_by_area for c in chans
        if key(e, c) in banks_all and len(banks_all[key(e, c)])] or [np.array([1.0])])
    vmax = float(np.percentile(pooled, 99))
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax, clip=True)
    cmap = plt.cm.inferno

    # One frame for everything: the largest half-extent any arena reaches.
    half = 0.0
    for e in envs_by_area:
        g = env_geom.get(e, {})
        half = max(half, g['env_R'] if g.get('is_circular') else
                   max(abs(g.get('x_min', 0)), abs(g.get('x_max', 0)),
                       abs(g.get('y_min', 0)), abs(g.get('y_max', 0))))
    lim = half * 1.04

    n_r, n_c = len(envs_by_area), len(chans)
    fig, axes = plt.subplots(n_r, n_c, squeeze=False,
                             figsize=(2.6 * n_c, 2.7 * n_r))
    for i, e in enumerate(envs_by_area):
        geom = env_geom.get(e, {})
        for j, c in enumerate(chans):
            ax = axes[i][j]
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            for sp in ax.spines.values():
                sp.set_color('0.85')
            if geom.get('is_circular'):
                ax.add_patch(plt.Circle((geom.get('env_cx', 0.0),
                                         geom.get('env_cy', 0.0)),
                                        geom['env_R'], fill=False,
                                        color='0.25', lw=1.1))
            elif 'x_min' in geom:
                ax.add_patch(plt.Rectangle(
                    (geom['x_min'], geom['y_min']),
                    geom['x_max'] - geom['x_min'],
                    geom['y_max'] - geom['y_min'],
                    fill=False, color='0.25', lw=1.1))
            for lm in geom.get('landmarks', []):
                ax.plot(lm[0], lm[1], 's', ms=2, color='#2b6cb0', zorder=5)

            b = banks_all.get(key(e, c))
            if b is not None and len(b):
                # Largest first, so the fine fields stay visible on top.
                for _, r in b.sort_values('area_env_m2', ascending=False).iterrows():
                    ax.add_patch(Ellipse(
                        (r.centroid_x, r.centroid_y),
                        2 * r.semi_major_m, 2 * r.semi_minor_m,
                        angle=np.degrees(r.orientation_rad),
                        facecolor=cmap(norm(r.area_env_m2)),
                        edgecolor='none', alpha=0.55, lw=0))
                ax.text(0.03, 0.03, f'n={len(b)}', transform=ax.transAxes,
                        fontsize=6, color='0.4')
            if i == 0:
                ax.set_title(c, fontsize=10)
            if j == 0:
                area = geom.get('env_area', float('nan'))
                ax.set_ylabel(f'{e}\n{area:.0f} m$^2$', fontsize=8)
    # A single 5 m bar, since every panel is at the same scale. Placed in axes
    # fractions above the field count rather than in data coordinates, so it
    # cannot land on top of the label whatever the arena's extent.
    ax0 = axes[0][0]
    x0 = -lim + 0.06 * 2 * lim
    y0 = -lim + 0.14 * 2 * lim
    ax0.plot([x0, x0 + 5.0], [y0, y0], '-', color='k', lw=2)
    ax0.text(x0, y0 + 0.03 * 2 * lim, '5 m', fontsize=6)

    fig.tight_layout(rect=(0, 0, 0.90, 0.94))
    # Added after tight_layout: a manually placed colour bar is not a
    # tight_layout-compatible axes and warns if it exists during the call.
    cax = fig.add_axes([0.92, 0.15, 0.014, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('field area (m$^2$)', fontsize=9)
    cb.ax.tick_params(labelsize=7)
    ticks = [t for t in (0.05, 0.25, 1, 2, 4, 8) if t <= vmax]
    cb.set_ticks(ticks + [vmax])
    cb.set_ticklabels([f'{t:g}' for t in ticks] + [f'$\\geq${vmax:.1f}'])
    fig.suptitle('S2  admitted fields drawn on the arena, coloured by field '
                 'area\nevery panel at the same physical scale and the same '
                 'colour scale', fontsize=11)
    _save(fig, fig_dir, 'S2_field_maps.png')


def fig_size_vs_scale(summary, fig_dir):
    """S5: the size ladder against area.

    Median field size, the max/min spread and the number of occupied scale
    bands. Together these say whether a larger space buys a wider range of
    scales or merely a uniformly coarser one -- which is the question
    Experiment 3 is built on, and the reason this experiment is its
    prerequisite.
    """
    s = at_operating_point(summary)
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
    fig.suptitle('S3  the size ladder against arena area — does a larger space '
                 'buy a WIDER range of scales, or a uniformly coarser one?\n'
                 'filled circles joined = the area sweep (shape fixed); open '
                 'squares = an elongated arena, not on the area curve',
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, fig_dir, 'S3_size_vs_scale.png')


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
        base = at_operating_point(s)
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
                            f'{self.out_dir}/eliav_lengths.csv',
                            f'{self.out_dir}/threshold_invariance.csv')
                if os.path.exists(p)]

    def body(self):
        s = self.results
        if s is None or not len(s):
            return 'No field libraries were produced.'
        S = self.section
        base = at_operating_point(s)
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
            'CV IS MEASURED BUT NOT PLOTTED, and not compared to Fig 6E. '
            'Pooled across bands it describes a six-band mixture spanning two '
            'orders of magnitude, and its trend across area tracks how many '
            'bands are occupied rather than any field size. Within a band it '
            'is fixed by the band definition: bands are geometric in radius '
            'at ratio 1.6, so areas span 2.56x and a uniform spread over that '
            'gives CV = 25%, which is what we measure (23-32%). Harland\'s '
            '70-101 sits between the two, and neither of ours is comparable '
            'until there is a model of how a recording samples cells from '
            'this library. The numbers are in band_summary.csv.', '',
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
                  'Neither available lever thins the fine end at present. '
                  'Raising the Rule 8 floor does not work: the median lands '
                  'at about 2x the floor wherever the floor is put, so '
                  'choosing it chooses the answer. Rule 2 (split-half '
                  'reliability) would be the principled route -- reliability '
                  'rose monotonically with band under the old 0.25 m binning, '
                  'so a threshold removes fine fields for being '
                  'unreproducible rather than for being small -- but at the '
                  'lattice bin the two half-maps land on disjoint bins and '
                  'every IoU is exactly 0. Scoring the halves on a coarser '
                  'grid than the one used for field extent would restore it.']
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
                'Run over AREA_ENVS (circ_lm8_r3, circ_lm8_r6, '
                'circ_lm8_r10 — 28.3 to 314.2 m^2, 11.1x against '
                'Harland\'s 8.8x) for the comparison this experiment is '
                'named after.'])))

        base = base.sort_values(['env_area_m2', 'channel'])
        # --- the Eliav comparison, in one dimension -----------------------
        el = getattr(self, 'eliav', None)
        if el is not None and len(el):
            el = at_operating_point(el)
        if el is not None and len(el):
            L = ['Eliav report field size as a LENGTH along a tunnel, not as '
                 'an area, so ours is measured the same way: how far each '
                 'field reaches along the arena\'s long axis. Only elongated '
                 'arenas appear here; a disc has no long axis.', '']
            for env_name, g in el.groupby('env'):
                span = g.arena_length_m.iloc[0]
                L.append(f'{env_name} — {span:.1f} m long, '
                         f'{int(g.n_fields.median())} fields per channel')
                L.append(f'  median field length   '
                         f'{g.len_median_m.median():.2f} m  '
                         f'({100*g.frac_of_arena_length.median():.1f}% of the '
                         f'arena\'s length)')
                L.append(f'  mean field length     '
                         f'{g.len_mean_m.median():.2f} m')
                L.append(f'  longest / shortest    '
                         f'{g.len_max_min_ratio.median():.1f}x   '
                         f'(90th/10th percentile {g.len_p90_p10_ratio.median():.1f}x)')
            L += ['',
                  'Eliav, for reference:']
            for k in ELIAV_MEAN_LEN_M:
                L.append(f'  {k:14s} mean field {ELIAV_MEAN_LEN_M[k]:.1f} m, '
                         f'within-neuron size ratio {ELIAV_SIZE_RATIO[k]:.1f}x')
            L += ['',
                  'Their 6 m segment is the comparison that matters: a short '
                  'stretch of the same tunnel, where mean field size fell from '
                  '5.9 m to 1.5 m and the size ratio from 4.4 to 1.6. That is '
                  'their evidence that a spread of scales is a property of a '
                  'LARGE space rather than of the hippocampus, and it is the '
                  'claim a 10 m corridor can speak to.', '',
                  'Two mismatches to carry into any comparison. Their ratio is '
                  'WITHIN a neuron and ours is across the population, because '
                  'a single-centroid cluster owns one field -- these are '
                  'different quantities and the population spread is the wider '
                  'of the two by construction. And their tunnel is a true '
                  'one-dimensional flight path, while ours is 2 m wide, so a '
                  'field here has a width the bat\'s did not.']
            out.append(S('THE ELIAV COMPARISON — field length along the arena',
                         '\n'.join(L)))

        # --- shape at matched area ----------------------------------------
        if 'aspect' in base.columns and (base.aspect != 1.0).any():
            L = []
            # Pair on area to ~0.1 m^2: the rectangle's walls are written to
            # 4 dp, so its area differs from the disc's pi*r^2 in the fifth
            # decimal and an exact groupby would never pair them.
            for area, g in base.groupby(base.env_area_m2.round(1)):
                circ = g[g.aspect == 1.0]
                oth = g[g.aspect != 1.0]
                if not len(circ) or not len(oth):
                    continue
                L.append(f'At {area:.1f} m^2, {circ.env.iloc[0]} against '
                         f'{", ".join(sorted(oth.env.unique()))}:')
                for col, lbl, pct in (
                        ('area_median_m2', 'median field area (m^2)', False),
                        ('coverage_median', 'coverage per field', True),
                        ('n_fields', 'fields admitted', False),
                        ('n_bands_occupied', 'bands occupied', False)):
                    cv_, ov = circ[col].median(), oth[col].median()
                    f = (lambda v: f'{100*v:.2f}%') if pct else (
                        lambda v: f'{v:.4g}')
                    L.append(f'  {lbl:26s} disc {f(cv_):>9s}   '
                             f'other {f(ov):>9s}   '
                             f'x{ov/cv_:.2f}' if cv_ else f'  {lbl}: n/a')
            if L:
                L += ['',
                      'Same area, same landmark count, different boundary. A '
                      'large difference here means the size distribution is '
                      'reading boundary geometry rather than enclosure scale, '
                      'and the area trend above should be read with that in '
                      'mind. This pair is deliberately excluded from the area '
                      'trend: two arenas at one area, one of them a different '
                      'shape, would let shape leak into a slope that is meant '
                      'to be about area alone.']
                out.append(S('SHAPE AT MATCHED AREA', '\n'.join(L)))

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
    p.add_argument('--envs',
                   default=','.join(AREA_ENVS + ELIAV_ENVS),
                   help='default is the area sweep (AREA_ENVS, 28.3-314 m^2, '
                        'the axis Harland vary) plus the corridor '
                        '(ELIAV_ENVS, 10 x 2 m). The corridor has aspect 5, '
                        'so it is excluded from the area trend by '
                        'construction.')
    p.add_argument('--channels', default=','.join(CHANNELS))
    p.add_argument('--settings',
                   default=','.join(f'{p}:{t:g}' for p, t in SETTINGS),
                   help='EXTENT_PCTL:ACT_THRESH pairs, comma separated')
    p.add_argument('--lam', type=float, default=0.0,
                   help='LAMBDA. 0 = feature only, as everywhere else.')
    p.add_argument('--split-half-iou-min', default='none', metavar='LIST',
                   help='Rule 2: reject a field whose split-half IoU is below '
                        'this. CURRENTLY UNUSABLE and the run will refuse it: '
                        'at the lattice bin the two half-maps occupy disjoint '
                        'bins, so every IoU is exactly 0 and any threshold '
                        'rejects the whole library. It was informative under '
                        'the old 0.25 m binning (0.45 at band 0 rising to 0.69 '
                        'at band 5) and needs the halves scored on a coarser '
                        'grid before it works again. Takes a comma list once '
                        'that is fixed, since Rule 2 only re-runs the cheap '
                        'admission stage.')
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
    ious = [None if tok.strip().lower() in ('none', 'off', '')
            else float(tok) for tok in args.split_half_iou_min.split(',')
            if tok.strip()] or [None]
    global PRIMARY_IOU
    PRIMARY_IOU = ious[0]
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
                                USE_GPU=not args.no_gpu))
    device = R.pick_device(use_gpu=not args.no_gpu)

    print('=' * 72)
    print('Scale distribution | what shape is our field-size distribution')
    print(f'  envs     : {envs}')
    print(f'  channels : {chans}')
    print(f'  settings : {[(p, t) for p, t in settings]}  (EXTENT_PCTL, ACT_THRESH)')
    print(f'  LAMBDA   : {base_C["LAMBDA"]}')
    print('  Rule 2   : ' + ', '.join(
        'off (measured, not enforced)' if i is None else f'IoU >= {i:g}'
        for i in ious))
    print(f'  areas    : {"varies — Fig 6E readable" if len(envs) > 1 else "one"}'
          '  (a single area cannot speak to Harland 3F-G or 6E)')
    print('  note     : EXTENT_PCTL saturates at 65 and the sweep is settled;')
    print('             pass --settings to re-open it. ACT_THRESH cancels')
    print('             under SIGMA_MODE=quantile and is not a knob.')
    print('=' * 72, flush=True)

    banks_all, sum_rows, fit_rows, inv_rows = {}, [], [], []
    band_rows, env_geom, eliav_rows = [], {}, []
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
            banks = build_banks(e, c, blocks, xy, env, settings, ious,
                                base_C, device, out_dir, args.use_cache)
            # The ACT_THRESH identity is about the mask, so check it at one
            # Rule 2 setting rather than once per threshold.
            for row in threshold_invariance(
                    {(p, t): b for (p, t, i), b in banks.items()
                     if i == ious[0]}):
                inv_rows.append(dict(row, env=e, channel=c))
            for (p, t, iou), bank in banks.items():
                banks_all[(e, c, p, t, iou)] = bank
                if not len(bank):
                    continue
                tag = dict(env=e, channel=c, extent_pctl=p, act_thresh=t,
                           split_half_iou_min=(np.nan if iou is None
                                               else float(iou)),
                           env_area_m2=float(env['env_area']))
                d = describe(bank, env, base_C)
                sum_rows.append({**tag, **d})
                band_rows.extend(band_table(bank, env, base_C, tag))
                eliav_rows.extend(eliav_lengths(bank, env, tag))
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
    eliav = pd.DataFrame(eliav_rows)
    if len(eliav):
        eliav.to_csv(f'{out_dir}/eliav_lengths.csv', index=False)
    bands = pd.DataFrame(band_rows)
    if len(bands):
        bands = bands.sort_values(['env_area_m2', 'channel', 'scale_band'])
        bands.to_csv(f'{out_dir}/band_summary.csv', index=False)
    fits.to_csv(f'{out_dir}/fits.csv', index=False)
    if inv is not None:
        inv.to_csv(f'{out_dir}/threshold_invariance.csv', index=False)

    trends = scale_trends(at_operating_point(summary))
    if len(trends):
        trends.to_csv(f'{out_dir}/scale_trends.csv', index=False)

    _f = at_operating_point(fits)
    winners = _f[(_f.variable == 'area') & (_f.d_aic == 0)].form.value_counts()

    print('\nfigures:', flush=True)
    # Scale order, so S1 reads small -> mega down the page.
    envs_by_area = list(summary.sort_values('env_area_m2')
                        .drop_duplicates('env').env)
    fig_distributions(banks_all, fits, envs_by_area, chans, env_geom, fig_dir)
    fig_field_maps(banks_all, envs_by_area, chans, env_geom, fig_dir)
    fig_size_vs_scale(summary, fig_dir)
    prune_orphan_figures(fig_dir)

    rep = ScaleDistributionReport(env_name=','.join(envs), out_dir=out_dir,
                                  fig_dir=fig_dir, results=summary,
                                  log_path=os.environ.get('REALM_LOG_PATH'))
    rep.fits, rep.invariance, rep.winners, rep.trends = fits, inv, winners, trends
    rep.bands, rep.eliav = bands, eliav
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
