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
every library, and the one that fits best is reported beside how well the
other two do. Picking a favourite in advance and reporting only its fit would
be the one result this experiment cannot support.

Every fit knows the size window. Rules 8 and 9 admit a field only between the
floor and the ceiling, so each form's density is renormalised to that window
before it is fitted -- a truncated fit. Without it the comparison is rigged:
an exponential fitted from zero puts its highest density exactly where the
floor has removed every field, is penalised for that gap, and log-normal wins
whatever the true shape. On synthetic exponential libraries cut at the r = 3
floor, the untruncated fit picked log-normal in 50 of 50.

Goodness of fit is reported three ways, because Harland's statistic and a
model-selection statistic answer different questions:

  r_hist   Pearson r between the binned density and the fitted pdf at the bin
           centres. This is *their* statistic, and the only one comparable to
           the published 0.995 and 0.985. It is also a weak discriminator: on
           a heavy-fine-end histogram all three forms clear r = 0.9, which is
           why it must not be read alone.
  ks       Kolmogorov-Smirnov distance against the truncated form, with a
           parametric-bootstrap p drawn from that same truncated form. The
           analytic p is anticonservative when the parameters were estimated
           from the same sample, so it is not used.
  aic      2k - 2 logL. This is the discriminator: `winner` is its argmin, and
           `aic_weight` (the Akaike weight, 0-1) says how decisively it wins.

The threshold caveat
--------------------
Harland show their exponential fit becomes quasi-linear at a lower
field-detection threshold, so distribution shape is not threshold-independent
and the comparison is meaningless without checking ours.

`EXTENT_PCTL` saturates at 65 (run_field_recovery, against ideal place cells
of known size), so the default is that single operating point. The first full
run of this experiment swept 50 / 65 / 80 and found log-normal winning at every
setting, but those fits ignored the size window and would have picked
log-normal whatever the shape, so that sweep says nothing about whether the
shape holds across settings. `--settings 50:0.5,65:0.5,80:0.5` re-opens it.

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

Arenas
------
Two of the three targets above are claims about ENVIRONMENT SCALE. Fig 3F-G is
a scale-dependent shape claim -- exponential in the megaspace, Gaussian in the
small environments -- and Fig 6E is CV against enclosure area. Neither can be
read from datasets that hold area constant.

The default is all nine arenas, each sampled at ~N_TARGET positions so sample
count is not a covariate. Each has one declared role, and the role alone
decides which comparisons it enters:

  area sweep      circ_lm8_r3, _r6, _r10   the only arenas on the area trend
  Eliav corridor  corr_lm8_l10w2           the only arena scored on length
  square          corr_lm8_l10w10          10 x 10 m, two panels on each wall
  no landmarks    circ_lm0_r3, circ_lm0_r6, corr_lm0_l10w10, corr_lm0_l10w2
                                           each compared with its lm8 twin

Roles are declared rather than read off aspect ratio. Aspect was the old test,
and it would have put the square (aspect 1) on the disc trend and into the
corridor comparison.

Scales
------
Fields are grouped into scales 0 (finest) to 5 (coarsest): geometric in radius
at ratio 1.6, starting from the Rule 8 floor of that arena. The bank column is
still called `scale_band`; it is the same thing.

Truncation
----------
Rules 8 and 9 bound field size by construction -- floor at Harland's smallest
measured field (0.023 m^2 in 18.6 m^2), ceiling at 20% of arena area. The fits
account for that window, but the window is still the rule speaking: the
fraction of fields resting on each bound is reported beside every fit, and
agreement at the fine end is partly assumed rather than found.

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
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import matplotlib.colors as mcolors
from scipy import optimize, special, stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch
import rules as R
from realm_tools.experiment_lib.reporting import ExperimentReport

# The area sweep: varying area at fixed shape and landmark count -- small,
# medium, mega. Only these three enter the trend against area.
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
# Deliberately not area-matched to any disc, and kept off the area trend by
# its role: it is here to be scored on field length, not on area.
ELIAV_ENVS = ['corr_lm8_l10w2']

# The 10 x 10 m square: eight panels at equal arc length around the perimeter,
# which puts two on each wall. Not area-matched to a disc (100 m^2 against the
# r = 6 disc's 113), so it is reported beside the area sweep, not on it.
SQUARE_ENVS = ['corr_lm8_l10w10']

# No-landmark copies. Each has exactly the walls and position grid of its lm8
# twin and no panels, so the landmarks are the only difference within a pair.
NO_LANDMARK_ENVS = ['circ_lm0_r3', 'circ_lm0_r6', 'corr_lm0_l10w10',
                    'corr_lm0_l10w2']

ALL_ENVS = AREA_ENVS + ELIAV_ENVS + SQUARE_ENVS + NO_LANDMARK_ENVS

# Declared, never inferred from geometry: aspect ratio cannot tell the square
# from a disc, and an lm0 disc has exactly the area of its lm8 twin, so either
# test would put an arena into a comparison it was not built for.
ROLES = {**{e: 'area sweep' for e in AREA_ENVS},
         **{e: 'Eliav corridor' for e in ELIAV_ENVS},
         **{e: 'square' for e in SQUARE_ENVS},
         **{e: 'no landmarks' for e in NO_LANDMARK_ENVS}}


def env_role(name):
    """The comparison an arena belongs to. 'other' keeps it out of all of them."""
    return ROLES.get(name, 'other')


def landmark_twin(name):
    """The lm8 arena an lm0 arena copies: circ_lm0_r3 -> circ_lm8_r3."""
    return name.replace('_lm0_', '_lm8_') if '_lm0_' in name else None

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

# Rule 12's coverage threshold, as the rules define it. Read from there
# rather than written down again, so the two cannot drift apart. It is 0 --
# coverage is measured and reported, NOT enforced, and is not one of the three
# admission rules. A run at any other value restores the filter and is a
# DIFFERENT EXPERIMENT, kept apart from this one everywhere: its own cache
# directory, its own figure directory, and its own line in the report. See
# --tiling-frac-min.
DEFAULT_TILING = float(R.DEFAULT_CFG['TILING_FRAC_MIN'])


def coverage_tag(tiling_frac_min):
    """The coverage setting as it appears in a bank FILENAME, always present.

    Unconditional, and that is the point. When Rule 12 stopped being an
    admission rule the default moved from 0.50 to 0, so a filename carrying no
    coverage marker -- as every bank built before that change does -- would be
    claimed by the new default and read back as a library it is not. Two
    different libraries must never be able to share a name. An old
    `..._roff_bank.csv` matches no key this function can produce, so it is
    inert rather than dangerous.
    """
    return f'_cov{float(tiling_frac_min):g}'


def tiling_suffix(tiling_frac_min):
    """The DIRECTORY suffix a coverage setting earns, empty at the default.

    Empty at the default so the standard model's outputs live in the unadorned
    path every other script points at. Restoring the filter sends a run
    somewhere else entirely, because it is a different experiment.
    """
    tf = float(tiling_frac_min)
    return '' if abs(tf - DEFAULT_TILING) < 1e-12 else f'_cov{tf:g}'


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
SCALES = list(range(6))    # scale occupancy reported over scales 0-5
HIST_PCTL = 95             # S1 shows each library up to this percentile

# Figure colours, checked with a palette validator (OKLab distance, Machado
# colour-blind simulation) on the light surface rather than chosen by eye.
#
# The three fitted forms are identities, so each takes a categorical slot. Their
# curves cross, so the check is all-pairs: worst colour-blind pair dE 9.2 (>= 8
# target), worst normal-vision pair 24.0. Aqua is under 3:1 against the
# surface, which the legend and each panel's "best:" label relieve.
FORM_COLORS = {'lognormal': '#2a78d6', 'exponential': '#eb6834',
               'gaussian': '#1baf7a'}
# Field size runs along a colour gradient, not a single hue: viridis reversed,
# so a small field is green and a large one deep purple, through teal and blue
# on the way, with lightness falling steadily. A one-hue blue ramp was tried
# first and could not hold six separable scales -- consecutive steps sat ~0.047
# apart in OKLab lightness against a 0.06 target -- and read as flat besides.
#
# The ramp starts a quarter of the way in, the first point whose colour clears
# 2:1 contrast against the surface (#5cc863, 2.06:1). Viridis reversed opens on
# yellow, which measures 1.63:1: an outline that faint is barely visible, and
# it would have been the finest fields wearing it, the most numerous ones.
#
# A deliberate departure from "one hue for magnitude": viridis is monotone in
# lightness and colour-blind safe, the figure carries a colourbar to read it
# by, and colour here encodes size alone, never identity.
SIZE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'vpce_size', matplotlib.colormaps['viridis_r'](np.linspace(0.25, 1.0, 256)))
# One colour per scale for S2a, taken from the middle of that scale's slice of
# the same ramp, so a scale's panel there matches the colour its fields take in
# S2b.
SCALE_COLORS = [mcolors.to_hex(SIZE_CMAP(t)) for t in np.linspace(0.08, 0.92, 6)]
# Kept thin: at 1.7 pt the few scale-5 outlines were the darkest, heaviest ink
# in a panel of thousands and buried everything finer.
SCALE_LINEWIDTHS = [0.3, 0.4, 0.5, 0.65, 0.85, 1.1]
INK, INK_2, MUTED = '#0b0b0b', '#52514e', '#898781'   # primary, secondary, muted text
RULE_GRAY = '#c3c2b7'                                 # axes and histogram bars
SURFACE = '#fcfcfb'

# Published reference values, for the report and the figures.
HARLAND_CV = {'CA1 small': 70.0, 'CA1 medium': 85.0, 'CA1 megaspace': 101.0}
HARLAND_COVERAGE = (0.09, 0.13)
# Harland's megaspace is 8.8x their small environment. Our area sweep is
# 11.1x, so the two spans are comparable and their per-quantity changes across
# that span are numbers to hit rather than directions.
HARLAND_AREA_RATIO = 8.8
# Coverage per cell SATURATES: only ~2 percentage points higher in the
# megaspace than in the small environment despite 8.8x the area.
HARLAND_COVERAGE_RISE_PP = 2.0
# Their reported fit quality in the r_hist statistic, and the headline
# shape number from the megaspace.
HARLAND_R_EXPON, HARLAND_R_GAUSS = 0.995, 0.985
HARLAND_FRAC_UNDER_1M2 = 0.78

# Area counts as varied only when it spans at least this ratio, rather than
# `nunique() > 1`, so two arenas meant to share an area can never be read as
# an area axis over a rounding difference.
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
    ('n_scales_occupied',  'scales occupied',               +1, 'ours',
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

# Location is pinned at 0 for the two positive-support forms: field area cannot
# be negative, and a free location would let each fit slide its own origin,
# which makes the AIC comparison between forms meaningless. `k` counts the
# parameters each fit estimates.
FORMS = {
    'lognormal':   dict(dist=stats.lognorm, k=2),    # shape s, scale
    'exponential': dict(dist=stats.expon,   k=1),    # scale
    'gaussian':    dict(dist=stats.norm,    k=2),    # mean, sd
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


def _params(name, theta, lo, hi):
    """Optimiser vector -> scipy's positional parameters.

    The optimiser works on log(shape) and log(scale), so it can never propose
    a negative one and needs no bounds. The Gaussian's mean is mapped into
    (lo, hi) the same way; _log_lik says why it is held there.
    """
    if name == 'lognormal':
        return (float(np.exp(theta[0])), 0.0, float(np.exp(theta[1])))
    if name == 'exponential':
        return (0.0, float(np.exp(theta[0])))
    return (float(lo + (hi - lo) * special.expit(theta[0])),
            float(np.exp(theta[1])))


def _log_std_normal_mass(z_lo, z_hi):
    """log(Phi(z_hi) - Phi(z_lo)), from whichever tail keeps it accurate."""
    if z_lo > 0:
        a, b = special.log_ndtr(-z_lo), special.log_ndtr(-z_hi)
    else:
        a, b = special.log_ndtr(z_hi), special.log_ndtr(z_lo)
    return a + np.log1p(-np.exp(b - a)) if b < a else -np.inf


def _suff(x):
    """The sums every log-likelihood here needs, so a fit never revisits x."""
    lx = np.log(x)
    return len(x), float(x.sum()), float((x * x).sum()), float(lx.sum()), float((lx * lx).sum())


def _log_lik(name, theta, S, lo, hi):
    """Log-likelihood of one form truncated to [lo, hi], from sufficient sums.

    Each form's summed log density depends on the data only through a few
    sums, so an evaluation costs the same at 30 fields as at 3000. Evaluating
    over the fields instead made a 200-draw bootstrap take ~6 minutes per
    library.

    The Gaussian's mean is held inside the window. A Gaussian peaking far below
    the floor shows only its falling tail inside the window, which is an
    exponential under another name: unconstrained, it won 6 of 30 synthetic
    exponential libraries with a mean of -43 m^2. Harland's Gaussian is a bell
    with its peak in range, and that is the shape this one fits.
    """
    n, s1, s2, l1, l2 = S
    if name == 'lognormal':
        s, m = np.exp(theta[0]), theta[1]
        ll = (-l1 - n * np.log(s) - 0.5 * n * np.log(2 * np.pi)
              - (l2 - 2 * m * l1 + n * m * m) / (2 * s * s))
        lm = _log_std_normal_mass((np.log(lo) - m) / s, (np.log(hi) - m) / s)
    elif name == 'exponential':
        scale = np.exp(theta[0])
        ll = -n * np.log(scale) - s1 / scale
        lm = -lo / scale + np.log(-np.expm1(-(hi - lo) / scale))
    else:
        mu, sd = _params(name, theta, lo, hi)
        ll = (-n * np.log(sd) - 0.5 * n * np.log(2 * np.pi)
              - (s2 - 2 * mu * s1 + n * mu * mu) / (2 * sd * sd))
        lm = _log_std_normal_mass((lo - mu) / sd, (hi - mu) / sd)
    return ll - n * lm


def _start(name, S, lo, hi):
    """Optimiser starting point: the untruncated fit, from the same sums."""
    n, s1, s2, l1, l2 = S
    if name == 'lognormal':
        m = l1 / n
        return np.array([0.5 * np.log(max(l2 / n - m * m, 1e-12)), m])
    if name == 'exponential':
        return np.array([np.log(s1 / n)])
    mean, var = s1 / n, max(s2 / n - (s1 / n) ** 2, 1e-12)
    u = np.clip((mean - lo) / (hi - lo), 1e-3, 1 - 1e-3)
    return np.array([special.logit(u), 0.5 * np.log(var)])


def _ks_stat(x, cdf):
    """Kolmogorov-Smirnov distance between a sample and a CDF."""
    x = np.sort(x)
    F = cdf(x)
    i = np.arange(1, len(x) + 1)
    return float(max((i / len(x) - F).max(), (F - (i - 1) / len(x)).max()))


class Truncated:
    """One form restricted to [lo, hi], its density renormalised to that window.

    The probability of the window, P(lo <= X <= hi), is taken from whichever
    tail keeps it accurate: from the CDF when lo is in the lower half, from the
    survival function otherwise. The plain difference cdf(hi) - cdf(lo)
    cancels to zero when the window sits far into the upper tail, which a
    Gaussian fitted to a heavy fine end readily does.
    """

    def __init__(self, dist, par, lo, hi):
        self.dist, self.par, self.lo, self.hi = dist, tuple(par), lo, hi
        self.upper = dist.cdf(lo, *par) >= 0.5
        if self.upper:
            a, b = dist.logsf(lo, *par), dist.logsf(hi, *par)
        else:
            a, b = dist.logcdf(hi, *par), dist.logcdf(lo, *par)
        self.log_mass = (a + np.log1p(-np.exp(b - a))
                         if np.isfinite(a) and b < a else -np.inf)

    @property
    def ok(self):
        return np.isfinite(self.log_mass)

    def logpdf(self, x):
        return self.dist.logpdf(x, *self.par) - self.log_mass

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= self.lo) & (x <= self.hi)
        return np.where(inside, np.exp(self.logpdf(x)), 0.0)

    def cdf(self, x):
        d, p, m = self.dist, self.par, np.exp(self.log_mass)
        c = ((d.sf(self.lo, *p) - d.sf(x, *p)) if self.upper
             else (d.cdf(x, *p) - d.cdf(self.lo, *p))) / m
        return np.clip(c, 0.0, 1.0)

    def rvs(self, n, rng):
        """n draws from the truncated form, by inverting within the window.

        Returns an empty array rather than raising when the window's endpoints
        do not come back finite and ordered -- which happens when a scale
        parameter has collapsed, making both endpoints NaN. rng.uniform raises
        OverflowError on that, and the caller only wants to skip the draw.
        """
        d, p = self.dist, self.par
        a, b = ((d.sf(self.hi, *p), d.sf(self.lo, *p)) if self.upper
                else (d.cdf(self.lo, *p), d.cdf(self.hi, *p)))
        if not (np.isfinite(a) and np.isfinite(b)) or not b > a:
            return np.empty(0)
        u = rng.uniform(a, b, n)
        return d.isf(u, *p) if self.upper else d.ppf(u, *p)


#: Distinct values a sample needs before three forms can be told apart, and
#: the relative spread it needs to be a distribution rather than a point mass.
#: A scale that Rule 12 was deleting is typically pinned at the Rule 8 area
#: floor, so relaxing the coverage requirement readily produces both.
MIN_DISTINCT = 5
MIN_REL_SPREAD = 1e-6

#: A scale parameter at or below this is a point mass, not a distribution.
#: Fits that reach it are refused rather than returned, because everything
#: downstream -- the truncated mass, the CDF, the bootstrap draw -- divides by
#: it.
SCALE_FLOOR = 1e-12


def fit_truncated(name, x, lo, hi, start=None):
    """Maximum-likelihood fit of one form truncated to [lo, hi].

    Returns (params, loglik, theta), theta being the optimiser vector a
    bootstrap refit can start from, or None if no parameters put the sample
    inside the window with finite likelihood.
    """
    S = _suff(x)

    def nll(theta):
        # Nelder-Mead probes freely, including where a scale parameter
        # underflows to exactly 0. `_params` returns Python floats, so such a
        # probe raises ZeroDivisionError rather than producing inf the way a
        # numpy division would, and np.errstate does not catch that. An
        # unreachable point is worth +inf, not a dead run: this is what killed
        # a whole eight-arena job on one degenerate library.
        try:
            with np.errstate(all='ignore'):
                v = -_log_lik(name, theta, S, lo, hi)
        except (ZeroDivisionError, FloatingPointError, OverflowError):
            return np.inf
        return v if np.isfinite(v) else np.inf

    theta0 = _start(name, S, lo, hi) if start is None else start
    res = optimize.minimize(nll, theta0, method='Nelder-Mead',
                            options=dict(xatol=1e-7, fatol=1e-7, maxiter=4000))
    if not np.isfinite(res.fun):
        return None
    par = _params(name, res.x, lo, hi)
    # The scale is the last element of every form's parameter tuple here.
    if not np.all(np.isfinite(par)) or float(par[-1]) <= SCALE_FLOOR:
        return None
    return par, -float(res.fun), res.x


def fit_forms(x, lo, hi, n_boot=N_BOOT, seed=0):
    """Fit all three forms to one sample, each truncated to [lo, hi].

    lo and hi are the admission window the sample passed through -- the Rule 8
    floor and Rule 9 ceiling, in the same units as x. Fitting without them
    lets log-normal win whatever the true shape (module docstring).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < MIN_FIELDS:
        return []
    # A sample with no spread is a point mass and there is nothing to compare
    # three forms on. Refused here, with nothing returned, rather than left to
    # fail somewhere inside the optimiser: a library whose fields all sit on
    # the Rule 8 area floor is a real outcome -- it is what a scale looks like
    # when Rule 12's coverage test is not there to delete it -- and it has to
    # be survivable, not fatal.
    med = float(np.median(x))
    if (len(np.unique(x)) < MIN_DISTINCT or
            (med > 0 and (float(x.max()) - float(x.min())) / med
             < MIN_REL_SPREAD)):
        return []
    # Fields exactly on a bound can land a rounding error outside it, where
    # their likelihood is zero. Widen the window by that much and no more.
    lo, hi = min(lo, float(x.min())), max(hi, float(x.max()))
    dens, centres, nb = _hist(x)
    rng = np.random.default_rng(seed)
    out = []
    for name, spec in FORMS.items():
        fit = fit_truncated(name, x, lo, hi)
        if fit is None:
            continue
        par, ll, theta = fit
        T = Truncated(spec['dist'], par, lo, hi)
        # The guard has been on Truncated all along and was never consulted.
        # Without it a form whose window carries no probability goes on to
        # produce a NaN CDF and an undrawable bootstrap.
        if not T.ok:
            continue
        k = spec['k']

        # Harland's statistic: fitted pdf against the binned density.
        pdf_c = T.pdf(centres)
        r_hist = (float(np.corrcoef(dens, pdf_c)[0, 1])
                  if np.std(pdf_c) > 0 and np.std(dens) > 0 else np.nan)

        # KS with a parametric-bootstrap p. Each synthetic sample is drawn
        # from the fitted truncated form and refitted the same way, so it
        # carries the same window and the same parameter-estimation optimism
        # as the observed statistic, which the analytic p does not.
        ks = _ks_stat(x, T.cdf)
        n_ge = n_ok = 0
        with np.errstate(all='ignore'):
            for _ in range(n_boot):
                xs = T.rvs(len(x), rng)
                xs = xs[np.isfinite(xs) & (xs > 0)]
                if len(xs) < MIN_FIELDS:
                    continue
                refit = fit_truncated(name, xs, lo, hi, start=theta)
                if refit is None:
                    continue
                Ts = Truncated(spec['dist'], refit[0], lo, hi)
                if not Ts.ok:
                    continue
                n_ge += _ks_stat(xs, Ts.cdf) >= ks
                n_ok += 1
        out.append(dict(form=name, n=len(x), n_bins=nb,
                        params=[float(v) for v in par],
                        trunc_lo=float(lo), trunc_hi=float(hi),
                        loglik=ll, k=k, aic=2 * k - 2 * ll,
                        r_hist=r_hist, ks=ks,
                        ks_p_boot=(n_ge + 1) / (n_ok + 1) if n_ok else np.nan))
    if out:
        a_min = min(d['aic'] for d in out)
        w = np.exp(-0.5 * np.array([d['aic'] - a_min for d in out]))
        best = min(out, key=lambda d: d['aic'])['form']
        for d, wi in zip(out, w / w.sum()):
            d['winner'] = best
            d['d_aic'] = d['aic'] - a_min
            d['aic_weight'] = float(wi)
    return out


# --------------------------------------------------------------- statistics

def describe(bank, env, C):
    """The non-fit descriptors: CV, extremes, scales, coverage, truncation."""
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
        # The declared role decides which comparisons a row enters; aspect and
        # landmark count are carried for the tables, never used to decide it.
        role=env.get('role', 'other'),
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
    scale = bank.scale_band.to_numpy(dtype=int)
    for s in SCALES:
        d[f'scale{s}_frac'] = float(np.mean(scale == s))
    d['scale6plus_frac'] = float(np.mean(scale > max(SCALES)))
    d['n_scales_occupied'] = int(len(np.unique(scale)))
    return d


def scale_table(bank, env, C, tag):
    """One row per scale: the library is a tiling at every scale.

    A tiling at scale s needs about arena/s tiles, so the finest scale always
    holds most of the library and always sets any pooled median, mean or CV.
    Measured on the first full sweep: scale 0 is 61-65% of every channel's
    library, and the pooled median coverage (0.26%) is simply scale 0's.

    Per scale the picture is different -- scales 4 and 5 sit at 7-9% and 15-17%
    of the arena, bracketing Harland's 9-13% -- so the model does reach their
    size and the pooled statistic hides it. Report both; the per-scale numbers
    are the ones comparable to a recorded sample of cells.
    """
    if not len(bank):
        return []
    area = float(env['env_area'])
    rows = []
    for scale, g in bank.groupby('scale_band'):
        a = g.area_env_m2.to_numpy(dtype=float)
        rows.append(dict(
            tag, scale=int(scale), n_fields=len(a),
            share_of_library=float(len(a) / len(bank)),
            area_median_m2=float(np.median(a)),
            coverage_median=float(np.median(a) / area),
            # Total floor this scale lays down, as a multiple of the arena.
            # Around 1.0 means the scale tiles it once.
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

    for semi-axes a, b at orientation t. Only the Eliav corridor is scored this
    way. A disc has no long axis, and the square's "long axis" is an arbitrary
    pick between two equal sides, so for every other arena this returns
    nothing.
    """
    if not len(bank) or env.get('role') != 'Eliav corridor':
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
    # The area sweep only: discs at one landmark count. The corridor and the
    # square differ in shape, and an lm0 disc sits at exactly its twin's area,
    # so letting either in would put shape or landmarks into a slope that is
    # meant to be about area alone.
    if 'role' in summary.columns:
        summary = summary[summary.role == 'area sweep']
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
            # Our direction, with a deadband: a change under 10% across an 11x
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
        # The coverage setting is always in the name -- see coverage_tag.
        cov = coverage_tag(base_C.get('TILING_FRAC_MIN', DEFAULT_TILING))
        return f'{out_dir}/{env_name}/{cname}_p{p}_t{t:g}_r{r}{cov}_bank.csv'

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


def _save(fig, fig_dir, name, dpi=150):
    p = os.path.join(fig_dir, name)
    fig.savefig(p, dpi=dpi, bbox_inches='tight')
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


def _style_axes(ax):
    """Recessive chrome: hairline left and bottom axes, muted ticks, no box."""
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(RULE_GRAY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=6, colors=MUTED, length=2, width=0.6)


def _env_label(e, geom):
    """Row label: arena name, area and role."""
    area = geom.get('env_area')
    return (f'{e}\n' + ('' if area is None else f'{area:.0f} m$^2$ · ')
            + geom.get('role', env_role(e)))


def fig_distributions(banks_all, fits, envs, chans, env_geom, fig_dir):
    """S1: the size histogram per arena x channel, with all three fits drawn.

    Arena on the ROW and channel on the COLUMN, each named once.

    Linear axes, each panel stopped at its own HIST_PCTL percentile. A library
    is mostly fine-scale fields with a long, thin coarse tail, so a linear axis
    drawn out to the largest field packs almost every bar against zero.
    Stopping at the 95th percentile gives the width to the bulk of the
    distribution. The fields beyond the cut are counted in the panel, stay in
    the bar heights' denominator, and stay in every fit -- only the view is cut.

    Bars are the percentage of ALL the library's fields in each bin, so the
    hidden tail is not quietly shared out among the visible bars. Each curve is
    a form's truncated density times the bin width: the percentage that form
    predicts for a bin. The best fit by AIC is drawn heavier and named in the
    panel. The dotted line is the Rule 8 floor, below which no field can exist.

    A channel that collapsed still gets a panel, labelled with its field
    count, so a real failure does not look like a plotting gap.
    """
    key = lambda e, c: (e, c, DEFAULT_PCTL, DEFAULT_T, PRIMARY_IOU)
    envs = [e for e in envs if any(key(e, c) in banks_all for c in chans)]
    if not envs:
        return
    fig, axes = plt.subplots(len(envs), len(chans), squeeze=False,
                             figsize=(2.9 * len(chans), 2.2 * len(envs) + 1.0))
    fig.patch.set_facecolor(SURFACE)
    fo = at_operating_point(fits)
    for i, e in enumerate(envs):
        geom = env_geom.get(e, {})
        floor = R.DEFAULT_CFG['RULE8_AREA_FRAC'] * geom.get('env_area', np.nan)
        for j, c in enumerate(chans):
            ax = axes[i][j]
            _style_axes(ax)
            if i == 0:
                ax.set_title(c, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f'{_env_label(e, geom)}\n\n% of fields',
                              fontsize=7.5, color=INK_2)

            b = banks_all.get(key(e, c))
            n = 0 if b is None else len(b)
            if n < MIN_FIELDS:
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, f'{n} field' + ('' if n == 1 else 's') +
                        '\ntoo few to fit', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color=INK_2)
                continue

            x = b.area_env_m2.to_numpy(dtype=float)
            cut = float(np.percentile(x, HIST_PCTL))
            shown = x[x <= cut]
            # Freedman-Diaconis on the part that is drawn, kept to 12-40 bins
            # so a panel is neither a comb nor a handful of blocks.
            edges = np.histogram_bin_edges(shown, bins='fd', range=(0.0, cut))
            if not 12 <= len(edges) - 1 <= 40:
                edges = np.linspace(0.0, cut,
                                    int(np.clip(len(edges) - 1, 12, 40)) + 1)
            counts, _ = np.histogram(shown, bins=edges)
            bw = float(edges[1] - edges[0])
            bars = 100.0 * counts / n
            ax.bar(edges[:-1], bars, width=bw, align='edge', color=RULE_GRAY,
                   edgecolor=SURFACE, linewidth=0.5)
            top = float(bars.max())

            best = None
            sub = (fo[(fo.env == e) & (fo.channel == c) & (fo.variable == 'area')]
                   if len(fo) else fo)
            for _, row in sub.iterrows():
                T = Truncated(FORMS[row.form]['dist'], json.loads(row.params),
                              row.trunc_lo, row.trunc_hi)
                gx = np.linspace(row.trunc_lo, cut, 300)
                curve = 100.0 * bw * T.pdf(gx)
                won = row.d_aic == 0
                ax.plot(gx, curve, color=FORM_COLORS[row.form],
                        lw=2.0 if won else 1.0, alpha=1.0 if won else 0.8,
                        zorder=3 if won else 2, solid_capstyle='round')
                top = max(top, float(np.nanmax(curve)))
                if won:
                    best = row
            if np.isfinite(floor):
                ax.axvline(floor, color=MUTED, lw=0.8, ls=':', zorder=1)

            note = [f'n = {n:,}']
            if best is not None:
                note.append(f'best: {best.form} (r = {best.r_hist:.3f})')
            note.append(f'{n - len(shown):,} fields > {cut:.2g} m$^2$ not shown')
            ax.text(0.98, 0.97, '\n'.join(note), transform=ax.transAxes,
                    ha='right', va='top', fontsize=6, color=INK_2,
                    linespacing=1.35)
            ax.set_xlim(0.0, cut)
            ax.set_ylim(0.0, top * 1.45)
            if i == len(envs) - 1:
                ax.set_xlabel('field area (m$^2$)', fontsize=8, color=INK_2)

    handles = ([Line2D([], [], color=FORM_COLORS[f], lw=2.0) for f in FORMS]
               + [Patch(color=RULE_GRAY), Line2D([], [], color=MUTED, lw=0.8, ls=':')])
    labels = list(FORMS) + ['fields in each bin', 'Rule 8 floor']
    height = fig.get_figheight()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1 - 0.62 / height))
    fig.suptitle('S1  field-size distribution per arena and channel\n'
                 f'linear axes; each panel stops at its own {HIST_PCTL}th '
                 'percentile (the tail stays in the fit); heavier curve = best '
                 'fit by AIC, each form fitted within the Rule 8/9 size window',
                 fontsize=10, color=INK, y=1 - 0.08 / height)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / height))
    _save(fig, fig_dir, 'S1_size_distributions.png')


def _half_extents(geom):
    """Half width and half height of the box that holds an arena, centred on it."""
    if geom.get('is_circular'):
        return float(geom['env_R']), float(geom['env_R'])
    return (float(max(abs(geom.get('x_min', 0.0)), abs(geom.get('x_max', 0.0)))),
            float(max(abs(geom.get('y_min', 0.0)), abs(geom.get('y_max', 0.0)))))


# Below the arena every map panel keeps a strip this fraction of its half width
# deep, which holds the scale bar and the field count clear of any field.
MAP_MARGIN = 0.18


def _map_frame(geom):
    """A map panel's half width, half height, and height/width ratio.

    Taken from the arena's own shape, so a 10 x 2 m corridor gets a short wide
    panel rather than a strip floating in a square of empty floor.
    """
    hx, hy = _half_extents(geom)
    lim_x, lim_y = 1.04 * hx, 1.04 * hy
    return lim_x, lim_y, (2 * lim_y + MAP_MARGIN * lim_x) / (2 * lim_x)


def _draw_arena(ax, geom, lim_x, lim_y):
    """Blank map panel: equal aspect, no ticks, the wall and the landmarks."""
    ax.set_facecolor(SURFACE)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_y - MAP_MARGIN * lim_x, lim_y)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if geom.get('is_circular'):
        ax.add_patch(plt.Circle((geom.get('env_cx', 0.0), geom.get('env_cy', 0.0)),
                                geom['env_R'], fill=False, color=INK_2, lw=0.9,
                                zorder=10))
    elif 'x_min' in geom:
        ax.add_patch(plt.Rectangle((geom['x_min'], geom['y_min']),
                                   geom['x_max'] - geom['x_min'],
                                   geom['y_max'] - geom['y_min'],
                                   fill=False, color=INK_2, lw=0.9, zorder=10))
    for lx, ly in geom.get('landmarks', []):
        ax.plot(lx, ly, 's', ms=2.4, color=INK, zorder=11)


def _scale_bar(ax, lim_x, lim_y):
    """A round-length bar about a fifth of the panel wide, in the bottom margin."""
    length = max(v for v in (0.1, 0.2, 0.5, 1, 2, 5, 10) if v <= 0.4 * lim_x)
    x0, y0 = -0.96 * lim_x, -lim_y - 0.55 * MAP_MARGIN * lim_x
    ax.plot([x0, x0 + length], [y0, y0], color=INK, lw=1.6,
            solid_capstyle='butt', zorder=12)
    ax.text(x0 + length + 0.05 * lim_x, y0, f'{length:g} m', fontsize=6,
            color=INK_2, va='center')


def _field_ellipses(bank):
    """Each field's Rule 7 ellipse, as patches for one PatchCollection."""
    return [Ellipse((r.centroid_x, r.centroid_y), 2 * r.semi_major_m,
                    2 * r.semi_minor_m, angle=np.degrees(r.orientation_rad))
            for r in bank.itertuples()]


def _count_label(ax, n):
    ax.text(0.98, 0.015, f'n = {n:,}', transform=ax.transAxes, ha='right',
            va='bottom', fontsize=6, color=INK_2)


def fig_scale_maps(banks_all, envs, chans, env_geom, fig_dir, C):
    """S2a: one figure per arena -- channels on the rows, scales on the columns.

    A library is roughly two-thirds scale-0 fields, and drawn together they
    bury everything coarser. Split out, every scale gets a panel of its own:
    a column shows how one scale tiles the arena, and reading along a row
    shows the ladder for one channel. Every panel in a figure is the same
    arena at the same size, so field sizes compare directly across the page.

    Column headings give each scale's radius range in metres for this arena.
    Scales are geometric from the Rule 8 floor, which is a fixed fraction of
    arena area, so scale k is physically larger in a larger arena; that is why
    this is one figure per arena rather than one per channel.
    """
    key = lambda e, c: (e, c, DEFAULT_PCTL, DEFAULT_T, PRIMARY_IOU)
    for e in envs:
        if not any(key(e, c) in banks_all for c in chans):
            continue
        geom = env_geom.get(e, {})
        lim_x, lim_y, ratio = _map_frame(geom)
        area = float(geom['env_area'])
        r_min = np.sqrt(C['RULE8_AREA_FRAC'] * area / np.pi)
        r_max = np.sqrt(C['RULE9_AREA_FRAC'] * area / np.pi)
        fig, axes = plt.subplots(len(chans), len(SCALES), squeeze=False,
                                 figsize=(2.25 * len(SCALES),
                                          len(chans) * (2.25 * ratio + 0.35) + 1.1))
        fig.patch.set_facecolor(SURFACE)
        for i, c in enumerate(chans):
            b = banks_all.get(key(e, c))
            for s in SCALES:
                ax = axes[i][s]
                _draw_arena(ax, geom, lim_x, lim_y)
                g = b[b.scale_band == s] if b is not None and len(b) else b
                n = 0 if g is None else len(g)
                if n:
                    ax.add_collection(PatchCollection(
                        _field_ellipses(g),
                        facecolors=[mcolors.to_rgba(SCALE_COLORS[s], 0.22)],
                        edgecolors=[mcolors.to_rgba(SCALE_COLORS[s], 0.95)],
                        linewidths=0.6, zorder=2))
                _count_label(ax, n)
                if i == 0:
                    lo_r = r_min * C['BAND_RATIO'] ** s
                    hi_r = min(r_min * C['BAND_RATIO'] ** (s + 1), r_max)
                    which = ' (finest)' if s == 0 else (
                        ' (coarsest)' if s == SCALES[-1] else '')
                    ax.set_title(f'scale {s}{which}\nradius {lo_r:.2g}–{hi_r:.2g} m',
                                 fontsize=8.5, color=INK)
                if s == 0:
                    ax.set_ylabel(c, fontsize=10, color=INK)
        _scale_bar(axes[0][0], lim_x, lim_y)
        height = fig.get_figheight()
        fig.suptitle(f'S2a  {e}: admitted fields split by scale\n'
                     f'{area:.0f} m$^2$ · {geom.get("role", env_role(e))} · '
                     f'{geom.get("n_landmarks", 0)} landmarks (black squares) · '
                     f'scales are geometric in radius at ratio '
                     f'{C["BAND_RATIO"]:g} from the Rule 8 floor',
                     fontsize=10, color=INK, y=1 - 0.1 / height)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / height))
        # 110 dpi: eight of these at 150 dpi came to ~27 MB, past the mail
        # attachment budget before anything else was attached.
        _save(fig, fig_dir, f'S2a_scales_{e}.png', dpi=110)


def fig_field_outlines(banks_all, envs, chans, env_geom, fig_dir):
    """S2b: every admitted field drawn on the arena, coloured by its own size.

    Arena on the row, channel on the column. Each arena fills its own panel so
    the finest fields of the small arenas stay visible; the bar on each row
    gives that row's size. Row heights follow each arena's shape, so the
    corridor is a short wide row rather than a strip in a square of blank.

    Each field is a near-transparent fill under a strong outline, both taking
    the same colour: the field's radius on a light-to-dark ramp, read off the
    colourbar in metres. A solid fill hides whatever lies beneath it, and with
    thousands of overlapping fields per library only the top layer would show.

    The ramp is logarithmic, which is what keeps the scales legible within it.
    Scales are geometric in radius at ratio 1.6, so each occupies an equal
    slice of the colour axis, while a field at the top of its scale still
    reads darker than one at the bottom -- which six flat class colours could
    not show. One ramp serves every panel, so a colour means the same size in
    every arena and the r = 10 disc's fields really are drawn darker than the
    r = 3 disc's.

    Line width grows with scale as a second cue, and fine fields are drawn
    first so the rare coarse ones sit on top of the carpet rather than under
    it.
    """
    key = lambda e, c: (e, c, DEFAULT_PCTL, DEFAULT_T, PRIMARY_IOU)
    envs = [e for e in envs if any(key(e, c) in banks_all for c in chans)]
    if not envs:
        return
    pooled = np.concatenate(
        [banks_all[key(e, c)].radius_env_m.to_numpy(dtype=float)
         for e in envs for c in chans
         if key(e, c) in banks_all and len(banks_all[key(e, c)])]
        or [np.array([0.1, 1.0])])
    lo = max(float(np.min(pooled)), 1e-3)
    hi = max(float(np.max(pooled)), lo * 1.01)
    norm = mcolors.LogNorm(vmin=lo, vmax=hi, clip=True)

    frames = [_map_frame(env_geom.get(e, {})) for e in envs]
    # Row heights follow each arena's shape, but never fall below what a row's
    # own label needs: a 10 x 2 m corridor row is a third the height of a disc
    # row, and at that height its rotated label runs into its neighbour's.
    ratios = [max(f[2], 0.55) for f in frames]
    fig, axes = plt.subplots(len(envs), len(chans), squeeze=False,
                             figsize=(2.5 * len(chans),
                                      sum(2.5 * r + 0.35 for r in ratios) + 1.2),
                             gridspec_kw=dict(height_ratios=ratios))
    fig.patch.set_facecolor(SURFACE)
    for i, e in enumerate(envs):
        geom = env_geom.get(e, {})
        lim_x, lim_y, _ = frames[i]
        for j, c in enumerate(chans):
            ax = axes[i][j]
            _draw_arena(ax, geom, lim_x, lim_y)
            b = banks_all.get(key(e, c))
            n = 0 if b is None else len(b)
            for s in SCALES:
                g = b[b.scale_band == s] if n else None
                if g is None or not len(g):
                    continue
                # One colour per field, from its own radius: a wash for the
                # fill, the same colour at full strength for the outline.
                rgba = SIZE_CMAP(norm(g.radius_env_m.to_numpy(dtype=float)))
                face, edge = rgba.copy(), rgba.copy()
                face[:, 3], edge[:, 3] = 0.05, 0.95
                ax.add_collection(PatchCollection(
                    _field_ellipses(g), facecolors=face, edgecolors=edge,
                    linewidths=SCALE_LINEWIDTHS[s], zorder=2 + s))
            _count_label(ax, n)
            if i == 0:
                ax.set_title(c, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(_env_label(e, geom), fontsize=7.5, color=INK_2)
                _scale_bar(ax, lim_x, lim_y)

    # Colour is the size ramp now, so the legend shows the other cue: line
    # width by scale. Drawing these keys in a ramp colour would claim a size
    # each scale does not have, since a scale spans a range of sizes and its
    # position on the ramp moves with the arena.
    labels = [f'scale {s}' + (' (finest)' if s == 0 else
                              ' (coarsest)' if s == SCALES[-1] else '')
              for s in SCALES] + ['landmark']
    handles = ([Line2D([], [], color=INK_2, lw=max(0.8, 1.8 * SCALE_LINEWIDTHS[s]))
                for s in SCALES]
               + [Line2D([], [], color=INK, marker='s', ms=4, ls='')])
    height = fig.get_figheight()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1 - 0.62 / height))
    fig.suptitle('S2b  admitted fields, coloured by field size\n'
                 'green small to purple large, on one logarithmic ramp shared '
                 'by every panel; line width grows with scale and coarse '
                 'fields are drawn on top; each arena fills its own panel',
                 fontsize=10, color=INK, y=1 - 0.08 / height)
    fig.tight_layout(rect=(0, 0, 0.93, 1 - 0.95 / height))
    # Added after tight_layout: a manually placed colour bar is not a
    # tight_layout-compatible axes and warns if it exists during the call.
    # A short bar centred on the figure. Stretched over the full height it
    # reads as a ninth column of data rather than a key.
    cax = fig.add_axes([0.945, 0.40, 0.011, 0.20])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SIZE_CMAP), cax=cax)
    cb.set_label('field radius (m)', fontsize=9, color=INK_2)
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10) if lo <= t <= hi]
    if ticks:
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{t:g}' for t in ticks])
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_visible(False)
    # 130 dpi: at 150 this one figure was 11 MB of antialiased outlines.
    _save(fig, fig_dir, 'S2b_field_outlines.png', dpi=130)


def _scale_panel(ax, s, col, ylabel, pct=False):
    """One quantity against arena area, a line per channel.

    Only the area sweep is joined: the trend is about area at fixed shape and
    landmark count. Every other arena is an open marker at its own area -- a
    circle for a no-landmark disc, a square for a box -- so it reads as a
    comparison beside the curve rather than a point on it.
    """
    sweep = s[s.role == 'area sweep'] if 'role' in s.columns else s
    other = s[s.role != 'area sweep'] if 'role' in s.columns else s.iloc[:0]
    for c in sorted(s.channel.unique()):
        colour = CHANNEL_COLORS.get(c, '0.4')
        g = sweep[sweep.channel == c].sort_values('env_area_m2')
        if len(g):
            ax.plot(g.env_area_m2, 100 * g[col] if pct else g[col], 'o-',
                    ms=5, lw=1.2, color=colour, label=c)
        h = other[other.channel == c]
        for is_disc, marker in ((True, 'o'), (False, 's')):
            k = h[h.env.str.startswith('circ_') == is_disc]
            if len(k):
                ax.plot(k.env_area_m2, 100 * k[col] if pct else k[col], marker,
                        ms=7, mfc='none', mew=1.4, color=colour)
    ax.set_xlabel('arena area (m$^2$)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def fig_size_vs_scale(summary, fig_dir):
    """S3: the size ladder against area.

    Median field size, the max/min spread and the number of occupied
    scales. Together these say whether a larger space buys a wider range of
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
    _scale_panel(axes[2], s, 'n_scales_occupied', 'scales occupied')
    axes[0].set_title('typical field size', fontsize=8)
    axes[1].set_title('size spread', fontsize=8)
    axes[2].set_title('scale diversity', fontsize=8)
    for ax in axes:
        ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle('S3  the size ladder against arena area — does a larger space '
                 'buy a WIDER range of scales, or a uniformly coarser one?\n'
                 'joined circles = the area sweep (shape and landmarks fixed); '
                 'open circles = no-landmark discs; open squares = the corridor '
                 'and the squares, not on the curve',
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
        # A relaxed Rule 12 says so in the subject line: two runs of this
        # experiment otherwise arrive as two identical-looking mails.
        tf = getattr(self, 'tiling_frac_min', DEFAULT_TILING)
        pre = ('' if abs(tf - DEFAULT_TILING) < 1e-12 else
               '[Rule 12 coverage OFF] ' if tf <= 0 else
               f'[Rule 12 coverage {tf:g}] ')
        return (f'{pre}CV {base.cv_area_pct.median():.0f}%, '
                f'max/min {base.area_max_min_ratio.median():.1f}x, {top}')

    def figures(self):
        # What this run wrote, not what is lying in the directory, most useful
        # first. The mail attaches in order and skips whatever no longer fits
        # the size budget, and the eight per-arena S2a maps are the bulk.
        rank = {'S1': 0, 'S2b': 1, 'S3': 2}
        return sorted(FIGURES_WRITTEN,
                      key=lambda p: (rank.get(os.path.basename(p).split('_')[0], 3), p))

    def data_files(self):
        return [p for p in (f'{self.out_dir}/summary.csv',
                            f'{self.out_dir}/fits.csv',
                            f'{self.out_dir}/scale_trends.csv',
                            f'{self.out_dir}/scale_summary.csv',
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

        # A run with the coverage requirement moved is a different experiment
        # and says so before anything else, because every number below it is
        # drawn from a different set of libraries.
        tf = getattr(self, 'tiling_frac_min', DEFAULT_TILING)
        if abs(tf - DEFAULT_TILING) > 1e-12:
            out.append(S('THE COVERAGE FILTER IS SWITCHED BACK ON IN THIS RUN',
                         '\n'.join([
                f'TILING_FRAC_MIN = {tf:g}, against the default of '
                f'{DEFAULT_TILING:g}. Coverage is normally measured and '
                'reported but admits nothing; here Rule 12 is deleting scales '
                'again, so this run has FOUR admission rules instead of the '
                'standard three (size range, contiguity, competition).', '',
                'Rule 12 drops a scale whose admitted fields, unioned, cover '
                'less than that fraction of the floor, and keeps the '
                'contiguous run of qualifying scales around the best-covered '
                'one. It polices both ends of the ladder: coarse scales fail '
                'for having too few nodes that large, fine scales because a '
                'tiling at that resolution would need more fields than the '
                'tree produces.', '',
                'So the libraries below are NOT the ones the rest of the '
                'series is built on. The comparison to make is against the '
                'standard run of this same experiment -- same arenas, same '
                'channels, same seed, same code -- whose outputs are in '
                'data_cache/scale_distribution; this run\'s are in '
                f'data_cache/scale_distribution{tiling_suffix(tf)}.', '',
                'Read the scale occupancy and the fitted form first. Rule 12 '
                'removes whole scales from both ends, so switching it back on '
                'NARROWS the size range by construction -- a smaller CV and a '
                'narrower max/min are expected and are not themselves '
                'findings. Whether the FITTED FORM differs, and which scales '
                'the filter removed, are the questions.'])))

        # --- the shape, which is the actual question ----------------------
        w = self.winners
        shape = [
            f'{len(base)} environment x channel libraries at the operating '
            f'point (EXTENT_PCTL {DEFAULT_PCTL}), '
            f'{int(base.n_fields.sum())} fields in total.', '',
            'CV IS MEASURED BUT NOT PLOTTED, and not compared to Fig 6E. '
            'Pooled across scales it describes a six-scale mixture spanning two '
            'orders of magnitude, and its trend across area tracks how many '
            'scales are occupied rather than any field size. Within a scale it '
            'is fixed by the scale definition: scales are geometric in radius '
            'at ratio 1.6, so areas span 2.56x and a uniform spread over that '
            'gives CV = 25%, which is what we measure (23-32%). Harland\'s '
            '70-101 sits between the two, and neither of ours is comparable '
            'until there is a model of how a recording samples cells from '
            'this library. The numbers are in scale_summary.csv.', '',
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

        # --- best fit, and how it sits against the two papers -------------
        forms = ['Every library is fitted with all three published forms, each '
                 'told that fields cannot fall below the Rule 8 floor or above '
                 'the Rule 9 ceiling. Without that an exponential is penalised '
                 'for the empty gap below the floor, and log-normal wins '
                 'whatever the true shape.', '']
        best = fb[fb.d_aic == 0]
        if len(best):
            forms.append(f'Best fit by AIC across {len(best)} libraries: ' +
                         ', '.join(f'{k} {v}' for k, v in
                                   best.form.value_counts().items()) +
                         f'. Median Akaike weight of the winner '
                         f'{best.aic_weight.median():.2f} (1 = decisive, '
                         f'1/3 = the three forms fit about equally).')
            forms.append('')
            order = [e for e in getattr(self, 'env_order', []) if e in set(best.env)]
            order += sorted(set(best.env) - set(order))
            for env_name in order:
                g = best[best.env == env_name]
                forms.append(f'  {env_name:17s} {env_role(env_name):15s} '
                             + ', '.join(f'{k} {v}' for k, v in
                                         g.form.value_counts().items())
                             + f'   (of {len(g)} channels)')

            def _against(env_name, form, published):
                g = fb[fb.env == env_name]
                if not len(g):
                    return f'    {env_name} is not in this run.'
                won = int(((g.form == form) & (g.d_aic == 0)).sum())
                r = g[g.form == form].r_hist.median()
                return (f'    {env_name}: {form} is the best fit in {won} of '
                        f'{g.channel.nunique()} channels; median r_hist for '
                        f'{form} {r:.3f}' + ('' if published is None else
                                             f' against the published {published}')
                        + '.')

            forms += ['', 'Against the two papers:',
                      '  Harland, megaspace: negative exponential.',
                      _against(AREA_ENVS[-1], 'exponential', HARLAND_R_EXPON),
                      '  Harland, small environments: Gaussian.',
                      _against(AREA_ENVS[0], 'gaussian', HARLAND_R_GAUSS),
                      '  Eliav: log-normal.',
                      f'    log-normal is the best fit in '
                      f'{int((best.form == "lognormal").sum())} of {len(best)} '
                      f'libraries across every arena in this run.',
                      _against(ELIAV_ENVS[0], 'lognormal', None),
                      '',
                      f'"Megaspace" here is the r = 10 disc and "small" the r = 3 '
                      f'disc: the ratio between them (11.1x) matches Harland\'s '
                      f'{HARLAND_AREA_RATIO}x, the absolute areas do not -- '
                      f'their megaspace is smaller than our r = 3 disc.', '']
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
        out.append(S('BEST FIT, AND THE TWO PAPERS', '\n'.join(forms)))

        # --- the threshold caveat -----------------------------------------
        inv = self.invariance
        cav = [
            'Harland show their exponential fit becomes quasi-linear at a '
            'lower detection threshold, so distribution shape is not '
            'threshold-independent and the comparison is meaningless without '
            'checking ours.', '',
            '  EXTENT_PCTL saturates at 65, established by run_field_recovery '
            'against ideal place cells of known size, so the default is that '
            'single operating point.',
            '  The first full run swept 50 / 65 / 80 and found log-normal '
            'winning at every setting, but those fits ignored the size window '
            'and would have picked log-normal whatever the shape. Whether the '
            'shape holds across settings is therefore open: re-run with '
            '--settings 50:0.5,65:0.5,80:0.5 to check.', '',
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
        out.append(S('THE THRESHOLD CAVEAT', '\n'.join(cav)))

        # --- what bounds the distribution ---------------------------------
        out.append(S('TRUNCATION', '\n'.join([
            'Rules 8 and 9 bound field size by construction: floor at '
            'Harland\'s smallest measured field (0.023 m^2 in 18.6 m^2), '
            'ceiling at 20% of arena area. The fits above account for that '
            'window, but the window itself is the rule, not the model.',
            f'  fields resting on the Rule 8 floor    median '
            f'{100*base.frac_at_floor.median():.1f}%',
            f'  fields resting on the Rule 9 ceiling  median '
            f'{100*base.frac_at_ceiling.median():.1f}%',
            '',
            'Mass piling against a bound is the rule speaking, not the model, '
            'and the fine end is floored at exactly the value Harland measured '
            '-- so agreement at the fine end is partly assumed rather than '
            'found.'])))

        # --- the library is a tiling at every scale -----------------------
        sc = getattr(self, 'scales', None)
        if sc is not None and len(sc):
            # Column names deliberately not `cov`/`n`: those collide with
            # pandas Series methods and attribute access silently returns the
            # method instead of the value.
            g = sc.groupby('scale').agg(
                n_f=('n_fields', 'sum'), share_lib=('share_of_library', 'median'),
                cov_med=('coverage_median', 'median'),
                tile_mult=('tiling_multiple', 'median'),
                iou_med=('split_half_iou_median', 'median'))
            L = ['A field library is a TILING AT EVERY SCALE, not a sample of '
                 'cells. A tiling at scale s needs about arena/s tiles, so the '
                 'finest scale necessarily holds most of the library and '
                 'necessarily sets any pooled median, mean or CV. Read the '
                 'scales, not the pool. Medians across every library in the '
                 'run:', '',
                 f'  {"scale":>5s} {"fields":>7s} {"share":>7s} '
                 f'{"median cov":>11s} {"tiling":>7s} {"IoU":>5s}']
            for s_, r in g.iterrows():
                L.append(f'  {int(s_):>5d} {int(r.n_f):>7d} {100*r.share_lib:>6.1f}% '
                         f'{100*r.cov_med:>10.2f}% {r.tile_mult:>7.2f} {r.iou_med:>5.2f}')
            hi = g[(g.cov_med >= HARLAND_COVERAGE[0]) &
                   (g.cov_med <= HARLAND_COVERAGE[1])]
            near = g[(g.cov_med >= 0.5 * HARLAND_COVERAGE[0]) &
                     (g.cov_med <= 2.0 * HARLAND_COVERAGE[1])]
            L += ['',
                  f'"tiling" is the floor a scale lays down as a multiple of '
                  f'the arena; ~1 means the scale covers it once. IoU is '
                  f'split-half reliability, which is 0 everywhere until Rule 2 '
                  f'is scored on a coarser grid (below).']
            if len(near):
                L += ['',
                      f'THE MODEL DOES REACH HARLAND\'S SIZE. Scale(s) '
                      f'{", ".join(str(int(i)) for i in near.index)} sit at '
                      + ', '.join(f'{100*v:.1f}%' for v in near.cov_med) +
                      f' of the arena, against their 9-13% per cell'
                      + ('' if len(hi) else ' (bracketing it rather than '
                         'landing inside)') + '. The pooled median is far '
                      f'below that only because scale '
                      f'{int(g.share_lib.idxmax())} is '
                      f'{100*g.share_lib.max():.0f}% of the library.']
            else:
                L += ['',
                      'No scale reaches Harland\'s 9-13% per cell. This is a '
                      'divergence in the coarse tail, not an artifact of '
                      'pooling, and should be reported as such.']
            L += ['',
                  'Neither available lever thins the fine end at present. '
                  'Raising the Rule 8 floor does not work: the median lands '
                  'at about 2x the floor wherever the floor is put, so '
                  'choosing it chooses the answer. Rule 2 (split-half '
                  'reliability) would be the principled route -- reliability '
                  'rose monotonically with scale under the old 0.25 m binning, '
                  'so a threshold removes fine fields for being '
                  'unreproducible rather than for being small -- but at the '
                  'lattice bin the two half-maps land on disjoint bins and '
                  'every IoU is exactly 0. Scoring the halves on a coarser '
                  'grid than the one used for field extent would restore it.']
            out.append(S('PER SCALE — read this before any pooled number',
                         '\n'.join(L)))

        # --- what changes as scale changes -------------------------------
        # The area sweep only. The other arenas have their own sections, and
        # letting them in here would put shape or landmarks into a trend that
        # is meant to be about area alone.
        sweep = base[base.role == 'area sweep'] if 'role' in base.columns else base
        areas = sorted((sweep if len(sweep) else base).env_area_m2.unique())
        tr = getattr(self, 'trends', None)
        if len(sweep) and area_varies(sweep) and tr is not None and len(tr):
            lo, hi = min(areas), max(areas)
            L = [f'{len(areas)} area-sweep arenas, {lo:.1f} to {hi:.0f} m^2 '
                 f'({hi/lo:.1f}x). '
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
                f'This run holds {sweep.env.nunique()} of the '
                f'{len(AREA_ENVS)} area-sweep arenas, so Harland '
                'Fig 6E (CV against enclosure area) and the 3F/3G contrast (a '
                'negative exponential in the megaspace against a Gaussian in '
                'the small environments) CANNOT be read at all. Both are claims '
                'about scale, and scale is not varied.', '',
                'What this run does establish is the shape in the arenas it '
                'holds -- a control, and a prerequisite for reading the area '
                'sweep, but not a test of either published claim.', '',
                'Run over AREA_ENVS (circ_lm8_r3, circ_lm8_r6, '
                'circ_lm8_r10 — 28.3 to 314.2 m^2, 11.1x against '
                'Harland\'s 8.8x) for the comparison this experiment is '
                'named after.'])))

        # Area order, the lm8 arena before its lm0 twin, channels within.
        base = base.sort_values(['env_area_m2', 'n_landmarks', 'env', 'channel'],
                                ascending=[True, False, True, True])
        # --- the Eliav comparison, in one dimension -----------------------
        el = getattr(self, 'eliav', None)
        if el is not None and len(el):
            el = at_operating_point(el)
        if el is not None and len(el):
            L = ['Eliav report field size as a LENGTH along a tunnel, not as '
                 'an area, so ours is measured the same way: how far each '
                 'field reaches along the arena\'s long axis. Only the Eliav '
                 'corridor appears here: a disc has no long axis, and a '
                 'square\'s would be an arbitrary pick of two equal sides.', '']
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

        # --- with and without landmarks ------------------------------------
        def _cell(rows, col, fmt):
            return fmt.format(rows[col].iloc[0]) if len(rows) else '-'

        def _best_form(env_name, c):
            g = fb[(fb.env == env_name) & (fb.channel == c) & (fb.d_aic == 0)]
            return g.form.iloc[0] if len(g) else '-'

        L = []
        for lm0 in sorted(base.env.unique()):
            lm8 = landmark_twin(lm0)
            if lm8 is None:
                continue
            with_lm, without = base[base.env == lm8], base[base.env == lm0]
            if not len(with_lm) or not len(without):
                continue
            L += [f'{lm8} / {lm0}, {with_lm.env_area_m2.iloc[0]:.0f} m^2',
                  f'  {"channel":8s} {"fields":>13s} {"median area m^2":>19s} '
                  f'{"scales":>7s}   best fit']
            for c in [c for c in CHANNELS if c in set(base.channel)]:
                a, b_ = with_lm[with_lm.channel == c], without[without.channel == c]
                L.append(
                    f'  {c:8s} '
                    f'{_cell(a, "n_fields", "{:.0f}"):>6s}/'
                    f'{_cell(b_, "n_fields", "{:.0f}"):<6s} '
                    f'{_cell(a, "area_median_m2", "{:.3g}"):>9s}/'
                    f'{_cell(b_, "area_median_m2", "{:.3g}"):<9s} '
                    f'{_cell(a, "n_scales_occupied", "{:.0f}"):>3s}/'
                    f'{_cell(b_, "n_scales_occupied", "{:.0f}"):<3s}   '
                    f'{_best_form(lm8, c)} / {_best_form(lm0, c)}')
            L.append('')
        if L:
            L += ['Every pair reads lm8 / lm0: the same walls and the same '
                  'position grid, with the panels the only difference, so any '
                  'gap between the two numbers is what the landmarks do.']
            out.append(S('WITH AND WITHOUT LANDMARKS', '\n'.join(L)))

        cols = ['env', 'role', 'env_area_m2', 'channel', 'n_fields', 'cv_area_pct',
                'area_min_m2', 'area_median_m2', 'area_max_m2',
                'area_max_min_ratio', 'coverage_median', 'frac_under_1m2',
                'n_scales_occupied', 'frac_at_floor', 'frac_at_ceiling']
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
                   default=','.join(ALL_ENVS),
                   help='default is all nine arenas (ALL_ENVS). Each has a '
                        'declared role in ROLES that decides which comparisons '
                        'it enters: only the area sweep is on the area trend, '
                        'only the Eliav corridor is scored on length, and each '
                        'no-landmark arena is compared with its lm8 twin. An '
                        'arena missing from ROLES is reported but enters none '
                        'of them.')
    p.add_argument('--channels', default=','.join(CHANNELS))
    p.add_argument('--settings',
                   default=','.join(f'{p}:{t:g}' for p, t in SETTINGS),
                   help='EXTENT_PCTL:ACT_THRESH pairs, comma separated')
    p.add_argument('--lam', type=float, default=0.0,
                   help='LAMBDA. 0 = feature only, as everywhere else.')
    p.add_argument('--tiling-frac-min', type=float, default=DEFAULT_TILING,
                   metavar='FRAC',
                   help='Rule 12: the fraction of the floor a scale\'s '
                        'fields must cover, unioned, for Rule 12 to keep that '
                        f'scale. DEFAULT {DEFAULT_TILING:g} -- coverage is '
                        'MEASURED AND REPORTED BUT NOT ENFORCED, and is not '
                        'one of the three admission rules (size range, '
                        'contiguity, competition). Setting it above 0 restores '
                        'the filter for a comparison run and sends every '
                        'output to ..._cov<value>, its own cache and figures, '
                        'so a filtered run cannot be mistaken for the '
                        'standard one.')
    p.add_argument('--split-half-iou-min', default='none', metavar='LIST',
                   help='Rule 2: reject a field whose split-half IoU is below '
                        'this. CURRENTLY UNUSABLE and the run will refuse it: '
                        'at the lattice bin the two half-maps occupy disjoint '
                        'bins, so every IoU is exactly 0 and any threshold '
                        'rejects the whole library. It was informative under '
                        'the old 0.25 m binning (0.45 at scale 0 rising to 0.69 '
                        'at scale 5) and needs the halves scored on a coarser '
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

    suffix = tiling_suffix(args.tiling_frac_min)
    out_dir = f'{REPO}/data_cache/scale_distribution{suffix}'
    fig_dir = f'{HERE}/figures/scale_distribution{suffix}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    base_C = R.resolve_cfg(dict(LAMBDA=args.lam, RANDOM_SEED=args.seed,
                                USE_GPU=not args.no_gpu,
                                TILING_FRAC_MIN=float(args.tiling_frac_min)))
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
    print(f'  Rule 12  : coverage >= {base_C["TILING_FRAC_MIN"]:g}'
          + ('  (the operating point)' if not suffix else
             '  <-- OFF, the coverage requirement is DISABLED'
             if base_C['TILING_FRAC_MIN'] <= 0 else '  <-- RELAXED')
          + ('' if not suffix else
             f'\n             outputs go to ...{suffix}, not to the '
             f'operating point\'s'))
    print(f'  areas    : {"varies — Fig 6E readable" if len(envs) > 1 else "one"}'
          '  (a single area cannot speak to Harland 3F-G or 6E)')
    print('  note     : EXTENT_PCTL saturates at 65 and the sweep is settled;')
    print('             pass --settings to re-open it. ACT_THRESH cancels')
    print('             under SIGMA_MODE=quantile and is not a knob.')
    print('=' * 72, flush=True)

    banks_all, sum_rows, fit_rows, inv_rows = {}, [], [], []
    scale_rows, env_geom, eliav_rows = [], {}, []
    missing = []
    # Libraries whose distribution could not be fitted: `fit_none` for the
    # ones fit_forms declined (too few fields, too few distinct values, no
    # spread), `fit_failed` for anything that raised. Both are reported rather
    # than leaving a silent gap in fits.csv.
    fit_none, fit_failed = [], []
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
        # The role decides which comparisons the arena enters. Shape and
        # landmark count are carried for the tables and the figures only.
        env['role'] = env_role(e)
        env['n_landmarks'] = len(root.findall('landmark'))
        env['aspect'] = (1.0 if env.get('is_circular') else
                         (env['x_max'] - env['x_min']) /
                         (env['y_max'] - env['y_min']))
        env_geom[e] = dict(env, landmarks=[
            (float(l.get('x')), float(l.get('y')))
            for l in root.findall('landmark')])
        print(f'  arena area {env["env_area"]:.1f} m^2, aspect '
              f'{env["aspect"]:.2f}, {env["n_landmarks"]} landmarks, '
              f'role: {env["role"]}, {len(xy)} locations', flush=True)
        if env['role'] == 'other':
            print(f'  !! {e} has no role in ROLES: it is reported, but enters '
                  f'no comparison', flush=True)

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
                scale_rows.extend(scale_table(bank, env, base_C, tag))
                eliav_rows.extend(eliav_lengths(bank, env, tag))
                # Area is Harland's unit; equivalent diameter is the closest
                # thing we have to Eliav's 1D field width, so both are fitted,
                # each within the Rule 8/9 window expressed in its own unit.
                a_lo = base_C['RULE8_AREA_FRAC'] * env['env_area']
                a_hi = base_C['RULE9_AREA_FRAC'] * env['env_area']
                for var, x, lo, hi in (
                        ('area', bank.area_env_m2, a_lo, a_hi),
                        ('diameter', 2.0 * bank.radius_env_m,
                         2.0 * np.sqrt(a_lo / np.pi), 2.0 * np.sqrt(a_hi / np.pi))):
                    # One library's fit must not cost the other twenty-three.
                    # Each of these is an hour of GPU time upstream of here,
                    # and a degenerate library is a real outcome rather than a
                    # bug in the run -- so it is recorded and stepped over.
                    # `fit_forms` already refuses a point mass by returning
                    # nothing; this catches whatever it does not.
                    try:
                        got = fit_forms(x, lo, hi, n_boot=args.n_boot,
                                        seed=args.seed)
                    except Exception as ex:                    # noqa: BLE001
                        got = []
                        fit_failed.append(dict(env=e, channel=c, variable=var,
                                               n_fields=len(bank),
                                               error=f'{type(ex).__name__}: {ex}'))
                        print(f'  !! [{c}] {var} fit failed, continuing: '
                              f'{type(ex).__name__}: {ex}', flush=True)
                    if not got:
                        n_uniq = int(pd.Series(np.asarray(x, dtype=float))
                                     .nunique())
                        fit_none.append(dict(env=e, channel=c, variable=var,
                                             n_fields=len(bank),
                                             n_distinct=n_uniq))
                    for d in got:
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
    scales = pd.DataFrame(scale_rows)
    if len(scales):
        scales = scales.sort_values(['env_area_m2', 'env', 'channel', 'scale'])
        scales.to_csv(f'{out_dir}/scale_summary.csv', index=False)
    fits.to_csv(f'{out_dir}/fits.csv', index=False)
    if inv is not None:
        inv.to_csv(f'{out_dir}/threshold_invariance.csv', index=False)

    trends = scale_trends(at_operating_point(summary))
    if len(trends):
        trends.to_csv(f'{out_dir}/scale_trends.csv', index=False)
    if fit_none or fit_failed:
        unfit = pd.DataFrame(
            [dict(r, reason='declined: no spread to fit') for r in fit_none] +
            [dict(r, reason='raised') for r in fit_failed])
        unfit.to_csv(f'{out_dir}/unfitted.csv', index=False)
        print(f'\n!! {len(fit_none)} library x variable fits declined and '
              f'{len(fit_failed)} raised -- see {out_dir}/unfitted.csv',
              flush=True)

    _f = at_operating_point(fits)
    winners = _f[(_f.variable == 'area') & (_f.d_aic == 0)].form.value_counts()

    print('\nfigures:', flush=True)
    # Area order, and within one area the lm8 arena before its lm0 twin, so the
    # figures read small -> mega down the page with each pair side by side.
    envs_by_area = list(summary.sort_values(['env_area_m2', 'n_landmarks', 'env'],
                                            ascending=[True, False, True])
                        .drop_duplicates('env').env)
    fig_distributions(banks_all, fits, envs_by_area, chans, env_geom, fig_dir)
    fig_scale_maps(banks_all, envs_by_area, chans, env_geom, fig_dir, base_C)
    fig_field_outlines(banks_all, envs_by_area, chans, env_geom, fig_dir)
    fig_size_vs_scale(summary, fig_dir)
    prune_orphan_figures(fig_dir)

    rep = ScaleDistributionReport(env_name=','.join(envs), out_dir=out_dir,
                                  fig_dir=fig_dir, results=summary,
                                  log_path=os.environ.get('REALM_LOG_PATH'))
    rep.fits, rep.invariance, rep.winners, rep.trends = fits, inv, winners, trends
    rep.scales, rep.eliav, rep.env_order = scales, eliav, envs_by_area
    rep.tiling_frac_min = float(base_C['TILING_FRAC_MIN'])
    rep.fit_none, rep.fit_failed = fit_none, fit_failed
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
