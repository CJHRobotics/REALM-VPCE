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

# The six collected datasets: four landmark counts on one disc, plus the two
# geometry arenas. The landmark counts each divide the clock face evenly from
# noon, so no panel sits on a camera view seam.
# Same area (~28.3 m^2), varying cue density and shape. A control: it says
# whether the distribution moves with anything other than scale.
ENVS = ['circ_lm2_r0', 'circ_lm4_r0', 'circ_lm8_r0', 'circ_lm12_r0',
        'rect_lm8_r0', 'corr_lm8_r0']

# Varying area at fixed shape and landmark count, 4.91 -> 314.16 m^2, a 64x
# range. This is the axis Harland's design actually varies, and the only one
# on which Figs 3F-G and 6E can be read at all -- see the module docstring.
#   --envs "$(python -c 'import run_scale_distribution as m;
#                        print(",".join(m.AREA_ENVS))')"
AREA_ENVS = ['circ_lm8_rad1p25', 'circ_lm8_rad2p0', 'circ_lm8_r0',
             'circ_lm8_rad3p5', 'circ_lm8_rad6p0', 'circ_lm8_rad10p0']
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
HARLAND_R_EXPON, HARLAND_R_GAUSS = 0.995, 0.985
HARLAND_FRAC_UNDER_1M2 = 0.78

# Area counts as varied only when it spans at least this ratio. Not
# `nunique() > 1`: the corridor is 28.224 m^2 against the discs' 28.274, a
# 0.2% rounding difference that would otherwise be read as an area axis and
# produce a Fig 6E plot out of six points that all share one scale.
AREA_SPAN_MIN = 2.0


def area_varies(s):
    """Does this set of runs span enough area to speak to Harland Fig 6E?"""
    a = s.env_area_m2.dropna()
    return len(a) > 0 and float(a.max()) / float(a.min()) >= AREA_SPAN_MIN


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
    )
    band = bank.scale_band.to_numpy(dtype=int)
    for b in BANDS:
        d[f'band{b}_frac'] = float(np.mean(band == b))
    d['band6plus_frac'] = float(np.mean(band > max(BANDS)))
    d['n_bands_occupied'] = int(len(np.unique(band)))
    return d


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

def _save(fig, fig_dir, name):
    p = os.path.join(fig_dir, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}', flush=True)


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


def fig_cv(summary, fig_dir):
    """S2: coefficient of variation of field size — Harland Fig 6E.

    Plotted against whichever axis the given datasets actually vary. With the
    area sweep that is arena area, and the figure is a direct reading of
    Fig 6E. With the same-area six it can only be cue density and shape, and
    the figure answers the weaker control question instead — is the CV moved
    by anything other than scale? The axis is chosen from the data rather
    than fixed, so the same code serves both and neither is mislabelled.

    Harland's 70/85/101 are drawn as a reference scale. They are a trend to
    compare against only when our own x axis is area.
    """
    s = summary[(summary.extent_pctl == DEFAULT_PCTL) &
                (summary.act_thresh == DEFAULT_T)]
    if not len(s):
        return
    by_area = area_varies(s)
    if by_area:
        panels = [('env_area_m2', 'arena area (m$^2$)', s)]
    else:
        panels = [('n_landmarks', 'landmark count (disc)',
                   s[s.aspect == 1.0]),
                  ('aspect', 'aspect ratio (8 landmarks)',
                   s[s.n_landmarks == 8])]
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(6.4 * len(panels), 4.4))
    for ax, (xcol, xlabel, d) in zip(axes[0], panels):
        for c in sorted(d.channel.unique()):
            g = d[d.channel == c].sort_values(xcol)
            if len(g):
                ax.plot(g[xcol], g.cv_area_pct, 'o-', ms=5, lw=1.2,
                        color=CHANNEL_COLORS.get(c, '0.4'), label=c)
        for k, v in HARLAND_CV.items():
            ax.axhline(v, color='k', ls=':', lw=0.9)
            ax.text(ax.get_xlim()[1], v, f'  {k} {v:g}', fontsize=6,
                    va='center')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('CV of field area (%)')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(
        'S2  coefficient of variation of field size against arena area '
        '(Harland Fig 6E)' if by_area else
        'S2  coefficient of variation of field size. Area is held constant '
        'across these datasets,\nso the dotted Harland Fig 6E values are a '
        'reference scale, not a trend to fit.', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, fig_dir, 'S2_cv.png')


def fig_coverage(summary, fig_dir):
    """S3: per-field arena coverage against Harland's 9-13% band."""
    s = summary[(summary.extent_pctl == DEFAULT_PCTL) &
                (summary.act_thresh == DEFAULT_T)]
    if not len(s):
        return
    fig, ax = plt.subplots(figsize=(max(7.0, 0.42 * len(s)), 4.4))
    xs = np.arange(len(s))
    lo = 100 * (s.coverage_median - s.coverage_q25).to_numpy()
    hi = 100 * (s.coverage_q75 - s.coverage_median).to_numpy()
    ax.errorbar(xs, 100 * s.coverage_median.to_numpy(), yerr=[lo, hi],
                fmt='o', ms=4, lw=1, capsize=2,
                color='#333333', ecolor='0.6')
    ax.axhspan(100 * HARLAND_COVERAGE[0], 100 * HARLAND_COVERAGE[1],
               color='#2ca02c', alpha=0.18, label='Harland 9-13%')
    ax.set_ylim(bottom=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{e}\n{c}' for e, c in zip(s.env, s.channel)],
                       rotation=90, fontsize=5)
    ax.set_ylabel('arena covered per field (%)')
    ax.set_title('S3  fraction of the arena covered by one field '
                 '(median, IQR)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _save(fig, fig_dir, 'S3_coverage.png')


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
        import glob
        return sorted(glob.glob(f'{self.fig_dir}/*.png'))

    def data_files(self):
        return [p for p in (f'{self.out_dir}/summary.csv',
                            f'{self.out_dir}/fits.csv',
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

        # --- can these datasets speak to scale at all? --------------------
        areas = sorted(base.env_area_m2.unique())
        if area_varies(base):
            lo, hi = min(areas), max(areas)
            cv_lo = base[base.env_area_m2 == lo].cv_area_pct.median()
            cv_hi = base[base.env_area_m2 == hi].cv_area_pct.median()
            out.append(S('CV AGAINST AREA — Harland Fig 6E', '\n'.join([
                f'{len(areas)} arena areas, {lo:.1f} to {hi:.0f} m^2 '
                f'({hi/lo:.0f}x). This is the axis Harland vary, so 6E can be '
                f'read directly.', '',
                f'  CV at {lo:.1f} m^2   {cv_lo:.0f}%',
                f'  CV at {hi:.0f} m^2   {cv_hi:.0f}%',
                f'  Harland      ' +
                ', '.join(f'{k} {v:g}' for k, v in HARLAND_CV.items()), '',
                'Their claim is that CV RISES with enclosure area. Read the '
                'direction first and the absolute values second: matching the '
                'direction is the result, and sitting at their exact 70/85/101 '
                'is not expected from a different agent in a different arena.',
                '',
                'One confound is built into the sweep and cannot be removed '
                'from it: the landmarks are a fixed 0.75 m, so a panel '
                'subtends less of the image in a larger arena, and enclosure '
                'size is confounded with cue salience. That is a genuine '
                'property of fixed-size cues rather than a defect — Harland\'s '
                'room cues were fixed too — but at r = 1.25 eight panels cover '
                '76% of the circumference, which is closer to a ring of flags '
                'than to a room with landmarks in it. Treat the endpoints as '
                'the weakest points of the curve.'])))
        else:
            out.append(S('SCALE IS NOT VARIED IN THIS RUN', '\n'.join([
                f'Every dataset here is within a factor '
                f'{max(areas)/min(areas):.2f} of {areas[0]:.1f} m^2, so '
                f'Harland Fig 6E '
                '(CV against enclosure area) and the 3F/3G contrast (a '
                'negative exponential in the megaspace against a Gaussian in '
                'the small environments) CANNOT be read at all. Both are '
                'claims about scale, and scale is held constant.', '',
                'What this run does establish is the shape at one scale, and '
                'whether cue density or arena shape move it — a control, and '
                'a prerequisite for reading the area sweep, but not a test of '
                'either published claim.', '',
                'Run over AREA_ENVS (circ_lm8_rad1p25 .. rad10p0, 4.91 to '
                '314.16 m^2) for the comparison this experiment is named '
                'after.'])))

        cols = ['env', 'channel', 'n_fields', 'env_area_m2', 'cv_area_pct',
                'area_min_m2', 'area_median_m2', 'area_max_m2',
                'area_max_min_ratio', 'coverage_median', 'frac_under_1m2',
                'n_bands_occupied', 'frac_at_floor', 'frac_at_ceiling']
        out.append(S(f'Per environment and channel (EXTENT_PCTL {DEFAULT_PCTL})',
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
                                USE_GPU=not args.no_gpu))
    device = R.pick_device(use_gpu=not args.no_gpu)

    print('=' * 72)
    print('Scale distribution | what shape is our field-size distribution')
    print(f'  envs     : {envs}')
    print(f'  channels : {chans}')
    print(f'  settings : {[(p, t) for p, t in settings]}  (EXTENT_PCTL, ACT_THRESH)')
    print(f'  LAMBDA   : {base_C["LAMBDA"]}')
    print(f'  areas    : {"varies — Fig 6E readable" if len(envs) > 1 else "one"}'
          '  (a single area cannot speak to Harland 3F-G or 6E)')
    print('  note     : EXTENT_PCTL saturates at 65 and the sweep is settled;')
    print('             pass --settings to re-open it. ACT_THRESH cancels')
    print('             under SIGMA_MODE=quantile and is not a knob.')
    print('=' * 72, flush=True)

    banks_all, sum_rows, fit_rows, inv_rows = {}, [], [], []
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
                tag = dict(env=e, channel=c, extent_pctl=p, act_thresh=t)
                sum_rows.append(dict(tag, **describe(bank, env, base_C)))
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
    fits.to_csv(f'{out_dir}/fits.csv', index=False)
    if inv is not None:
        inv.to_csv(f'{out_dir}/threshold_invariance.csv', index=False)

    winners = (fits[(fits.variable == 'area') & (fits.d_aic == 0) &
                    (fits.extent_pctl == DEFAULT_PCTL) &
                    (fits.act_thresh == DEFAULT_T)].form.value_counts())

    print('\nfigures:', flush=True)
    fig_distributions(banks_all, fits, envs, chans, fig_dir)
    fig_cv(summary, fig_dir)
    fig_coverage(summary, fig_dir)

    rep = ScaleDistributionReport(env_name=','.join(envs), out_dir=out_dir,
                                  fig_dir=fig_dir, results=summary,
                                  log_path=os.environ.get('REALM_LOG_PATH'))
    rep.fits, rep.invariance, rep.winners = fits, inv, winners
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
