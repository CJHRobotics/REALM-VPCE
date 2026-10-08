"""Every number the paper reports, recomputed from the analyses' caches.

One row per number in the manuscript: where it appears, what it is, the value
the text gives now, how the new value is computed, and the new value. Going
down the list updates the paper. The tables (funnel by scale, coverage, the
two elongation tables, the validation table) are given cell by cell.

Every number is computed from the cached outputs of the analyses that the
rerun produces -- the prune audit, the scale distribution libraries and their
tables, field geometry, and the extent validation -- and where an experiment
already has a function for a statistic, that function is called, so a number
here is the number its report would give. Nothing is fitted or rebuilt.

A number that cannot be computed (a cache missing, a column renamed) is
written with status ERROR and the reason, and the rest are still produced.

Usage
    python paper_numbers.py                       # -> figures/paper/paper_numbers.csv
    python paper_numbers.py --cache DIR --out FILE
"""
import argparse
import json
import os
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f'{HERE}/../..')
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import rules as R                                   # noqa: E402
import run_scale_distribution as SD                 # noqa: E402

ENVS = ['circ_lm8_r3', 'circ_lm8_r6', 'circ_lm0_r3', 'circ_lm0_r6',
        'corr_lm8_l10w2', 'corr_lm0_l10w2', 'corr_lm8_l10w10',
        'corr_lm0_l10w10']                          # Table 1's order
CHANS = ['hog', 'color', 'spatial', 'lidar', 'visual', 'all']
CIRC = [e for e in ENVS if e.startswith('circ')]
SQUARE = ['corr_lm8_l10w10', 'corr_lm0_l10w10']
CORR = ['corr_lm8_l10w2', 'corr_lm0_l10w2']
NAME = SD.CHANNEL_LABEL                             # spatial -> downscaled
ALPHA = 0.05


# ------------------------------------------------------------------ format

def pct(x, d=1):
    return f'{x:.{d}f}%'


def num(x, d=0):
    if isinstance(x, (int, np.integer)) or d == 0:
        return f'{int(round(float(x))):,}'
    return f'{x:.{d}f}'


def rng_(lo, hi, d=1, unit=''):
    return f'{lo:.{d}f}{unit}–{hi:.{d}f}{unit}'


def star(rho, q):
    return f'{rho:+.2f}'.replace('+', '') + ('*' if not q < ALPHA else '')


# ------------------------------------------------------------------- data

class Data:
    """The analyses' outputs, each loaded the first time a number asks."""

    def __init__(self, cache):
        self.cache = cache
        self._memo = {}

    def _get(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    # prune audit -----------------------------------------------------------
    @property
    def pairs(self):
        return self._get('pairs', lambda: self._csv('prune_audit/prune_audit_pairs.csv'))

    @property
    def pscales(self):
        def load():
            d = self._csv('prune_audit/prune_audit_scales.csv')
            d['scale'] = d.scale.astype(str)
            return d
        return self._get('pscales', load)

    # scale distribution ----------------------------------------------------
    @property
    def banks(self):
        def load():
            out_dir = f'{self.cache}/scale_distribution'
            key = (SD.DEFAULT_PCTL, SD.DEFAULT_T, SD.PRIMARY_IOU)
            banks, lost = {}, []
            for e in ENVS:
                for c in CHANS:
                    p = SD.bank_path(out_dir, e, c, *key, SD.DEFAULT_TILING)
                    if os.path.exists(p):
                        banks[(e, c)] = pd.read_csv(p)
                    else:
                        lost.append(os.path.relpath(p, self.cache))
            if lost:
                raise RuntimeError(f'{len(lost)} libraries missing, e.g. {lost[0]}')
            return banks
        return self._get('banks', load)

    def env(self, e):
        def load():
            root = ET.parse(f'{REPO}/simulation/worlds/environments/vpce/'
                            f'{e}.xml').getroot()
            env = R.build_env(None, root)
            env['landmarks'] = [(float(l.get('x')), float(l.get('y')))
                                for l in root.findall('landmark')]
            return env
        return self._get(('env', e), load)

    @property
    def desc(self):
        """SD.describe for every library: the scale run's own summary."""
        def load():
            rows = []
            for (e, c), b in self.banks.items():
                d = SD.describe(b, self.env(e), R.DEFAULT_CFG)
                rows.append(dict(d, env=e, channel=c))
            return pd.DataFrame(rows)
        return self._get('desc', load)

    @property
    def redund(self):
        def load():
            d = self._csv('scale_distribution/redundancy.csv')
            m = (d.extent_pctl == SD.DEFAULT_PCTL) & np.isclose(d.act_thresh, SD.DEFAULT_T)
            if 'split_half_iou_min' in d:
                m &= d.split_half_iou_min.isna()
            return d[m]
        return self._get('redund', load)

    @property
    def fits(self):
        def load():
            d = self._csv('scale_distribution/fits.csv')
            m = (d.extent_pctl == SD.DEFAULT_PCTL) & np.isclose(d.act_thresh, SD.DEFAULT_T)
            return d[m]
        return self._get('fits', load)

    # field geometry --------------------------------------------------------
    @property
    def corr(self):
        def load():
            d = self._csv('field_geometry/correlations.csv')
            return d[d.subset == 'all fields']
        return self._get('corr', load)

    @property
    def fields(self):
        return self._get('fields', lambda: self._csv('field_geometry/fields.csv'))

    # extent validation -----------------------------------------------------
    @property
    def ev(self):
        def load():
            import run_extent_validation as EV
            cache_dir = f'{self.cache}/extent_validation'
            data = EV.load_cache(cache_dir, None)
            if not data['envs']:
                raise RuntimeError(f'no extent-validation results in {cache_dir}')
            units = {m.get('field_unit', 'cluster') for m in data['meta'].values()}
            if units != {R.DEFAULT_CFG['FIELD_UNIT']}:
                raise RuntimeError(f'extent-validation cache built with field '
                                   f'unit {units}, not '
                                   f'{R.DEFAULT_CFG["FIELD_UNIT"]!r}; rerun it')
            rec, ctl = data['rec'], data['ctl']
            q = sorted(rec.q.unique())
            cur = EV.pooled_curves(rec, ctl, q, EV.N_BOOT, 0)
            out = dict(EV=EV, data=data, q=np.asarray(q, float), cur=cur,
                       crit=EV.criteria(cur), cells=EV.cell_optima(rec),
                       lev_scale=EV.level_optima(rec, 'scale', q, EV.N_BOOT, 0)[0],
                       lev_cont=EV.level_optima(rec, 'contour', q, EV.N_BOOT, 0)[0],
                       down=EV.downstream_table(data['pipe']))
            return out
        return self._get('ev', load)

    def _csv(self, rel):
        p = f'{self.cache}/{rel}'
        if not os.path.exists(p):
            raise RuntimeError(f'missing {p}')
        return pd.read_csv(p)


# ---------------------------------------------------------------- entries

ENTRIES = []


def entry(id, where, what, old, how):
    def wrap(fn):
        ENTRIES.append(dict(id=id, where=where, what=what, old=old, how=how,
                            fn=fn))
        return fn
    return wrap


def table(prefix, where, cells):
    """Register one table cell per (id, what, old, how, fn) tuple."""
    for cid, what, old, how, fn in cells:
        ENTRIES.append(dict(id=f'{prefix}.{cid}', where=where, what=what,
                            old=old, how=how, fn=fn))


# Helpers over the prune audit ---------------------------------------------

def _p(d):
    return d.pairs


def share(d, num_col, den_col='n_candidates', rows=None):
    p = _p(d) if rows is None else rows
    return 100.0 * p[num_col].sum() / p[den_col].sum()


def env_rows(d, envs):
    return d.pairs[d.pairs.env.isin(envs)]


def selected_share(rows):
    return 100.0 * rows.n_admitted.sum() / rows.n_candidates.sum()


# ======================================================== 5.1 field selection

W_SEL = '05-results.tex, §5.1 Field selection'


@entry('sel.cand_per_set', W_SEL, 'Candidate fields per set: min, max, median',
       '2492–2773 (median 2556)', 'prune_audit_pairs.n_candidates over the 48 sets')
def _(d):
    n = d.pairs.n_candidates
    return f'{n.min():,}–{n.max():,} (median {int(np.median(n)):,})'


@entry('sel.clusters_per_set', W_SEL,
       'NEW: clusters per set, which the candidate fields now split into',
       '(not in the paper)', 'prune_audit_pairs.n_clusters')
def _(d):
    n = d.pairs.n_clusters
    return f'{n.min():,}–{n.max():,} (median {int(np.median(n)):,}); total {n.sum():,}'


@entry('sel.cand_total', W_SEL, 'Candidate fields evaluated, all 48 sets',
       '122,800', 'sum of n_candidates')
def _(d):
    return num(d.pairs.n_candidates.sum())


@entry('sel.selected_total', W_SEL, 'Fields selected, all 48 sets', '28,402',
       'sum of n_admitted (prune audit)')
def _(d):
    return num(d.pairs.n_admitted.sum())


@entry('sel.pct_size', W_SEL, '% of candidate fields rejected by size', '63.3%',
       '(n_candidates - pass_size) / n_candidates, pooled')
def _(d):
    p = d.pairs
    return pct(100.0 * (p.n_candidates - p.pass_size).sum() / p.n_candidates.sum())


@entry('sel.pct_competition', W_SEL, '% of candidate fields rejected by competition',
       '12.9%', '(pass_contiguity - pass_competition) / n_candidates, pooled')
def _(d):
    p = d.pairs
    return pct(100.0 * (p.pass_contiguity - p.pass_competition).sum() / p.n_candidates.sum())


@entry('sel.pct_selected', W_SEL, '% of candidate fields selected', '23.1%',
       'n_admitted / n_candidates, pooled')
def _(d):
    return pct(share(d, 'n_admitted'))


@entry('sel.pct_multifield', W_SEL,
       '% of candidate fields selected as subfields of multifield units '
       '(was: % with multifield activity, excluded)', '0.7%',
       'n_admitted_multifield / n_candidates, pooled')
def _(d):
    return pct(share(d, 'n_admitted_multifield'))


@entry('sel.competition_of_reaching', W_SEL,
       '% of candidate fields reaching competition that it rejected', '35.8%',
       '(pass_contiguity - pass_competition) / pass_contiguity')
def _(d):
    p = d.pairs
    return pct(100.0 * (p.pass_contiguity - p.pass_competition).sum() / p.pass_contiguity.sum())


@entry('size.below_floor', W_SEL + ', ¶Size', '% of candidate fields below the lower bound',
       '63.1%', "prune_audit_scales rows '< floor'")
def _(d):
    s = d.pscales
    return pct(100.0 * s[s.scale == '< floor'].n_candidates.sum() / d.pairs.n_candidates.sum())


@entry('size.above_ceiling', W_SEL + ', ¶Size',
       '% and number of candidate fields above the upper bound', '0.2% (266)',
       "prune_audit_scales rows '> ceiling'")
def _(d):
    s = d.pscales
    n = int(s[s.scale == '> ceiling'].n_candidates.sum())
    return f'{pct(100.0 * n / d.pairs.n_candidates.sum())} ({n:,})'


@entry('size.median_cand_radius_small_circ', W_SEL + ', ¶Size',
       'Median candidate equivalent radius in the small circular enclosure, '
       'range across feature spaces', '0.090–0.094 m',
       'prune_audit_pairs.cand_radius_median_m, circ_lm8_r3 (circ_lm0_r3 in brackets)')
def _(d):
    def r(e):
        v = d.pairs[d.pairs.env == e].cand_radius_median_m
        return f'{v.min():.3f}–{v.max():.3f} m'
    return f'{r("circ_lm8_r3")} ({r("circ_lm0_r3")})'


@entry('size.lower_bound_small_circ', W_SEL + ', ¶Size',
       'Lower bound as a radius, small circular enclosure (unchanged by the rule)',
       '0.105 m', 'prune_audit_pairs.floor_radius_m')
def _(d):
    return f'{d.pairs[d.pairs.env == "circ_lm8_r3"].floor_radius_m.iloc[0]:.3f} m'


@entry('comp.count', W_SEL + ', ¶Same-scale competition',
       'Candidate fields rejected by competition', '15,809',
       'sum of pass_contiguity - pass_competition')
def _(d):
    p = d.pairs
    return num((p.pass_contiguity - p.pass_competition).sum())


@entry('comp.rate_scales_0_4', W_SEL + ', ¶Same-scale competition',
       'Competition rate among fields reaching it, range over scales 0–4',
       '35.4%–38.7%', 'per scale, pooled: 1 - n_pass_competition / n_candidates')
def _(d):
    s = d.pscales[d.pscales.scale.isin([str(i) for i in range(5)])]
    g = s.groupby('scale')[['n_candidates', 'n_pass_competition']].sum()
    r = 100.0 * (1 - g.n_pass_competition / g.n_candidates)
    return rng_(r.min(), r.max(), 1, '%')


@entry('comp.rate_scale_5', W_SEL + ', ¶Same-scale competition',
       'Competition rate at the coarsest scale', '14.1%', 'as above, scale 5')
def _(d):
    s = d.pscales[d.pscales.scale == '5']
    return pct(100.0 * (1 - s.n_pass_competition.sum() / s.n_candidates.sum()))


@entry('comp.cand_ratio_fine_coarse', W_SEL + ', ¶Same-scale competition',
       'Candidate fields at the finest scale over those at the coarsest', 'about 170',
       'sum n_candidates at scale 0 / at scale 5')
def _(d):
    s = d.pscales
    a = s[s.scale == '0'].n_candidates.sum()
    b = s[s.scale == '5'].n_candidates.sum()
    return f'{a / b:.0f}' if b else 'no candidates at scale 5'


def _funnel_scale_cells():
    old = [["0.18", "27187", "35.4%", "63.9%", "1.0%"], ["0.29", "10782", "36.0%", "62.9%", "1.8%"],
           ["0.47", "4403", "36.8%", "60.3%", "4.6%"], ["0.74", "1822", "38.7%", "57.5%", "6.3%"],
           ["1.19", "709", "37.8%", "57.1%", "8.2%"], ["1.28", "156", "14.1%", "82.1%", "4.5%"]]
    cells = []
    for sc in range(6):
        def g(d, sc=sc):
            s = d.pscales[d.pscales.scale == str(sc)]
            return s
        cells += [
            (f'scale{sc}.median_radius', f'Scale {sc}: median radius (m)', old[sc][0],
             'median over the 48 sets of the scale\'s median candidate radius',
             lambda d, g=g: f'{g(d).median_radius_m.median():.2f}'),
            (f'scale{sc}.candidates', f'Scale {sc}: candidate fields passing size', old[sc][1],
             'sum of n_candidates', lambda d, g=g: num(g(d).n_candidates.sum())),
            (f'scale{sc}.competition', f'Scale {sc}: % rejected by competition', old[sc][2],
             '1 - n_pass_competition / n_candidates',
             lambda d, g=g: pct(100.0 * (1 - g(d).n_pass_competition.sum() / g(d).n_candidates.sum()))),
            (f'scale{sc}.selected', f'Scale {sc}: % selected', old[sc][3],
             'n_admitted / n_candidates',
             lambda d, g=g: pct(100.0 * g(d).n_admitted.sum() / g(d).n_candidates.sum())),
            (f'scale{sc}.multifield', f'Scale {sc}: % of selected fields in multifield units',
             old[sc][4], 'n_admitted_multifield / n_admitted (the caption\'s definition)',
             lambda d, g=g: pct(100.0 * g(d).n_admitted_multifield.sum() / max(g(d).n_admitted.sum(), 1))),
        ]
    return cells


table('tab_funnel_scale', W_SEL + ', Table funnel-scale', _funnel_scale_cells())


W_DIFF = W_SEL + ', ¶Differences between environments and feature spaces'


@entry('diff.selected_circ_square_lm8', W_DIFF,
       '% selected, range over the four circular enclosures and the square with landmarks',
       '25–27%', 'n_admitted / n_candidates per environment, feature spaces pooled')
def _(d):
    v = [selected_share(env_rows(d, [e])) for e in CIRC + ['corr_lm8_l10w10']]
    return rng_(min(v), max(v), 1, '%')


@entry('diff.selected_square_lm0', W_DIFF, '% selected, square without landmarks',
       '21.4%', 'as above')
def _(d):
    return pct(selected_share(env_rows(d, ['corr_lm0_l10w10'])))


@entry('diff.selected_corridor', W_DIFF,
       '% selected, corridor with and without landmarks', '20.8% and 13.5%', 'as above')
def _(d):
    return ' and '.join(pct(selected_share(env_rows(d, [e]))) for e in CORR)


@entry('diff.corridor_size', W_DIFF, '% removed by size in the corridors (range)',
       '68–69%', '(n_candidates - pass_size) / n_candidates per corridor')
def _(d):
    v = [100.0 * (r.n_candidates.sum() - r.pass_size.sum()) / r.n_candidates.sum()
         for r in (env_rows(d, [e]) for e in CORR)]
    return rng_(min(v), max(v), 1, '%')


@entry('diff.corridor_lm0_competition', W_DIFF,
       'Competition rate among fields reaching it, corridor without landmarks',
       '54.3%', '(pass_contiguity - pass_competition) / pass_contiguity')
def _(d):
    r = env_rows(d, ['corr_lm0_l10w2'])
    return pct(100.0 * (r.pass_contiguity - r.pass_competition).sum() / r.pass_contiguity.sum())


@entry('diff.selected_by_feature_space', W_DIFF,
       '% selected per feature space, environments pooled (paper: downscaled 20.7, '
       'lidar 26.0, all 23.4)', 'downscaled 20.7%, lidar 26.0%, all 23.4%',
       'n_admitted / n_candidates per feature space')
def _(d):
    p = d.pairs
    v = {c: 100.0 * p[p.channel == c].n_admitted.sum() / p[p.channel == c].n_candidates.sum()
         for c in CHANS}
    return ', '.join(f'{NAME[c]} {v[c]:.1f}%' for c in CHANS)


@entry('diff.selected_set_range', W_DIFF, '% selected, lowest and highest set',
       '6.0% (downscaled, corr_lm0_l10w2) to 28.3% (lidar, corr_lm0_l10w10)',
       'n_admitted / n_candidates per set')
def _(d):
    p = d.pairs.assign(s=100.0 * d.pairs.n_admitted / d.pairs.n_candidates)
    lo, hi = p.loc[p.s.idxmin()], p.loc[p.s.idxmax()]
    return (f'{lo.s:.1f}% ({NAME[lo.channel]}, {lo.env}) to '
            f'{hi.s:.1f}% ({NAME[hi.channel]}, {hi.env})')


# ===================================================== 5.2 field sizes

W_SIZE = '05-results.tex, §5.2 The distribution of field sizes'


@entry('scale.total', W_SIZE, 'Selected fields in the 48 sets (scale run)', '28,413',
       'rows over the 48 libraries')
def _(d):
    return num(sum(len(b) for b in d.banks.values()))


@entry('scale.per_set', W_SIZE, 'Fields per set: min, max, median', '172–714 (median 641)',
       'library sizes')
def _(d):
    n = d.desc.n_fields
    return f'{n.min()}–{n.max()} (median {int(np.median(n))})'


@entry('scale.sets_missing_a_scale', W_SIZE, 'Sets missing a scale, and which',
       'three sets in the corridor, each missing the coarsest scale',
       'n_scales_occupied < 6')
def _(d):
    m = d.desc[d.desc.n_scales_occupied < 6]
    if not len(m):
        return 'none'
    out = []
    for r in m.itertuples():
        have = set(d.banks[(r.env, r.channel)].scale_band.unique())
        out.append(f'{r.env}/{NAME[r.channel]} (missing {sorted(set(range(6)) - have)})')
    return f'{len(m)}: ' + '; '.join(out)


@entry('scale.shares', W_SIZE + ', ¶Shape', '% of fields at scales 0–5, pooled',
       '61.1, 24.2, 9.3, 3.6, 1.4, 0.4', 'all fields of the 48 libraries')
def _(d):
    s = pd.concat([b.scale_band for b in d.banks.values()])
    return ', '.join(f'{100.0 * (s == k).mean():.1f}' for k in range(6))


@entry('scale.ratio_between_scales', W_SIZE + ', ¶Shape',
       'Fields at each scale relative to the scale below (geometric mean, 0–5)',
       'roughly 0.4', 'geometric mean of n(s+1)/n(s)')
def _(d):
    s = pd.concat([b.scale_band for b in d.banks.values()])
    n = np.array([(s == k).sum() for k in range(6)], float)
    r = n[1:] / n[:-1]
    r = r[np.isfinite(r) & (r > 0)]
    return f'{np.exp(np.log(r).mean()):.2f}'


@entry('scale.median_over_floor', W_SIZE + ', ¶Shape',
       'Median field area over the lower bound: median of sets (range)',
       '2.2 (1.92–2.29)', 'describe().area_median_over_floor per set')
def _(d):
    v = d.desc.area_median_over_floor
    return f'{v.median():.1f} ({v.min():.2f}–{v.max():.2f})'


@entry('scale.within_2x_floor', W_SIZE + ', ¶Shape', '% of fields under twice the bound',
       '44%', 'pooled over all fields')
def _(d):
    n = (d.desc.frac_within_2x_floor * d.desc.n_fields).sum() / d.desc.n_fields.sum()
    return pct(100.0 * n, 0)


def _lognormal(d):
    f = d.fits
    out = []
    for (e, c, v), g in f.groupby(['env', 'channel', 'variable']):
        best = g.loc[g.aic.idxmin()]
        ln = g[g.form == 'lognormal']
        row = dict(env=e, channel=c, variable=v, converged=len(ln) > 0,
                   preferred=best.form == 'lognormal')
        if len(ln):
            s_, scale = json.loads(ln.params.iloc[0])[:2]
            row['mode'] = scale * np.exp(-s_ ** 2)
            row['lo'] = ln.trunc_lo.iloc[0]
            row['ks_p'] = ln.ks_p_boot.iloc[0]
        out.append(row)
    return pd.DataFrame(out)


@entry('fits.converged', W_SIZE + ', ¶Shape (fits)',
       'Log-normal fits that converged, of 96 (area and diameter)', '93 of 96',
       'fits.csv: a log-normal row per library and variable')
def _(d):
    t = _lognormal(d)
    return f'{int(t.converged.sum())} of {48 * 2}'


@entry('fits.preferred', W_SIZE + ', ¶Shape (fits)',
       'Converged log-normal fits that AIC preferred', 'every one',
       'lowest AIC among the three forms')
def _(d):
    t = _lognormal(d)
    c = t[t.converged]
    return f'{int(c.preferred.sum())} of {len(c)}'


@entry('fits.peak_below_floor', W_SIZE + ', ¶Shape (fits)',
       'Log-normal fits whose peak lies below the lower bound', 'every case',
       'mode = scale * exp(-s^2) < trunc_lo')
def _(d):
    t = _lognormal(d)
    c = t[t.converged]
    return f'{int((c["mode"] < c.lo).sum())} of {len(c)}'


@entry('fits.ks_rejected', W_SIZE + ', ¶Shape (fits)',
       'Sets in which the KS bootstrap rejected the log-normal (area)', 'all 48',
       'ks_p_boot < 0.05, variable area')
def _(d):
    t = _lognormal(d)
    a = t[(t.variable == 'area') & t.converged]
    return f'{int((a.ks_p < 0.05).sum())} of {len(a)}'


W_SPREAD = W_SIZE + ', ¶Spread of field sizes'


@entry('spread.cv', W_SPREAD, 'CV of field area: median across sets (IQR)',
       '232% (223–238%)', 'describe().cv_area_pct')
def _(d):
    v = d.desc.cv_area_pct
    return f'{v.median():.0f}% ({v.quantile(.25):.0f}–{v.quantile(.75):.0f}%)'


@entry('spread.max_min', W_SPREAD,
       'Largest over smallest field in a set, typical (median): area (radius)',
       '149 (12.2)', 'describe().area_max_min_ratio, radius_max_min_ratio')
def _(d):
    return f'{d.desc.area_max_min_ratio.median():.0f} ({d.desc.radius_max_min_ratio.median():.1f})'


@entry('spread.median_cover', W_SPREAD, 'Median field as % of environment area',
       '0.27%', 'median over sets of describe().coverage_median')
def _(d):
    return pct(100.0 * d.desc.coverage_median.median(), 2)


@entry('spread.four_finest', W_SPREAD, '% of fields in the four finest scales', '98%',
       'pooled')
def _(d):
    s = pd.concat([b.scale_band for b in d.banks.values()])
    return pct(100.0 * (s <= 3).mean(), 0)


def _desc_where(d, envs):
    return d.desc[d.desc.env.isin(envs)]


@entry('spread.cv_by_fs_outside_corr', W_SPREAD,
       'Median CV per feature space, circular and square enclosures: lowest and highest',
       '228% (all) to 244% (color)', 'median over those environments per feature space')
def _(d):
    v = _desc_where(d, CIRC + SQUARE).groupby('channel').cv_area_pct.median()
    return f'{v.min():.0f}% ({NAME[v.idxmin()]}) to {v.max():.0f}% ({NAME[v.idxmax()]})'


@entry('spread.ratio_by_fs_outside_corr', W_SPREAD,
       'Median largest/smallest ratio per feature space, circular and square: lowest and highest',
       '144 (color) to 156 (downscaled)', 'as above, area_max_min_ratio')
def _(d):
    v = _desc_where(d, CIRC + SQUARE).groupby('channel').area_max_min_ratio.median()
    return f'{v.min():.0f} ({NAME[v.idxmin()]}) to {v.max():.0f} ({NAME[v.idxmax()]})'


@entry('spread.all_six_scales_outside_corr', W_SPREAD,
       'Every feature space occupied all six scales in circular and square enclosures',
       'yes', 'n_scales_occupied == 6 in every such set')
def _(d):
    t = _desc_where(d, CIRC + SQUARE)
    bad = t[t.n_scales_occupied < 6]
    return 'yes' if not len(bad) else 'NO: ' + ', '.join(f'{r.env}/{NAME[r.channel]}' for r in bad.itertuples())


@entry('spread.fields_by_fs', W_SPREAD,
       'Median fields per set per feature space: most and fewest',
       'lidar 679, downscaled 612', 'median n_fields over environments')
def _(d):
    v = d.desc.groupby('channel').n_fields.median()
    return f'{NAME[v.idxmax()]} {v.max():.0f}, {NAME[v.idxmin()]} {v.min():.0f}'


@entry('spread.corridor_lm8_narrow', W_SPREAD,
       'corr_lm8_l10w2: CV and largest/smallest ratio for downscaled and visual; '
       'whether they lack the coarsest scale', '156% and 165%; 74 and 96; lack it',
       'describe() for those two sets')
def _(d):
    t = d.desc[(d.desc.env == 'corr_lm8_l10w2')].set_index('channel')
    out = []
    for c in ('spatial', 'visual'):
        r = t.loc[c]
        has5 = (d.banks[('corr_lm8_l10w2', c)].scale_band == 5).any()
        out.append(f'{NAME[c]}: CV {r.cv_area_pct:.0f}%, ratio {r.area_max_min_ratio:.0f}, '
                   f'{"has" if has5 else "lacks"} scale 5')
    return '; '.join(out)


@entry('spread.corridor_lm8_lidar', W_SPREAD, 'corr_lm8_l10w2, lidar: CV and ratio',
       '237%; 155', 'describe()')
def _(d):
    r = d.desc[(d.desc.env == 'corr_lm8_l10w2') & (d.desc.channel == 'lidar')].iloc[0]
    return f'{r.cv_area_pct:.0f}%; {r.area_max_min_ratio:.0f}'


W_ENV = W_SIZE + ', ¶Environment size and landmarks'


def _med_area(d, e):
    return float(np.median(pd.concat([d.banks[(e, c)].area_env_m2 for c in CHANS])))


@entry('env.median_area_small_large', W_ENV,
       'Median field area, small -> large circular enclosure, with and without landmarks (m²)',
       '0.075 -> 0.307 (with); 0.076 -> 0.302 (without)',
       'median over all fields of the environment, feature spaces pooled')
def _(d):
    return (f'{_med_area(d, "circ_lm8_r3"):.3f} -> {_med_area(d, "circ_lm8_r6"):.3f} (with); '
            f'{_med_area(d, "circ_lm0_r3"):.3f} -> {_med_area(d, "circ_lm0_r6"):.3f} (without)')


def _env_med(d, e, col):
    return float(_desc_where(d, [e])[col].median())


@entry('env.median_over_floor', W_ENV,
       'Median field over the lower bound: small and large enclosure, with; without',
       '2.14 and 2.19 with; 2.18 and 2.16 without',
       'median over the environment\'s six sets of area_median_over_floor')
def _(d):
    f = lambda e: _env_med(d, e, 'area_median_over_floor')
    return (f'{f("circ_lm8_r3"):.2f} and {f("circ_lm8_r6"):.2f} with; '
            f'{f("circ_lm0_r3"):.2f} and {f("circ_lm0_r6"):.2f} without')


@entry('env.median_cover', W_ENV, 'Median field as % of each circular enclosure', '0.27%',
       'coverage_median, median over each enclosure\'s sets (range)')
def _(d):
    v = [100.0 * _env_med(d, e, 'coverage_median') for e in CIRC]
    return rng_(min(v), max(v), 2, '%')


@entry('env.cv', W_ENV, 'CV: small and large enclosure, with; without',
       '227% and 237% with; 231% and 233% without', 'median over sets of cv_area_pct')
def _(d):
    f = lambda e: _env_med(d, e, 'cv_area_pct')
    return (f'{f("circ_lm8_r3"):.0f}% and {f("circ_lm8_r6"):.0f}% with; '
            f'{f("circ_lm0_r3"):.0f}% and {f("circ_lm0_r6"):.0f}% without')


@entry('env.max_min', W_ENV, 'Largest/smallest ratio: small -> large, with; without',
       '150 -> 156 with; 144 -> 157 without', 'median over sets of area_max_min_ratio')
def _(d):
    f = lambda e: _env_med(d, e, 'area_max_min_ratio')
    return (f'{f("circ_lm8_r3"):.0f} -> {f("circ_lm8_r6"):.0f} with; '
            f'{f("circ_lm0_r3"):.0f} -> {f("circ_lm0_r6"):.0f} without')


def _n(d, e, c):
    return len(d.banks[(e, c)])


@entry('env.lidar_landmark_diff_circ_square', W_ENV,
       'lidar: largest difference in field count with vs without landmarks, '
       'circular and square (fields, %)', 'at most 18 fields (2.7%)',
       '|n(lm8) - n(lm0)| per geometry; % of the no-landmark set')
def _(d):
    pairs = [('circ_lm8_r3', 'circ_lm0_r3'), ('circ_lm8_r6', 'circ_lm0_r6'),
             ('corr_lm8_l10w10', 'corr_lm0_l10w10')]
    v = [(abs(_n(d, a, 'lidar') - _n(d, b, 'lidar')),
          100.0 * abs(_n(d, a, 'lidar') - _n(d, b, 'lidar')) / _n(d, b, 'lidar')) for a, b in pairs]
    k = max(v)
    return f'at most {k[0]} fields ({k[1]:.1f}%)'


@entry('env.lidar_landmark_diff_corridor', W_ENV,
       'lidar: difference with vs without landmarks in the corridor', '60 fields (11%)',
       'as above')
def _(d):
    a, b = _n(d, 'corr_lm8_l10w2', 'lidar'), _n(d, 'corr_lm0_l10w2', 'lidar')
    return f'{abs(a - b)} fields ({100.0 * abs(a - b) / b:.0f}%)'


def _lm_factor(d, a, b):
    return {c: _n(d, a, c) / _n(d, b, c) for c in ('hog', 'color', 'spatial', 'visual', 'all')}


@entry('env.landmark_factor_corridor', W_ENV,
       'Landmark factor on field count, visual feature spaces, corridor: lowest and highest',
       '1.2 (color) to 2.8 (downscaled)', 'n(lm8) / n(lm0)')
def _(d):
    f = _lm_factor(d, 'corr_lm8_l10w2', 'corr_lm0_l10w2')
    lo, hi = min(f, key=f.get), max(f, key=f.get)
    return f'{f[lo]:.1f} ({NAME[lo]}) to {f[hi]:.1f} ({NAME[hi]})'


@entry('env.landmark_factor_square', W_ENV,
       'Landmark factor, visual feature spaces, square: range', '1.1 to 1.3', 'as above')
def _(d):
    f = _lm_factor(d, 'corr_lm8_l10w10', 'corr_lm0_l10w10')
    return f'{min(f.values()):.1f} to {max(f.values()):.1f}'


@entry('env.landmark_factor_circ', W_ENV,
       'Landmark factor, visual feature spaces, circular enclosures: range', '0.93 to 1.01',
       'as above, both sizes')
def _(d):
    f = list(_lm_factor(d, 'circ_lm8_r3', 'circ_lm0_r3').values()) + \
        list(_lm_factor(d, 'circ_lm8_r6', 'circ_lm0_r6').values())
    return f'{min(f):.2f} to {max(f):.2f}'


@entry('env.median_over_floor_range', W_ENV,
       'Median field over the bound, range over every environment', '2.1–2.2 times',
       'median per environment of area_median_over_floor')
def _(d):
    v = [_env_med(d, e, 'area_median_over_floor') for e in ENVS]
    return rng_(min(v), max(v), 1)


# ===================================================== 5.3 coverage

W_COV = '05-results.tex, §5.3 Coverage of the environment'


def _coverage_cells():
    old = {"circ_lm8_r3": ["5.1", "1.13", "5.5", "1.18", "5.2", "1.15", "5.3", "1.12", "5.6", "1.16", "5.0", "1.12"],
           "circ_lm8_r6": ["5.5", "1.27", "5.1", "1.18", "5.3", "1.21", "5.6", "1.14", "5.5", "1.18", "5.7", "1.18"],
           "circ_lm0_r3": ["5.9", "1.20", "5.6", "1.16", "5.0", "1.20", "5.0", "1.13", "5.4", "1.14", "5.6", "1.17"],
           "circ_lm0_r6": ["5.4", "1.17", "5.1", "1.19", "5.0", "1.17", "5.4", "1.15", "6.1", "1.25", "5.9", "1.20"],
           "corr_lm8_l10w2": ["2.9", "1.12", "4.5", "1.16", "2.7", "1.14", "5.5", "1.24", "3.7", "1.13", "4.5", "1.15"],
           "corr_lm0_l10w2": ["2.9", "1.10", "3.7", "1.14", "2.0", "1.02", "5.1", "1.17", "2.7", "1.09", "3.4", "1.08"],
           "corr_lm8_l10w10": ["5.0", "1.22", "5.3", "1.17", "4.6", "1.15", "5.4", "1.16", "6.0", "1.22", "5.6", "1.19"],
           "corr_lm0_l10w10": ["4.4", "1.11", "4.0", "1.11", "3.8", "1.05", "5.7", "1.16", "4.3", "1.11", "5.0", "1.14"]}
    cells = []
    for e in ENVS:
        for j, c in enumerate(CHANS):
            def asr(d, e=e, c=c):
                r = d.redund[(d.redund.env == e) & (d.redund.channel == c)]
                return f'{r.redundancy.iloc[0]:.1f}'
            def wsr(d, e=e, c=c):
                r = d.redund[(d.redund.env == e) & (d.redund.channel == c)]
                return f'{r.redundancy_within_scale.iloc[0]:.2f}'
            cells += [(f'{e}.{NAME[c]}.ASR', f'{e}, {NAME[c]}: ASR', old[e][2 * j],
                       'redundancy.csv: redundancy', asr),
                      (f'{e}.{NAME[c]}.WSR', f'{e}, {NAME[c]}: WSR', old[e][2 * j + 1],
                       'redundancy.csv: redundancy_within_scale', wsr)]
    return cells


table('tab_coverage', W_COV + ', Table coverage', _coverage_cells())


@entry('cov.asr', W_COV, 'Across-scale redundancy: median (IQR) over the 48 sets',
       '5.1 (4.5–5.5)', 'redundancy')
def _(d):
    v = d.redund.redundancy
    return f'{v.median():.1f} ({v.quantile(.25):.1f}–{v.quantile(.75):.1f})'


@entry('cov.wsr', W_COV, 'Within-scale redundancy: median, and maximum', '1.16; never above 1.27',
       'redundancy_within_scale')
def _(d):
    v = d.redund.redundancy_within_scale
    return f'{v.median():.2f}; max {v.max():.2f}'


@entry('cov.asr_lidar', W_COV, 'ASR for lidar, range over environments', '5.0–5.7', 'as above')
def _(d):
    v = d.redund[d.redund.channel == 'lidar'].redundancy
    return rng_(v.min(), v.max(), 1)


@entry('cov.asr_corridor', W_COV, 'Median ASR in the corridor, against elsewhere',
       '3.6, against 5.4', 'medians over sets')
def _(d):
    r = d.redund
    return (f'{r[r.env.isin(CORR)].redundancy.median():.1f}, against '
            f'{r[~r.env.isin(CORR)].redundancy.median():.1f}')


@entry('cov.asr_corridor_hog_downscaled', W_COV,
       'ASR in the corridor for hog and downscaled, range', '2.0–2.9', 'as above')
def _(d):
    r = d.redund
    v = r[r.env.isin(CORR) & r.channel.isin(['hog', 'spatial'])].redundancy
    return rng_(v.min(), v.max(), 1)


# ===================================================== 5.4 geometry

W_GEO = '05-results.tex, §5.4 Field shape near a boundary'
PAIR_SCALE, PAIR_WALL = 'scale vs elongation', 'wall distance vs elongation'


def _corr(d, grouping, pair, **k):
    c = d.corr[(d.corr.grouping == grouping) & (d.corr.pair == pair)]
    for key, v in k.items():
        c = c[c[key] == v]
    if not len(c):
        raise RuntimeError(f'no correlation for {grouping} {pair} {k}')
    return c.iloc[0]


def _elong_env_cells():
    old = {"circ_lm8_r3": ["3804", "-0.04*", "-0.59"], "circ_lm8_r6": ["4064", "0.03*", "-0.66"],
           "circ_lm0_r3": ["3903", "-0.02*", "-0.59"], "circ_lm0_r6": ["4152", "0.02*", "-0.67"],
           "corr_lm8_l10w2": ["3240", "0.03*", "-0.31"], "corr_lm0_l10w2": ["2087", "0.10", "-0.12"],
           "corr_lm8_l10w10": ["3867", "0.04", "-0.52"], "corr_lm0_l10w10": ["3296", "-0.01*", "-0.46"]}
    cells = []
    for e in ENVS:
        cells += [
            (f'{e}.fields', f'{e}: selected fields', old[e][0], 'n of the env correlation',
             lambda d, e=e: num(_corr(d, 'env', PAIR_SCALE, env=e).n)),
            (f'{e}.scale', f'{e}: rho, elongation with scale (* = q >= 0.05)', old[e][1],
             "correlations.csv grouping 'env'",
             lambda d, e=e: star(*_corr(d, 'env', PAIR_SCALE, env=e)[['rho', 'q']])),
            (f'{e}.wall', f'{e}: rho, elongation with wall distance', old[e][2],
             "correlations.csv grouping 'env'",
             lambda d, e=e: star(*_corr(d, 'env', PAIR_WALL, env=e)[['rho', 'q']])),
        ]
        for sc in range(5):
            cells.append((f'{e}.wall_scale{sc}',
                          f'{e}: rho with wall distance at scale {sc}', 'tbd',
                          "correlations.csv grouping 'env x scale'",
                          lambda d, e=e, sc=sc: star(*_corr(d, 'env x scale', PAIR_WALL,
                                                            env=e, scale=sc)[['rho', 'q']])))
    cells += [
        ('all.fields', 'All environments: fields', '28413', "grouping 'pooled'",
         lambda d: num(_corr(d, 'pooled', PAIR_SCALE).n)),
        ('all.scale', 'All environments: rho with scale', '0.01*', "grouping 'pooled'",
         lambda d: star(*_corr(d, 'pooled', PAIR_SCALE)[['rho', 'q']])),
        ('all.wall', 'All environments: rho with wall distance', '-0.38', "grouping 'pooled'",
         lambda d: star(*_corr(d, 'pooled', PAIR_WALL)[['rho', 'q']])),
    ]
    return cells


table('tab_elong_env', W_GEO + ', Table elong-env', _elong_env_cells())


def _elong_channel_cells():
    old = {"circ_lm8_r3": ["-0.60", "-0.37", "-0.74", "-0.96", "-0.61", "-0.62"],
           "circ_lm8_r6": ["-0.83", "-0.38", "-0.78", "-0.97", "-0.75", "-0.78"],
           "circ_lm0_r3": ["-0.39", "-0.74", "-0.96", "-0.97", "-0.74", "-0.76"],
           "circ_lm0_r6": ["-0.75", "-0.73", "-0.93", "-0.97", "-0.87", "-0.91"],
           "corr_lm8_l10w2": ["-0.35", "-0.46", "-0.41", "-0.34", "-0.37", "-0.31"],
           "corr_lm0_l10w2": ["-0.14", "-0.07*", "-0.17", "-0.32", "-0.04*", "-0.15"],
           "corr_lm8_l10w10": ["-0.64", "-0.17", "-0.59", "-0.86", "-0.53", "-0.56"],
           "corr_lm0_l10w10": ["-0.59", "-0.40", "-0.60", "-0.85", "-0.55", "-0.63"]}
    return [(f'{e}.{NAME[c]}', f'{e}, {NAME[c]}: rho with wall distance', old[e][j],
             "correlations.csv grouping 'env x channel'",
             lambda d, e=e, c=c: star(*_corr(d, 'env x channel', PAIR_WALL,
                                             env=e, channel=c)[['rho', 'q']]))
            for e in ENVS for j, c in enumerate(CHANS)]


table('tab_elong_channel', W_GEO + ', Table elong-channel', _elong_channel_cells())


def _med_elong_by_scale(d, envs):
    f = d.fields[d.fields.env.isin(envs) & (d.fields.scale <= 4)]
    return f.groupby(['env', 'scale']).elongation.median()


@entry('geo.elong_scale_range', W_GEO + ', ¶Elongation and scale',
       'Median elongation within a scale (0–4): circular and square; corridor',
       '1.3–2.3; 2–5.5', 'fields.csv, median per environment and scale')
def _(d):
    a = _med_elong_by_scale(d, CIRC + SQUARE)
    b = _med_elong_by_scale(d, CORR)
    return f'{a.min():.1f}–{a.max():.1f}; {b.min():.1f}–{b.max():.1f}'


@entry('geo.less_elongated_with_landmarks', W_GEO + ', ¶Elongation and scale',
       'Fields less elongated with landmarks than without, in every geometry',
       'yes', 'median elongation per environment, lm8 vs lm0')
def _(d):
    m = d.fields.groupby('env').elongation.median()
    pairs = [('circ_lm8_r3', 'circ_lm0_r3'), ('circ_lm8_r6', 'circ_lm0_r6'),
             ('corr_lm8_l10w10', 'corr_lm0_l10w10'), ('corr_lm8_l10w2', 'corr_lm0_l10w2')]
    parts = [f'{a} {m[a]:.2f} vs {m[b]:.2f}' for a, b in pairs]
    ok = all(m[a] < m[b] for a, b in pairs)
    return ('yes' if ok else 'NOT in every geometry') + ' (' + '; '.join(parts) + ')'


@entry('geo.near_wall', W_GEO + ', ¶Elongation and distance from the wall',
       'Median elongation nearest the wall -> at the center/midline, circular and square',
       'about 1.8–3.5 -> about 1.1–1.2',
       'fields.csv, wall distance < 0.1 and >= 0.9, per environment')
def _(d):
    f = d.fields[d.fields.env.isin(CIRC + SQUARE)]
    near = f[f.wall_dist_norm < 0.1].groupby('env').elongation.median()
    far = f[f.wall_dist_norm >= 0.9].groupby('env').elongation.median()
    return f'{near.min():.1f}–{near.max():.1f} -> {far.min():.1f}–{far.max():.1f}'


@entry('geo.scale_corr_feature_spaces', W_GEO + ', ¶Elongation and scale',
       "Significant scale-elongation correlations per feature space (q < 0.05), and their signs",
       'few; most for color, differing signs', "grouping 'env x channel'")
def _(d):
    c = d.corr[(d.corr.grouping == 'env x channel') & (d.corr.pair == PAIR_SCALE) & (d.corr.q < ALPHA)]
    if not len(c):
        return 'none'
    g = c.groupby('channel').rho.agg(['size', 'min', 'max'])
    return '; '.join(f'{NAME[ch]} {int(r["size"])} ({r["min"]:+.2f} to {r["max"]:+.2f})'
                     for ch, r in g.iterrows())


# ===================================================== 5.5 multifield

W_MF = '05-results.tex, §5.5 Multifield activity'


def _units(d, envs=None, chans=None):
    rows = []
    for (e, c), b in d.banks.items():
        if (envs and e not in envs) or (chans and c not in chans):
            continue
        m = b[b.multifield.astype(bool)]
        for nid, g in m.groupby('node_id'):
            rows.append(dict(env=e, channel=c, node_id=nid, n_sub=len(g),
                             ratio=g.area_env_m2.max() / g.area_env_m2.min(),
                             multiscale=g.scale_band.nunique() > 1,
                             cx=g.centroid_x.to_numpy(), cy=g.centroid_y.to_numpy()))
    return pd.DataFrame(rows)


def _fields(d, envs=None):
    b = [b for (e, c), b in d.banks.items() if not envs or e in envs]
    return pd.concat(b) if b else pd.DataFrame()


@entry('mf.count', W_MF,
       'Multifield units; their selected subfields; % of selected fields; % of all candidate fields',
       '848 candidate fields (1.9% of those passing size), 0.7% of all',
       'libraries: units with two or more selected subfields; prune audit for candidates')
def _(d):
    u = _units(d)
    f = _fields(d)
    nf = int(f.multifield.astype(bool).sum())
    return (f'{len(u):,} units, {nf:,} fields = {100.0 * nf / len(f):.1f}% of selected fields, '
            f'{100.0 * d.pairs.n_admitted_multifield.sum() / d.pairs.n_candidates.sum():.1f}% of all candidate fields')


@entry('mf.by_feature_space', W_MF, 'Multifield units per feature space',
       'color 346, downscaled 271, hog 154, visual 66, all 11, lidar 0 (candidate fields)',
       'units with two or more selected subfields, environments pooled')
def _(d):
    u = _units(d)
    n = u.groupby('channel').size() if len(u) else pd.Series(dtype=int)
    return ', '.join(f'{NAME[c]} {int(n.get(c, 0))}' for c in CHANS)


@entry('mf.by_geometry', W_MF,
       '% of selected fields in multifield units: corridor, square, circular',
       '5.5%, 1.6%, 0.6% (of candidate fields passing size)', 'libraries, pooled per geometry')
def _(d):
    def f(envs):
        x = _fields(d, envs)
        return 100.0 * x.multifield.astype(bool).mean()
    return f'corridor {f(CORR):.1f}%, square {f(SQUARE):.1f}%, circular {f(CIRC):.1f}%'


@entry('mf.sets_with_none', W_MF, 'Sets with no multifield unit', '29 of 48',
       'libraries with no multifield row')
def _(d):
    n = sum(not b.multifield.astype(bool).any() for b in d.banks.values())
    return f'{n} of {len(d.banks)}'


@entry('mf.by_scale', W_MF,
       '% of selected fields in multifield units: finest scale and scale 4',
       '1.0% to 8.2%', 'libraries, pooled, per scale')
def _(d):
    f = _fields(d)
    v = f.groupby('scale_band').multifield.apply(lambda x: 100.0 * x.astype(bool).mean())
    return ', '.join(f'scale {int(k)} {x:.1f}%' for k, x in v.items())


@entry('mf.multiscale', W_MF,
       "NEW: multifield units whose subfields lie at different scales (AW's question)",
       '(not in the paper)', 'units whose selected subfields span more than one scale')
def _(d):
    u = _units(d)
    if not len(u):
        return 'no multifield units'
    by = u.groupby('env').multiscale.agg(['sum', 'size'])
    return (f'{int(u.multiscale.sum())} of {len(u)} units; by environment: '
            + ', '.join(f'{e} {int(r["sum"])}/{int(r["size"])}' for e, r in by.iterrows()))


@entry('mf.corr_hog_subfields', W_MF,
       'corr_lm8_l10w2, hog: multifield units and how many subfields each',
       '127 of 147 had two or more: 91 two, 36 three', 'units by number of selected subfields')
def _(d):
    u = _units(d, ['corr_lm8_l10w2'], ['hog'])
    if not len(u):
        return 'no multifield units'
    vc = u.n_sub.value_counts().sort_index()
    return f'{len(u)} units: ' + ', '.join(f'{int(v)} with {k}' for k, v in vc.items())


@entry('mf.corr_hog_spacing', W_MF,
       'corr_lm8_l10w2, hog: spacing of a unit\'s subfields along the corridor vs landmark spacing',
       'close to the landmark spacing', 'median gap between neighboring subfield centers (x)')
def _(d):
    u = _units(d, ['corr_lm8_l10w2'], ['hog'])
    if not len(u):
        return 'no multifield units'
    gaps = np.concatenate([np.diff(np.sort(x)) for x in u.cx if len(x) > 1])
    lx = sorted({round(x, 2) for x, y in d.env('corr_lm8_l10w2')['landmarks'] if y > 0})
    return (f'median {np.median(gaps):.2f} m (IQR {np.percentile(gaps, 25):.2f}–'
            f'{np.percentile(gaps, 75):.2f}); landmark spacing {np.median(np.diff(lx)):.2f} m')


@entry('mf.corr_hog_size_ratio', W_MF + '; 06-discussion.tex, §Multifield place cells',
       'corr_lm8_l10w2, hog: largest subfield over smallest, mean over units',
       '1.5', 'mean of max/min subfield area per unit')
def _(d):
    u = _units(d, ['corr_lm8_l10w2'], ['hog'])
    return f'{u.ratio.mean():.1f}' if len(u) else 'no multifield units'


@entry('mf.size_ratio_all', W_MF + '; 06-discussion.tex',
       'NEW: largest subfield over smallest, mean over every multifield unit',
       '(not in the paper; 1.5 was one set)', 'as above, all sets')
def _(d):
    u = _units(d)
    return f'{u.ratio.mean():.1f} (median {u.ratio.median():.1f})' if len(u) else 'none'


# ===================================================== validation (A2, Methods)

W_A2 = 'A2-extent-percentile.tex'
W_METH = '03-methods.tex, ¶Choice of the extent percentile'


def _ev(d):
    return d.ev


def _op(e):
    return int(np.flatnonzero(np.isclose(e['q'], 80.0))[0])


@entry('val.overlap_op', W_METH + '; ' + W_A2,
       'Median overlap of projected field and footprint at q = 80', '0.69',
       'criteria()["accuracy"]["value_op"]')
def _(d):
    return f'{_ev(d)["crit"]["accuracy"]["value_op"]:.2f}'


@entry('val.area_op', W_METH + '; ' + W_A2, 'Median area ratio at q = 80', '0.95',
       'criteria()["calibration"]["value_op"]')
def _(d):
    return f'{_ev(d)["crit"]["calibration"]["value_op"]:.2f}'


@entry('tab_valid.overlap', W_A2 + ', Table valid',
       'Overlap: best q (95% CI); nearly as good; at q = 80 (CI)',
       '80 (75–80); 70–85; 0.69 (0.64–0.74)', 'criteria() and the pooled curve band at 80')
def _(d):
    e = _ev(d); a = e['crit']['accuracy']; i = _op(e)
    lo, hi = e['cur']['band']['iou'][0][i], e['cur']['band']['iou'][1][i]
    return (f'{a["best"]:.0f} ({a["ci"][0]:.0f}–{a["ci"][1]:.0f}); {a["range"][0]:.0f}–'
            f'{a["range"][1]:.0f}; {a["value_op"]:.2f} ({lo:.2f}–{hi:.2f})')


@entry('tab_valid.area', W_A2 + ', Table valid',
       'Area ratio: best q (CI); nearly as good; at q = 80 (CI)',
       '82 (79–85); 75–85; 0.95 (0.88–1.03)', 'as above, calibration')
def _(d):
    e = _ev(d); c = e['crit']['calibration']; i = _op(e)
    lo, hi = (2 ** e['cur']['band']['lr'][0][i], 2 ** e['cur']['band']['lr'][1][i])
    return (f'{c["best"]:.0f} ({c["ci"][0]:.0f}–{c["ci"][1]:.0f}); {c["range"][0]:.0f}–'
            f'{c["range"][1]:.0f}; {c["value_op"]:.2f} ({lo:.2f}–{hi:.2f})')


@entry('val.area_at_q', W_A2, 'Area ratio at q = 65, 90, 95, 100', '0.67; 1.3, 1.7, 2.4',
       '2 ** median log2 area ratio on the pooled curve')
def _(d):
    e = _ev(d)
    v = {q: 2 ** e['cur']['point']['lr'][int(np.flatnonzero(np.isclose(e['q'], q))[0])]
         for q in (65, 90, 95, 100)}
    return f'{v[65]:.2f}; {v[90]:.1f}, {v[95]:.1f}, {v[100]:.1f}'


@entry('val.pair_best_q', W_A2, 'Best q per environment-feature-space pair: median (IQR; range)',
       '80 (75–85; 40–95)', 'cell_optima().best_q over informative pairs')
def _(d):
    c = _ev(d)['cells']; c = c[c.informative]
    b = c.best_q
    return f'{b.median():.0f} ({b.quantile(.25):.0f}–{b.quantile(.75):.0f}; {b.min():.0f}–{b.max():.0f})'


@entry('val.pairs_95', W_A2, 'Informative pairs reaching 95% of their own best overlap at q = 80',
       '29 of 47', 'cell_optima().pct_of_best >= 95')
def _(d):
    c = _ev(d)['cells']; c = c[c.informative]
    return f'{int((c.pct_of_best >= 95).sum())} of {len(c)}'


@entry('val.uninformative', W_A2, 'Pairs matching their footprints poorly at every q, and best overlap',
       'one (best overlap 0.24)', 'cell_optima(), not informative')
def _(d):
    c = _ev(d)['cells']; c = c[~c.informative]
    return (f'{len(c)}: ' + ', '.join(f'{r.env}/{NAME.get(r.channel, r.channel)} {r.best_iou:.2f}'
                                     for r in c.itertuples())) if len(c) else 'none'


@entry('val.scales', W_A2, 'q = 80 nearly as good for how many scales; best q finest -> coarsest',
       'five of six (coarsest best 70); 90 -> 70', 'level_optima by scale')
def _(d):
    l = _ev(d)['lev_scale'].sort_values('level')
    return (f'{int(l.op_inside.sum())} of {len(l)}; best q by scale '
            + ', '.join(f'{int(r.level)}: {r.best_q:.0f}' for r in l.itertuples()))


@entry('val.contours', W_A2, 'q = 80 nearly as good for how many wall contours; best q near -> far',
       'all four; 75 -> 90', 'level_optima by contour')
def _(d):
    l = _ev(d)['lev_cont'].sort_values('level')
    return (f'{int(l.op_inside.sum())} of {len(l)}; best q by contour '
            + ', '.join(f'{r.level:g}: {r.best_q:.0f}' for r in l.itertuples()))


@entry('val.by_env', W_A2, 'Median best q by environment',
       '85–90 small circular; 75–80 large circular, square, corridor with landmarks; 50 corridor without',
       'cell_optima().best_q, median per environment')
def _(d):
    c = _ev(d)['cells']
    m = c.groupby('env').best_q.median()
    return ', '.join(f'{e} {m[e]:.0f}' for e in ENVS if e in m)


@entry('val.downstream_65', W_A2 + '; ' + W_METH,
       'Sets at q = 65: median fields as % of q = 80 (range); median radius as % of q = 80',
       '83% (77–91%); 98.8%', 'downstream_table() at q = 65')
def _(d):
    t = _ev(d)['down']; t = t[np.isclose(t.q, 65)]
    return (f'{t.n_fields_pct_of_op.median():.0f}% ({t.n_fields_pct_of_op.min():.0f}–'
            f'{t.n_fields_pct_of_op.max():.0f}%); {t.median_radius_m_pct_of_op.median():.1f}%')


@entry('val.downstream_50', W_A2 + '; ' + W_METH,
       'Sets at q = 50: median fields as % of q = 80 (range); median radius as % of q = 80',
       '67% (56–76%); 97.6%', 'downstream_table() at q = 50')
def _(d):
    t = _ev(d)['down']; t = t[np.isclose(t.q, 50)]
    return (f'{t.n_fields_pct_of_op.median():.0f}% ({t.n_fields_pct_of_op.min():.0f}–'
            f'{t.n_fields_pct_of_op.max():.0f}%); {t.median_radius_m_pct_of_op.median():.1f}%')


@entry('val.discrimination', W_A2,
       'NEW: hand-crafted clusters admitted at q = 80, and non-fields admitted, under the subfield rule '
       '(two-lobed groups are now admitted as two fields)', '(see the validation report)',
       'criteria()["discrimination"] tpr_op, fpr_op')
def _(d):
    x = _ev(d)['crit']['discrimination']
    return f'{100 * x["tpr_op"]:.0f}% of hand-crafted clusters, {100 * x["fpr_op"]:.0f}% of non-fields'


# ------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', default=f'{REPO}/data_cache')
    p.add_argument('--out', default=f'{HERE}/figures/paper/paper_numbers.csv')
    p.add_argument('--only', default='', help='comma-separated id prefixes')
    args = p.parse_args()

    d = Data(os.path.abspath(args.cache))
    only = [s for s in args.only.split(',') if s]
    rows, n_err = [], 0
    for e in ENTRIES:
        if only and not any(e['id'].startswith(o) for o in only):
            continue
        try:
            new, status = e['fn'](d), 'ok'
        except Exception as ex:                                 # noqa: BLE001
            new, status = f'{type(ex).__name__}: {ex}', 'ERROR'
            n_err += 1
            if os.environ.get('PAPER_NUMBERS_DEBUG'):
                traceback.print_exc()
        rows.append(dict(id=e['id'], location=e['where'], quantity=e['what'],
                         paper_value=e['old'], new_value=new,
                         computed_as=e['how'], status=status))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'{len(rows)} numbers -> {args.out}  ({n_err} could not be computed)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
