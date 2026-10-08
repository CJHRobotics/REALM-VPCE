"""The paper's figures, under the file names the manuscript includes.

Each figure is drawn by the function its experiment script uses -- imported,
not copied -- from one of two inputs:

  cache (default)  the tables and grids a finished run left in data_cache/.
                   Nothing is recomputed, no dataset is read, no GPU is used.
                   This is what slurm/paper_figures.sh runs on GAIVI.
  --fake           synthetic input shaped like the cache, for checking a
                   layout locally before a job is submitted. The shapes and
                   numbers are made up and nothing drawn from them is a result.

Both go through the same drawing and saving path, so a layout checked with
--fake is the layout the job produces. Arena geometry is real in both: --fake
reads it from the world XML.

Text renders in DejaVu Sans either way. GAIVI has neither Arial nor Liberation
Sans and falls through to it, so pinning it here keeps a local preview honest.

Usage
    python paper_figures.py                    # every figure, from the cache
    python paper_figures.py valid-q            # one of them
    python paper_figures.py --fake             # every figure, synthetic input
    python paper_figures.py --list             # what is here

Output: one PNG (300 dpi) and one PDF per figure, named as in the paper's
figures/ folder, in figures/paper/ (cache) or figures/preview/ (--fake). Copy
the PNGs into vpce-paper/figures/ to update the manuscript.
"""
import argparse
import os
import shutil
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(f'{HERE}/../..')
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import rules as R                                   # noqa: E402
import run_extent_validation as EV                  # noqa: E402
import run_field_geometry as FG                     # noqa: E402
import run_multifield_examples as MF                # noqa: E402
import run_prune_audit as PA                        # noqa: E402
import run_scale_distribution as SD                 # noqa: E402

CACHE = f'{REPO}/data_cache'


class PaperFigures:
    """Stands in for an experiment's own saver: same .save(fig, name), but it
    writes under the paper's file name and nothing else -- no multi-page PDF,
    no pruning of the experiment's figure folder."""

    def __init__(self, out_dir, names):
        self.dir, self.names, self.written = out_dir, names, []
        os.makedirs(out_dir, exist_ok=True)

    def save(self, fig, name, mail=True):
        if name not in self.names:
            plt.close(fig)
            return
        for ext, kw in (('png', dict(dpi=300)), ('pdf', {})):
            p = f'{self.dir}/{self.names[name]}.{ext}'
            fig.savefig(p, bbox_inches='tight', pad_inches=0.03, **kw)
            self.written.append(p)
        plt.close(fig)
        print(f'  {self.dir}/{self.names[name]}.png', flush=True)

    def adopt(self, path, name):
        """Take a file an experiment saved itself, under the paper's name."""
        ext = os.path.splitext(path)[1]
        dest = f'{self.dir}/{self.names[name]}{ext}'
        shutil.move(path, dest)
        self.written.append(dest)
        print(f'  {dest}', flush=True)


# ------------------------------------------------------------ valid-q (Fig. 3)

def valid_q_cache(cache):
    """The example V1 was built for: the cluster the validation run picked at
    EXAMPLE_SCALE, whose overlap at the q in use is nearest that scale's
    median. The pick is made during the run, so a cache built at another q
    holds a different example -- refused rather than drawn under a caption
    that says q = 80."""
    cache_dir = f'{cache}/extent_validation'
    env, cname = EV.EXAMPLE_ENV, EV.GALLERY_CHANNEL
    data = EV.load_cache(cache_dir, [env])
    if env not in data['envs']:
        raise RuntimeError(f'no extent-validation results for {env} in '
                           f'{cache_dir}; run slurm/extent_validation.sh')
    q_run = float(data['meta'][env].get('q_op', np.nan))
    if not np.isclose(q_run, EV.Q_OP):
        raise RuntimeError(f'{cache_dir}/{env} was built at q = {q_run:g}, but '
                           f'q in use is {EV.Q_OP:g}: its example cluster was '
                           f'picked at the old value. Re-run '
                           f'slurm/extent_validation.sh for {env}.')
    ex = EV._example(data, env, cname)
    if ex is None:
        raise RuntimeError(f'{cache_dir}/{env} holds no V1 example for channel '
                           f'{cname}; the recovery stage did not run for it')
    print(f'  example: {env} / {cname}, cluster {ex["gid"]} '
          f'(cache from git {data["meta"][env].get("git", "?")}, '
          f'{data["meta"][env].get("started", "?")})', flush=True)
    return ex


def valid_q_fake(rng):
    """One hand-crafted cluster in the 6 m disc, made up.

    The footprint is a disc of radius 1.75 m on the left. Its feature distance
    grows away from a point inside it, faster across x than along it and with
    smooth noise, so the field grows lopsidedly to the right as q rises -- as
    the real example does.
    """
    geom, xe, ye, X, Y, in_env, ba = _fake_arena(EV.EXAMPLE_ENV)
    cx, cy, r = -3.6, 0.0, 1.75
    imask = (np.hypot(X - cx, Y - cy) <= r) & in_env
    dx, dy = X - (cx + 0.3), Y - cy
    d2 = (np.where(dx > 0, 0.45, 1.0) * dx) ** 2 + dy ** 2
    d2 = d2 * (1 + 0.25 * _smooth_noise(X.shape, rng))
    bin_d2 = np.where(in_env, d2, np.nan)
    # Members' distances to their centroid: the footprint's own, jittered, so
    # a few outside bins sit inside the members' spread.
    dc2 = np.sort(bin_d2[imask] * rng.uniform(0.85, 1.6, imask.sum()))
    return dict(bin_d2=bin_d2, dc2=dc2, imask=imask, in_env=in_env,
                x_edges=xe, y_edges=ye, bin_area=ba, geom=geom)


# ----------------------------------------------------------- funnel (Fig. 4)

def funnel_cache(cache):
    """The prune audit's per-library counts. Its CSV records no q, so the date
    it was written is printed: a file older than the move to q = 80
    (29 September 2026) is from the old operating point."""
    path = f'{cache}/prune_audit/prune_audit_pairs.csv'
    if not os.path.exists(path):
        raise RuntimeError(f'no prune audit in {path}; run slurm/prune_audit.sh')
    print(f'  {path}  (written {_written(path)})', flush=True)
    return pd.read_csv(path)


def funnel_fake(rng):
    """Every library's counts, made up around the shares the last run gave:
    about 62% too small or large, 1-2% multifield, 13% competition."""
    area = {'circ_lm8_r3': 28.27, 'circ_lm0_r3': 28.27, 'circ_lm8_r6': 113.1,
            'circ_lm0_r6': 113.1, 'corr_lm8_l10w10': 100.0,
            'corr_lm0_l10w10': 100.0, 'corr_lm8_l10w2': 20.0,
            'corr_lm0_l10w2': 20.0}
    rows = []
    for e, a in area.items():
        corridor = e.endswith('w2')
        for c in ('hog', 'color', 'spatial', 'lidar', 'visual', 'all'):
            n = int(rng.integers(2200, 2800))
            size = int(n * rng.uniform(0.66, 0.70) if corridor
                       else n * rng.uniform(0.60, 0.63))
            contig = int(n * (rng.uniform(0.015, 0.03) if corridor
                              else rng.uniform(0.002, 0.01)))
            compete = int(n * rng.uniform(0.09, 0.16))
            rows.append(dict(env=e, channel=c, env_area_m2=a, n_candidates=n,
                             pass_size=n - size, pass_contiguity=n - size - contig,
                             pass_competition=n - size - contig - compete,
                             n_admitted=n - size - contig - compete))
    return pd.DataFrame(rows)


def funnel_draw(pairs, figs):
    """P1 saves its own file; it is drawn to a scratch folder and adopted."""
    with tempfile.TemporaryDirectory() as tmp:
        for p in PA.fig_rule_shares(pairs, tmp):
            figs.adopt(p, 'P1_rule_shares')


# ------------------------------------------------------ size-dist (Fig. 6)

SIZE_ENVS = ['circ_lm8_r3', 'circ_lm0_r3', 'circ_lm8_r6', 'circ_lm0_r6',
             'corr_lm8_l10w10', 'corr_lm0_l10w10', 'corr_lm8_l10w2',
             'corr_lm0_l10w2']


def _env_geom(env_name):
    """An arena's outline, area and landmarks, from the world XML, built as
    the scale experiment's --report-only builds them."""
    root = ET.parse(f'{REPO}/simulation/worlds/environments/vpce/'
                    f'{env_name}.xml').getroot()
    env = R.build_env(None, root)
    marks = [(float(l.get('x')), float(l.get('y')))
             for l in root.findall('landmark')]
    return dict(env, landmarks=marks, n_landmarks=len(marks))


def size_dist_cache(cache):
    """Every library at the operating point, read as --report-only reads
    them. The file names carry q (`_p80_`), so a library built at another q
    is not found rather than drawn."""
    out_dir = f'{cache}/scale_distribution'
    key = (SD.DEFAULT_PCTL, SD.DEFAULT_T, SD.PRIMARY_IOU)
    banks, lost = {}, []
    for e in SIZE_ENVS:
        for c in SD.CHANNELS:
            path = SD.bank_path(out_dir, e, c, *key, SD.DEFAULT_TILING)
            if os.path.exists(path):
                banks[(e, c) + key] = pd.read_csv(path)
            else:
                lost.append(os.path.relpath(path, out_dir))
    if not banks:
        raise RuntimeError(f'no libraries at q = {SD.DEFAULT_PCTL} in '
                           f'{out_dir}; run slurm/scale_distribution.sh')
    print(f'  {len(banks)} libraries from {out_dir}', flush=True)
    if lost:
        print(f'  !! {len(lost)} missing, drawn without them: '
              + ', '.join(lost[:4]) + (' ...' if len(lost) > 4 else ''),
              flush=True)
    return banks


def size_dist_fake(rng):
    """Made-up libraries with everything the map figures draw: log-normal
    field areas between each arena's floor and ceiling, a scale from the
    model's own banding, an ellipse per field, and a center inside the arena.
    About as many fields as a real library, most of them at the finest scale.
    """
    key = (SD.DEFAULT_PCTL, SD.DEFAULT_T, SD.PRIMARY_IOU)
    C = R.DEFAULT_CFG
    banks = {}
    for e in SIZE_ENVS:
        g = _env_geom(e)
        lo = C['RULE8_AREA_FRAC'] * g['env_area']
        hi = C['RULE9_AREA_FRAC'] * g['env_area']
        r_min = np.sqrt(lo / np.pi)
        for c in SD.CHANNELS:
            n = int(rng.integers(450, 700))
            area = np.minimum(lo * (1 + rng.lognormal(0.3, 1.0, n)), hi)
            r = np.sqrt(area / np.pi)
            el = 1 + rng.exponential(0.35, n)
            a, b = r * np.sqrt(el), r / np.sqrt(el)
            if g.get('is_circular'):
                rho = np.sqrt(rng.uniform(0, 1, n)) * np.maximum(g['env_R'] - r, 0)
                th = rng.uniform(0, 2 * np.pi, n)
                cx, cy = rho * np.cos(th), rho * np.sin(th)
            else:
                cx = rng.uniform(g['x_min'] + r, np.maximum(g['x_max'] - r, g['x_min'] + r))
                cy = rng.uniform(g['y_min'] + r, np.maximum(g['y_max'] - r, g['y_min'] + r))
            banks[(e, c) + key] = pd.DataFrame(dict(
                area_env_m2=area, radius_env_m=r,
                scale_band=np.minimum(R.assign_bands(r, r_min, C['BAND_RATIO']), 5),
                centroid_x=cx, centroid_y=cy, semi_major_m=a, semi_minor_m=b,
                orientation_rad=rng.uniform(0, np.pi, n)))
    return banks


def size_dist_draw(banks, figs):
    """S1 saves its own PNG and PDF into the folder it is given."""
    geom = {e: _env_geom(e) for e in SIZE_ENVS}
    with tempfile.TemporaryDirectory() as tmp:
        SD.fig_sizes_by_arena(banks, SIZE_ENVS, SD.CHANNELS, geom, tmp)
        for ext in ('png', 'pdf'):
            figs.adopt(f'{tmp}/S1_sizes_by_arena.{ext}', 'S1_sizes_by_arena')


def scale_maps_draw(banks, figs):
    """S2a, one figure per arena; the paper prints circ_lm8_r6's, and the
    other seven come along under the same naming."""
    geom = {e: _env_geom(e) for e in SIZE_ENVS}
    with tempfile.TemporaryDirectory() as tmp:
        SD.fig_scale_maps(banks, SIZE_ENVS, SD.CHANNELS, geom, tmp,
                          R.DEFAULT_CFG)
        for e in SIZE_ENVS:
            p = f'{tmp}/S2a_scales_{e}.png'
            if os.path.exists(p):
                figs.adopt(p, f'S2a_scales_{e}')


def outlines_draw(banks, figs):
    """S2b, every library on one sheet, its rows in the order the other
    figures use: each arena directly followed by its copy without landmarks."""
    geom = {e: _env_geom(e) for e in SIZE_ENVS}
    with tempfile.TemporaryDirectory() as tmp:
        SD.fig_field_outlines(banks, SIZE_ENVS, SD.CHANNELS, geom, tmp)
        figs.adopt(f'{tmp}/S2b_field_outlines.png', 'S2b_field_outlines')


def size_fits_cache(cache):
    """Figure 12: the same libraries as Figure 6, with the fits the scale run
    left in fits.csv."""
    path = f'{cache}/scale_distribution/fits.csv'
    if not os.path.exists(path):
        raise RuntimeError(f'no fits in {path}; run slurm/scale_distribution.sh')
    print(f'  {path}  (written {_written(path)})', flush=True)
    return dict(banks=size_dist_cache(cache), fits=pd.read_csv(path))


def size_fits_fake(rng):
    """The fake libraries with no fits: histograms only, for the layout."""
    return dict(banks=size_dist_fake(rng), fits=pd.DataFrame(
        columns=['env', 'channel', 'variable', 'form', 'params', 'trunc_lo',
                 'trunc_hi', 'd_aic', 'extent_pctl', 'act_thresh',
                 'split_half_iou_min']))


def size_fits_draw(data, figs):
    geom = {e: _env_geom(e) for e in SIZE_ENVS}
    with tempfile.TemporaryDirectory() as tmp:
        SD.fig_distributions(data['banks'], data['fits'], SIZE_ENVS,
                             SD.CHANNELS, geom, tmp)
        figs.adopt(f'{tmp}/S1_sizes_with_fits.png', 'S1_sizes_with_fits')


# ------------------------------------------- elongation (Figs. 7 and 8)

def _written(path):
    """When a cached table was written: none of these record their q, so the
    date is the check -- before 29 September 2026 means the old q = 65."""
    import time
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))


def fields_cache(cache):
    path = f'{cache}/field_geometry/fields.csv'
    if not os.path.exists(path):
        raise RuntimeError(f'no field geometry in {path}; run '
                           f'slurm/field_geometry.sh')
    print(f'  {path}  (written {_written(path)})', flush=True)
    return pd.read_csv(path)


def fields_fake(rng):
    """Made-up fields for all eight arenas and six feature spaces: fewer at
    each coarser scale, elongated near the wall and more so in the corridor."""
    area = {'circ_lm8_r3': 28.27, 'circ_lm0_r3': 28.27, 'circ_lm8_r6': 113.1,
            'circ_lm0_r6': 113.1, 'corr_lm8_l10w10': 100.0,
            'corr_lm0_l10w10': 100.0, 'corr_lm8_l10w2': 20.0,
            'corr_lm0_l10w2': 20.0}
    parts = []
    for e, a in area.items():
        k = 3.0 if e.endswith('w2') else 1.0
        for c in ('hog', 'color', 'spatial', 'lidar', 'visual', 'all'):
            scale = rng.choice(6, 600, p=[.55, .22, .12, .06, .03, .02])
            wd = rng.uniform(0, 1, 600)
            el = 1 + k * np.exp(-3 * wd) * rng.lognormal(0, 0.5, 600) \
                + rng.exponential(0.25, 600)
            parts.append(pd.DataFrame(dict(env=e, channel=c, env_area_m2=a,
                                           scale=scale, wall_dist_norm=wd,
                                           elongation=el)))
    return pd.concat(parts, ignore_index=True)


def _geometry_draw(fields, figs, name, draw):
    """The geometry figures save into FG.FIG_DIR; point it at a scratch
    folder for the call and adopt what lands there."""
    keep = FG.FIG_DIR
    with tempfile.TemporaryDirectory() as tmp:
        FG.FIG_DIR = tmp
        try:
            draw(fields, f'{name}.png')
        finally:
            FG.FIG_DIR = keep
        figs.adopt(f'{tmp}/{name}.png', name)


def elong_scale_draw(fields, figs):
    _geometry_draw(fields, figs, 'G1_elongation_by_scale',
                   FG.fig_elongation_by_scale)


def elong_wall_draw(fields, figs):
    # The same call finish() makes, so the paper's G2 is the report's.
    _geometry_draw(fields, figs, 'G2_elongation_vs_wall',
                   lambda f, n: FG._vs_distance(
                       f, 'elongation', n, 'Elongation against wall distance',
                       'elongation', hline=1.0))


# ---------------------------------------------- correlation grid (Fig. 9)

def rho_cache(cache):
    path = f'{cache}/field_geometry/correlations.csv'
    if not os.path.exists(path):
        raise RuntimeError(f'no correlations in {path}; run '
                           f'slurm/field_geometry.sh')
    print(f'  {path}  (written {_written(path)})', flush=True)
    return pd.read_csv(path)


def rho_fake(rng):
    """Made-up Spearman correlations: near zero with scale, negative with
    wall distance and weaker in the corridors, a star where q < 0.05."""
    rows = []
    for e in SIZE_ENVS:
        weak = 0.4 if e.endswith('w2') else 1.0
        for _, _, label in FG.PAIRS:
            centre = 0.0 if label.startswith('scale') else -0.7 * weak
            for c in list(SD.CHANNELS) + [None]:
                rho = float(np.clip(centre + rng.normal(0, 0.12), -0.99, 0.99))
                rows.append(dict(grouping='env' if c is None else 'env x channel',
                                 subset='all fields', env=e, channel=c,
                                 pair=label, rho=rho,
                                 q=0.001 if abs(rho) > 0.1 else 0.3))
    return pd.DataFrame(rows)


def rho_draw(corr, figs):
    _geometry_draw(corr, figs, 'G4_correlation_summary', FG.fig_rho_summary)


# ------------------------------------------------------- multifield (Fig. 10)
#
# One panel per environment, top to bottom. Each panel pools the multifield
# clusters run_multifield_examples.py found in EVERY feature space it has
# cached for that environment: the figure shows that the model produces
# multifield responses, not which feature space does, so none is named.

MULTIFIELD_ENVS = ['corr_lm8_l10w2', 'corr_lm0_l10w10']
MULTIFIELD_N = 6          # as many as there are hues that stay apart


def _multifield_panel(env, clusters, xe, ye, geom):
    picked = MF.pick_superimposed(clusters, MULTIFIELD_N)
    n_ms = sum(len(set(c['scales'])) > 1 for c in clusters)
    print(f'  {env}: {len(clusters)} multifield clusters, {n_ms} multiscale; '
          f'drawing {len(picked)}: ' + '; '.join(
              f'{c.get("feature_space", "")} {c["node_id"]} scales '
              f'{"/".join(map(str, c["scales"]))}' for c in picked), flush=True)
    return dict(env=env, geom=geom, x_edges=xe, y_edges=ye, clusters=picked)


def multifield_cache(cache):
    """Every multifield unit in the environment's libraries, all feature
    spaces pooled: a unit with two or more selected subfields, each drawn as
    its fitted ellipse. Read from the scale run's libraries, so it is the
    model's own selection, not a separate rebuild.

    Overlap between units is judged on the ellipses rasterized to a 5 cm
    grid, which is all pick_superimposed needs of them."""
    key = (SD.DEFAULT_PCTL, SD.DEFAULT_T, SD.PRIMARY_IOU)
    panels = []
    for env in MULTIFIELD_ENVS:
        geom = _env_geom(env)
        if geom.get('is_circular'):
            r = geom['env_R']; x0, x1, y0, y1 = -r, r, -r, r
        else:
            x0, x1, y0, y1 = geom['x_min'], geom['x_max'], geom['y_min'], geom['y_max']
        xe, ye = np.arange(x0, x1 + 0.025, 0.05), np.arange(y0, y1 + 0.025, 0.05)
        X, Y = np.meshgrid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]),
                           indexing='ij')
        clusters, have = [], []
        for c in SD.CHANNELS:
            p = SD.bank_path(f'{cache}/scale_distribution', env, c, *key,
                             SD.DEFAULT_TILING)
            if not os.path.exists(p):
                continue
            b = pd.read_csv(p)
            if 'multifield' not in b:
                raise RuntimeError(f'{p} has no multifield column: it was built '
                                   f'before the subfield rule; rerun it')
            m = b[b.multifield.astype(bool)]
            have.append(f'{SD.channel_label(c, long=False)} {m.node_id.nunique()}')
            for nid, g in m.groupby('node_id'):
                ell = list(zip(g.centroid_x, g.centroid_y, g.semi_major_m,
                               g.semi_minor_m, g.orientation_rad))
                masks = []
                for (x, y, a, bb, th) in ell:
                    u = (X - x) * np.cos(th) + (Y - y) * np.sin(th)
                    v = -(X - x) * np.sin(th) + (Y - y) * np.cos(th)
                    masks.append((u / a) ** 2 + (v / bb) ** 2 <= 1)
                areas = g.area_env_m2.to_numpy()
                clusters.append(dict(node_id=int(nid), feature_space=c,
                                     scale=int(g.scale_band.min()),
                                     balance=float(np.sort(areas)[-2] / areas.max()),
                                     areas=areas, scales=g.scale_band.to_numpy(),
                                     ellipses=ell, masks=np.stack(masks)))
        if not have:
            raise RuntimeError(f'no libraries for {env} in {cache}/scale_distribution')
        print(f'  {env}: multifield units by feature space: ' + ', '.join(have),
              flush=True)
        if not clusters:
            raise RuntimeError(f'{env}: no multifield unit in any feature space')
        panels.append(_multifield_panel(env, clusters, xe, ye, geom))
    return panels


def multifield_fake(rng):
    """Made-up multifield clusters: two or three round-ish subfields each --
    at the landmark spacing in the corridor, mirrored across the square --
    and about a third of them with subfields at different scales."""
    panels = []
    for env in MULTIFIELD_ENVS:
        geom, xe, ye, X, Y, in_env, ba = _fake_arena(env)
        wide = (geom['x_max'] - geom['x_min']) / (geom['y_max'] - geom['y_min']) >= 2.5
        r_min = 0.09 if wide else 0.21
        clusters = []
        for _ in range(40):
            n_sub = int(rng.choice([2, 2, 3]))
            s0 = int(rng.integers(0, 4))
            ss = [s0] * n_sub
            if rng.random() < 0.35:
                ss[-1] = min(s0 + int(rng.integers(1, 3)), 5)
            if wide:
                step = 10.0 / 4
                x0 = rng.uniform(-5 + 0.6, 5 - 0.6 - step * (n_sub - 1))
                y = rng.uniform(-0.7, 0.7)
                centres = [(x0 + k * step, y) for k in range(n_sub)]
            else:
                # Mirror images across the square's center lines: a view
                # repeated by the room's symmetry, which is what the real
                # ones in an arena without landmarks tend to be.
                x, y = rng.uniform(-4.2, 4.2, 2)
                centres = [(x, y), (-x, y), (x, -y)][:n_sub]
            masks, areas = [], []
            for (cx, cy), sc in zip(centres, ss):
                r = r_min * 1.6 ** (sc + rng.uniform(0.2, 0.8))
                e = rng.uniform(1.0, 1.5)
                m = (((X - cx) / (r * e)) ** 2 + ((Y - cy) * e / r) ** 2 <= 1) & in_env
                masks.append(m)
                areas.append(m.sum() * ba)
            # Subfields are separate patches by definition: a fake cluster
            # whose own subfields touch is not one, so it is dropped.
            if min(areas) <= 0 or np.sum(masks, axis=0).max() > 1:
                continue
            areas = np.array(areas)
            clusters.append(dict(node_id=60000 + len(clusters), scale=s0,
                                 balance=float(np.sort(areas)[-2] / areas.max()),
                                 areas=areas,
                                 scales=MF.subfield_scales(areas, r_min, 1.6),
                                 masks=np.stack(masks)))
        panels.append(_multifield_panel(env, clusters, xe, ye, geom))
    return panels


# ------------------------------------------------------------------ registry
#
# name: (paper label, {experiment's figure name: paper file name},
#        cache loader, fake builder, drawing function(data, figs))

FIGURES = {
    'valid-q': ('fig:valid-q (Fig. 3)',
                {'V1_what_q_does': 'valid_V1_what_q_does'},
                valid_q_cache, valid_q_fake, EV.fig_mechanism),
    'funnel': ('fig:funnel (Fig. 4)',
               {'P1_rule_shares': 'prune_P1_rule_shares'},
               funnel_cache, funnel_fake, funnel_draw),
    'size-dist': ('fig:size-dist (Fig. 6)',
                  {'S1_sizes_by_arena': 'scale_S1_sizes_by_arena'},
                  size_dist_cache, size_dist_fake, size_dist_draw),
    'scale-maps': ('fig:scale-maps (Fig. 5)',
                   {f'S2a_scales_{e}': f'scale_S2a_{e}' for e in SIZE_ENVS},
                   size_dist_cache, size_dist_fake, scale_maps_draw),
    'outlines': ('fig:supp-outlines (Fig. 13)',
                 {'S2b_field_outlines': 'scale_S2b_field_outlines'},
                 size_dist_cache, size_dist_fake, outlines_draw),
    'size-fits': ('fig:supp-size-fits (Fig. 12)',
                  {'S1_sizes_with_fits': 'scale_S1_sizes_with_fits'},
                  size_fits_cache, size_fits_fake, size_fits_draw),
    'elong-scale': ('fig:elong-scale (Fig. 7)',
                    {'G1_elongation_by_scale': 'geom_G1_elongation_by_scale'},
                    fields_cache, fields_fake, elong_scale_draw),
    'elong-wall': ('fig:elong-wall (Fig. 8)',
                   {'G2_elongation_vs_wall': 'geom_G2_elongation_vs_wall'},
                   fields_cache, fields_fake, elong_wall_draw),
    'geom-rho': ('fig:geom-rho (Fig. 9)',
                 {'G4_correlation_summary': 'geom_G4_correlation_summary'},
                 rho_cache, rho_fake, rho_draw),
    'multifield': ('fig:multifield (Fig. 10)',
                   {'M00_multifield_map': 'multifield_M00_overview'},
                   multifield_cache, multifield_fake, MF.fig_multifield_map),
}


# ------------------------------------------------------------- fake helpers

def _fake_arena(env_name, bin_m=0.1, keep_out=0.2):
    """Real geometry from the world XML, and a floor grid shaped like the
    cache's."""
    root = ET.parse(f'{REPO}/simulation/worlds/environments/vpce/'
                    f'{env_name}.xml').getroot()
    env = R.build_env(None, root)
    landmarks = [(float(l.get('x')), float(l.get('y')))
                 for l in root.findall('landmark')]
    geom = dict(env, landmarks=landmarks, n_landmarks=len(landmarks))
    if env.get('is_circular'):
        r = env['env_R']
        x0, x1, y0, y1 = -r, r, -r, r
    else:
        x0, x1, y0, y1 = env['x_min'], env['x_max'], env['y_min'], env['y_max']
    xe = np.arange(x0, x1 + bin_m / 2, bin_m)
    ye = np.arange(y0, y1 + bin_m / 2, bin_m)
    X, Y = np.meshgrid(EV._centres(xe), EV._centres(ye), indexing='ij')
    if env.get('is_circular'):
        in_env = np.hypot(X, Y) <= env['env_R'] - keep_out
    else:
        in_env = ((X >= x0 + keep_out) & (X <= x1 - keep_out) &
                  (Y >= y0 + keep_out) & (Y <= y1 - keep_out))
    return geom, xe, ye, X, Y, in_env, bin_m ** 2


def _smooth_noise(shape, rng, width=6):
    """Spatially smooth noise in [-1, 1]: what makes a fake field ragged."""
    from scipy import ndimage
    n = ndimage.gaussian_filter(rng.standard_normal(shape), width)
    return n / np.abs(n).max()


# ---------------------------------------------------------------------- mail

def mail_figures(out, job, status, log, send=None):
    """Mail every PNG in `out`, in as many messages as the attachment budget
    needs, each saying which part it is and what it carries.

    One message used to carry them all, and the mailer drops whatever no
    longer fits its budget: the eight S2a sheets come before S2b by name and
    used it up, so the paper's Figure 13 was silently left behind. Now the
    paper's own figures go first, every file lands in some message, and a
    file too large for any message is named in the first one.
    """
    import glob
    from realm_tools.experiment_lib import reporting
    send = send or reporting.send_email
    budget = reporting.DEFAULT_MAX_ATTACHMENT_BYTES
    pngs = sorted(glob.glob(f'{out}/*.png'))
    # The paper's figures first; the S2a sheets for arenas the paper does not
    # print come last.
    extra = lambda p: ('scale_S2a_' in p and not p.endswith('scale_S2a_circ_lm8_r6.png'))
    pngs.sort(key=lambda p: (extra(p), p))
    too_big = [p for p in pngs if os.path.getsize(p) > budget]
    # The first message also carries the log and any CSV beside the figures
    # (the paper's numbers), so it starts with less room.
    first = sorted(glob.glob(f'{out}/*.csv')) + ([log] if os.path.exists(log) else [])
    batches, cur = [], []
    room = budget - sum(os.path.getsize(p) for p in first)
    for p in (p for p in pngs if p not in too_big):
        size = os.path.getsize(p)
        if size > room and cur:
            batches.append(cur)
            cur, room = [], budget
        cur.append(p)
        room -= size
    if cur:
        batches.append(cur)
    batches = batches or [[]]
    name = lambda p: p.rsplit('/', 1)[-1]
    verdict = 'all drawn' if str(status) == '0' else 'SOME FAILED - see the log'
    everything = '\n'.join(f'  {name(p)}' for p in pngs) or '  (none)'
    for k, batch in enumerate(batches, 1):
        part = f', part {k} of {len(batches)}' if len(batches) > 1 else ''
        body = (f'{len(pngs)} figure(s) drawn, named as in vpce-paper/figures/'
                f'{part}.\n\nAttached here:\n'
                + ('\n'.join(f'  {name(p)}' for p in batch
                             + (first[:-1] if k == 1 and first and first[-1] == log
                                else first if k == 1 else [])) or '  (none)'))
        if len(batches) > 1:
            body += f'\n\nAll of them:\n{everything}'
        if k == 1 and too_big:
            body += ('\n\nToo large to mail, on disk only:\n'
                     + '\n'.join(f'  {name(p)} ({os.path.getsize(p) / 1e6:.0f} MB)'
                                  for p in too_big))
        body += f'\n\nPDFs are beside them in {out}.\n'
        send(f'[REALM-VPCE] paper figures, {verdict} (job {job}){part}', body,
             attachments=batch + (first if k == 1 else []))
    return len(batches)


# ---------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('which', nargs='*', metavar='FIGURE',
                   help=f'any of: {", ".join(FIGURES)} (default: all)')
    p.add_argument('--fake', action='store_true',
                   help='synthetic input, for a local layout check')
    p.add_argument('--out', default='',
                   help='output folder (default figures/paper, or '
                        'figures/preview with --fake)')
    p.add_argument('--cache', default=CACHE,
                   help='data_cache folder to read (default: the repo\'s)')
    p.add_argument('--seed', type=int, default=0, help='for --fake')
    p.add_argument('--list', action='store_true')
    p.add_argument('--mail', nargs=3, metavar=('JOB', 'STATUS', 'LOG'),
                   help='mail what is in --out instead of drawing (the job '
                        'calls this once the figures are written)')
    args = p.parse_args()

    if args.mail:
        out = args.out or f'{HERE}/figures/paper'
        n = mail_figures(out, *args.mail)
        print(f'mailed in {n} message(s)')
        return 0

    if args.list:
        for k, (label, names, *_) in FIGURES.items():
            print(f'  {k:10s} {label:24s} -> {", ".join(names.values())}')
        return 0
    bad = [w for w in args.which if w not in FIGURES]
    if bad:
        p.error(f'unknown figure(s) {", ".join(bad)}; --list shows them')
    which = args.which or list(FIGURES)
    cache = os.path.abspath(args.cache)
    out = args.out or f'{HERE}/figures/{"preview" if args.fake else "paper"}'

    EV._style()
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    # The paper-size figures draw inside SD.PAPER_RC, which names Arial
    # first; pin that too, or a Mac preview silently renders in Arial.
    SD.PAPER_RC['font.sans-serif'] = ['DejaVu Sans']
    print(f'paper figures: {", ".join(which)}  '
          f'({"FAKE input" if args.fake else "from " + cache})', flush=True)

    # One figure failing -- a cache not built yet, say -- must not cost the
    # others theirs. Each is reported, and the exit status says if any failed.
    failed = []
    for name in which:
        label, names, from_cache, fake, draw = FIGURES[name]
        print(f'\n{name}  {label}', flush=True)
        saver = PaperFigures(out, names)
        try:
            data = fake(np.random.default_rng(args.seed)) if args.fake \
                else from_cache(cache)
            draw(data, saver)
            missing = [n for n in names.values()
                       if f'{out}/{n}.png' not in saver.written]
            if missing:
                raise RuntimeError(f'drawn, but not written: {", ".join(missing)}')
        except Exception as e:                                  # noqa: BLE001
            print(f'  !! {name} FAILED: {e}', flush=True)
            if not isinstance(e, RuntimeError):
                traceback.print_exc()
            failed.append(name)

    print(f'\nfigures -> {out}')
    if failed:
        print(f'FAILED: {", ".join(failed)}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
