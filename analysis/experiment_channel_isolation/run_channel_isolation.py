"""Channel isolation x spatial-weighting sweep for rule-governed place fields.

Builds a place-field bank for every (feature channel, lambda) pair and
records how the fields that form depend on which sensory channel the agent
has and on how much positional binding is allowed.

Channels
    hog / color / spatial / lidar   each isolated on its own
    visual                          hog + color + spatial
    all                             everything, including lidar

Lidar is range-limited: beams beyond LIDAR_MAX_RANGE are replaced by a
sentinel and flagged in a companion in-range channel, so the agent's
distance perception is bounded.

Rules in force: 1 (contiguity), 2 (reliability), 4 (spatial weighting),
7 (anisotropy), 8/9 (size window), 11 (same-scale competition),
12 (tiling stop). See VPCE-Brain "Agglomeration Rules".

Usage
    python run_channel_isolation.py [env_name] [options]

    --channels hog,color,spatial,lidar,visual,all
    --lambdas  0,0.1,0.5,2
    --subsample N        run on a random N locations (0 = all)
    --no-gpu             force CPU
    --no-plots           skip figure generation
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch
import rules as R


LIDAR_MAX_RANGE = 5.0        # metres — the agent's distance perception limit
LIDAR_SENTINEL  = -1.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('env', nargs='?', default='circ_lm8_r0')
    ap.add_argument('--channels', default='hog,color,spatial,lidar,visual,all')
    ap.add_argument('--lambdas', default='0,0.1,0.5,2')
    ap.add_argument('--subsample', type=int, default=0)
    ap.add_argument('--bin-m', type=float, default=R.DEFAULT_CFG['BIN_M'])
    ap.add_argument('--lidar-max-range', type=float, default=LIDAR_MAX_RANGE)
    ap.add_argument('--no-gpu', action='store_true')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--figures-only', action='store_true',
                    help='rebuild figures from cached banks; no agglomeration')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def load_cached(out_root, channel_names, lambdas):
    """Reload banks, reports and diagnostics written by a previous run.

    Lets figures be iterated on without repeating the agglomeration, which
    is the expensive part and does not change when only a plot changes.
    """
    banks, reports, rows = {}, {}, []
    for cname in channel_names:
        for lam in lambdas:
            d = f'{out_root}/{cname}/lam{lam:g}'
            if not os.path.exists(f'{d}/bank.csv'):
                continue
            try:
                banks[(cname, lam)] = pd.read_csv(f'{d}/bank.csv')
            except pd.errors.EmptyDataError:      # bank written before the
                banks[(cname, lam)] = pd.DataFrame()   # explicit-schema fix
            rep = {}
            if os.path.exists(f'{d}/report.json'):
                with open(f'{d}/report.json') as f:
                    rep = json.load(f)
                # json turns the band-keyed coverage map into strings
                rep['coverage'] = {int(k): v for k, v in rep.get('coverage', {}).items()}
            if os.path.exists(f'{d}/diagnostics.npz'):
                rep.update({k: v for k, v in np.load(f'{d}/diagnostics.npz').items()})
            reports[(cname, lam)] = rep
            rows.append((cname, lam))
    print(f'  loaded {len(banks)} cached run(s) from {out_root}')
    return banks, reports


def main():
    args = parse_args()
    env_name = args.env
    channel_names = [c.strip() for c in args.channels.split(',') if c.strip()]
    lambdas = [float(x) for x in args.lambdas.split(',') if x.strip()]

    data_path = f'{REPO}/data/vpce/collect_data/{env_name}.h5'
    xml_path  = f'{REPO}/simulation/worlds/environments/vpce/{env_name}.xml'
    out_root  = f'{REPO}/data_cache/channel_isolation/{env_name}'
    fig_dir   = f'{HERE}/figures/{env_name}'
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=' * 72)
    print(f'Channel isolation  |  env={env_name}')
    print(f'  channels : {channel_names}')
    print(f'  lambdas  : {lambdas}')
    print(f'  lidar    : masked beyond {args.lidar_max_range} m -> {LIDAR_SENTINEL} + in-range flag')
    print('=' * 72)

    if args.figures_only:
        import plots
        blocks, xy = ch.load_channel_blocks(
            data_path, lidar_max_range=args.lidar_max_range,
            lidar_sentinel=LIDAR_SENTINEL, lidar_mask_channel=True, verbose=False)
        xml_root = ET.parse(xml_path).getroot() if os.path.exists(xml_path) else None
        env = R.build_env(xy, xml_root)
        banks, reports = load_cached(out_root, channel_names, lambdas)
        metrics_df = (pd.read_csv(f'{out_root}/metrics.csv')
                      if os.path.exists(f'{out_root}/metrics.csv') else pd.DataFrame())
        plots.make_all(banks, reports, metrics_df, env, xml_root, fig_dir,
                       env_name, channel_names, lambdas)
        print(f'\nFigures -> {fig_dir}')
        return

    # ---------------------------------------------------------------- load
    t_load = time.time()
    blocks, xy = ch.load_channel_blocks(
        data_path,
        lidar_max_range=args.lidar_max_range,
        lidar_sentinel=LIDAR_SENTINEL,
        lidar_mask_channel=True,
    )
    xml_root = ET.parse(xml_path).getroot() if os.path.exists(xml_path) else None
    env = R.build_env(xy, xml_root)

    if args.subsample and args.subsample < len(xy):
        sel = np.random.default_rng(args.seed).choice(len(xy), args.subsample, replace=False)
        sel.sort()
        blocks = {k: v[sel] for k, v in blocks.items()}
        xy = xy[sel]
        print(f'  subsampled to {len(xy)} locations')

    N = len(xy)
    print(f'  environment: {"disc" if env["is_circular"] else "box"}  '
          f'area={env["env_area"]:.1f} m^2  '
          f'density={N/env["env_area"]:.1f} locations/m^2')
    print(f'  load: {time.time()-t_load:.1f}s')

    device = R.pick_device(use_gpu=not args.no_gpu)
    rng = np.random.default_rng(args.seed)

    # Positional normaliser for Rule 4, shared by every run so lambda means
    # the same thing everywhere.
    ij = rng.integers(0, N, size=(2, 2_000_000))
    keep = ij[0] != ij[1]
    xy_med = float(np.median(((xy[ij[0][keep]] - xy[ij[1][keep]]) ** 2).sum(1)))
    xy_med = xy_med if xy_med > 0 else 1.0

    # ---------------------------------------------------------------- runs
    all_banks, all_reports, metrics = {}, {}, []

    for cname in channel_names:
        if cname not in ch.CHANNEL_SETS:
            print(f'  !! unknown channel set {cname!r}, skipping')
            continue
        keys = ch.CHANNEL_SETS[cname]
        if any(k not in blocks for k in keys):
            print(f'  !! {cname}: missing block(s), skipping')
            continue

        print(f'\n----- channel: {cname}  ({" + ".join(keys)}) -----')
        X = ch.assemble(blocks, keys, normalize=True)
        print(f'  feature matrix: {X.shape}  ({X.nbytes/1e9:.2f} GB)')

        # Rule 4 only adds a positional term, so the feature distances are
        # computed once here and reused by every lambda.
        D2_feat = R.feature_sq_distances(X, device=device)
        feat_med = R._median_offdiag(D2_feat, rng)
        print(f'  median feature d^2: {feat_med:.4g}')

        for lam in lambdas:
            tag = f'{cname}/lam{lam:g}'
            print(f'\n  === {tag} ===')
            t0 = time.time()
            cfg = dict(LAMBDA=lam, BIN_M=args.bin_m, RANDOM_SEED=args.seed,
                       USE_GPU=not args.no_gpu)
            bank_df, kept_mu, report = R.build_bank(
                X, xy, env, D2_feat, feat_med, xy_med,
                cfg=cfg, device=device, tag=tag)

            out_dir = f'{out_root}/{cname}/lam{lam:g}'
            os.makedirs(out_dir, exist_ok=True)
            bank_df.to_csv(f'{out_dir}/bank.csv', index=False)
            np.save(f'{out_dir}/bank_mu.npy', kept_mu.astype(np.float32))
            np.savez_compressed(
                f'{out_dir}/diagnostics.npz',
                cand_cc_frac=report['cand_cc_frac'],
                cand_split_half_iou=report['cand_split_half_iou'],
                cand_pass_size=report['cand_pass_size'],
                cand_r_eq=report['cand_r_eq'],
                cand_elongation=report['cand_elongation'],
                cand_sigma_ratio=report['cand_sigma_ratio'],
            )
            serialisable = {k: v for k, v in report.items()
                            if not isinstance(v, np.ndarray)}
            with open(f'{out_dir}/report.json', 'w') as f:
                json.dump(serialisable, f, indent=2, default=float)

            all_banks[(cname, lam)] = bank_df
            all_reports[(cname, lam)] = report

            row = dict(channel=cname, lam=lam, n_fields=len(bank_df),
                       runtime_s=round(time.time() - t0, 1),
                       n_candidates=report['n_candidates'],
                       frag_rate=report['frag_rate'],
                       median_cc_frac=report['median_cc_frac'],
                       unreliable_rate=report['unreliable_rate'],
                       median_split_half_iou=report['median_split_half_iou'],
                       band_lo=report['band_lo'], band_hi=report['band_hi'],
                       # Recorded so a channel that yields no fields is
                       # diagnosable rather than silently empty: a channel
                       # whose candidates are all near-global has failed to
                       # localise, which is itself the result.
                       median_cand_radius_m=float(np.median(report['cand_r_eq'])),
                       frac_cand_above_cap=float(
                           (report['cand_r_eq'] > report['r_max']).mean()),
                       frac_cand_below_floor=float(
                           (report['cand_r_eq'] < report['r_min']).mean()),
                       median_sigma_ratio=report['median_sigma_ratio'])
            for name, cnt in report['funnel']:
                row[f'funnel_{name}'] = cnt
            if len(bank_df):
                row.update(
                    median_radius_m=float(bank_df['radius_env_m'].median()),
                    median_elongation=float(bank_df['elongation'].median()),
                    max_radius_m=float(bank_df['radius_env_m'].max()),
                    min_radius_m=float(bank_df['radius_env_m'].min()),
                    radius_ratio=float(bank_df['radius_env_m'].max()
                                       / max(bank_df['radius_env_m'].min(), 1e-9)),
                    median_dist_to_wall_m=float(bank_df['dist_to_wall_m'].median()),
                )
                # The Paper 1 measurement: do fields grow away from the wall?
                # Reported, never enforced — wall distance is not a rule.
                if len(bank_df) >= 5:
                    row['corr_radius_wall'] = float(
                        np.corrcoef(bank_df['dist_to_wall_m'],
                                    bank_df['radius_env_m'])[0, 1])
            metrics.append(row)
            print(f'  -> {len(bank_df)} fields in {row["runtime_s"]}s -> {out_dir}')

        del X, D2_feat

    metrics_df = pd.DataFrame(metrics)
    metrics_path = f'{out_root}/metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f'\nMetrics -> {metrics_path}')
    if len(metrics_df):
        cols = [c for c in ('channel', 'lam', 'n_fields', 'median_radius_m',
                            'median_elongation', 'frag_rate', 'corr_radius_wall',
                            'median_sigma_ratio', 'frac_cand_above_cap',
                            'band_lo', 'runtime_s') if c in metrics_df.columns]
        # Tagged so slurm/send_report.py keeps these lines in the emailed
        # summary — its filter drops anything it doesn't recognise, and this
        # table is the result.
        for line in metrics_df[cols].to_string(index=False).splitlines():
            print(f'[summary] {line}')

    # ---------------------------------------------------------------- plots
    if not args.no_plots and all_banks:
        import plots
        plots.make_all(all_banks, all_reports, metrics_df, env, xml_root,
                       fig_dir, env_name, channel_names, lambdas)
        print(f'\nFigures -> {fig_dir}')


if __name__ == '__main__':
    main()
