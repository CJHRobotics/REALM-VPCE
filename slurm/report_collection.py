"""Email a summary of what a collection job actually produced.

SLURM's own --mail-type=END says a job finished; it does not say whether the
dataset is usable. This reports what was collected -- arena geometry, sample
counts, feature blocks, and the blank-frame verdict -- so the run can be
judged from the mail rather than by opening files afterwards.

    python report_collection.py <maze> [<maze> ...] [--elapsed SECONDS]
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from check_dataset import summarize, format_summary          # noqa: E402
from realm_tools.experiment_lib.reporting import send_email   # noqa: E402

DATA = f'{REPO}/data/vpce/collect_data'
ENVS = f'{REPO}/simulation/worlds/environments/vpce'


def arena(maze):
    """Geometry and landmarks from the world file, for context in the mail."""
    path = f'{ENVS}/{maze}.xml'
    if not os.path.exists(path):
        return '  (world file not found)'
    root = ET.parse(path).getroot()
    lms = root.findall('landmark')
    sizes = {(l.get('width'), l.get('height')) for l in lms}
    cw = root.findall('circular_wall')
    if cw:
        r = float(cw[0].get('radius'))
        geom = f'disc, radius {r:g} m, area {np.pi*r**2:.2f} m2'
    else:
        walls = [w for w in root.findall('wall') if w.get('type') == 'boundary']
        xs = [float(w.get(k)) for w in walls for k in ('x1', 'x2')]
        ys = [float(w.get(k)) for w in walls for k in ('y1', 'y2')]
        w_, h_ = max(xs) - min(xs), max(ys) - min(ys)
        geom = f'rectangle {w_:.2f} x {h_:.2f} m, area {w_*h_:.2f} m2'
    return (f'  arena     {geom}\n'
            f'  landmarks {len(lms)} panels, size '
            f'{", ".join(sorted("x".join(s) for s in sizes))} m')


# Mirrors the fallback in collect_data.py: the landmark-count arenas
# (circ_lm4/6/10_r0) share the circ_lm8_r0 grid rather than carrying one
# each, since they differ only in the number of panels on the wall.
POSITIONS_FALLBACK = 'circ_lm8_r0_positions.csv'


def grid(maze):
    csv = f'{ENVS}/positions/{maze}_positions.csv'
    shared = ''
    if not os.path.exists(csv):
        csv = f'{ENVS}/positions/{POSITIONS_FALLBACK}'
        shared = f'  (shared: {POSITIONS_FALLBACK})'
    if not os.path.exists(csv):
        return '  (position grid not found)'
    import pandas as pd
    d = pd.read_csv(csv)
    u = np.unique(np.round(d.x.to_numpy(), 6))
    dd = np.diff(u)
    dd = dd[dd > 1e-9]
    s = float(np.median(dd)) if len(dd) else float('nan')
    return f'  grid      {len(d):,} positions, spacing {s:.4f} m{shared}'


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('mazes', nargs='+')
    ap.add_argument('--elapsed', type=float, default=None,
                    help='job wall time in seconds')
    ns = ap.parse_args()
    args, elapsed = ns.mazes, ns.elapsed

    job = os.environ.get('SLURM_JOB_ID', 'local')
    node = os.environ.get('SLURMD_NODENAME', os.uname().nodename)

    lines = ['Webots data collection on GAIVI', '',
             f'job      {job}   node {node}']
    if elapsed:
        lines.append(f'elapsed  {elapsed/3600:.2f} h')
    lines.append('')

    all_ok, found = True, 0
    for maze in args:
        path = f'{DATA}/{maze}.h5'
        lines += ['=' * 60, maze, '=' * 60, arena(maze), grid(maze), '']
        if not os.path.exists(path):
            lines += ['  NO DATASET WRITTEN -- the run did not produce this '
                      'arena. See the job log.', '']
            all_ok = False
            continue
        found += 1
        s = summarize(path)
        all_ok &= s['ok']
        lines += [format_summary(s), '']

    verdict = ('All datasets look usable.' if all_ok and found else
               'PROBLEMS -- see above. Do not run experiments on these.')
    lines += ['=' * 60, verdict]

    body = '\n'.join(lines)
    print(body, flush=True)
    tag = 'OK' if all_ok and found else 'CHECK'
    send_email(f'[REALM-VPCE] collection {tag} — {", ".join(args)}', body)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
