"""Did the headless renderer actually produce images?

The failure this guards against is silent: a headless Webots that returns
blank frames still writes a dataset of the right shape and size, and every
feature is identical at every position. That is indistinguishable from a
successful run until an experiment produces nonsense weeks later.

Constant image features across positions means blank frames, not a boring
arena -- even an empty room looks different from different places.

    python check_dataset.py [path.h5 ...]      # defaults to the newest
"""

import glob
import os
import sys

import h5py
import numpy as np

IMAGE_KEYS = ('hog', 'color_hist', 'spatial')
ALL_KEYS = IMAGE_KEYS + ('lidar',)
SAMPLE = 4000


def summarize(path):
    """Stats for one dataset. Shared with the collection report."""
    out = dict(path=path, name=os.path.basename(path)[:-3],
               size_bytes=os.path.getsize(path), blocks={}, blank=[], ok=True)
    with h5py.File(path, 'r') as f:
        out['rows'] = int(f['x'].shape[0]) if 'x' in f else -1
        if 'x' in f and 'theta' in f:
            n_theta = len(np.unique(np.round(np.asarray(f['theta'][:200]), 6)))
            out['headings'] = n_theta
            out['positions'] = out['rows'] // max(n_theta, 1)
            xs, ys = np.asarray(f['x'][:]), np.asarray(f['y'][:])
            out['extent'] = (float(xs.min()), float(xs.max()),
                             float(ys.min()), float(ys.max()))
        for k in ALL_KEYS:
            if k not in f:
                continue
            a = np.asarray(f[k][: min(SAMPLE, f[k].shape[0])], dtype=np.float32)
            spread = float(a.std(axis=0).mean())
            out['blocks'][k] = dict(shape=tuple(int(v) for v in f[k].shape),
                                    spread=spread,
                                    finite=float(np.isfinite(a).mean()))
            # Only image channels gate: a lidar scan can legitimately be
            # constant in a rotationally symmetric arena, an image cannot.
            if spread < 1e-8 and k in IMAGE_KEYS:
                out['blank'].append(k)
    out['ok'] = not out['blank']
    return out


def format_summary(s):
    L = [f"{s['name']}   {s['size_bytes']/1e9:.2f} GB",
         f"  rows {s['rows']:,}"
         + (f"   positions {s['positions']:,} x {s['headings']} headings"
            if 'positions' in s else '')]
    if 'extent' in s:
        x0, x1, y0, y1 = s['extent']
        L.append(f"  extent  x [{x0:.2f}, {x1:.2f}]   y [{y0:.2f}, {y1:.2f}]")
    for k, b in s['blocks'].items():
        flag = '   <-- CONSTANT: blank frames' if (b['spread'] < 1e-8
                                                   and k in IMAGE_KEYS) else ''
        L.append(f"  {k:11s} {str(b['shape']):18s} std across positions "
                 f"{b['spread']:.6g}  finite {100*b['finite']:5.1f}%{flag}")
    L.append('  OK: image features vary across positions' if s['ok'] else
             '  FAIL: renderer returned blank frames. Do not use this dataset.')
    return '\n'.join(L)


def check(path):
    s = summarize(path)
    print(format_summary(s))
    return s['ok']


if __name__ == '__main__':
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob('data/vpce/collect_data/*.h5'),
                       key=os.path.getmtime)[-1:]
    if not paths:
        print('no datasets found')
        sys.exit(0)
    sys.exit(0 if all(check(p) for p in paths) else 1)
