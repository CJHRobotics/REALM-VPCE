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


def check(path):
    ok = True
    with h5py.File(path, 'r') as f:
        n = f['x'].shape[0] if 'x' in f else -1
        print(f'{path}  ({n} rows)')
        for k in ALL_KEYS:
            if k not in f:
                print(f'  {k:11s} absent')
                continue
            a = np.asarray(f[k][: min(SAMPLE, f[k].shape[0])], dtype=np.float32)
            spread = float(a.std(axis=0).mean())
            finite = float(np.isfinite(a).mean())
            flag = ''
            if spread < 1e-8:
                flag = '   <-- CONSTANT: blank frames'
                if k in IMAGE_KEYS:
                    ok = False
            print(f'  {k:11s} {str(f[k].shape):18s} per-feature std across '
                  f'positions {spread:.6g}  finite {100*finite:5.1f}%{flag}')
    print('  OK: image features vary across positions' if ok else
          '  FAIL: the renderer returned blank frames. Do not use this dataset.')
    return ok


if __name__ == '__main__':
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob('data/vpce/collect_data/*.h5'),
                       key=os.path.getmtime)[-1:]
    if not paths:
        print('no datasets found')
        sys.exit(0)
    sys.exit(0 if all(check(p) for p in paths) else 1)
