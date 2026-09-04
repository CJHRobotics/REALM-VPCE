"""Does a channel carry position information at all?

Field count is a poor diagnostic: it is the end of a long pipeline, and a
channel can produce almost no admitted fields either because its features are
uninformative or because its candidates are rejected downstream. Two wrong
explanations for the colour channel's collapse were reached by reasoning from
field counts alone.

This measures the input side directly. For a sample of positions, find each
one's nearest neighbour in FEATURE space and report how far away that
neighbour is in REAL space. An informative channel returns a spatial
neighbour: the 2026-08-21 report measured 4-14 cm this way, with hog at 5 cm
and colour worst at 14 cm. A channel carrying no position information returns
an arbitrary location, so the figure approaches the mean distance between two
random points in the arena -- about 9 m in a 10 m disc.

That separates the two possibilities cleanly. If colour now decodes to metres
rather than centimetres, its features lost their spatial signal in
collection, and no change to the admission rules will bring the fields back.

    python check_channel_locality.py [env ...] [--n 4000]
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import channels as ch


def decode_error(X, xy, n, rng, chunk=512):
    """Median real-space distance to each position's feature-space neighbour."""
    idx = np.sort(rng.choice(len(xy), min(n, len(xy)), replace=False))
    P = X[idx].astype(np.float32)
    Psq = (P ** 2).sum(1)
    out = np.empty(len(idx))
    for a in range(0, len(idx), chunk):
        b = min(a + chunk, len(idx))
        d2 = Psq[a:b, None] - 2.0 * (P[a:b] @ P.T) + Psq[None, :]
        np.fill_diagonal(d2[:, a:b], np.inf)
        nn = np.argmin(d2, axis=1)
        out[a:b] = np.hypot(xy[idx[a:b], 0] - xy[idx[nn], 0],
                            xy[idx[a:b], 1] - xy[idx[nn], 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('envs', nargs='*',
                    default=['circ_lm2_r0', 'circ_lm4_r0',
                             'circ_lm8_r0', 'circ_lm12_r0'])
    ap.add_argument('--channels', default='hog,color,spatial,lidar,visual,all')
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    chans = [c.strip() for c in a.channels.split(',') if c.strip()]

    print(f'nearest feature-space neighbour, median distance in real space')
    print(f'(sample of {a.n} positions; ~9 m means no position information '
          f'in a 10 m disc)\n')
    hdr = f'{"env":16s}' + ''.join(f'{c:>10s}' for c in chans)
    print(hdr + '\n' + '-' * len(hdr))

    for e in a.envs:
        path = f'{REPO}/data/vpce/collect_data/{e}.h5'
        if not os.path.exists(path):
            print(f'{e:16s}  (no dataset)')
            continue
        blocks, xy = ch.load_channel_blocks(path, verbose=False)
        rng = np.random.default_rng(a.seed)
        row = f'{e:16s}'
        for c in chans:
            if any(k not in blocks for k in ch.CHANNEL_SETS[c]):
                row += f'{"--":>10s}'
                continue
            X = ch.assemble(blocks, ch.CHANNEL_SETS[c], normalize=True)
            med = float(np.median(decode_error(X, xy, a.n, rng)))
            row += f'{med:>9.2f}m'
            del X
        print(row, flush=True)
        del blocks


if __name__ == '__main__':
    main()
