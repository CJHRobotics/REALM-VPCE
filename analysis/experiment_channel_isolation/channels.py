"""Feature-channel loading for the channel-isolation experiment.

Each descriptor channel is loaded as its own block so the agglomeration can
be run on one channel at a time (hog / color / spatial / lidar) as well as
on combinations. Two things differ from `load_selective_features`:

1. **Range-limited lidar.** Real range perception is bounded. Any beam
   returning further than `lidar_max_range` — and any non-finite beam,
   which is the sensor reporting "no hit" — is replaced by a sentinel and
   flagged in a companion binary channel. The clustering therefore sees
   "I cannot see that far" as an explicit signal rather than as a very
   large number.

2. **Per-block scaling.** Channels have wildly different dimensionalities
   and magnitudes (hog 30240-d, lidar 720-d). Left raw, whichever block has
   the largest norm dominates every distance. Each block is divided by its
   RMS row norm so that all blocks contribute equally to squared distance,
   which is what makes the isolated-channel and combined runs comparable.

Visual blocks are concatenated across all 8 headings (fixed-heading 360°
view), matching the convention used elsewhere in the project.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# NOTE: feature_block_sizes lives in realm_tools.image_lib.image_feature_lib,
# which imports cv2 at module level. It is only needed for datasets in the
# legacy single-'multimodal_features' layout, so it is imported lazily inside
# that branch. Importing it here would make every run of this analysis depend
# on a working OpenCV build — which is how a NumPy upgrade that broke cv2's
# ABI took down a run that never touches an image.


VISUAL_KEYS = ('hog', 'color_hist', 'spatial')

# Channel groupings run by the experiment. Each entry names the blocks
# that get concatenated for that configuration.
CHANNEL_SETS = {
    'hog'    : ('hog',),
    'color'  : ('color_hist',),
    'spatial': ('spatial',),
    'lidar'  : ('lidar',),
    'visual' : ('hog', 'color_hist', 'spatial'),
    'all'    : ('hog', 'color_hist', 'spatial', 'lidar'),
}


def mask_lidar(lidar, max_range=5.0, sentinel=-1.0, with_mask_channel=True):
    """Bound the agent's distance perception.

    Beams returning beyond `max_range`, and beams that are non-finite
    (the sensor's "no hit within range" code), are set to `sentinel`.

    Parameters
    ----------
    lidar             : (n, n_beams) raw range image, metres
    max_range         : float  perception limit
    sentinel          : float  value written for out-of-range beams
    with_mask_channel : bool   append a binary in-range indicator per beam,
                               doubling the block width

    Returns
    -------
    (n, n_beams) or (n, 2 * n_beams) float32

    Notes
    -----
    The sentinel alone puts a discontinuity in the feature space — a wall at
    4.99 m and a wall at 5.01 m sit `max_range - sentinel` apart. The
    companion mask channel makes that jump interpretable instead of
    accidental: two locations that both see nothing within range agree on
    the mask, whatever their sentinel values do.
    """
    lidar = np.asarray(lidar, dtype=np.float32)
    in_range = np.isfinite(lidar) & (lidar <= max_range)
    out = np.where(in_range, lidar, np.float32(sentinel)).astype(np.float32)
    if not with_mask_channel:
        return out
    return np.concatenate([out, in_range.astype(np.float32)], axis=1)


def load_channel_blocks(data_path,
                        n_orientations   = 8,
                        lidar_max_range  = 5.0,
                        lidar_sentinel   = -1.0,
                        lidar_mask_channel = True,
                        lidar_heading_index = 0,
                        verbose = True):
    """Load every feature channel separately, plus positions.

    Returns
    -------
    blocks : dict  name -> (n_loc, d) float32, one entry per available channel
    xy     : (n_loc, 2) float32 positions
    """
    import h5py

    blocks = {}
    with h5py.File(data_path, 'r') as f:
        keys = set(f.keys())
        xs = np.asarray(f['x'][:], dtype=np.float64)
        ys = np.asarray(f['y'][:], dtype=np.float64)
        n_loc = len(xs) // n_orientations

        # --- visual blocks -------------------------------------------------
        if VISUAL_KEYS[0] in keys:
            raw_visual = {k: np.asarray(f[k][:], dtype=np.float32)
                          for k in VISUAL_KEYS if k in keys}
        elif 'multimodal_features' in keys:
            # Legacy layout: one concatenated vector, split by known widths.
            from realm_tools.image_lib.image_feature_lib import feature_block_sizes
            combined_all = np.asarray(f['multimodal_features'][:], dtype=np.float32)
            sizes, offset = feature_block_sizes(), 0
            raw_visual = {}
            for k in VISUAL_KEYS:
                n = sizes[k]
                raw_visual[k] = combined_all[:, offset:offset + n]
                offset += n
        else:
            raw_visual = {}

        for name, arr in raw_visual.items():
            # (n_loc * n_orient, d) -> (n_loc, n_orient * d), fixed heading order
            blocks[name] = arr.reshape(n_loc, n_orientations, -1).reshape(n_loc, -1)

        # --- lidar ---------------------------------------------------------
        if 'lidar' in keys:
            lidar = np.asarray(f['lidar'][:], dtype=np.float32)
            if lidar.shape[0] == n_loc:
                pass                                    # one scan per location
            elif lidar.shape[0] == n_loc * n_orientations:
                lidar = lidar.reshape(n_loc, n_orientations, -1)[:, lidar_heading_index, :]
            else:
                raise ValueError(f'unexpected lidar shape {lidar.shape}')
            if verbose:
                fin = np.isfinite(lidar)
                beyond = float((fin & (lidar > lidar_max_range)).mean())
                print(f'  lidar: {100*(~fin).mean():.1f}% non-finite, '
                      f'{100*beyond:.1f}% beyond {lidar_max_range} m '
                      f'-> {100*(beyond + (~fin).mean()):.1f}% masked to {lidar_sentinel}')
            blocks['lidar'] = mask_lidar(lidar,
                                         max_range=lidar_max_range,
                                         sentinel=lidar_sentinel,
                                         with_mask_channel=lidar_mask_channel)

    xy = np.stack([xs.reshape(n_loc, n_orientations)[:, 0],
                   ys.reshape(n_loc, n_orientations)[:, 0]], axis=1).astype(np.float32)

    if verbose:
        print(f'  locations: {n_loc}')
        for k, v in blocks.items():
            print(f'  block {k:11s} {v.shape}')
    return blocks, xy


def assemble(blocks, names, normalize=True):
    """Concatenate the named blocks into one feature matrix.

    With `normalize`, each block is divided by its RMS row norm first, so
    every block contributes equal expected squared distance regardless of
    its width or units. Without it, hog (30240-d) would swamp lidar (720-d)
    in any combined configuration.
    """
    parts = []
    for name in names:
        if name not in blocks:
            raise KeyError(f'channel {name!r} not present in this dataset')
        b = blocks[name].astype(np.float32, copy=True)
        if normalize:
            # Accumulated in chunks: casting a 30240-d block to float64 in
            # one go would double a multi-GB array to compute one scalar.
            total, n = 0.0, b.shape[0]
            for s in range(0, n, 2048):
                total += float((b[s:s + 2048].astype(np.float64) ** 2).sum())
            rms = float(np.sqrt(total / n)) if n else 0.0
            if rms > 0:
                b /= np.float32(rms)
        parts.append(b)
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
