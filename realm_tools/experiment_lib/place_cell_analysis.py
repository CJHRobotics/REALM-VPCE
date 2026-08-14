"""
place_cell_analysis.py

Reusable helpers for the notebook experiments: load a POV dataset with the
standard heading-collapsing convention, fit a visual-place-cell ensemble with
GMM clustering, and compute per-cell spatial-field metrics from population
activations.

These wrap patterns previously duplicated inside individual notebooks so new
experiments can just import them.
"""

import warnings

import numpy as np

from realm_tools.experiment_lib.loggers import PovDataset, combine_pov_observations
from realm_tools.image_lib.image_feature_lib import feature_block_sizes
from realm_tools.place_cell_lib import VisualPlaceCellEnsemble, fit_gmm


def load_selective_features(
    data_path,
    use_hog        = True,
    use_color_hist = True,
    use_spatial    = True,
    use_lidar      = False,
    n_combine      = 8,
    n_orientations = 8,
    combine_mode   = 'concat',
    lidar_heading_index = 0,
):
    """Load a subset of descriptor blocks, combined across the heading window.

    Handles both storage formats transparently:
      * new format — each descriptor stored as its own HDF5 field
        (`hog`, `color_hist`, `spatial`) plus optional `lidar`.
      * legacy format — a single `multimodal_features` field with HOG,
        color_hist and spatial concatenated in that order. Split back
        into blocks via `feature_block_sizes()`.

    Visual blocks are combined across the heading window in the same way
    as `combine_pov_observations` (mean or concat). The lidar block is
    location-tied: only the scan at heading `lidar_heading_index` is
    kept per location and appended as-is.

    Parameters
    ----------
    data_path        : str   path to the collected `.h5`
    use_hog / use_color_hist / use_spatial / use_lidar : bool  toggle blocks
    n_combine        : int   size of the heading window (default 8 = full turn)
    n_orientations   : int   POV captures per location (default 8)
    combine_mode     : {'mean', 'concat'}
    lidar_heading_index : int  which heading's lidar to keep (default 0)

    Returns
    -------
    features : (N, D) np.ndarray  combined feature vector per location
                                   (or per heading row if n_combine != n_orientations)
    xy       : (M, 2) np.ndarray  matching (x, y) positions
    """
    if combine_mode not in ('mean', 'concat'):
        raise ValueError(f"combine_mode must be 'mean' or 'concat', got {combine_mode!r}")

    ds = PovDataset.load_dataset(data_path)

    # Pull each requested visual block. New format = separate fields;
    # legacy format = slice a saved multimodal_features vector by known
    # block sizes.
    blocks = {}
    for key in ('hog', 'color_hist', 'spatial'):
        if key in ds._fields:
            blocks[key] = np.asarray(ds._fields[key])
    if not blocks and 'multimodal_features' in ds._fields:
        combined_all = np.asarray(ds.multimodal_features)
        sizes        = feature_block_sizes()
        offset       = 0
        for key in ('hog', 'color_hist', 'spatial'):
            n = sizes[key]
            blocks[key] = combined_all[:, offset:offset + n]
            offset += n

    parts = []
    if use_hog        and 'hog'        in blocks: parts.append(blocks['hog'])
    if use_color_hist and 'color_hist' in blocks: parts.append(blocks['color_hist'])
    if use_spatial    and 'spatial'    in blocks: parts.append(blocks['spatial'])

    if not parts and not use_lidar:
        raise ValueError("Enable at least one feature block (hog / color_hist / spatial / lidar).")

    xs = np.asarray(ds.x); ys = np.asarray(ds.y)
    n_loc = len(xs) // n_orientations

    # Combine visual blocks across the heading window.
    if parts:
        per_heading = np.concatenate(parts, axis=1)                  # (N * n_orient, D_vis)
        per_heading = per_heading.reshape(n_loc, n_orientations, -1)
        if n_combine == n_orientations:
            if combine_mode == 'mean':
                combined = per_heading.mean(axis=1)                   # (n_loc, D_vis)
            else:                                                     # 'concat'
                combined = per_heading.reshape(n_loc, -1)             # (n_loc, n_orient * D_vis)
            per_location = True
        else:
            raise NotImplementedError(
                'sliding-window combine is not yet supported by load_selective_features; '
                'use n_combine == n_orientations'
            )
    else:
        combined = np.empty((n_loc, 0), dtype=np.float32)
        per_location = True

    if use_lidar and 'lidar' in ds._fields:
        lidar = np.asarray(ds.lidar)
        # Support both storage layouts (see combine_pov_observations).
        if lidar.shape[0] == n_loc:
            lidar_n = lidar.astype(np.float32)
        elif lidar.shape[0] == n_loc * n_orientations:
            lidar_n = lidar.reshape(n_loc, n_orientations, -1)[:, lidar_heading_index, :] \
                          .astype(np.float32)
        else:
            raise ValueError(f'unexpected lidar shape {lidar.shape}')
        finite  = np.isfinite(lidar_n)
        clip_hi = np.nanmax(lidar_n[finite]) if finite.any() else 1.0
        lidar_n = np.where(finite, lidar_n, clip_hi).astype(np.float32)
        combined = np.concatenate([combined, lidar_n], axis=1)
    elif use_lidar:
        raise KeyError(f'use_lidar=True but no `lidar` field in {data_path}')

    xs_out = xs.reshape(n_loc, n_orientations)[:, 0]
    ys_out = ys.reshape(n_loc, n_orientations)[:, 0]
    xy = np.stack([xs_out, ys_out], axis=1)
    return combined.astype(np.float32), xy


def load_lidar_only(data_path, n_orientations=8, heading_index=0):
    """Load the lidar range image as the sole feature vector.

    The collector stores one lidar scan per location (broadcast into every
    heading row). This helper unpacks that back into one row per location by
    slicing every `n_orientations`-th record at `heading_index` (0 = the
    north-facing capture).

    Parameters
    ----------
    data_path      : str
    n_orientations : int   number of POV captures per location (default 8)
    heading_index  : int   which heading's lidar to keep (default 0 = north)

    Returns
    -------
    features : (N, D_lidar) np.ndarray  north-facing lidar per location
    xy       : (N, 2)       np.ndarray  (x, y) per location
    """
    ds       = PovDataset.load_dataset(data_path)
    lidar    = np.asarray(ds.lidar)
    xs       = np.asarray(ds.x)
    ys       = np.asarray(ds.y)
    n_loc    = len(xs) // n_orientations
    if lidar.shape[0] == n_loc:                                # new per-location layout
        pass
    elif lidar.shape[0] == n_loc * n_orientations:             # legacy broadcast layout
        lidar = lidar.reshape(n_loc, n_orientations, -1)[:, heading_index, :]
    else:
        raise ValueError(f'unexpected lidar shape {lidar.shape}')
    xs       = xs.reshape(n_loc, n_orientations)[:, heading_index]
    ys       = ys.reshape(n_loc, n_orientations)[:, heading_index]
    # Lidar returns infinity for max-range hits; clip so distances are finite.
    max_range = np.nanmax(lidar[np.isfinite(lidar)]) if np.any(np.isfinite(lidar)) else 1.0
    lidar    = np.where(np.isfinite(lidar), lidar, max_range).astype(np.float32)
    xy       = np.stack([xs, ys], axis=1)
    return lidar, xy


def load_condition(data_path, n_combine=8, combine_mode='mean'):
    """Load a PovDataset and collapse per-orientation observations.

    Parameters
    ----------
    data_path    : str  path to a `.h5` produced by `collect_data`
    n_combine    : int  window passed to `combine_pov_observations`
    combine_mode : {'mean', 'concat'} — how to combine features within the
                   window.  `'mean'` averages (rotation-invariant), `'concat'`
                   stacks in a fixed heading order (heading-preserving,
                   n_combine × the feature dim).

    Returns
    -------
    features : (N, D) np.ndarray  multimodal features per (collapsed) location
    xy       : (N, 2) np.ndarray  (x, y) position per location
    """
    ds       = PovDataset.load_dataset(data_path)
    combined = combine_pov_observations(ds, n_combine=n_combine, combine_mode=combine_mode)
    features = np.asarray(combined['multimodal_features'])
    xy       = np.stack([combined['x'], combined['y']], axis=1)
    return features, xy


def fit_visual_place_cells(features, n_cells, reg_covar=1e-3, random_state=0):
    """Fit a GMM and wrap it in a VisualPlaceCellEnsemble.

    Silences the sklearn convergence warnings that are common at high K on
    the high-dimensional VPCE features.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        centers, radii = fit_gmm(
            features,
            n_components=n_cells,
            reg_covar=reg_covar,
            random_state=random_state,
        )
    return VisualPlaceCellEnsemble(centers, radii)


def field_metrics(activations, xy, arena_radius=None):
    """Per-cell spatial-field metrics via Monte-Carlo integration of activity.

    Treats each sampled `(x, y)` as a bin of equal area `A_total / N`, so
    integrating the activation over the arena is just a sum:

        field_area   = Σ_i activityᵢ · (A_total / N) = mean(activity) · A_total
        field_radius = sqrt(field_area / π)

    The field centre is the argmax location — the `(x, y)` sample with the
    highest activation for that cell.  Assumes the samples cover the arena
    roughly uniformly (otherwise the equal-area weighting is biased).

    Parameters
    ----------
    activations  : (N, K) np.ndarray
    xy           : (N, 2) np.ndarray  — arena positions
    arena_radius : float or None — radius (m) of the (circular) arena.
                   If None, inferred as `max(|xy|)`.

    Returns
    -------
    dict:
        field_area    : (K,)   integrated activity over the arena     (m²)
        field_radius  : (K,)   sqrt(area / π) — equivalent-disc radius (m)
        mean_activity : (K,)   mean activation over samples (= area / A_total)
        peak_xy       : (K, 2) argmax location for each cell            (m)
        peak_value    : (K,)   activation value at the argmax location
        arena_radius  : (scalar) R used
        arena_area    : (scalar) π R²
        sample_area   : (scalar) A_total / N — area contributed by each sample
    """
    a  = np.asarray(activations, dtype=np.float64)                  # (N, K)
    xy = np.asarray(xy,          dtype=np.float64)                  # (N, 2)
    N, K = a.shape

    R           = float(arena_radius) if arena_radius is not None \
                  else float(np.hypot(xy[:, 0], xy[:, 1]).max())
    arena_area  = float(np.pi * R ** 2)
    sample_area = arena_area / max(N, 1)

    mean_activity = a.mean(axis=0)                                   # (K,)
    field_area    = mean_activity * arena_area                       # (K,)
    field_radius  = np.sqrt(field_area / np.pi)                      # (K,)

    peak_idx      = np.argmax(a, axis=0)                             # (K,)
    peak_xy       = xy[peak_idx]                                     # (K, 2)
    peak_value    = a[peak_idx, np.arange(K)]                        # (K,)

    return {
        'field_area'   : field_area,
        'field_radius' : field_radius,
        'mean_activity': mean_activity,
        'peak_xy'      : peak_xy,
        'peak_value'   : peak_value,
        'arena_radius' : R,
        'arena_area'   : arena_area,
        'sample_area'  : sample_area,
    }
