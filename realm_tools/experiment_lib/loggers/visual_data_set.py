import os
import numpy as np
import h5py


class PovDataset:
    """
    Generic per-observation dataset. Each observation is a set of named
    fields (e.g. x, y, theta, multimodal_features, landmark_mask, ...).
    Fields are not fixed up front — pass whatever fields you collect to
    add_observation, and they'll be saved/loaded under those same names.
    """

    def __init__(self):
        self._fields = {}

    def add_observation(self, **fields):
        """
        Adds a new datapoint to the dataset. Accepts arbitrary named fields,
        e.g. add_observation(x=x, y=y, theta=theta, multimodal_features=feat).
        """
        for name, value in fields.items():
            self._fields.setdefault(name, []).append(value)

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name)

    def save_dataset(self, filename):
        """
        Save the dataset to an HDF5 file, one dataset per field.
        Creates the directory if it does not exist.
        """
        if not filename.endswith('.h5'):
            filename += '.h5'
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with h5py.File(filename, 'w') as f:
            for name, values in self._fields.items():
                f.create_dataset(name, data=np.array(values), compression='gzip')

        n = len(next(iter(self._fields.values()), []))
        print(f"Dataset saved to {filename}  ({n} observations)")

    @staticmethod
    def load_dataset(filename):
        """
        Load a dataset from an HDF5 file.
        """
        if not filename.endswith('.h5'):
            filename += '.h5'
        dataset = PovDataset()
        with h5py.File(filename, 'r') as f:
            for name in f.keys():
                dataset._fields[name] = list(f[name][:])
        return dataset


def combine_pov_observations(dataset, n_combine, n_orientations=8, combine_mode='mean'):
    """
    Combine per-orientation observations at each (x, y) location into datapoints.

    Assumes `dataset` was collected with `n_orientations` POV captures per
    location, stored as consecutive rows in a fixed orientation order (as
    produced by collect_data.py).

    Parameters
    ----------
    dataset        : PovDataset
    n_combine      : int — number of adjacent orientations (a centered,
                     circular window) to combine into each datapoint.
                     Must be odd and <= n_orientations, or equal to
                     n_orientations to collapse each location to a single
                     datapoint (like the original all-headings combine).
    n_orientations : int — number of POV captures per location (default 8)
    combine_mode   : {'mean', 'concat'}
                     'mean'   → average feature vectors in the window (default).
                                Rotation-invariant, so locations that look
                                similar under a heading rotation alias together.
                     'concat' → concatenate feature vectors in a fixed
                                orientation order.  Preserves heading structure
                                so rotationally-equivalent locations become
                                distinguishable, at the cost of `n_combine ×`
                                the feature dimensionality.

    Returns
    -------
    dict with keys: x, y, theta, multimodal_features, landmark_mask, landmark_azimuths
    landmark_mask is the union (logical OR) of the windowed orientations.
    landmark_azimuths is taken from the center orientation only (azimuths are
    tied to a single viewpoint and aren't meaningful to average). When
    n_combine == n_orientations, theta and landmark_azimuths are NaN since no
    single orientation applies.
    """
    if n_combine != n_orientations and n_combine % 2 == 0:
        raise ValueError("n_combine must be odd (a centered window) or equal to n_orientations")
    if combine_mode not in ('mean', 'concat'):
        raise ValueError(f"combine_mode must be 'mean' or 'concat', got {combine_mode!r}")

    features = np.array(dataset.multimodal_features)
    xs       = np.array(dataset.x)
    ys       = np.array(dataset.y)
    thetas   = np.array(dataset.theta)
    masks    = np.array(dataset.landmark_mask)
    azimuths = np.array(dataset.landmark_azimuths)

    n_locations = len(xs) // n_orientations
    features = features.reshape(n_locations, n_orientations, -1)
    xs       = xs.reshape(n_locations, n_orientations)[:, 0]
    ys       = ys.reshape(n_locations, n_orientations)[:, 0]
    thetas   = thetas.reshape(n_locations, n_orientations)
    masks    = masks.reshape(n_locations, n_orientations, -1)
    azimuths = azimuths.reshape(n_locations, n_orientations, -1)

    # Optional lidar field: kept per-location (single north-facing scan),
    # not combined across headings. Handles two storage layouts:
    #   * new  — (n_locations, D_lidar): one scan per position
    #   * old  — (n_locations * n_orientations, D_lidar): scan broadcast
    #            to every heading row (kept for backward compatibility).
    lidar_north = None
    if 'lidar' in dataset._fields:
        lidar_all = np.array(dataset.lidar)
        if lidar_all.shape[0] == n_locations:
            lidar_north = lidar_all
        elif lidar_all.shape[0] == n_locations * n_orientations:
            lidar_north = lidar_all.reshape(n_locations, n_orientations, -1)[:, 0, :]
        else:
            raise ValueError(f'unexpected lidar shape {lidar_all.shape} — '
                             f'expected ({n_locations}, D) or ({n_locations*n_orientations}, D)')

    if n_combine == n_orientations:
        if combine_mode == 'mean':
            combined = features.mean(axis=1)                       # (n_locations, D)
        else:  # 'concat' — fixed orientation order (0, 1, ..., n_orientations - 1)
            combined = features.reshape(n_locations, n_orientations * features.shape[-1])
        if lidar_north is not None:
            combined = np.concatenate([combined, lidar_north], axis=-1)
        return {
            'x': xs,
            'y': ys,
            'theta': np.full(n_locations, np.nan),
            'multimodal_features': combined,
            'landmark_mask': masks.max(axis=1),
            'landmark_azimuths': np.full_like(azimuths[:, 0], np.nan),
        }

    half = n_combine // 2
    window_offsets = np.arange(-half, half + 1)
    window_indices = (np.arange(n_orientations)[:, None] + window_offsets[None, :]) % n_orientations

    windowed_features = features[:, window_indices]                # (n_loc, n_orient, n_combine, D)
    if combine_mode == 'mean':
        combined_features = windowed_features.mean(axis=2)         # (n_loc, n_orient, D)
    else:  # 'concat' — heading-centred window, order (-half, ..., +half)
        combined_features = windowed_features.reshape(
            n_locations, n_orientations, n_combine * features.shape[-1]
        )
    combined_mask = masks[:, window_indices].max(axis=2)           # (n_loc, n_orient, n_landmarks)

    if lidar_north is not None:
        # Broadcast the location-tied north-facing scan across all heading
        # rows so every row carries the same lidar block.
        lidar_broadcast = np.broadcast_to(
            lidar_north[:, None, :],
            (n_locations, n_orientations, lidar_north.shape[-1]),
        )
        combined_features = np.concatenate([combined_features, lidar_broadcast], axis=-1)

    return {
        'x': np.repeat(xs, n_orientations),
        'y': np.repeat(ys, n_orientations),
        'theta': thetas.reshape(-1),
        'multimodal_features': combined_features.reshape(n_locations * n_orientations, -1),
        'landmark_mask': combined_mask.reshape(n_locations * n_orientations, -1),
        'landmark_azimuths': azimuths.reshape(n_locations * n_orientations, -1),
    }
