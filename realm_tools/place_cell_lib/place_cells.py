"""
place_cells.py

Vectorised place cell ensembles for the REALM / VPCE framework.

Two ensemble types share the same RBF core:

    SpatialPlaceCellEnsemble   — cells defined in 2D (x, y) space
                                  used by navigation / RL agents

    VisualPlaceCellEnsemble    — cells defined in high-dimensional feature
                                  space (the VPCE model)

Both expose the same interface:

    ensemble.activate(inputs)  →  np.ndarray  shape (N, K)

where N is the number of observations and K is the number of place cells.
Activations are computed with a single vectorised pass — no per-cell Python
loops, no mutable state stored inside cell objects.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def rbf_activations(features, centers, radii):
    """
    Compute the row-normalised RBF activation matrix.

    Each row i contains the normalised activation of all K place cells
    for observation i.  Normalisation ensures the population vector sums
    to 1 at every observation, giving a probability-like encoding.

    Uses the identity  ||a-b||² = ||a||² - 2a·b + ||b||²  to compute all
    pairwise distances via matrix multiplication, avoiding the O(N·K·D)
    memory cost of the naive broadcast approach.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Observation feature vectors.  For SpatialPlaceCellEnsemble D=2.
    centers  : np.ndarray, shape (K, D)
        Place cell centroids in feature space.
    radii    : np.ndarray, shape (K,)
        RBF sigma (receptive field width) per cell.

    Returns
    -------
    np.ndarray, shape (N, K)  —  row-normalised activations in [0, 1].
    """
    features = np.atleast_2d(features)
    f_sq     = np.sum(features ** 2, axis=1, keepdims=True)   # (N, 1)
    c_sq     = np.sum(centers  ** 2, axis=1)                   # (K,)
    fc       = features @ centers.T                             # (N, K)
    dist_sq  = np.maximum(f_sq - 2 * fc + c_sq, 0)            # (N, K)
    raw      = np.exp(-dist_sq / (2 * radii ** 2))             # (N, K)
    row_sums = raw.sum(axis=1, keepdims=True)
    return raw / np.where(row_sums == 0, 1, row_sums)


# ---------------------------------------------------------------------------
# Ensemble classes
# ---------------------------------------------------------------------------

class SpatialPlaceCellEnsemble:
    """
    Place cell ensemble defined in 2D (x, y) physical space.

    Replaces the old PlaceCell / PlaceCellNetwork pair with a fully
    vectorised implementation.  No per-cell objects, no mutable activity
    state — activation is computed on demand and returned as a numpy array.

    Parameters
    ----------
    centers : array-like, shape (K, 2)   — cell centres in metres
    radii   : array-like, shape (K,)     — RBF sigma per cell in metres

    Example
    -------
    >>> ensemble = SpatialPlaceCellEnsemble(centers, radii)
    >>> act = ensemble.activate(np.array([[x, y]]))   # (1, K)
    >>> act = ensemble.activate(positions)             # (N, K)
    """

    def __init__(self, centers, radii):
        self.centers = np.asarray(centers, dtype=np.float64)   # (K, 2)
        self.radii   = np.asarray(radii,   dtype=np.float64)   # (K,)

    @property
    def n_cells(self):
        return len(self.radii)

    def activate(self, positions):
        """
        Compute normalised activations for one or more (x, y) positions.

        Parameters
        ----------
        positions : array-like, shape (N, 2) or (2,)

        Returns
        -------
        np.ndarray, shape (N, K)
        """
        return rbf_activations(
            np.asarray(positions, dtype=np.float64),
            self.centers,
            self.radii,
        )


class VisualPlaceCellEnsemble:
    """
    Visual Place Cell (VPCE) ensemble defined in high-dimensional feature space.

    Replaces the old VisiualPlaceCell / VisualPlaceCellNetwork pair.
    Centers and radii are produced by fit_kmeans / fit_gmm in clustering.py
    and are stored in the PlaceCellEnsemble HDF5 model file.

    Parameters
    ----------
    centers : array-like, shape (K, D)   — cluster centroids in feature space
    radii   : array-like, shape (K,)     — RBF sigma per cell

    Example
    -------
    >>> ensemble = VisualPlaceCellEnsemble(centers, radii)
    >>> act = ensemble.activate(features_pca)   # (N, K)
    >>> act = ensemble.activate(single_obs)     # (1, K)
    """

    def __init__(self, centers, radii):
        self.centers = np.asarray(centers, dtype=np.float64)   # (K, D)
        self.radii   = np.asarray(radii,   dtype=np.float64)   # (K,)

    @property
    def n_cells(self):
        return len(self.radii)

    @property
    def feature_dim(self):
        return self.centers.shape[1]

    def activate(self, features):
        """
        Compute normalised activations for one or more feature vectors.

        Parameters
        ----------
        features : array-like, shape (N, D) or (D,)

        Returns
        -------
        np.ndarray, shape (N, K)
        """
        return rbf_activations(
            np.asarray(features, dtype=np.float64),
            self.centers,
            self.radii,
        )
