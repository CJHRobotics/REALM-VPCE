import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
def fit_kmeans(features, n_clusters):
    """
    Cluster feature vectors using k-means.

    Args:
        features   (np.ndarray): shape (N, feature_dim)
        n_clusters (int):        number of place cells to form

    Returns:
        centers (np.ndarray): shape (n_clusters, feature_dim)
        radii   (np.ndarray): shape (n_clusters,)
                              mean distance from each point to its assigned center
    """
    features = np.asarray(features, dtype=np.float64)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(features)
    centers = kmeans.cluster_centers_
    labels  = kmeans.labels_
    radii = np.array([
        np.max(np.linalg.norm(features[labels == i] - centers[i], axis=1))
        for i in range(n_clusters)
    ])
    return centers, radii
def fit_gmm(features, n_components, reg_covar=1e-3):
    """
    Cluster feature vectors using a Gaussian Mixture Model.

    Args:
        features     (np.ndarray): shape (N, feature_dim)
        n_components (int):        number of place cells to form
        reg_covar    (float):      regularisation added to diagonal covariance to
                                   prevent component collapse on high-dimensional or
                                   float32 data.  Increase (e.g. 1e-2) if fitting
                                   raises ill-defined covariance errors. Default 1e-3.

    Returns:
        centers (np.ndarray): shape (n_components, feature_dim)
        radii   (np.ndarray): shape (n_components,)
    """
    # float64 is required for numerical stability in high-dimensional GMM fitting
    features = np.asarray(features, dtype=np.float64)

    gmm = GaussianMixture(n_components=n_components, random_state=42,
                          covariance_type='diag', reg_covar=reg_covar)
    gmm.fit(features)
    centers = gmm.means_
    labels  = gmm.predict(features)
    radii = np.array([
        np.max(np.linalg.norm(features[labels == i] - centers[i], axis=1))
        if np.any(labels == i) else 1.0
        for i in range(n_components)
    ])
    return centers, radii
