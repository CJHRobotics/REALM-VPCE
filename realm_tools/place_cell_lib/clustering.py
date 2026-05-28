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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(features)
    centers = kmeans.cluster_centers_
    labels  = kmeans.labels_
    radii = np.array([
        np.mean(np.linalg.norm(features[labels == i] - centers[i], axis=1))
        for i in range(n_clusters)
    ])
    return centers, radii
def fit_gmm(features, n_components):
    """
    Cluster feature vectors using a Gaussian Mixture Model.

    Args:
        features     (np.ndarray): shape (N, feature_dim)
        n_components (int):        number of place cells to form

    Returns:
        centers (np.ndarray): shape (n_components, feature_dim)
        radii   (np.ndarray): shape (n_components,)
                              isotropic sigma approximated from the diagonal covariance
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(features)
    centers = gmm.means_
    radii = np.array([
        np.sqrt(np.mean(np.diag(gmm.covariances_[i])))
        for i in range(n_components)
    ])
    return centers, radii
