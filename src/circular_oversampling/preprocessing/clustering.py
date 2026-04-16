"""
Unified clustering interface for minority-class seed selection.

Provides a single entry-point :func:`cluster_minority` that dispatches to
either *K-Means* or *Hierarchical Agglomerative Clustering* (HAC) and
returns cluster labels together with cluster centres.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering


def _compute_centers(X, labels):
    """Compute cluster centres as the mean of assigned points.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points.
    labels : ndarray of shape (n_samples,)
        Cluster assignments.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        Mean of each cluster.
    """
    unique_labels = np.sort(np.unique(labels))
    centers = np.empty((len(unique_labels), X.shape[1]), dtype=np.float64)
    for i, lab in enumerate(unique_labels):
        centers[i] = X[labels == lab].mean(axis=0)
    return centers


def cluster_minority(X, method="kmeans", n_clusters=3, random_state=42):
    """Cluster minority-class samples.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix of *minority* samples only.
    method : {"kmeans", "hac"}, default="kmeans"
        Clustering algorithm to use.

        * ``"kmeans"`` -- :class:`sklearn.cluster.KMeans`.
        * ``"hac"`` -- :class:`sklearn.cluster.AgglomerativeClustering`
          (Ward linkage).

    n_clusters : int, default=3
        Number of clusters to form.
    random_state : int, default=42
        Random seed (used by K-Means; HAC is deterministic).

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster label for each sample.
    centers : ndarray of shape (n_clusters, n_features)
        Cluster centres (K-Means native centres or HAC mean centres).

    Raises
    ------
    ValueError
        If *method* is not one of the supported values.
    """
    X = np.asarray(X, dtype=np.float64)
    method = method.lower().strip()

    # Clamp n_clusters to the number of available samples.
    n_clusters = min(n_clusters, X.shape[0])

    if method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )
        model.fit(X)
        return model.labels_, model.cluster_centers_

    if method == "hac":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        )
        labels = model.fit_predict(X)
        centers = _compute_centers(X, labels)
        return labels, centers

    raise ValueError(
        f"Unknown clustering method '{method}'. "
        f"Supported values: 'kmeans', 'hac'."
    )
