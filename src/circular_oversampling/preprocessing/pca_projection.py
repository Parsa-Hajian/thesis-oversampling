"""
PCA-based dimensionality reduction for the circular-oversampling pipeline.

When ``use_pca=True`` (the default) and the input data has more than two
features, the oversampling geometry operates in a 2-D PCA subspace and
synthetic points are lifted back via the inverse transform.

When ``use_pca=False`` all geometry is performed directly in the original
d-dimensional feature space: circles become d-dimensional hyperballs and all
distance / clustering operations use the full feature vector.  This avoids
the information loss inherent in projecting to two principal components but
requires the geometric primitives (ball sampling, vMF directional bias) to
operate in d dimensions.
"""

import numpy as np
from sklearn.decomposition import PCA


def to_2d(X, random_state=0, use_pca=True):
    """Project feature matrix *X* into the working space.

    When ``use_pca=True`` (default) the data is projected to a 2-D PCA
    subspace.  When ``use_pca=False`` the data is returned as-is and all
    downstream geometry operates in the original d-dimensional space.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input feature matrix.
    random_state : int, default=0
        Random seed forwarded to :class:`sklearn.decomposition.PCA`.
    use_pca : bool, default=True
        If ``False``, skip PCA projection entirely.

    Returns
    -------
    X_proj : ndarray of shape (n_samples, 2) if use_pca else (n_samples, d)
        The working-space representation.
    pca_model : PCA or None
        The fitted PCA transformer, or ``None`` when PCA was not applied
        (either because use_pca=False or the input was already 2-D).
    """
    X = np.asarray(X, dtype=np.float64)

    if not use_pca:
        return X.copy(), None

    if X.shape[1] == 2:
        return X.copy(), None

    pca = PCA(n_components=2, random_state=random_state)
    X2d = pca.fit_transform(X)
    return X2d, pca


def from_2d(X_proj, pca_model):
    """Map working-space points back to the original feature space.

    Parameters
    ----------
    X_proj : ndarray of shape (n_samples, d_proj)
        Points in the working space (2-D PCA or original d-D).
    pca_model : PCA or None
        The fitted PCA model returned by :func:`to_2d`.  When ``None`` the
        data is returned unchanged (already in original space or use_pca=False).

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Points in the original feature space.
    """
    X_proj = np.asarray(X_proj, dtype=np.float64)

    if pca_model is None:
        return X_proj.copy()

    return pca_model.inverse_transform(X_proj)
