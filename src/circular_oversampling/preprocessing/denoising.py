"""
Denoising wrappers for removing noisy minority samples before oversampling.

Supports three cleaning strategies accessed through a unified
:func:`denoise` interface:

* **Tomek links** -- remove majority-side samples that form Tomek links with
  minority samples (via ``imblearn.under_sampling.TomekLinks``).
* **Edited Nearest Neighbours (ENN)** -- remove samples misclassified by
  their k-nearest neighbours (via
  ``imblearn.under_sampling.EditedNearestNeighbours``).
* **DBSCAN** -- fit :class:`sklearn.cluster.DBSCAN` on the minority class
  and discard minority points flagged as noise (label ``-1``).
"""

import numpy as np


def _denoise_tomek(X, y):
    """Remove Tomek links using *imblearn*.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)

    Returns
    -------
    X_clean : ndarray
    y_clean : ndarray
    """
    from imblearn.under_sampling import TomekLinks

    tl = TomekLinks()
    X_clean, y_clean = tl.fit_resample(X, y)
    return X_clean, y_clean


def _denoise_enn(X, y):
    """Remove samples misclassified by their nearest neighbours.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)

    Returns
    -------
    X_clean : ndarray
    y_clean : ndarray
    """
    from imblearn.under_sampling import EditedNearestNeighbours

    enn = EditedNearestNeighbours()
    X_clean, y_clean = enn.fit_resample(X, y)
    return X_clean, y_clean


def _denoise_dbscan(X, y, eps=0.5, min_samples=5):
    """Remove minority-class noise detected by DBSCAN.

    DBSCAN is fitted on the *minority* subset only.  Minority points whose
    DBSCAN label is ``-1`` (noise) are dropped; all majority points are
    kept unchanged.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    eps : float, default=0.5
        DBSCAN neighbourhood radius.
    min_samples : int, default=5
        DBSCAN minimum samples per core point.

    Returns
    -------
    X_clean : ndarray
    y_clean : ndarray
    """
    from sklearn.cluster import DBSCAN

    classes, counts = np.unique(y, return_counts=True)
    min_lab = classes[np.argmin(counts)]

    minority_mask = (y == min_lab)
    majority_mask = ~minority_mask

    X_min = X[minority_mask]
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = db.fit_predict(X_min)

    # Keep minority points that are *not* noise.
    keep_minority = (db_labels != -1)
    X_min_clean = X_min[keep_minority]
    y_min_clean = y[minority_mask][keep_minority]

    # Recombine with all majority points.
    X_clean = np.vstack([X[majority_mask], X_min_clean])
    y_clean = np.concatenate([y[majority_mask], y_min_clean])

    return X_clean, y_clean


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

_METHODS = {
    "tomek": _denoise_tomek,
    "enn": _denoise_enn,
    "dbscan": _denoise_dbscan,
}


def denoise(X, y, method=None):
    """Apply denoising to remove noisy minority samples.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target labels.
    method : {None, "tomek", "enn", "dbscan"}, default=None
        Denoising strategy.  ``None`` is a no-op that returns the input
        arrays unchanged.

    Returns
    -------
    X_clean : ndarray
        Cleaned feature matrix (may be smaller than the input).
    y_clean : ndarray
        Corresponding cleaned target labels.

    Raises
    ------
    ValueError
        If *method* is not one of the supported values.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if method is None:
        return X.copy(), y.copy()

    method = method.lower().strip()
    if method not in _METHODS:
        raise ValueError(
            f"Unknown denoising method '{method}'. "
            f"Supported values: {list(_METHODS.keys())}."
        )

    return _METHODS[method](X, y)
