"""
Wrappers for baseline oversampling methods from **imbalanced-learn**.

Every baseline inherits from :class:`~src.core.base.BaseOversampler` so that
the evaluation pipeline can treat them identically to the project's custom
circular oversamplers.  Each wrapper delegates to the corresponding
``imblearn`` class while exposing the same ``fit_resample(X, y)`` interface.

Registered baselines
--------------------
=================  ==========================================
Key                Method
=================  ==========================================
``"none"``         No oversampling (identity pass-through)
``"ros"``          Random over-sampling
``"smote"``        SMOTE
``"borderline"``   Borderline-SMOTE (kind="borderline-1")
``"adasyn"``       ADASYN
``"svm_smote"``    SVM-SMOTE
``"kmeans_smote"`` KMeans-SMOTE
=================  ==========================================
"""

import numpy as np
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)

from src.core.base import BaseOversampler


# ---------------------------------------------------------------------------
# Concrete baseline classes
# ---------------------------------------------------------------------------

class NoOversampling(BaseOversampler):
    """Identity pass-through -- returns the data unchanged.

    Useful as a baseline to measure classifier performance without any
    oversampling at all.
    """

    def fit_resample(self, X, y):
        """Return copies of *X* and *y* without modification."""
        return np.array(X, dtype=np.float64, copy=True), np.array(y, copy=True)


class RandomOversampling(BaseOversampler):
    """Random over-sampling with replacement.

    Duplicates existing minority samples until the desired class balance is
    reached.

    Parameters
    ----------
    sampling_strategy : float, default=1.0
        Desired minority-to-majority ratio after resampling.
    random_state : int or None, default=42
    """

    def fit_resample(self, X, y):
        sampler = RandomOverSampler(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )
        return sampler.fit_resample(X, y)


class SMOTEOversampling(BaseOversampler):
    """Synthetic Minority Over-sampling Technique (SMOTE).

    Parameters
    ----------
    k_neighbors : int, default=5
    sampling_strategy : float, default=1.0
    random_state : int or None, default=42
    """

    def __init__(self, k_neighbors=5, sampling_strategy=1.0, random_state=42):
        super().__init__(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        sampler = SMOTE(
            k_neighbors=self.k_neighbors,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )
        return sampler.fit_resample(X, y)

    def get_params(self):
        params = super().get_params()
        params["k_neighbors"] = self.k_neighbors
        return params


class BorderlineSMOTEOversampling(BaseOversampler):
    """Borderline-SMOTE (kind 1).

    Focuses synthetic sample generation on minority instances near the
    decision boundary.

    Parameters
    ----------
    k_neighbors : int, default=5
    m_neighbors : int, default=10
    sampling_strategy : float, default=1.0
    random_state : int or None, default=42
    """

    def __init__(
        self, k_neighbors=5, m_neighbors=10, sampling_strategy=1.0,
        random_state=42
    ):
        super().__init__(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors

    def fit_resample(self, X, y):
        sampler = BorderlineSMOTE(
            k_neighbors=self.k_neighbors,
            m_neighbors=self.m_neighbors,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            kind="borderline-1",
        )
        return sampler.fit_resample(X, y)

    def get_params(self):
        params = super().get_params()
        params["k_neighbors"] = self.k_neighbors
        params["m_neighbors"] = self.m_neighbors
        return params


class ADASYNOversampling(BaseOversampler):
    """Adaptive Synthetic Sampling (ADASYN).

    Generates more synthetic samples for minority instances that are harder
    to learn.

    Parameters
    ----------
    n_neighbors : int, default=5
    sampling_strategy : float, default=1.0
    random_state : int or None, default=42
    """

    def __init__(self, n_neighbors=5, sampling_strategy=1.0, random_state=42):
        super().__init__(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
        self.n_neighbors = n_neighbors

    def fit_resample(self, X, y):
        sampler = ADASYN(
            n_neighbors=self.n_neighbors,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )
        return sampler.fit_resample(X, y)

    def get_params(self):
        params = super().get_params()
        params["n_neighbors"] = self.n_neighbors
        return params


class SVMSMOTEOversampling(BaseOversampler):
    """SVM-SMOTE -- SMOTE guided by SVM support vectors.

    Generates synthetic samples along the directions defined by support
    vectors of an SVM trained on the original data.

    Parameters
    ----------
    k_neighbors : int, default=5
    m_neighbors : int, default=10
    sampling_strategy : float, default=1.0
    random_state : int or None, default=42
    """

    def __init__(
        self, k_neighbors=5, m_neighbors=10, sampling_strategy=1.0,
        random_state=42
    ):
        super().__init__(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors

    def fit_resample(self, X, y):
        sampler = SVMSMOTE(
            k_neighbors=self.k_neighbors,
            m_neighbors=self.m_neighbors,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )
        return sampler.fit_resample(X, y)

    def get_params(self):
        params = super().get_params()
        params["k_neighbors"] = self.k_neighbors
        params["m_neighbors"] = self.m_neighbors
        return params


class KMeansSMOTEOversampling(BaseOversampler):
    """KMeans-SMOTE -- cluster-based SMOTE.

    Applies k-means clustering before generating synthetic samples, avoiding
    noisy regions.

    Parameters
    ----------
    k_neighbors : int, default=5
    kmeans_estimator : int, default=3
        Number of clusters for KMeans.
    sampling_strategy : float, default=1.0
    random_state : int or None, default=42
    """

    def __init__(
        self, k_neighbors=5, kmeans_estimator=3, sampling_strategy=1.0,
        random_state=42
    ):
        super().__init__(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
        self.k_neighbors = k_neighbors
        self.kmeans_estimator = kmeans_estimator

    def fit_resample(self, X, y):
        sampler = KMeansSMOTE(
            k_neighbors=self.k_neighbors,
            kmeans_estimator=self.kmeans_estimator,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            cluster_balance_threshold="auto",
        )
        return sampler.fit_resample(X, y)

    def get_params(self):
        params = super().get_params()
        params["k_neighbors"] = self.k_neighbors
        params["kmeans_estimator"] = self.kmeans_estimator
        return params


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINES = {
    "none": NoOversampling,
    "ros": RandomOversampling,
    "smote": SMOTEOversampling,
    "borderline_smote": BorderlineSMOTEOversampling,
    "adasyn": ADASYNOversampling,
    "svm_smote": SVMSMOTEOversampling,
    "kmeans_smote": KMeansSMOTEOversampling,
}


def get_baseline(name, **kwargs):
    """Instantiate a baseline oversampler by name.

    Parameters
    ----------
    name : str
        One of the keys in :data:`BASELINES`.
    **kwargs
        Forwarded to the constructor (e.g. ``sampling_strategy``,
        ``random_state``, ``k_neighbors``).

    Returns
    -------
    oversampler : BaseOversampler

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in BASELINES:
        raise KeyError(
            f"Unknown baseline '{name}'. Available: {list(BASELINES.keys())}"
        )
    return BASELINES[name](**kwargs)


def get_all_baselines(**kwargs):
    """Return a dict mapping every baseline name to a fresh instance.

    Parameters
    ----------
    **kwargs
        Common keyword arguments forwarded to every constructor.

    Returns
    -------
    baselines : dict[str, BaseOversampler]
    """
    return {name: cls(**kwargs) for name, cls in BASELINES.items()}


def list_baseline_names():
    """Return the list of registered baseline names.

    Returns
    -------
    names : list[str]
    """
    return list(BASELINES.keys())
