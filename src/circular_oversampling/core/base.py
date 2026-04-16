"""
Abstract base class for all oversampling methods in the project.

Every oversampler -- whether a custom circular method or an imbalanced-learn
wrapper -- must subclass :class:`BaseOversampler` and implement
:meth:`fit_resample`.  This guarantees a uniform API that the evaluation
pipeline can rely on.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseOversampler(ABC):
    """Base class that all oversamplers must inherit from.

    Subclasses are required to implement :meth:`fit_resample`, which accepts
    the feature matrix *X* and label vector *y* and returns their oversampled
    counterparts.

    Parameters
    ----------
    sampling_strategy : float or int, default=1.0
        Controls how many synthetic minority samples are generated.

        * **float** -- target minority-to-majority ratio.  ``1.0`` means full
          balance (minority count == majority count after oversampling).
        * **int** -- absolute number of synthetic samples to produce.

    minority_label : int, str, or None, default=None
        If provided, this value is used as the minority class label.  When
        ``None`` the minority label is inferred as the least-frequent class.

    random_state : int or None, default=42
        Seed for reproducibility.

    use_pca : bool, default=True
        If ``True`` (default), all geometric operations are performed in a
        2-D PCA subspace and synthetic points are lifted back to the original
        dimensionality via the inverse PCA transform.
        If ``False``, all geometry (ball sampling, clustering, KNN) is
        performed directly in the original d-dimensional feature space.
        The native-dimension mode avoids PCA information loss but uses
        d-dimensional hyperballs; for high-dimensional data (d > 10) the
        volume of the hypersphere concentrates near its surface, so results
        may differ from the 2-D PCA mode.
    """

    def __init__(self, sampling_strategy=1.0, minority_label=None,
                 random_state=42, use_pca=True):
        self.sampling_strategy = sampling_strategy
        self.minority_label = minority_label
        self.random_state = random_state
        self.use_pca = use_pca

    @abstractmethod
    def fit_resample(self, X, y):
        """Oversample the minority class and return the augmented dataset.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Target labels (binary).

        Returns
        -------
        X_resampled : ndarray of shape (n_samples_new, n_features)
            Augmented feature matrix (original + synthetic samples).
        y_resampled : ndarray of shape (n_samples_new,)
            Augmented target labels.
        """

    # ------------------------------------------------------------------
    # Convenience helpers wrapping src.utils.helpers
    # ------------------------------------------------------------------

    def _get_minority_label(self, y):
        """Identify the minority class label.

        Uses the caller-supplied ``self.minority_label`` when available;
        otherwise falls back to the least-frequent class in *y*.
        """
        from src.utils.helpers import minority_label as _minority_label
        return _minority_label(y, self.minority_label)

    def _get_n_synth(self, y, min_lab):
        """Compute the number of synthetic samples to generate."""
        from src.utils.helpers import n_synth_from_strategy
        return n_synth_from_strategy(y, min_lab, self.sampling_strategy)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self):
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.get_params().items()
        )
        return f"{type(self).__name__}({params})"

    def get_params(self):
        """Return a dictionary of constructor parameters and their values."""
        return {
            "sampling_strategy": self.sampling_strategy,
            "minority_label": self.minority_label,
            "random_state": self.random_state,
            "use_pca": self.use_pca,
        }
