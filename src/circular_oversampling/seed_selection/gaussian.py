"""
Gaussian (random) seed selection baseline.

Simple random selection of seeds from the minority class,
optionally with Gaussian weighting toward the center.
"""

import numpy as np


class GaussianSeedSelector:
    """
    Baseline seed selector using random or Gaussian-weighted selection.

    Parameters
    ----------
    method : str
        "random" for uniform random, "gaussian" for center-weighted.
    random_state : int
        Random seed.
    """

    def __init__(self, method="random", random_state=42):
        self.method = method
        self.random_state = random_state

    def select(self, X_minority, n_seeds):
        """
        Select seeds randomly or with Gaussian weighting.

        Parameters
        ----------
        X_minority : ndarray, shape (n, d)
        n_seeds : int

        Returns
        -------
        seed_indices : ndarray, shape (n_seeds,)
        score : float
            Always 0.0 for baseline.
        scores_log : dict
            Empty dict for baseline.
        """
        X = np.asarray(X_minority, dtype=float)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        if n_seeds >= n:
            return np.arange(n), 0.0, {}

        if self.method == "gaussian":
            # Gaussian weighting: points closer to center are more likely
            center = X.mean(axis=0)
            dists = np.linalg.norm(X - center, axis=1)
            # Inverse distance weighting with Gaussian kernel
            sigma = np.median(dists) + 1e-12
            weights = np.exp(-0.5 * (dists / sigma) ** 2)
            weights = weights / weights.sum()
            indices = rng.choice(n, size=n_seeds, replace=False, p=weights)
        else:
            # Uniform random
            indices = rng.choice(n, size=n_seeds, replace=False)

        return indices, 0.0, {}
