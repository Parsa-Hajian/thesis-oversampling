"""
Benchmark Circular-SMOTE oversampler.

For each synthetic sample the algorithm selects a random minority seed,
picks one of its *k* nearest minority neighbours, forms a circle (centred
at their midpoint with radius = half-distance), and samples a point
uniformly inside the resulting disk.

When ``use_pca=True`` (default) and the original feature space has more than
two dimensions, the geometry is performed in a 2-D PCA subspace and the
resulting synthetic points are lifted back via the inverse PCA transform.
When ``use_pca=False``, the circle becomes a d-dimensional hyperball and
uniform sampling uses the d-D ball formula.

Chunked processing is used to keep memory bounded when generating large
numbers of synthetic samples.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.core.base import BaseOversampler
from src.preprocessing.pca_projection import to_2d, from_2d
from src.utils.geometry import circle_from_pair, uniform_in_disk_vec, uniform_in_ball_batch


class CircularSMOTE(BaseOversampler):
    """Circular-SMOTE baseline oversampler.

    Parameters
    ----------
    sampling_strategy : float or int, default=1.0
        Desired minority-to-majority ratio (float) or absolute count (int).
    k : int, default=5
        Number of nearest minority neighbours used for pair formation.
    minority_label : int, str, or None, default=None
        Explicit minority label; inferred if ``None``.
    random_state : int or None, default=42
        Seed for reproducibility.
    use_pca : bool, default=True
        If ``True``, project to 2-D PCA before sampling.  If ``False``,
        operate directly in the original d-dimensional feature space.
    chunk_size : int, default=20000
        Maximum number of synthetic points generated per batch to limit
        peak memory usage.
    """

    def __init__(self, sampling_strategy=1.0, k=5, minority_label=None,
                 random_state=42, use_pca=True, chunk_size=20000):
        super().__init__(
            sampling_strategy=sampling_strategy,
            minority_label=minority_label,
            random_state=random_state,
            use_pca=use_pca,
        )
        self.k = k
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_resample(self, X, y):
        """Oversample the minority class using Circular-SMOTE.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        X_resampled : ndarray of shape (n_samples_new, n_features)
        y_resampled : ndarray of shape (n_samples_new,)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        min_lab = self._get_minority_label(y)
        n_synth = self._get_n_synth(y, min_lab)
        if n_synth == 0:
            return X.copy(), y.copy()

        X_min = X[y == min_lab]
        n_min = len(X_min)

        # Project to working space (2-D PCA or original d-D).
        X_min_2d, pca_model = to_2d(
            X_min, random_state=self.random_state or 0, use_pca=self.use_pca
        )

        # KNN on minority points in working space.
        k_eff = min(self.k, n_min - 1)
        if k_eff < 1:
            # Fewer than 2 minority points -- duplicate with tiny jitter.
            synth_2d = np.tile(X_min_2d[0], (n_synth, 1))
            synth_2d += rng.normal(0, 1e-6, synth_2d.shape)
            X_synth = from_2d(synth_2d, pca_model)
            return (
                np.vstack([X, X_synth]),
                np.concatenate([y, np.full(n_synth, min_lab)]),
            )

        nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto")
        nn.fit(X_min_2d)
        # distances shape (n_min, k_eff+1); first column is self-distance ~0
        _, nn_indices = nn.kneighbors(X_min_2d)

        # Generate synthetic samples in chunks.
        synth_chunks = []
        remaining = n_synth

        while remaining > 0:
            batch = min(remaining, self.chunk_size)
            synth_chunks.append(
                self._generate_batch(X_min_2d, nn_indices, batch, k_eff, rng)
            )
            remaining -= batch

        synth_2d = np.vstack(synth_chunks)

        # Lift back to original dimensionality.
        X_synth = from_2d(synth_2d, pca_model)

        X_resampled = np.vstack([X, X_synth])
        y_resampled = np.concatenate([y, np.full(n_synth, min_lab)])
        return X_resampled, y_resampled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_batch(X_min_2d, nn_indices, batch_size, k_eff, rng):
        """Generate a batch of synthetic points in the working space.

        For each synthetic sample:
          1. Pick a random minority anchor.
          2. Pick a random KNN neighbour of the anchor.
          3. Form a circle / hyperball (midpoint centre, half-distance radius).
          4. Sample uniformly inside the ball.
        """
        n_min = len(X_min_2d)
        d = X_min_2d.shape[1]

        # Random anchors.
        anchor_idx = rng.integers(0, n_min, size=batch_size)
        # Random neighbour index (columns 1..k_eff are actual neighbours).
        neigh_col = rng.integers(1, k_eff + 1, size=batch_size)
        neigh_idx = nn_indices[anchor_idx, neigh_col]

        anchors = X_min_2d[anchor_idx]
        neighbours = X_min_2d[neigh_idx]

        # Vectorised ball construction.
        centers = 0.5 * (anchors + neighbours)
        radii = 0.5 * np.linalg.norm(anchors - neighbours, axis=1)
        radii = np.maximum(radii, 1e-12)

        # Vectorised uniform ball sampling (2-D fast path; d-D general path).
        if d == 2:
            return uniform_in_disk_vec(rng, centers, radii)
        return uniform_in_ball_batch(rng, centers, radii)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_params(self):
        params = super().get_params()
        params.update({
            "k": self.k,
            "chunk_size": self.chunk_size,
        })
        return params


