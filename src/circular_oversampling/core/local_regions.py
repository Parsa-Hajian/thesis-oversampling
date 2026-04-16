"""
Local-region-aware circular oversampler (Algorithm 2).

This algorithm refines the basic circular approach by performing a local
analysis inside every candidate circle before sampling:

Pipeline
--------
1. Build a KNN graph on the minority class.
2. For each candidate circle (formed from a seed--neighbour pair):
   a. Check that at least ``N_min`` minority points lie inside the circle.
   b. Run a local K-means inside the circle to form Voronoi sub-regions.
   c. Assign each sub-region a sampling probability proportional to
      ``(n_k + eps)^beta`` where ``n_k`` is the count in that sub-region.
   d. (Optional) Evaluate a local certainty model; if uncertainty exceeds
      the threshold, skip this circle.
3. For each synthetic point: pick a sub-region by the weighted probability,
   then sample uniformly inside the intersection of that Voronoi cell and
   the bounding disk via rejection sampling.

Post-hoc denoising (Tomek links or ENN) can be applied optionally.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from src.core.base import BaseOversampler
from src.preprocessing.pca_projection import to_2d, from_2d
from src.utils.geometry import (
    circle_from_pair,
    points_in_circle,
    uniform_in_disk_vec,
    uniform_in_ball_batch,
    assign_voronoi,
    sample_in_voronoi_cell,
)
from src.utils.helpers import EPS


class LocalRegions(BaseOversampler):
    """Local-region-aware oversampler.

    Parameters
    ----------
    sampling_strategy : float or int, default=1.0
        Desired minority-to-majority ratio (float) or absolute count (int).
    k_seed : int, default=10
        KNN neighbours for seed--neighbour pairing.
    N_min : int, default=8
        Minimum minority points required inside a circle for it to be
        eligible.
    local_k_max : int, default=6
        Maximum number of local K-means clusters inside each circle.
    min_points_per_cluster : int, default=15
        If the circle has fewer than ``local_k_max * min_points_per_cluster``
        points, the local K is reduced proportionally.
    beta : float, default=1.7
        Exponent controlling how strongly sub-region counts influence
        sampling probability.  Larger values bias sampling toward denser
        sub-regions.
    certainty_threshold : float, default=0.80
        Minimum KNN-based certainty score for a circle to be used.  Set to
        0.0 to disable the certainty check.
    clustering_method : str, default="kmeans"
        ``"kmeans"`` (only K-means is used for local sub-regions).
    denoise_method : str or None, default=None
        Post-hoc denoising (``"tomek"`` or ``"enn"``).  ``None`` disables it.
    minority_label : int, str, or None, default=None
        Explicit minority label; inferred if ``None``.
    random_state : int or None, default=42
        Seed for reproducibility.
    use_pca : bool, default=True
        If ``True``, project to 2-D PCA before all geometry.  If ``False``,
        circles become d-D hyperballs; the certainty check and Voronoi
        partitioning operate in the original feature space.
    chunk_size : int, default=20000
        Maximum synthetic points per generation pass.
    """

    def __init__(
        self,
        sampling_strategy=1.0,
        k_seed=10,
        N_min=8,
        local_k_max=6,
        min_points_per_cluster=15,
        beta=1.7,
        certainty_threshold=0.80,
        clustering_method="kmeans",
        denoise_method=None,
        minority_label=None,
        random_state=42,
        use_pca=True,
        chunk_size=20000,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            minority_label=minority_label,
            random_state=random_state,
            use_pca=use_pca,
        )
        self.k_seed = k_seed
        self.N_min = N_min
        self.local_k_max = local_k_max
        self.min_points_per_cluster = min_points_per_cluster
        self.beta = beta
        self.certainty_threshold = certainty_threshold
        self.clustering_method = clustering_method
        self.denoise_method = denoise_method
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_resample(self, X, y):
        """Oversample the minority class using local-region estimation.

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

        # --- Project to working space (2-D PCA or original d-D) ---
        X_min_2d, pca_model = to_2d(
            X_min, random_state=self.random_state or 0, use_pca=self.use_pca
        )
        # Project the full dataset for the certainty check (same projection).
        X_all_2d, _ = to_2d(
            X, random_state=self.random_state or 0, use_pca=self.use_pca
        )

        # --- KNN graph on minority (2-D) ---
        k_eff = min(self.k_seed, n_min - 1)
        if k_eff < 1:
            # Fewer than 2 minority points.
            synth_2d = np.tile(X_min_2d[0], (n_synth, 1))
            synth_2d += rng.normal(0, 1e-6, synth_2d.shape)
            X_synth = from_2d(synth_2d, pca_model)
            return (
                np.vstack([X, X_synth]),
                np.concatenate([y, np.full(n_synth, min_lab)]),
            )

        nn = NearestNeighbors(n_neighbors=k_eff + 1)
        nn.fit(X_min_2d)
        _, nn_indices = nn.kneighbors(X_min_2d)

        # --- Pre-compute eligible circles ---
        # Each circle is (anchor_idx, neigh_idx, center_2d, radius,
        #                  local_centroids, region_probs).
        circles = self._build_eligible_circles(
            X_min_2d, nn_indices, k_eff, X_all_2d, y, min_lab, rng
        )

        if len(circles) == 0:
            # No eligible circles -- fall back to basic circular sampling.
            synth_2d = self._fallback(X_min_2d, nn_indices, k_eff, n_synth, rng)
            X_synth = from_2d(synth_2d, pca_model)
            X_res = np.vstack([X, X_synth])
            y_res = np.concatenate([y, np.full(n_synth, min_lab)])
            return X_res, y_res

        # --- Generate synthetic samples ---
        synth_2d = self._generate(circles, n_synth, rng)

        # Lift back.
        X_synth = from_2d(synth_2d, pca_model)
        X_res = np.vstack([X, X_synth])
        y_res = np.concatenate([y, np.full(n_synth, min_lab)])

        # --- Optional denoising ---
        if self.denoise_method is not None:
            X_res, y_res = self._denoise(X_res, y_res, min_lab)

        return X_res, y_res

    # ------------------------------------------------------------------
    # Build eligible circles
    # ------------------------------------------------------------------

    def _build_eligible_circles(self, X_min_2d, nn_indices, k_eff,
                                X_all_2d, y, min_lab, rng):
        """Pre-compute all eligible circles with local sub-regions.

        Returns a list of dicts, each containing:
            center, radius, local_centroids, region_probs
        """
        n_min = len(X_min_2d)
        circles = []

        for i in range(n_min):
            for col in range(1, k_eff + 1):
                j = nn_indices[i, col]
                center, radius = circle_from_pair(X_min_2d[i], X_min_2d[j])

                # --- Check minimum points ---
                mask_inside = points_in_circle(X_min_2d, center, radius)
                n_inside = mask_inside.sum()
                if n_inside < self.N_min:
                    continue

                # --- Certainty check (80 % threshold) ---
                if self.certainty_threshold > 0:
                    certainty = self._circle_certainty(
                        center, radius, X_all_2d, y, min_lab
                    )
                    if certainty < self.certainty_threshold:
                        continue

                # --- Local K-means ---
                X_inside = X_min_2d[mask_inside]
                local_k = self._effective_local_k(n_inside)

                if local_k <= 1:
                    # Single region -- uniform disk sampling.
                    circles.append({
                        "center": center,
                        "radius": radius,
                        "local_centroids": center.reshape(1, 2),
                        "region_probs": np.array([1.0]),
                    })
                    continue

                seed = int(rng.integers(0, 2**31))
                km = KMeans(n_clusters=local_k, n_init=5, random_state=seed)
                sub_labels = km.fit_predict(X_inside)
                local_centroids = km.cluster_centers_

                # Voronoi region probabilities ~ (n_k + eps)^beta.
                counts = np.array([
                    np.sum(sub_labels == c) for c in range(local_k)
                ], dtype=np.float64)
                weights = (counts + EPS) ** self.beta
                region_probs = weights / weights.sum()

                circles.append({
                    "center": center,
                    "radius": radius,
                    "local_centroids": local_centroids,
                    "region_probs": region_probs,
                })

        return circles

    def _effective_local_k(self, n_inside):
        """Determine the number of local K-means clusters to use."""
        max_k = self.local_k_max
        if n_inside < max_k * self.min_points_per_cluster:
            max_k = max(1, n_inside // max(self.min_points_per_cluster, 1))
        return max(1, min(max_k, n_inside))

    def _circle_certainty(self, center, radius, X_all_2d, y, min_lab):
        """Estimate certainty that the circle is in a minority-dominated
        region using a KNN classifier on all points inside the circle.

        Returns the fraction of points inside the circle that are minority.
        """
        mask = points_in_circle(X_all_2d, center, radius)
        if mask.sum() == 0:
            return 0.0
        y_inside = y[mask]
        n_min_inside = np.sum(y_inside == min_lab)
        return n_min_inside / len(y_inside)

    # ------------------------------------------------------------------
    # Synthetic sample generation
    # ------------------------------------------------------------------

    def _generate(self, circles, n_synth, rng):
        """Generate *n_synth* synthetic points using the eligible circles."""
        n_circles = len(circles)
        synth = np.empty((n_synth, 2), dtype=np.float64)
        idx = 0

        while idx < n_synth:
            # Pick a random eligible circle.
            ci = rng.integers(0, n_circles)
            circ = circles[ci]
            center = circ["center"]
            radius = circ["radius"]
            local_centroids = circ["local_centroids"]
            region_probs = circ["region_probs"]
            n_regions = len(region_probs)

            # Pick a sub-region.
            region_idx = rng.choice(n_regions, p=region_probs)

            if n_regions == 1:
                # Uniform disk sampling.
                pt = uniform_in_disk_vec(
                    rng,
                    center.reshape(1, 2),
                    np.array([radius]),
                )
                synth[idx] = pt[0]
            else:
                # Rejection-sample inside the Voronoi cell of the chosen
                # sub-region within the bounding disk.
                pt = sample_in_voronoi_cell(
                    rng, local_centroids, region_idx, center, radius,
                    n_samples=1, max_rejection_iters=200,
                )
                synth[idx] = pt[0]

            idx += 1

        return synth

    # ------------------------------------------------------------------
    # Fallback (basic circular sampling when no circles are eligible)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(X_min_2d, nn_indices, k_eff, n_synth, rng):
        """Basic circular-SMOTE fallback (dimension-agnostic)."""
        n_min = len(X_min_2d)
        d = X_min_2d.shape[1]
        anchor_idx = rng.integers(0, n_min, size=n_synth)
        neigh_col = rng.integers(1, k_eff + 1, size=n_synth)
        neigh_idx = nn_indices[anchor_idx, neigh_col]

        anchors = X_min_2d[anchor_idx]
        neighbours = X_min_2d[neigh_idx]
        centers = 0.5 * (anchors + neighbours)
        radii = 0.5 * np.linalg.norm(anchors - neighbours, axis=1)
        radii = np.maximum(radii, EPS)

        if d == 2:
            return uniform_in_disk_vec(rng, centers, radii)
        return uniform_in_ball_batch(rng, centers, radii)

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    def _denoise(self, X, y, min_lab):
        """Apply optional post-hoc denoising."""
        method = self.denoise_method
        if method is None:
            return X, y

        if method == "tomek":
            from imblearn.under_sampling import TomekLinks
            tl = TomekLinks()
            return tl.fit_resample(X, y)

        if method == "enn":
            from imblearn.under_sampling import EditedNearestNeighbours
            enn = EditedNearestNeighbours()
            return enn.fit_resample(X, y)

        raise ValueError(f"Unknown denoise method: {method!r}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_params(self):
        params = super().get_params()
        params.update({
            "k_seed": self.k_seed,
            "N_min": self.N_min,
            "local_k_max": self.local_k_max,
            "min_points_per_cluster": self.min_points_per_cluster,
            "beta": self.beta,
            "certainty_threshold": self.certainty_threshold,
            "clustering_method": self.clustering_method,
            "denoise_method": self.denoise_method,
            "chunk_size": self.chunk_size,
        })
        return params


