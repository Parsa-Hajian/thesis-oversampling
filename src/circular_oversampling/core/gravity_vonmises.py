"""
Gravity-biased Von Mises circular oversampler (Algorithm 1).

This algorithm combines clustering-based gravity scoring with Von Mises
directional sampling inside circles formed between minority point pairs.

Pipeline
--------
1. Cluster minority points (K-means or HAC).
2. For each cluster compute spread, area, density, sparsity, and a composite
   gravity score.
3. Compute density-weighted *gravity centres* per cluster.
4. For each synthetic sample: form a circle between a seed and a KNN
   neighbour, derive the Von Mises direction (toward the gravity centre)
   and concentration kappa (from distance- and gravity-based scores), then
   sample from the Von Mises distribution inside the disk.

The *cross_cluster* (border restriction) variation forces the seed's
neighbour to come from the nearest *other* cluster.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

from src.core.base import BaseOversampler
from src.preprocessing.pca_projection import to_2d, from_2d
from src.utils.geometry import (
    circle_from_pair, vonmises_in_disk, vmf_in_ball, uniform_in_ball_batch,
)
from src.utils.helpers import EPS


# ======================================================================
# Main class
# ======================================================================

class GravityVonMises(BaseOversampler):
    """Gravity-biased Von Mises oversampler.

    Parameters
    ----------
    sampling_strategy : float or int, default=1.0
        Desired minority-to-majority ratio (float) or absolute count (int).
    K : int, default=3
        Number of minority clusters.
    k_nn : int, default=5
        Neighbours for local-scale estimation (used in gravity-centre
        weighting).
    k_seed : int, default=5
        Neighbours for seed-neighbour pairing.
    kappa_max : float, default=20.0
        Maximum Von Mises concentration parameter.
    gamma : float, default=2.0
        Exponent mapping the combined score to kappa.
    alpha : float, default=0.6
        Blending between the arithmetic mean (``0.5*(s_d + s_g)``) and the
        product (``s_d * s_g``) of the distance and gravity sub-scores.
    clustering_method : str, default="kmeans"
        ``"kmeans"`` or ``"hac"`` (hierarchical agglomerative clustering).
    cross_cluster : bool, default=False
        If ``True``, enforce the border-restriction variation where the
        neighbour is taken from the nearest *other* cluster.
    denoise_method : str or None, default=None
        Post-hoc denoising method (``"tomek"`` or ``"enn"``).  Applied after
        synthesis.  ``None`` disables denoising.
    minority_label : int, str, or None, default=None
        Explicit minority label; inferred if ``None``.
    random_state : int or None, default=42
        Seed for reproducibility.
    use_pca : bool, default=True
        If ``True``, project to 2-D PCA before all geometry.  If ``False``,
        operate in the original d-D feature space using von Mises-Fisher
        directional sampling inside d-D hyperballs.
    chunk_size : int, default=20000
        Maximum synthetic points per batch.
    """

    def __init__(
        self,
        sampling_strategy=1.0,
        K=3,
        k_nn=5,
        k_seed=5,
        kappa_max=20.0,
        gamma=2.0,
        alpha=0.6,
        clustering_method="kmeans",
        cross_cluster=False,
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
        self.K = K
        self.k_nn = k_nn
        self.k_seed = k_seed
        self.kappa_max = kappa_max
        self.gamma = gamma
        self.alpha = alpha
        self.clustering_method = clustering_method
        self.cross_cluster = cross_cluster
        self.denoise_method = denoise_method
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_resample(self, X, y):
        """Oversample the minority class using gravity-biased Von Mises.

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

        # --- Clustering ---
        K_eff = min(self.K, n_min)
        cluster_labels = self._cluster(X_min_2d, K_eff, rng)

        # --- Gravity scoring per cluster ---
        gravity_info = self._compute_gravity_info(X_min_2d, cluster_labels,
                                                  K_eff)

        # --- KNN graph for pairing ---
        if self.cross_cluster:
            nn_indices = self._build_cross_cluster_nn(
                X_min_2d, cluster_labels, gravity_info["centroids"], rng
            )
        else:
            k_eff = min(self.k_seed, n_min - 1)
            nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto")
            nn.fit(X_min_2d)
            _, nn_indices = nn.kneighbors(X_min_2d)

        # --- Generate synthetic samples ---
        synth_chunks = []
        remaining = n_synth
        while remaining > 0:
            batch = min(remaining, self.chunk_size)
            synth_chunks.append(
                self._generate_batch(
                    X_min_2d, nn_indices, cluster_labels,
                    gravity_info, batch, rng,
                )
            )
            remaining -= batch

        synth_2d = np.vstack(synth_chunks)

        # Lift back.
        X_synth = from_2d(synth_2d, pca_model)

        X_res = np.vstack([X, X_synth])
        y_res = np.concatenate([y, np.full(n_synth, min_lab)])

        # --- Optional denoising ---
        if self.denoise_method is not None:
            X_res, y_res = self._denoise(X_res, y_res, min_lab)

        return X_res, y_res

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster(self, X, K, rng):
        """Return integer cluster labels for *X*."""
        if K <= 1:
            return np.zeros(len(X), dtype=int)

        if self.clustering_method == "hac":
            model = AgglomerativeClustering(n_clusters=K)
            return model.fit_predict(X)

        # Default: K-means.
        seed = int(rng.integers(0, 2**31))
        model = KMeans(n_clusters=K, n_init=10, random_state=seed)
        return model.fit_predict(X)

    # ------------------------------------------------------------------
    # Gravity info
    # ------------------------------------------------------------------

    def _compute_gravity_info(self, X, cluster_labels, K):
        """Compute per-cluster gravity metrics.

        Works in any dimensionality d (2-D PCA or native d-D).

        Returns a dict with keys: centroids, gravity_scores, gravity_centres,
        per_point_cluster, g_min, g_max.
        """
        d = X.shape[1]
        centroids = np.empty((K, d), dtype=np.float64)
        spreads = np.empty(K)
        densities = np.empty(K)
        sparsities = np.empty(K)

        for c in range(K):
            mask = cluster_labels == c
            Xc = X[mask]
            n_c = len(Xc)
            centroid = Xc.mean(axis=0) if n_c > 0 else X.mean(axis=0)
            centroids[c] = centroid

            # Spread: mean distance to centroid.
            if n_c > 1:
                dists_to_cent = np.linalg.norm(Xc - centroid, axis=1)
                spread_c = dists_to_cent.mean()
            else:
                spread_c = EPS
            spreads[c] = max(spread_c, EPS)

            # Area (2-D circle approximation).
            area_c = np.pi * spreads[c] ** 2 + EPS

            # Density.
            densities[c] = n_c / area_c

            # Sparsity: mean k-NN distance within cluster.
            if n_c > 1:
                k_local = min(self.k_nn, n_c - 1)
                nn_local = NearestNeighbors(n_neighbors=k_local + 1)
                nn_local.fit(Xc)
                dists_nn, _ = nn_local.kneighbors(Xc)
                # Exclude self-distance (column 0).
                sparsities[c] = dists_nn[:, 1:].mean()
            else:
                sparsities[c] = EPS

        # Gravity score per cluster: g_c = density / (spread * sparsity).
        g_scores = densities / (spreads * sparsities + EPS)

        # Gravity centres (density-weighted).
        gravity_centres = np.empty((K, d), dtype=np.float64)
        for c in range(K):
            mask = cluster_labels == c
            Xc = X[mask]
            if len(Xc) == 0:
                gravity_centres[c] = centroids[c]
                continue
            # Weight: inverse local scale.
            local_scales = self._local_scales(Xc)
            weights = 1.0 / (local_scales + EPS)
            gravity_centres[c] = np.average(Xc, axis=0, weights=weights)

        g_min = g_scores.min()
        g_max = g_scores.max()

        return {
            "centroids": centroids,
            "gravity_scores": g_scores,
            "gravity_centres": gravity_centres,
            "g_min": g_min,
            "g_max": g_max,
        }

    def _local_scales(self, X):
        """Mean k-NN distance for each point in *X*."""
        n = len(X)
        if n <= 1:
            return np.full(n, EPS)
        k = min(self.k_nn, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        return dists[:, 1:].mean(axis=1)

    # ------------------------------------------------------------------
    # Cross-cluster (border restriction) neighbour pairing
    # ------------------------------------------------------------------

    def _build_cross_cluster_nn(self, X, cluster_labels, centroids, rng):
        """Build a neighbour index where each seed's neighbour comes from the
        nearest *other* cluster (by gravity-centre distance).

        Returns nn_indices of shape (n_min, k_eff+1) where column 0 is self.
        """
        n = len(X)
        K = len(centroids)
        k_eff = min(self.k_seed, n - 1)

        # For each cluster, find the nearest other cluster.
        nearest_other = np.empty(K, dtype=int)
        for c in range(K):
            dists = np.linalg.norm(centroids - centroids[c], axis=1)
            dists[c] = np.inf  # Exclude self.
            nearest_other[c] = np.argmin(dists)

        nn_indices = np.zeros((n, k_eff + 1), dtype=int)
        nn_indices[:, 0] = np.arange(n)  # Self in column 0.

        for c in range(K):
            mask_c = np.where(cluster_labels == c)[0]
            other = nearest_other[c]
            mask_other = np.where(cluster_labels == other)[0]

            if len(mask_other) == 0:
                # Fallback: pair with own cluster.
                mask_other = mask_c

            k_cross = min(k_eff, len(mask_other))
            if k_cross == 0:
                nn_indices[mask_c, 1:] = mask_c[0]
                continue

            nn_cross = NearestNeighbors(n_neighbors=k_cross)
            nn_cross.fit(X[mask_other])
            _, idx_local = nn_cross.kneighbors(X[mask_c])
            # Map local indices back to global.
            idx_global = mask_other[idx_local]

            for col in range(min(k_eff, k_cross)):
                nn_indices[mask_c, col + 1] = idx_global[:, col]
            # If k_eff > k_cross, repeat last column.
            for col in range(k_cross, k_eff):
                nn_indices[mask_c, col + 1] = idx_global[:, -1]

        return nn_indices

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def _generate_batch(self, X_min_2d, nn_indices, cluster_labels,
                        gravity_info, batch_size, rng):
        """Generate *batch_size* synthetic points in the working space.

        For 2-D PCA mode: uses Von Mises angular sampling inside each disk.
        For native d-D mode: uses a vectorised vMF-like directional bias inside
        each d-dimensional hyperball (Gaussian-direction normalisation method).
        """
        n_min = len(X_min_2d)
        d = X_min_2d.shape[1]
        k_eff = nn_indices.shape[1] - 1  # Column 0 is self.

        # Pick random anchors and neighbours.
        anchor_idx = rng.integers(0, n_min, size=batch_size)
        neigh_col = rng.integers(1, k_eff + 1, size=batch_size)
        neigh_idx = nn_indices[anchor_idx, neigh_col]

        anchors = X_min_2d[anchor_idx]
        neighbours = X_min_2d[neigh_idx]

        # Ball per pair.
        centers = 0.5 * (anchors + neighbours)
        radii = 0.5 * np.linalg.norm(anchors - neighbours, axis=1)
        radii = np.maximum(radii, EPS)

        # Cluster of each anchor.
        anchor_clusters = cluster_labels[anchor_idx]

        # --- Compute per-sample kappa and direction ---
        gc = gravity_info["gravity_centres"]       # (K, d)
        g_scores = gravity_info["gravity_scores"]
        g_min = gravity_info["g_min"]
        g_max = gravity_info["g_max"]

        gc_for_anchor = gc[anchor_clusters]        # (batch, d)
        diffs = gc_for_anchor - centers            # (batch, d)

        # Distance score s_d.
        dists_to_gc = np.linalg.norm(diffs, axis=1)
        d_min = dists_to_gc.min()
        d_max = dists_to_gc.max()
        s_d = (d_max - dists_to_gc) / (d_max - d_min + EPS)

        # Gravity score s_g.
        g_per_sample = g_scores[anchor_clusters]
        s_g = (g_per_sample - g_min) / (g_max - g_min + EPS)

        # Combined kappa.
        s_combined = (self.alpha * 0.5 * (s_d + s_g)
                      + (1.0 - self.alpha) * s_d * s_g)
        kappa = self.kappa_max * (s_combined ** self.gamma)

        if d == 2:
            # --- 2-D: Von Mises angle + sqrt-radius ---
            mu_angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            theta = np.array([
                rng.vonmises(mu_angles[i], kappa[i]) for i in range(batch_size)
            ])
            u = rng.random(batch_size)
            r = radii * np.sqrt(u)
            synth = np.empty((batch_size, 2), dtype=np.float64)
            synth[:, 0] = centers[:, 0] + r * np.cos(theta)
            synth[:, 1] = centers[:, 1] + r * np.sin(theta)
        else:
            # --- d-D: vMF-like directional bias via Gaussian normalisation ---
            # Normalise the direction toward the gravity centre.
            mu_dirs = diffs / (dists_to_gc[:, np.newaxis] + EPS)  # (batch, d)
            # Sample isotropic Gaussian directions and bias toward mu.
            raw_dirs = rng.standard_normal((batch_size, d))
            biased = raw_dirs + kappa[:, np.newaxis] * mu_dirs
            norms = np.linalg.norm(biased, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-30)
            directions = biased / norms
            # Volume-uniform radius in d-D: r ~ R * U^{1/d}.
            u = rng.random(batch_size)
            r = radii * (u ** (1.0 / d))
            synth = centers + r[:, np.newaxis] * directions

        return synth

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
            "K": self.K,
            "k_nn": self.k_nn,
            "k_seed": self.k_seed,
            "kappa_max": self.kappa_max,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "clustering_method": self.clustering_method,
            "cross_cluster": self.cross_cluster,
            "denoise_method": self.denoise_method,
            "chunk_size": self.chunk_size,
        })
        return params


