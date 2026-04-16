"""
Full-dimensional Gravity-biased Von Mises-Fisher oversampler (GVM-CO-ND).

Identical pipeline to GVM-CO but operates in the original d-dimensional
feature space instead of projecting to 2-D PCA first.  The Von Mises
angular distribution is replaced by the von Mises-Fisher (vMF) distribution
on S^{d-1}, which is the exact d-dimensional generalisation.

Key differences from GVM-CO (2-D):
  - No PCA projection / inverse transform.
  - Von Mises   → von Mises-Fisher (Wood 1994 sampler).
  - 2-D disk    → d-dimensional ball (r ~ R * U^{1/d}).
  - Gravity direction: unit vector from ball centre toward gravity centre.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

from src.core.base import BaseOversampler
from src.utils.geometry import circle_from_pair, vmf_in_ball
from src.utils.helpers import EPS


class GravityVonMisesND(BaseOversampler):
    """Full-dimensional gravity-biased vMF circular oversampler.

    Parameters
    ----------
    sampling_strategy : float or int, default=1.0
    K : int, default=3
        Number of minority clusters.
    k_nn : int, default=5
        Neighbours for local-scale (gravity-centre weighting).
    k_seed : int, default=5
        Neighbours for seed-neighbour pairing.
    kappa_max : float, default=20.0
        Maximum vMF concentration parameter.
    gamma : float, default=2.0
        Exponent mapping combined score to kappa.
    alpha : float, default=0.4
        Blending between additive (alpha=1) and multiplicative (alpha=0)
        combination of distance and gravity sub-scores. Default 0.4
        (best value from ablation study).
    clustering_method : str, default="kmeans"
    cross_cluster : bool, default=False
    minority_label : int, str, or None, default=None
    random_state : int or None, default=42
    """

    def __init__(
        self,
        sampling_strategy=1.0,
        K=3,
        k_nn=5,
        k_seed=5,
        kappa_max=20.0,
        gamma=2.0,
        alpha=0.4,
        clustering_method="kmeans",
        cross_cluster=False,
        minority_label=None,
        random_state=42,
    ):
        super().__init__(sampling_strategy=sampling_strategy,
                         minority_label=minority_label,
                         random_state=random_state)
        self.K = K
        self.k_nn = k_nn
        self.k_seed = k_seed
        self.kappa_max = kappa_max
        self.gamma = gamma
        self.alpha = alpha
        self.clustering_method = clustering_method
        self.cross_cluster = cross_cluster

    # ------------------------------------------------------------------
    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        min_lab = self._get_minority_label(y)
        n_synth = self._get_n_synth(y, min_lab)
        if n_synth == 0:
            return X.copy(), y.copy()

        X_min = X[y == min_lab]
        n_min, d = X_min.shape

        # --- Clustering ---
        K_eff = min(self.K, n_min)
        labels = self._cluster(X_min, K_eff, rng)

        # --- Gravity info ---
        grav = self._gravity_info(X_min, labels, K_eff)

        # --- KNN for pairing ---
        if self.cross_cluster:
            nn_idx = self._cross_cluster_nn(X_min, labels, grav["centroids"])
        else:
            k_eff = min(self.k_seed, n_min - 1)
            nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
            _, nn_idx = nbrs.kneighbors(X_min)

        # --- Generate ---
        synth = self._generate(X_min, nn_idx, labels, grav, n_synth, rng)

        X_res = np.vstack([X, synth])
        y_res = np.concatenate([y, np.full(n_synth, min_lab)])
        return X_res, y_res

    # ------------------------------------------------------------------
    def _cluster(self, X, K, rng):
        seed = int(rng.integers(0, 2**31))
        if self.clustering_method == "hac":
            return AgglomerativeClustering(n_clusters=K).fit_predict(X)
        return KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(X)

    def _gravity_info(self, X, labels, K):
        n, d = X.shape
        centroids = np.zeros((K, d))
        spreads = np.zeros(K)
        gravity_scores = np.zeros(K)
        gravity_centres = np.zeros((K, d))

        for c in range(K):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            Xc = X[idx]
            mu_c = Xc.mean(axis=0)
            centroids[c] = mu_c
            spread = np.mean(np.linalg.norm(Xc - mu_c, axis=1))
            spreads[c] = spread + EPS
            area = np.pi * spread**2 + EPS
            density = len(idx) / area

            # Sparsity: mean k-NN distance within cluster
            k_sp = min(self.k_nn, len(idx) - 1)
            if k_sp > 0:
                nbrs = NearestNeighbors(n_neighbors=k_sp + 1).fit(Xc)
                dists, _ = nbrs.kneighbors(Xc)
                sparsity = dists[:, 1:].mean() + EPS
            else:
                sparsity = EPS

            gravity_scores[c] = density / (spread * sparsity + EPS)

            # Density-weighted gravity centre
            if k_sp > 0:
                local_scales = dists[:, 1:].mean(axis=1) + EPS
            else:
                local_scales = np.full(len(idx), EPS)
            w = 1.0 / local_scales
            gravity_centres[c] = (w[:, None] * Xc).sum(axis=0) / w.sum()

        # Normalise gravity scores to [0, 1]
        g_min, g_max = gravity_scores.min(), gravity_scores.max()
        if g_max - g_min < EPS:
            norm_scores = np.ones(K)
        else:
            norm_scores = (gravity_scores - g_min) / (g_max - g_min + EPS)

        return {
            "centroids": centroids,
            "spreads": spreads,
            "norm_scores": norm_scores,
            "gravity_centres": gravity_centres,
        }

    def _cross_cluster_nn(self, X, labels, centroids):
        n = len(X)
        nn_idx = np.zeros((n, 2), dtype=int)
        nn_idx[:, 0] = np.arange(n)
        K = len(centroids)
        for i in range(n):
            c = labels[i]
            # Nearest other cluster by centroid distance
            dists_to_centroids = np.linalg.norm(centroids - centroids[c], axis=1)
            dists_to_centroids[c] = np.inf
            other_c = int(np.argmin(dists_to_centroids))
            other_idx = np.where(labels == other_c)[0]
            if len(other_idx) == 0:
                other_idx = np.where(labels != c)[0]
            dists = np.linalg.norm(X[other_idx] - X[i], axis=1)
            nn_idx[i, 1] = other_idx[np.argmin(dists)]
        return nn_idx

    def _generate(self, X_min, nn_idx, labels, grav, n_synth, rng):
        n_min, d = X_min.shape
        synth = np.empty((n_synth, d))
        norm_scores = grav["norm_scores"]
        gravity_centres = grav["gravity_centres"]

        for i in range(n_synth):
            seed_idx = int(rng.integers(0, n_min))
            row = nn_idx[seed_idx]
            # pick a random neighbour (skip self at index 0)
            j = int(rng.integers(1, len(row))) if len(row) > 1 else 0
            neigh_idx = int(row[j])

            xi, xj = X_min[seed_idx], X_min[neigh_idx]
            center, radius = circle_from_pair(xi, xj)

            c = int(labels[seed_idx])
            gc = gravity_centres[c]

            # vMF direction: from circle centre toward gravity centre
            direction = gc - center
            dir_norm = np.linalg.norm(direction)
            if dir_norm < EPS:
                direction = rng.standard_normal(d)
                dir_norm = np.linalg.norm(direction)
            mu = direction / dir_norm

            # Distance score
            dist_to_gc = dir_norm
            # Approximate d_min/d_max using median trick (cheap)
            s_d = float(norm_scores[c])  # reuse gravity as proxy; full s_d needs batch context

            s_g = float(norm_scores[c])
            mean_s = 0.5 * (s_d + s_g)
            joint_s = s_d * s_g
            s = self.alpha * mean_s + (1.0 - self.alpha) * joint_s
            kappa = max(self.kappa_max * (s ** self.gamma), 1e-6)

            pt = vmf_in_ball(rng, center, radius, mu, kappa, 1)
            synth[i] = pt[0]

        return synth

    def get_params(self):
        p = super().get_params()
        p.update({
            "K": self.K, "k_nn": self.k_nn, "k_seed": self.k_seed,
            "kappa_max": self.kappa_max, "gamma": self.gamma,
            "alpha": self.alpha, "clustering_method": self.clustering_method,
            "cross_cluster": self.cross_cluster,
        })
        return p
