"""
Geometry-preserving seed selection for oversampling.

Selects minority seed points that best preserve the geometric coverage,
distributional similarity, and local structure of the original minority class.

Pipeline:
    1. Cluster minority using K-Means
    2. Project to PCA space (k components)
    3. Generate N random candidate seed sets
    4. Score each candidate by NHOP + AGTP - JSD (maximize) and Z (minimize)
    5. Return the best candidate

References:
    - Hajiannejad (2025), Seed Selection for Oversampling
    - Analogous to outlier-aware sample selection in CBR
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.seed_selection.metrics import (
    normalized_histogram_overlap_percentage,
    jensen_shannon_divergence,
    agtp_score,
    smoothness_score,
)


class SeedSelector:
    """
    Geometry-preserving seed selector.

    Parameters
    ----------
    n_candidates : int
        Number of random candidate seed sets to evaluate.
    n_pcs : int
        Number of PCA components for scoring.
    k_clusters : int
        Number of K-Means clusters for stratified sampling.
    k_topo : int
        k for topological similarity in AGTP.
    jsd_weight : float
        Weight for JSD penalty in the composite score.
    z_weight : float
        Weight for smoothness penalty: score = (NHOP + AGTP) - jsd_weight * JSD - z_weight * Z.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_candidates=100,
        n_pcs=2,
        k_clusters=5,
        k_topo=5,
        jsd_weight=0.3,
        z_weight=0.5,
        random_state=42,
    ):
        self.n_candidates = n_candidates
        self.n_pcs = n_pcs
        self.k_clusters = k_clusters
        self.k_topo = k_topo
        self.jsd_weight = jsd_weight
        self.z_weight = z_weight
        self.random_state = random_state

    def select(self, X_minority, n_seeds):
        """
        Select the best seed set from the minority class.

        Parameters
        ----------
        X_minority : ndarray, shape (n, d)
            All minority class points.
        n_seeds : int
            Number of seeds to select.

        Returns
        -------
        seed_indices : ndarray, shape (n_seeds,)
            Indices into X_minority of the selected seeds.
        best_score : float
            Score of the best candidate.
        scores_log : dict
            Detailed scores of the best candidate (HOP, AGTP, Z).
        """
        X = np.asarray(X_minority, dtype=float)
        n, d = X.shape

        if n_seeds >= n:
            return np.arange(n), 0.0, {"nhop": 1.0, "agtp": 1.0, "jsd": 0.0, "z": 0.0}

        rng = np.random.default_rng(self.random_state)

        # Project to PCA space for scoring
        n_pcs = min(self.n_pcs, d, n)
        if n_pcs < d:
            pca = PCA(n_components=n_pcs, random_state=self.random_state)
            X_pca = pca.fit_transform(X)
        else:
            X_pca = X.copy()

        # Cluster minority for stratified candidate generation
        k = min(self.k_clusters, n)
        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(X_pca)
        labels = km.labels_

        # Cluster sizes for proportional allocation
        cluster_sizes = np.bincount(labels, minlength=k)
        cluster_fracs = cluster_sizes / cluster_sizes.sum()
        seeds_per_cluster = np.round(cluster_fracs * n_seeds).astype(int)

        # Fix rounding: adjust largest cluster
        diff = n_seeds - seeds_per_cluster.sum()
        seeds_per_cluster[np.argmax(cluster_sizes)] += diff

        # Ensure no cluster gets more seeds than it has points
        for c in range(k):
            seeds_per_cluster[c] = min(seeds_per_cluster[c], cluster_sizes[c])
        # Redistribute excess
        remaining = n_seeds - seeds_per_cluster.sum()
        for c in np.argsort(-cluster_sizes):
            if remaining <= 0:
                break
            can_add = cluster_sizes[c] - seeds_per_cluster[c]
            add = min(remaining, can_add)
            seeds_per_cluster[c] += add
            remaining -= add

        best_score = -np.inf
        best_indices = None
        best_log = None

        for _ in range(self.n_candidates):
            # Generate candidate: stratified random sampling
            candidate = []
            for c in range(k):
                cluster_idx = np.where(labels == c)[0]
                n_pick = int(seeds_per_cluster[c])
                if n_pick > 0 and len(cluster_idx) > 0:
                    picked = rng.choice(cluster_idx, size=n_pick, replace=False)
                    candidate.extend(picked.tolist())

            candidate = np.array(candidate)
            if len(candidate) == 0:
                continue

            # Score candidate in PCA space
            X_seeds_pca = X_pca[candidate]

            nhop = normalized_histogram_overlap_percentage(X_pca, X_seeds_pca)
            jsd = jensen_shannon_divergence(X_pca, X_seeds_pca)
            agtp = agtp_score(X_pca, X_seeds_pca, k=self.k_topo)
            z = smoothness_score(X_seeds_pca)

            score = (nhop + agtp) - self.jsd_weight * jsd - self.z_weight * z

            if score > best_score:
                best_score = score
                best_indices = candidate
                best_log = {"nhop": nhop, "agtp": agtp, "jsd": jsd, "z": z}

        if best_indices is None:
            # Fallback: random selection
            best_indices = rng.choice(n, size=n_seeds, replace=False)
            best_log = {"nhop": 0.0, "agtp": 0.0, "jsd": 0.0, "z": 0.0}

        return best_indices, best_score, best_log
