"""
Seed selection scoring metrics: NHOP, JSD, AGTP, Z.

These metrics evaluate how well a candidate seed set preserves the
statistical and geometric properties of the original minority class.

References:
    - Seed selection strategy from Hajiannejad (2025)
    - Normalized Histogram Overlap Percentage (NHOP) for marginal distribution similarity
    - Jensen-Shannon Divergence (JSD) for distributional similarity
    - AGTP for geometric + topological similarity
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


def normalized_histogram_overlap_percentage(X_original, X_seeds, n_bins=30):
    """
    Compute Normalized Histogram Overlap Percentage (NHOP) across all dimensions.

    Before computing histograms, both sets are min-max normalized per dimension
    using the combined range, ensuring that both distributions are on the same
    [0, 1] scale. Histograms are then built on shared bin edges over this
    normalized range and converted to probability distributions.

    Parameters
    ----------
    X_original : ndarray, shape (n, d)
        Original minority points (in PCA space).
    X_seeds : ndarray, shape (m, d)
        Selected seed points (in PCA space).
    n_bins : int
        Number of histogram bins per dimension.

    Returns
    -------
    nhop : float
        Average NHOP in [0, 1]. 1 = perfect marginal match.
    """
    X_original = np.asarray(X_original, dtype=float)
    X_seeds = np.asarray(X_seeds, dtype=float)
    d = X_original.shape[1]

    hops = np.zeros(d)
    for dim in range(d):
        vals_orig = X_original[:, dim]
        vals_seed = X_seeds[:, dim]

        # Common range from both sets
        lo = min(vals_orig.min(), vals_seed.min())
        hi = max(vals_orig.max(), vals_seed.max())
        if hi - lo < EPS:
            hops[dim] = 1.0
            continue

        # Min-max normalize both to [0, 1] using common range
        vals_orig_norm = (vals_orig - lo) / (hi - lo)
        vals_seed_norm = (vals_seed - lo) / (hi - lo)

        # Shared bin edges on normalized [0, 1] range
        edges = np.linspace(0.0, 1.0, n_bins + 1)

        # Normalized histograms
        h_orig, _ = np.histogram(vals_orig_norm, bins=edges, density=True)
        h_seed, _ = np.histogram(vals_seed_norm, bins=edges, density=True)

        # Convert to probabilities
        bin_width = edges[1] - edges[0]
        p_orig = h_orig * bin_width
        p_seed = h_seed * bin_width

        # Overlap = sum of min
        hops[dim] = np.sum(np.minimum(p_orig, p_seed))

    return float(np.mean(hops))


# Backward compatibility alias
histogram_overlap_percentage = normalized_histogram_overlap_percentage


def jensen_shannon_divergence(X_original, X_seeds, n_bins=30):
    """
    Compute Jensen-Shannon divergence averaged across dimensions.

    Lower is better (0 = identical distributions).

    Parameters
    ----------
    X_original : ndarray, shape (n, d)
    X_seeds : ndarray, shape (m, d)
    n_bins : int

    Returns
    -------
    js : float
        Average JS divergence across dimensions.
    """
    X_original = np.asarray(X_original, dtype=float)
    X_seeds = np.asarray(X_seeds, dtype=float)
    d = X_original.shape[1]

    js_vals = np.zeros(d)
    for dim in range(d):
        vals_orig = X_original[:, dim]
        vals_seed = X_seeds[:, dim]

        lo = min(vals_orig.min(), vals_seed.min())
        hi = max(vals_orig.max(), vals_seed.max())
        if hi - lo < EPS:
            js_vals[dim] = 0.0
            continue

        edges = np.linspace(lo, hi, n_bins + 1)
        bin_width = edges[1] - edges[0]

        h_orig, _ = np.histogram(vals_orig, bins=edges, density=True)
        h_seed, _ = np.histogram(vals_seed, bins=edges, density=True)

        p = h_orig * bin_width + EPS
        q = h_seed * bin_width + EPS

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        m = 0.5 * (p + q)
        js_vals[dim] = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

    return float(np.mean(js_vals))


def geometric_similarity(X_original, X_seeds):
    """
    Compare mean and covariance structure between original and seed sets.

    Uses normalized Frobenius distance between covariance matrices and
    Euclidean distance between means.

    Parameters
    ----------
    X_original : ndarray, shape (n, d)
    X_seeds : ndarray, shape (m, d)

    Returns
    -------
    sim : float
        Similarity in [0, 1]. 1 = identical mean and covariance.
    """
    mean_orig = np.mean(X_original, axis=0)
    mean_seed = np.mean(X_seeds, axis=0)

    # Mean similarity
    mean_dist = np.linalg.norm(mean_orig - mean_seed)
    scale = np.linalg.norm(mean_orig) + EPS
    mean_sim = max(0.0, 1.0 - mean_dist / scale)

    # Covariance similarity
    if X_original.shape[0] > 1 and X_seeds.shape[0] > 1:
        cov_orig = np.cov(X_original.T)
        cov_seed = np.cov(X_seeds.T)
        if cov_orig.ndim == 0:
            cov_orig = np.array([[cov_orig]])
            cov_seed = np.array([[cov_seed]])
        cov_dist = np.linalg.norm(cov_orig - cov_seed, "fro")
        cov_scale = np.linalg.norm(cov_orig, "fro") + EPS
        cov_sim = max(0.0, 1.0 - cov_dist / cov_scale)
    else:
        cov_sim = 0.0

    return 0.5 * (mean_sim + cov_sim)


def topological_similarity(X_original, X_seeds, k=5):
    """
    Compare kNN distance distributions between original and seed sets.

    Computes kNN distances for both sets and measures histogram overlap
    of the distance distributions.

    Parameters
    ----------
    X_original : ndarray, shape (n, d)
    X_seeds : ndarray, shape (m, d)
    k : int
        Number of nearest neighbors.

    Returns
    -------
    sim : float
        Similarity in [0, 1]. 1 = identical kNN distance distributions.
    """
    n_orig = X_original.shape[0]
    n_seed = X_seeds.shape[0]

    k_orig = min(k + 1, n_orig)
    k_seed = min(k + 1, n_seed)

    if k_orig <= 1 or k_seed <= 1:
        return 0.0

    nn_orig = NearestNeighbors(n_neighbors=k_orig).fit(X_original)
    d_orig, _ = nn_orig.kneighbors(X_original)
    knn_dists_orig = d_orig[:, 1:].flatten()

    nn_seed = NearestNeighbors(n_neighbors=k_seed).fit(X_seeds)
    d_seed, _ = nn_seed.kneighbors(X_seeds)
    knn_dists_seed = d_seed[:, 1:].flatten()

    # Histogram overlap of kNN distances
    lo = min(knn_dists_orig.min(), knn_dists_seed.min())
    hi = max(knn_dists_orig.max(), knn_dists_seed.max())
    if hi - lo < EPS:
        return 1.0

    n_bins = 30
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_width = edges[1] - edges[0]

    h_orig, _ = np.histogram(knn_dists_orig, bins=edges, density=True)
    h_seed, _ = np.histogram(knn_dists_seed, bins=edges, density=True)

    p_orig = h_orig * bin_width
    p_seed = h_seed * bin_width

    return float(np.sum(np.minimum(p_orig, p_seed)))


def agtp_score(X_original, X_seeds, k=5):
    """
    Average Geometric + Topological Similarity (AGTP).

    AGTP = 0.5 * (geometric_similarity + topological_similarity)

    Parameters
    ----------
    X_original : ndarray, shape (n, d)
    X_seeds : ndarray, shape (m, d)
    k : int
        k for topological similarity.

    Returns
    -------
    agtp : float
        Score in [0, 1]. Higher is better.
    """
    geo = geometric_similarity(X_original, X_seeds)
    topo = topological_similarity(X_original, X_seeds, k=k)
    return 0.5 * (geo + topo)


def smoothness_score(X_seeds):
    """
    Compute smoothness (Z) of inter-seed distances.

    Measures the regularity of spacing between seeds. Lower Z means
    more evenly distributed seeds.

    Computed as the coefficient of variation of sorted nearest-neighbor
    distances among seeds.

    Parameters
    ----------
    X_seeds : ndarray, shape (m, d)

    Returns
    -------
    z : float
        Smoothness score. Lower is better (more regular spacing).
    """
    m = X_seeds.shape[0]
    if m <= 1:
        return 0.0

    k = min(2, m)
    nn = NearestNeighbors(n_neighbors=k).fit(X_seeds)
    d, _ = nn.kneighbors(X_seeds)
    nn_dists = d[:, 1] if d.shape[1] > 1 else d[:, 0]

    # Coefficient of variation
    mean_d = np.mean(nn_dists)
    std_d = np.std(nn_dists)

    if mean_d < EPS:
        return 0.0

    return float(std_d / mean_d)
