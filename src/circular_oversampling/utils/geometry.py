"""
Geometric sampling and rotation primitives for circular oversampling.

All sampling routines produce *uniform* distributions inside disks (2-D) or
balls (arbitrary dimension).  Vectorised ("batch") variants operate on arrays
of centres / radii so that the inner loop stays inside NumPy.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 2-D disk sampling
# ---------------------------------------------------------------------------

def uniform_in_disk_vec(rng, centers, radii):
    """Sample one point uniformly inside each disk (2-D).

    Uses the square-root-radius trick to ensure uniform area coverage.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator instance.
    centers : ndarray of shape (n, 2)
        Centre of each disk.
    radii : ndarray of shape (n,) or (n, 1)
        Radius of each disk.

    Returns
    -------
    points : ndarray of shape (n, 2)
        One uniformly-sampled point per disk.
    """
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64).ravel()
    n = centers.shape[0]

    # Uniform angle in [0, 2*pi).
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    # Radius scaled by sqrt(U) for uniform area distribution.
    r = radii * np.sqrt(rng.uniform(0.0, 1.0, size=n))

    offsets = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return centers + offsets


# ---------------------------------------------------------------------------
# High-dimensional ball sampling
# ---------------------------------------------------------------------------

def uniform_in_ball(rng, center, R, dim):
    """Sample a single point uniformly inside a *dim*-dimensional ball.

    Uses the standard method: sample a direction uniformly on the unit sphere
    (via normalised Gaussian vector) and a radius distributed as
    ``U^{1/dim}`` to get uniform volume coverage.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator instance.
    center : ndarray of shape (dim,)
        Centre of the ball.
    R : float
        Radius of the ball.
    dim : int
        Dimensionality.

    Returns
    -------
    point : ndarray of shape (dim,)
        A single uniformly-sampled point inside the ball.
    """
    center = np.asarray(center, dtype=np.float64)
    # Random direction on the unit sphere.
    direction = rng.standard_normal(dim)
    norm = np.linalg.norm(direction)
    if norm < 1e-30:
        # Degenerate case -- fall back to the centre itself.
        return center.copy()
    direction /= norm

    # Radial component: r ~ R * U^{1/d} for uniform volume.
    r = R * (rng.uniform() ** (1.0 / dim))
    return center + r * direction


def uniform_in_ball_batch(rng, centers, radii):
    """Vectorised uniform sampling inside high-dimensional balls.

    Generates one point per (centre, radius) pair.  The dimensionality is
    inferred from ``centers.shape[1]``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator instance.
    centers : ndarray of shape (n, d)
        Centre of each ball.
    radii : ndarray of shape (n,) or (n, 1)
        Radius of each ball.

    Returns
    -------
    points : ndarray of shape (n, d)
        One uniformly-sampled point per ball.
    """
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64).ravel()
    n, d = centers.shape

    # Random directions on the unit sphere.
    directions = rng.standard_normal((n, d))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    # Guard against near-zero norms (astronomically unlikely but safe).
    norms = np.maximum(norms, 1e-30)
    directions /= norms

    # Radial scaling for uniform volume in d dimensions.
    u = rng.uniform(size=n)
    r = radii * (u ** (1.0 / d))  # shape (n,)

    return centers + r[:, np.newaxis] * directions


# ---------------------------------------------------------------------------
# 2-D rotation utilities
# ---------------------------------------------------------------------------

def rotate_2d(v, angle):
    """Rotate a single 2-D vector by *angle* radians (counter-clockwise).

    Parameters
    ----------
    v : ndarray of shape (2,)
        The vector to rotate.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    v_rot : ndarray of shape (2,)
        The rotated vector.
    """
    v = np.asarray(v, dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1],
                     s * v[0] + c * v[1]])


def rotate_batch_2d(V, angles):
    """Rotate an array of 2-D vectors by corresponding angles.

    Parameters
    ----------
    V : ndarray of shape (n, 2)
        Vectors to rotate (one per row).
    angles : ndarray of shape (n,)
        Per-vector rotation angles in radians.

    Returns
    -------
    V_rot : ndarray of shape (n, 2)
        Rotated vectors.
    """
    V = np.asarray(V, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64).ravel()

    c = np.cos(angles)
    s = np.sin(angles)

    x_rot = c * V[:, 0] - s * V[:, 1]
    y_rot = s * V[:, 0] + c * V[:, 1]

    return np.column_stack([x_rot, y_rot])


# ---------------------------------------------------------------------------
# Circle construction from point pairs
# ---------------------------------------------------------------------------

EPS = 1e-12


def circle_from_pair(x_i, x_j):
    """Compute the circle defined by a pair of points.

    The circle is centred at the midpoint with radius equal to half the
    Euclidean distance between the two points.

    Parameters
    ----------
    x_i, x_j : array-like of shape (d,)
        Endpoints of the pair.

    Returns
    -------
    center : ndarray of shape (d,)
        Midpoint.
    radius : float
        Half the Euclidean distance (clamped above EPS).
    """
    x_i = np.asarray(x_i, dtype=np.float64)
    x_j = np.asarray(x_j, dtype=np.float64)
    center = 0.5 * (x_i + x_j)
    radius = 0.5 * np.linalg.norm(x_i - x_j)
    return center, max(radius, EPS)


# ---------------------------------------------------------------------------
# Points-in-circle test
# ---------------------------------------------------------------------------

def points_in_circle(X, center, radius):
    """Return a boolean mask of points inside a circle (any dimension).

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Point cloud.
    center : array-like of shape (d,)
        Centre of the circle / sphere.
    radius : float
        Radius.

    Returns
    -------
    mask : ndarray of shape (n,), dtype bool
    """
    dists = np.linalg.norm(X - np.asarray(center, dtype=np.float64), axis=1)
    return dists <= radius


# ---------------------------------------------------------------------------
# Von Mises sampling inside a disk
# ---------------------------------------------------------------------------

def vonmises_in_disk(rng, center, radius, mu, kappa, n_samples):
    """Sample points inside a 2-D disk with Von Mises angular bias.

    The angle is drawn from a Von Mises distribution (favouring direction
    *mu* with concentration *kappa*) while the radial component is drawn
    uniformly within the disk (r ~ R * sqrt(U)).

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator instance.
    center : array-like of shape (2,)
        Centre of the disk.
    radius : float
        Radius of the disk.
    mu : float
        Mean direction (radians) for the Von Mises distribution.
    kappa : float
        Concentration parameter (0 = uniform, large = concentrated).
    n_samples : int
        Number of points to draw.

    Returns
    -------
    points : ndarray of shape (n_samples, 2)
        Sampled 2-D coordinates.
    """
    center = np.asarray(center, dtype=np.float64)
    theta = rng.vonmises(mu, kappa, n_samples)
    u = rng.random(n_samples)
    r = radius * np.sqrt(u)
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return points + center


# ---------------------------------------------------------------------------
# Voronoi-style region assignment (by nearest centroid)
# ---------------------------------------------------------------------------

def assign_voronoi(X, centroids):
    """Assign each point to the closest centroid (Voronoi partitioning).

    Parameters
    ----------
    X : ndarray of shape (n, d)
    centroids : ndarray of shape (k, d)

    Returns
    -------
    labels : ndarray of shape (n,), dtype int
        Index of the closest centroid for each point.
    """
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def sample_in_voronoi_cell(rng, all_centroids, target_idx, center, radius,
                           n_samples, max_rejection_iters=500):
    """Sample uniformly in the intersection of a Voronoi cell and a ball.

    Works in any dimension d.  When d == 2 uses :func:`uniform_in_disk_vec`
    for efficiency; for d > 2 uses :func:`uniform_in_ball_batch`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random state.
    all_centroids : ndarray of shape (k, d)
        All Voronoi centroids.
    target_idx : int
        Index of the target cell.
    center : array-like of shape (d,)
        Centre of the bounding ball.
    radius : float
        Radius of the bounding ball.
    n_samples : int
        Desired number of samples.
    max_rejection_iters : int
        Maximum number of rejection iterations before falling back.

    Returns
    -------
    samples : ndarray of shape (n_samples, d)
    """
    if n_samples <= 0:
        return np.empty((0, all_centroids.shape[1]), dtype=np.float64)

    center = np.asarray(center, dtype=np.float64)
    d = center.shape[0]
    collected = []
    remaining = n_samples

    for _ in range(max_rejection_iters):
        batch = max(remaining * 4, 64)
        c_tile = np.tile(center, (batch, 1))
        r_fill = np.full(batch, radius)
        if d == 2:
            candidates = uniform_in_disk_vec(rng, c_tile, r_fill)
        else:
            candidates = uniform_in_ball_batch(rng, c_tile, r_fill)
        labels = assign_voronoi(candidates, all_centroids)
        accept = labels == target_idx
        if accept.any():
            collected.append(candidates[accept])
            remaining -= int(accept.sum())
        if remaining <= 0:
            break

    if len(collected) == 0:
        return all_centroids[target_idx] + rng.normal(
            0, radius * 0.05, (n_samples, d)
        )

    result = np.concatenate(collected, axis=0)[:n_samples]
    if len(result) < n_samples:
        shortfall = n_samples - len(result)
        filler = all_centroids[target_idx] + rng.normal(
            0, radius * 0.05, (shortfall, d)
        )
        result = np.concatenate([result, filler], axis=0)
    return result


# ---------------------------------------------------------------------------
# Von Mises-Fisher (vMF) directional sampling -- d-dimensional generalisation
# ---------------------------------------------------------------------------

def sample_vmf(rng, mu, kappa, n_samples):
    """Sample from the von Mises-Fisher distribution on S^{d-1}.

    Uses the Wood (1994) acceptance-rejection algorithm for d >= 3 and
    falls back to numpy's vonmises sampler for d == 2.

    Parameters
    ----------
    rng : numpy.random.Generator
    mu : ndarray of shape (d,)
        Mean direction (will be L2-normalised internally).
    kappa : float
        Concentration parameter. 0 = uniform on sphere; large = concentrated.
    n_samples : int

    Returns
    -------
    samples : ndarray of shape (n_samples, d)
        Unit vectors sampled from vMF(mu, kappa).

    References
    ----------
    Wood, A. T. A. (1994). Simulation of the von Mises Fisher distribution.
    Communications in Statistics - Simulation and Computation, 23(1), 157-164.
    """
    mu = np.asarray(mu, dtype=np.float64)
    d = len(mu)
    norm = np.linalg.norm(mu)
    if norm < 1e-30:
        # Degenerate: sample uniformly on sphere
        z = rng.standard_normal((n_samples, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z
    mu = mu / norm

    if d == 2:
        # Reduce to standard von Mises
        mu_angle = np.arctan2(mu[1], mu[0])
        angles = rng.vonmises(mu_angle, kappa, n_samples)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    # Wood (1994) algorithm for d >= 3
    # ----------------------------------
    # Step 1: solve for b via Newton's method
    b = _vmf_wood_b(kappa, d)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1) * np.log(1.0 - x0 ** 2)

    w_samples = np.empty(n_samples)
    accepted = 0
    while accepted < n_samples:
        needed = (n_samples - accepted) * 3 + 64  # over-sample for efficiency
        # Beta sample
        z = rng.beta((d - 1) / 2.0, (d - 1) / 2.0, size=needed)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        # Acceptance criterion
        u = rng.uniform(size=needed)
        log_accept = kappa * w + (d - 1) * np.log(1.0 - x0 * w) - c
        mask = np.log(u) < log_accept
        good = w[mask]
        take = min(len(good), n_samples - accepted)
        w_samples[accepted: accepted + take] = good[:take]
        accepted += take

    # Step 2: sample uniform (d-1)-sphere and combine
    v = rng.standard_normal((n_samples, d - 1))
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_norm = np.maximum(v_norm, 1e-30)
    v = v / v_norm  # uniform on S^{d-2}

    w = w_samples[:, np.newaxis]                       # (n, 1)
    x = np.concatenate(
        [np.sqrt(1.0 - w ** 2) * v, w], axis=1       # (n, d)
    )

    # Step 3: rotate so that e_d aligns with mu
    return _vmf_rotate(x, mu)


def _vmf_wood_b(kappa, d):
    """Compute the Wood (1994) parameter b for vMF sampling."""
    dm1 = d - 1
    sqrt_term = np.sqrt(4.0 * kappa ** 2 + dm1 ** 2)
    b = dm1 / (2.0 * kappa + sqrt_term - 2.0 * kappa)
    return b


def _vmf_rotate(x, mu):
    """Rotate samples so that the last standard basis vector e_d maps to mu.

    Uses a Householder reflection: H = I - 2 * v v^T where
    v = (e_d - mu) / ||e_d - mu||.

    Parameters
    ----------
    x : ndarray of shape (n, d)
        Samples with last coordinate aligned with e_d.
    mu : ndarray of shape (d,), unit vector
        Target mean direction.

    Returns
    -------
    ndarray of shape (n, d)
    """
    d = len(mu)
    e_d = np.zeros(d)
    e_d[-1] = 1.0

    diff = e_d - mu
    diff_norm = np.linalg.norm(diff)
    if diff_norm < 1e-10:
        return x  # mu already is e_d
    v = diff / diff_norm

    # Householder: H x = x - 2 * v (v^T x)
    vTx = x @ v        # (n,)
    return x - 2.0 * vTx[:, np.newaxis] * v[np.newaxis, :]


def perturb_directions_nd(V, ang_std, rng):
    """Perturb an array of unit direction vectors with small Gaussian noise.

    Adds isotropic Gaussian noise of standard deviation *ang_std* to each
    direction vector and re-normalises, providing a rotation-like perturbation
    that works in any dimension d.  For d == 2 the result is equivalent to
    rotating by an angle drawn from N(0, ang_std^2).

    Parameters
    ----------
    V : ndarray of shape (n, d)
        Unit direction vectors (will be re-normalised internally).
    ang_std : float
        Standard deviation of the per-component Gaussian noise (radians for
        d == 2; dimensionally equivalent for d > 2).
    rng : numpy.random.Generator

    Returns
    -------
    V_perturbed : ndarray of shape (n, d)
        Perturbed, re-normalised direction vectors.
    """
    V = np.asarray(V, dtype=np.float64)
    noise = rng.normal(0.0, ang_std, size=V.shape)
    V_perturbed = V + noise
    norms = np.linalg.norm(V_perturbed, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    return V_perturbed / norms


def sample_radial_nd(rng, r_inner, r_outer, dim, n):
    """Sample radii uniformly in a d-dimensional spherical shell.

    Uses the volume-uniform formula
    ``r = (U * (r_outer^d - r_inner^d) + r_inner^d)^{1/d}``
    which degenerates to the familiar sqrt trick when d == 2.

    Parameters
    ----------
    rng : numpy.random.Generator
    r_inner, r_outer : ndarray of shape (n,)
        Inner and outer radii for each sample.
    dim : int
        Dimensionality of the ambient space.
    n : int
        Number of samples.

    Returns
    -------
    radii : ndarray of shape (n,)
    """
    u = rng.random(n)
    r_in_d = r_inner ** dim
    r_out_d = r_outer ** dim
    return (u * (r_out_d - r_in_d) + r_in_d) ** (1.0 / dim)


def vmf_in_ball(rng, center, radius, mu, kappa, n_samples):
    """Sample points inside a d-dimensional ball with vMF directional bias.

    The direction of each point from the ball centre is drawn from vMF(mu, kappa);
    the radial distance is drawn uniformly (r ~ R * U^{1/d}).

    Parameters
    ----------
    rng : numpy.random.Generator
    center : ndarray of shape (d,)
    radius : float
    mu : ndarray of shape (d,)
        Mean direction for vMF (will be normalised).
    kappa : float
        vMF concentration.
    n_samples : int

    Returns
    -------
    points : ndarray of shape (n_samples, d)
    """
    center = np.asarray(center, dtype=np.float64)
    d = len(center)
    directions = sample_vmf(rng, mu, kappa, n_samples)   # (n, d), unit vectors
    u = rng.uniform(size=n_samples)
    r = radius * (u ** (1.0 / d))
    return center + r[:, np.newaxis] * directions
