"""
Visualization of synthetic point generation for each algorithm.

Produces zoomed-out context plots and zoomed-in circle/layer detail plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
from matplotlib.collections import PatchCollection


def plot_before_after(X, y, X_res, y_res, title="", figsize=(12, 5)):
    """
    Side-by-side plot: original vs. oversampled dataset.

    Parameters
    ----------
    X, y : original data
    X_res, y_res : oversampled data
    title : str
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    classes = np.unique(y)
    colors = ["#1f77b4", "#ff7f0e"]

    # Original
    for i, c in enumerate(classes):
        mask = y == c
        label = f"Class {c} (n={mask.sum()})"
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=10, alpha=0.6, label=label)
    ax1.set_title("BEFORE (original)")
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal")

    # Oversampled
    n_orig = len(X)
    for i, c in enumerate(classes):
        mask_orig = (y_res[:n_orig] == c)
        mask_synth = np.zeros(len(y_res), dtype=bool)
        mask_synth[n_orig:] = (y_res[n_orig:] == c)

        if mask_orig.any():
            ax2.scatter(X_res[:n_orig][mask_orig, 0], X_res[:n_orig][mask_orig, 1],
                       c=colors[i], s=10, alpha=0.6, label=f"Class {c} (orig)")
        if mask_synth.any():
            ax2.scatter(X_res[mask_synth, 0], X_res[mask_synth, 1],
                       c=colors[i], s=10, alpha=0.3, marker="x", label=f"Class {c} (synth)")

    ax2.set_title(f"AFTER {title}")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")

    plt.tight_layout()
    return fig


def plot_circle_generation(center, radius, seed, neighbor, synthetic,
                           minority_points=None, cluster_labels=None,
                           gravity_centers=None, title="", figsize=(6, 6)):
    """
    Zoomed-in view of one circle generation step.

    Parameters
    ----------
    center : (2,) circle center
    radius : float
    seed : (2,) seed point
    neighbor : (2,) neighbor point
    synthetic : (2,) generated point
    minority_points : (n, 2) optional context points
    cluster_labels : (n,) optional cluster assignments
    gravity_centers : (K, 2) optional gravity center locations
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Circle
    circle = CirclePatch(center, radius, fill=False, edgecolor="gray",
                         linewidth=1.5, linestyle="--")
    ax.add_patch(circle)

    # Context minority points
    if minority_points is not None:
        if cluster_labels is not None:
            unique_labels = np.unique(cluster_labels)
            cmap = plt.cm.Set2
            for i, cl in enumerate(unique_labels):
                mask = cluster_labels == cl
                ax.scatter(minority_points[mask, 0], minority_points[mask, 1],
                          c=[cmap(i)], s=15, alpha=0.4, label=f"Cluster {cl}")
        else:
            ax.scatter(minority_points[:, 0], minority_points[:, 1],
                      c="steelblue", s=15, alpha=0.4, label="Minority")

    # Gravity centers
    if gravity_centers is not None:
        ax.scatter(gravity_centers[:, 0], gravity_centers[:, 1],
                  c="purple", s=100, marker="D", edgecolors="black",
                  linewidths=1, zorder=5, label="Gravity centers")

    # Seed, neighbor, synthetic
    ax.scatter(*seed, c="blue", s=80, marker="X", zorder=5, label="Seed $x_i$")
    ax.scatter(*neighbor, c="green", s=80, marker="+", linewidths=2, zorder=5,
              label="Neighbor $x_j$")
    ax.scatter(*synthetic, c="red", s=80, marker="*", zorder=5, label="Synthetic")

    # Circle center
    ax.scatter(*center, c="black", s=30, marker="o", zorder=5)

    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    margin = radius * 0.3
    ax.set_xlim(center[0] - radius - margin, center[0] + radius + margin)
    ax.set_ylim(center[1] - radius - margin, center[1] + radius + margin)

    plt.tight_layout()
    return fig


def plot_vonmises_circle(center, radius, kappa, mu, seed, neighbor, synthetic,
                         title="", figsize=(6, 6)):
    """
    Zoomed-in circle with Von Mises probability shading.

    Parameters
    ----------
    center : (2,) circle center
    radius : float
    kappa : float, Von Mises concentration
    mu : float, Von Mises mean direction (radians)
    seed, neighbor, synthetic : (2,) points
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Shaded probability on circle
    theta_grid = np.linspace(-np.pi, np.pi, 200)
    r_grid = np.linspace(0, radius, 50)
    T, R = np.meshgrid(theta_grid, r_grid)

    # Von Mises density on angle
    from scipy.stats import vonmises
    vm_pdf = vonmises.pdf(T, kappa, loc=mu)
    # Uniform on radius (weighted by r for area element)
    density = vm_pdf * R / (radius**2 / 2 + 1e-12)

    X_grid = center[0] + R * np.cos(T)
    Y_grid = center[1] + R * np.sin(T)

    ax.pcolormesh(X_grid, Y_grid, density, cmap="YlOrRd", alpha=0.5, shading="auto")

    # Circle outline
    circle = CirclePatch(center, radius, fill=False, edgecolor="gray",
                         linewidth=1.5, linestyle="--")
    ax.add_patch(circle)

    # Direction rays for each cluster
    ray_len = radius * 1.1
    ax.annotate("", xy=(center[0] + ray_len * np.cos(mu),
                        center[1] + ray_len * np.sin(mu)),
               xytext=center,
               arrowprops=dict(arrowstyle="->", color="red", lw=2))

    # Points
    ax.scatter(*seed, c="blue", s=80, marker="X", zorder=5, label="Seed")
    ax.scatter(*neighbor, c="green", s=80, marker="+", linewidths=2, zorder=5, label="Neighbor")
    ax.scatter(*synthetic, c="red", s=80, marker="*", zorder=5, label="Synthetic")

    ax.set_title(f"{title}\n$\\kappa$={kappa:.2f}, $\\mu$={np.degrees(mu):.1f}°")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    margin = radius * 0.3
    ax.set_xlim(center[0] - radius - margin, center[0] + radius + margin)
    ax.set_ylim(center[1] - radius - margin, center[1] + radius + margin)

    plt.tight_layout()
    return fig


def plot_layered_segments(center, R, n_layers, minority_2d, segments=None,
                          synthetic=None, title="", figsize=(6, 6)):
    """
    Visualization of layered segmental oversampling.

    Parameters
    ----------
    center : (2,) center point
    R : float, outer radius
    n_layers : int
    minority_2d : (n, 2)
    segments : list of (start, end) pairs, optional
    synthetic : (m, 2) synthetic points, optional
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Layer rings
    radii = np.linspace(0, R, n_layers + 1)
    for r in radii:
        circle = CirclePatch(center, r, fill=False, edgecolor="lightgray",
                             linewidth=0.5, linestyle=":")
        ax.add_patch(circle)

    # Outer boundary
    outer = CirclePatch(center, R, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(outer)

    # Minority points colored by layer
    dists = np.linalg.norm(minority_2d - center, axis=1)
    layer_ids = np.searchsorted(radii, dists, side="right") - 1
    layer_ids = np.clip(layer_ids, 0, n_layers - 1)

    cmap = plt.cm.viridis
    for layer in range(n_layers):
        mask = layer_ids == layer
        if mask.any():
            color = cmap(layer / max(n_layers - 1, 1))
            ax.scatter(minority_2d[mask, 0], minority_2d[mask, 1],
                      c=[color], s=15, alpha=0.6)

    # Segments
    if segments is not None:
        for start, end in segments[:20]:  # limit to 20 for readability
            ax.plot([start[0], end[0]], [start[1], end[1]],
                   c="orange", linewidth=1, alpha=0.5)

    # Synthetic points
    if synthetic is not None:
        ax.scatter(synthetic[:, 0], synthetic[:, 1],
                  c="red", s=20, marker="*", alpha=0.7, label="Synthetic")

    # Center
    ax.scatter(*center, c="black", s=50, marker="+", linewidths=2, zorder=5, label="Center")

    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    margin = R * 0.2
    ax.set_xlim(center[0] - R - margin, center[0] + R + margin)
    ax.set_ylim(center[1] - R - margin, center[1] + R + margin)

    plt.tight_layout()
    return fig
