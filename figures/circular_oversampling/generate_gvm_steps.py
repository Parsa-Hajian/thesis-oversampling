#!/usr/bin/env python3
"""
Generate step-by-step PDF visualizations for GVM-CO algorithm (all 3 variations).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.stats import vonmises
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import os

# ── Global settings ──────────────────────────────────────────────────────────
SEED = 42
DPI = 300
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

C_MAJ = '#4878CF'
C_MIN = '#D65F5F'
C_SYN = '#2CA02C'
C_BORDER = '#FF8C00'
C_LIGHT = '#CCCCCC'
C_DARK = '#333333'
C_PURPLE = '#9467BD'
C_TEAL = '#17BECF'
CLUSTER_COLORS = ['#D65F5F', '#E6A817', '#9467BD']


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format='pdf')
    plt.close(fig)
    print(f'  Saved {path}')


def make_dataset(seed=SEED):
    rng = np.random.RandomState(seed)
    maj1 = rng.randn(25, 2) * 0.7 + np.array([3.0, 3.0])
    maj2 = rng.randn(20, 2) * 0.6 + np.array([-1.5, -0.5])
    maj3 = rng.randn(15, 2) * 0.5 + np.array([1.0, -2.0])
    X_maj = np.vstack([maj1, maj2, maj3])
    min1 = rng.randn(8, 2) * 0.35 + np.array([0.5, 1.2])
    min2 = rng.randn(6, 2) * 0.30 + np.array([-0.5, 0.5])
    min3 = rng.randn(4, 2) * 0.25 + np.array([1.5, -0.3])
    X_min = np.vstack([min1, min2, min3])
    return X_maj, X_min


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Raw imbalanced dataset
# ═══════════════════════════════════════════════════════════════════════════
def gvm_step1():
    X_maj, X_min = make_dataset()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=45, alpha=0.5, edgecolors='white',
               linewidth=0.3, label=f'Majority ($n={len(X_maj)}$)', zorder=2)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=55, alpha=0.9, edgecolors='black',
               linewidth=0.4, label=f'Minority ($n={len(X_min)}$)', marker='s', zorder=3)
    ax.set_title('Step 1: Imbalanced Dataset', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    # Add IR annotation
    ir = len(X_maj) / len(X_min)
    ax.text(0.98, 0.02, f'IR = {ir:.1f}:1', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    savefig(fig, 'gvm_step1_data.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: K-means clustering of minority
# ═══════════════════════════════════════════════════════════════════════════
def gvm_step2():
    X_maj, X_min = make_dataset()
    km = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(X_min)
    labels = km.labels_
    centers = km.cluster_centers_

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=30, alpha=0.2, edgecolors='none',
               label='Majority', zorder=1)
    for c in range(3):
        mask = labels == c
        ax.scatter(X_min[mask, 0], X_min[mask, 1], c=CLUSTER_COLORS[c], s=60,
                   alpha=0.9, edgecolors='black', linewidth=0.5, marker='s',
                   label=f'Cluster {c+1} ($n_c={mask.sum()}$)', zorder=3)
    ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=120,
               edgecolors='white', linewidth=1.5, label='Cluster centers', zorder=4)
    # Draw convex hull for each cluster
    from scipy.spatial import ConvexHull
    for c in range(3):
        mask = labels == c
        pts = X_min[mask]
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            for simplex in hull.simplices:
                ax.plot(pts[simplex, 0], pts[simplex, 1], c=CLUSTER_COLORS[c],
                        linewidth=1.0, alpha=0.5, linestyle='--')
    ax.set_title('Step 2: K-Means Clustering of Minority', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    savefig(fig, 'gvm_step2_clustering.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Per-cluster gravity metrics
# ═══════════════════════════════════════════════════════════════════════════
def gvm_step3():
    X_maj, X_min = make_dataset()
    km = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(X_min)
    labels = km.labels_
    centers = km.cluster_centers_

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.15, edgecolors='none', zorder=1)

    for c in range(3):
        mask = labels == c
        pts = X_min[mask]
        mu_c = pts.mean(axis=0)
        spread = np.mean(np.linalg.norm(pts - mu_c, axis=1))

        # Density-weighted centroid (gravity center)
        nn = NearestNeighbors(n_neighbors=min(3, len(pts))).fit(pts)
        dists, _ = nn.kneighbors(pts)
        local_scale = dists[:, -1] + 1e-8
        weights = 1.0 / local_scale
        gravity_center = np.average(pts, weights=weights, axis=0)

        density = len(pts) / (np.pi * spread**2 + 1e-8)
        sparsity = np.mean(dists[:, 1:])

        ax.scatter(pts[:, 0], pts[:, 1], c=CLUSTER_COLORS[c], s=55, alpha=0.9,
                   edgecolors='black', linewidth=0.4, marker='s', zorder=3)

        # Spread circle
        circle = plt.Circle(mu_c, spread, fill=False, edgecolor=CLUSTER_COLORS[c],
                            linewidth=1.2, linestyle='--', alpha=0.6, zorder=2)
        ax.add_patch(circle)

        # Gravity center (star)
        ax.scatter(*gravity_center, c=CLUSTER_COLORS[c], marker='*', s=200,
                   edgecolors='black', linewidth=0.8, zorder=5)

        # Annotation
        ax.annotate(f'$C_{c+1}$: $\\rho$={density:.1f}\n'
                    f'spread={spread:.2f}\nsparsity={sparsity:.2f}',
                    xy=mu_c, xytext=(mu_c[0]+0.6, mu_c[1]+0.5),
                    fontsize=7.5, ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=CLUSTER_COLORS[c], alpha=0.15),
                    arrowprops=dict(arrowstyle='->', color=C_DARK, lw=0.8))

    # Legend
    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CLUSTER_COLORS[i],
               markersize=8, label=f'Cluster {i+1}') for i in range(3)]
    handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                          markersize=12, label='Gravity center'))
    handles.append(Line2D([0], [0], linestyle='--', color='gray', label='Spread radius'))
    ax.legend(handles=handles, loc='upper left', framealpha=0.9, fontsize=7.5)
    ax.set_title('Step 3: Cluster Metrics & Gravity Centers', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    savefig(fig, 'gvm_step3_gravity.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: KNN pairing
# ═══════════════════════════════════════════════════════════════════════════
def gvm_step4():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED)
    seeds = rng.choice(len(X_min), 8, replace=False)
    seed_pts = X_min[seeds]

    nn = NearestNeighbors(n_neighbors=2).fit(X_min)
    _, idx = nn.kneighbors(seed_pts)
    neighbors = idx[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.15, edgecolors='none', zorder=1)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=40, alpha=0.4, edgecolors='none',
               marker='s', zorder=2)

    # Highlight seeds
    ax.scatter(seed_pts[:, 0], seed_pts[:, 1], c=C_MIN, s=80, alpha=1.0,
               edgecolors='black', linewidth=1.0, marker='s', label='Selected seeds', zorder=4)

    # Draw pairing lines
    for i in range(len(seeds)):
        p1 = seed_pts[i]
        p2 = X_min[neighbors[i]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=C_BORDER, linewidth=1.5,
                alpha=0.8, zorder=3)
        ax.scatter(*p2, c=C_BORDER, s=60, marker='D', edgecolors='black',
                   linewidth=0.5, zorder=4)

    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_MIN,
               markeredgecolor='black', markersize=9, label='Seed'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=C_BORDER,
               markeredgecolor='black', markersize=8, label='KNN neighbor'),
        Line2D([0], [0], color=C_BORDER, linewidth=1.5, label='Pair connection'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9)
    ax.set_title('Step 4: KNN Seed–Neighbor Pairing', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    savefig(fig, 'gvm_step4_pairing.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Circle construction from pairs
# ═══════════════════════════════════════════════════════════════════════════
def gvm_step5():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED)
    seeds = rng.choice(len(X_min), 6, replace=False)
    seed_pts = X_min[seeds]

    nn = NearestNeighbors(n_neighbors=2).fit(X_min)
    _, idx = nn.kneighbors(seed_pts)
    neighbors = idx[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.12, edgecolors='none', zorder=1)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=35, alpha=0.3, edgecolors='none',
               marker='s', zorder=2)

    for i in range(len(seeds)):
        p1 = seed_pts[i]
        p2 = X_min[neighbors[i]]
        center = (p1 + p2) / 2
        radius = np.linalg.norm(p1 - p2) / 2

        circle = plt.Circle(center, radius, fill=True, facecolor=C_SYN, alpha=0.08,
                            edgecolor=C_SYN, linewidth=1.2, linestyle='-', zorder=2)
        ax.add_patch(circle)

        # Center dot
        ax.scatter(*center, c=C_DARK, s=20, marker='+', linewidths=1.2, zorder=5)

        # Pair points
        ax.scatter(*p1, c=C_MIN, s=70, marker='s', edgecolors='black', linewidth=0.6, zorder=4)
        ax.scatter(*p2, c=C_BORDER, s=60, marker='D', edgecolors='black', linewidth=0.5, zorder=4)

        # Diameter line
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=C_DARK, linewidth=0.6,
                linestyle=':', alpha=0.5, zorder=3)

        # Annotate one circle
        if i == 0:
            ax.annotate(f'$r = \\frac{{d(x_i, x_j)}}{{2}} = {radius:.2f}$',
                        xy=center, xytext=(center[0]+0.8, center[1]+0.6),
                        fontsize=8, ha='left',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color=C_DARK, lw=0.8))

    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_MIN,
               markeredgecolor='black', markersize=8, label='Seed $x_i$'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=C_BORDER,
               markeredgecolor='black', markersize=7, label='Neighbor $x_j$'),
        Line2D([0], [0], marker='+', color=C_DARK, markersize=8, linestyle='None',
               label='Circle center'),
        plt.Circle((0, 0), 0.1, fill=True, facecolor=C_SYN, alpha=0.15,
                   edgecolor=C_SYN, label='Generation circle'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9)
    ax.set_title('Step 5: Circle Construction from Pairs', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    savefig(fig, 'gvm_step5_circles.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Von Mises directional bias
# ═══════════════════════════════════════════════════════════════════════════
def gvm_step6():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Von Mises on a circle
    ax1 = axes[0]
    center = np.array([0, 0])
    radius = 1.0
    gravity_dir = np.array([0.7, 0.5])
    gravity_dir = gravity_dir / np.linalg.norm(gravity_dir)
    mu_angle = np.arctan2(gravity_dir[1], gravity_dir[0])

    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta),
             c=C_DARK, linewidth=1.0, alpha=0.3)

    # Von Mises samples for different kappa
    rng = np.random.RandomState(SEED)
    for kappa, color, label_k in [(2, '#AADDAA', '$\\kappa=2$ (low)'),
                                   (8, '#55BB55', '$\\kappa=8$ (med)'),
                                   (25, '#227722', '$\\kappa=25$ (high)')]:
        angles = vonmises.rvs(kappa, loc=mu_angle, size=80, random_state=rng)
        r_samp = np.sqrt(rng.uniform(0, 1, 80)) * radius
        xs = center[0] + r_samp * np.cos(angles)
        ys = center[1] + r_samp * np.sin(angles)
        ax1.scatter(xs, ys, c=color, s=12, alpha=0.6, label=label_k, zorder=3)

    # Gravity direction arrow
    ax1.annotate('', xy=center + gravity_dir * 1.3, xytext=center,
                 arrowprops=dict(arrowstyle='->', color=C_BORDER, lw=2.5))
    ax1.text(gravity_dir[0] * 1.45, gravity_dir[1] * 1.45, '$\\mu_c$\n(gravity)',
             fontsize=8, ha='center', color=C_BORDER, fontweight='bold')
    ax1.scatter(*center, c='black', s=30, zorder=5, marker='+', linewidths=1.5)
    ax1.set_xlim(-1.7, 1.7); ax1.set_ylim(-1.7, 1.7)
    ax1.set_aspect('equal')
    ax1.set_title('Von Mises Sampling\n(varying $\\kappa$)', fontweight='bold', fontsize=11)
    ax1.legend(loc='lower left', fontsize=7.5, framealpha=0.9)
    ax1.grid(True, alpha=0.15)
    ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$x_2$')

    # Right: Von Mises PDF for different kappa values
    ax2 = axes[1]
    theta_plot = np.linspace(-np.pi, np.pi, 300)
    for kappa, color, ls in [(2, '#AADDAA', '-'), (8, '#55BB55', '--'), (25, '#227722', '-.')]:
        pdf = vonmises.pdf(theta_plot, kappa, loc=0)
        ax2.plot(theta_plot, pdf, color=color, linewidth=2.0, linestyle=ls,
                 label=f'$\\kappa={kappa}$')
    ax2.axvline(0, color=C_BORDER, linestyle=':', linewidth=1.0, alpha=0.6, label='$\\mu_c$ direction')
    ax2.set_xlabel('Angle $\\theta$ (radians)')
    ax2.set_ylabel('Probability density')
    ax2.set_title('Von Mises PDF $f(\\theta; \\mu, \\kappa)$', fontweight='bold', fontsize=11)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax2.set_xlim(-np.pi, np.pi)
    ax2.grid(True, alpha=0.15)

    fig.suptitle('Step 6: Von Mises Directional Bias', fontweight='bold', fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, 'gvm_step6_vonmises.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Final generation with bias visible
# ═══════════════════════════════════════════════════════════════════════════
def gvm_step7():
    X_maj, X_min = make_dataset()
    km = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(X_min)
    labels = km.labels_

    rng = np.random.RandomState(SEED)
    seeds = rng.choice(len(X_min), 8, replace=False)
    seed_pts = X_min[seeds]
    nn_model = NearestNeighbors(n_neighbors=2).fit(X_min)
    _, idx = nn_model.kneighbors(seed_pts)
    neighbors = idx[:, 1]

    # Compute gravity centers per cluster
    gravity_centers = {}
    for c in range(3):
        mask = labels == c
        pts = X_min[mask]
        nn_c = NearestNeighbors(n_neighbors=min(3, len(pts))).fit(pts)
        d, _ = nn_c.kneighbors(pts)
        w = 1.0 / (d[:, -1] + 1e-8)
        gravity_centers[c] = np.average(pts, weights=w, axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.12, edgecolors='none', zorder=1)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=45, alpha=0.5, edgecolors='black',
               linewidth=0.3, marker='s', zorder=2)

    all_syn = []
    for i in range(len(seeds)):
        p1 = seed_pts[i]
        p2 = X_min[neighbors[i]]
        center = (p1 + p2) / 2
        radius = np.linalg.norm(p1 - p2) / 2 + 1e-8
        seed_label = labels[seeds[i]]
        gc = gravity_centers[seed_label]

        # Von Mises direction toward gravity center
        direction = gc - center
        mu_angle = np.arctan2(direction[1], direction[0])
        kappa = 8.0

        circle = plt.Circle(center, radius, fill=False, edgecolor=C_SYN,
                            linewidth=0.8, linestyle='--', alpha=0.4, zorder=2)
        ax.add_patch(circle)

        # Generate synthetic points
        n_synth = 5
        angles = vonmises.rvs(kappa, loc=mu_angle, size=n_synth, random_state=rng)
        radii = np.sqrt(rng.uniform(0, 1, n_synth)) * radius
        syn_x = center[0] + radii * np.cos(angles)
        syn_y = center[1] + radii * np.sin(angles)
        syn_pts = np.column_stack([syn_x, syn_y])
        all_syn.append(syn_pts)

    all_syn = np.vstack(all_syn)
    ax.scatter(all_syn[:, 0], all_syn[:, 1], c=C_SYN, s=35, alpha=0.8,
               edgecolors='black', linewidth=0.3, marker='^', zorder=4,
               label=f'Synthetic ($n={len(all_syn)}$)')

    # Gravity centers
    for c in range(3):
        ax.scatter(*gravity_centers[c], c=CLUSTER_COLORS[c], marker='*', s=180,
                   edgecolors='black', linewidth=0.8, zorder=5)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_MAJ,
               markersize=7, alpha=0.5, label='Majority'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_MIN,
               markeredgecolor='black', markersize=8, label='Minority'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=C_SYN,
               markeredgecolor='black', markersize=8, label='Synthetic'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markersize=12, label='Gravity center'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9, fontsize=8)
    ax.set_title('Step 7: Synthetic Points with Von Mises Bias', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    savefig(fig, 'gvm_step7_generation.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Variation A: Standard (same-cluster pairing)
# ═══════════════════════════════════════════════════════════════════════════
def gvm_var_standard():
    X_maj, X_min = make_dataset()
    km = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(X_min)
    labels = km.labels_

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.12, edgecolors='none', zorder=1)

    for c in range(3):
        mask = labels == c
        pts = X_min[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=CLUSTER_COLORS[c], s=55, alpha=0.9,
                   edgecolors='black', linewidth=0.5, marker='s', zorder=3)

        if len(pts) >= 2:
            nn = NearestNeighbors(n_neighbors=2).fit(pts)
            _, idx = nn.kneighbors(pts)
            for i in range(len(pts)):
                j = idx[i, 1]
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                        c=CLUSTER_COLORS[c], linewidth=1.5, alpha=0.5, zorder=2)
                center = (pts[i] + pts[j]) / 2
                radius = np.linalg.norm(pts[i] - pts[j]) / 2
                circ = plt.Circle(center, radius, fill=True, facecolor=CLUSTER_COLORS[c],
                                  alpha=0.04, edgecolor=CLUSTER_COLORS[c],
                                  linewidth=0.6, linestyle='--', zorder=1)
                ax.add_patch(circ)

    handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=CLUSTER_COLORS[i],
                      markeredgecolor='black', markersize=8, label=f'Cluster {i+1}')
               for i in range(3)]
    handles.append(Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.5,
                          label='Same-cluster pair'))
    ax.legend(handles=handles, loc='upper left', framealpha=0.9, fontsize=8)
    ax.set_title('Variation A: Standard (Same-Cluster Pairing)', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    ax.text(0.98, 0.02, 'Pairs within same cluster\n→ preserves cluster structure',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    savefig(fig, 'gvm_var_standard.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Variation B: Cross-cluster pairing
# ═══════════════════════════════════════════════════════════════════════════
def gvm_var_crosscluster():
    X_maj, X_min = make_dataset()
    km = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(X_min)
    labels = km.labels_
    centers = km.cluster_centers_

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.12, edgecolors='none', zorder=1)

    for c in range(3):
        mask = labels == c
        ax.scatter(X_min[mask, 0], X_min[mask, 1], c=CLUSTER_COLORS[c], s=55, alpha=0.9,
                   edgecolors='black', linewidth=0.5, marker='s', zorder=3)

    # Cross-cluster pairing: for each point, find nearest in different cluster
    rng = np.random.RandomState(SEED)
    drawn_pairs = set()
    for i in range(len(X_min)):
        ci = labels[i]
        other_mask = labels != ci
        other_pts = X_min[other_mask]
        other_indices = np.where(other_mask)[0]
        if len(other_pts) == 0:
            continue
        dists = np.linalg.norm(other_pts - X_min[i], axis=1)
        j_local = np.argmin(dists)
        j = other_indices[j_local]
        pair_key = (min(i, j), max(i, j))
        if pair_key in drawn_pairs:
            continue
        drawn_pairs.add(pair_key)

        ax.plot([X_min[i, 0], X_min[j, 0]], [X_min[i, 1], X_min[j, 1]],
                c=C_BORDER, linewidth=1.2, alpha=0.6, linestyle='-', zorder=2)
        center = (X_min[i] + X_min[j]) / 2
        radius = np.linalg.norm(X_min[i] - X_min[j]) / 2
        circ = plt.Circle(center, radius, fill=True, facecolor=C_BORDER,
                          alpha=0.04, edgecolor=C_BORDER,
                          linewidth=0.6, linestyle='--', zorder=1)
        ax.add_patch(circ)

    handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=CLUSTER_COLORS[i],
                      markeredgecolor='black', markersize=8, label=f'Cluster {i+1}')
               for i in range(3)]
    handles.append(Line2D([0], [0], color=C_BORDER, linewidth=1.5, alpha=0.6,
                          label='Cross-cluster pair'))
    ax.legend(handles=handles, loc='upper left', framealpha=0.9, fontsize=8)
    ax.set_title('Variation B: Cross-Cluster (Border) Pairing', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    ax.text(0.98, 0.02, 'Pairs across cluster boundaries\n→ fills inter-cluster gaps',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    savefig(fig, 'gvm_var_crosscluster.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Variation C: Full-dataset clustering
# ═══════════════════════════════════════════════════════════════════════════
def gvm_var_fulldataset():
    X_maj, X_min = make_dataset()
    X_all = np.vstack([X_maj, X_min])
    y_all = np.array([0]*len(X_maj) + [1]*len(X_min))

    km = KMeans(n_clusters=5, random_state=SEED, n_init=10).fit(X_all)
    all_labels = km.labels_

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: full-dataset clustering
    ax1 = axes[0]
    fd_colors = ['#4878CF', '#D65F5F', '#2CA02C', '#FF8C00', '#9467BD']
    for c in range(5):
        mask = all_labels == c
        maj_in = mask & (y_all == 0)
        min_in = mask & (y_all == 1)
        ax1.scatter(X_all[maj_in, 0], X_all[maj_in, 1], c=fd_colors[c], s=30,
                    alpha=0.4, edgecolors='none', marker='o', zorder=2)
        ax1.scatter(X_all[min_in, 0], X_all[min_in, 1], c=fd_colors[c], s=60,
                    alpha=0.9, edgecolors='black', linewidth=0.5, marker='s', zorder=3)

        # Cluster center
        ax1.scatter(*km.cluster_centers_[c], c=fd_colors[c], marker='X', s=100,
                    edgecolors='black', linewidth=1.0, zorder=4)

        # Minority ratio
        n_min_c = min_in.sum()
        n_total_c = mask.sum()
        rho = n_min_c / (n_total_c + 1e-8)
        ax1.annotate(f'$\\rho_{c+1}$={rho:.2f}', xy=km.cluster_centers_[c],
                     xytext=(km.cluster_centers_[c][0]+0.3, km.cluster_centers_[c][1]+0.4),
                     fontsize=7.5, ha='left',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor=fd_colors[c], alpha=0.2),
                     arrowprops=dict(arrowstyle='->', lw=0.6))

    ax1.set_title('(a) Full-Dataset Clustering\n(K-Means on majority + minority)',
                  fontweight='bold', fontsize=10)
    ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$x_2$')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.15)

    # Right: minority-dominated clusters selected
    ax2 = axes[1]
    ax2.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.12, edgecolors='none', zorder=1)

    for c in range(5):
        mask = all_labels == c
        min_in = mask & (y_all == 1)
        n_min_c = min_in.sum()
        n_total_c = mask.sum()
        rho = n_min_c / (n_total_c + 1e-8)

        if rho > 0.1:  # minority-dominated
            pts = X_all[min_in]
            ax2.scatter(pts[:, 0], pts[:, 1], c=fd_colors[c], s=65, alpha=0.9,
                        edgecolors='black', linewidth=0.6, marker='s', zorder=3)
            # Highlight cluster region
            from scipy.spatial import ConvexHull
            cluster_pts = X_all[mask]
            if len(cluster_pts) >= 3:
                hull = ConvexHull(cluster_pts)
                hull_pts = cluster_pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax2.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.06, color=fd_colors[c])
                ax2.plot(hull_pts[:, 0], hull_pts[:, 1], c=fd_colors[c], linewidth=1.0,
                         linestyle='--', alpha=0.4)
            ax2.annotate(f'Selected\n$\\rho={rho:.2f}$', xy=km.cluster_centers_[c],
                         xytext=(km.cluster_centers_[c][0]+0.5, km.cluster_centers_[c][1]-0.5),
                         fontsize=7.5, color=fd_colors[c], fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color=fd_colors[c], lw=0.8))

    ax2.set_title('(b) Minority-Dominated Clusters\n(oversampling targets)',
                  fontweight='bold', fontsize=10)
    ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$x_2$')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.15)

    fig.suptitle('Variation C: Full-Dataset Clustering (K-Means SMOTE Style)',
                 fontweight='bold', fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, 'gvm_var_fulldataset.pdf')


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating GVM-CO step-by-step figures...')
    gvm_step1()
    gvm_step2()
    gvm_step3()
    gvm_step4()
    gvm_step5()
    gvm_step6()
    gvm_step7()
    gvm_var_standard()
    gvm_var_crosscluster()
    gvm_var_fulldataset()
    print('Done! 10 GVM-CO figures generated.')
