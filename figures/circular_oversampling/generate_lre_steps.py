#!/usr/bin/env python3
"""
Generate step-by-step PDF visualizations for LRE-CO algorithm.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import os

SEED = 42
DPI = 300
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'legend.fontsize': 8.5, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'axes.linewidth': 0.8, 'lines.linewidth': 1.2,
    'figure.dpi': DPI, 'savefig.dpi': DPI, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

C_MAJ = '#4878CF'; C_MIN = '#D65F5F'; C_SYN = '#2CA02C'
C_BORDER = '#FF8C00'; C_LIGHT = '#CCCCCC'; C_DARK = '#333333'
C_PURPLE = '#9467BD'; C_TEAL = '#17BECF'
REGION_COLORS = ['#E6A817', '#9467BD', '#17BECF', '#D65F5F', '#2CA02C']


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format='pdf')
    plt.close(fig)
    print(f'  Saved {path}')


def make_minority(seed=SEED):
    rng = np.random.RandomState(seed)
    min1 = rng.randn(10, 2) * 0.4 + np.array([0.5, 1.0])
    min2 = rng.randn(7, 2) * 0.35 + np.array([-0.5, 0.3])
    min3 = rng.randn(5, 2) * 0.25 + np.array([1.2, -0.2])
    return np.vstack([min1, min2, min3])


def make_dataset(seed=SEED):
    rng = np.random.RandomState(seed)
    maj1 = rng.randn(25, 2) * 0.7 + np.array([3.0, 3.0])
    maj2 = rng.randn(20, 2) * 0.6 + np.array([-1.5, -0.5])
    X_maj = np.vstack([maj1, maj2])
    X_min = make_minority(seed)
    return X_maj, X_min


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: KNN pairing
# ═══════════════════════════════════════════════════════════════════════════
def lre_step1():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED)
    seeds_idx = rng.choice(len(X_min), 8, replace=False)
    seed_pts = X_min[seeds_idx]

    nn = NearestNeighbors(n_neighbors=2).fit(X_min)
    _, idx = nn.kneighbors(seed_pts)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.15, edgecolors='none', zorder=1)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=40, alpha=0.4, edgecolors='none',
               marker='s', zorder=2)
    ax.scatter(seed_pts[:, 0], seed_pts[:, 1], c=C_MIN, s=80, alpha=1.0,
               edgecolors='black', linewidth=1.0, marker='s', zorder=4)
    for i in range(len(seeds_idx)):
        p1 = seed_pts[i]
        p2 = X_min[idx[i, 1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=C_BORDER, linewidth=1.5, alpha=0.8, zorder=3)
        ax.scatter(*p2, c=C_BORDER, s=55, marker='D', edgecolors='black', linewidth=0.5, zorder=4)

    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_MIN,
               markeredgecolor='black', markersize=9, label='Seed'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=C_BORDER,
               markeredgecolor='black', markersize=8, label='KNN neighbor'),
        Line2D([0], [0], color=C_BORDER, linewidth=1.5, label='Pair connection'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9)
    ax.set_title('Step 1: KNN Seed–Neighbor Pairing', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    savefig(fig, 'lre_step1_pairs.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Circle construction with inside points
# ═══════════════════════════════════════════════════════════════════════════
def lre_step2():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED)
    seeds_idx = rng.choice(len(X_min), 6, replace=False)
    seed_pts = X_min[seeds_idx]
    nn = NearestNeighbors(n_neighbors=2).fit(X_min)
    _, idx = nn.kneighbors(seed_pts)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.12, edgecolors='none', zorder=1)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=35, alpha=0.3, edgecolors='none',
               marker='s', zorder=2)

    for i in range(len(seeds_idx)):
        p1 = seed_pts[i]; p2 = X_min[idx[i, 1]]
        center = (p1 + p2) / 2
        radius = np.linalg.norm(p1 - p2) / 2

        # Points inside circle
        dists_to_center = np.linalg.norm(X_min - center, axis=1)
        inside_mask = dists_to_center <= radius
        n_inside = inside_mask.sum()

        color = C_SYN if n_inside >= 3 else '#AAAAAA'
        circ = plt.Circle(center, radius, fill=True, facecolor=color, alpha=0.08,
                          edgecolor=color, linewidth=1.0, linestyle='-', zorder=1)
        ax.add_patch(circ)

        # Highlight inside points
        inside_pts = X_min[inside_mask]
        ax.scatter(inside_pts[:, 0], inside_pts[:, 1], c=color, s=50, marker='s',
                   edgecolors='black', linewidth=0.4, alpha=0.7, zorder=3)

        # Count annotation
        ax.text(center[0], center[1] - radius - 0.12, f'$n_{{in}}={n_inside}$',
                fontsize=7.5, ha='center', va='top', color=color, fontweight='bold')

    ax.set_title('Step 2: Circles with Inside-Point Count', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    savefig(fig, 'lre_step2_circles.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Certainty threshold filtering
# ═══════════════════════════════════════════════════════════════════════════
def lre_step3():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED)
    seeds_idx = rng.choice(len(X_min), 8, replace=False)
    seed_pts = X_min[seeds_idx]
    nn = NearestNeighbors(n_neighbors=2).fit(X_min)
    _, idx = nn.kneighbors(seed_pts)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.12, edgecolors='none', zorder=1)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=40, alpha=0.4, edgecolors='none',
               marker='s', zorder=2)

    for i in range(len(seeds_idx)):
        p1 = seed_pts[i]; p2 = X_min[idx[i, 1]]
        center = (p1 + p2) / 2
        radius = np.linalg.norm(p1 - p2) / 2

        dists = np.linalg.norm(X_min - center, axis=1)
        n_inside = (dists <= radius).sum()

        # Simulate certainty based on inside count
        certainty = min(1.0, n_inside / 5.0) * (0.6 + 0.4 * rng.random())
        accepted = certainty >= 0.80

        if accepted:
            circ = plt.Circle(center, radius, fill=True, facecolor=C_SYN, alpha=0.08,
                              edgecolor=C_SYN, linewidth=1.5, linestyle='-', zorder=1)
            marker_label = f'$\\tau={certainty:.2f}$ (pass)'
            text_color = C_SYN
        else:
            circ = plt.Circle(center, radius, fill=False,
                              edgecolor='#CC3333', linewidth=1.5, linestyle='--', zorder=1)
            marker_label = f'$\\tau={certainty:.2f}$ (fail)'
            text_color = '#CC3333'
        ax.add_patch(circ)
        ax.text(center[0], center[1] + radius + 0.08, marker_label,
                fontsize=7, ha='center', va='bottom', color=text_color, fontweight='bold')

    # Threshold line legend
    handles = [
        Line2D([0], [0], color=C_SYN, linewidth=2, label='Accepted ($\\tau \\geq 0.80$)'),
        Line2D([0], [0], color='#CC3333', linewidth=2, linestyle='--',
               label='Rejected ($\\tau < 0.80$)'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9, fontsize=9)
    ax.set_title('Step 3: Certainty Threshold Filtering ($\\tau \\geq 0.80$)',
                 fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    savefig(fig, 'lre_step3_certainty.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Local K-means inside one circle
# ═══════════════════════════════════════════════════════════════════════════
def lre_step4():
    rng = np.random.RandomState(SEED)
    # Generate points inside a circle for demonstration
    center = np.array([0.0, 0.0])
    radius = 1.5
    n_pts = 30
    angles = rng.uniform(0, 2*np.pi, n_pts)
    radii = np.sqrt(rng.uniform(0, 1, n_pts)) * radius
    pts = np.column_stack([center[0] + radii * np.cos(angles),
                           center[1] + radii * np.sin(angles)])

    km = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(pts)
    local_labels = km.labels_
    local_centers = km.cluster_centers_

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Draw the circle
    circ = plt.Circle(center, radius, fill=True, facecolor='#F0F0F0', alpha=0.3,
                      edgecolor=C_DARK, linewidth=1.5, zorder=1)
    ax.add_patch(circ)

    for c in range(4):
        mask = local_labels == c
        ax.scatter(pts[mask, 0], pts[mask, 1], c=REGION_COLORS[c], s=55,
                   edgecolors='black', linewidth=0.4, marker='s', zorder=3,
                   label=f'Sub-cluster {c+1} ($n={mask.sum()}$)')
    ax.scatter(local_centers[:, 0], local_centers[:, 1], c='black', marker='X', s=100,
               edgecolors='white', linewidth=1.5, zorder=4, label='Sub-cluster centers')

    # Draw connection lines between points and their centers
    for c in range(4):
        mask = local_labels == c
        for pt in pts[mask]:
            ax.plot([pt[0], local_centers[c, 0]], [pt[1], local_centers[c, 1]],
                    c=REGION_COLORS[c], linewidth=0.4, alpha=0.3, zorder=2)

    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=7.5)
    ax.set_title('Step 4: Local K-Means Inside Circle', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.15)
    savefig(fig, 'lre_step4_local_kmeans.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Voronoi partition inside circle
# ═══════════════════════════════════════════════════════════════════════════
def lre_step5():
    rng = np.random.RandomState(SEED)
    center = np.array([0.0, 0.0])
    radius = 1.5

    # Generate inside-circle points and cluster
    n_pts = 30
    angles = rng.uniform(0, 2*np.pi, n_pts)
    radii_pts = np.sqrt(rng.uniform(0, 1, n_pts)) * radius
    pts = np.column_stack([radii_pts * np.cos(angles), radii_pts * np.sin(angles)])

    km = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(pts)
    local_centers = km.cluster_centers_

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Circle boundary
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), c=C_DARK, linewidth=1.5, zorder=5)

    # Voronoi from centers — clip to circle
    # Create a fine grid and assign each point to nearest center
    grid_x = np.linspace(-radius, radius, 300)
    grid_y = np.linspace(-radius, radius, 300)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    in_circle = np.linalg.norm(grid_pts, axis=1) <= radius

    dists_to_centers = np.array([np.linalg.norm(grid_pts - lc, axis=1) for lc in local_centers])
    region_labels = np.argmin(dists_to_centers, axis=0)
    region_labels[~in_circle] = -1

    region_grid = region_labels.reshape(gx.shape)
    for c in range(4):
        region_mask = region_grid == c
        ax.contourf(gx, gy, (region_grid == c).astype(float), levels=[0.5, 1.5],
                    colors=[REGION_COLORS[c]], alpha=0.15)

    # Region probabilities
    counts = np.array([(km.labels_ == c).sum() for c in range(4)])
    probs = counts / counts.sum()
    for c in range(4):
        ax.text(local_centers[c, 0], local_centers[c, 1] - 0.2,
                f'$P_{{R_{c+1}}}={probs[c]:.2f}$',
                fontsize=8, ha='center', fontweight='bold', color=REGION_COLORS[c],
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))

    ax.scatter(pts[:, 0], pts[:, 1], c=[REGION_COLORS[l] for l in km.labels_],
               s=45, edgecolors='black', linewidth=0.4, marker='s', zorder=4)
    ax.scatter(local_centers[:, 0], local_centers[:, 1], c='black', marker='X', s=100,
               edgecolors='white', linewidth=1.5, zorder=5)

    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')
    ax.set_title('Step 5: Voronoi Partition with Region Probabilities',
                 fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.grid(True, alpha=0.1)
    savefig(fig, 'lre_step5_voronoi.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Distribution estimation
# ═══════════════════════════════════════════════════════════════════════════
def lre_step6():
    rng = np.random.RandomState(SEED)
    center = np.array([0.0, 0.0])
    radius = 1.5

    # Points inside a region (simulate one Voronoi cell)
    pts = rng.randn(15, 2) * 0.4 + np.array([0.3, 0.2])
    # Keep only inside circle
    mask = np.linalg.norm(pts - center, axis=1) <= radius
    pts = pts[mask]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: KDE contours
    ax1 = axes[0]
    theta = np.linspace(0, 2*np.pi, 200)
    ax1.plot(radius * np.cos(theta), radius * np.sin(theta), c=C_DARK, linewidth=1.2)

    if len(pts) >= 3:
        kde = gaussian_kde(pts.T, bw_method=0.4)
        grid_x = np.linspace(-radius, radius, 100)
        grid_y = np.linspace(-radius, radius, 100)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_flat = np.vstack([gx.ravel(), gy.ravel()])
        z = kde(grid_flat).reshape(gx.shape)
        # Mask outside circle
        dist_grid = np.sqrt(gx**2 + gy**2)
        z[dist_grid > radius] = np.nan
        ax1.contourf(gx, gy, z, levels=10, cmap='Greens', alpha=0.5)
        ax1.contour(gx, gy, z, levels=6, colors=C_SYN, linewidths=0.6, alpha=0.7)

    ax1.scatter(pts[:, 0], pts[:, 1], c=C_MIN, s=55, edgecolors='black',
                linewidth=0.5, marker='s', zorder=4)
    ax1.set_title('(a) KDE Density Estimation', fontweight='bold', fontsize=10)
    ax1.set_xlim(-2, 2); ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$x_2$')
    ax1.grid(True, alpha=0.1)

    # Right: Sampling from estimated distribution
    ax2 = axes[1]
    ax2.plot(radius * np.cos(theta), radius * np.sin(theta), c=C_DARK, linewidth=1.2)
    ax2.scatter(pts[:, 0], pts[:, 1], c=C_MIN, s=55, edgecolors='black',
                linewidth=0.5, marker='s', zorder=4, label='Original')

    # Generate synthetic from KDE
    if len(pts) >= 3:
        syn = kde.resample(20, seed=SEED).T
        # Clip to circle
        syn_in = syn[np.linalg.norm(syn - center, axis=1) <= radius]
        ax2.scatter(syn_in[:, 0], syn_in[:, 1], c=C_SYN, s=40, marker='^',
                    edgecolors='black', linewidth=0.3, alpha=0.8, zorder=5,
                    label=f'Synthetic ($n={len(syn_in)}$)')

    ax2.set_title('(b) Sampling from Estimated Distribution', fontweight='bold', fontsize=10)
    ax2.set_xlim(-2, 2); ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$x_2$')
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.1)

    fig.suptitle('Step 6: Distribution Estimation & Sampling', fontweight='bold', fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, 'lre_step6_distribution.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Final generation
# ═══════════════════════════════════════════════════════════════════════════
def lre_step7():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED)
    seeds_idx = rng.choice(len(X_min), 6, replace=False)
    seed_pts = X_min[seeds_idx]
    nn = NearestNeighbors(n_neighbors=2).fit(X_min)
    _, idx = nn.kneighbors(seed_pts)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, s=25, alpha=0.12, edgecolors='none', zorder=1)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=45, alpha=0.5, edgecolors='black',
               linewidth=0.3, marker='s', zorder=2)

    all_syn = []
    for i in range(len(seeds_idx)):
        p1 = seed_pts[i]; p2 = X_min[idx[i, 1]]
        center = (p1 + p2) / 2
        radius = np.linalg.norm(p1 - p2) / 2 + 0.01

        circ = plt.Circle(center, radius, fill=False, edgecolor=C_SYN,
                          linewidth=0.8, linestyle='--', alpha=0.4, zorder=1)
        ax.add_patch(circ)

        # Generate from local distribution (uniform for simplicity in viz)
        n_syn = 5
        a = rng.uniform(0, 2*np.pi, n_syn)
        r = np.sqrt(rng.uniform(0, 1, n_syn)) * radius
        syn = np.column_stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a)])
        all_syn.append(syn)

    all_syn = np.vstack(all_syn)
    ax.scatter(all_syn[:, 0], all_syn[:, 1], c=C_SYN, s=35, alpha=0.8,
               edgecolors='black', linewidth=0.3, marker='^', zorder=4,
               label=f'Synthetic ($n={len(all_syn)}$)')

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_MAJ,
               markersize=7, alpha=0.5, label='Majority'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_MIN,
               markeredgecolor='black', markersize=8, label='Minority'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=C_SYN,
               markeredgecolor='black', markersize=8, label='Synthetic'),
    ]
    ax.legend(handles=handles, loc='upper left', framealpha=0.9)
    ax.set_title('Step 7: Final Synthetic Points (LRE-CO)', fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    savefig(fig, 'lre_step7_generation.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Overview: 2x2 panel
# ═══════════════════════════════════════════════════════════════════════════
def lre_overview():
    rng = np.random.RandomState(SEED)
    X_maj, X_min = make_dataset()

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # (a) Circle formation
    ax = axes[0, 0]
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=45, alpha=0.6,
               edgecolors='black', linewidth=0.3, marker='s', zorder=2)
    seeds_idx = rng.choice(len(X_min), 5, replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(X_min)
    _, idx = nn.kneighbors(X_min[seeds_idx])
    for i in range(len(seeds_idx)):
        p1 = X_min[seeds_idx[i]]; p2 = X_min[idx[i, 1]]
        c = (p1 + p2) / 2; r = np.linalg.norm(p1 - p2) / 2
        circ = plt.Circle(c, r, fill=True, facecolor=C_SYN, alpha=0.06,
                          edgecolor=C_SYN, linewidth=1.0, zorder=1)
        ax.add_patch(circ)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=C_BORDER, lw=1.0, alpha=0.6)
    ax.set_title('(a) Circle Formation', fontweight='bold', fontsize=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    # (b) Certainty check
    ax = axes[0, 1]
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=35, alpha=0.4, marker='s', zorder=2)
    for i in range(len(seeds_idx)):
        p1 = X_min[seeds_idx[i]]; p2 = X_min[idx[i, 1]]
        c = (p1 + p2) / 2; r = np.linalg.norm(p1 - p2) / 2
        cert = 0.5 + 0.5 * rng.random()
        ok = cert >= 0.80
        circ = plt.Circle(c, r, fill=ok, facecolor=C_SYN if ok else 'none',
                          alpha=0.08 if ok else 1.0,
                          edgecolor=C_SYN if ok else '#CC3333',
                          linewidth=1.5, linestyle='-' if ok else '--', zorder=1)
        ax.add_patch(circ)
        ax.text(c[0], c[1], f'{cert:.2f}', fontsize=7, ha='center',
                color=C_SYN if ok else '#CC3333', fontweight='bold')
    ax.set_title('(b) Certainty Filtering ($\\tau \\geq 0.80$)', fontweight='bold', fontsize=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    # (c) Local clustering + Voronoi
    ax = axes[1, 0]
    center = np.array([0.5, 0.8])
    radius = 0.8
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta),
            c=C_DARK, linewidth=1.2)
    inside_mask = np.linalg.norm(X_min - center, axis=1) <= radius
    inside_pts = X_min[inside_mask]
    if len(inside_pts) >= 3:
        km = KMeans(n_clusters=min(3, len(inside_pts)), random_state=SEED, n_init=5).fit(inside_pts)
        for cc in range(km.n_clusters):
            m = km.labels_ == cc
            ax.scatter(inside_pts[m, 0], inside_pts[m, 1], c=REGION_COLORS[cc], s=55,
                       edgecolors='black', linewidth=0.4, marker='s', zorder=4)
    ax.set_title('(c) Local K-Means + Voronoi', fontweight='bold', fontsize=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    # (d) Generation
    ax = axes[1, 1]
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, s=40, alpha=0.5,
               edgecolors='black', linewidth=0.3, marker='s', zorder=2)
    syn_pts = rng.randn(25, 2) * 0.3 + np.array([0.3, 0.8])
    ax.scatter(syn_pts[:, 0], syn_pts[:, 1], c=C_SYN, s=30, marker='^',
               edgecolors='black', linewidth=0.3, alpha=0.8, zorder=3)
    ax.set_title('(d) Final Generation', fontweight='bold', fontsize=10)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    fig.suptitle('LRE-CO: Local Region Estimation Circular Oversampling — Overview',
                 fontweight='bold', fontsize=13, y=1.01)
    fig.tight_layout()
    savefig(fig, 'lre_overview.pdf')


if __name__ == '__main__':
    print('Generating LRE-CO step-by-step figures...')
    lre_step1()
    lre_step2()
    lre_step3()
    lre_step4()
    lre_step5()
    lre_step6()
    lre_step7()
    lre_overview()
    print('Done! 8 LRE-CO figures generated.')
