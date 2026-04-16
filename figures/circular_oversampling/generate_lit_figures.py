#!/usr/bin/env python3
"""
Generate high-quality PDF visualizations for ALL oversampling methods
discussed in Chapter 3 of the thesis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Circle, Ellipse, FancyBboxPatch
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import gaussian_kde, vonmises
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
import os

# ── Global settings ──────────────────────────────────────────────────────────
SEED = 42
FIG_W, FIG_H = 6, 5
DPI = 300
OUT_DIR = '/Users/parsahajiannejad/Desktop/circular-oversampling/thesis/figures'

# Academic style
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

# Color palette
C_MAJ = '#4878CF'    # blue majority
C_MIN = '#D65F5F'    # red minority
C_SYN = '#2CA02C'    # green synthetic
C_BORDER = '#FF8C00' # orange for borderline/highlighted
C_LIGHT = '#CCCCCC'  # light gray
C_DARK = '#333333'   # dark
C_PURPLE = '#9467BD'
C_TEAL = '#17BECF'


def make_dataset(seed=SEED):
    """Create a consistent 2D toy dataset: ~40 majority + ~12 minority."""
    rng = np.random.RandomState(seed)
    # Majority class – two loose blobs
    maj1 = rng.randn(22, 2) * 0.8 + np.array([2.0, 2.0])
    maj2 = rng.randn(18, 2) * 0.7 + np.array([-1.0, -0.5])
    X_maj = np.vstack([maj1, maj2])
    # Minority class – small cluster near boundary with some overlap
    X_min = rng.randn(12, 2) * 0.55 + np.array([0.3, 0.8])
    return X_maj, X_min


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format='pdf')
    plt.close(fig)
    print(f'  Saved {path}')


def base_ax(fig, title):
    ax = fig.add_subplot(111)
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    return ax


def plot_base(ax, X_maj, X_min, maj_alpha=0.35, min_alpha=0.8):
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, marker='o', s=40,
               alpha=maj_alpha, edgecolors='white', linewidth=0.4, label='Majority', zorder=2)
    ax.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, marker='s', s=50,
               alpha=min_alpha, edgecolors='black', linewidth=0.5, label='Minority', zorder=3)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SMOTE
# ═══════════════════════════════════════════════════════════════════════════════
def fig_smote():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 1)
    nn = NearestNeighbors(n_neighbors=4).fit(X_min)
    dists, idxs = nn.kneighbors(X_min)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'SMOTE')
    plot_base(ax, X_maj, X_min)

    synth = []
    for i in range(len(X_min)):
        j = idxs[i, rng.randint(1, 4)]
        lam = rng.rand()
        s = X_min[i] + lam * (X_min[j] - X_min[i])
        synth.append(s)
        ax.plot([X_min[i, 0], X_min[j, 0]], [X_min[i, 1], X_min[j, 1]],
                color=C_LIGHT, linewidth=0.8, zorder=1)
        ax.annotate('', xy=(s[0], s[1]),
                     xytext=((X_min[i, 0] + s[0]) / 2, (X_min[i, 1] + s[1]) / 2),
                     arrowprops=dict(arrowstyle='-', color=C_SYN, lw=0.5))

    synth = np.array(synth)
    ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=35,
               edgecolors='black', linewidth=0.4, label='Synthetic', zorder=4)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'Linear interpolation\nbetween $k$-NN pairs',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_smote.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Borderline-SMOTE
# ═══════════════════════════════════════════════════════════════════════════════
def fig_borderline_smote():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 2)
    X_all = np.vstack([X_maj, X_min])
    nn_all = NearestNeighbors(n_neighbors=6).fit(X_all)
    _, idxs_all = nn_all.kneighbors(X_min)

    n_maj = len(X_maj)
    borderline_mask = np.zeros(len(X_min), dtype=bool)
    for i in range(len(X_min)):
        neighbors = idxs_all[i, 1:]
        n_maj_neighbors = np.sum(neighbors < n_maj)
        if 2 <= n_maj_neighbors <= 4:
            borderline_mask[i] = True

    # Ensure at least 4 borderline points for visual clarity
    if borderline_mask.sum() < 4:
        dists_to_maj = np.min(np.linalg.norm(X_min[:, None] - X_maj[None, :], axis=2), axis=1)
        order = np.argsort(dists_to_maj)
        for idx in order:
            borderline_mask[idx] = True
            if borderline_mask.sum() >= 5:
                break

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'Borderline-SMOTE')
    plot_base(ax, X_maj, X_min, min_alpha=0.4)

    # Highlight borderline points
    border_pts = X_min[borderline_mask]
    safe_pts = X_min[~borderline_mask]
    ax.scatter(border_pts[:, 0], border_pts[:, 1], c=C_BORDER, marker='s', s=80,
               edgecolors='black', linewidth=0.8, label='Borderline minority', zorder=5)
    ax.scatter(safe_pts[:, 0], safe_pts[:, 1], c=C_MIN, marker='s', s=50,
               edgecolors='black', linewidth=0.5, alpha=0.4, zorder=3)

    # Draw danger zone
    for pt in border_pts:
        circle = plt.Circle(pt, 0.35, fill=False, linestyle='--', linewidth=0.7,
                            edgecolor=C_BORDER, alpha=0.6, zorder=4)
        ax.add_patch(circle)

    # Generate from borderline only
    nn_min = NearestNeighbors(n_neighbors=3).fit(X_min)
    _, idxs_min = nn_min.kneighbors(border_pts)
    synth = []
    for i in range(len(border_pts)):
        j = idxs_min[i, rng.randint(1, 3)]
        lam = rng.rand()
        s = border_pts[i] + lam * (X_min[j] - border_pts[i])
        synth.append(s)
        ax.plot([border_pts[i, 0], X_min[j, 0]], [border_pts[i, 1], X_min[j, 1]],
                color=C_BORDER, linewidth=0.5, alpha=0.4, zorder=1)

    synth = np.array(synth)
    ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=35,
               edgecolors='black', linewidth=0.4, label='Synthetic', zorder=6)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'Generation only from\nborderline minority points',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_borderline_smote.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Safe-Level SMOTE
# ═══════════════════════════════════════════════════════════════════════════════
def fig_safe_level_smote():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 3)
    X_all = np.vstack([X_maj, X_min])
    n_maj = len(X_maj)
    k = 5
    nn_all = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
    _, idxs_all = nn_all.kneighbors(X_min)

    # Compute safe levels
    safe_levels = np.zeros(len(X_min))
    for i in range(len(X_min)):
        neighbors = idxs_all[i, 1:]
        safe_levels[i] = np.sum(neighbors >= n_maj) / k  # fraction minority neighbors

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'Safe-Level SMOTE')
    plot_base(ax, X_maj, X_min, min_alpha=0.3)

    # Color minority by safe level
    sc = ax.scatter(X_min[:, 0], X_min[:, 1], c=safe_levels, cmap='RdYlGn',
                    marker='s', s=70, edgecolors='black', linewidth=0.6,
                    vmin=0, vmax=1, zorder=5)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Safe level', fontsize=9)

    # Generate – place synthetic closer to safer endpoint
    nn_min = NearestNeighbors(n_neighbors=3).fit(X_min)
    _, idxs_min = nn_min.kneighbors(X_min)
    synth = []
    for i in range(len(X_min)):
        j = idxs_min[i, rng.randint(1, 3)]
        sl_i, sl_j = safe_levels[i], safe_levels[j]
        if sl_i + sl_j == 0:
            ratio = 0.5
        else:
            ratio = sl_j / (sl_i + sl_j)  # bias toward safer
        lam = rng.rand() * ratio
        s = X_min[i] + lam * (X_min[j] - X_min[i])
        synth.append(s)
        ax.annotate('', xy=s, xytext=X_min[i],
                     arrowprops=dict(arrowstyle='->', color=C_SYN, lw=0.6, alpha=0.5))

    synth = np.array(synth)
    ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=30,
               edgecolors='black', linewidth=0.4, label='Synthetic', zorder=6)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'Synthetic points biased\ntoward safer endpoint',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_safe_level_smote.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ADASYN
# ═══════════════════════════════════════════════════════════════════════════════
def fig_adasyn():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 4)
    X_all = np.vstack([X_maj, X_min])
    n_maj = len(X_maj)
    k = 5
    nn_all = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
    _, idxs_all = nn_all.kneighbors(X_min)

    # Difficulty = fraction of majority neighbors
    difficulty = np.zeros(len(X_min))
    for i in range(len(X_min)):
        neighbors = idxs_all[i, 1:]
        difficulty[i] = np.sum(neighbors < n_maj) / k

    # Normalize to get generation weights
    if difficulty.sum() > 0:
        weights = difficulty / difficulty.sum()
    else:
        weights = np.ones(len(X_min)) / len(X_min)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'ADASYN')
    plot_base(ax, X_maj, X_min, min_alpha=0.5)

    # Show generation clouds proportional to difficulty
    nn_min = NearestNeighbors(n_neighbors=3).fit(X_min)
    _, idxs_min = nn_min.kneighbors(X_min)

    all_synth = []
    for i in range(len(X_min)):
        n_gen = max(1, int(weights[i] * 30))
        # Draw cloud radius
        radius = 0.1 + difficulty[i] * 0.4
        circle = plt.Circle(X_min[i], radius, fill=True, alpha=0.1,
                            facecolor=C_BORDER, edgecolor=C_BORDER,
                            linestyle='--', linewidth=0.5, zorder=1)
        ax.add_patch(circle)
        for _ in range(n_gen):
            j = idxs_min[i, rng.randint(1, 3)]
            lam = rng.rand()
            s = X_min[i] + lam * (X_min[j] - X_min[i])
            all_synth.append(s)

    all_synth = np.array(all_synth)
    ax.scatter(all_synth[:, 0], all_synth[:, 1], c=C_SYN, marker='D', s=18,
               alpha=0.6, edgecolors='none', label='Synthetic', zorder=4)

    # Annotate difficulty on a couple of points
    hardest = np.argmax(difficulty)
    easiest = np.argmin(difficulty)
    ax.annotate(f'Hard ($\\hat{{r}}$={difficulty[hardest]:.2f})',
                xy=X_min[hardest], xytext=(X_min[hardest, 0] + 0.7, X_min[hardest, 1] + 0.5),
                fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.7, color=C_DARK),
                color=C_DARK, zorder=10)
    ax.annotate(f'Easy ($\\hat{{r}}$={difficulty[easiest]:.2f})',
                xy=X_min[easiest], xytext=(X_min[easiest, 0] - 1.2, X_min[easiest, 1] - 0.6),
                fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.7, color=C_DARK),
                color=C_DARK, zorder=10)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'More synthetics near\nharder-to-classify points',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_adasyn.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SVM-SMOTE
# ═══════════════════════════════════════════════════════════════════════════════
def fig_svm_smote():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 5)
    X_all = np.vstack([X_maj, X_min])
    y_all = np.array([0] * len(X_maj) + [1] * len(X_min))

    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_all, y_all)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'SVM-SMOTE')

    # Decision boundary
    x_range = np.linspace(X_all[:, 0].min() - 1, X_all[:, 0].max() + 1, 200)
    y_range = np.linspace(X_all[:, 1].min() - 1, X_all[:, 1].max() + 1, 200)
    xx, yy = np.meshgrid(x_range, y_range)
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[C_MAJ, C_DARK, C_MIN],
               linestyles=['--', '-', '--'], linewidths=[0.7, 1.0, 0.7], zorder=1)
    ax.contourf(xx, yy, Z, levels=[-1, 1], colors=[C_LIGHT], alpha=0.15, zorder=0)

    plot_base(ax, X_maj, X_min, min_alpha=0.5)

    # Highlight support vectors that are minority
    sv_indices = svm.support_
    sv_min_mask = y_all[sv_indices] == 1
    sv_min = X_all[sv_indices[sv_min_mask]]
    ax.scatter(sv_min[:, 0], sv_min[:, 1], c=C_BORDER, marker='*', s=180,
               edgecolors='black', linewidth=0.5, label='Minority SVs (seeds)', zorder=7)

    # SMOTE from support vectors
    nn_min = NearestNeighbors(n_neighbors=3).fit(X_min)
    _, idxs_min = nn_min.kneighbors(sv_min)
    synth = []
    for i in range(len(sv_min)):
        for _ in range(2):
            j = idxs_min[i, rng.randint(1, 3)]
            lam = rng.rand()
            s = sv_min[i] + lam * (X_min[j] - sv_min[i])
            synth.append(s)
            ax.plot([sv_min[i, 0], X_min[j, 0]], [sv_min[i, 1], X_min[j, 1]],
                    color=C_BORDER, linewidth=0.5, alpha=0.3, zorder=1)

    synth = np.array(synth)
    ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=35,
               edgecolors='black', linewidth=0.4, label='Synthetic', zorder=6)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'Interpolation from\nminority support vectors',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_svm_smote.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 6. K-Means SMOTE
# ═══════════════════════════════════════════════════════════════════════════════
def fig_kmeans_smote():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 6)
    X_all = np.vstack([X_maj, X_min])
    y_all = np.array([0] * len(X_maj) + [1] * len(X_min))

    km = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(X_all)
    labels = km.labels_

    # Identify minority-enriched clusters
    cluster_min_frac = np.zeros(4)
    for c in range(4):
        mask = labels == c
        if mask.sum() > 0:
            cluster_min_frac[c] = y_all[mask].mean()

    enriched = cluster_min_frac > (len(X_min) / len(X_all))  # above global IR

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, '$K$-Means SMOTE')
    plot_base(ax, X_maj, X_min, maj_alpha=0.25, min_alpha=0.5)

    # Draw cluster boundaries via Voronoi-like shading
    colors_cluster = ['#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896']
    for c in range(4):
        mask = labels == c
        pts = X_all[mask]
        if len(pts) < 3:
            continue
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            verts = pts[hull.vertices]
            # Expand slightly
            center = verts.mean(axis=0)
            verts_exp = center + 1.15 * (verts - center)
            poly = Polygon(verts_exp, alpha=0.12 if enriched[c] else 0.05,
                           facecolor=colors_cluster[c],
                           edgecolor=colors_cluster[c] if not enriched[c] else C_SYN,
                           linewidth=1.5 if enriched[c] else 0.5,
                           linestyle='-' if enriched[c] else '--',
                           zorder=0)
            ax.add_patch(poly)
            if enriched[c]:
                ax.text(center[0], center[1] + 0.15, f'Cluster {c}\n(enriched)',
                        fontsize=7, ha='center', color=C_SYN, fontweight='bold', zorder=10)
        except Exception:
            pass

    # SMOTE within enriched clusters
    synth = []
    for c in range(4):
        if not enriched[c]:
            continue
        mask_min = (labels[len(X_maj):] == c)
        min_in_cluster = X_min[mask_min]
        if len(min_in_cluster) < 2:
            continue
        nn_c = NearestNeighbors(n_neighbors=min(3, len(min_in_cluster))).fit(min_in_cluster)
        _, idxs_c = nn_c.kneighbors(min_in_cluster)
        for i in range(len(min_in_cluster)):
            j = idxs_c[i, rng.randint(1, min(3, len(min_in_cluster)))]
            lam = rng.rand()
            s = min_in_cluster[i] + lam * (min_in_cluster[j] - min_in_cluster[i])
            synth.append(s)

    if len(synth) > 0:
        synth = np.array(synth)
        ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=35,
                   edgecolors='black', linewidth=0.4, label='Synthetic', zorder=6)

    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c='black', marker='+', s=100, linewidth=1.5, label='Centroids', zorder=8)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'SMOTE applied only in\nminority-enriched clusters',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_kmeans_smote.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Cluster-Based Random Oversampling (CBR)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_cbr():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 7)

    km = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit(X_min)
    labels_min = km.labels_

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'Cluster-Based Random Oversampling')
    plot_base(ax, X_maj, X_min, maj_alpha=0.25, min_alpha=0.3)

    cluster_colors = [C_MIN, C_BORDER, C_PURPLE]
    synth_all = []
    for c in range(3):
        mask = labels_min == c
        pts = X_min[mask]
        # Color cluster
        ax.scatter(pts[:, 0], pts[:, 1], c=cluster_colors[c], marker='s', s=60,
                   edgecolors='black', linewidth=0.5, zorder=5,
                   label=f'Minority cluster {c + 1}')
        # Draw convex hull
        if len(pts) >= 3:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(pts)
                verts = pts[hull.vertices]
                center = verts.mean(axis=0)
                verts_exp = center + 1.2 * (verts - center)
                from matplotlib.patches import Polygon
                poly = Polygon(verts_exp, alpha=0.1, facecolor=cluster_colors[c],
                               edgecolor=cluster_colors[c], linewidth=1.0, linestyle='--', zorder=0)
                ax.add_patch(poly)
            except Exception:
                pass

        # Random oversample within cluster (duplicate with noise)
        n_needed = max(3, 5 - len(pts))
        for _ in range(n_needed):
            idx = rng.randint(len(pts))
            s = pts[idx] + rng.randn(2) * 0.08
            synth_all.append(s)

    synth_all = np.array(synth_all)
    ax.scatter(synth_all[:, 0], synth_all[:, 1], c=C_SYN, marker='D', s=30,
               edgecolors='black', linewidth=0.4, label='Synthetic', zorder=6)

    ax.legend(loc='upper left', framealpha=0.9, fontsize=7.5)
    ax.text(0.97, 0.03, 'Each cluster oversampled\nindependently',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_cbr.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 8. KDE Oversampling
# ═══════════════════════════════════════════════════════════════════════════════
def fig_kde():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 8)

    kde = gaussian_kde(X_min.T, bw_method=0.4)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'KDE Oversampling')

    # Density surface
    x_range = np.linspace(X_min[:, 0].min() - 1.5, X_min[:, 0].max() + 1.5, 150)
    y_range = np.linspace(X_min[:, 1].min() - 1.5, X_min[:, 1].max() + 1.5, 150)
    xx, yy = np.meshgrid(x_range, y_range)
    Z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    cf = ax.contourf(xx, yy, Z, levels=10, cmap='Reds', alpha=0.3, zorder=0)
    ax.contour(xx, yy, Z, levels=6, colors=C_MIN, linewidths=0.4, alpha=0.5, zorder=1)

    plot_base(ax, X_maj, X_min, maj_alpha=0.25)

    # Sample from KDE
    synth = kde.resample(size=20, seed=SEED).T
    ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=30,
               edgecolors='black', linewidth=0.4, label='Synthetic (KDE samples)', zorder=6)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('KDE density', fontsize=9)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'Samples drawn from\nestimated density surface',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_kde.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 9. GMM Oversampling
# ═══════════════════════════════════════════════════════════════════════════════
def fig_gmm():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 9)

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=SEED)
    gmm.fit(X_min)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'GMM Oversampling')
    plot_base(ax, X_maj, X_min, maj_alpha=0.25)

    # Draw Gaussian ellipses
    comp_colors = [C_MIN, C_ORANGE_ALT, C_PURPLE] if False else [C_MIN, C_BORDER, C_PURPLE]
    for k in range(3):
        mean = gmm.means_[k]
        cov = gmm.covariances_[k]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        for nsig in [1, 2]:
            w, h = 2 * nsig * np.sqrt(eigenvalues)
            ell = Ellipse(xy=mean, width=w, height=h, angle=angle,
                          fill=False, edgecolor=comp_colors[k],
                          linewidth=1.0 if nsig == 1 else 0.5,
                          linestyle='-' if nsig == 1 else '--', alpha=0.7, zorder=2)
            ax.add_patch(ell)
        ax.plot(mean[0], mean[1], '+', color=comp_colors[k], markersize=12,
                markeredgewidth=2, zorder=8)
        ax.text(mean[0] + 0.15, mean[1] + 0.15, f'$\\mathcal{{N}}_{k + 1}$',
                fontsize=9, color=comp_colors[k], fontweight='bold', zorder=10)

    # Sample from GMM
    synth, _ = gmm.sample(20)
    ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=30,
               edgecolors='black', linewidth=0.4, label='Synthetic (GMM samples)', zorder=6)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'Samples from fitted\nGaussian mixture components',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_gmm.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 10. GAN-Based Oversampling (Conceptual diagram)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_gan():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('GAN-Based Oversampling', fontweight='bold', pad=10)

    # Noise input
    box_props = dict(boxstyle='round,pad=0.4', facecolor='#E8E8E8', edgecolor=C_DARK, linewidth=1.2)
    ax.text(1.0, 6.0, 'Noise\n$\\mathbf{z} \\sim \\mathcal{N}(0, I)$', fontsize=9,
            ha='center', va='center', bbox=box_props, zorder=5)

    # Generator
    gen_box = dict(boxstyle='round,pad=0.5', facecolor='#AED581', edgecolor='#558B2F', linewidth=1.5)
    ax.text(4.0, 6.0, 'Generator\n$G(\\mathbf{z})$', fontsize=10,
            ha='center', va='center', bbox=gen_box, fontweight='bold', zorder=5)

    # Arrow noise -> generator
    ax.annotate('', xy=(2.8, 6.0), xytext=(1.8, 6.0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=C_DARK))

    # Fake samples
    ax.text(7.0, 6.0, 'Fake\nminority\nsamples', fontsize=9,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9', edgecolor=C_SYN, linewidth=1.2),
            zorder=5)
    ax.annotate('', xy=(5.8, 6.0), xytext=(5.2, 6.0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=C_DARK))

    # Real minority data
    ax.text(7.0, 2.0, 'Real\nminority\nsamples', fontsize=9,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFCDD2', edgecolor=C_MIN, linewidth=1.2),
            zorder=5)

    # Discriminator
    disc_box = dict(boxstyle='round,pad=0.5', facecolor='#90CAF9', edgecolor='#1565C0', linewidth=1.5)
    ax.text(4.0, 2.0, 'Discriminator\n$D(\\mathbf{x})$', fontsize=10,
            ha='center', va='center', bbox=disc_box, fontweight='bold', zorder=5)

    # Arrows to discriminator
    ax.annotate('', xy=(5.5, 2.0), xytext=(5.9, 2.0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=C_DARK))
    ax.annotate('', xy=(4.0, 5.0), xytext=(4.0, 2.9),
                arrowprops=dict(arrowstyle='<-', lw=1.5, color=C_DARK))
    # Connect fake to disc via down+left
    ax.annotate('', xy=(7.0, 5.1), xytext=(7.0, 2.9),
                arrowprops=dict(arrowstyle='-', lw=1.0, color=C_LIGHT, linestyle='--'))

    # Output
    ax.text(1.0, 2.0, 'Real / Fake\n$D(\\mathbf{x}) \\in [0,1]$', fontsize=9,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=1.2),
            zorder=5)
    ax.annotate('', xy=(2.2, 2.0), xytext=(2.8, 2.0),
                arrowprops=dict(arrowstyle='<-', lw=1.5, color=C_DARK))

    # Feedback loop
    ax.annotate('', xy=(1.0, 5.1), xytext=(1.0, 2.9),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#E53935',
                                connectionstyle='arc3,rad=-0.3', linestyle='--'))
    ax.text(-0.2, 4.0, 'Adversarial\nloss', fontsize=7.5, ha='center', va='center',
            color='#E53935', style='italic')

    # Caption
    ax.text(5.0, 0.5, 'Generator learns to produce realistic minority samples;\n'
                       'Discriminator learns to distinguish real from fake.',
            fontsize=8.5, ha='center', va='center', style='italic', color=C_DARK)
    savefig(fig, 'lit_gan.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 11. LoRAS (Manifold-Based)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_loras():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 11)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'LoRAS (Manifold-Based Oversampling)')
    plot_base(ax, X_maj, X_min, maj_alpha=0.25)

    # For each minority point, estimate local affine subspace via neighbors
    nn = NearestNeighbors(n_neighbors=4).fit(X_min)
    _, idxs = nn.kneighbors(X_min)

    synth = []
    for i in range(len(X_min)):
        neighbors = X_min[idxs[i, 1:]]
        centroid = neighbors.mean(axis=0)
        # Local subspace = convex combination with random weights
        for _ in range(2):
            w = rng.dirichlet(np.ones(len(neighbors)))
            s = (w[:, None] * neighbors).sum(axis=0)
            # Add small noise
            s += rng.randn(2) * 0.05
            synth.append(s)
        # Draw local subspace region
        from matplotlib.patches import Polygon
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(neighbors)
            verts = neighbors[hull.vertices]
            poly = Polygon(verts, alpha=0.08, facecolor=C_TEAL,
                           edgecolor=C_TEAL, linewidth=0.7, linestyle='-', zorder=1)
            ax.add_patch(poly)
        except Exception:
            pass

    synth = np.array(synth)
    ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=30,
               edgecolors='black', linewidth=0.4, label='Synthetic', zorder=6)

    # Annotate
    ax.text(0.97, 0.03, 'Convex combinations in\nlocal affine subspaces',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    ax.legend(loc='upper left', framealpha=0.9)
    savefig(fig, 'lit_loras.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 12. RBO (Radial-Based Oversampling)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_rbo():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 12)

    # Local density estimation via kNN distance
    nn = NearestNeighbors(n_neighbors=4).fit(X_min)
    dists, _ = nn.kneighbors(X_min)
    avg_dist = dists[:, 1:].mean(axis=1)
    radii = avg_dist * 0.7  # generation radius proportional to local density

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'RBO (Radial-Based Oversampling)')
    plot_base(ax, X_maj, X_min, maj_alpha=0.25)

    synth = []
    for i in range(len(X_min)):
        r = radii[i]
        circle = plt.Circle(X_min[i], r, fill=True, alpha=0.08,
                            facecolor=C_TEAL, edgecolor=C_TEAL,
                            linewidth=0.8, linestyle='-', zorder=1)
        ax.add_patch(circle)
        # Generate points within radius
        for _ in range(2):
            angle = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0, r)
            s = X_min[i] + rad * np.array([np.cos(angle), np.sin(angle)])
            synth.append(s)

    synth = np.array(synth)
    ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=30,
               edgecolors='black', linewidth=0.4, label='Synthetic', zorder=6)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'Radial generation with radius\nproportional to local density',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_rbo.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 13. DBSMOTE
# ═══════════════════════════════════════════════════════════════════════════════
def fig_dbsmote():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 13)

    db = DBSCAN(eps=0.6, min_samples=2).fit(X_min)
    labels = db.labels_
    core_mask = np.zeros(len(X_min), dtype=bool)
    core_mask[db.core_sample_indices_] = True

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'DBSMOTE')
    plot_base(ax, X_maj, X_min, maj_alpha=0.25, min_alpha=0.3)

    # Highlight core points
    ax.scatter(X_min[core_mask, 0], X_min[core_mask, 1], c=C_BORDER, marker='s', s=70,
               edgecolors='black', linewidth=0.7, label='Core points', zorder=5)
    ax.scatter(X_min[~core_mask, 0], X_min[~core_mask, 1], c=C_MIN, marker='s', s=40,
               edgecolors='black', linewidth=0.4, alpha=0.4, label='Border/noise', zorder=4)

    # Pseudo-centroids per cluster
    unique_labels = set(labels) - {-1}
    synth = []
    for c in unique_labels:
        mask = labels == c
        cluster_pts = X_min[mask]
        centroid = cluster_pts.mean(axis=0)
        ax.plot(centroid[0], centroid[1], '*', color=C_PURPLE, markersize=15,
                markeredgecolor='black', markeredgewidth=0.5, zorder=8)

        # Generate toward pseudo-centroid from core points
        core_in_cluster = X_min[mask & core_mask]
        for pt in core_in_cluster:
            lam = rng.rand() * 0.8
            s = pt + lam * (centroid - pt)
            synth.append(s)
            ax.annotate('', xy=s, xytext=pt,
                        arrowprops=dict(arrowstyle='->', color=C_SYN, lw=0.6, alpha=0.4))

    if len(synth) > 0:
        synth = np.array(synth)
        ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=30,
                   edgecolors='black', linewidth=0.4, label='Synthetic', zorder=6)

    # Add centroid to legend
    ax.plot([], [], '*', color=C_PURPLE, markersize=10, markeredgecolor='black',
            markeredgewidth=0.5, label='Pseudo-centroid')

    ax.legend(loc='upper left', framealpha=0.9, fontsize=7.5)
    ax.text(0.97, 0.03, 'Generation from core points\ntoward pseudo-centroids',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_dbsmote.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Noise Filtering (Tomek Links + ENN)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_noise_filtering():
    X_maj, X_min = make_dataset()
    X_all = np.vstack([X_maj, X_min])
    y_all = np.array([0] * len(X_maj) + [1] * len(X_min))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W * 1.6, FIG_H))
    fig.suptitle('Noise Filtering Methods', fontweight='bold', fontsize=12, y=1.02)

    # ── Tomek Links ──
    ax1.set_title('Tomek Links', fontsize=11)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')

    nn = NearestNeighbors(n_neighbors=2).fit(X_all)
    _, idxs = nn.kneighbors(X_all)

    tomek_pairs = []
    for i in range(len(X_all)):
        j = idxs[i, 1]
        if idxs[j, 1] == i and y_all[i] != y_all[j]:
            if (min(i, j), max(i, j)) not in [(p[0], p[1]) for p in tomek_pairs]:
                tomek_pairs.append((min(i, j), max(i, j)))

    ax1.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, marker='o', s=40,
                alpha=0.35, edgecolors='white', linewidth=0.4, label='Majority', zorder=2)
    ax1.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, marker='s', s=50,
                alpha=0.8, edgecolors='black', linewidth=0.5, label='Minority', zorder=3)

    for i, j in tomek_pairs[:6]:  # Show up to 6 pairs
        ax1.plot([X_all[i, 0], X_all[j, 0]], [X_all[i, 1], X_all[j, 1]],
                 color=C_BORDER, linewidth=1.5, linestyle='-', zorder=4)
        ax1.scatter([X_all[i, 0], X_all[j, 0]], [X_all[i, 1], X_all[j, 1]],
                    facecolors='none', edgecolors=C_BORDER, s=120, linewidth=1.5, zorder=5)
        # X mark on the majority point to remove
        remove_idx = i if y_all[i] == 0 else j
        ax1.scatter(X_all[remove_idx, 0], X_all[remove_idx, 1],
                    marker='x', c='red', s=80, linewidth=2, zorder=6)

    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.text(0.97, 0.03, 'Mutual NN pairs of\nopposite classes removed',
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=7.5,
             style='italic', color=C_DARK,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))

    # Add Tomek link to legend
    ax1.plot([], [], color=C_BORDER, linewidth=1.5, label='Tomek link')
    ax1.scatter([], [], marker='x', c='red', s=60, linewidth=2, label='Removed')
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=7)

    # ── ENN ──
    ax2.set_title('Edited Nearest Neighbours (ENN)', fontsize=11)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')

    k = 3
    nn_enn = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
    _, idxs_enn = nn_enn.kneighbors(X_all)

    misclassified = []
    for i in range(len(X_all)):
        neighbors = idxs_enn[i, 1:]
        neighbor_labels = y_all[neighbors]
        majority_vote = 1 if neighbor_labels.sum() > k / 2 else 0
        if majority_vote != y_all[i]:
            misclassified.append(i)

    ax2.scatter(X_maj[:, 0], X_maj[:, 1], c=C_MAJ, marker='o', s=40,
                alpha=0.35, edgecolors='white', linewidth=0.4, label='Majority', zorder=2)
    ax2.scatter(X_min[:, 0], X_min[:, 1], c=C_MIN, marker='s', s=50,
                alpha=0.8, edgecolors='black', linewidth=0.5, label='Minority', zorder=3)

    for i in misclassified:
        ax2.scatter(X_all[i, 0], X_all[i, 1], facecolors='none',
                    edgecolors=C_BORDER, s=130, linewidth=1.5, zorder=5)
        ax2.scatter(X_all[i, 0], X_all[i, 1], marker='x', c='red',
                    s=80, linewidth=2, zorder=6)

    ax2.scatter([], [], facecolors='none', edgecolors=C_BORDER, s=80,
                linewidth=1.5, label='Misclassified by $k$-NN')
    ax2.scatter([], [], marker='x', c='red', s=60, linewidth=2, label='Removed')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=7)
    ax2.text(0.97, 0.03, 'Points misclassified by\ntheir $k$ neighbours removed',
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=7.5,
             style='italic', color=C_DARK,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))

    fig.tight_layout()
    savefig(fig, 'lit_noise_filtering.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Circular-SMOTE (proposed baseline)
# ═══════════════════════════════════════════════════════════════════════════════
def _circular_smote_core(X_min, rng, n_synth=15):
    """Generate using Circular-SMOTE: uniform sampling within disk between two points."""
    nn = NearestNeighbors(n_neighbors=3).fit(X_min)
    _, idxs = nn.kneighbors(X_min)
    synth = []
    pairs = []
    for _ in range(n_synth):
        i = rng.randint(len(X_min))
        j = idxs[i, rng.randint(1, 3)]
        center = (X_min[i] + X_min[j]) / 2
        radius = np.linalg.norm(X_min[i] - X_min[j]) / 2
        # Uniform sampling in disk
        angle = rng.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(rng.uniform())
        s = center + r * np.array([np.cos(angle), np.sin(angle)])
        synth.append(s)
        pairs.append((i, j, center, radius))
    return np.array(synth), pairs


def fig_circular_smote():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 15)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'Circular-SMOTE (Proposed Baseline)')
    plot_base(ax, X_maj, X_min, maj_alpha=0.25)

    synth, pairs = _circular_smote_core(X_min, rng, n_synth=15)

    # Draw a few representative circles
    drawn = set()
    for idx, (i, j, center, radius) in enumerate(pairs):
        key = (min(i, j), max(i, j))
        if key not in drawn and len(drawn) < 5:
            drawn.add(key)
            circle = plt.Circle(center, radius, fill=True, alpha=0.06,
                                facecolor=C_TEAL, edgecolor=C_TEAL,
                                linewidth=0.8, linestyle='-', zorder=1)
            ax.add_patch(circle)
            # Mark endpoints
            ax.plot([X_min[i, 0], X_min[j, 0]], [X_min[i, 1], X_min[j, 1]],
                    color=C_LIGHT, linewidth=0.6, linestyle=':', zorder=1)

    ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=30,
               edgecolors='black', linewidth=0.4, label='Synthetic', zorder=6)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'Uniform sampling within\ndiameter-defined disk',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_circular_smote.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 16. GVM-CO (proposed Alg. 1)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_gvm_co():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 16)

    gravity_center = X_min.mean(axis=0)

    nn = NearestNeighbors(n_neighbors=3).fit(X_min)
    _, idxs = nn.kneighbors(X_min)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'GVM-CO (Algorithm 1: Von Mises Angular Bias)')
    plot_base(ax, X_maj, X_min, maj_alpha=0.25)

    # Mark gravity center
    ax.plot(gravity_center[0], gravity_center[1], '*', color=C_PURPLE,
            markersize=18, markeredgecolor='black', markeredgewidth=0.6, zorder=9)
    ax.annotate('Gravity center', xy=gravity_center,
                xytext=(gravity_center[0] + 0.8, gravity_center[1] - 0.6),
                fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.7, color=C_DARK),
                color=C_PURPLE, fontweight='bold', zorder=10)

    synth = []
    drawn = set()
    for _ in range(15):
        i = rng.randint(len(X_min))
        j = idxs[i, rng.randint(1, 3)]
        center = (X_min[i] + X_min[j]) / 2
        radius = np.linalg.norm(X_min[i] - X_min[j]) / 2

        # Von Mises angular bias toward gravity center
        dir_to_gravity = gravity_center - center
        mu = np.arctan2(dir_to_gravity[1], dir_to_gravity[0])
        kappa = 2.0  # concentration
        angle = vonmises.rvs(kappa, loc=mu, random_state=rng)
        r = radius * np.sqrt(rng.uniform())
        s = center + r * np.array([np.cos(angle), np.sin(angle)])
        synth.append(s)

        key = (min(i, j), max(i, j))
        if key not in drawn and len(drawn) < 4:
            drawn.add(key)
            circle = plt.Circle(center, radius, fill=True, alpha=0.06,
                                facecolor=C_TEAL, edgecolor=C_TEAL,
                                linewidth=0.8, zorder=1)
            ax.add_patch(circle)
            # Show Von Mises concentration direction
            ax.annotate('', xy=(center[0] + radius * 0.7 * np.cos(mu),
                                center[1] + radius * 0.7 * np.sin(mu)),
                        xytext=center,
                        arrowprops=dict(arrowstyle='->', lw=1.0, color=C_PURPLE, alpha=0.6))

    synth = np.array(synth)
    ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=30,
               edgecolors='black', linewidth=0.4, label='Synthetic', zorder=6)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'Von Mises angular bias\ntoward gravity center',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_gvm_co.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 17. LRE-CO (proposed Alg. 2)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_lre_co():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 17)

    nn = NearestNeighbors(n_neighbors=3).fit(X_min)
    _, idxs = nn.kneighbors(X_min)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'LRE-CO (Algorithm 2: Voronoi Rejection)')
    plot_base(ax, X_maj, X_min, maj_alpha=0.25)

    # Pick a representative pair and show Voronoi inside disk
    i, j = 3, idxs[3, 1]
    center = (X_min[i] + X_min[j]) / 2
    radius = np.linalg.norm(X_min[i] - X_min[j]) / 2

    circle = plt.Circle(center, radius, fill=True, alpha=0.06,
                        facecolor=C_TEAL, edgecolor=C_TEAL,
                        linewidth=1.2, zorder=1)
    ax.add_patch(circle)

    # Create Voronoi seeds inside the disk for illustration
    vor_seeds = []
    for pt in [X_min[i], X_min[j], center]:
        vor_seeds.append(pt)
    # Add nearby minority points
    nearby = X_min[np.linalg.norm(X_min - center, axis=1) < radius * 2]
    for pt in nearby:
        if not any(np.allclose(pt, vs) for vs in vor_seeds):
            vor_seeds.append(pt)
    vor_seeds = np.array(vor_seeds)

    if len(vor_seeds) >= 4:
        try:
            vor = Voronoi(vor_seeds)
            # Draw Voronoi edges clipped to view
            for simplex in vor.ridge_vertices:
                if -1 not in simplex:
                    v0, v1 = vor.vertices[simplex]
                    ax.plot([v0[0], v1[0]], [v0[1], v1[1]],
                            color=C_PURPLE, linewidth=0.6, alpha=0.5, linestyle='--', zorder=2)
        except Exception:
            pass

    # Rejection sampling: generate in disk, keep if in minority Voronoi cell
    synth = []
    rejected = []
    X_all = np.vstack([X_maj, X_min])
    for _ in range(40):
        angle = rng.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(rng.uniform())
        candidate = center + r * np.array([np.cos(angle), np.sin(angle)])
        # Accept if closer to minority than majority
        d_min = np.min(np.linalg.norm(X_min - candidate, axis=1))
        d_maj = np.min(np.linalg.norm(X_maj - candidate, axis=1))
        if d_min < d_maj:
            synth.append(candidate)
        else:
            rejected.append(candidate)
        if len(synth) >= 12:
            break

    if len(synth) > 0:
        synth = np.array(synth)
        ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=30,
                   edgecolors='black', linewidth=0.4, label='Accepted', zorder=6)
    if len(rejected) > 0:
        rejected = np.array(rejected)
        ax.scatter(rejected[:, 0], rejected[:, 1], c='gray', marker='x', s=25,
                   alpha=0.5, linewidth=0.8, label='Rejected', zorder=5)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'Voronoi partitioning with\nrejection sampling in disk',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_lre_co.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 18. LS-CO (proposed Alg. 3)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_ls_co():
    X_maj, X_min = make_dataset()
    rng = np.random.RandomState(SEED + 18)

    nn = NearestNeighbors(n_neighbors=3).fit(X_min)
    _, idxs = nn.kneighbors(X_min)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = base_ax(fig, 'LS-CO (Algorithm 3: Layered Segments)')
    plot_base(ax, X_maj, X_min, maj_alpha=0.25)

    # Pick a representative pair
    i, j = 5, idxs[5, 1]
    center = (X_min[i] + X_min[j]) / 2
    radius = np.linalg.norm(X_min[i] - X_min[j]) / 2

    # Draw layered rings
    n_layers = 4
    n_segments = 6
    colors_rings = plt.cm.YlOrRd(np.linspace(0.15, 0.6, n_layers))
    for layer in range(n_layers):
        r_inner = radius * layer / n_layers
        r_outer = radius * (layer + 1) / n_layers
        for seg in range(n_segments):
            theta1 = 360 * seg / n_segments
            theta2 = 360 * (seg + 1) / n_segments
            wedge = mpatches.Wedge(center, r_outer, theta1, theta2,
                                    width=r_outer - r_inner,
                                    facecolor=colors_rings[layer],
                                    edgecolor='white', linewidth=0.3,
                                    alpha=0.15, zorder=1)
            ax.add_patch(wedge)

    # Outer circle boundary
    circle = plt.Circle(center, radius, fill=False,
                        edgecolor=C_TEAL, linewidth=1.2, zorder=2)
    ax.add_patch(circle)

    # Generate samples in different layers and segments
    synth = []
    for _ in range(15):
        ii = rng.randint(len(X_min))
        jj = idxs[ii, rng.randint(1, 3)]
        c = (X_min[ii] + X_min[jj]) / 2
        rad = np.linalg.norm(X_min[ii] - X_min[jj]) / 2
        # Pick a layer and segment
        layer = rng.randint(n_layers)
        seg = rng.randint(n_segments)
        r_inner = rad * layer / n_layers
        r_outer = rad * (layer + 1) / n_layers
        r = np.sqrt(rng.uniform(r_inner**2, r_outer**2))
        theta = rng.uniform(2 * np.pi * seg / n_segments, 2 * np.pi * (seg + 1) / n_segments)
        s = c + r * np.array([np.cos(theta), np.sin(theta)])
        synth.append(s)

    synth = np.array(synth)
    ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=30,
               edgecolors='black', linewidth=0.4, label='Synthetic', zorder=6)

    # Annotate layers
    ax.annotate('Layer 1\n(inner)', xy=(center[0], center[1] + radius * 0.15),
                fontsize=6.5, ha='center', color=C_DARK, zorder=10)
    ax.annotate('Layer 4\n(outer)', xy=(center[0] + radius * 0.8, center[1]),
                fontsize=6.5, ha='center', color=C_DARK, zorder=10)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.text(0.97, 0.03, 'Layered rings and angular\nsegments within disk',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            style='italic', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_LIGHT, alpha=0.9))
    savefig(fig, 'lit_ls_co.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 19. Comparison: SMOTE vs Circular-SMOTE vs GVM-CO
# ═══════════════════════════════════════════════════════════════════════════════
def fig_comparison():
    X_maj, X_min = make_dataset()

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 2.2, FIG_H * 0.85))
    fig.suptitle('Comparison: SMOTE vs. Circular-SMOTE vs. GVM-CO', fontweight='bold', fontsize=12, y=1.03)

    methods = ['SMOTE', 'Circular-SMOTE', 'GVM-CO']

    for ax_idx, (ax, method) in enumerate(zip(axes, methods)):
        ax.set_title(method, fontsize=11, fontweight='bold')
        ax.set_xlabel('$x_1$')
        if ax_idx == 0:
            ax.set_ylabel('$x_2$')
        plot_base(ax, X_maj, X_min, maj_alpha=0.25)

        rng = np.random.RandomState(SEED + 100 + ax_idx)
        nn = NearestNeighbors(n_neighbors=3).fit(X_min)
        _, idxs = nn.kneighbors(X_min)
        gravity_center = X_min.mean(axis=0)

        synth = []
        for _ in range(15):
            i = rng.randint(len(X_min))
            j = idxs[i, rng.randint(1, 3)]

            if method == 'SMOTE':
                lam = rng.rand()
                s = X_min[i] + lam * (X_min[j] - X_min[i])
            elif method == 'Circular-SMOTE':
                center = (X_min[i] + X_min[j]) / 2
                radius = np.linalg.norm(X_min[i] - X_min[j]) / 2
                angle = rng.uniform(0, 2 * np.pi)
                r = radius * np.sqrt(rng.uniform())
                s = center + r * np.array([np.cos(angle), np.sin(angle)])
            else:  # GVM-CO
                center = (X_min[i] + X_min[j]) / 2
                radius = np.linalg.norm(X_min[i] - X_min[j]) / 2
                dir_to_gravity = gravity_center - center
                mu = np.arctan2(dir_to_gravity[1], dir_to_gravity[0])
                angle = vonmises.rvs(2.0, loc=mu, random_state=rng)
                r = radius * np.sqrt(rng.uniform())
                s = center + r * np.array([np.cos(angle), np.sin(angle)])
            synth.append(s)

            # Draw connection
            if method == 'SMOTE':
                ax.plot([X_min[i, 0], X_min[j, 0]], [X_min[i, 1], X_min[j, 1]],
                        color=C_LIGHT, linewidth=0.5, zorder=1)

        synth = np.array(synth)
        ax.scatter(synth[:, 0], synth[:, 1], c=C_SYN, marker='D', s=25,
                   edgecolors='black', linewidth=0.3, label='Synthetic', zorder=6)

        if method != 'SMOTE':
            # Draw a couple of representative circles
            for k in range(3):
                ii = k * 3
                jj = idxs[ii, 1]
                center = (X_min[ii] + X_min[jj]) / 2
                radius = np.linalg.norm(X_min[ii] - X_min[jj]) / 2
                circle = plt.Circle(center, radius, fill=False, alpha=0.3,
                                    edgecolor=C_TEAL, linewidth=0.6, linestyle='--', zorder=1)
                ax.add_patch(circle)

        if method == 'GVM-CO':
            ax.plot(gravity_center[0], gravity_center[1], '*', color=C_PURPLE,
                    markersize=12, markeredgecolor='black', markeredgewidth=0.4, zorder=9)

        # Set same axis limits
        all_pts = np.vstack([X_maj, X_min])
        margin = 1.0
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

    # Single legend at bottom
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_MAJ, markersize=7, label='Majority'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_MIN, markersize=7, label='Minority'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=C_SYN, markersize=7, label='Synthetic'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, framealpha=0.9, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    savefig(fig, 'lit_comparison.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Generating all oversampling method figures...\n')

    generators = [
        ('SMOTE', fig_smote),
        ('Borderline-SMOTE', fig_borderline_smote),
        ('Safe-Level SMOTE', fig_safe_level_smote),
        ('ADASYN', fig_adasyn),
        ('SVM-SMOTE', fig_svm_smote),
        ('K-Means SMOTE', fig_kmeans_smote),
        ('CBR', fig_cbr),
        ('KDE', fig_kde),
        ('GMM', fig_gmm),
        ('GAN', fig_gan),
        ('LoRAS', fig_loras),
        ('RBO', fig_rbo),
        ('DBSMOTE', fig_dbsmote),
        ('Noise Filtering', fig_noise_filtering),
        ('Circular-SMOTE', fig_circular_smote),
        ('GVM-CO', fig_gvm_co),
        ('LRE-CO', fig_lre_co),
        ('LS-CO', fig_ls_co),
        ('Comparison', fig_comparison),
    ]

    for name, func in generators:
        print(f'[{name}]')
        try:
            func()
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback
            traceback.print_exc()

    print('\nDone. All figures saved to:', OUT_DIR)
