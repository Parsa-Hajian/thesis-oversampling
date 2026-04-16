#!/usr/bin/env python3
"""
Generate step-by-step PDF visualizations for the Seed Selection strategy.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
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
C_BORDER = '#FF8C00'; C_DARK = '#333333'
C_PURPLE = '#9467BD'; C_TEAL = '#17BECF'; C_LIGHT = '#CCCCCC'
CLUSTER_COLORS = ['#D65F5F', '#E6A817', '#9467BD', '#17BECF', '#2CA02C']


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format='pdf')
    plt.close(fig)
    print(f'  Saved {path}')


def make_minority_hd(n=80, d=5, seed=SEED):
    """High-dimensional minority data with clusters."""
    rng = np.random.RandomState(seed)
    c1 = rng.randn(30, d) * 0.5 + np.array([1]*d)
    c2 = rng.randn(25, d) * 0.4 + np.array([-1]*d)
    c3 = rng.randn(15, d) * 0.6 + np.array([0.5, -0.5] + [0]*(d-2))
    c4 = rng.randn(10, d) * 0.3 + np.array([-0.5, 1] + [0]*(d-2))
    return np.vstack([c1, c2, c3, c4])


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Clustering minority class
# ═══════════════════════════════════════════════════════════════════════════
def seed_step1():
    X = make_minority_hd()
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    X_2d = pca.transform(X)
    km = KMeans(n_clusters=5, random_state=SEED, n_init=10).fit(X_2d)

    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(5):
        mask = km.labels_ == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=CLUSTER_COLORS[c], s=50,
                   alpha=0.8, edgecolors='black', linewidth=0.3, marker='s',
                   label=f'Cluster {c+1} ($n={mask.sum()}$)')
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c='black', marker='X', s=120, edgecolors='white', linewidth=1.5, zorder=5)

    # Proportional allocation annotation
    total_seeds = 20
    allocs = [(km.labels_ == c).sum() / len(X) * total_seeds for c in range(5)]
    for c in range(5):
        ax.annotate(f'$m_{c+1}={allocs[c]:.0f}$',
                    xy=km.cluster_centers_[c], xytext=(km.cluster_centers_[c][0]+0.5,
                    km.cluster_centers_[c][1]+0.5),
                    fontsize=7.5, color=CLUSTER_COLORS[c],
                    arrowprops=dict(arrowstyle='->', lw=0.6, color=CLUSTER_COLORS[c]),
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))

    ax.set_title('Step 1: K-Means Clustering for Stratified Sampling', fontweight='bold', pad=10)
    ax.set_xlabel('PC$_1$'); ax.set_ylabel('PC$_2$')
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    savefig(fig, 'seed_step1_cluster.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: PCA projection
# ═══════════════════════════════════════════════════════════════════════════
def seed_step2():
    X = make_minority_hd()
    pca = PCA(random_state=SEED).fit(X)
    var_ratio = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: 2D projection
    ax1 = axes[0]
    X_2d = pca.transform(X)[:, :2]
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=C_MIN, s=40, alpha=0.7,
                edgecolors='black', linewidth=0.3, marker='s')
    ax1.set_title('(a) PCA Projection ($k=2$)', fontweight='bold', fontsize=10)
    ax1.set_xlabel('PC$_1$'); ax1.set_ylabel('PC$_2$')
    ax1.grid(True, alpha=0.15)
    ax1.text(0.02, 0.98, f'Explained var: {var_ratio[0]+var_ratio[1]:.1%}',
             transform=ax1.transAxes, ha='left', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # Right: explained variance bar chart
    ax2 = axes[1]
    n_comp = min(5, len(var_ratio))
    x_pos = np.arange(n_comp)
    ax2.bar(x_pos, var_ratio[:n_comp] * 100, color=C_MIN, alpha=0.7,
            edgecolor='black', linewidth=0.5)
    cumulative = np.cumsum(var_ratio[:n_comp]) * 100
    ax2.plot(x_pos, cumulative, 'o-', color=C_BORDER, linewidth=2, markersize=6)
    for i in range(n_comp):
        ax2.text(i, var_ratio[i]*100 + 1.5, f'{var_ratio[i]*100:.1f}%',
                 ha='center', fontsize=8, fontweight='bold')
    ax2.axhline(80, color=C_SYN, linestyle='--', linewidth=1, alpha=0.6, label='80% threshold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'PC$_{i+1}$' for i in range(n_comp)])
    ax2.set_ylabel('Variance Explained (%)')
    ax2.set_title('(b) Explained Variance Ratio', fontweight='bold', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.15, axis='y')

    fig.suptitle('Step 2: PCA Projection for Scoring', fontweight='bold', fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, 'seed_step2_pca.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Candidate seed sets
# ═══════════════════════════════════════════════════════════════════════════
def seed_step3():
    X = make_minority_hd()
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    X_2d = pca.transform(X)
    km = KMeans(n_clusters=5, random_state=SEED, n_init=10).fit(X_2d)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    n_seeds = 20

    for trial, ax in enumerate(axes.flat):
        rng = np.random.RandomState(SEED + trial * 7)
        # Stratified sampling
        selected = []
        for c in range(5):
            cluster_idx = np.where(km.labels_ == c)[0]
            n_c = max(1, int(len(cluster_idx) / len(X) * n_seeds))
            chosen = rng.choice(cluster_idx, min(n_c, len(cluster_idx)), replace=False)
            selected.extend(chosen)
        selected = np.array(selected[:n_seeds])

        # Plot all points faded
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=C_LIGHT, s=25, alpha=0.4,
                   edgecolors='none', zorder=1)
        # Highlight selected seeds
        for c in range(5):
            mask = np.isin(selected, np.where(km.labels_ == c)[0])
            if mask.any():
                ax.scatter(X_2d[selected[mask], 0], X_2d[selected[mask], 1],
                           c=CLUSTER_COLORS[c], s=60, alpha=0.9, edgecolors='black',
                           linewidth=0.5, marker='s', zorder=3)

        ax.set_title(f'Candidate {trial+1} ($m={len(selected)}$)',
                     fontweight='bold', fontsize=10)
        ax.set_xlabel('PC$_1$'); ax.set_ylabel('PC$_2$')
        ax.grid(True, alpha=0.1)

    fig.suptitle('Step 3: Candidate Seed Sets (Stratified Random Sampling)',
                 fontweight='bold', fontsize=12, y=1.01)
    fig.tight_layout()
    savefig(fig, 'seed_step3_candidates.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: NHOP computation
# ═══════════════════════════════════════════════════════════════════════════
def seed_step4():
    X = make_minority_hd()
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    X_2d = pca.transform(X)
    rng = np.random.RandomState(SEED)
    seeds = rng.choice(len(X), 20, replace=False)
    X_seeds = X_2d[seeds]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for dim, ax in enumerate(axes):
        vals_orig = X_2d[:, dim]
        vals_seed = X_seeds[:, dim]

        # Min-max normalize
        lo = min(vals_orig.min(), vals_seed.min())
        hi = max(vals_orig.max(), vals_seed.max())
        vals_orig_norm = (vals_orig - lo) / (hi - lo + 1e-8)
        vals_seed_norm = (vals_seed - lo) / (hi - lo + 1e-8)

        n_bins = 15
        edges = np.linspace(0, 1, n_bins + 1)
        h_orig, _ = np.histogram(vals_orig_norm, bins=edges, density=True)
        h_seed, _ = np.histogram(vals_seed_norm, bins=edges, density=True)
        h_orig = h_orig / (h_orig.sum() + 1e-8)
        h_seed = h_seed / (h_seed.sum() + 1e-8)
        overlap = np.minimum(h_orig, h_seed)

        bin_centers = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]

        ax.bar(bin_centers - width*0.15, h_orig, width=width*0.3, color=C_MIN, alpha=0.6,
               edgecolor='black', linewidth=0.3, label='Original')
        ax.bar(bin_centers + width*0.15, h_seed, width=width*0.3, color=C_SYN, alpha=0.6,
               edgecolor='black', linewidth=0.3, label='Seeds')

        # Shade overlap
        for b in range(n_bins):
            ov = overlap[b]
            if ov > 0:
                ax.bar(bin_centers[b], ov, width=width*0.6, color=C_BORDER, alpha=0.25,
                       edgecolor='none')

        nhop_dim = overlap.sum()
        ax.set_title(f'PC$_{dim+1}$: NHOP$_{dim+1}$ = {nhop_dim:.3f}', fontweight='bold', fontsize=10)
        ax.set_xlabel('Normalized value $[0, 1]$')
        ax.set_ylabel('Proportion')
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.15, axis='y')

    nhop_total = sum(np.minimum(
        np.histogram((X_2d[:, d] - X_2d[:, d].min()) / (X_2d[:, d].max() - X_2d[:, d].min() + 1e-8),
                     bins=15, density=False)[0],
        np.histogram((X_seeds[:, d] - X_2d[:, d].min()) / (X_2d[:, d].max() - X_2d[:, d].min() + 1e-8),
                     bins=15, density=False)[0]
    ).sum() for d in range(2)) / 2

    fig.suptitle(f'Step 4: NHOP — Normalized Histogram Overlap',
                 fontweight='bold', fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, 'seed_step4_nhop.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: JSD computation
# ═══════════════════════════════════════════════════════════════════════════
def seed_step5():
    rng = np.random.RandomState(SEED)

    # Create two distributions for illustration
    n_bins = 20
    x = np.linspace(0, 1, 200)

    # P: original-like
    P_raw = 0.5 * np.exp(-((x-0.3)**2) / 0.02) + 0.3 * np.exp(-((x-0.7)**2) / 0.03)
    P = P_raw / P_raw.sum()
    # Q: seed-like (slightly shifted)
    Q_raw = 0.6 * np.exp(-((x-0.35)**2) / 0.025) + 0.2 * np.exp(-((x-0.65)**2) / 0.04)
    Q = Q_raw / Q_raw.sum()
    M = (P + Q) / 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) P and Q
    ax1 = axes[0]
    ax1.fill_between(x, P, alpha=0.3, color=C_MIN, label='$P$ (original)')
    ax1.plot(x, P, color=C_MIN, linewidth=1.5)
    ax1.fill_between(x, Q, alpha=0.3, color=C_SYN, label='$Q$ (seeds)')
    ax1.plot(x, Q, color=C_SYN, linewidth=1.5)
    ax1.set_title('(a) Distributions $P$ and $Q$', fontweight='bold', fontsize=10)
    ax1.set_xlabel('Normalized value'); ax1.set_ylabel('Probability')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.15)

    # (b) Mixture M
    ax2 = axes[1]
    ax2.fill_between(x, M, alpha=0.3, color=C_BORDER, label='$M = (P+Q)/2$')
    ax2.plot(x, M, color=C_BORDER, linewidth=2)
    ax2.plot(x, P, color=C_MIN, linewidth=1, alpha=0.5, linestyle='--')
    ax2.plot(x, Q, color=C_SYN, linewidth=1, alpha=0.5, linestyle='--')
    ax2.set_title('(b) Mixture $M = (P+Q)/2$', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Normalized value'); ax2.set_ylabel('Probability')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.15)

    # (c) KL divergences and JSD
    ax3 = axes[2]
    # Compute KL and JSD
    P_safe = np.clip(P, 1e-10, None)
    Q_safe = np.clip(Q, 1e-10, None)
    M_safe = np.clip(M, 1e-10, None)
    kl_pm = np.sum(P_safe * np.log(P_safe / M_safe))
    kl_qm = np.sum(Q_safe * np.log(Q_safe / M_safe))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm

    bars = ax3.bar(['$D_{KL}(P||M)$', '$D_{KL}(Q||M)$', '$JSD(P,Q)$'],
                   [kl_pm, kl_qm, jsd],
                   color=[C_MIN, C_SYN, C_BORDER], alpha=0.7, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, [kl_pm, kl_qm, jsd]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax3.set_title('(c) JSD Components', fontweight='bold', fontsize=10)
    ax3.set_ylabel('Divergence')
    ax3.grid(True, alpha=0.15, axis='y')

    fig.suptitle('Step 5: Jensen–Shannon Divergence', fontweight='bold', fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, 'seed_step5_jsd.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: AGTP — Geometric similarity
# ═══════════════════════════════════════════════════════════════════════════
def seed_step6():
    X = make_minority_hd()
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    X_2d = pca.transform(X)
    rng = np.random.RandomState(SEED)
    seeds = rng.choice(len(X), 20, replace=False)
    X_seeds = X_2d[seeds]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Plot points
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=C_MIN, s=30, alpha=0.3, edgecolors='none',
               marker='s', zorder=2, label='Original')
    ax.scatter(X_seeds[:, 0], X_seeds[:, 1], c=C_SYN, s=50, alpha=0.8,
               edgecolors='black', linewidth=0.4, marker='s', zorder=3, label='Seeds')

    # Means
    mu_orig = X_2d.mean(axis=0)
    mu_seeds = X_seeds.mean(axis=0)
    ax.scatter(*mu_orig, c=C_MIN, marker='*', s=200, edgecolors='black',
               linewidth=1, zorder=5)
    ax.scatter(*mu_seeds, c=C_SYN, marker='*', s=200, edgecolors='black',
               linewidth=1, zorder=5)

    # Covariance ellipses
    for data, color, label in [(X_2d, C_MIN, 'Original'), (X_seeds, C_SYN, 'Seeds')]:
        cov = np.cov(data.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        for ns in [1, 2]:
            width = 2 * ns * np.sqrt(eigenvalues[1])
            height = 2 * ns * np.sqrt(eigenvalues[0])
            ellipse = Ellipse(xy=data.mean(axis=0), width=width, height=height,
                              angle=angle, fill=False, edgecolor=color,
                              linewidth=1.0 + (2-ns)*0.5, linestyle='--' if ns==2 else '-',
                              alpha=0.5 + ns*0.15, zorder=4)
            ax.add_patch(ellipse)

    # Mean difference arrow
    ax.annotate('', xy=mu_seeds, xytext=mu_orig,
                arrowprops=dict(arrowstyle='<->', color=C_BORDER, lw=2))
    mid = (mu_orig + mu_seeds) / 2
    mu_diff = np.linalg.norm(mu_orig - mu_seeds)
    ax.text(mid[0] + 0.2, mid[1] + 0.2, f'$\\|\\Delta\\mu\\| = {mu_diff:.3f}$',
            fontsize=8, color=C_BORDER, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8))

    # Compute G_sim
    cov_orig = np.cov(X_2d.T)
    cov_seeds = np.cov(X_seeds.T)
    mu_sim = max(0, 1 - np.linalg.norm(mu_orig - mu_seeds) / (np.linalg.norm(mu_orig) + 1e-8))
    cov_sim = max(0, 1 - np.linalg.norm(cov_orig - cov_seeds, 'fro') / (np.linalg.norm(cov_orig, 'fro') + 1e-8))
    g_sim = 0.5 * (mu_sim + cov_sim)

    ax.text(0.98, 0.02, f'$G_{{sim}} = \\frac{{1}}{{2}}({mu_sim:.3f} + {cov_sim:.3f}) = {g_sim:.3f}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    ax.set_title('Step 6: AGTP — Geometric Similarity ($G_{sim}$)', fontweight='bold', pad=10)
    ax.set_xlabel('PC$_1$'); ax.set_ylabel('PC$_2$')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    savefig(fig, 'seed_step6_agtp_geom.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: AGTP — Topological similarity
# ═══════════════════════════════════════════════════════════════════════════
def seed_step7():
    X = make_minority_hd()
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    X_2d = pca.transform(X)
    rng = np.random.RandomState(SEED)
    seeds = rng.choice(len(X), 20, replace=False)
    X_seeds = X_2d[seeds]
    k = 5

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # (a) kNN graph for original
    ax1 = axes[0]
    nn_orig = NearestNeighbors(n_neighbors=k+1).fit(X_2d)
    dists_orig, idx_orig = nn_orig.kneighbors(X_2d)
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=C_MIN, s=25, alpha=0.6, edgecolors='none', marker='s')
    # Draw some kNN edges
    for i in range(0, len(X_2d), 4):
        for j in idx_orig[i, 1:k+1]:
            ax1.plot([X_2d[i, 0], X_2d[j, 0]], [X_2d[i, 1], X_2d[j, 1]],
                     c=C_MIN, linewidth=0.3, alpha=0.2)
    ax1.set_title('(a) $k$-NN Graph (Original)', fontweight='bold', fontsize=10)
    ax1.set_xlabel('PC$_1$'); ax1.set_ylabel('PC$_2$')
    ax1.grid(True, alpha=0.1)

    # (b) kNN graph for seeds
    ax2 = axes[1]
    nn_seeds = NearestNeighbors(n_neighbors=min(k+1, len(X_seeds))).fit(X_seeds)
    dists_seeds, idx_seeds = nn_seeds.kneighbors(X_seeds)
    ax2.scatter(X_seeds[:, 0], X_seeds[:, 1], c=C_SYN, s=40, alpha=0.8,
                edgecolors='black', linewidth=0.3, marker='s')
    for i in range(len(X_seeds)):
        for j in idx_seeds[i, 1:]:
            ax2.plot([X_seeds[i, 0], X_seeds[j, 0]], [X_seeds[i, 1], X_seeds[j, 1]],
                     c=C_SYN, linewidth=0.5, alpha=0.3)
    ax2.set_title('(b) $k$-NN Graph (Seeds)', fontweight='bold', fontsize=10)
    ax2.set_xlabel('PC$_1$'); ax2.set_ylabel('PC$_2$')
    ax2.grid(True, alpha=0.1)

    # (c) kNN distance histograms comparison
    ax3 = axes[2]
    knn_dists_orig = dists_orig[:, 1:].flatten()
    knn_dists_seeds = dists_seeds[:, 1:].flatten()
    n_bins = 20
    lo = min(knn_dists_orig.min(), knn_dists_seeds.min())
    hi = max(knn_dists_orig.max(), knn_dists_seeds.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    h1, _ = np.histogram(knn_dists_orig, bins=edges, density=True)
    h2, _ = np.histogram(knn_dists_seeds, bins=edges, density=True)
    h1 = h1 / (h1.sum() + 1e-8)
    h2 = h2 / (h2.sum() + 1e-8)
    overlap = np.minimum(h1, h2).sum()

    bin_centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    ax3.bar(bin_centers - width*0.15, h1, width=width*0.3, color=C_MIN, alpha=0.6,
            edgecolor='black', linewidth=0.3, label='Original')
    ax3.bar(bin_centers + width*0.15, h2, width=width*0.3, color=C_SYN, alpha=0.6,
            edgecolor='black', linewidth=0.3, label='Seeds')
    ax3.set_title(f'(c) $k$-NN Distance Histograms\n$T_{{sim}} = {overlap:.3f}$',
                  fontweight='bold', fontsize=10)
    ax3.set_xlabel('$k$-NN distance'); ax3.set_ylabel('Proportion')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.15, axis='y')

    fig.suptitle('Step 7: AGTP — Topological Similarity ($T_{sim}$)',
                 fontweight='bold', fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, 'seed_step7_agtp_topo.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 8: Smoothness Z
# ═══════════════════════════════════════════════════════════════════════════
def seed_step8():
    rng = np.random.RandomState(SEED)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # (a) Regular spacing (low Z)
    ax1 = axes[0]
    n_seeds = 12
    angles = np.linspace(0, 2*np.pi, n_seeds, endpoint=False)
    regular_pts = np.column_stack([np.cos(angles), np.sin(angles)]) * 1.5
    regular_pts += rng.normal(0, 0.05, regular_pts.shape)  # tiny jitter

    ax1.scatter(regular_pts[:, 0], regular_pts[:, 1], c=C_SYN, s=70,
                edgecolors='black', linewidth=0.6, marker='s', zorder=3)
    # NN distances
    nn = NearestNeighbors(n_neighbors=2).fit(regular_pts)
    dists, idx = nn.kneighbors(regular_pts)
    nn_dists = dists[:, 1]
    z_regular = nn_dists.std() / (nn_dists.mean() + 1e-8)

    for i in range(len(regular_pts)):
        j = idx[i, 1]
        ax1.plot([regular_pts[i, 0], regular_pts[j, 0]],
                 [regular_pts[i, 1], regular_pts[j, 1]],
                 c=C_BORDER, linewidth=1.0, alpha=0.6, zorder=2)
        mid = (regular_pts[i] + regular_pts[j]) / 2
        ax1.text(mid[0]+0.08, mid[1]+0.08, f'{nn_dists[i]:.2f}', fontsize=6,
                 color=C_DARK, alpha=0.7)

    ax1.set_title(f'(a) Regular Spacing: $Z = {z_regular:.3f}$', fontweight='bold', fontsize=10)
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.1)
    ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$x_2$')

    # (b) Irregular spacing (high Z)
    ax2 = axes[1]
    # Cluster some points close together, others far apart
    irregular_pts = np.vstack([
        rng.normal(0, 0.15, (6, 2)),  # tight cluster
        rng.normal(0, 0.15, (3, 2)) + np.array([3, 3]),  # far away
        rng.normal(0, 0.15, (3, 2)) + np.array([-2, 2]),  # medium distance
    ])
    ax2.scatter(irregular_pts[:, 0], irregular_pts[:, 1], c=C_SYN, s=70,
                edgecolors='black', linewidth=0.6, marker='s', zorder=3)

    nn2 = NearestNeighbors(n_neighbors=2).fit(irregular_pts)
    dists2, idx2 = nn2.kneighbors(irregular_pts)
    nn_dists2 = dists2[:, 1]
    z_irregular = nn_dists2.std() / (nn_dists2.mean() + 1e-8)

    for i in range(len(irregular_pts)):
        j = idx2[i, 1]
        ax2.plot([irregular_pts[i, 0], irregular_pts[j, 0]],
                 [irregular_pts[i, 1], irregular_pts[j, 1]],
                 c=C_BORDER, linewidth=1.0, alpha=0.6, zorder=2)
        mid = (irregular_pts[i] + irregular_pts[j]) / 2
        ax2.text(mid[0]+0.08, mid[1]+0.08, f'{nn_dists2[i]:.2f}', fontsize=6,
                 color=C_DARK, alpha=0.7)

    ax2.set_title(f'(b) Irregular Spacing: $Z = {z_irregular:.3f}$', fontweight='bold', fontsize=10)
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.1)
    ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$x_2$')

    fig.suptitle('Step 8: Smoothness Regulariser $Z$ (Coefficient of Variation of NN Distances)',
                 fontweight='bold', fontsize=11, y=1.02)
    fig.tight_layout()
    savefig(fig, 'seed_step8_smoothness.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Step 9: Composite scoring
# ═══════════════════════════════════════════════════════════════════════════
def seed_step9():
    # Simulated scores for best vs worst candidate
    candidates = {
        'Best': {'nhop': 0.92, 'agtp': 0.85, 'jsd': 0.03, 'z': 0.25},
        'Median': {'nhop': 0.78, 'agtp': 0.70, 'jsd': 0.12, 'z': 0.45},
        'Worst': {'nhop': 0.55, 'agtp': 0.48, 'jsd': 0.28, 'z': 0.82},
    }
    w_j, w_z = 0.3, 0.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Component breakdown
    ax1 = axes[0]
    x_pos = np.arange(4)
    width = 0.25
    colors_bar = {'Best': C_SYN, 'Median': C_BORDER, 'Worst': C_MIN}

    for i, (name, scores) in enumerate(candidates.items()):
        vals = [scores['nhop'], scores['agtp'], -w_j * scores['jsd'], -w_z * scores['z']]
        bars = ax1.bar(x_pos + i*width, vals, width, color=colors_bar[name], alpha=0.7,
                       edgecolor='black', linewidth=0.5, label=name)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 * np.sign(val),
                     f'{val:.3f}', ha='center', fontsize=6.5, fontweight='bold')

    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(['NHOP', 'AGTP', '$-w_j$·JSD', '$-w_z$·$Z$'])
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('Score Component')
    ax1.set_title('(a) Score Components', fontweight='bold', fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.15, axis='y')

    # (b) Final composite scores
    ax2 = axes[1]
    names = list(candidates.keys())
    final_scores = []
    for name, scores in candidates.items():
        s = (scores['nhop'] + scores['agtp']) - w_j * scores['jsd'] - w_z * scores['z']
        final_scores.append(s)

    bars = ax2.barh(names, final_scores, color=[colors_bar[n] for n in names],
                    alpha=0.7, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, final_scores):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    # Arrow pointing to best
    ax2.annotate('← Selected', xy=(final_scores[0], 0), xytext=(final_scores[0]+0.15, 0.3),
                 fontsize=10, color=C_SYN, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C_SYN, lw=1.5))

    ax2.set_xlabel('Composite Score = (NHOP + AGTP) − $w_j$·JSD − $w_z$·$Z$')
    ax2.set_title('(b) Final Composite Scores', fontweight='bold', fontsize=10)
    ax2.grid(True, alpha=0.15, axis='x')

    fig.suptitle('Step 9: Composite Scoring & Selection', fontweight='bold', fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, 'seed_step9_scoring.pdf')


# ═══════════════════════════════════════════════════════════════════════════
# Overview: 2x2 panel
# ═══════════════════════════════════════════════════════════════════════════
def seed_overview():
    X = make_minority_hd()
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    X_2d = pca.transform(X)
    km = KMeans(n_clusters=5, random_state=SEED, n_init=10).fit(X_2d)
    rng = np.random.RandomState(SEED)
    seeds = rng.choice(len(X), 20, replace=False)
    X_seeds = X_2d[seeds]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # (a) Clustering + candidates
    ax = axes[0, 0]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=C_LIGHT, s=20, alpha=0.4, edgecolors='none')
    for c in range(5):
        mask = np.isin(seeds, np.where(km.labels_ == c)[0])
        if mask.any():
            ax.scatter(X_seeds[mask, 0], X_seeds[mask, 1], c=CLUSTER_COLORS[c], s=55,
                       edgecolors='black', linewidth=0.4, marker='s', zorder=3)
    ax.set_title('(a) Stratified Seed Selection', fontweight='bold', fontsize=10)
    ax.set_xlabel('PC$_1$'); ax.set_ylabel('PC$_2$')
    ax.grid(True, alpha=0.1)

    # (b) NHOP + JSD
    ax = axes[0, 1]
    dim = 0
    vals_orig = X_2d[:, dim]
    vals_seed = X_seeds[:, dim]
    lo = min(vals_orig.min(), vals_seed.min())
    hi = max(vals_orig.max(), vals_seed.max())
    n_bins = 12
    edges = np.linspace(0, 1, n_bins + 1)
    h1, _ = np.histogram((vals_orig - lo) / (hi - lo + 1e-8), bins=edges, density=True)
    h2, _ = np.histogram((vals_seed - lo) / (hi - lo + 1e-8), bins=edges, density=True)
    h1 = h1 / (h1.sum() + 1e-8)
    h2 = h2 / (h2.sum() + 1e-8)
    bc = (edges[:-1] + edges[1:]) / 2
    w = edges[1] - edges[0]
    ax.bar(bc - w*0.15, h1, width=w*0.3, color=C_MIN, alpha=0.6, label='Original')
    ax.bar(bc + w*0.15, h2, width=w*0.3, color=C_SYN, alpha=0.6, label='Seeds')
    ax.set_title('(b) NHOP + JSD Histograms', fontweight='bold', fontsize=10)
    ax.set_xlabel('Normalized value'); ax.set_ylabel('Proportion')
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.15, axis='y')

    # (c) AGTP ellipses
    ax = axes[1, 0]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=C_MIN, s=20, alpha=0.3, edgecolors='none', marker='s')
    ax.scatter(X_seeds[:, 0], X_seeds[:, 1], c=C_SYN, s=40, alpha=0.7,
               edgecolors='black', linewidth=0.3, marker='s')
    for data, color in [(X_2d, C_MIN), (X_seeds, C_SYN)]:
        cov = np.cov(data.T)
        evals, evecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        ell = Ellipse(xy=data.mean(axis=0), width=4*np.sqrt(evals[1]),
                      height=4*np.sqrt(evals[0]), angle=angle, fill=False,
                      edgecolor=color, linewidth=1.5, linestyle='--', zorder=4)
        ax.add_patch(ell)
    ax.set_title('(c) AGTP: Mean + Covariance', fontweight='bold', fontsize=10)
    ax.set_xlabel('PC$_1$'); ax.set_ylabel('PC$_2$')
    ax.grid(True, alpha=0.1)

    # (d) Best seed set
    ax = axes[1, 1]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=C_LIGHT, s=20, alpha=0.3, edgecolors='none')
    ax.scatter(X_seeds[:, 0], X_seeds[:, 1], c=C_SYN, s=65, alpha=0.9,
               edgecolors='black', linewidth=0.6, marker='s', zorder=3)
    ax.set_title('(d) Best Seed Set Selected', fontweight='bold', fontsize=10)
    ax.set_xlabel('PC$_1$'); ax.set_ylabel('PC$_2$')
    ax.grid(True, alpha=0.1)
    ax.text(0.5, 0.02, 'Score = (NHOP + AGTP) − $w_j$·JSD − $w_z$·$Z$',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    fig.suptitle('Seed Selection Pipeline — Overview', fontweight='bold', fontsize=13, y=1.01)
    fig.tight_layout()
    savefig(fig, 'seed_overview.pdf')


if __name__ == '__main__':
    print('Generating Seed Selection step-by-step figures...')
    seed_step1()
    seed_step2()
    seed_step3()
    seed_step4()
    seed_step5()
    seed_step6()
    seed_step7()
    seed_step8()
    seed_step9()
    seed_overview()
    print('Done! 10 Seed Selection figures generated.')
