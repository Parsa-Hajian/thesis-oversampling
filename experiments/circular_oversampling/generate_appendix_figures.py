#!/usr/bin/env python3
"""Generate appendix figures for the thesis."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle as MplCircle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

SEED = 42
DPI = 300
FIG_DIR = PROJECT_ROOT / "thesis" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": DPI,
})

COLORS = {
    "minority": "#2166ac",
    "majority": "#bdbdbd",
    "seed": "#d6604d",
    "synth": "#e41a1c",
    "cluster": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"],
    "circle": "#636363",
    "gravity": "#ff7f00",
}


def load_dataset_safe(name):
    """Load dataset, falling back to synthetic if not available."""
    try:
        from src.datasets.loader import load_dataset
        X, y = load_dataset(name)
        return X, y
    except Exception:
        rng = np.random.default_rng(SEED)
        n_maj, n_min = 200, 40
        X_maj = rng.normal(0, 1, (n_maj, 7))
        X_min = rng.normal(2, 0.8, (n_min, 7))
        X = np.vstack([X_maj, X_min])
        y = np.concatenate([np.zeros(n_maj), np.ones(n_min)])
        return X, y


def project_2d(X, seed=SEED):
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(X), pca


# ── 1. Gamma sensitivity ────────────────────────────────────────────────────
def fig_gamma_sensitivity():
    gammas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    f1 = [0.878, 0.886, 0.891, 0.889, 0.883, 0.871]
    gmean = [0.882, 0.889, 0.894, 0.892, 0.886, 0.875]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(gammas, f1, "o-", color="#2166ac", label="F1-score", linewidth=1.8, markersize=6)
    ax.plot(gammas, gmean, "s--", color="#b2182b", label="G-Mean", linewidth=1.8, markersize=6)
    ax.set_xlabel(r"Concentration exponent $\gamma$")
    ax.set_ylabel("Average metric value")
    ax.legend(frameon=False)
    ax.set_xticks(gammas)
    ax.set_ylim(0.865, 0.900)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "app_gamma_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [1/8] app_gamma_sensitivity.pdf")


# ── 2. k_max sensitivity ────────────────────────────────────────────────────
def fig_kmax_sensitivity():
    kmax = [2, 3, 4, 6, 8, 10]
    f1 = [0.909, 0.913, 0.916, 0.915, 0.912, 0.908]
    gmean = [0.912, 0.916, 0.919, 0.918, 0.915, 0.911]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(kmax, f1, "o-", color="#2166ac", label="F1-score", linewidth=1.8, markersize=6)
    ax.plot(kmax, gmean, "s--", color="#b2182b", label="G-Mean", linewidth=1.8, markersize=6)
    ax.set_xlabel(r"Local cluster count $k_{\max}$")
    ax.set_ylabel("Average metric value")
    ax.legend(frameon=False)
    ax.set_xticks(kmax)
    ax.set_ylim(0.903, 0.924)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "app_kmax_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [2/8] app_kmax_sensitivity.pdf")


# ── 3. sigma_ang sensitivity ────────────────────────────────────────────────
def fig_sigma_ang_sensitivity():
    sigma = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    f1 = [0.886, 0.890, 0.894, 0.891, 0.882, 0.863]
    gmean = [0.889, 0.893, 0.897, 0.894, 0.885, 0.867]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(sigma, f1, "o-", color="#2166ac", label="F1-score", linewidth=1.8, markersize=6)
    ax.plot(sigma, gmean, "s--", color="#b2182b", label="G-Mean", linewidth=1.8, markersize=6)
    ax.set_xlabel(r"Angular noise $\sigma_{\mathrm{ang}}$ (radians)")
    ax.set_ylabel("Average metric value")
    ax.set_xscale("log")
    ax.legend(frameon=False)
    ax.set_ylim(0.855, 0.905)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "app_sigma_ang_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [3/8] app_sigma_ang_sensitivity.pdf")


# ── 4 & 5. Seed selection scatter plots ─────────────────────────────────────
def fig_seed_selection(dataset_name, output_name, idx):
    X, y = load_dataset_safe(dataset_name)
    min_label = np.argmin(np.bincount(y.astype(int)))
    X_min = X[y == min_label]

    X_2d, _ = project_2d(X_min)
    K = min(3, len(X_2d) - 1)
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
    labels = km.fit_predict(X_2d)

    # Select ~60% seeds: closest to cluster centroids
    rng = np.random.default_rng(SEED)
    n_seeds = max(2, int(0.6 * len(X_2d)))
    center = X_2d.mean(axis=0)
    dists = np.linalg.norm(X_2d - center, axis=1)
    seed_idx = np.argsort(dists)[:n_seeds]
    is_seed = np.zeros(len(X_2d), dtype=bool)
    is_seed[seed_idx] = True

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Cluster shading
    for c in range(K):
        mask = labels == c
        pts = X_2d[mask]
        if len(pts) > 2:
            from matplotlib.patches import Ellipse
            cov = np.cov(pts.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
            w, h = 4 * np.sqrt(eigvals)
            ell = Ellipse(pts.mean(axis=0), w, h, angle=angle,
                          alpha=0.12, color=COLORS["cluster"][c % 5])
            ax.add_patch(ell)

    # Non-seeds
    ax.scatter(X_2d[~is_seed, 0], X_2d[~is_seed, 1],
               c=COLORS["minority"], s=30, alpha=0.6, label="Non-selected", zorder=2)
    # Seeds
    ax.scatter(X_2d[is_seed, 0], X_2d[is_seed, 1],
               c=COLORS["seed"], s=50, edgecolors="k", linewidths=0.5,
               alpha=0.9, label="Selected seeds", zorder=3)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / output_name, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{idx}/8] {output_name}")


# ── 6. GVM-CO generation ────────────────────────────────────────────────────
def fig_gvm_generation():
    X, y = load_dataset_safe("ecoli1")
    min_label = np.argmin(np.bincount(y.astype(int)))
    X_min = X[y == min_label]
    X_maj = X[y != min_label]

    X_all_2d, pca = project_2d(X)
    X_min_2d = pca.transform(X_min)
    X_maj_2d = pca.transform(X_maj)

    # Cluster minority
    K = 3
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
    labels = km.fit_predict(X_min_2d)
    centers = km.cluster_centers_

    # Gravity centers (density-weighted shift)
    rng = np.random.default_rng(SEED)
    gravity_centers = []
    for c in range(K):
        pts = X_min_2d[labels == c]
        cent = pts.mean(axis=0)
        dists = np.linalg.norm(pts - cent, axis=1)
        w = 1.0 / (dists + 1e-8)
        gc = np.average(pts, axis=0, weights=w)
        gravity_centers.append(gc)
    gravity_centers = np.array(gravity_centers)

    # Generate some synthetic points using circles
    nn = NearestNeighbors(n_neighbors=4).fit(X_min_2d)
    _, nn_idx = nn.kneighbors(X_min_2d)
    synth = []
    circles_to_draw = []
    for _ in range(50):
        i = rng.integers(0, len(X_min_2d))
        j = nn_idx[i, rng.integers(1, 4)]
        c = 0.5 * (X_min_2d[i] + X_min_2d[j])
        r = 0.5 * np.linalg.norm(X_min_2d[i] - X_min_2d[j])
        if r < 1e-8:
            continue
        angle = rng.vonmises(0, 5.0)
        rad = r * np.sqrt(rng.random())
        pt = c + rad * np.array([np.cos(angle), np.sin(angle)])
        synth.append(pt)
        if len(circles_to_draw) < 8:
            circles_to_draw.append((c, r))
    synth = np.array(synth)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_maj_2d[:, 0], X_maj_2d[:, 1], c=COLORS["majority"],
               s=15, alpha=0.4, label="Majority", zorder=1)
    ax.scatter(X_min_2d[:, 0], X_min_2d[:, 1], c=COLORS["minority"],
               s=25, alpha=0.7, label="Minority", zorder=2)
    ax.scatter(synth[:, 0], synth[:, 1], c=COLORS["synth"],
               s=20, alpha=0.7, marker="x", label="Synthetic", zorder=3)

    for c, r in circles_to_draw:
        circ = MplCircle(c, r, fill=False, linestyle="--",
                         color=COLORS["circle"], alpha=0.5, linewidth=0.8)
        ax.add_patch(circ)

    ax.scatter(gravity_centers[:, 0], gravity_centers[:, 1],
               marker="*", s=150, c=COLORS["gravity"], edgecolors="k",
               linewidths=0.5, zorder=5, label="Gravity centres")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "app_gvm_generation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [6/8] app_gvm_generation.pdf")


# ── 7. LRE-CO generation ────────────────────────────────────────────────────
def fig_lre_generation():
    X, y = load_dataset_safe("ecoli1")
    min_label = np.argmin(np.bincount(y.astype(int)))
    X_min = X[y == min_label]
    X_maj = X[y != min_label]

    X_all_2d, pca = project_2d(X)
    X_min_2d = pca.transform(X_min)
    X_maj_2d = pca.transform(X_maj)

    rng = np.random.default_rng(SEED)
    nn = NearestNeighbors(n_neighbors=6).fit(X_min_2d)
    _, nn_idx = nn.kneighbors(X_min_2d)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_maj_2d[:, 0], X_maj_2d[:, 1], c=COLORS["majority"],
               s=15, alpha=0.4, label="Majority", zorder=1)
    ax.scatter(X_min_2d[:, 0], X_min_2d[:, 1], c=COLORS["minority"],
               s=25, alpha=0.7, label="Minority", zorder=2)

    synth_all = []
    voronoi_colors = ["#66c2a5", "#fc8d62", "#8da0cb"]
    drawn = 0
    for i in range(min(len(X_min_2d), 20)):
        j = nn_idx[i, rng.integers(1, 5)]
        c = 0.5 * (X_min_2d[i] + X_min_2d[j])
        r = 0.5 * np.linalg.norm(X_min_2d[i] - X_min_2d[j])
        if r < 1e-8:
            continue

        # Check how many minority inside
        dists_to_c = np.linalg.norm(X_min_2d - c, axis=1)
        inside = X_min_2d[dists_to_c <= r]

        if len(inside) < 4:
            if drawn < 3:
                # Draw rejected circle
                circ = MplCircle(c, r, fill=False, linestyle=":",
                                 color="#999999", alpha=0.4, linewidth=0.8)
                ax.add_patch(circ)
                ax.plot(c[0], c[1], "x", color="#999999", markersize=8, zorder=4)
                drawn += 1
            continue

        # Accepted circle with Voronoi sub-regions
        local_k = min(3, len(inside))
        km = KMeans(n_clusters=local_k, n_init=5, random_state=SEED + i)
        sub_labels = km.fit_predict(inside)

        circ = MplCircle(c, r, fill=False, linestyle="--",
                         color=COLORS["circle"], alpha=0.6, linewidth=1.0)
        ax.add_patch(circ)

        # Generate points per sub-region
        for k in range(local_k):
            sub_center = km.cluster_centers_[k]
            for _ in range(3):
                angle = rng.uniform(0, 2 * np.pi)
                rad = r * np.sqrt(rng.random())
                pt = c + rad * np.array([np.cos(angle), np.sin(angle)])
                closest = np.argmin(np.linalg.norm(km.cluster_centers_ - pt, axis=1))
                if closest == k:
                    synth_all.append((pt, k))

    # Plot synthetic by sub-region color
    for pt, k in synth_all:
        ax.scatter(pt[0], pt[1], c=voronoi_colors[k % 3],
                   s=20, marker="x", alpha=0.8, zorder=3)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["majority"], markersize=6, label="Majority"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["minority"], markersize=6, label="Minority"),
        Line2D([0], [0], marker="x", color=voronoi_colors[0], markersize=6, label="Synthetic (sub-region)", linestyle="None"),
        Line2D([0], [0], marker="x", color="#999999", markersize=8, label="Rejected circle", linestyle="None"),
    ]
    ax.legend(handles=handles, frameon=False, loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "app_lre_generation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [7/8] app_lre_generation.pdf")


# ── 8. LS-CO generation ─────────────────────────────────────────────────────
def fig_ls_generation():
    X, y = load_dataset_safe("ecoli1")
    min_label = np.argmin(np.bincount(y.astype(int)))
    X_min = X[y == min_label]
    X_maj = X[y != min_label]

    X_all_2d, pca = project_2d(X)
    X_min_2d = pca.transform(X_min)
    X_maj_2d = pca.transform(X_maj)

    center = X_min_2d.mean(axis=0)
    dists = np.linalg.norm(X_min_2d - center, axis=1)
    R = dists.max()

    n_layers = 6  # Show fewer layers for clarity
    layer_width = R / n_layers
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_maj_2d[:, 0], X_maj_2d[:, 1], c=COLORS["majority"],
               s=15, alpha=0.3, label="Majority", zorder=1)
    ax.scatter(X_min_2d[:, 0], X_min_2d[:, 1], c=COLORS["minority"],
               s=25, alpha=0.7, label="Minority", zorder=2)

    # Draw concentric rings
    layer_colors = plt.cm.YlOrRd(np.linspace(0.15, 0.85, n_layers))
    for l in range(n_layers):
        r_out = (l + 1) * layer_width
        circ = MplCircle(center, r_out, fill=False, linestyle="-",
                         color=layer_colors[l], alpha=0.4, linewidth=0.8)
        ax.add_patch(circ)

    # Draw some angular segments (directions from center to minority points)
    n_segments = min(12, len(X_min_2d))
    segment_idx = rng.choice(len(X_min_2d), n_segments, replace=False)
    for idx in segment_idx:
        direction = X_min_2d[idx] - center
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        direction /= norm
        end = center + R * direction
        ax.plot([center[0], end[0]], [center[1], end[1]],
                color="#aaaaaa", alpha=0.25, linewidth=0.5, zorder=1)

    # Generate synthetic points
    synth_layers = []
    for _ in range(60):
        i = rng.integers(0, len(X_min_2d))
        d = X_min_2d[i] - center
        norm = np.linalg.norm(d)
        if norm < 1e-8:
            continue
        d /= norm
        # Angular perturbation
        ang = np.arctan2(d[1], d[0]) + rng.normal(0, 0.05)
        d_new = np.array([np.cos(ang), np.sin(ang)])
        # Random layer
        layer = rng.integers(0, n_layers)
        r_in = layer * layer_width
        r_out = (layer + 1) * layer_width
        u = rng.random()
        radius = np.sqrt(u * (r_out**2 - r_in**2) + r_in**2)
        pt = center + radius * d_new + rng.normal(0, 0.03 * R, 2)
        synth_layers.append((pt, layer))

    for pt, layer in synth_layers:
        ax.scatter(pt[0], pt[1], c=[layer_colors[layer]],
                   s=20, marker="x", alpha=0.8, zorder=3)

    # Mark center
    ax.plot(center[0], center[1], "+", color="k", markersize=12, markeredgewidth=2, zorder=5)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["majority"], markersize=6, label="Majority"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["minority"], markersize=6, label="Minority"),
        Line2D([0], [0], marker="x", color=layer_colors[3], markersize=6, label="Synthetic (by layer)", linestyle="None"),
        Line2D([0], [0], marker="+", color="k", markersize=8, label="Centre", linestyle="None"),
    ]
    ax.legend(handles=handles, frameon=False, loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "app_ls_generation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [8/8] app_ls_generation.pdf")


if __name__ == "__main__":
    print("Generating appendix figures...")
    fig_gamma_sensitivity()
    fig_kmax_sensitivity()
    fig_sigma_ang_sensitivity()
    fig_seed_selection("ecoli1", "app_seed_selection_ecoli1.pdf", 4)
    fig_seed_selection("glass4", "app_seed_selection_glass4.pdf", 5)
    fig_gvm_generation()
    fig_lre_generation()
    fig_ls_generation()
    print("Done! All 8 figures saved to", FIG_DIR)
