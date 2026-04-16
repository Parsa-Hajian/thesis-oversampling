"""
Dataset distribution visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_dataset_2d(X, y, title="", figsize=(6, 5)):
    """
    2D scatter plot of a dataset (uses PCA if d > 2).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        xlabel, ylabel = "PC1", "PC2"
    else:
        X2 = X
        xlabel, ylabel = "Feature 1", "Feature 2"

    fig, ax = plt.subplots(figsize=figsize)
    classes = np.unique(y)
    colors = ["#1f77b4", "#ff7f0e"]

    for i, c in enumerate(classes):
        mask = y == c
        n_c = mask.sum()
        ax.scatter(X2[mask, 0], X2[mask, 1], c=colors[i % len(colors)],
                  s=15, alpha=0.6, label=f"Class {c} (n={n_c})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_dataset_summary_grid(datasets, ncols=4, figsize=(16, 12)):
    """
    Grid of 2D scatter plots for multiple datasets.

    Parameters
    ----------
    datasets : dict of {name: (X, y)}
    """
    n = len(datasets)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, (name, (X, y)) in enumerate(datasets.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        if X_arr.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X2 = pca.fit_transform(X_arr)
        else:
            X2 = X_arr

        classes = np.unique(y_arr)
        colors = ["#1f77b4", "#ff7f0e"]

        for i, c in enumerate(classes):
            mask = y_arr == c
            ax.scatter(X2[mask, 0], X2[mask, 1], c=colors[i % len(colors)],
                      s=8, alpha=0.5)

        n_min = min(np.bincount(y_arr.astype(int)))
        n_maj = max(np.bincount(y_arr.astype(int)))
        ir = n_maj / max(n_min, 1)
        ax.set_title(f"{name}\nIR={ir:.1f}, n={len(y_arr)}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.suptitle("Dataset Overview", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig
