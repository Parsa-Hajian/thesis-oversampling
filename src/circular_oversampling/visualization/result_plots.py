"""
Result visualization: heatmaps, bar charts, critical difference diagrams, box plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_result_heatmap(results_df, metric="g_mean", figsize=(14, 8), cmap="YlGnBu"):
    """
    Heatmap of metric values: methods (rows) x datasets (columns).

    Parameters
    ----------
    results_df : DataFrame
        Must have columns: 'method', 'dataset', and the metric column.
    metric : str
        Which metric to plot.
    """
    pivot = results_df.pivot_table(values=metric, index="method", columns="dataset")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax,
                linewidths=0.5, cbar_kws={"label": metric})
    ax.set_title(f"{metric} — Methods x Datasets")
    plt.tight_layout()
    return fig


def plot_average_rank_bar(results_df, metric="g_mean", figsize=(10, 6)):
    """
    Bar chart of average rank per method across datasets.

    Parameters
    ----------
    results_df : DataFrame
        Must have columns: 'method', 'dataset', and the metric column.
    """
    pivot = results_df.pivot_table(values=metric, index="dataset", columns="method")

    # Rank per dataset (higher metric = rank 1)
    ranks = pivot.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("viridis", len(avg_ranks))
    bars = ax.barh(avg_ranks.index, avg_ranks.values, color=colors)

    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_title(f"Average Rank by {metric}")
    ax.invert_yaxis()

    # Add rank values
    for bar, val in zip(bars, avg_ranks.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
               f"{val:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_critical_difference(avg_ranks, cd, method_names, alpha=0.05, figsize=(10, 3)):
    """
    Nemenyi critical difference diagram.

    Parameters
    ----------
    avg_ranks : dict or array
        Average rank per method.
    cd : float
        Critical difference value.
    method_names : list of str
    """
    if isinstance(avg_ranks, dict):
        names = list(avg_ranks.keys())
        ranks = np.array([avg_ranks[n] for n in names])
    else:
        names = list(method_names)
        ranks = np.asarray(avg_ranks)

    # Sort by rank
    order = np.argsort(ranks)
    ranks = ranks[order]
    names = [names[i] for i in order]

    n = len(names)
    fig, ax = plt.subplots(figsize=figsize)

    # Axis
    lo = min(ranks) - 0.5
    hi = max(ranks) + 0.5
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)

    # Tick marks
    ax.set_xticks(np.arange(1, n + 1))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Average Rank")

    # No y-axis
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Place methods
    y_top = 0.8
    y_bot = 0.2
    half = n // 2

    for i, (name, rank) in enumerate(zip(names, ranks)):
        if i < half:
            y = y_top
            va = "bottom"
            ax.plot([rank, rank], [y - 0.05, y], "k-", lw=1)
        else:
            y = y_bot
            va = "top"
            ax.plot([rank, rank], [y, y + 0.05], "k-", lw=1)

        ax.text(rank, y, f" {name} ({rank:.2f})", ha="center", va=va, fontsize=8)

    # CD bar
    ax.plot([lo + 0.1, lo + 0.1 + cd], [0.5, 0.5], "k-", lw=3)
    ax.text(lo + 0.1 + cd / 2, 0.55, f"CD={cd:.2f}", ha="center", fontsize=9)

    # Connect groups (methods within CD of each other)
    for i in range(n):
        for j in range(i + 1, n):
            if ranks[j] - ranks[i] < cd:
                y_line = y_top + 0.02 * (j - i) if i < half else y_bot - 0.02 * (j - i)
                ax.plot([ranks[i], ranks[j]], [y_line, y_line], "k-", lw=2, alpha=0.5)

    ax.set_title(f"Critical Difference Diagram ($\\alpha$={alpha})")
    plt.tight_layout()
    return fig


def plot_metric_boxplots(fold_results, metric="g_mean", figsize=(12, 6)):
    """
    Box plots of fold-level metric distributions per method.

    Parameters
    ----------
    fold_results : DataFrame
        Must have columns: 'method', 'fold', and the metric column.
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = fold_results.groupby("method")[metric].median().sort_values(ascending=False).index
    data = [fold_results[fold_results["method"] == m][metric].values for m in methods]

    bp = ax.boxplot(data, labels=methods, vert=True, patch_artist=True)

    colors = sns.color_palette("Set2", len(methods))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric)
    ax.set_title(f"Distribution of {metric} across folds")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def plot_performance_vs_ir(results_df, metric="g_mean", dataset_info=None, figsize=(8, 6)):
    """
    Scatter plot: imbalance ratio vs. performance gain over SMOTE.

    Parameters
    ----------
    results_df : DataFrame
    metric : str
    dataset_info : dict of {dataset_name: {"ir": float, ...}}
    """
    if dataset_info is None:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Compute gain over SMOTE per dataset per method
    smote_scores = results_df[results_df["method"] == "smote"].set_index("dataset")[metric]

    methods = [m for m in results_df["method"].unique() if m != "smote"]
    cmap = plt.cm.Set1

    for i, method in enumerate(methods):
        method_scores = results_df[results_df["method"] == method].set_index("dataset")[metric]
        common = method_scores.index.intersection(smote_scores.index)

        irs = [dataset_info.get(d, {}).get("ir", 1.0) for d in common]
        gains = [(method_scores[d] - smote_scores[d]) for d in common]

        ax.scatter(irs, gains, c=[cmap(i)], s=40, alpha=0.7, label=method)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Imbalance Ratio")
    ax.set_ylabel(f"{metric} gain over SMOTE")
    ax.set_title(f"Performance gain vs. Imbalance Ratio ({metric})")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    return fig
