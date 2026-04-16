"""
Generate all figures for the paper and thesis.

Reads experiment results and produces publication-quality plots.

Usage:
    python experiments/generate_figures.py results/tables/full_results_*.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.result_plots import (
    plot_result_heatmap,
    plot_average_rank_bar,
    plot_critical_difference,
    plot_metric_boxplots,
    plot_performance_vs_ir,
)
from src.datasets.registry import DATASETS
from src.evaluation.statistical_tests import critical_difference_data


def save_fig(fig, name, output_dir, formats=("pdf", "png")):
    """Save figure in multiple formats."""
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def generate_all_figures(results_csv, output_dir="results/figures"):
    """Generate all paper figures from results CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} results")

    metrics = ["g_mean_median", "auc_median", "f_measure_median", "balanced_accuracy_median"]

    for metric in metrics:
        if metric not in df.columns:
            continue

        metric_short = metric.replace("_median", "")

        # Average across classifiers
        agg = df.groupby(["dataset", "method"])[metric].mean().reset_index()
        agg.rename(columns={metric: metric_short}, inplace=True)

        # 1. Heatmap
        fig = plot_result_heatmap(agg, metric=metric_short)
        save_fig(fig, f"heatmap_{metric_short}", output_dir)

        # 2. Average rank bar chart
        fig = plot_average_rank_bar(agg, metric=metric_short)
        save_fig(fig, f"avgrank_{metric_short}", output_dir)

        # 3. Critical difference diagram
        pivot = agg.pivot_table(values=metric_short, index="dataset", columns="method")
        if pivot.shape[0] >= 3 and pivot.shape[1] >= 3:
            cd_data = critical_difference_data(
                pivot.values, list(pivot.columns), alpha=0.05
            )
            fig = plot_critical_difference(
                cd_data["avg_ranks"], cd_data["cd"], list(pivot.columns)
            )
            save_fig(fig, f"cd_diagram_{metric_short}", output_dir)

        # 4. Performance vs IR scatter
        fig = plot_performance_vs_ir(agg, metric=metric_short, dataset_info=DATASETS)
        if fig is not None:
            save_fig(fig, f"perf_vs_ir_{metric_short}", output_dir)

    # 5. Per-classifier results for main metric
    main_metric = "g_mean_median"
    if main_metric in df.columns:
        for clf in df["classifier"].unique():
            clf_df = df[df["classifier"] == clf]
            agg_clf = clf_df.groupby(["dataset", "method"])[main_metric].mean().reset_index()
            agg_clf.rename(columns={main_metric: "g_mean"}, inplace=True)

            fig = plot_result_heatmap(agg_clf, metric="g_mean")
            save_fig(fig, f"heatmap_gmean_{clf}", output_dir)

    print(f"\nAll figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", help="Path to results CSV")
    parser.add_argument("--output", default="results/figures",
                       help="Output directory for figures")
    args = parser.parse_args()

    generate_all_figures(args.results_csv, args.output)


if __name__ == "__main__":
    main()
