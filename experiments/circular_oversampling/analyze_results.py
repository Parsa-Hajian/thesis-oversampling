"""
Aggregate experiment results, run statistical tests, and generate summary tables.

Usage:
    python experiments/analyze_results.py results/tables/full_results_*.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.statistical_tests import (
    friedman_test, holms_posthoc, nemenyi_test, critical_difference_data,
)


def compute_average_ranks(df, metric="g_mean_median"):
    """Compute average rank per method across datasets."""
    pivot = df.pivot_table(values=metric, index="dataset", columns="method")
    ranks = pivot.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()
    return avg_ranks


def run_statistical_analysis(df, metric="g_mean_median", alpha=0.05):
    """Run Friedman + post-hoc tests."""
    pivot = df.pivot_table(values=metric, index="dataset", columns="method")
    matrix = pivot.values  # (n_datasets, n_methods)
    method_names = list(pivot.columns)

    print(f"\n{'='*60}")
    print(f"Statistical Analysis for: {metric}")
    print(f"{'='*60}")

    # Friedman test
    stat, p_val = friedman_test(matrix)
    print(f"\nFriedman test: statistic={stat:.4f}, p-value={p_val:.6f}")

    if p_val < alpha:
        print(f"  -> Significant at alpha={alpha}. Running post-hoc tests...")

        # Holm's test
        print("\nHolm's post-hoc pairwise comparisons:")
        holm_df = holms_posthoc(matrix, method_names)
        print(holm_df.to_string())

        # Nemenyi test
        print("\nNemenyi post-hoc pairwise comparisons:")
        nem_df = nemenyi_test(matrix, method_names)
        print(nem_df.to_string())

        # Critical difference
        cd_data = critical_difference_data(matrix, method_names, alpha=alpha)
        print(f"\nAverage ranks: {cd_data['avg_ranks']}")
        print(f"Critical difference: {cd_data['cd']:.4f}")

        return {
            "friedman_stat": stat,
            "friedman_p": p_val,
            "holm": holm_df,
            "nemenyi": nem_df,
            "cd_data": cd_data,
        }
    else:
        print(f"  -> NOT significant at alpha={alpha}.")
        return {"friedman_stat": stat, "friedman_p": p_val}


def generate_latex_table(df, metric="g_mean_median", caption="", label=""):
    """Generate LaTeX table from results pivot."""
    pivot = df.pivot_table(values=metric, index="method", columns="dataset")

    # Add average rank column
    ranks = pivot.rank(axis=1, ascending=False)
    pivot["Avg Rank"] = ranks.mean(axis=1)
    pivot = pivot.sort_values("Avg Rank")

    # Bold best per column
    latex = pivot.to_latex(float_format="%.4f", caption=caption, label=label)
    return latex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", help="Path to results CSV")
    parser.add_argument("--metrics", nargs="+",
                       default=["g_mean_median", "auc_median", "f_measure_median",
                               "balanced_accuracy_median"])
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} results from {args.results_csv}")
    print(f"Datasets: {df['dataset'].nunique()}")
    print(f"Methods: {df['method'].nunique()}")
    print(f"Classifiers: {df['classifier'].nunique()}")

    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"Metric {metric} not found, skipping.")
            continue

        # Average across classifiers for statistical tests
        agg = df.groupby(["dataset", "method"])[metric].mean().reset_index()

        # Average ranks
        avg_ranks = compute_average_ranks(agg, metric)
        print(f"\nAverage ranks for {metric}:")
        print(avg_ranks.to_string())

        # Statistical tests
        stats = run_statistical_analysis(agg, metric, args.alpha)

        # LaTeX table
        latex = generate_latex_table(agg, metric,
                                    caption=f"Results for {metric}",
                                    label=f"tab:{metric}")
        latex_path = output_dir / f"table_{metric}.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()
